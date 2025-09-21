import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from transformers import GPT2TokenizerFast
except Exception:
    GPT2TokenizerFast = None

from config import DistillConfig, ModelConfig
from data import HFTextDataset, ToyCharDataset
from distillation import PredictorMLP, SelfDistillTrainer
from model import GPTLike
from scheduler import CosineWithWarmup
from utils import evaluate, prune_transformer, set_seed

def main():
    p = argparse.ArgumentParser()
    # data
    p.add_argument('--dataset', type=str, default=None, choices=[None, 'wikitext2', 'wikitext103', 'openwebtext'])
    p.add_argument('--toy_text', type=str, default=None)
    p.add_argument('--max_seq_len', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--accum_steps', type=int, default=1)
    # model
    p.add_argument('--vocab_size', type=int, default=50257)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--d_ff', type=int, default=2048)
    p.add_argument('--n_layer', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--layerdrop_p', type=float, default=0.0)
    # distill
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--lambda_kd', type=float, default=0.5)
    p.add_argument('--lambda_sd', type=float, default=0.2)
    p.add_argument('--projector_dim', type=int, default=256)
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--freeze_milestones', type=int, nargs='*', default=[20000, 60000])
    p.add_argument('--freeze_fracs', type=float, nargs='*', default=[0.25, 0.5])
    # train
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.95])
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--warmup_steps', type=int, default=2000)
    p.add_argument('--train_steps', type=int, default=20000)
    p.add_argument('--eval_every', type=int, default=1000)
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--seed', type=int, default=1337)
    # ckpt & resume
    p.add_argument('--out_dir', type=str, default='runs/selfdistil_t')
    p.add_argument('--resume', type=str, default=None)
    # pruning / continuation
    p.add_argument('--prune_frac', type=float, default=0.0)
    p.add_argument('--continue_steps', type=int, default=0)

    args = p.parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # Data
    if args.dataset is None and args.toy_text is None:
        raise SystemExit("Provide --dataset or --toy_text")

    if args.dataset is not None:
        assert GPT2TokenizerFast is not None, "Install transformers & datasets"
        tok = GPT2TokenizerFast.from_pretrained('gpt2')
        tok.pad_token = tok.eos_token
        train_ds = HFTextDataset(args.dataset, 'train', tok, args.max_seq_len)
        val_ds = HFTextDataset(args.dataset, 'validation' if args.dataset != 'openwebtext' else 'train[:1%]', tok, args.max_seq_len)
        vocab_size = tok.vocab_size
    else:
        text = (args.toy_text or "SelfDistil‑T toy data") * 100
        train_ds = ToyCharDataset(text, seq_len=args.max_seq_len)
        val_ds = ToyCharDataset(text, seq_len=args.max_seq_len)
        vocab_size = train_ds.vocab_size

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # Model & trainer
    mcfg = ModelConfig(vocab_size=vocab_size, d_model=args.d_model, n_head=args.n_head, d_ff=args.d_ff,
                       n_layer=args.n_layer, max_seq_len=args.max_seq_len, dropout=args.dropout)
    student = GPTLike(mcfg, layerdrop_p=args.layerdrop_p).to(device)
    dcfg = DistillConfig(temperature=args.temperature, lambda_kd=args.lambda_kd, lambda_sd=args.lambda_sd,
                         projector_dim=args.projector_dim, ema_decay=args.ema_decay,
                         freeze_milestones=tuple(args.freeze_milestones), freeze_fracs=tuple(args.freeze_fracs))
    trainer = SelfDistillTrainer(student, dcfg, device)

    # Optim & sched
    params = [p for p in list(student.parameters()) + list(trainer.predictors.parameters()) if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay)
    sched = CosineWithWarmup(optim, warmup_steps=args.warmup_steps, total_steps=args.train_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Resume
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        student.load_state_dict(ckpt['student'])
        trainer.teacher.load_state_dict(ckpt['teacher'])
        trainer.predictors.load_state_dict(ckpt['predictors'])
        optim.load_state_dict(ckpt['optim'])
        sched.load_state_dict(ckpt['sched'])
        global_step = ckpt['global_step']
        trainer.frozen_prefix = ckpt.get('frozen_prefix', 0)
        print(f"Resumed from {args.resume} @ step {global_step}")

    # Optional pruning before continuation
    if args.prune_frac > 0.0:
        L = len(student.blocks)
        k = int(round((1.0 - args.prune_frac) * L))
        # keep  equally spaced indices
        keep = sorted(set([round(i * (L-1) / max(1,k-1)) for i in range(k)]))
        student_pruned = prune_transformer(student, keep).to(device)
        trainer.student = student_pruned
        trainer.teacher = GPTLike(student_pruned.cfg, layerdrop_p=0.0).to(device)
        trainer.teacher.load_state_dict(student_pruned.state_dict())
        for p in trainer.teacher.parameters(): p.requires_grad = False
        trainer.predictors = nn.ModuleList([PredictorMLP(student_pruned.cfg.d_model, dcfg.projector_dim) for _ in range(student_pruned.cfg.n_layer)]).to(device)
        print(f"Pruned transformer: kept layers {keep}")

    # Training loop
    student.train(); trainer.teacher.train()
    t0 = time.time()
    running = {}
    while global_step < args.train_steps:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            trainer.maybe_freeze(global_step)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss, stats = trainer.compute_losses(batch, train_mode=True)
                loss = loss / max(1, args.accum_steps)
            scaler.scale(loss).backward()
            if (global_step + 1) % args.accum_steps == 0:
                if args.max_grad_norm is not None:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(trainer.student.parameters(), args.max_grad_norm)
                scaler.step(optim)
                scaler.update()
                sched.step()
                trainer.update_teacher()
            # logging
            for k, v in stats.items():
                running[k] = running.get(k, []) + [v]
            if global_step % args.log_every == 0:
                msg = {k: round(sum(v)/len(v), 4) for k, v in running.items()}
                lr_now = optim.param_groups[0]['lr']
                msg['lr'] = round(lr_now, 7)
                msg['step'] = global_step
                msg['frozen_prefix'] = trainer.frozen_prefix
                print(msg, flush=True)
                running = {}
            if global_step % args.eval_every == 0 and global_step > 0:
                eval_stats = evaluate(trainer, val_loader, device)
                print({**{'step': global_step}, **{k: round(v, 4) for k, v in eval_stats.items()}}, flush=True)
                # save ckpt
                ckpt = {
                    'student': trainer.student.state_dict(),
                    'teacher': trainer.teacher.state_dict(),
                    'predictors': trainer.predictors.state_dict(),
                    'optim': optim.state_dict(),
                    'sched': sched.state_dict(),
                    'global_step': global_step,
                    'frozen_prefix': trainer.frozen_prefix,
                }
                torch.save(ckpt, os.path.join(args.out_dir, f'ckpt_{global_step}.pt'))
            global_step += 1
            if global_step >= args.train_steps:
                break
    dt = time.time() - t0
    print(f"Training finished in {dt/60:.1f} min. Final step: {global_step}")

    # Optional continuation after pruning
    if args.continue_steps > 0:
        print(f"Continuing training for {args.continue_steps} steps after pruning…")
        end_step = global_step + args.continue_steps
        while global_step < end_step:
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optim.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    loss, _ = trainer.compute_losses(batch, train_mode=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(trainer.student.parameters(), args.max_grad_norm)
                scaler.step(optim)
                scaler.update()
                sched.step()
                trainer.update_teacher()
                global_step += 1
                if global_step >= end_step:
                    break
        print("Continuation finished.")

if __name__ == '__main__':
    main()
