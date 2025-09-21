import math
import random
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from distillation import SelfDistillTrainer
from model import GPTLike

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



@torch.no_grad()
def evaluate(trainer: SelfDistillTrainer, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    trainer.student.eval(); trainer.teacher.eval()
    losses = []
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, _ = trainer.compute_losses(batch, train_mode=False)
        losses.append(loss.item())
    ppl = math.exp(sum(losses)/len(losses)) if losses else float('inf')
    trainer.student.train(); trainer.teacher.train()
    return {'val_loss': sum(losses)/len(losses), 'val_ppl': ppl}

def prune_transformer(model: GPTLike, keep_indices: List[int]) -> GPTLike:
    """Return a new GPTLike with only blocks at keep_indices (ascending)."""
    cfg = model.cfg
    new_model = GPTLike(cfg, layerdrop_p=getattr(model, 'layerdrop_p', 0.0))
    with torch.no_grad():
        new_model.tok_emb.weight.copy_(model.tok_emb.weight)
        for i, ki in enumerate(keep_indices):
            new_model.blocks[i].load_state_dict(model.blocks[ki].state_dict())
        new_model.ln_f.load_state_dict(model.ln_f.state_dict())
        new_model.head.weight.copy_(model.head.weight)
    # Shrink module list logically
    new_model.blocks = nn.ModuleList(new_model.blocks[:len(keep_indices)])
    new_model.cfg.n_layer = len(keep_indices)
    return new_model
