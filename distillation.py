from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DistillConfig
from model import GPTLike
from ema import EMA

class PredictorMLP(nn.Module):
    def __init__(self, d_model: int, projector_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, projector_dim),
            nn.GELU(),
            nn.Linear(projector_dim, d_model)
        )
    def forward(self, x):
        return self.net(x)

class SelfDistillTrainer:
    def __init__(self, student: GPTLike, cfg: DistillConfig, device: torch.device):
        self.student = student
        self.teacher = GPTLike(student.cfg, layerdrop_p=0.0)  # teacher never drops layers
        self.teacher.load_state_dict(student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.ema = EMA(self.student, decay=cfg.ema_decay)
        self.cfg = cfg
        self.device = device
        self.predictors = nn.ModuleList([PredictorMLP(student.cfg.d_model, cfg.projector_dim) for _ in range(student.cfg.n_layer)])
        self.predictors.to(device)
        self.frozen_prefix = 0

    @torch.no_grad()
    def update_teacher(self):
        self.ema.update(self.student)
        self.ema.copy_to(self.teacher)

    def maybe_freeze(self, global_step: int):
        for milestone, frac in zip(self.cfg.freeze_milestones, self.cfg.freeze_fracs):
            if global_step == milestone:
                k = int(frac * self.student.cfg.n_layer)
                for i, blk in enumerate(self.student.blocks):
                    if i < k:
                        for p in blk.parameters():
                            p.requires_grad = False
                self.frozen_prefix = max(self.frozen_prefix, k)

    def layer_map(self, l_s: int, L_s: int, L_t: int) -> int:
        return min(L_t - 1, max(0, round(l_s * (L_t / L_s))))

    def kd_loss(self, logits_s: torch.Tensor, logits_t: torch.Tensor, T: float) -> torch.Tensor:
        s = F.log_softmax(logits_s / T, dim=-1)
        t = F.softmax(logits_t / T, dim=-1)
        return F.kl_div(s, t, reduction='batchmean') * (T * T)

    def sd_loss(self, h_s: List[torch.Tensor], h_t: List[torch.Tensor]) -> torch.Tensor:
        losses = []
        Ls, Lt = len(h_s), len(h_t)
        for l in range(self.frozen_prefix, Ls):
            map_to = self.layer_map(l, Ls, Lt)
            s_h = h_s[l]
            with torch.no_grad():
                t_h = h_t[map_to].detach()
            p = self.predictors[l](s_h)
            p = F.normalize(p, dim=-1)
            t = F.normalize(t_h, dim=-1)
            loss = 1.0 - (p * t).sum(dim=-1).mean()
            losses.append(loss)
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)

    def compute_losses(self, batch: Dict[str, torch.Tensor], train_mode: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        idx, targets = batch['input_ids'], batch['labels']
        logits_s, h_s = self.student(idx, return_hidden=True, frozen_prefix=self.frozen_prefix, train_mode=train_mode)
        with torch.no_grad():
            logits_t, h_t = self.teacher(idx, return_hidden=True, frozen_prefix=0, train_mode=False)
        loss_lm = F.cross_entropy(logits_s.view(-1, logits_s.size(-1)), targets.view(-1))
        loss_kd = self.kd_loss(logits_s, logits_t, self.cfg.temperature)
        loss_sd = self.sd_loss(h_s, h_t)
        total = loss_lm + self.cfg.lambda_kd * loss_kd + self.cfg.lambda_sd * loss_sd
        stats = {
            'loss_total': float(total.detach().item()),
            'loss_lm': float(loss_lm.detach().item()),
            'loss_kd': float(loss_kd.detach().item()),
            'loss_sd': float(loss_sd.detach().item()),
        }
        return total, stats
