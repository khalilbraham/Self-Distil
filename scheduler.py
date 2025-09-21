import math
import torch

class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lrs.append(base_lr * step / max(1, self.warmup_steps))
            else:
                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                progress = min(max(progress, 0.0), 1.0)
                lrs.append(0.5 * base_lr * (1 + math.cos(math.pi * progress)))
        return lrs
