import torch
import torch.nn as nn

class EMA:
    """Exponential Moving Average for teacher parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow and p.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * p.data + self.decay * self.shadow[name]

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow and p.requires_grad:
                p.data.copy_(self.shadow[name])
