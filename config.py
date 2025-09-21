from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 512
    n_head: int = 8
    d_ff: int = 2048
    n_layer: int = 8
    max_seq_len: int = 512
    dropout: float = 0.1

@dataclass
class DistillConfig:
    temperature: float = 2.0
    lambda_kd: float = 0.5
    lambda_sd: float = 0.2
    projector_dim: int = 256
    ema_decay: float = 0.999
    freeze_milestones: Tuple[int, ...] = (20000, 60000)
    freeze_fracs: Tuple[float, ...] = (0.25, 0.5)

@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    train_steps: int = 20000
    eval_every: int = 1000
    log_every: int = 100
    accum_steps: int = 1
    amp: bool = True
    layerdrop_p: float = 0.0
