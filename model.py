import math
import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T]

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer('mask', torch.tril(torch.ones(8192, 8192)).view(1, 1, 8192, 8192))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.out(y))
        return y

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTLike(nn.Module):
    def __init__(self, cfg: ModelConfig, layerdrop_p: float = 0.0):
        super().__init__()
        self.cfg = cfg
        self.layerdrop_p = layerdrop_p
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, return_hidden: bool = False, frozen_prefix: int = 0, train_mode: bool = True) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        B, T = idx.size()
        x = self.tok_emb(idx)
        x = self.pos_emb(x)
        x = self.drop(x)
        hiddens = []
        L = len(self.blocks)
        for li, blk in enumerate(self.blocks):
            # LayerDrop only during training and not for frozen layers
            if train_mode and li >= frozen_prefix and self.layerdrop_p > 0.0:
                if random.random() < self.layerdrop_p:
                    # skip updating this block but keep representation
                    hiddens.append(x) if return_hidden else None
                    continue
            x = blk(x)
            if return_hidden:
                hiddens.append(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return (logits, hiddens) if return_hidden else (logits, None)
