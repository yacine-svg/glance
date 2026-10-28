"""Glance: a GPT-style decoder-only transformer, written out in full.

Architecture (GPT-2 family):
  token embedding + learned position embedding
  → N × [ x + Attention(LayerNorm(x)) ; x + MLP(LayerNorm(x)) ]   (pre-norm residual blocks)
  → final LayerNorm → linear head tied to the token embedding

The names of the parameters here are the contract with train/export.py,
train/reference.py (NumPy) and lib/engine/model.ts (browser).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 4096
    ctx: int = 256  # context length (tokens)
    d: int = 384  # model width
    n_layer: int = 6
    n_head: int = 6
    dropout: float = 0.1

    def dict(self):
        return asdict(self)


class Attention(nn.Module):
    """Multi-head causal self-attention: each token looks back at the ones before it."""

    def __init__(self, c: Config):
        super().__init__()
        assert c.d % c.n_head == 0
        self.h, self.hd = c.n_head, c.d // c.n_head
        self.qkv = nn.Linear(c.d, 3 * c.d)
        self.proj = nn.Linear(c.d, c.d)
        self.attn_drop = nn.Dropout(c.dropout)
        self.resid_drop = nn.Dropout(c.dropout)
        self.dropout = c.dropout

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.h, self.hd).transpose(1, 2)  # (B, H, T, hd)
        k = k.view(B, T, self.h, self.hd).transpose(1, 2)
        v = v.view(B, T, self.h, self.hd).transpose(1, 2)

        if return_attn:
            # explicit version, so we can look at the attention weights
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
            y = att @ v
        else:
            # same maths, fused kernel (FlashAttention on GPU) for training speed
            att = None
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), att


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.fc = nn.Linear(c.d, 4 * c.d)
        self.proj = nn.Linear(4 * c.d, c.d)
        self.act = nn.GELU(approximate="tanh")  # the browser engine uses the same tanh approximation
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        return self.drop(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d)
        self.attn = Attention(c)
        self.ln2 = nn.LayerNorm(c.d)
        self.mlp = MLP(c)

    def forward(self, x, return_attn=False):
        a, att = self.attn(self.ln1(x), return_attn)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, att


class Glance(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab_size, c.d)
        self.wpe = nn.Embedding(c.ctx, c.d)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d)
        self.lm_head = nn.Linear(c.d, c.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying: fewer parameters, better small models

        self.apply(self._init)
        # GPT-2 trick: scale the residual projections so the residual stream doesn't grow with depth
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters()) - self.wpe.weight.numel()

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_attn: bool = False):
        B, T = idx.shape
        assert T <= self.c.ctx, f"sequence of {T} tokens is longer than the context ({self.c.ctx})"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        attns = []
        for block in self.blocks:
            x, att = block(x, return_attn)
            attns.append(att)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, (attns if return_attn else None)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, n: int, temperature: float = 0.8, top_k: int | None = 40):
        for _ in range(n):
            logits, _, _ = self(idx[:, -self.c.ctx :])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(logits.softmax(-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    def optimizer(self, lr: float, weight_decay: float = 0.1, betas=(0.9, 0.95)):
        """AdamW with weight decay on matrices only (not on biases, LayerNorm or embeddings' norms)."""
        decay = [p for n, p in self.named_parameters() if p.dim() >= 2]
        no_decay = [p for n, p in self.named_parameters() if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        fused = torch.cuda.is_available()
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
