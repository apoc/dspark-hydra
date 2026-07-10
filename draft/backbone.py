"""Draft backbone: stack of injected-attention + (MoE|dense) FFN layers (§4.1)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import InjectedAttention, RMSNorm
from .moe import make_ffn


class DraftLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)
        self.attn = InjectedAttention(cfg)
        self.ffn = make_ffn(cfg)   # DraftMoE (routed by target) or DraftFFN (dense)

    def forward(self, x, ctx, d, C):
        x = x + self.attn(self.attn_norm(x), ctx)
        x = self.ffn(x, d, C)      # MoE/dense own their norm + residual
        return x


class DraftBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([DraftLayer(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)

    def forward(self, x, ctx, d, C):
        for layer in self.layers:
            x = layer(x, ctx, d, C)
        return self.final_norm(x)
