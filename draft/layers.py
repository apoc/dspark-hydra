"""Draft transformer primitives: RMSNorm, rotary embedding, and an attention module
that attends over injected target context (KV) plus causal draft positions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def build_rope(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    ang = torch.outer(pos, inv)              # (T, hd/2)
    emb = torch.cat([ang, ang], dim=-1)      # (T, hd)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x:(B,H,T,hd); cos/sin:(T,hd)
    hd = x.shape[-1]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos[None, None] + rot * sin[None, None]


class InjectedAttention(nn.Module):
    """Multi-head attention over [injected context KV | causal draft positions].

    Query = draft positions (T=gamma). Keys/Values = concat(context tokens, draft
    positions). Context tokens are visible to all draft positions (prefix); draft
    positions are causal among themselves. RoPE is applied to draft Q/K only.
    """

    def __init__(self, cfg):
        super().__init__()
        H, nh = cfg.hidden_size, cfg.n_heads
        self.nh, self.hd = nh, H // nh
        self.q = nn.Linear(H, H, bias=False)
        self.k = nn.Linear(H, H, bias=False)
        self.v = nn.Linear(H, H, bias=False)
        self.o = nn.Linear(H, H, bias=False)
        self.kc = nn.Linear(H, H, bias=False)  # context key
        self.vc = nn.Linear(H, H, bias=False)  # context value
        self.theta = cfg.rope_theta

    def _split(self, x, B, T):
        return x.view(B, T, self.nh, self.hd).transpose(1, 2)  # (B,nh,T,hd)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # x:(B,T,H) draft positions; ctx:(B,Tc,H) injected context
        B, T, H = x.shape
        Tc = ctx.shape[1]
        cos, sin = build_rope(T, self.hd, self.theta, x.device, x.dtype)

        q = apply_rope(self._split(self.q(x), B, T), cos, sin)
        k = apply_rope(self._split(self.k(x), B, T), cos, sin)
        v = self._split(self.v(x), B, T)
        kc = self._split(self.kc(ctx), B, Tc)
        vc = self._split(self.vc(ctx), B, Tc)

        keys = torch.cat([kc, k], dim=2)     # (B,nh,Tc+T,hd)
        vals = torch.cat([vc, v], dim=2)
        scores = (q @ keys.transpose(-1, -2)) / (self.hd ** 0.5)  # (B,nh,T,Tc+T)
        # mask: context fully visible; draft causal
        cmask = torch.zeros(T, Tc, device=x.device)
        dmask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        mask = torch.cat([cmask, dmask], dim=1)
        scores = scores + mask[None, None]
        attn = scores.softmax(-1)
        out = attn @ vals                    # (B,nh,T,hd)
        out = out.transpose(1, 2).reshape(B, T, H)
        return self.o(out)
