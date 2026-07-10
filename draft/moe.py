"""Domain-routed MoE draft FFN — the core novelty (§4.2).

K draft experts (SwiGLU, width moe_intermediate_size), top-k' active, plus an always-on
shared expert. Routing derived from the target, not recomputed:

  hard    (Variant A): g = C @ softmax(d); top-k' groups. C frozen.
  soft    (Variant B): tiny draft router R_d(h); distilled toward C @ softmax(d).
  scratch (control)  : R_d(h) trained only on drafting loss (no target signal).
  dense   (control)  : single FFN of matched active FLOPs (k' * I), no routing.

`d` is the target's router logits at l* (borrowed, zero extra target compute).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_up = nn.Linear(hidden, 2 * inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class DraftFFN(nn.Module):
    """Dense control: single FFN with matched active FLOPs (width = k' * I)."""

    def __init__(self, cfg):
        super().__init__()
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)
        self.ffn = SwiGLU(cfg.hidden_size, cfg.k_prime * cfg.moe_intermediate_size)

    def forward(self, x, d=None, C=None):
        return x + self.ffn(self.norm(x))


class DraftMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)
        self.experts = nn.ModuleList([SwiGLU(cfg.hidden_size, cfg.moe_intermediate_size) for _ in range(cfg.K)])
        self.shared = SwiGLU(cfg.hidden_size, cfg.moe_intermediate_size) if cfg.n_shared else None
        if cfg.router_mode in ("soft", "scratch"):
            self.router = nn.Linear(cfg.hidden_size, cfg.K, bias=False)
        self.last_group_logits = None   # (N,K) distillation target = C @ softmax(d) (soft)
        self.last_router_logits = None  # (N,K) draft router output (soft/scratch)
        self.last_gate = None           # (N,K) selection weights for aux balance loss

    def _group_scores(self, h, d, C):
        """Return group scores (N,K). hard/soft distill target = C @ softmax(d)."""
        cfg = self.cfg
        if cfg.router_mode == "hard":
            p = torch.softmax(d.float(), dim=-1)          # (N,E)
            return p @ C.to(p.dtype).t()                  # (N,K)
        # soft caches the distillation target; both soft/scratch use the draft router
        if cfg.router_mode == "soft":
            p = torch.softmax(d.float(), dim=-1)
            self.last_group_logits = (p @ C.to(p.dtype).t())
        self.last_router_logits = self.router(h)
        return self.last_router_logits

    def forward(self, x, d, C):
        cfg = self.cfg
        h = self.norm(x)
        N = h.shape[:-1].numel()
        hf = h.reshape(N, cfg.hidden_size)
        df = d.reshape(N, -1) if d is not None else None

        scores = self._group_scores(hf, df, C)            # (N,K)
        gate = torch.softmax(scores, dim=-1)
        topw, topi = gate.topk(cfg.k_prime, dim=-1)       # (N,k')
        topw = (topw / (topw.sum(-1, keepdim=True) + 1e-9)).to(hf.dtype)  # renorm active, match dtype
        self.last_gate = gate

        out = torch.zeros_like(hf)
        for slot in range(cfg.k_prime):
            idx = topi[:, slot]                           # (N,)
            w = topw[:, slot].unsqueeze(-1)
            for e in range(cfg.K):
                m = idx == e
                if m.any():
                    out[m] += w[m] * self.experts[e](hf[m])
        if self.shared is not None:
            out = out + self.shared(hf)
        return x + out.reshape_as(x)


def make_ffn(cfg):
    return DraftFFN(cfg) if cfg.router_mode == "dense" else DraftMoE(cfg)
