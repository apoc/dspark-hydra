"""KV injection (DFlash-style, §4.1).

Concatenate the target's full-attention hidden states at the injection layers and
project once: H_ctx = RMSNorm(W_c [H^{l1};...;H^{lm}]). H_ctx is injected as extra
context key/value positions into every draft layer's attention.

Rationale: only full-attention layers hold a per-position global-softmax context;
linear (DeltaNet) layers keep a recurrent SSM state and are poor injection sources.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import RMSNorm


class KVInjector(nn.Module):
    """Project target inject-layer hiddens into draft context KV tokens.

    Accepts either a single anchor context (B, n_inject, H) -> (B, 1, H) [summary /
    EAGLE-style, default] or a per-position prefix (B, T_ctx, n_inject, H) ->
    (B, T_ctx, H) [prefix mode]. The projection operates on the layer-concat feature
    dim, so the same weights serve any number of context positions.
    """

    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Linear(cfg.n_inject * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)
        self.n_inject = cfg.n_inject
        self.hidden = cfg.hidden_size

    def forward(self, hidden_inject: torch.Tensor) -> torch.Tensor:
        if hidden_inject.dim() == 3:            # (B, n_inject, H) -> single context token
            B = hidden_inject.shape[0]
            flat = hidden_inject.reshape(B, self.n_inject * self.hidden)
            return self.norm(self.proj(flat)).unsqueeze(1)
        # (B, T_ctx, n_inject, H) -> (B, T_ctx, H)
        B, Tc = hidden_inject.shape[:2]
        flat = hidden_inject.reshape(B, Tc, self.n_inject * self.hidden)
        return self.norm(self.proj(flat))
