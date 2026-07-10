"""Confidence head (§4.4).

c_k = sigma(w . [h_k ; W1[x_{k-1}]]), supervised by c_k* = 1 - 1/2 ||p_k^d - p_k^t||_1.
Shares W1 (the r-dim token feature) with the Markov head. Post-hoc Sequential
Temperature Scaling (STS) is applied at eval time; here we output the raw logit + prob.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConfidenceHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w = nn.Linear(cfg.hidden_size + cfg.markov_rank, 1, bias=True)

    def forward(self, h: torch.Tensor, token_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h:(B,γ,H), token_feat:(B,γ,r) -> (logit:(B,γ), prob:(B,γ))."""
        logit = self.w(torch.cat([h, token_feat], dim=-1)).squeeze(-1)
        return logit, torch.sigmoid(logit)
