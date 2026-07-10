"""Semi-autoregressive Markov head (§4.3).

p_k^d = softmax(U_k + B_k) where
  U_k = LM_head(h_k)                     base logits from the draft hidden (frozen head)
  B_k = W1[x_{k-1}] @ W2                  low-rank (r) transition bias on the prev token

Keeps p_k^d an exact per-token categorical -> speculative rejection sampling stays
lossless. W1 (V,r) is shared with the confidence head. LM head is borrowed & frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MarkovHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W1 = nn.Embedding(cfg.vocab_size, cfg.markov_rank)   # W1[x] : (r,)
        self.W2 = nn.Linear(cfg.markov_rank, cfg.vocab_size, bias=False)
        nn.init.zeros_(self.W2.weight)   # start with zero transition bias (B_k = 0)

    def token_feature(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        """W1[x_{k-1}] : (..., r) — shared with the confidence head."""
        return self.W1(prev_tokens)

    def forward(self, h: torch.Tensor, prev_tokens: torch.Tensor, lm_head) -> tuple[torch.Tensor, torch.Tensor]:
        """h:(B,γ,H), prev_tokens:(B,γ) -> (logits:(B,γ,V), token_feat:(B,γ,r))."""
        U = lm_head(h)                                    # frozen base logits
        feat = self.token_feature(prev_tokens)            # (B,γ,r)
        B = self.W2(feat)                                 # (B,γ,V)
        return U + B, feat
