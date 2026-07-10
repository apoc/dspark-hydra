"""Method 3 (§5.3) — learned collapse (upper bound).

C is a trainable K×256 row-stochastic matrix (softmax over the 256 target experts),
trained end-to-end with the drafting losses. Most flexible, least interpretable; used
to bound achievable tau. Can be initialized from a co-activation/weight C (logits =
log of a one-hot-ish prior) or uniformly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedCollapse(nn.Module):
    def __init__(self, k: int, num_experts: int = 256, init_C: torch.Tensor | None = None):
        super().__init__()
        if init_C is not None:
            # logits whose row-softmax approximates init_C (per-expert membership)
            logits = torch.log(init_C.clamp_min(1e-4))
        else:
            logits = torch.zeros(k, num_experts)
        self.logits = nn.Parameter(logits)

    def C(self) -> torch.Tensor:
        """Row-stochastic collapse map (each draft group is a distribution over experts)."""
        return torch.softmax(self.logits, dim=-1)

    def group_scores(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """expert_probs:(...,E) target router softmax -> group scores (...,K)."""
        return expert_probs @ self.C().t()
