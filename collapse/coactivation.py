"""Method 1 (default, §5.1) — co-activation clustering.

Accumulate a 256x256 co-activation matrix M[i,j] = #(experts i,j both in top-8) at
source layer l* over the calibration dump, normalize (PMI), and spectral-cluster into
K groups. Experts firing on the same tokens serve the same content -> same draft expert.
"""

from __future__ import annotations

import torch

from .kmeans import labels_to_C, spectral_labels


def coactivation_matrix(dump_dir: str, top_k: int = 8, num_experts: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (M:(E,E) co-activation counts, sel:(E,) marginal selection counts)."""
    from target.dump import load_shards

    M = torch.zeros(num_experts, num_experts, dtype=torch.float64)
    sel = torch.zeros(num_experts, dtype=torch.float64)
    for sd, _ in load_shards(dump_dir):
        rs = sd["router_star"].float()               # (N,E) logits
        idx = rs.topk(top_k, dim=-1).indices          # (N,k)
        # marginal counts
        sel.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel(), dtype=torch.float64))
        # pairwise co-activation via per-token one-hot outer product, batched
        onehot = torch.zeros(rs.shape[0], num_experts)
        onehot.scatter_(1, idx, 1.0)
        M += (onehot.T.double() @ onehot.double())
    M.fill_diagonal_(0)
    return M, sel


def pmi_affinity(M: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """Positive PMI normalization of the co-activation matrix -> symmetric affinity."""
    total = sel.sum()
    p_i = sel / (total + 1e-12)
    p_ij = M / (M.sum() + 1e-12)
    denom = p_i[:, None] * p_i[None, :]
    pmi = torch.log((p_ij + 1e-12) / (denom + 1e-12))
    return pmi.clamp(min=0.0)


def build_coactivation_C(dump_dir: str, k: int, top_k: int = 8, num_experts: int = 256, seed: int = 0):
    """Build the collapse map C via co-activation + spectral clustering.

    Returns (C:(k,E), info dict).
    """
    M, sel = coactivation_matrix(dump_dir, top_k, num_experts)
    A = pmi_affinity(M, sel)
    labels = spectral_labels(A, k, seed=seed)
    C = labels_to_C(labels, k, num_experts)
    info = {
        "method": "co_activation",
        "labels": labels,
        "sel_counts": sel,
        "coactivation_nnz": int((M > 0).sum().item()),
        "group_sizes": [int((labels == g).sum().item()) for g in range(k)],
    }
    return C, info
