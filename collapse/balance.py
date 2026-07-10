"""Load-balance analysis for a collapse map C (§5.4).

Reused target routing can starve some draft experts. Given C and the calibration dump,
compute how often each draft group is selected (top-k') and its total gate mass, so we
can detect skew and, if needed, rebalance.
"""

from __future__ import annotations

import torch


def group_load(dump_dir: str, C: torch.Tensor, k_prime: int = 2) -> dict:
    """Per-draft-group selection frequency + gate mass under reused routing.

    For each token: p = softmax(router_star); g = p @ C^T (K,); select top-k' groups.
    """
    from target.dump import load_shards

    K = C.shape[0]
    sel = torch.zeros(K, dtype=torch.float64)
    mass = torch.zeros(K, dtype=torch.float64)
    n = 0
    Ct = C.t().float()
    for sd, _ in load_shards(dump_dir):
        p = torch.softmax(sd["router_star"].float(), dim=-1)  # (N,E)
        g = p @ Ct                                            # (N,K)
        top = g.topk(min(k_prime, K), dim=-1).indices         # (N,k')
        sel.scatter_add_(0, top.reshape(-1), torch.ones(top.numel(), dtype=torch.float64))
        mass += g.sum(0).double()
        n += p.shape[0]

    freq = sel / max(n, 1)
    mass_frac = mass / (mass.sum() + 1e-12)
    ideal = k_prime / K
    return {
        "tokens": n,
        "select_freq": freq,               # per group, fraction of tokens selecting it
        "mass_frac": mass_frac,            # per group, share of total group mass
        "cv_freq": float((freq.std() / (freq.mean() + 1e-12)).item()),   # coeff of variation
        "min_freq": float(freq.min().item()),
        "max_freq": float(freq.max().item()),
        "ideal_freq": ideal,
        "starved_groups": int((freq < 0.2 * ideal).sum().item()),
    }
