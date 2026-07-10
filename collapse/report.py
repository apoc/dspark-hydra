"""Domain analysis for a collapse map (adapts the ESFT profiler's routing analysis).

Answers, on calibration data: do domains activate different target experts at l*
(the premise of router reuse), and does the collapse map C give domain-separated
draft groups (block-diagonal group x domain -> hypothesis plausible)?
"""

from __future__ import annotations

import torch

from target.dump import DOMAINS


def per_domain_gate_mass(dump_dir: str, num_experts: int = 256) -> dict[int, torch.Tensor]:
    """Accumulate summed router softmax mass per expert, per domain, at l*."""
    from target.dump import load_shards

    mass = {d: torch.zeros(num_experts, dtype=torch.float64) for d in range(len(DOMAINS))}
    for sd, _ in load_shards(dump_dir):
        p = torch.softmax(sd["router_star"].float(), dim=-1)
        dom = sd["domain_id"].long()
        for d in mass:
            m = dom == d
            if m.any():
                mass[d] += p[m].sum(0).double()
    return mass


def hot_set(mass_row: torch.Tensor, frac: float = 0.8) -> set[int]:
    if mass_row.sum() == 0:
        return set()
    order = mass_row.argsort(descending=True)
    cum = mass_row[order].cumsum(0) / mass_row.sum()
    k = int((cum < frac).sum().item()) + 1
    return set(order[:k].tolist())


def domain_overlap(mass: dict[int, torch.Tensor]) -> dict:
    """Pairwise Jaccard overlap of domain hot-expert sets (@80% mass)."""
    hs = {d: hot_set(m) for d, m in mass.items()}
    out = {}
    doms = [d for d in hs if hs[d]]
    for i in doms:
        for j in doms:
            if j <= i:
                continue
            a, b = hs[i], hs[j]
            jac = len(a & b) / max(len(a | b), 1)
            out[f"{DOMAINS[i]}~{DOMAINS[j]}"] = round(jac, 3)
    return {"hot_set_sizes": {DOMAINS[d]: len(hs[d]) for d in hs}, "jaccard": out}


def group_domain_composition(dump_dir: str, C: torch.Tensor, k_prime: int = 2) -> torch.Tensor:
    """(K, num_domains): fraction of each domain's tokens whose top draft group is g."""
    from target.dump import load_shards

    K = C.shape[0]
    nd = len(DOMAINS)
    counts = torch.zeros(K, nd, dtype=torch.float64)
    tot = torch.zeros(nd, dtype=torch.float64)
    Ct = C.t().float()
    for sd, _ in load_shards(dump_dir):
        p = torch.softmax(sd["router_star"].float(), dim=-1)
        g = (p @ Ct).argmax(-1)          # top group per token
        dom = sd["domain_id"].long()
        for d in range(nd):
            m = dom == d
            if m.any():
                counts[:, d].scatter_add_(0, g[m], torch.ones(int(m.sum()), dtype=torch.float64))
                tot[d] += int(m.sum())
    return counts / (tot[None, :] + 1e-12)
