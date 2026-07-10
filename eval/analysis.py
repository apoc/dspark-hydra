"""Phase 5 analyses (§8.1-8.2).

- position_wise_acceptance: acceptance at draft position k conditioned on 1..k-1 accepted
  (DSpark Fig 2). Shows suffix decay and whether routing helps deep positions.
- specialization_heatmap: draft-expert (group) activation frequency x domain. Expect
  block-diagonal for reused-router MoE (E1/E2), uniform for dense/scratch (RQ3).
"""

from __future__ import annotations

import torch


def position_wise_acceptance(n_accs: list[int], gamma: int) -> list[float]:
    """From raw accepted-counts per round, P(accept at k | reached k), k=1..gamma."""
    reached = [0] * (gamma + 1)   # reached[k] = #rounds with >= k-1 accepts (reached position k)
    accepted = [0] * (gamma + 1)  # accepted[k] = #rounds with >= k accepts
    for na in n_accs:
        for k in range(1, gamma + 1):
            if na >= k - 1:
                reached[k] += 1
            if na >= k:
                accepted[k] += 1
    return [accepted[k] / reached[k] if reached[k] else 0.0 for k in range(1, gamma + 1)]


@torch.no_grad()
def specialization_heatmap(draft, cfg, dump_dir: str, C=None, device="cuda") -> torch.Tensor:
    """(K, n_domains): fraction of each domain's tokens whose top draft group is g,
    under the trained draft's actual routing (hard: C@softmax(d); soft/scratch: R_d(h)
    is context-dependent, so we approximate with the C-collapsed target router for hard
    and the draft router applied to target hiddens for soft/scratch is out of scope here;
    for hard/E1 this is the routing the draft truly uses)."""
    from target.dump import DOMAINS, load_shards

    K = cfg.K
    nd = len(DOMAINS)
    counts = torch.zeros(K, nd, dtype=torch.float64)
    tot = torch.zeros(nd, dtype=torch.float64)
    Ct = C.t().float().to(device) if C is not None else None
    for sd, _ in load_shards(dump_dir):
        d = sd["router_star"].float().to(device)
        dom = sd["domain_id"].long()
        if Ct is not None:
            g = (torch.softmax(d, -1) @ Ct).argmax(-1).cpu()     # hard reuse routing
        else:
            continue  # soft/scratch routing needs draft hiddens; measured during eval
        for di in range(nd):
            m = dom == di
            if m.any():
                counts[:, di].scatter_add_(0, g[m], torch.ones(int(m.sum()), dtype=torch.float64))
                tot[di] += int(m.sum())
    return counts / (tot[None, :] + 1e-12)
