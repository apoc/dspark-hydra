"""Phase 2: build the 256->K expert-collapse map C (§5).

Methods: co_activation (default), weight_cluster, learned (init only here; trained in
Phase 4). Optionally attach weight-space warm-init centroids aligned to C's own groups
(co-activation routing + weight init, the §5.2 combined mode). Emits C, centroids,
balance stats, and a domain-overlap report.

Run on Spark:
    ~/devel/vllm/venv/bin/python scripts/build_C.py --dump data/calib_v1 --method co_activation \
        --K 16 --k-prime 2 --warm-init --out data/collapse/coact_k16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from collapse.balance import group_load  # noqa: E402
from collapse.coactivation import build_coactivation_C  # noqa: E402
from collapse.report import domain_overlap, group_domain_composition, per_domain_gate_mass  # noqa: E402
from collapse.weight_cluster import build_weight_C, load_expert_weights  # noqa: E402
from target.dump import DOMAINS  # noqa: E402


def warm_init_centroids(model_path: str, layer: int, labels: torch.Tensor, k: int):
    """Weight-space centroid per C group (mean of member experts) for draft warm-init."""
    gate_up, down = load_expert_weights(model_path, layer)
    cg = torch.stack([gate_up[labels == g].mean(0) if (labels == g).any() else gate_up.mean(0) for g in range(k)])
    cd = torch.stack([down[labels == g].mean(0) if (labels == g).any() else down.mean(0) for g in range(k)])
    return {"gate_up_proj": cg.to(torch.bfloat16), "down_proj": cd.to(torch.bfloat16)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(_REPO_ROOT / "data" / "calib_smoke"))
    ap.add_argument("--method", choices=["co_activation", "weight_cluster"], default="co_activation")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--k-prime", type=int, default=2)
    ap.add_argument("--warm-init", action="store_true")
    ap.add_argument("--balance", action="store_true", help="capacity-constrained clustering (§5.4)")
    ap.add_argument("--out", default=str(_REPO_ROOT / "data" / "collapse" / "coact_k16"))
    ap.add_argument("--model", default=None, help="model path for weight clustering / warm-init")
    args = ap.parse_args()

    mcfg = yaml.safe_load(open(_REPO_ROOT / "configs" / "model.yaml"))
    ccfg = yaml.safe_load(open(_REPO_ROOT / "configs" / "corpus.yaml"))
    l_star = ccfg["dump"]["l_star"]
    num_experts = mcfg["target"]["num_experts"]
    top_k = mcfg["target"]["num_experts_per_tok"]
    import os
    model_path = os.path.expanduser(args.model or mcfg["target"]["local_path"])

    print(f"building C: method={args.method} K={args.K} l*={l_star} dump={args.dump}")
    if args.method == "co_activation":
        C, info = build_coactivation_C(args.dump, args.K, top_k=top_k, num_experts=num_experts, balanced=args.balance)
    else:
        C, info = build_weight_C(model_path, l_star, args.K, num_experts=num_experts)

    labels = info["labels"]
    print("group sizes:", info["group_sizes"])
    assert (C.sum(0) == 1).all(), "C is not a partition (each expert in exactly one group)"
    empty = [g for g, s in enumerate(info["group_sizes"]) if s == 0]
    print("empty groups:", empty or "none")

    tensors = {"C": C, "labels": labels.to(torch.int32)}
    if args.warm_init or args.method == "weight_cluster":
        cents = info.get("centroids") or warm_init_centroids(model_path, l_star, labels, args.K)
        tensors["centroid_gate_up_proj"] = cents["gate_up_proj"]
        tensors["centroid_down_proj"] = cents["down_proj"]
        print(f"warm-init centroids: gate_up{tuple(cents['gate_up_proj'].shape)} down{tuple(cents['down_proj'].shape)}")

    bal = group_load(args.dump, C, k_prime=args.k_prime)
    print(f"balance: cv_freq={bal['cv_freq']:.3f} min={bal['min_freq']:.4f} max={bal['max_freq']:.4f} "
          f"ideal={bal['ideal_freq']:.4f} starved={bal['starved_groups']}")

    mass = per_domain_gate_mass(args.dump, num_experts=num_experts)
    overlap = domain_overlap(mass)
    print("domain hot-set sizes:", overlap["hot_set_sizes"])
    print("domain Jaccard overlap @80%:", overlap["jaccard"])
    comp = group_domain_composition(args.dump, C, k_prime=args.k_prime)
    print("group x domain composition (rows=groups, cols=" + ",".join(DOMAINS) + "):")
    for g in range(args.K):
        row = "  ".join(f"{comp[g, d]:.2f}" for d in range(len(DOMAINS)))
        print(f"  g{g:02d}: {row}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "C.safetensors"))
    report = {
        "method": args.method, "K": args.K, "k_prime": args.k_prime, "l_star": l_star,
        "dump": args.dump, "group_sizes": info["group_sizes"], "empty_groups": empty,
        "balance": {k: (v if not torch.is_tensor(v) else v.tolist()) for k, v in bal.items()},
        "domain_overlap": overlap,
        "group_domain_composition": comp.tolist(),
    }
    json.dump(report, open(out / "report.json", "w"), indent=2)
    print(f"\nsaved C + report -> {out}")


if __name__ == "__main__":
    main()
