"""Method 2 (§5.2) — weight-similarity clustering (warm-init bonus).

Cluster the 256 target expert FFNs at layer l* into K groups by a compact SVD-spectrum
descriptor, and return per-group weight centroids so each draft expert can be warm-
initialized from its cluster centroid (valid when draft expert width mirrors the target,
moe_intermediate_size=512).
"""

from __future__ import annotations

import json
import os

import torch
from safetensors import safe_open

from .kmeans import kmeans, labels_to_C


def _expert_weight_keys(layer: int) -> tuple[str, str]:
    base = f"model.language_model.layers.{layer}.mlp.experts"
    return f"{base}.gate_up_proj", f"{base}.down_proj"


def load_expert_weights(model_path: str, layer: int, device: str = "cpu"):
    """Return (gate_up:(E,2I,H), down:(E,H,I)) for the experts at `layer`."""
    model_path = os.path.expanduser(model_path)
    wm = json.load(open(os.path.join(model_path, "model.safetensors.index.json")))["weight_map"]
    gk, dk = _expert_weight_keys(layer)
    out = {}
    for key in (gk, dk):
        with safe_open(os.path.join(model_path, wm[key]), framework="pt", device=device) as f:
            out[key] = f.get_tensor(key).float()
    return out[gk], out[dk]


def svd_descriptor(gate_up: torch.Tensor, down: torch.Tensor, r: int = 16) -> torch.Tensor:
    """Per-expert descriptor: top-r singular values of each matrix + Frobenius norms."""
    E = gate_up.shape[0]
    feats = []
    for e in range(E):
        sg = torch.linalg.svdvals(gate_up[e])[:r]
        sd = torch.linalg.svdvals(down[e])[:r]
        feats.append(torch.cat([sg, sd, gate_up[e].norm().reshape(1), down[e].norm().reshape(1)]))
    x = torch.stack(feats)
    return (x - x.mean(0)) / (x.std(0) + 1e-6)  # standardize


def build_weight_C(model_path: str, layer: int, k: int, num_experts: int = 256, r: int = 16, seed: int = 0):
    """Cluster experts by weight similarity. Returns (C:(k,E), info incl. centroids)."""
    gate_up, down = load_expert_weights(model_path, layer)
    desc = svd_descriptor(gate_up, down, r=r)
    labels, _ = kmeans(desc, k, seed=seed)
    C = labels_to_C(labels, k, num_experts)
    # weight-space centroids for warm-init (mean of member experts)
    cent_gate_up = torch.stack([gate_up[labels == g].mean(0) if (labels == g).any() else gate_up.mean(0) for g in range(k)])
    cent_down = torch.stack([down[labels == g].mean(0) if (labels == g).any() else down.mean(0) for g in range(k)])
    info = {
        "method": "weight_cluster",
        "labels": labels,
        "group_sizes": [int((labels == g).sum().item()) for g in range(k)],
        "centroids": {"gate_up_proj": cent_gate_up.to(torch.bfloat16), "down_proj": cent_down.to(torch.bfloat16)},
    }
    return C, info
