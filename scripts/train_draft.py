"""Phase 4: train one draft variant (§6 run matrix).

Borrows the target's frozen embedding + LM head, loads the collapse map C for hard/soft,
warm-inits MoE experts from centroids if present, and trains on the dump.

Run on Spark (tmux for real runs):
    $PY scripts/train_draft.py --variant E1_hard --dump data/calib_v1 \
        --C data/collapse/coact_k16/C.safetensors --epochs 10 --out ckpts/E1_hard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from draft.config import DraftConfig  # noqa: E402
from draft.model import DraftModel  # noqa: E402
from target.loader import load_embed_lm_head  # noqa: E402
from train.loop import train_draft  # noqa: E402

VARIANT_MODE = {
    "B3_dense": "dense", "E1_hard": "hard", "E2_soft": "soft", "C1_scratch": "scratch",
}


def warm_init_experts(model, C_tensors):
    """Copy per-group weight centroids into draft experts (if width mirrors target)."""
    if "centroid_gate_up_proj" not in C_tensors:
        return 0
    cg = C_tensors["centroid_gate_up_proj"]  # (K, 2I, H)
    cd = C_tensors["centroid_down_proj"]     # (K, H, I)
    n = 0
    for layer in model.backbone.layers:
        moe = layer.ffn
        if not hasattr(moe, "experts"):
            continue
        for g, expert in enumerate(moe.experts):
            if expert.gate_up.weight.shape == cg[g].shape and expert.down.weight.shape == cd[g].shape:
                with torch.no_grad():
                    expert.gate_up.weight.copy_(cg[g].to(expert.gate_up.weight.dtype))
                    expert.down.weight.copy_(cd[g].to(expert.down.weight.dtype))
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANT_MODE), required=True)
    ap.add_argument("--dump", default=str(_REPO_ROOT / "data" / "calib_v1"))
    ap.add_argument("--C", default=None, help="collapse map C.safetensors (hard/soft)")
    ap.add_argument("--warm-init", action="store_true")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--k-prime", type=int, default=2)
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mode = VARIANT_MODE[args.variant]
    print(f"loading frozen embed + LM head (~2GB) ...", flush=True)
    embed, lm_head, tc = load_embed_lm_head(device="cuda")
    device = embed.weight.device
    dtype = embed.weight.dtype

    cfg = DraftConfig(
        hidden_size=tc.hidden_size, vocab_size=tc.vocab_size,
        n_layers=args.n_layers, gamma=args.gamma, K=args.K, k_prime=args.k_prime,
        moe_intermediate_size=tc.moe_intermediate_size,
        num_target_experts=tc.num_experts, router_mode=mode,
    )
    model = DraftModel(cfg).to(device, dtype)

    C = None
    if mode in ("hard", "soft"):
        assert args.C, f"{args.variant} needs --C"
        Ct = load_file(args.C)
        C = Ct["C"].to(device, dtype)
        if args.warm_init:
            n = warm_init_experts(model, Ct)
            print(f"warm-initialized {n} experts from centroids")

    out = train_draft(model, cfg, args.dump, embed, lm_head, C,
                      device=device, epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, max_steps=args.max_steps)

    save_dir = Path(args.out or (_REPO_ROOT / "ckpts" / args.variant))
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": out.state_dict(), "cfg": vars(cfg)}, save_dir / "draft.pt")
    print(f"saved -> {save_dir / 'draft.pt'}")


if __name__ == "__main__":
    main()
