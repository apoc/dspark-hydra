"""Phase 5: offline accepted-length (tau) per domain for a trained draft (§8.1).

Loads the full target (verification) + a draft checkpoint, streams held-out prompts per
domain, runs speculative decoding (scheduler OFF, fixed gamma, temp 1.0), and reports
mean tau per domain + macro-average. Lossless by construction (§8.3 gate covers this).

Run on Spark (tmux): $PY scripts/eval_tau.py --ckpt ckpts/E1_hard/draft.pt \
    --C data/collapse/coact_k16/C.safetensors --per-domain 20 --out reports/tau_E1_hard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from draft.config import DraftConfig  # noqa: E402
from draft.model import DraftModel  # noqa: E402
from eval.spec_decode import spec_decode  # noqa: E402
from target.loader import load_target  # noqa: E402
from target.corpus import stream_prompts  # noqa: E402


def build_prompt_ids(target, text, mode, ptoks):
    tok = target.tokenizer
    if mode == "completion":
        return tok(text, return_tensors="pt", truncation=True, max_length=ptoks).input_ids
    msgs = [{"role": "user", "content": text}]
    try:
        r = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    except TypeError:
        r = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    return r["input_ids"] if isinstance(r, dict) or hasattr(r, "__getitem__") else r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--C", default=None)
    ap.add_argument("--corpus", default=str(_REPO_ROOT / "configs" / "corpus.yaml"))
    ap.add_argument("--per-domain", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--skip", type=int, default=10000, help="offset into stream for held-out prompts")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = DraftConfig(**ckpt["cfg"])
    corpus = yaml.safe_load(open(args.corpus))

    target = load_target()
    device = next(target.model.parameters()).device
    draft = DraftModel(cfg)
    draft.load_state_dict(ckpt["model"])
    draft = draft.to(device, next(target.model.parameters()).dtype).eval()
    C = load_file(args.C)["C"].to(device) if args.C else None
    inj = corpus["dump"]["inject_layers"]
    l_star = corpus["dump"]["l_star"]

    results = {}
    for dom, dspec in corpus["domains"].items():
        mode = dspec.get("mode", "chat")
        prompts = stream_prompts(dspec["hf"], args.skip + args.per_domain,
                                 strip_gutenberg_flag=dspec.get("strip_gutenberg", False))[args.skip:]
        prompts = prompts[:args.per_domain]
        taus = []
        gen = torch.Generator(device=device).manual_seed(args.seed)
        for p in prompts:
            ids = build_prompt_ids(target, p, mode, dspec.get("prompt_tokens", 128))
            _, t = spec_decode(target, draft, cfg, ids, C=C, max_new=args.max_new,
                               inject_layers=inj, l_star=l_star, gen=gen)
            if t:
                taus.append(sum(t) / len(t))
        mean = sum(taus) / max(len(taus), 1)
        results[dom] = {"mean_tau": mean, "n_prompts": len(taus)}
        print(f"[{dom}] mean_tau={mean:.3f} over {len(taus)} prompts", flush=True)

    macro = sum(v["mean_tau"] for v in results.values()) / max(len(results), 1)
    results["macro_avg_tau"] = macro
    print(f"\nMACRO-AVG tau = {macro:.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"ckpt": args.ckpt, "results": results}, open(args.out, "w"), indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
