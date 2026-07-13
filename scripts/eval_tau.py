"""Phase 5: offline accepted-length (tau) per domain for a trained draft (§8.1).

Loads the full target (verification) + a draft checkpoint, streams held-out prompts per
domain, runs speculative decoding (scheduler OFF, fixed gamma, temp 1.0), and reports
mean tau per domain + macro-average. Lossless by construction (§8.3 gate covers this).

Run on Spark (tmux): $PY scripts/eval_tau.py --ckpt ckpts/E1_hard/draft.pt \
    --C data/collapse/coact_k16/C.safetensors --per-domain 20 --out reports/tau_E1_hard.json
"""

from __future__ import annotations

import argparse
import hashlib
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
from eval.analysis import position_wise_acceptance  # noqa: E402
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
    ap.add_argument("--per-domain", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--skip", type=int, default=200,
                    help="confirmation-block offset; MUST clear dump[0:100] and any prior eval block")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="seed replicates; per-(domain,prompt,replicate) RNG derived deterministically")
    ap.add_argument("--out", default=None)
    ap.add_argument("--d-source", choices=["star", "agg"], default="star",
                    help="routing descriptor source (RQ5); MUST match the ckpt's training d-source / C router_source")
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

    def derive_seed(dom, pi, s):
        # deterministic, order-independent, checkpoint-aligned RNG per (domain, prompt, replicate)
        return int(hashlib.sha256(f"{dom}:{pi}:{s}".encode()).hexdigest()[:8], 16)

    results = {}
    if args.out and Path(args.out).exists():
        try:
            prev = json.load(open(args.out)).get("results", {})
            results = {k: v for k, v in prev.items() if k != "macro_avg_tau"}
            print(f"resuming eval: {list(results)} already done", flush=True)
        except Exception:
            results = {}

    def _save():
        if not args.out:
            return
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"ckpt": args.ckpt, "C": args.C, "gamma": cfg.gamma, "max_new": args.max_new,
                   "skip": args.skip, "per_domain": args.per_domain, "seeds": args.seeds,
                   "results": results}, open(args.out, "w"), indent=2)
    for dom, dspec in corpus["domains"].items():
        if dom in results:
            continue
        mode = dspec.get("mode", "chat")
        pool = stream_prompts(dspec["hf"], args.skip + args.per_domain,
                              strip_gutenberg_flag=dspec.get("strip_gutenberg", False))
        prompts = pool[args.skip:args.skip + args.per_domain]   # disjoint confirmation block; NO tail wrap
        if len(prompts) < args.per_domain:
            print(f"[{dom}] WARNING: {len(prompts)} held-out prompts past skip={args.skip} "
                  f"(wanted {args.per_domain}) -- reduced-N block", flush=True)
        tau_by_prompt, naccs = [], []
        td = tv = ta = 0.0
        nrounds = 0
        gen_tokens = 0.0
        for pi, p in enumerate(prompts):
            ids = build_prompt_ids(target, p, mode, dspec.get("prompt_tokens", 128))
            per_seed = []
            for s in args.seeds:
                gen = torch.Generator(device=device).manual_seed(derive_seed(dom, pi, s))
                _, t, na, tim = spec_decode(target, draft, cfg, ids, C=C, max_new=args.max_new,
                                            inject_layers=inj, l_star=l_star, gen=gen, time_it=True,
                                            d_source=args.d_source)
                td += tim["t_draft"]; tv += tim["t_verify"]; ta += tim["t_anchor"]
                nrounds += tim["rounds"]
                if t:
                    per_seed.append(sum(t) / len(t))
                    gen_tokens += sum(t)              # total generated tokens (for weight-consistent L)
                    naccs.extend(na)
            tau_by_prompt.append(per_seed)               # aligned with prompt index pi
        prompt_means = [sum(ps) / len(ps) for ps in tau_by_prompt if ps]
        mean = sum(prompt_means) / max(len(prompt_means), 1)
        posacc = position_wise_acceptance(naccs, cfg.gamma) if naccs else []
        results[dom] = {
            "mean_tau": mean,
            "norm_tau": mean / (cfg.gamma + 1),
            "n_prompts": len(prompt_means),
            "n_seeds": len(args.seeds),
            "tau_by_prompt_seed": tau_by_prompt,   # len==#prompts; per-seed list each (raw, paired bootstrap)
            "position_acceptance": posacc,
            "latency_ms": {"t_anchor": 1e3 * ta / max(nrounds, 1),
                           "t_draft": 1e3 * td / max(nrounds, 1),
                           "t_verify": 1e3 * tv / max(nrounds, 1), "rounds": nrounds,
                           # measured offline latency of THIS engine: it runs BOTH an anchor
                           # forward and a verify forward per round. The anchor signal is a
                           # residual/bonus token absent from the prior verify input, so it
                           # cannot be cached away -> no cached-engine projection reported.
                           "gen_tokens": gen_tokens,
                           "L_offline_ms_per_tok": 1e3 * (ta + td + tv) / max(gen_tokens, 1e-9)},
        }
        lat = results[dom]["latency_ms"]
        print(f"[{dom}] mean_tau={mean:.3f} norm={mean / (cfg.gamma + 1):.3f} "
              f"n={len(prompt_means)}x{len(args.seeds)} pos_acc={['%.2f' % x for x in posacc]} "
              f"t_draft={lat['t_draft']:.1f}ms t_verify={lat['t_verify']:.1f}ms", flush=True)
        _save()

    macro = sum(v["mean_tau"] for v in results.values()) / max(len(results), 1)
    results["macro_avg_tau"] = macro
    print(f"\nMACRO-AVG tau = {macro:.3f}")
    _save()
    if args.out:
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
