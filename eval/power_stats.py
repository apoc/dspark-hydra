"""Exploratory (DSpark-faithful) power analysis of tau across draft variants.

No pass/fail gate. Reports, per domain and pooled-macro:
  - mean tau per variant (+ normalized tau/(gamma+1), since gamma=5 != paper block 7),
  - PAIRED contrast Delta = variant - reference per prompt (same prompts/order across
    variants), with a cluster-bootstrap-by-prompt 95% CI (resamples prompts),
  - relative gain Delta/tau_ref in %, with a labeled 5% reference line (NOT a threshold),
  - measured per-round latency + cached-engine projection.

Pairing unit = prompt; per-prompt tau is the mean over the fixed seed replicates, so the
bootstrap resamples prompts only and inference is CONDITIONAL on those decoding seeds.

Usage: $PY eval/power_stats.py --dir reports --ref B3_dense \
    --variants B3_dense E1_hard E2_soft C1_scratch --out reports/power_summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_SKIP_KEYS = {"macro_avg_tau"}


def load(dir_: Path, variant: str) -> dict:
    p = dir_ / f"tau_{variant}_power.json"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    return json.loads(p.read_text())


def domains(res: dict) -> list[str]:
    return [k for k in res["results"] if k not in _SKIP_KEYS]


def prompt_taus(res: dict, dom: str) -> np.ndarray:
    """Per-prompt tau = mean over seeds; NaN where a prompt produced no rounds."""
    rows = res["results"][dom]["tau_by_prompt_seed"]
    return np.array([np.mean(r) if r else np.nan for r in rows], dtype=float)


def paired_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Align by prompt index (truncate to min length), drop NaN in either."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    return a[m] - b[m]


def boot_ci(delta: np.ndarray, nboot: int = 10000, seed: int = 0):
    if delta.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, delta.size, size=(nboot, delta.size))
    means = delta[idx].mean(axis=1)
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reports")
    ap.add_argument("--ref", default="B3_dense")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--ref-line", type=float, default=0.05, help="labeled relative reference line (not a gate)")
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--out", default=None, help="path prefix -> writes .md and .json")
    args = ap.parse_args()

    dir_ = Path(args.dir)
    data = {v: load(dir_, v) for v in args.variants}
    ref = data[args.ref]
    doms = domains(ref)
    gamma = ref["gamma"]

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"# Power analysis (exploratory, DSpark-faithful) - ref={args.ref}, gamma={gamma}")
    emit(f"tau = accepted length incl. bonus token (DSpark Sec4.1 fn4); norm = tau/(gamma+1)={gamma+1}.")
    emit(f"Contrast = paired per-prompt Delta (variant-ref); CI = cluster-bootstrap-by-prompt (prompts resampled,")
    emit(f"conditional on the fixed decoding seeds {ref['seeds']}). {args.ref_line*100:.0f}% line is a REFERENCE, not a gate.\n")

    # --- per-variant tau + latency table ---
    emit("## Mean tau + latency per variant")
    emit("| variant | domain | n | tau | norm | t_anchor | t_draft | t_verify | L_offline ms/tok |")
    emit("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for v in args.variants:
        r = data[v]["results"]
        for dom in doms:
            d = r[dom]
            lat = d.get("latency_ms", {})
            emit(f"| {v} | {dom} | {d['n_prompts']} | {d['mean_tau']:.3f} | {d['norm_tau']:.3f} | "
                 f"{lat.get('t_anchor', float('nan')):.1f} | {lat.get('t_draft', float('nan')):.1f} | "
                 f"{lat.get('t_verify', float('nan')):.1f} | {lat.get('L_offline_ms_per_tok', float('nan')):.2f} |")
        macro = sum(r[dm]["mean_tau"] for dm in doms) / len(doms)
        emit(f"| {v} | **macro** | - | **{macro:.3f}** | {macro/(gamma+1):.3f} | | | | |")
    emit("")

    # --- paired contrasts vs ref ---
    summary = {"ref": args.ref, "gamma": gamma, "seeds": ref["seeds"], "ref_line": args.ref_line, "contrasts": {}}
    emit(f"## Paired contrasts vs {args.ref} (Delta = variant - {args.ref})")
    emit("| variant | domain | mean Delta | rel% | 95% CI (abs) | CI lower rel% | >ref-line? |")
    emit("|---|---|--:|--:|:--:|--:|:--:|")
    for v in args.variants:
        if v == args.ref:
            continue
        summary["contrasts"][v] = {}
        pooled = []
        for dom in doms:
            a = prompt_taus(data[v], dom)
            b = prompt_taus(ref, dom)
            delta = paired_delta(a, b)
            pooled.append(delta)
            tau_ref = np.nanmean(prompt_taus(ref, dom))
            mean, lo, hi = boot_ci(delta, args.nboot)
            rel = 100 * mean / tau_ref if tau_ref else float("nan")
            rel_lo = 100 * lo / tau_ref if tau_ref else float("nan")
            flag = "yes" if rel_lo >= args.ref_line * 100 else ("~" if rel >= args.ref_line * 100 else "no")
            emit(f"| {v} | {dom} | {mean:+.3f} | {rel:+.1f}% | [{lo:+.3f}, {hi:+.3f}] | {rel_lo:+.1f}% | {flag} |")
            summary["contrasts"][v][dom] = {"mean": mean, "ci": [lo, hi], "rel_pct": rel,
                                            "ci_lower_rel_pct": rel_lo, "n": int(delta.size)}
        # pooled-macro (all prompts across domains as paired units)
        alld = np.concatenate(pooled) if pooled else np.array([])
        macro_ref = np.mean([np.nanmean(prompt_taus(ref, dm)) for dm in doms])
        mean, lo, hi = boot_ci(alld, args.nboot)
        rel = 100 * mean / macro_ref if macro_ref else float("nan")
        rel_lo = 100 * lo / macro_ref if macro_ref else float("nan")
        flag = "yes" if rel_lo >= args.ref_line * 100 else ("~" if rel >= args.ref_line * 100 else "no")
        emit(f"| {v} | **macro (pooled)** | {mean:+.3f} | {rel:+.1f}% | [{lo:+.3f}, {hi:+.3f}] | {rel_lo:+.1f}% | {flag} |")
        summary["contrasts"][v]["macro_pooled"] = {"mean": mean, "ci": [lo, hi], "rel_pct": rel,
                                                    "ci_lower_rel_pct": rel_lo, "n": int(alld.size)}
    emit("")
    emit("Flag: 'yes' = CI lower bound clears the reference line; '~' = point estimate clears but CI does not;")
    emit("'no' = below. This is descriptive only (exploratory report; no accept/reject decision).")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out + ".md").write_text("\n".join(lines) + "\n")
        Path(args.out + ".json").write_text(json.dumps(summary, indent=2))
        print(f"\nsaved -> {args.out}.md / {args.out}.json")


if __name__ == "__main__":
    main()
