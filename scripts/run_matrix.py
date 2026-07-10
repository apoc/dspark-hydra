"""Phase 4-5 orchestrator: build C, train every §6 variant, eval per-domain tau, compare.

Runs each step as a subprocess for isolation. Intended for a long tmux run once the
calibration dump is complete (has manifest.json).

Run on Spark (tmux):
    $PY scripts/run_matrix.py --dump data/calib_v1 --epochs 10 --per-domain-eval 20
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# variant -> needs C?
VARIANTS = [
    ("B3_dense", False),
    ("E1_hard", True),
    ("E2_soft", True),
    ("C1_scratch", False),
]


def run(cmd: list[str], log):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"  exit={p.returncode} (log: {log})", flush=True)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(_REPO_ROOT / "data" / "calib_v1"))
    ap.add_argument("--C", default=str(_REPO_ROOT / "data" / "collapse" / "coact_k16"))
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--per-domain-eval", type=int, default=20)
    ap.add_argument("--variants", nargs="*", default=[v for v, _ in VARIANTS])
    args = ap.parse_args()

    assert (Path(args.dump) / "manifest.json").exists(), f"dump not complete: {args.dump}"
    logs = _REPO_ROOT / "logs"; logs.mkdir(exist_ok=True)
    reports = _REPO_ROOT / "reports"; reports.mkdir(exist_ok=True)
    ckdir = _REPO_ROOT / "ckpts"
    C_file = Path(args.C) / "C.safetensors"

    # Phase 2: build C (co-activation + warm-init) if missing
    if not C_file.exists():
        run([PY, "scripts/build_C.py", "--dump", args.dump, "--method", "co_activation",
             "--K", str(args.K), "--warm-init", "--out", args.C], logs / "build_C.log")

    summary = {}
    for variant in args.variants:
        needs_C = dict(VARIANTS)[variant]
        out = ckdir / variant
        cmd = [PY, "scripts/train_draft.py", "--variant", variant, "--dump", args.dump,
               "--K", str(args.K), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
               "--out", str(out)]
        if needs_C:
            cmd += ["--C", str(C_file), "--warm-init"]
        rc = run(cmd, logs / f"train_{variant}.log")
        if rc != 0:
            summary[variant] = {"error": "train failed"}
            continue

        tau_out = reports / f"tau_{variant}.json"
        ecmd = [PY, "scripts/eval_tau.py", "--ckpt", str(out / "draft.pt"),
                "--per-domain", str(args.per_domain_eval), "--out", str(tau_out)]
        if needs_C:
            ecmd += ["--C", str(C_file)]
        run(ecmd, logs / f"eval_{variant}.log")
        if tau_out.exists():
            summary[variant] = json.load(open(tau_out))["results"]

    # comparison table
    print("\n=== RUN MATRIX SUMMARY (macro-avg tau) ===")
    for v, r in summary.items():
        macro = r.get("macro_avg_tau", "n/a") if isinstance(r, dict) else "n/a"
        print(f"  {v:12s} macro_tau={macro}")
    json.dump(summary, open(reports / "run_matrix.json", "w"), indent=2)
    print(f"\nsaved -> {reports/'run_matrix.json'}")


if __name__ == "__main__":
    main()
