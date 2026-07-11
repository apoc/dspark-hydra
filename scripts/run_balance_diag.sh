#!/usr/bin/env bash
# Balance-constrained-C diagnostic (run in tmux on Spark).
# Rebuilds C with capacity-constrained clustering (§5.4, zero starved groups), retrains
# E1_hard + E2_soft on it (saturation-stopped), re-evals per-domain tau, and compares to
# the original (unbalanced) run. B3/C1 unaffected (they don't use C). Outputs to separate
# dirs so the original results stay intact.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/devel/vllm/venv/bin/python
DUMP=${DUMP:-data/calib_v1}
CBAL=${CBAL:-data/collapse/coact_k16_bal}

echo "[$(date +%T)] warming model page cache"
ls ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/blobs/* | xargs -P8 -I{} cat {} >/dev/null || true

echo "[$(date +%T)] building balance-constrained C -> $CBAL"
$PY scripts/build_C.py --dump "$DUMP" --method co_activation --K 16 --balance --warm-init --out "$CBAL"

for v in E1_hard E2_soft; do
  echo "[$(date +%T)] TRAIN $v (balanced C)"
  $PY scripts/train_draft.py --variant "$v" --dump "$DUMP" --C "$CBAL/C.safetensors" --warm-init \
      --patience 6 --min-steps 1000 --out "ckpts_bal/$v"
  echo "[$(date +%T)] EVAL $v (balanced C)"
  $PY scripts/eval_tau.py --ckpt "ckpts_bal/$v/draft.pt" --C "$CBAL/C.safetensors" \
      --per-domain 12 --max-new 64 --out "reports/tau_${v}_bal.json"
done

echo "[$(date +%T)] DONE. balanced-C results:"
for v in E1_hard E2_soft; do
  $PY -c "import json;r=json.load(open('reports/tau_${v}_bal.json'))['results'];print('$v BAL', {k:(round(x['mean_tau'],3) if isinstance(x,dict) else round(x,3)) for k,x in r.items()})"
done
echo "compare prose vs original: E1 0.513 / E2 0.570 (unbalanced) ; B3 dense 0.533"
