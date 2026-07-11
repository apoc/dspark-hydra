#!/usr/bin/env bash
# Phase-5 power re-eval (exploratory / DSpark-faithful). Runs on Spark, inside tmux:
#   tmux new -d -s power 'bash scripts/eval_power.sh 2>&1 | tee logs/power.log'
# Env knobs (probe vs full): PER_DOMAIN, MAXNEW, SKIP, SEEDS, WITH_BAL, VARIANTS.
#   probe:  PER_DOMAIN=2 SEEDS="0" MAXNEW=32 WITH_BAL=0 VARIANTS="B3_dense" bash scripts/eval_power.sh
set -uo pipefail

PY=${PY:-$HOME/devel/vllm/venv/bin/python}
PER_DOMAIN=${PER_DOMAIN:-100}
MAXNEW=${MAXNEW:-64}
SKIP=${SKIP:-200}
SEEDS=${SEEDS:-0 1 2}
WITH_BAL=${WITH_BAL:-1}
REPORTS=reports
COACT=data/collapse/coact_k16/C.safetensors
BALC=data/collapse/coact_k16_bal/C.safetensors
mkdir -p "$REPORTS" logs

# fast pinned-load prep: warm the HF blob page cache (unpinned H2D is ~100x slower)
echo "[warm] priming HF blob cache..."
ls "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/blobs/* \
  | xargs -P8 -I{} cat {} >/dev/null 2>&1 || true

# name  ckpt  C-flag
declare -a ROWS=(
  "B3_dense|ckpts/B3_dense/draft.pt|"
  "E1_hard|ckpts/E1_hard/draft.pt|--C $COACT"
  "E2_soft|ckpts/E2_soft/draft.pt|--C $COACT"
  "C1_scratch|ckpts/C1_scratch/draft.pt|"
)
if [ "$WITH_BAL" = "1" ]; then
  ROWS+=("E1_hard_bal|ckpts_bal/E1_hard/draft.pt|--C $BALC")
  ROWS+=("E2_soft_bal|ckpts_bal/E2_soft/draft.pt|--C $BALC")
fi

VARS=""
for row in "${ROWS[@]}"; do
  IFS='|' read -r name ckpt cflag <<< "$row"
  # allow VARIANTS env to restrict which rows run
  if [ -n "${VARIANTS:-}" ] && [[ " $VARIANTS " != *" $name "* ]]; then continue; fi
  echo "=== eval $name (ckpt=$ckpt ${cflag:-no-C}) ==="
  t0=$(date +%s)
  # shellcheck disable=SC2086
  if $PY scripts/eval_tau.py --ckpt "$ckpt" $cflag \
        --per-domain "$PER_DOMAIN" --max-new "$MAXNEW" --skip "$SKIP" --seeds $SEEDS \
        --out "$REPORTS/tau_${name}_power.json"; then
    echo "    $name ok ($(( $(date +%s) - t0 ))s)"
    VARS="$VARS $name"            # only successful evals enter analysis
  else
    rc=$?
    echo "    !! $name FAILED (rc=$rc) after $(( $(date +%s) - t0 ))s -- excluded from analysis"
    rm -f "$REPORTS/tau_${name}_power.json"
  fi
done

# analysis (needs the reference present)
if [[ " $VARS " == *" B3_dense "* ]]; then
  echo "=== power_stats ==="
  # shellcheck disable=SC2086
  $PY eval/power_stats.py --dir "$REPORTS" --ref B3_dense --variants $VARS \
      --out "$REPORTS/power_summary"
fi
echo "DONE"
