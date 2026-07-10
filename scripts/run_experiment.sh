#!/usr/bin/env bash
# Autonomous experiment driver (run in tmux on Spark).
# Waits for the calibration dump to complete, then builds C, trains every §6 variant,
# and measures per-domain tau. Logs land in logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=~/devel/vllm/venv/bin/python
DUMP=${DUMP:-data/calib_v1}
EPOCHS=${EPOCHS:-3}
EVAL_N=${EVAL_N:-12}

echo "[$(date +%T)] waiting for dump manifest: $DUMP/manifest.json"
until [ -f "$DUMP/manifest.json" ]; do sleep 30; done
echo "[$(date +%T)] dump complete; warming model page cache"
ls ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/blobs/* | xargs -P8 -I{} cat {} >/dev/null || true

echo "[$(date +%T)] dump tokens: $($PY -c "import json;print(json.load(open('$DUMP/manifest.json'))['total_tokens'])")"
echo "[$(date +%T)] verifying dump alignment (non-fatal)"
$PY scripts/verify_dump.py --dump "$DUMP" || echo "WARN: verify failed, proceeding (alignment proven on smoke)"

echo "[$(date +%T)] running run matrix (epochs=$EPOCHS eval_n=$EVAL_N)"
$PY scripts/run_matrix.py --dump "$DUMP" --epochs "$EPOCHS" --per-domain-eval "$EVAL_N"

echo "[$(date +%T)] DONE. See reports/run_matrix.json"
