#!/bin/bash
# Phase C stage 5: powered eval of the 4 v3 variants (sharded by variant).
# Submit AFTER training:
#   sbatch --array=0-3 scripts/hotdog_eval.sbatch
#   # then: power_stats vs B3_dense_v3, and E2 vs C1 (RQ2 primary endpoint)
# n=50/domain x 2 seeds (v2-matched, for a clean v2->v3 comparison), --skip 5000 => held-out
# block DISJOINT from the v3 training range (prompts[0:5000]/domain). --requeue: eval_tau
# resumes per-domain from its own partial JSON (no sbatch-level skip -> partials are safe).
#SBATCH -p a100-preempt
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=150G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH -J eval_v3
#SBATCH -o logs/eval_v3_%a.log
set -euo pipefail
cd "$HOME/devel/dspark-hydra"
PY=~/devel/finetune-venv/bin/python
C=data/collapse/coact_k16_v3/C.safetensors
VARIANTS=(B3_dense E1_hard E2_soft C1_scratch)
V="${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
case "$V" in
  E1_hard|E2_soft) CF="--C $C --d-source star" ;;
  *)               CF="" ;;
esac
OUT="reports/tau_${V}_v3_power.json"
echo "=== eval $V on $(hostname) $(date) -> $OUT ==="
$PY -u scripts/eval_tau.py --ckpt "ckpts_v3/$V/draft.pt" $CF \
  --per-domain 50 --seeds 0 1 --skip 5000 --max-new 32 --d-source star --out "$OUT"
echo "=== $V done $(date) ==="
