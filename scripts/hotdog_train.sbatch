#!/bin/bash
# Phase C stage 4: train the 4 draft variants on calib_v3 (preempt-safe via loop ckpt/resume).
# Submit AFTER build_C:
#   cd ~/devel/dspark-hydra && mkdir -p logs
#   sbatch --array=0-3 scripts/hotdog_train.sbatch
# array idx -> variant. --requeue + train.loop ckpt_path (ckpts_v3/<V>/ckpt.pt) => a preempted
# run resumes from the last eval (<=200 steps lost); a finished variant (draft.pt) self-skips.
# patience 12 + min-steps 12000 = the autostop fix (learned routers have delayed descent).
#SBATCH -p a100-preempt
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH -J train_v3
#SBATCH -o logs/train_v3_%a.log
set -euo pipefail
cd "$HOME/devel/dspark-hydra"
PY=~/devel/finetune-venv/bin/python
DUMP=data/calib_v3
C=data/collapse/coact_k16_v3/C.safetensors
VARIANTS=(B3_dense E1_hard E2_soft C1_scratch)
V="${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
COMMON="--variant $V --dump $DUMP --n-layers 4 --K 16 --k-prime 2 --gamma 5 \
  --patience 12 --min-steps 12000 --eval-interval 200 --out ckpts_v3/$V"
case "$V" in
  E1_hard|E2_soft) EXTRA="--C $C --warm-init --d-source star" ;;
  *)               EXTRA="" ;;
esac
echo "=== train $V on $(hostname) $(date) ==="
$PY -u scripts/train_draft.py $COMMON $EXTRA
echo "=== $V done $(date) ==="
