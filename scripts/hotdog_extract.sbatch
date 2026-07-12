#!/bin/bash
# Phase C stage 2: sharded, preempt-safe HF teacher-force extraction on hotdog A100.
# Submit AFTER stage-1 generation completes:
#   cd ~/devel/dspark-hydra && mkdir -p logs
#   sbatch --array=0-15 --export=ALL,NSHARDS=16 scripts/hotdog_extract.sbatch
#   # then merge:  ~/devel/finetune-venv/bin/python scripts/merge_dump.py --dump data/calib_v3
# Each array task = one part on one A100 (HF, finetune-venv). --requeue -> a preempted part
# is redone from scratch (its manifest.json is only written on completion, so it self-skips
# once done). Keep parts small (large NSHARDS) so a preemption redoes <=1 part.
#SBATCH -p a100-preempt
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH -J ext_v3
#SBATCH -o logs/ext_v3_%a.log
set -euo pipefail
cd "$HOME/devel/dspark-hydra"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${NSHARDS:-16}"
mkdir -p data/calib_v3
echo "=== extract part $SHARD/$NSHARDS on $(hostname) $(date) ==="
~/devel/finetune-venv/bin/python -u scripts/dump_from_responses.py \
  --responses data/gen_v3 --shard "$SHARD" --nshards "$NSHARDS" \
  --out "data/calib_v3/part_${SHARD}"
echo "=== part $SHARD done $(date) ==="
