#!/bin/bash
# Phase C stage 1: sharded, preempt-safe vLLM generation on hotdog A100.
# Submit:  cd ~/devel/dspark-hydra && mkdir -p logs
#          sbatch --array=0-3 --export=ALL,NSHARDS=4,PERDOMAIN=5000 scripts/hotdog_gen.sbatch
# Each array task = one data-parallel shard on one A100 (--tp 1). --requeue -> on preemption
# Slurm requeues and gen_responses.py resumes from its per-shard JSONL (skips done uids).
#SBATCH -p a100-preempt
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=110G
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH -J gen_v3
#SBATCH -o logs/gen_v3_%a.log
set -euo pipefail
cd "$HOME/devel/dspark-hydra"
export PATH="$HOME/devel/vllm-venv/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_FLASHINFER=0
MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${NSHARDS:-4}"
PERDOMAIN="${PERDOMAIN:-5000}"
mkdir -p data/gen_v3
echo "=== gen shard $SHARD/$NSHARDS perdomain=$PERDOMAIN on $(hostname) $(date) ==="
python -u scripts/gen_responses.py \
  --per-domain "$PERDOMAIN" --tp 1 --enforce-eager --max-model-len 4096 \
  --chunk 1024 --shard "$SHARD" --nshards "$NSHARDS" \
  --model "$MODEL" \
  --out "data/gen_v3/shard_${SHARD}.jsonl"
echo "=== shard $SHARD done $(date) ==="
