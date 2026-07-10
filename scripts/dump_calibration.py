"""Phase 1: run target over calibration corpus; dump per-token:
  - pre-LM-head hidden (dim 2048, bf16)
  - full-attn layer hiddens for the KV-injection set
  - router logits at ℓ* (and aggregate)
  - target next-token distribution (top-p truncated or full subset)
  - sampled token

Output: sharded .pt / .safetensors files + a loader.

Usage:
    python scripts/dump_calibration.py --config configs/model.yaml --out-dir data/calib/
"""
# TODO (Phase 1): implement instrumentation hooks and dump loop
