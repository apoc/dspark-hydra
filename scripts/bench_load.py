"""Microbench: locate the slow path in target loading on GB10 unified memory.

Measures CPU read bandwidth, H2D transfer (unpinned/pinned), and safetensors
direct-to-cuda, on a few large expert tensors. Run on Spark.
"""
import glob
import os
import time

import torch
from safetensors import safe_open

SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/"
    "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
)


def main():
    shard = sorted(glob.glob(SNAP + "/model-*.safetensors"))[5]

    # CPU read (warm cache)
    t = time.time()
    ts = []
    with safe_open(shard, framework="pt", device="cpu") as f:
        for k in list(f.keys())[:8]:
            ts.append(f.get_tensor(k))
    nb = sum(x.numel() * x.element_size() for x in ts)
    dt = time.time() - t
    print(f"CPU read: {len(ts)} tensors {nb/1e9:.2f}GB in {dt:.2f}s = {nb/1e9/dt:.2f} GB/s")

    big = max(ts, key=lambda x: x.numel())
    gb = big.numel() * big.element_size() / 1e9
    print(f"big tensor {tuple(big.shape)} {big.dtype} = {gb:.2f}GB")

    # H2D unpinned
    torch.cuda.synchronize(); t = time.time()
    _ = big.to("cuda"); torch.cuda.synchronize()
    dt = time.time() - t
    print(f"H2D unpinned: {gb/dt:.3f} GB/s ({dt:.2f}s)")

    # H2D pinned
    t = time.time(); bp = big.pin_memory(); dt_pin = time.time() - t
    torch.cuda.synchronize(); t = time.time()
    _ = bp.to("cuda", non_blocking=True); torch.cuda.synchronize()
    dt = time.time() - t
    print(f"pin_memory: {dt_pin:.2f}s; H2D pinned: {gb/dt:.3f} GB/s ({dt:.2f}s)")

    # safetensors direct to cuda
    torch.cuda.synchronize(); t = time.time()
    with safe_open(shard, framework="pt", device="cuda") as f:
        k = list(f.keys())[0]
        x = f.get_tensor(k)
    torch.cuda.synchronize()
    xb = x.numel() * x.element_size() / 1e9
    dt = time.time() - t
    print(f"safetensors->cuda (1 tensor {xb:.2f}GB): {xb/dt:.3f} GB/s ({dt:.2f}s)")


if __name__ == "__main__":
    main()
