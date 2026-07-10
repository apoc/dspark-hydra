"""Benchmark the fast load path for GB10 unified memory.

Finding: unpinned H2D on GB10 is ~0.16 GB/s (pathological); pinned is ~18 GB/s.
So: load on CPU (near-free from warm page cache), then stream each tensor to
CUDA through pinned memory, freeing the CPU copy as we go to stay under 128GB.

Run on Spark:
    ~/devel/vllm/venv/bin/python scripts/bench_load.py
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def peak_gb():
    return torch.cuda.max_memory_allocated() / 1e9


def main():
    from target.loader import load_target, move_to_cuda_pinned, resolve_model_path
    from transformers import AutoModelForImageTextToText

    path = resolve_model_path()

    t = time.time()
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cpu")
    model.eval()
    if hasattr(model.model, "visual"):
        model.model.visual = None
    print(f"CPU load: {time.time()-t:.1f}s")

    t = time.time()
    move_to_cuda_pinned(model)
    torch.cuda.synchronize()
    print(f"pinned stream CPU->CUDA: {time.time()-t:.1f}s; peak GPU alloc={peak_gb():.1f}GB")

    # sanity forward
    from target.hooks import extract
    from target.loader import Target
    tok = __import__("transformers").AutoTokenizer.from_pretrained(path)
    tgt = Target(model=model, tokenizer=tok, text_config=cfg.get_text_config(), path=path)
    ids = tok("Speculative decoding is", return_tensors="pt").input_ids
    t = time.time()
    res = extract(tgt, ids)
    print(f"forward ok in {time.time()-t:.2f}s; router[39]={tuple(res.router_at(39).shape)}")


if __name__ == "__main__":
    main()
