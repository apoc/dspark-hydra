"""Phase C stage 1: vLLM batched target-response generation -> JSONL of token ids.

Fast replacement for the HF `generate` loop (the calibration-dump bottleneck: ~15 s/seq
on GB10). Runs on a multi-GPU box (e.g. 2xA100, --tp 2). Stores prompt/response TOKEN IDS
(not text) so stage 2 (`dump_from_responses.py`, HF teacher-force) aligns byte-exactly --
no re-tokenization drift. Sampling matches configs/corpus.yaml `generation` (the same
distribution the current dump draws from).

Run (hotdog A100, via srun on a *-preempt partition; needs the venv `bin` on PATH for ninja, and
flashinfer disabled — env set at import below). The 35B fits one 80 GB A100, so --tp 1 works:
    srun -p a100-preempt --gres=gpu:1 -c 8 --mem=110G -t 08:00:00 \
        env PATH="$HOME/devel/vllm-venv/bin:$PATH" \
        ~/devel/vllm-venv/bin/python scripts/gen_responses.py --per-domain 5000 --tp 1 --enforce-eager --out data/gen_v2.jsonl
    # smoke: --limit 20 (same env)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

# vLLM on hotdog (multimodal Qwen3.6-35B, driven text-only): use vLLM's native Torch sampler.
# flashinfer's top-k/top-p kernel JIT-compiles against curand.h, which the CUDA-13 install lacks
# (no root). Must be set BEFORE `import vllm` (done in main()).
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER", "0")

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from target.corpus import stream_prompts  # noqa: E402


def build_prompt_ids(tok, text: str, mode: str, prompt_tokens: int, enable_thinking: bool) -> list[int]:
    """Match target/dump.generate_response prompt construction exactly."""
    if mode == "completion":
        return tok(text, truncation=True, max_length=prompt_tokens).input_ids
    msgs = [{"role": "user", "content": text}]
    try:
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, enable_thinking=enable_thinking)
    except TypeError:
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True)
    ids = enc.input_ids if hasattr(enc, "input_ids") else enc["input_ids"]
    return ids[0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(_REPO_ROOT / "configs" / "corpus.yaml"))
    ap.add_argument("--model", default=None, help="override target path (defaults to model.yaml)")
    ap.add_argument("--per-domain", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="total prompts split by domain fraction")
    ap.add_argument("--tp", type=int, default=2, help="tensor-parallel size")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--enforce-eager", action="store_true", help="skip CUDA-graph capture (GB10 workaround)")
    ap.add_argument("--max-num-seqs", type=int, default=256, help="max concurrent sequences")
    ap.add_argument("--chunk", type=int, default=1024, help="generate+flush per chunk (preempt-safe resume unit)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ccfg = yaml.safe_load(open(args.corpus))
    mcfg = yaml.safe_load(open(_REPO_ROOT / "configs" / "model.yaml"))
    model_path = os.path.expanduser(args.model or mcfg["target"]["local_path"])
    gcfg = ccfg["generation"]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # build all prompt-token-id lists first (cheap), then one big batched generate
    recs: list[tuple[str, str, str, list[int]]] = []  # (uid, domain, mode, prompt_ids)
    for dom, dspec in ccfg["domains"].items():
        if args.per_domain is not None:
            n = args.per_domain
        elif args.limit is not None:
            n = max(1, int(round(args.limit * dspec["fraction"])))
        else:
            n = 100
        mode = dspec.get("mode", "chat")
        ptoks = dspec.get("prompt_tokens", 128)
        prompts = stream_prompts(dspec["hf"], n, strip_gutenberg_flag=dspec.get("strip_gutenberg", False))
        print(f"[{dom}] {len(prompts)} prompts (mode={mode})", flush=True)
        for i, p in enumerate(prompts):
            recs.append((f"{dom}:{i}", dom, mode,
                         build_prompt_ids(tok, p, mode, ptoks, gcfg.get("enable_thinking", False))))

    llm = LLM(model=model_path, tensor_parallel_size=args.tp, dtype="bfloat16",
              trust_remote_code=True, gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len, enforce_eager=args.enforce_eager,
              max_num_seqs=args.max_num_seqs,
              limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=gcfg.get("temperature", 0.7), top_p=gcfg.get("top_p", 0.8),
                        top_k=gcfg.get("top_k", 20), max_tokens=gcfg.get("max_new_tokens", 256))

    # resume: skip uids already in the output; chunked append+flush so preemption loses <=1 chunk
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                pass
    todo = [r for r in recs if r[0] not in done]
    print(f"{len(recs)} total, {len(done)} already done, {len(todo)} to generate", flush=True)

    t0 = time.time()
    ntok = 0
    with open(out_path, "a") as f:
        for s in range(0, len(todo), args.chunk):
            batch = todo[s:s + args.chunk]
            outs = llm.generate([{"prompt_token_ids": pid} for _, _, _, pid in batch], sp)
            for (uid, dom, mode, pid), o in zip(batch, outs):
                rid = list(o.outputs[0].token_ids)
                ntok += len(rid)
                f.write(json.dumps({"uid": uid, "domain": dom, "mode": mode,
                                    "prompt_ids": pid, "response_ids": rid}) + "\n")
            f.flush()
            print(f"  chunk: {s + len(batch)}/{len(todo)} seqs, {ntok} tok, {time.time() - t0:.0f}s", flush=True)
    dt = time.time() - t0
    print(f"generated {len(todo)} seqs, {ntok} response tokens in {dt:.0f}s "
          f"({ntok / max(dt, 1):.0f} tok/s) -> {args.out}")


if __name__ == "__main__":
    main()
