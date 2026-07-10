"""Phase 1: produce the calibration dump (§7.4).

For each domain in configs/corpus.yaml: stream prompts, generate a target response
(non-thinking / completion), teacher-force one forward, and persist per-token records
(hiddens + router logits + next token) to sharded safetensors.

Run on Spark:
    ~/devel/vllm/venv/bin/python scripts/dump_calibration.py --limit 20 --out data/calib_smoke
    ~/devel/vllm/venv/bin/python scripts/dump_calibration.py --per-domain 5000 --out data/calib
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from target.corpus import stream_prompts  # noqa: E402
from target.dump import ShardWriter, extract_records, generate_response  # noqa: E402
from target.loader import load_target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(_REPO_ROOT / "configs" / "corpus.yaml"))
    ap.add_argument("--out", default=str(_REPO_ROOT / "data" / "calib_smoke"))
    ap.add_argument("--per-domain", type=int, default=None, help="prompts per domain")
    ap.add_argument("--limit", type=int, default=None, help="total prompts (split by fraction)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.corpus))
    gen_cfg = cfg["generation"]
    dcfg = cfg["dump"]
    inject_layers = dcfg["inject_layers"]
    l_star = dcfg["l_star"]

    print("loading target ...", flush=True)
    t0 = time.time()
    target = load_target()
    print(f"target loaded in {time.time()-t0:.1f}s", flush=True)

    writer = ShardWriter(args.out, tokens_per_shard=dcfg["tokens_per_shard"])

    for dom, dspec in cfg["domains"].items():
        if args.per_domain is not None:
            n = args.per_domain
        elif args.limit is not None:
            n = max(1, int(round(args.limit * dspec["fraction"])))
        else:
            n = 5
        mode = dspec.get("mode", "chat")
        ptoks = dspec.get("prompt_tokens", 128)
        print(f"\n[{dom}] streaming {n} prompts ({dspec['hf']['path']}, mode={mode})", flush=True)
        prompts = stream_prompts(dspec["hf"], n, strip_gutenberg_flag=dspec.get("strip_gutenberg", False))
        print(f"[{dom}] got {len(prompts)} prompts; generating+dumping ...", flush=True)

        td = time.time()
        for j, ptxt in enumerate(prompts):
            full_ids, rstart = generate_response(target, ptxt, gen_cfg, mode=mode, prompt_tokens=ptoks)
            rec = extract_records(target, full_ids, rstart, inject_layers, l_star)
            writer.add(rec, dom)
            if (j + 1) % 10 == 0:
                print(f"  [{dom}] {j+1}/{len(prompts)}  tokens={writer._n + sum(m['n'] for m in writer._manifest)}  ({time.time()-td:.0f}s)", flush=True)

    meta = writer.close()
    print(f"\nDUMP DONE: {meta['sequences']} seqs, {meta['total_tokens']} tokens, "
          f"{len(meta['shards'])} shards -> {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
