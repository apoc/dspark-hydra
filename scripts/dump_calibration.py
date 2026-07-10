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

from target.dump import ShardWriter, extract_records, generate_response  # noqa: E402
from target.loader import load_target  # noqa: E402


def _strip_gutenberg(text: str) -> str | None:
    """Return the book body between the START/END markers, else None."""
    up = text
    s = up.find("*** START OF THE PROJECT GUTENBERG")
    if s != -1:
        s = up.find("\n", s)
        e = up.find("*** END OF THE PROJECT GUTENBERG")
        body = up[s:e if e != -1 else None].strip()
    else:
        body = text.strip()
    # skip any residual front matter; require enough body to form a context + continuation
    if len(body) < 2000:
        return None
    return body[500:]  # nudge past chapter headers / TOC


def stream_prompts(hf: dict, n: int, strip_gutenberg: bool = False):
    from datasets import load_dataset

    kw = {"streaming": True, "split": hf["split"]}
    if hf.get("name"):
        ds = load_dataset(hf["path"], hf["name"], **kw)
    else:
        ds = load_dataset(hf["path"], **kw)
    field = hf["field"]
    out = []
    for ex in ds:
        txt = ex.get(field)
        if not txt:
            continue
        txt = str(txt)
        if strip_gutenberg:
            txt = _strip_gutenberg(txt)
            if txt is None:
                continue
        if len(txt.strip()) > 0:
            out.append(txt)
        if len(out) >= n:
            break
    return out


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
        prompts = stream_prompts(dspec["hf"], n, strip_gutenberg=dspec.get("strip_gutenberg", False))
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
