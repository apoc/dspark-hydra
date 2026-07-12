"""Phase C stage 2: HF teacher-force extraction -- preempt-safe & shardable.

Reads gen_responses JSONL (a file, or a DIR of shard_*.jsonl), teacher-forces one HF
forward per sequence, and writes draft-training shards (same schema as dump_calibration).
Sharded across Slurm array tasks via --shard/--nshards (strided over the concatenated
input); each task writes its own --out part dir. seq_id is derived from the global uid
(DOMAIN_ID*100000 + per-domain index) so it is UNIQUE across parts -- required by
train.data.WindowDataset, which groups tokens by seq_id. Merge with scripts/merge_dump.py.

Preempt: a part whose manifest.json already exists is skipped on requeue (part-level resume;
keep parts small via a larger --nshards so a preemption redoes <=1 part).

Run (array, one task per part):
    sbatch --array=0-15 --export=ALL,NSHARDS=16 scripts/hotdog_extract.sbatch
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from target.dump import DOMAIN_ID, ShardWriter, extract_records  # noqa: E402
from target.loader import load_target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="JSONL file or dir of shard_*.jsonl")
    ap.add_argument("--corpus", default=str(_REPO_ROOT / "configs" / "corpus.yaml"))
    ap.add_argument("--out", required=True, help="output part dir")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "manifest.json").exists():
        print(f"{args.out}/manifest.json exists -> part already done, skipping", flush=True)
        return

    dcfg = yaml.safe_load(open(args.corpus))["dump"]
    inj, l_star = dcfg["inject_layers"], dcfg["l_star"]

    p = Path(args.responses)
    files = sorted(glob.glob(str(p / "*.jsonl"))) if p.is_dir() else [str(p)]
    recs_in = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                if r.get("response_ids"):
                    recs_in.append(r)
    recs_in = recs_in[args.shard::args.nshards]
    print(f"shard {args.shard}/{args.nshards}: {len(recs_in)} seqs from {len(files)} file(s)", flush=True)

    print("loading target (HF) ...", flush=True)
    t0 = time.time()
    target = load_target()
    device = next(target.model.parameters()).device
    print(f"target loaded in {time.time() - t0:.0f}s", flush=True)

    writer = ShardWriter(args.out, tokens_per_shard=dcfg["tokens_per_shard"])
    td = time.time()
    for n, r in enumerate(recs_in, 1):
        full = torch.tensor(r["prompt_ids"] + r["response_ids"], dtype=torch.long).unsqueeze(0).to(device)
        rec = extract_records(target, full, len(r["prompt_ids"]), inj, l_star)
        if rec:
            dom = r["domain"]
            seq_id = DOMAIN_ID[dom] * 100000 + int(r["uid"].split(":")[1])
            writer.add(rec, dom, seq_id=seq_id)
        if n % 200 == 0:
            tot = writer._n + sum(m["n"] for m in writer._manifest)
            print(f"  {n}/{len(recs_in)} seqs  tokens={tot}  ({time.time() - td:.0f}s)", flush=True)

    meta = writer.close()
    print(f"DUMP DONE: {meta['sequences']} seqs, {meta['total_tokens']} tokens, "
          f"{len(meta['shards'])} shards -> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
