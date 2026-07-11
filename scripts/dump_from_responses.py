"""Phase C stage 2: HF teacher-force extraction from pre-generated responses.

Reads the JSONL from `gen_responses.py` (prompt/response token ids) and teacher-forces
one HF forward per sequence to persist the draft-training records (hidden_final,
hidden_inject, router_star, router_agg, next_token) -- identical schema to
`dump_calibration.py`, but with generation already done (stage 1). Reuses
`target/dump.extract_records` + `ShardWriter`.

Requires the target loaded via HF (router logits are not exposed by vLLM), so run on a
box that fits the model (one 80 GB A100, or device_map across 2). Token ids are used
verbatim -> byte-exact alignment with stage 1.

Run (on the GPU box):
    python scripts/dump_from_responses.py --responses data/gen_v2.jsonl --out data/calib_v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from target.dump import ShardWriter, extract_records  # noqa: E402
from target.loader import load_target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="JSONL from gen_responses.py")
    ap.add_argument("--corpus", default=str(_REPO_ROOT / "configs" / "corpus.yaml"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dcfg = yaml.safe_load(open(args.corpus))["dump"]
    inj, l_star = dcfg["inject_layers"], dcfg["l_star"]

    print("loading target (HF) ...", flush=True)
    t0 = time.time()
    target = load_target()
    device = next(target.model.parameters()).device
    print(f"target loaded in {time.time() - t0:.0f}s", flush=True)

    writer = ShardWriter(args.out, tokens_per_shard=dcfg["tokens_per_shard"])
    n = 0
    td = time.time()
    with open(args.responses) as f:
        for line in f:
            r = json.loads(line)
            if not r["response_ids"]:
                continue
            full = torch.tensor(r["prompt_ids"] + r["response_ids"], dtype=torch.long).unsqueeze(0).to(device)
            rstart = len(r["prompt_ids"])
            rec = extract_records(target, full, rstart, inj, l_star)
            if rec:
                writer.add(rec, r["domain"])
            n += 1
            if n % 100 == 0:
                tot = writer._n + sum(m["n"] for m in writer._manifest)
                print(f"  {n} seqs  tokens={tot}  ({time.time() - td:.0f}s)", flush=True)

    meta = writer.close()
    print(f"DUMP DONE: {meta['sequences']} seqs, {meta['total_tokens']} tokens, "
          f"{len(meta['shards'])} shards -> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
