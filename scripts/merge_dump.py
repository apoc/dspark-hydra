"""Merge Phase C extraction part dirs (<dump>/part_*/) into ONE dump: a unified
manifest.json referencing every shard by a dump-relative path (part_K/shard_NNNNN.safetensors),
so target.dump.load_shards / train.data.WindowDataset read it as a single dump directory.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="parent dir containing part_*/")
    args = ap.parse_args()

    d = Path(args.dump)
    part_manifests = sorted(glob.glob(str(d / "part_*" / "manifest.json")))
    if not part_manifests:
        raise SystemExit(f"no part_*/manifest.json under {d}")

    shards, seqs, toks = [], 0, 0
    for pm in part_manifests:
        part = Path(pm).parent.name
        m = json.load(open(pm))
        seqs += m.get("sequences", 0)
        for sh in m["shards"]:
            shards.append({"shard": len(shards), "n": sh["n"], "file": f"{part}/{sh['file']}"})
            toks += sh["n"]

    meta = {"shards": shards, "sequences": seqs, "total_tokens": toks, "keys": None,
            "parts": len(part_manifests)}
    json.dump(meta, open(d / "manifest.json", "w"), indent=2)
    print(f"merged {len(part_manifests)} parts -> {len(shards)} shards, {seqs} seqs, "
          f"{toks} tokens -> {d}/manifest.json")


if __name__ == "__main__":
    main()
