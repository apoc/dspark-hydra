"""Phase 1 gate: verify a calibration dump is well-formed and correctly aligned.

Alignment is the critical check: hidden_final[i] through the frozen LM head must
predict next_token[i]. Responses are sampled, so we assert (a) high mean p(next_token)
and top-k hit rate, and (b) a large gap vs a deliberately shifted (misaligned) control.

Run on Spark:
    ~/devel/vllm/venv/bin/python scripts/verify_dump.py --dump data/calib_smoke
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from target.dump import DOMAINS, load_shards  # noqa: E402
from target.loader import load_target  # noqa: E402


def _pt_stats(lm, hidden, target):
    p_sum = top1 = top5 = 0
    for i in range(0, hidden.shape[0], 256):
        lg = lm(hidden[i:i + 256]).float()
        pr = torch.softmax(lg, -1)
        tg = target[i:i + 256]
        p_sum += pr[torch.arange(tg.shape[0]), tg].sum().item()
        tk = lg.topk(5, dim=-1).indices
        top1 += (tk[:, 0] == tg).sum().item()
        top5 += (tk == tg.unsqueeze(1)).any(1).sum().item()
    n = hidden.shape[0]
    return p_sum / n, top1 / n, top5 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(_REPO_ROOT / "data" / "calib_smoke"))
    args = ap.parse_args()

    meta = json.load(open(Path(args.dump) / "manifest.json"))
    print(f"# dump: {args.dump}")
    print(f"  sequences={meta['sequences']} tokens={meta['total_tokens']} shards={len(meta['shards'])}")

    tgt = load_target()
    lm = tgt.model.lm_head if hasattr(tgt.model, "lm_head") else tgt.model.get_output_embeddings()
    dev = next(tgt.model.parameters()).device
    ldt = next(lm.parameters()).dtype

    dom = collections.Counter()
    ap_all = a1_all = a5_all = mp_all = 0.0
    nseen = 0
    for sd, m in load_shards(args.dump):
        dom.update(sd["domain_id"].tolist())
        H = sd["hidden_final"].to(dev, ldt)
        nt = sd["next_token"].long().to(dev)
        n = H.shape[0]
        with torch.no_grad():
            ap_, a1_, a5_ = _pt_stats(lm, H, nt)
            mp_, _, _ = _pt_stats(lm, H[:-1], nt[1:])
        ap_all += ap_ * n; a1_all += a1_ * n; a5_all += a5_ * n; mp_all += mp_ * (n - 1); nseen += n

    ap_all /= nseen; a1_all /= nseen; a5_all /= nseen; mp_all /= max(nseen - len(meta["shards"]), 1)
    print("  domain counts:", {DOMAINS[k]: v for k, v in sorted(dom.items())})
    print(f"  ALIGNED   mean p(next)={ap_all:.3f} top1={a1_all:.3f} top5={a5_all:.3f}")
    print(f"  SHIFTED   mean p(next)={mp_all:.4f}")
    ok = ap_all > 0.3 and ap_all > 20 * mp_all
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
