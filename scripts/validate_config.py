"""Phase 0 exit artifact.

(1) Assert every §1 config value against the downloaded checkpoint.
(2) Run a text-only forward and confirm we can hook, on transformers 5.x:
      (a) full-attention layer hidden states  (KV-injection source)
      (b) per-layer 256-way router logits      (domain descriptor)
      (c) the native MTP-1 head                (baseline B0)

Exit 0 = all checks pass. Non-zero = a mismatch (printed).

Run (on Spark):
    ~/devel/vllm/venv/bin/python scripts/validate_config.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from target.loader import load_target, resolve_model_path  # noqa: E402
from target.hooks import extract  # noqa: E402
from target.mtp import Qwen35MTPHead, load_mtp_state_dict, validate_mtp  # noqa: E402

# §1 expected values (text_config).
EXPECTED = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 2048,
    "vocab_size": 248320,
    "num_hidden_layers": 40,
    "full_attention_interval": 4,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "mtp_num_hidden_layers": 1,
    "tie_word_embeddings": False,
    "router_aux_loss_coef": 0.001,
}
EXPECTED_FULL_ATTN = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]

_fail = 0


def check(label: str, got, expected):
    global _fail
    ok = got == expected
    _fail += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {label:34s} got={got!r} expected={expected!r}")


def main() -> int:
    path = resolve_model_path()
    print(f"# checkpoint: {path}")
    cfg = json.load(open(os.path.join(os.path.expanduser(path), "config.json")))
    tc = cfg["text_config"]

    print("\n## §1 static config assertions")
    for k, v in EXPECTED.items():
        check(k, tc.get(k), v)
    # layer_types -> full-attn indices
    full_attn = [i for i, t in enumerate(tc["layer_types"]) if t == "full_attention"]
    check("full_attn_layer_indices", full_attn, EXPECTED_FULL_ATTN)
    check("num_experts_per_tok+shared", tc["num_experts_per_tok"], 8)

    print(f"\n# transformers: {__import__('transformers').__version__}  (spec used 4.57.1; version not asserted)")

    print("\n## loading target (text-only, vision dropped) ...")
    target = load_target()
    n_params = sum(p.numel() for p in target.model.parameters())
    print(f"  loaded; language-model params: {n_params/1e9:.2f}B; device={next(target.model.parameters()).device}")
    check("full_attention_layers()", target.full_attention_layers(), EXPECTED_FULL_ATTN)

    print("\n## (a)(b) live forward: hiddens + router logits")
    prompt = "The quick brown fox jumps over the lazy dog. Speculative decoding is"
    ids = target.tokenizer(prompt, return_tensors="pt").input_ids
    res = extract(target, ids)
    T = ids.shape[1]

    check("num hidden_states", len(res.hidden_states), tc["num_hidden_layers"] + 1)
    check("hidden dim", res.hidden_states[-1].shape[-1], tc["hidden_size"])
    check("num router_logits layers", len(res.router_logits), tc["num_hidden_layers"])
    r39 = res.router_at(39)
    check("router[39] expert dim", r39.shape[-1], tc["num_experts"])
    # descriptor at l*=39 for the last token
    d = torch.softmax(r39.float(), dim=-1)[-1]
    topk = torch.topk(d, tc["num_experts_per_tok"])
    print(f"  router[39] last-token top-8 experts: {topk.indices.tolist()}")
    print(f"  router[39] top-8 prob mass: {topk.values.sum().item():.3f}")
    fa = target.full_attention_layers()
    hctx = res.full_attn_hiddens(fa)
    check("full-attn hiddens count", len(hctx), len(EXPECTED_FULL_ATTN))
    check("full-attn hidden[39] shape", tuple(hctx[39].shape[-2:]), (T, tc["hidden_size"]))

    print("\n## (c) native MTP-1 head")
    sd = load_mtp_state_dict(path)
    for line in validate_mtp(sd, target.text_config):
        print(line)
    mtp = Qwen35MTPHead(target)
    missing, unexpected = mtp.load_weights(sd)
    print(f"  MTP weights loaded (layer missing={len(missing)}, unexpected={len(unexpected)})")
    last_h = res.hidden_states[-1]
    mtp_logits = mtp(last_h, ids)
    check("MTP logits shape", tuple(mtp_logits.shape), (1, T, tc["vocab_size"]))
    check("MTP logits finite", bool(torch.isfinite(mtp_logits).all().item()), True)
    nxt = mtp_logits[0, -1].argmax().item()
    print(f"  MTP predicts next token id={nxt} -> {target.tokenizer.decode([nxt])!r}")

    print(f"\n{'='*60}")
    if _fail:
        print(f"RESULT: {_fail} CHECK(S) FAILED")
        return 1
    print("RESULT: ALL PHASE-0 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
