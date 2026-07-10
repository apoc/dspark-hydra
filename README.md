# DSpark-Hydra — Domain-Routed Speculative Drafting via Target-Router Reuse

Research code for testing whether a MoE target model's **own expert router** can be
reused, for free, to drive a **domain-routed MoE draft model** for speculative
decoding — raising per-domain accepted length (τ) on high-entropy domains (chat,
prose) at equal active draft parameters, without touching the lossless guarantee.

Target model (frozen): **Qwen/Qwen3.6-35B-A3B** (`qwen3_5_moe`, 35B total / ~3B active,
256 experts × top-8, hidden 2048, 40 MoE layers, hybrid linear+full attention).

> Full design: `doc/dspark-qwen36-moe-router-reuse-experiment.md` (local only, gitignored).

## The idea in one paragraph

Speculative decoding drafts γ tokens with a small model; the target verifies them by
rejection sampling (lossless for **any** draft distribution). A single small draft
trained across math/code/chat/prose suffers **capacity dilution**. But the target MoE
**already contains a trained domain partitioner** — its per-token 256-way expert
router — and those logits are **already computed** during verification (zero marginal
cost). We collapse the 256 target experts → K draft experts (map `C`) and route a
domain-specialized draft MoE by the target's own decision. Routing changes only the
draft distribution `p^d`; it can never alter the target `p^t` or the acceptance rule,
so every variant is output-equivalent to the target alone (verified in Phase 6).

```mermaid
graph TD
  T[Qwen3.6-35B-A3B frozen forward] -->|full-attn hiddens 3,7..39| H[KV-inject H_ctx]
  T -->|256-way router logits at l*=39| D[domain descriptor d]
  D --> C[collapse map C: 256 to K]
  H --> DR[draft backbone, hidden 2048]
  C --> MOE[draft MoE: K experts, route by C·softmax d]
  DR --> MOE
  MOE --> MH[Markov semi-AR head r=256]
  MH --> OUT[gamma draft tokens + confidences] --> T
```

## Run matrix

| # | Name | Draft | Router | Purpose |
|---|---|---|---|---|
| B0 | Native MTP-1 | ships w/ model | — | production floor (via manual MTP head / vLLM) |
| B3 | DSpark-dense | semi-AR, single FFN | — | paper reproduction, primary control |
| **E1** | **DSpark-MoE-hard** | semi-AR + K-expert MoE | frozen `C` | **main experiment** |
| E2 | DSpark-MoE-soft | semi-AR + K-expert MoE | distilled `R_d` | hard vs soft reuse |
| C1 | DSpark-MoE-scratch | semi-AR + K-expert MoE | from-scratch | isolate reuse value |

All at **equal active FLOPs**. Win: E1 macro-τ ≥ B3 with biggest gains on chat+prose,
≤ +1pt latency, and E1 ≥ C1 (reused router beats from-scratch).

## Repo layout

```
configs/     model + train + variant YAMLs
target/      loader (text-only), hidden/router hooks, native MTP head
collapse/    256→K C-map builders (co-activation / weight / learned) + balance
draft/       backbone, kv_inject, moe_reused_router, markov_head, conf_head
train/       losses (ce/tv/conf/route/bal), STS calibration, loop
eval/        accepted-length τ, position-wise, specialization, losslessness
serving/     (optional) vLLM/SGLang integration + scheduler
scripts/     validate_config, dump_calibration, build_C, run_matrix
reports/     tables, figures, RQ writeups
```

## Infrastructure & workflow

- **Code is authored locally** (`~/devel/dspark-hydra`, macOS). It is the source of truth.
- **All compute runs on DGX Spark** (SSH `localhost:5555`, user `apoc`; GB10, 128GB unified, 1 GPU).
- **Sync:** local `git push` → `git pull` on Spark at `~/devel/dspark-hydra` (remote
  `github.com/apoc/dspark-hydra`, cloned via HTTPS). Never edit on Spark directly.
- **Python env on Spark:** `~/devel/vllm/venv` (transformers 5.9.0, torch 2.11.0+cu130,
  safetensors 0.7.0, CUDA). Do not install packages; the env is pre-provisioned.
- **Model on Spark:** `~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96.../`
  (BF16, 26 shards, 67 GB). FP8 variant also cached. Paths live in `configs/model.yaml`.
- **Fast loading (GB10):** unpinned H2D on GB10 is ~0.16 GB/s (a 67GB CUDA load takes
  ~450s). `target.loader.move_to_cuda_pinned` loads on CPU then streams to GPU through
  pinned memory (~18 GB/s) — full load ~7s warm, ~20s cold. Warm the page cache first:
  `ls <blobs>/* | xargs -P8 -I{} cat {} >/dev/null`.
- `doc/` and `banks/` are gitignored.

## Status

- [x] **Phase 0 — Env & validation.** `scripts/validate_config.py` asserts every §1
      config value and, via one text-only forward, confirms live access to
      (a) full-attn layer hiddens, (b) per-layer 256-way router logits, (c) the native
      MTP-1 head (loaded from checkpoint; HF drops `mtp.*` on load, so we load it
      directly). All checks pass on transformers 5.9.0.
- [ ] Phase 1 — Instrumentation dump (hiddens + router logits + target dists)
- [ ] Phase 2 — Collapse map `C`
- [ ] Phase 3 — Draft model (all variants)
- [ ] Phase 4 — Train
- [ ] Phase 5 — Offline τ eval
- [ ] Phase 6 — Correctness (lossless) gate
- [ ] Phase 7 — Serving (optional)
- [ ] Phase 8 — Report (RQ1–RQ6)

## Running Phase 0

```bash
# on Spark, after git pull
~/devel/vllm/venv/bin/python scripts/validate_config.py
```
