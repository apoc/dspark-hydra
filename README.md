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

> Operational details (Spark access, sync flow, Python env, model paths, fast-loading,
> run commands) live in `AGENTS.md` (gitignored). Code is authored locally and synced to
> the DGX Spark for all compute.
## Status

- [x] **Phase 0 — Env & validation.** `scripts/validate_config.py` asserts every §1
      config value and, via one text-only forward, confirms live access to
      (a) full-attn layer hiddens, (b) per-layer 256-way router logits, (c) the native
      MTP-1 head (loaded from checkpoint; HF drops `mtp.*` on load, so we load it
      directly). All checks pass on transformers 5.9.0.
- [x] **Phase 1 — Instrumentation dump.** `scripts/dump_calibration.py` generates target
      responses (chat / prose-completion) and teacher-forces one forward to persist per-token
      hiddens + router logits + next token to sharded safetensors (`target/dump.py`).
      `scripts/verify_dump.py` proves p^t-reconstruction alignment (hidden_final[i]→next_token[i]:
      mean p 0.87, top-1 0.94, vs 111× lower shifted control).
- [x] **Phase 2 — Collapse map `C`.** `collapse/` builds the 256→K map: co-activation (PMI +
      pure-torch spectral clustering, default), weight-similarity (+centroid warm-init), learned.
      `scripts/build_C.py` emits C + balance stats + domain-overlap report. Pure torch (no sklearn/scipy).
- [x] **Phase 3 — Draft model.** `draft/` = KV-injected backbone + domain-routed MoE
      (hard/soft/scratch + dense control) + Markov semi-AR head + confidence head.
      `scripts/test_draft.py` passes fwd/bwd for all §6 rows; active params matched
      (dense 2.03M ≈ MoE active 2.33M) at 3× MoE total capacity.
- [~] Phase 4 — Train (losses + windowed dataloader + loop; training runs)
- [ ] Phase 5 — Offline τ eval
- [ ] Phase 6 — Correctness (lossless) gate
- [ ] Phase 7 — Serving (optional)
- [ ] Phase 8 — Report (RQ1–RQ6)

## Phase status → run commands

Per-phase run commands are in `AGENTS.md`. Scripts: `validate_config.py` (0),
`dump_calibration.py` / `verify_dump.py` (1), `build_C.py` (2), `test_draft.py` (3),
`train_draft.py` (4), `test_losslessness.py` (6).
