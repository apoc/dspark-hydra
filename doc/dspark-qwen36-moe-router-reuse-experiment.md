# Experiment Spec — Domain-Routed Speculative Drafting via Target-Router Reuse (DSpark-MoE on Qwen3.6-35B-A3B)

> **Status:** Ready-to-start research design. Self-contained. Another engineer/agent can begin at Phase 0 without prior context.
> **Owner:** (assign)
> **Target model (frozen):** `Qwen/Qwen3.6-35B-A3B` (`qwen3_5_moe`). Runtime env on DGX Spark (GB10, 128GB unified): **transformers 5.9.0, torch 2.11.0+cu130, safetensors 0.7.0** (config.json stamps 4.57.1, but the checkpoint loads & runs on 5.9.0 — validated in Phase 0; package versions are not pinned to the spec). HF impl class `Qwen3_5MoeForConditionalGeneration`; load via `AutoModelForImageTextToText`, drive text-only.
> **Prereq reading:** DSpark paper (arXiv 2607.05147, DeepSeek-AI). This doc assumes its terminology (draft/target, accepted length τ, semi-autoregressive head, confidence head, prefix scheduler).

---

## 0. TL;DR

DSpark builds a small **semi-autoregressive** draft model that predicts N tokens per target forward pass, trained to match the target's output distribution so speculative rejection sampling stays **lossless**. A single small draft model trained across many domains suffers **capacity dilution** — its limited parameters must approximate the target across math/code/chat/prose simultaneously, and per-domain acceptance length (τ) varies a lot (≈6 math → ≈3.5 chat in the paper; prose expected worse).

**Hypothesis:** The target MoE model **already contains a trained domain partitioner** — its per-token expert router. We can **reuse that router signal** (for free, it is already computed during verification) to drive a **domain-routed MoE draft model**, so each draft expert specializes to a semantic partition instead of averaging over all domains. This should raise τ on divergent/out-of-distribution domains (chat, literary prose) at **equal active draft parameters**, without harming the lossless guarantee.

**Deliverable of the experiment:** measured per-domain τ and throughput for (native MTP-1) vs (dense DSpark) vs (**DSpark-MoE with reused router**), plus an ablation of the 256→K expert-collapse method and router source layer.

---

## 1. Grounded target-model facts (validate before building)

All values fetched from `https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main/config.json` (`text_config` unless noted). **Phase 0 must re-assert these against the downloaded checkpoint.**

| Field | Value | Consequence for the draft design |
|---|---|---|
| `model_type` | `qwen3_5_moe` | HF impl class `Qwen3_5MoeForConditionalGeneration`; MoE + hybrid-attention |
| `hidden_size` | **2048** | Draft hidden / KV-injection projection target dim |
| `vocab_size` | **248320** (padded) | Markov head `W1∈R^{V×r}`, `W2∈R^{r×V}`; shared frozen LM head |
| `num_hidden_layers` | **40** | KV-injection source-layer selection |
| `layer_types` | `[linear,linear,linear,full] × 10` | Only **full-attention** layers hold global softmax context |
| `full_attention_interval` | 4 | Full-attn layer indices (0-based): **3,7,11,15,19,23,27,31,35,39** |
| `num_attention_heads` / `num_key_value_heads` | 16 / 2 | GQA on full-attn layers; head_dim 256 |
| linear (Gated DeltaNet) | 16 key / 32 value heads, head_dim 128, conv 4 | Recurrent state, **not** a standard KV cache — avoid as injection source |
| `num_experts` | **256** | Source cardinality for the 256→K collapse |
| `num_experts_per_tok` | **8** (routed) `+1` shared | Router emits top-8 of 256 per token per layer (~3.1% sparsity) |
| `moe_intermediate_size` | **512** | Per-expert FFN width; draft experts can mirror or shrink |
| `shared_expert_intermediate_size` | 512 | Always-on shared expert (domain-agnostic component) |
| `mtp_num_hidden_layers` | **1** | **Native MTP head already exists** → our MTP-1 baseline |
| `mtp_use_dedicated_embeddings` | false | MTP shares token embeddings |
| `tie_word_embeddings` | **false** | Embedding matrix ≠ LM head; draft borrows **both** (frozen) |
| `router_aux_loss_coef` | 0.001 | Reference for any draft load-balance aux loss |
| `partial_rotary_factor` / `rope_theta` | 0.25 / 1e7 | RoPE dim = 64 on full-attn; mrope sections [11,11,10] |
| vision encoder | present (`out_hidden_size` 2048) | **Out of scope** — text-only drafting; strip/ignore image tokens 248053–248057 |

**Two facts that make this model an ideal testbed:**
1. **It already ships an MTP-1 head** — the exact production baseline DSpark beat. No baseline to build; it comes for free.
2. **Every layer is MoE with a 256-way router** — abundant, per-layer domain signal to mine and reuse.

---

## 2. Background & problem statement

### 2.1 Why a small draft dilutes across domains
DSpark's draft learns to minimize TV distance to the target distribution (`L_tv`, weight 0.9). But the target's distribution is shaped completely differently per domain — near-deterministic in code/math, high-entropy in chat/prose. A fixed-capacity draft trained on a mixture must compromise. Empirically (DSpark Table 1, Qwen3-4B): τ ≈ 6.1 (GSM8K) vs 3.5 (Alpaca). Literary prose is expected below chat because both the **task entropy** and the **train/serve distribution shift** are worse.

The ceiling is information-theoretic: even a perfect draft has
`E[τ] ≤ Σ_k (1 − ½‖p_k^d − p_k^t‖₁)` — high target entropy caps τ regardless of draft size. But a *sub-capacity* draft loses **additional** τ to cross-domain interference. **That interference is what this experiment attacks.**

### 2.2 The free domain signal
An MoE router is a learned function `token-context → distribution over 256 experts`. Trained on trillions of tokens, that partition strongly correlates with domain/syntax/entropy regime. During speculative verification the target runs a full forward pass anyway, so its router logits are **already computed** — a zero-marginal-cost domain descriptor. Prior speculative work (DSpark included) discards it.

### 2.3 Core hypothesis (novel — not in the DSpark paper)
> Conditioning a domain-routed MoE draft model on the **target's own (collapsed) router decision** yields higher per-domain τ than a dense draft of equal active parameters, primarily by removing cross-domain capacity dilution — and does so without a separately learned domain classifier.

---

## 3. Research questions

- **RQ1 (main):** At equal *active* draft params, does DSpark-MoE (reused router) beat dense DSpark on macro-avg τ, and especially on the highest-entropy domains (chat, prose)?
- **RQ2 (the "free lunch"):** Does reusing the target router match/beat a draft router **learned from scratch** at equal cost? I.e., is the target's partition actually the right one for drafting?
- **RQ3 (interference):** Does routing reduce **per-domain τ variance** and shrink out-of-distribution (prose) degradation vs dense?
- **RQ4 (mechanism):** Hard (frozen collapse map) vs soft (distilled draft router) reuse — accuracy/latency/robustness tradeoff.
- **RQ5 (signal source):** Which target layer's router carries the most drafting-useful domain signal (early/mid/late full-attn-adjacent MoE)? Single layer vs aggregate.
- **RQ6 (correctness, confirmatory):** Verify output distribution is unchanged (lossless) for every variant — routing only alters `p^d`, never the rejection rule.

---

## 4. System architecture

```
                    ┌─────────────────────── TARGET (frozen) ───────────────────────┐
 prompt ─▶ Qwen3.6-35B-A3B forward (verification pass)                               │
                    │   • full-attn layer hidden states  H^{(3,7,…,39)}   ──────────┼──▶ KV-injection context H_ctx
                    │   • per-layer 256-way router logits R^{(ℓ)}         ──────────┼──▶ domain descriptor  d
                    │   • bonus/anchor token x0, target dist p^t (for training/verify)
                    └───────────────────────────────────────────────────────────────┘
                                     │ (all borrowed, zero extra target compute)
                                     ▼
        ┌──────────────────────── DRAFT (trainable, small) ───────────────────────┐
        │  Parallel backbone  (≈3–5 layers, hidden 2048)                            │
        │     • anchor emb + (γ−1) mask embs, KV-inject H_ctx into every draft layer │
        │     • MoE layers: K draft experts, routed by COLLAPSED target router  d    │  ◀── the experiment
        │  Semi-AR sequential head (Markov, low-rank r=256): base logit U_k + B_k    │
        │  Confidence head: c_k = σ(w·[h_k ; W1[x_{k-1}]])                           │
        │  Shared FROZEN embedding + LM head (from target)                           │
        └───────────────────────────────────────────────────────────────────────────┘
                                     │ γ draft tokens + confidences
                                     ▼
        Hardware-aware prefix scheduler (unchanged from DSpark) ─▶ target verifies ─▶ new anchor
```

### 4.1 Draft backbone
- 3–5 transformer layers, hidden 2048 (match target to keep injection projections trivial; a 1024 variant with an up/down projection is an ablation).
- **KV injection (DFlash-style):** concatenate hidden states from a chosen set of **full-attention** target layers, project once: `H_ctx = RMSNorm(W_c [H^{(l1)};…;H^{(lm)}])`, inject into every draft layer's K/V along the sequence dim. Default source set: `{19, 31, 39}` (mid, late, final full-attn) + final pre-LM-head hidden — tuned in RQ5.
  - **Rationale (hybrid-model specific):** linear-attention layers keep a recurrent SSM state, not a per-position KV cache; their hidden states are a poorer "global context" handle. Full-attention layers are the correct injection source.

### 4.2 Domain-routed MoE draft layers — the core novelty
Each draft MoE layer holds **K draft experts** (default **K=16**), top-`k'` active (default **k'=2**), FFN width `moe_intermediate_size` = 512 (mirror target) or 1024 (capacity ablation), plus one always-on shared expert (mirrors target's shared expert = domain-agnostic base).

Routing is **derived from the target**, not recomputed:

- **Variant A — Hard reuse (frozen collapse map).**
  Precompute `C ∈ {0,1}^{K×256}` (or row-stochastic `C ∈ R_{≥0}^{K×256}`) mapping 256 target experts → K draft groups (Section 5). At inference, take the target router logits `d = R^{(ℓ*)} ∈ R^{256}` at source layer `ℓ*`, compute draft group scores `g = C · softmax(d) ∈ R^K`, select top-`k'` draft experts. `C` is frozen; **only draft expert weights train.**

- **Variant B — Soft reuse (distilled draft router).**
  Draft has a tiny router `R_d: R^{2048} → R^K` (one linear). Trained with a distillation term `L_route = KL(softmax(g_target-collapsed) ‖ softmax(R_d(h)))` so it learns the target's collapsed partition but can adapt. Adds negligible params/latency; tests whether adaptation beats frozen reuse (RQ4).

- **Control — From-scratch router (for RQ2).**
  Identical draft MoE but `R_d` trained only on the drafting losses (no target-router signal). Isolates the value of the reused signal.

- **Control — Dense (for RQ1/RQ3).**
  Replace draft MoE with a single FFN of matched **active** FLOPs (`k'·moe_intermediate_size`). Reproduces standard DSpark.

**Load balancing:** reused routing can starve some draft experts. Add a draft-side aux load-balance loss (coef ≈ `router_aux_loss_coef` = 0.001) in Variants B/control; for Variant A, verify balance empirically and, if skewed, rebalance `C` (Section 5.4).

### 4.3 Semi-autoregressive head
Markov head exactly as DSpark default: transition bias `B(x_{k-1},·) = W1[x_{k-1}] · W2`, low-rank `r=256`. Keeps `p_k^d = softmax(U_k + B_k)` an **exact per-token categorical** → rejection sampling stays lossless. (RNN head is an optional ablation; DSpark found marginal gains, higher complexity.)

### 4.4 Confidence head + scheduler
Unchanged from DSpark: `c_k = σ(w·[h_k; W1[x_{k-1}]])`, supervised by `c_k* = 1 − ½‖p_k^d − p_k^t‖₁`, post-hoc Sequential Temperature Scaling (STS) to ECE ≈ 1%. Hardware-aware prefix scheduler (Algorithm 1) and its 2-step-async production form are reused as-is; they are **orthogonal** to the routing change. Keep them **off** for offline τ measurement (isolate draft quality), **on** for any serving test.

### 4.5 What stays frozen / trainable
| Frozen | Trainable |
|---|---|
| Entire target model (attn, MoE, routers) | Draft backbone layers |
| Token embedding, LM head (borrowed) | Draft MoE experts (+ shared) |
| Collapse map `C` (Variant A) | Draft router `R_d` (Variant B / control) |
| | Markov head `W1,W2`; confidence head `w` |

---

## 5. The 256 → K expert-collapse map `C`

Build offline, once, on a calibration corpus (Section 7.1). Three candidate methods; **co-activation clustering is the default**, weight clustering gives free warm-init, learned map is the flexible upper-bound.

### 5.1 Method 1 — co-activation clustering (default)
1. Run target over calibration corpus; at chosen source layer `ℓ*`, record the top-8 expert set per token.
2. Accumulate a `256×256` co-activation matrix `M[i,j] = #(experts i,j fire together)`.
3. Normalize (PMI or row-normalize), cluster to K groups (spectral or agglomerative on `1−M̂`).
4. `C[g,i] = 1` iff expert i ∈ group g (optionally soft = normalized co-activation membership).
> Rationale: experts firing on the same tokens serve the same content → the same draft expert.

### 5.2 Method 2 — weight-similarity clustering (warm-init bonus)
Cluster the 256 expert FFN weight vectors (concat of gate/up/down, or their SVD) into K; **initialize each draft expert from its cluster centroid.** Gives the draft a domain-partitioned starting point so training refines TV distance within a partition rather than discovering structure. Can be combined with Method 1's routing.

### 5.3 Method 3 — learned collapse (upper bound)
`C` is a trainable `K×256` row-stochastic matrix (softmax over 256), trained end-to-end with drafting losses. Most flexible, least interpretable; use to bound achievable τ.

### 5.4 Rebalancing
If any draft expert receives ≪ or ≫ its share of tokens on a held-out set, re-solve clustering with a balance constraint (e.g., balanced k-means / capacity-constrained assignment) so groups carry comparable token mass.

### 5.5 Router source layer `ℓ*` (RQ5)
Candidates from full-attn-adjacent MoE: `{19, 27, 39}` and "aggregate = mean of softmaxed router logits over all full-attn layers." Default `ℓ*=39` (closest to output, most task-semantic); aggregate is the robustness variant. Optionally EMA the descriptor over recent tokens to denoise the single-token signal.

---

## 6. Baselines & variants (the run matrix)

| # | Name | Draft structure | Router | Purpose |
|---|---|---|---|---|
| B0 | **Native MTP-1** | ships with model | — | Production floor (like DSpark's MTP-1) |
| B1 | Eagle3 (AR) | 1-layer autoregressive | — | AR reference (retrain in DeepSpec) |
| B2 | DFlash (parallel) | 5-layer parallel, no seq head | — | Parallel reference |
| B3 | **DSpark-dense** | semi-AR, single FFN | — | Paper reproduction; primary control |
| E1 | **DSpark-MoE-hard** | semi-AR + K-expert MoE | Variant A (frozen `C`) | **Main experiment** |
| E2 | **DSpark-MoE-soft** | semi-AR + K-expert MoE | Variant B (distilled) | RQ4 |
| C1 | DSpark-MoE-scratch | semi-AR + K-expert MoE | from-scratch `R_d` | RQ2 (isolate reuse value) |

All of B1–E2 share: identical training data, KV-injection source set, γ, backbone depth, Markov head, confidence head, **equal active FLOPs** (dense B3 FFN width = `k'·512`). Report parameter counts (total & active) for every row.

---

## 7. Data & training

### 7.1 Corpora
- **Draft-training + calibration:** instruction mixture à la DSpark (Open-PerfectBlend style: ~39% math, ~39% code, ~18% chat, ~4% IF). **Add an explicit `prose/literary` slice** (e.g., public-domain long-form fiction incl. *War and Peace* translations, essays) to directly probe the high-entropy hypothesis (RQ3). Target ~1–1.5M prompts.
- **Responses regenerated by the target** (Qwen3.6-35B-A3B) in non-thinking mode with recommended sampling — draft learns *this* model's distribution.
- **Held-out calibration split** for STS and for building `C`.

### 7.2 Objective
`L = 0.1·L_ce + 0.9·L_tv + 1.0·L_conf (+ λ_route·L_route for E2; + λ_bal·L_bal for E2/C1)`
- `L_ce = −Σ w_k log p_k^d(x_k*)`, `L_tv = Σ w_k ‖p_k^d − p_k^t‖₁`, `L_conf` = BCE to `c_k*`.
- Position weights `w_k = exp(−(k−1)/γ)`.
- `λ_route`, `λ_bal` small (start 0.01, 0.001); tune.
- Draft block size **γ = 5** to mirror DSpark's production config and the model's MTP horizon; sweep `{4,8,12,16}` in ablations.
- ~10 epochs to convergence.

### 7.3 Data-generation efficiency (from DSpark §5.1 — reuse)
- **Communicate hidden states (dim 2048), not logits (248320):** cache target pre-LM-head hidden; apply the shared LM head locally only at sampled positions → O(d) not O(V) inter-worker traffic. **Also cache the router logits at `ℓ*`** in the same dump (tiny: 256 floats/token).
- **Anchor-bounded sequence packing:** sample fixed # of anchors/sequence, pack blocks via token-level attention indices (not 2D masks) to decouple draft cost from the target's long context.

### 7.4 Instrumentation dump (Phase 1 artifact)
Per calibration token, persist: pre-LM-head hidden (2048, bf16), full-attn layer hiddens for the injection set, router logits at `ℓ*` (and aggregate), target next-token distribution (top-p truncated or full for a subset), sampled token. This single dump feeds both `C`-construction and training.

---

## 8. Evaluation protocol

### 8.1 Offline (scheduler OFF, fixed block) — primary
Domains & benches (mirror DSpark + prose):
- **Math:** GSM8K, MATH500, AIME25
- **Code:** MBPP, HumanEval, LiveCodeBench
- **Chat:** MT-Bench, Alpaca, Arena-Hard
- **Prose (new):** held-out literary/long-form continuation set

Metrics:
- **Accepted length τ per round** (primary), per bench + macro-avg, temperature 1.0, chain drafting.
- **Position-wise conditional acceptance** (DSpark Fig 2 method): acceptance at position k conditioned on 1..k−1 accepted — shows suffix-decay mitigation and whether routing helps deep positions.
- **Acceptance rate**, **draft latency overhead** (per-round engine time; must stay ≈ dense, target <~2%).

### 8.2 Interference analysis (RQ3)
- Per-domain τ **variance** across variants (lower = less dilution).
- **OOD prose degradation:** τ(prose)/τ(macro) ratio, DSpark-MoE vs dense.
- **Expert-specialization heatmap:** draft-expert activation frequency × domain; expect block-diagonal structure for E1/E2, uniform for dense/scratch.

### 8.3 Correctness (RQ6, gating)
- Assert accepted-token distribution == target's: fixed seed, compare speculative vs pure-AR target generation; KL / exact-match of sampled sequences over a fixed prompt set. Any variant that diverges is a **bug** (routing must not touch the rejection rule).

### 8.4 Online (optional, if serving infra available)
- vLLM/SGLang integration; **tok/s/user vs aggregate throughput** Pareto vs B0 (native MTP-1).
- **Verification budget vs concurrency** (scheduler ON) — confirm load-adaptive pruning still holds with MoE draft.

### 8.5 Success criteria
- **Primary:** E1 (or E2) macro-avg τ ≥ B3 (dense) by a clear margin, with the **largest gains on chat + prose**, at equal active params, ≤ dense draft latency +1 pt.
- **RQ2:** E1 ≥ C1 (reused router ≥ from-scratch) → the target partition is genuinely useful.
- **RQ3:** per-domain τ variance(E1) < variance(B3); prose ratio improved.
- **Correctness:** all variants pass §8.3 exactly.

---

## 9. Ablations

| Axis | Values |
|---|---|
| K (draft experts) | 8, **16**, 32, 64 |
| k' (active) | 1, **2**, 4 |
| Collapse method | **co-activation**, weight-cluster(+init), learned |
| Router source `ℓ*` | 19, 27, **39**, aggregate |
| Descriptor smoothing | none, EMA over recent tokens |
| KV-injection set | {39}, **{19,31,39}**, all full-attn |
| Warm-init from centroids | on/off |
| Block size γ | 4, **5**, 8, 12, 16 |
| Reuse mode | hard (A), soft (B) |
| Expert FFN width | **512**, 1024 |

---

## 10. Phased execution plan (for the implementing agent)

> Skip formatters/linters/full test suites until the end; each phase has a concrete exit artifact.

- **Phase 0 — Env & validation.** Download checkpoint; load `text_config`; **assert every value in §1**; run a text-only forward, confirm you can (a) read full-attn layer hiddens, (b) read the 256-way router logits per MoE layer, (c) invoke the native MTP head. Exit: a `validate_config.py` that prints/asserts the table and a working hidden+router hook.
- **Phase 1 — Instrumentation dump.** Implement hooks (§7.4); produce the calibration dataset (hiddens + router logits + target dists). Exit: sharded dump + a loader.
- **Phase 2 — Collapse map `C`.** Implement Methods 1–3 (§5); pick default via balance + a quick τ probe. Exit: serialized `C`, cluster report, balance stats, (optional) centroid init tensors.
- **Phase 3 — Draft model.** Implement backbone + KV injection + reused-router MoE (Variants A/B + dense/scratch controls) + Markov head + confidence head, in DeepSpec (`github.com/deepseek-ai/DeepSpec`) or an equivalent training repo. Exit: forward/backward passes for all rows in §6 on toy data.
- **Phase 4 — Train.** Train B1–E2 with identical data/hparams (§7). Exit: converged checkpoints + training curves; **B3 reproduces DSpark-style τ ranges** as a sanity gate.
- **Phase 5 — Offline eval.** §8.1–8.2 across all domains incl. prose; position-wise curves; specialization heatmaps. Exit: results tables + figures.
- **Phase 6 — Correctness.** §8.3 for every variant. Exit: lossless-equivalence report.
- **Phase 7 — (optional) Serving.** Integrate + load test vs native MTP-1. Exit: Pareto + budget-vs-load plots.
- **Phase 8 — Report.** Answer RQ1–RQ6 with evidence; recommend ship/no-ship of router reuse.

---

## 11. Suggested repo layout

```
dspark-qwen36-moe/
├── configs/            # model + train + variant YAMLs (K, k', ℓ*, γ, injection set)
├── target/             # loading, hidden/router hooks, native MTP wrapper
├── collapse/           # C-map builders (co-activation, weight, learned) + balance
├── draft/              # backbone, kv_inject, moe_reused_router, markov_head, conf_head
├── train/              # losses (ce/tv/conf/route/bal), STS calibration, loop
├── eval/               # accepted-length, position-wise, specialization, losslessness
├── serving/            # (optional) vLLM/SGLang integration + scheduler
├── scripts/            # validate_config, dump_calibration, build_C, run_matrix
└── reports/            # tables, figures, RQ writeups
```

---

## 12. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Hybrid attention** — linear layers lack per-position KV context | Weak injection → low τ | Inject only from **full-attn** layers (3,7,…,39) + final hidden |
| Single-token router signal noisy | Unstable routing → no gain | Use late layer `ℓ*=39` / aggregate / EMA smoothing (RQ5) |
| Reused router starves some draft experts | Wasted capacity | Balance-constrained clustering (§5.4); aux load-balance loss (E2/C1) |
| MoE adds params → draft latency creeps up | Erodes speedup | Keep **active** FLOPs == dense; measure per-round latency (§8.1); K↑ only if latency flat |
| Target router partitions for *its* loss, not drafting | Reuse underperforms scratch | That is exactly **RQ2**; C1 control decides it empirically |
| Vision/multimodal tokens leak in | Corrupt text drafting | Strip image/video tokens (248053–248057); text-only pipeline |
| Confidence scheduler non-anticipating property | Lossless violation | Keep DSpark early-stop / 2-step-async unchanged; §8.3 gates every variant |
| Over-claiming correctness | Silent quality regression | Correctness (§8.3) is a **hard gate**, run before any τ claim |

---

## 13. Explicit non-goals
- No target fine-tuning (frozen throughout).
- No change to the rejection-sampling rule or scheduler math (routing affects only `p^d`).
- No multimodal/vision drafting.
- No new SLA/serving infra unless Phase 7 is opted in.

---

## 14. Correctness note (why this is safe by construction)
Speculative decoding is lossless for **any** draft distribution `p^d`: tokens are accepted with `min(1, p^t/p^d)` and rejections resample from the target-residual. Routing changes only *which* draft experts shape `p^d` — it cannot change the target `p^t` or the acceptance rule. Therefore **every variant here is output-equivalent to running the target alone**; the experiment moves only the *speed* axis (τ), never quality. §8.3 verifies this empirically as a guardrail against implementation bugs.

---

## 15. Provenance & confidence
- **Grounded (config-verified):** all §1 model facts (fetched from the HF `config.json`).
- **Grounded (paper-verified):** DSpark architecture, losses, scheduler, τ figures (arXiv 2607.05147).
- **[INFERENCE] / design proposals:** router-reuse hypothesis, 256→K collapse methods, injection-source choice for the hybrid stack, expected τ gains on chat/prose. These are hypotheses to be **measured**, not established results. Any τ/throughput number stated as an expectation is an estimate until Phase 5.
