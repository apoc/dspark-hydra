# Final report — Domain-Routed Speculative Drafting via Target-Router Reuse (DSpark-MoE on Qwen3.6-35B-A3B)

> **⚠ SUPERSEDED / UPDATE IN FLIGHT (do not act on §0 bottom line below).** The v2 E2/C1 rows
> were trained with an autostop (patience 6) that **prematurely stopped the learned-router variants**
> (E2 soft, C1 scratch) at 6400 steps / val_loss 4.28. A forced-longer retrain (E2_long, 19600 steps,
> val_loss **3.742** — best of all variants) shows E2-soft **beats dense on all four domains including
> prose** (macro 1.833 vs 1.797). The "Not supported / edge shrinks with data" conclusion below is a
> **training artifact** and is being rewritten pending C1_long (forced retrain) + corrected paired CIs.

**Target (tightened, spec §0):** at **equal active draft FLOPs**, does reusing the frozen target
MoE router to drive a domain-partitioned MoE draft raise accepted length τ over a dense draft
(better accuracy-per-active-FLOP)? Which domains benefit is a question, not a success gate.
Efficiency (same τ at smaller active cost) is a corollary; draft *speed* is explicitly not claimed
(draft latency is ~5 % of `L=(T_draft+T_verify)/τ`).

> **Bottom line:** **Not supported.** MoE routing buys a small, real gain on **code** (and, at
> small data, math) at equal active params, but (a) it does **not** help the hypothesized
> high-entropy domains (chat, prose), (b) it does **not** require reusing the target's router
> (from-scratch ≈ reuse at small data), and (c) **the advantage shrinks with data**: at 3× tokens
> the dense baseline catches up on math and prose, leaving routing's durable edge on **code only**,
> with a macro gain whose 95 % CI includes zero. τ is lossless throughout (§8.3 gate passed).

---

## 1. Setup

- **Target (frozen):** Qwen3.6-35B-A3B (bf16), DGX Spark (GB10). Draft ≈ **266 M active** (dense) /
  **454 M total, ~278 M active** (MoE, K=16, k'=2) — a *deliberately small* budget to isolate
  routing at equal active params (~14× below a production DSpark draft; see `preliminary_results.md`).
- **Variants** (share backbone, KV-injection, Markov + confidence heads, data, equal active FLOPs):
  **B3** dense · **E1** hard-reuse (frozen collapse map C) · **E2** soft-reuse (distilled draft
  router) · **C1** from-scratch (learned router, no target signal).
- **Two training scales:** **v1 = 83 k tokens** (400 seqs), **v2 = 246 k tokens** (1200 seqs, 3×),
  identical uniform per-domain sampling (token mix ≈ 29/20/20/30 math/code/chat/prose; matches v1,
  neither matches DSpark's 39/39/18/4 design — out of scope).
- **Eval:** offline τ (scheduler OFF, γ=5, T=1.0), **τ = n_acc+1** (DSpark Sec 4.1 fn4). 50 held-out
  prompts/domain × 2 seeds, max_new 32; **held-out confirmation blocks disjoint from training**
  (v1 skip=200; v2 skip=300, past the 300/domain training range). Paired **cluster-bootstrap-by-
  prompt** 95 % CIs (prompts resampled; conditional on the fixed seeds).

## 2. Results — mean τ (higher better; ceiling 6)

**v1 (83 k tokens):**

| variant | math | code | chat | prose | macro |
|---|--:|--:|--:|--:|--:|
| B3 dense | 1.820 | 1.759 | 1.845 | 1.579 | 1.751 |
| E1 hard | 1.868 | 1.845 | 1.837 | 1.535 | 1.771 |
| E2 soft | 1.897 | 1.854 | 1.825 | 1.591 | **1.792** |
| C1 scratch | 1.875 | 1.848 | 1.820 | 1.558 | 1.775 |

**v2 (246 k tokens, 3×):**

| variant | math | code | chat | prose | macro | train steps / val_loss |
|---|--:|--:|--:|--:|--:|--:|
| B3 dense | 1.876 | 1.870 | 1.805 | 1.639 | 1.797 | 12000 / 3.90 |
| E1 hard | 1.875 | 1.953 | 1.812 | 1.639 | **1.820** | 14800 / 3.76 |
| E2 soft | 1.844 | 1.784 | 1.758 | 1.576 | 1.740 | 6400 / 4.28 † |
| C1 scratch | 1.795 | 1.725 | 1.732 | 1.611 | 1.716 | 6400 / 4.28 † |

† E2/C1 early-plateaued (see §4, RQ4).

## 3. Paired contrasts vs dense (Δ = variant − B3; 95 % CI)

**v1 macro:** E1 +0.021 [−0.003,+0.044] (ns) · E2 **+0.041 [+0.019,+0.062]** (excl 0) · C1 +0.025
[−0.001,+0.049] (borderline). Per-domain: gains concentrated on **code** (+5 %) and **math** (+3–4 %);
chat/prose flat-to-negative.

**v2 macro:** E1 +0.022 [−0.000,+0.044] (**touches 0**) · E2 −0.057 [−0.078,−0.036] · C1 −0.082
[−0.103,−0.060]. E1 per-domain: **code +0.083 [+0.048,+0.120]** (excl 0) is the *only* significant
effect; **math −0.001, chat +0.007, prose 0.000 — all ties**.

## 4. Research questions

**RQ1 (does reused-router MoE beat dense at equal active params?).** **Weakly, and only on code.**
At v1, E2 (soft) beat dense by +2.3 % macro (CI excl 0) but below the 5 % reference line, on
code/math not chat/prose. At v2 (both cleanly trained), **E1-hard's only significant edge over dense
is code (+4.4 %); math and prose become ties as dense catches up; macro edge CI includes zero.** The
apparent v1 math/prose gains were largely a **low-data artifact**.

**RQ2 (is the *reused* router the lever?).** **No.** v1: from-scratch C1 (1.775) ≈ soft-reuse E2
(1.792) and beat hard E1 (1.771) — the benefit is *having a routed MoE*, not the target's partition.

**RQ3 (interference / prose).** **Not supported.** Routing never improved prose over dense at either
scale; at v2 dense's prose *rose* with data (1.579→1.639) and routing matched it (tie).

**RQ4 (hard vs soft reuse).** v1: soft ≥ scratch ≥ hard. v2: **reversed** — the learned-router
variants (E2 soft, C1 scratch) converged to a **worse minimum** (val 4.28) than frozen-C E1 (3.76)
and dense (3.90), plateauing early (oscillating, not descending). *(Confirmatory forced-12k-step E2
retrain: RESULT PENDING — will state whether the plateau is genuine or hparam-induced.)* Under
identical hparams, learning the draft router does not scale as well as a frozen map here.

**RQ5 (router source ℓ\*).** **Aggregate source ≠ prose lever.** Rebuilding C from the aggregate
router (`router_agg`, mean-softmax over full-attn layers) and re-training/evaluating (consistent
`d=log(agg)`) changed **nothing significant except +4 % E1 math**; chat/prose/macro were within noise
vs ℓ\*=39. Domain Jaccard is ~unchanged (chat~prose 0.66–0.68 at both scales) → prose is a
**target-partition separability** limit, invariant to sample size and to this source choice.
*Caveat:* the aggregate is a *blend*; a single sharp earlier/mid layer (ℓ\*=19/27) is untested
(needs a per-layer re-dump) and is the only clean RQ5 probe left — so RQ5 is not fully closed.

**RQ6 (lossless).** **Confirmed by construction + test.** Rejection sampler KL to p^t ≈ 3e-5
(adversarial drafts); the all-accept bonus-index bug was fixed and regression-tested. Routing only
reshapes p^d; output distribution is unchanged for every variant.

## 5. Scaling read (does the undertraining caveat change the conclusion?)

3× data **partially answers** v1's dominant caveat: it did **not** broaden routing's benefit — it
**narrowed** it (dense caught up on math/prose; only code survives). This is a *sharper negative*, not
a failure to measure. **Caveat:** 3× (246 k tokens) is still ~1 % of production scale (RedHat's public
GLM-5.2 DSpark used ~500 k prompts × full responses × 3 epochs), and our draft is ~14× smaller than a
production draft — so this shows **more data does not help routing at this budget**, leaving open
(untested) whether production-scale data + capacity would behave differently.

## 6. Recommendation

**Do not ship router-reuse on this evidence.** The one durable effect (code τ +4–5 % at equal active
params) does not require reusing the target router, does not extend to the high-entropy domains the
hypothesis targeted, and yields a macro gain indistinguishable from zero once the dense baseline is
adequately trained. If pursued further, the decisive next tests are: (a) **production-scale data +
larger draft** (isolate capacity from routing), (b) the **single-layer ℓ\* re-dump** (clean RQ5), and
(c) **hparam-tuned learned-router training** at scale (the E2/C1 plateau). Absent those, the frozen
dense DSpark draft is the recommended configuration.

## 7. Artifacts

Per-variant raw τ (per-prompt×seed): `reports/tau_{B3_dense,E1_hard,E2_soft,C1_scratch}{,_v2}_power.json`;
aggregate diagnostic: `*_agg_power.json`. Paired-bootstrap summaries: `power_summary{,_v2,_agg,
_agg_vs_star_E1,_agg_vs_star_E2}.{md,json}`. Preliminary write-up: `preliminary_results.md`. Spec +
errata: `doc/dspark-qwen36-moe-router-reuse-experiment.md`.
