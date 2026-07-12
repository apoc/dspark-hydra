# Final report — Domain-Routed Speculative Drafting via Target-Router Reuse (DSpark-MoE on Qwen3.6-35B-A3B)

**Target (tightened, spec §0):** at **equal active draft FLOPs**, does reusing the frozen target
MoE router to drive a domain-partitioned MoE draft raise accepted length τ over a dense draft
(better accuracy-per-active-FLOP)? Which domains benefit is a question, not a success gate.
Efficiency (same τ at smaller active cost) is a corollary; draft *speed* is explicitly not claimed
(draft latency is ~5 % of `L=(T_draft+T_verify)/τ`).

> **Bottom line (corrected after fixing a training-autostop artifact — see §5).**
> At 3× training data with **adequate training**, a MoE-routed draft **modestly but significantly
> beats a dense draft** of equal active params on macro-τ (E2 soft +2.0 %, C1 scratch +1.4 %; both
> 95 % CIs exclude 0), **including a significant prose gain** (C1 +3.9 %). **However:** (1) every
> effect is **below the 5 % relevance reference line**; (2) **the target's router contributes nothing.**
> The *frozen* reuse variant (E1 — the literal "free lunch") is the **weakest** routed variant and does
> **not** clear 0 vs dense ([−0.000,+0.044]); warm-starting from the target (E2, distilled) gives **no**
> edge over cold-start (C1, scratch: E2−C1 macro Δ CI includes 0). Only *learned* routers beat dense —
> where the router comes from is irrelevant. The paper's specific "free lunch from target-router reuse"
> novelty is **not validated**; the benefit is from *having a learned routed MoE draft*.
> τ is lossless throughout (§8.3 gate passed).

---

## 1. Setup

- **Target (frozen):** Qwen3.6-35B-A3B (bf16), DGX Spark (GB10). Draft ≈ **266 M active** (dense) /
  **454 M total, ~278 M active** (MoE, K=16, k'=2) — a *deliberately small* budget to isolate routing
  at equal active params (~14× below a production DSpark draft; RedHat's public GLM-5.2 DSpark
  speculator is 3.807 B, τ≈3.97).
- **Variants** (share backbone, KV-injection, Markov + confidence heads, data, equal active FLOPs):
  **B3** dense · **E1** hard-reuse (frozen collapse map C) · **E2** soft-reuse (distilled draft
  router) · **C1** from-scratch (learned router, no target signal).
- **Two training scales:** **v1 = 83 k tokens** (400 seqs), **v2 = 246 k tokens** (1200 seqs, 3×),
  identical uniform per-domain sampling (token mix ≈ 29/20/20/30 math/code/chat/prose; matches v1;
  neither matches DSpark's 39/39/18/4 design — out of scope).
- **Eval:** offline τ (scheduler OFF, γ=5, T=1.0), **τ = n_acc+1** (DSpark Sec 4.1 fn4). 50 held-out
  prompts/domain × 2 seeds, max_new 32; **held-out confirmation blocks disjoint from training**
  (v1 skip=200; v2 skip=300, past the 300/domain training range). Paired **cluster-bootstrap-by-
  prompt** 95 % CIs (prompts resampled; conditional on the fixed seeds).

## 2. Results — mean τ (higher better; ceiling 6)

**v1 (83 k tokens):**  *(learned-router E2/C1 likely undertrained here too — see §5; treat as lower bounds)*

| variant | math | code | chat | prose | macro |
|---|--:|--:|--:|--:|--:|
| B3 dense | 1.820 | 1.759 | 1.845 | 1.579 | 1.751 |
| E1 hard | 1.868 | 1.845 | 1.837 | 1.535 | 1.771 |
| E2 soft | 1.897 | 1.854 | 1.825 | 1.591 | 1.792 |
| C1 scratch | 1.875 | 1.848 | 1.820 | 1.558 | 1.775 |

**v2 (246 k tokens, 3×) — all variants trained to genuine saturation:**

| variant | math | code | chat | prose | macro | steps / val_loss |
|---|--:|--:|--:|--:|--:|--:|
| B3 dense | 1.876 | 1.870 | 1.805 | 1.639 | 1.797 | 12000 / 3.90 |
| E1 hard | 1.875 | 1.953 | 1.812 | 1.639 | 1.820 | 14800 / 3.76 |
| **E2 soft** | 1.932 | 1.902 | 1.816 | 1.681 | **1.833** | 19600 / 3.742 |
| C1 scratch | 1.912 | 1.895 | 1.781 | 1.703 | 1.823 | 20200 / 3.713 |

## 3. Paired contrasts, v2 (Δ = variant − B3 dense; 95 % cluster-bootstrap CI)

| variant | math | code | chat | prose | **macro** |
|---|--:|--:|--:|--:|--:|
| E1 hard | −0.001 | **+0.083** ✓ | +0.007 | +0.000 | +0.022 [−0.000,+0.044] |
| E2 soft | **+0.056** ✓ | +0.032 | +0.011 | +0.042 | **+0.035 [+0.014,+0.057]** ✓ |
| C1 scratch | +0.036 | +0.025 | −0.024 | **+0.065** ✓ | **+0.026 [+0.004,+0.048]** ✓ |

✓ = 95 % CI excludes 0. **All below the 5 % relevance line.** **RQ2 direct contrast E2−C1 (reuse −
scratch): macro +0.010 [−0.013, +0.032] — CI includes 0** (reuse not significantly better than scratch).

## 4. Research questions

**RQ1 (routed MoE vs dense at equal active params).** **Yes, modestly, at scale.** At v2 with adequate
training, E2 (+2.0 %) and C1 (+1.4 %) beat dense on macro-τ with CIs excluding 0; E1 (+1.2 %) touches
0. Gains are broad but small — no variant clears 5 %. (At v1 the ranking was similar but the
learned-router variants were undertrained, §5.)

**RQ2 (is the *reused* router the lever?).** **No — the central negative, in two parts.** (i) The
*frozen* target router (E1 — the paper's literal "free lunch") is the **weakest** routed variant:
macro Δ +0.022 vs dense with CI touching 0 ([−0.000,+0.044]) and the worst routed val_loss (3.76).
(ii) Warm-starting from the target (E2, distilled) gives **no** edge over cold-start (C1, scratch):
E2−C1 macro Δ +0.010, CI includes 0; C1 even has the best prose. So the target's router contributes
nothing — neither as-is nor as init. What helps is a *learned* routed MoE draft; its provenance is
irrelevant. The paper's "free lunch from reusing the target router" is not validated on this testbed.

**RQ3 (prose / interference).** **Improved at scale (with training).** Unlike v1 (routing flat/negative
on prose), v2 shows real prose gains — C1 +3.9 % (CI excl 0), E2 +2.6 % — once both data and training
are adequate. Prose is no longer the failure domain, though separability at ℓ\* is unchanged (§ RQ5).

**RQ4 (hard vs soft reuse) + methodological finding.** Adaptive routers (soft E2, scratch C1) reach
**lower val_loss** (3.71–3.74) than frozen-map hard E1 (3.76) and dense (3.90), and slightly higher τ.
**Critical caveat:** the saturation-autostop (patience 6) **systematically under-trained the
learned-router variants** — their loss plateaus *temporarily* (~step 6400, val 4.28) then resumes
descending (delayed router↔expert co-adaptation). A first v2 pass early-stopped E2/C1 there and made
them look *worse than dense* (macro 1.740/1.716) — a pure artifact. Forced ≥12 k-step retrains
(E2_long 19.6 k, C1_long 20.2 k) recovered them to the numbers above. Dense/hard (no learned router)
converged smoothly and were unaffected. **Lesson: patience-based early-stop needs a higher patience /
min-steps floor for learned-router drafts, or it produces false negatives.**

**RQ5 (router source ℓ\*).** **Aggregate source is not the prose lever.** Rebuilding C from the
aggregate router (mean-softmax over full-attn layers; consistent `d=log(agg)`) changed nothing
significant vs ℓ\*=39 except +4 % E1 math; domain Jaccard ≈ unchanged (chat~prose 0.66–0.68 at both
scales). Caveat: the aggregate is a *blend*; a single sharp earlier/mid layer (ℓ\*=19/27) is untested
(needs a per-layer re-dump) and is the only clean RQ5 probe left — RQ5 not fully closed.

**RQ6 (lossless).** **Confirmed by construction + test.** Rejection-sampler KL to p^t ≈ 3e-5; the
all-accept bonus-index bug was fixed and regression-tested. Routing only reshapes p^d.

## 5. Scaling read

3× data **strengthens** the routed-vs-dense case (v1 macro Δ +1.2–2.3 % → v2 +1.2–2.0 % but now with
significant prose gains and cleanly-trained learned routers), while leaving the **reuse-vs-scratch**
verdict negative. It also surfaced the autostop artifact (§ RQ4) that would otherwise have produced a
false negative. **Caveats:** (a) 3× (246 k tokens) is still ~1 % of production scale (RedHat ~500 k
prompts × full responses × 3 epochs); (b) the draft is ~14× smaller than production; (c) cross-scale
τ comparisons (v1 skip=200 vs v2 skip=300) are on **disjoint prompt blocks** — suggestive, not paired;
the within-scale paired contrasts (§3) are the clean evidence. So this shows routing helps *modestly*
at this budget and that reuse ≠ scratch, leaving open whether production scale/capacity would widen
either gap.

## 6. Recommendation

- **MoE routing in the draft is a mild, real positive** at equal active params (macro +1.4–2.0 % at
  3× data, incl. prose) — worth pursuing at production scale.
- **Do not adopt target-router *reuse* specifically** on this evidence: a from-scratch learned router
  is simpler and statistically as good (RQ2). The paper's reuse "free lunch" is unconfirmed here.
- Effects are **below 5 %**; before shipping, the decisive tests are: production-scale data + larger
  draft (isolate capacity), the single-layer ℓ\* re-dump (clean RQ5), and a learned-router-aware
  training schedule (the autostop fix). Absent those, dense DSpark remains the safe default.

## 7. Artifacts

Raw per-prompt×seed τ: `reports/tau_{B3_dense,E1_hard,E2_soft,C1_scratch}{,_v2}_power.json`, corrected
learned-router v2: `tau_{E2_soft_long,C1_scratch_long}_v2_power.json`; aggregate diagnostic:
`*_agg_power.json`. Paired-bootstrap summaries: `power_summary*.{md,json}` (incl. `_v2_corrected`,
`_v2_reuse_vs_scratch`, `_agg_vs_star_{E1,E2}`). Preliminary write-up: `preliminary_results.md`.
Spec + errata: `doc/dspark-qwen36-moe-router-reuse-experiment.md`.
