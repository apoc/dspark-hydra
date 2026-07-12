# Final report — Domain-Routed Speculative Drafting via Target-Router Reuse (DSpark-MoE on Qwen3.6-35B-A3B)

**Target (tightened, spec §0):** at **equal active draft FLOPs**, does reusing the frozen target
MoE router to drive a domain-partitioned MoE draft raise accepted length τ over a dense draft
(better accuracy-per-active-FLOP)? Which domains benefit is a question, not a success gate.
Efficiency (same τ at smaller active cost) is a corollary; draft *speed* is explicitly not claimed
(draft latency is ~5 % of `L=(T_draft+T_verify)/τ`).

> **Bottom line (corrected: autostop artifact fixed — §5; multiple-comparison correction applied — §3).**
> At 3× data with adequate training, an *adaptive*-router MoE draft shows a **small, correction-robust
> edge** over dense on macro-τ — but **only the soft variant E2** survives Holm correction over the
> primary family (macro **+2.0 %**, p=.001; math +3.0 %), alongside **E1's code gain** (+4.4 %). **All
> are below the 5 % relevance line — nothing clears both correction-significance *and* practical
> relevance.** The broader "routing helps everywhere incl. prose" reading does **not** survive: C1's
> macro (+1.4 %) and prose (+3.9 %) gains were uncorrected/exploratory and collapse under correction
> (98.75 % CIs include 0). **Reuse verdict (split):** (i) the *frozen* target router (E1 — the literal
> "free lunch") is **rejected** — it doesn't beat dense. (ii) *Distilled* reuse (E2) vs from-scratch
> (C1) is **statistically indistinguishable** (E2−C1 CI includes 0, p=.38) — a *failure to reject*, not
> proven equivalence; and E2 (reuse) is in fact the lone correction-robust dense-beater while C1
> (scratch) fails Holm, so a *small reuse edge we're underpowered to prove* can't be excluded. Net: the
> "free lunch" is **not demonstrated** — frozen reuse fails; distilled-reuse-vs-scratch is unresolved at
> this scale/power. What clearly helps is a *learned* routed MoE draft, and only modestly. τ lossless (§8.3).

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

## 3. Paired contrasts, v2 (Δ = variant − B3 dense; *uncorrected* 95 % cluster-bootstrap CI — Holm-corrected verdict in note)

| variant | math | code | chat | prose | **macro** |
|---|--:|--:|--:|--:|--:|
| E1 hard | −0.001 | **+0.083** ✓ | +0.007 | +0.000 | +0.022 [−0.000,+0.044] |
| E2 soft | **+0.056** ✓ | +0.032 | +0.011 | +0.042 | **+0.035 [+0.014,+0.057]** ✓ |
| C1 scratch | +0.036 | +0.025 | −0.024 | **+0.065** ✓ | **+0.026 [+0.004,+0.048]** ✓ |

✓ = *uncorrected* 95 % CI excludes 0 (exploratory). **With Holm correction** (4 primary macro tests;
12 per-domain), **only E2 macro (+2.0 %, p=.001), E2 math (+3.0 %), and E1 code (+4.4 %) survive**;
**C1 macro (+1.4 %) and the C1/E2 prose gains do NOT** (98.75 % CIs include 0) — treat those as
exploratory. **Every survivor is still below the 5 % relevance line.** **RQ2 contrast E2−C1 (reuse −
scratch): +0.010 [−0.013, +0.032], p=.38 — a robust null** (reuse no better than scratch).

## 4. Research questions

**RQ1 (routed MoE vs dense at equal active params).** **A small, correction-robust "yes" — but for the
adaptive *soft* variant only.** After Holm correction, **only E2 (soft) beats dense** on macro (+2.0 %,
p=.001) and math (+3.0 %); C1 (scratch, +1.4 %) is directionally similar and *statistically
indistinguishable from E2*, yet does **not** independently clear the corrected bar; frozen E1 touches
0. All gains are <5 %. Routing helps, but weakly and only via the adaptive variant that best exploits
the extra data. (v1 ranking similar; learned-router variants undertrained there, §5.)

**RQ2 (is the *reused* router the lever?).** **Not demonstrated — a split verdict.** (i) The
*frozen* target router (E1 — the paper's literal "free lunch") is the **weakest** routed variant:
macro Δ +0.022 vs dense with CI touching 0 ([−0.000,+0.044]) and the worst routed val_loss (3.76).
(ii) *Distilled* reuse (E2) vs from-scratch (C1) is **statistically indistinguishable** (E2−C1 macro
Δ +0.010, CI includes 0, p=.38) — but a *failure to reject* is **not** proven equivalence (no
TOST/margin pre-specified). Tension: E2 (reuse) is the **lone** correction-robust dense-beater while
C1 (scratch) fails Holm — more consistent with *underpowered to distinguish (possibly a small reuse
edge)* than with true equivalence. **Net:** frozen reuse is rejected; any distilled-reuse advantage is
unresolved at this scale/power. The paper's "free lunch from reuse" is **not demonstrated** here.

**RQ3 (prose / interference).** **Not a robust win.** Point estimates turned positive at v2 (C1 +3.9 %,
E2 +2.6 %) vs v1's flat/negative, but **neither survives multiple-comparison correction** (C1 prose
p=.029 > Holm thr .005; E2 prose p=.15) — the earlier "prose improved at scale" read was an
uncorrected artifact of 12 per-domain tests. Prose is *no longer clearly the failure domain*, but the
hypothesized high-entropy benefit is **not** demonstrated. Separability at ℓ\* unchanged (§ RQ5).

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

3× data **modestly strengthens** the routed-vs-dense signal (v1 macro Δ +1.2–2.3 % → v2 +1.2–2.0 %),
and cleanly-trained learned routers now sit at or above dense — but after correction only E2's
macro/math edge and E1's code gain survive (prose gains do **not**). It also surfaced the autostop
artifact (§ RQ4) that would otherwise have produced a false negative. **Caveats:** (a) 3× (246 k
tokens) is still ~1 % of production scale (RedHat ~500 k prompts × full responses × 3 epochs); (b) the
draft is ~14× smaller than production; (c) cross-scale τ comparisons (v1 skip=200 vs v2 skip=300) are
on **disjoint prompt blocks** — suggestive, not paired; the within-scale paired contrasts (§3) are the
clean evidence. So routing helps *modestly* at this budget; the reuse-vs-scratch question is
**unresolved** (frozen reuse rejected, distilled-vs-scratch underpowered), and whether production
scale/capacity would widen any gap is untested.

## 6. Recommendation

- **A learned routed MoE draft is a mild, real positive** at equal active params — but small
  (correction-robust only for E2's macro +2.0 % / math and E1's code gain; prose does *not* survive)
  and **all effects are <5 %**. Worth re-testing at production scale, not shipping on this.
- **Reuse specifically is unproven, not disproven:** *frozen* target-router reuse (E1) is rejected
  (doesn't beat dense), while *distilled* reuse (E2) is indistinguishable from a scratch router — can't
  confirm or refute a small edge (underpowered). **When indifferent at this power, from-scratch is the
  simpler default** (no router extraction/distillation); adopt distilled reuse only if a properly-powered run shows an edge.
- Effects are **below 5 %**; before shipping, the decisive tests are: production-scale data + larger
  draft (isolate capacity), the single-layer ℓ\* re-dump (clean RQ5), and a learned-router-aware
  training schedule (the autostop fix). Absent those, dense DSpark remains the safe default.

## 7. Artifacts

Raw per-prompt×seed τ: `reports/tau_{B3_dense,E1_hard,E2_soft,C1_scratch}{,_v2}_power.json`, corrected
learned-router v2: `tau_{E2_soft_long,C1_scratch_long}_v2_power.json`; aggregate diagnostic:
`*_agg_power.json`. Paired-bootstrap summaries: `power_summary*.{md,json}` (incl. `_v2_corrected`,
`_v2_reuse_vs_scratch`, `_agg_vs_star_{E1,E2}`). Preliminary write-up: `preliminary_results.md`.
Spec + errata: `doc/dspark-qwen36-moe-router-reuse-experiment.md`.
