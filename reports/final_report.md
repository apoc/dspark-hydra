# Final report — Domain-Routed Speculative Drafting via Target-Router Reuse (DSpark-MoE on Qwen3.6-35B-A3B)

**Target (tightened, spec §0):** at **equal active draft FLOPs**, does reusing the frozen target
MoE router to drive a domain-partitioned MoE draft raise accepted length τ over a dense draft
(better accuracy-per-active-FLOP)? Which domains benefit is a question, not a success gate.
Efficiency (same τ at smaller active cost) is a corollary; draft *speed* is explicitly not claimed
(draft latency is ~5 % of `L=(T_draft+T_verify)/τ`).

> **Bottom line (updated with the v3 17× scaled run — the reuse verdict is SCALE-DEPENDENT).**
> **The target-router *soft-reuse* advantage is null at small data but emerges strongly at scale.**
> At v1/v2 (83 k–246 k tokens) soft-reuse (E2) ≈ from-scratch (C1) — reuse was not demonstrably the
> lever. At **v3 (4.3 M tokens, ~17×)** the picture reverses and is **robust to Holm + Bonferroni
> correction**: E2 soft-reuse **beats dense by +4.4 % macro** ([+0.055,+0.108]; **math +7.6 % & code
> +7.4 % both clear the 5 % relevance line** — a first) **and beats from-scratch C1 by +5.3 % macro**
> ([+0.070,+0.122], p<1e-4). Meanwhile **frozen-reuse (E1) is significantly *worse* than dense** (−3.0 %)
> and **scratch (C1) ≈ dense** — so the win is specifically the *learnable, target-distilled* router,
> not routing per se and not a frozen map. **Caveats:** gains are on math/code/chat — **prose still does
> not benefit** (RQ3 unsupported); the v2→v3 reversal is confounded (17× data *and* an augmented code
> corpus + disjoint eval block), so the clean claim is the *within-v3* contrast, not a pure data-scale
> attribution; draft is still ~0.27 B (~14× below production) at absolute τ≈1.9 (vs ~4). τ lossless (§8.3).

---

## 1. Setup

- **Target (frozen):** Qwen3.6-35B-A3B (bf16), DGX Spark (GB10). Draft ≈ **266 M active** (dense) /
  **454 M total, ~278 M active** (MoE, K=16, k'=2) — a *deliberately small* budget to isolate routing
  at equal active params (~14× below a production DSpark draft; RedHat's public GLM-5.2 DSpark
  speculator is 3.807 B, τ≈3.97).
- **Variants** (share backbone, KV-injection, Markov + confidence heads, data, equal active FLOPs):
  **B3** dense · **E1** hard-reuse (frozen collapse map C) · **E2** soft-reuse (distilled draft
  router) · **C1** from-scratch (learned router, no target signal).
- **Three training scales:** **v1 = 83 k tokens** (400 seqs), **v2 = 246 k tokens** (1200 seqs, 3×),
  **v3 = 4.3 M tokens** (20 k seqs, ~17×; code corpus augmented with 18 k python-instructions so all
  domains reach 5 k prompts). v1/v2 uniform per-domain; token mix ≈ 29/20/20/30 math/code/chat/prose.
  Runs on the hotdog A100 cluster (Slurm, preempt) via vLLM batched generation + HF teacher-force
  extraction; v1/v2 on DGX Spark. Neither mix matches DSpark's 39/39/18/4 — out of scope.
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

**v3 (4.3 M tokens, ~17× v2; code corpus augmented with 18 k python-instructions; skip=5000):**

| variant | math | code | chat | prose | macro | val_loss |
|---|--:|--:|--:|--:|--:|--:|
| B3 dense | 1.898 | 1.839 | 1.874 | 1.769 | 1.845 | 4.20 |
| E1 hard | 1.878 | 1.839 | 1.810 | 1.632 | 1.790 | 4.31 |
| **E2 soft** | **2.042** | **1.975** | **1.963** | 1.726 | **1.926** | 4.06 |
| C1 scratch | 1.920 | 1.871 | 1.824 | 1.703 | 1.830 | 4.25 |

**v3 paired macro contrasts (Holm + Bonferroni-robust):** **E2−B3 +0.081 [+0.055,+0.108]** ✓
(math +7.6 %, code +7.4 % clear 5 %); **E2−C1 +0.097 [+0.070,+0.122]** ✓ (p<1e-4; chat clears 5 %,
prose n.s.); **E1−B3 −0.055 [−0.079,−0.032]** (frozen reuse significantly *worse* than dense);
C1−B3 −0.016 [−0.039,+0.008] (≈dense). All survive Holm @0.05 except C1−B3; Bonferroni 98.75 % CIs
exclude 0 for E2−B3, E2−C1, E1−B3. Eval n=50/domain×2 (v2-matched); effects large, amply powered.

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

**RQ1 (routed MoE vs dense at equal active params).** **Scale-dependent; strongly "yes" at v3.**
At v1/v2 only E2 (soft) beat dense, weakly (+2.0 % macro, <5 %). At **v3 (17×), E2 beats dense by
+4.4 % macro ([+0.055,+0.108]), with math +7.6 % and code +7.4 % clearing the 5 % relevance line** —
the first practically-relevant gains in the study. It is variant-specific: **scratch C1 ≈ dense
(−0.8 %, n.s.)** and **frozen E1 is significantly *worse* (−3.0 %)**. So "routing helps" only through
the *learnable, target-distilled* router (E2), not routing in general.

**RQ2 (is the *reused* router the lever?).** **Scale-dependent — NO at small data, YES at 17× data.**
At v1/v2 soft-reuse (E2) ≈ from-scratch (C1) (v2 E2−C1 macro +0.010, CI incl 0, p=.38) — reuse was not
demonstrably the lever. At **v3 the verdict reverses and is correction-robust: E2 (soft reuse) beats
C1 (scratch) by +0.097 macro ([+0.070,+0.122], p<1e-4, Holm & Bonferroni ✓)**, positive on all four
domains (chat clears 5 %; prose n.s.). Frozen reuse (E1) stays rejected (significantly worse than
dense), so the lever is specifically the *learnable, target-distilled* router — **the target's routing
knowledge pays off only once there is enough data to distil it.** This **supports the paper's
target-router-reuse hypothesis at scale** (with the §5 confound caveat).

**RQ3 (prose / interference).** **Not a robust win.** Point estimates turned positive at v2 (C1 +3.9 %,
E2 +2.6 %) vs v1's flat/negative, but **neither survives multiple-comparison correction** (C1 prose
p=.029 > Holm thr .005; E2 prose p=.15) — the earlier "prose improved at scale" read was an
uncorrected artifact of 12 per-domain tests. Prose is *no longer clearly the failure domain*, but the
hypothesized high-entropy benefit is **not** demonstrated. Separability at ℓ\* unchanged (§ RQ5).

**RQ4 (hard vs soft reuse) + methodological finding.** **Soft ≫ hard, decisively at v3:** soft-reuse
E2 (macro 1.926) beats hard-reuse E1 (1.790) by +0.136, and E1 is significantly *worse than dense*
(−0.055) — a **frozen** collapse map actively hurts while a **distilled/learnable** one wins; adaptive
routers also reach lower val_loss than frozen E1 at every scale. **Methodological caveat (v2):** the
saturation-autostop (patience 6) under-trained the learned-router variants (delayed router↔expert
descent: temporary plateau ~step 6400 then resumes) — a first v2 pass early-stopped E2/C1 and made them
look *worse than dense*, a pure artifact fixed by ≥12 k-step retrains (and patience-12 throughout v3).
Lesson: learned-router drafts need a higher patience / min-steps floor or you get false negatives.

**RQ5 (router source ℓ\*).** **Aggregate source is not the prose lever.** Rebuilding C from the
aggregate router (mean-softmax over full-attn layers; consistent `d=log(agg)`) changed nothing
significant vs ℓ\*=39 except +4 % E1 math; domain Jaccard ≈ unchanged (chat~prose 0.66–0.68 at both
scales). Caveat: the aggregate is a *blend*; a single sharp earlier/mid layer (ℓ\*=19/27) is untested
(needs a per-layer re-dump) and is the only clean RQ5 probe left — RQ5 not fully closed.

**RQ6 (lossless).** **Confirmed by construction + test.** Rejection-sampler KL to p^t ≈ 3e-5; the
all-accept bonus-index bug was fixed and regression-tested. Routing only reshapes p^d.

## 5. Scaling read

**v1→v2→v3 (83 k → 246 k → 4.3 M tokens).** The soft-reuse advantage *grows with data*: E2−dense macro
+2.3 %→+2.0 %→**+4.4 %**, and E2−scratch (the reuse-specific lever) +0.9 %→+1.0 %→**+5.3 %**, crossing
from n.s. to correction-robust between v2 and v3. **Confound:** v3 differs from v2 in *both* data volume
(17×) *and* corpus (code augmented with 18 k python-instructions) *and* eval block (skip 5000 vs 300),
so the reversal cannot be pinned on data scale alone — the clean evidence is the *within-v3* paired
contrast. **Standing caveats:** v3 is still ~1–8 % of production data; draft ~14× below production;
absolute τ≈1.9 vs ~4; prose never benefits. Whether production capacity widens or narrows the reuse gap
is untested (draft held at ~0.27 B).

## 6. Recommendation

- **At scale, soft target-router reuse (E2) is a real, correction-robust win** — +4.4 % macro over
  dense (math/code >5 %) and +5.3 % over a from-scratch router: the study's strongest positive and the
  first result clearing practical relevance. **Prefer soft-reuse over both dense and scratch when
  training data is ample.**
- **But it is data-gated and variant-specific:** at small data reuse ≈ scratch (no advantage), and
  *frozen* reuse (E1) actively hurts. Don't use a frozen collapse map; don't expect the win at small data.
- **Not yet shippable:** the reversal is confounded (data + corpus + eval-block), τ≈1.9 is far below
  production, prose never benefits, draft is ~14× undersized. Decisive next tests: isolate data-scale
  from corpus (fix corpus, sweep volume), a larger draft (capacity), the single-layer ℓ\* probe (RQ5),
  production-scale validation.

## 7. Artifacts

Raw per-prompt×seed τ: `reports/tau_{B3_dense,E1_hard,E2_soft,C1_scratch}{,_v2,_v3}_power.json`,
corrected learned-router v2: `tau_{E2_soft_long,C1_scratch_long}_v2_power.json`; aggregate diagnostic:
`*_agg_power.json`. Paired-bootstrap summaries: `power_summary*.{md,json}` (incl. `_v3`,
`_v3_reuse_vs_scratch`, `_v2_corrected`, `_agg_vs_star_{E1,E2}`). v3 pipeline: `scripts/{gen_responses,
dump_from_responses,merge_dump,build_C,train_draft,eval_tau}.py` + `hotdog_{gen,extract,train,eval}.sbatch`
(sharded, preempt-safe). Preliminary write-up: `preliminary_results.md`.
Spec + errata: `doc/dspark-qwen36-moe-router-reuse-experiment.md`.
