# Preliminary result — Domain-Routed Speculative Drafting via Target-Router Reuse

**Status: PRELIMINARY, on a probably-undertrained draft.** This supersedes the τ numbers in
`phase8_report.md` (which used a smaller 12-prompt eval and the old bonus-excluded τ). It is a
**sanity-scale signal, not a verdict** — see caveats. Target restated (spec §0, tightened):
*at equal active draft FLOPs, does reusing the frozen target MoE router to drive a
domain-partitioned MoE draft raise accepted length τ over a dense draft (better
accuracy-per-active-FLOP)? Which domains benefit is a question, not a success gate.*

## Setup (power run)

- **Target (frozen):** Qwen3.6-35B-A3B, bf16, DGX Spark (GB10).
- **Draft training dump:** 400 target-generated sequences, **83,746 tokens** (~1% of the spec's
  1–1.5M-prompt goal). Draft ≈ **266M active** (dense) / **278M active, 454M total** (MoE), a
  *deliberately small* budget to isolate routing at equal active params (~14× below a production
  DSpark draft — RedHat's GLM-5.2 speculator is 3.807B, τ≈3.97).
- **Variants** (all share backbone, KV-injection, Markov + confidence heads, data, equal active
  FLOPs): B3 dense · E1 hard-reuse (frozen C) · E2 soft-reuse (distilled router) · C1 from-scratch
  (learned router, no target signal).
- **Collapse map C:** co-activation, K=16, k'=2, ℓ*=39, warm-init.
- **Eval:** offline τ (scheduler OFF, γ=5, temp 1.0), **τ = n_acc + 1** (DSpark Sec 4.1 fn4,
  bonus-inclusive), **50 prompts/domain × 2 seeds**, max_new=32, held-out confirmation block
  (skip=200, disjoint from dump and prior sanity eval). Per-prompt paired **cluster-bootstrap-by-
  prompt** 95% CIs (prompts resampled; conditional on the fixed seeds {0,1}).

## Results — mean τ (higher = better; ceiling 6)

| domain | B3 dense | E1 hard | E2 soft | C1 scratch |
|---|--:|--:|--:|--:|
| math   | 1.820 | 1.868 | **1.897** | 1.875 |
| code   | 1.759 | 1.845 | **1.854** | 1.848 |
| chat   | **1.845** | 1.837 | 1.825 | 1.820 |
| prose  | 1.579 | 1.535 | **1.591** | 1.558 |
| **macro** | 1.751 | 1.771 | **1.792** | 1.775 |

## Paired contrasts vs B3 (Δ = variant − dense; 95% cluster-bootstrap CI)

| variant | macro Δ | rel% | 95% CI | signif. (CI excl. 0)? |
|---|--:|--:|:--:|:--:|
| E1 hard | +0.021 | +1.2% | [−0.003, +0.044] | no |
| C1 scratch | +0.025 | +1.4% | [−0.001, +0.049] | borderline |
| **E2 soft** | **+0.041** | **+2.3%** | **[+0.019, +0.062]** | **yes** |

Per-domain (CI-excludes-0 marked ✓): **code** E1 +4.9%✓ / E2 +5.4%✓ / C1 +5.0%✓; **math** E1
+2.6%✓ / E2 +4.2%✓ / C1 +3.0%✓; **chat** all slightly negative (ns); **prose** E1 −2.8%, E2
+0.8%, C1 −1.3% (all ns). Latency: MoE `t_draft` 13–16 ms vs dense 10 ms (slightly *slower*);
`L` is dominated by the two ~300 ms target forwards, so draft speed is immaterial.

## Interpretation (preliminary)

1. **A small, real MoE-drafting gain exists** — E2 (soft reuse) macro +2.3% with a paired CI
   that excludes zero. But it is **below the 5% relevance reference line**.
2. **The gain is on code + math** (low-entropy, structured), **not** the hypothesized chat/prose
   (flat-to-negative). Prose, the frontier, shows no benefit — consistent with the earlier
   diagnostic that prose is a **target-partition separability** limit at ℓ*=39, not a capacity
   or balance problem.
3. **Router *reuse* is not the lever (RQ2).** From-scratch C1 (1.775) essentially matches soft
   reuse (1.792) and *beats* hard reuse (1.771). The benefit comes from **having a routed MoE
   draft**, not from reusing the target's specific partition. RQ2's "reused ≥ scratch" is not
   supported at this scale.
4. **Mechanism (RQ4):** soft (distilled, adaptive) ≥ scratch ≥ hard (frozen). If any routing is
   used, the adaptive router is best; the frozen collapse map is the weakest.

## Caveats (why this is preliminary, not a verdict)

- **Undertrained:** 83k training tokens ≈ 1% of the spec target; absolute τ (~1.75, ceiling 6)
  is far below production DSpark (RedHat 3.97, ceiling 8). Draft quality is data-starved.
- **Small draft:** ~0.27B active, ~14× below a production draft — capacity is also unprobed.
- **γ=5** (ceiling 6) vs production block 7–8; limits headroom.
- Inference conditional on 2 decoding seeds; n=50 prompts/domain.

## Next steps (planned)

1. **ℓ\* aggregate diagnostic** — rebuild C from the aggregate router signal (`router_agg`,
   already in the dump) instead of ℓ*=39, retrain E1/E2, re-eval. Last cheap mechanistic check
   of whether a different router source rescues chat/prose.
2. **Scale to ~1M tokens** — online vLLM hidden-state streaming (per RedHat/`speculators` recipe)
   or batched offline dump; retrain all variants; re-measure. Tests whether the effect
   strengthens (or the null holds) once the draft is no longer data-starved.
3. **Final writeup** after the scaled run.
