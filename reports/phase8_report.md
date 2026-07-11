# Phase 8 — Report: Domain-Routed Speculative Drafting via Target-Router Reuse

**Status:** Preliminary **sanity-scale** result (Phase 4 exit gate), not the production-scale
study. One end-to-end pass of the full §6 run matrix on Qwen3.6-35B-A3B.

## Setup (what was actually run)

- **Target (frozen):** Qwen3.6-35B-A3B, bf16, on DGX Spark (GB10).
- **Calibration/training dump:** 400 target-generated sequences, **83,746 tokens**
  (math 24.4k, code 17.1k, chat 17.0k, prose 25.2k). Non-thinking chat + prose completion.
- **Collapse map `C`:** co-activation clustering, K=16, k′=2, ℓ*=39, warm-init from weight
  centroids. Balance cv=1.47, 7 (of 16) groups under-served.
- **Training:** knowledge-saturation autostop (validation-plateau early-stop, by-sequence
  split, min_steps=1000, best-ckpt restore). Saturation reached at 9.8k–15.6k steps.
- **Eval:** offline τ (scheduler OFF, γ=5, temp 1.0), 12 held-out prompts/domain, max_new=64.
- **τ definition:** mean accepted draft tokens per round (+1 bonus only when all γ accepted).
  Consistent across variants; the **relative** comparison is what matters.

## Variants (§6 — all share backbone, KV-injection, Markov + confidence heads, data, and equal active FLOPs)

| # | Name | Draft FFN | Routing | Isolates |
|---|---|---|---|---|
| B3 | **dense** | single FFN, width `k'·512`=1024 | none (every token → same FFN) | baseline (standard DSpark) |
| E1 | **hard-reuse** | MoE, K=16 experts, k'=2 active | **frozen** collapse map: pick top-2 draft groups from the target's own router `g = C·softmax(d)`; only expert weights train | value of reusing the target's partition (RQ1) |
| E2 | **soft-reuse** | MoE, K=16, k'=2 | draft's own tiny router `R_d(h)`, **distilled toward** `C·softmax(d)` (learns the target partition but can adapt) | frozen vs adaptive reuse (RQ4) |
| C1 | **from-scratch** | MoE, K=16, k'=2 | draft router `R_d(h)` trained on the drafting loss **only — no target-router signal** | is the reused signal actually useful, or would any learned router do? (RQ2 control) |

`d` = the target's 256-way router logits at layer ℓ*=39, computed for free during verification.
`C` = offline 256→16 co-activation collapse map. **B3 vs E1/E2** answers "does routing help";
**E1/E2 vs C1** answers "is the *target's* partition the right one".

## Diagnostic: balance-constrained `C` (does starvation explain the missing prose gain?)

The default co-activation `C` left 7/16 draft groups under-served and prose spread across
them, suggesting load imbalance might explain why prose τ didn't improve. We rebuilt `C`
with capacity-constrained clustering (equal 16-expert groups, **0 starved**, cv 1.47→1.07)
and **retrained** E1/E2 on it (swapping `C` requires retraining — the experts learn their
group's token distribution).

| variant | C | math | code | chat | prose | macro |
|---|---|---|---|---|---|---|
| E1 hard | unbalanced | 0.928 | 0.902 | 0.765 | 0.513 | 0.777 |
| E1 hard | **balanced** | 0.942 | 0.864 | 0.641 | **0.457** | 0.726 |
| E2 soft | unbalanced | 0.915 | 0.962 | 0.706 | 0.570 | 0.788 |
| E2 soft | **balanced** | 0.925 | 0.838 | 0.710 | **0.567** | 0.760 |

**Balance did not help — it hurt or was neutral.** Macro dropped for both (−0.05 E1, −0.03 E2);
the only gain was math (+0.01). Prose moved the *wrong* way for hard reuse (0.513→0.457) and
was unchanged for soft (0.570→0.567, within 12-prompt noise). Interpretation: the starved
groups reflected the **real, imbalanced** co-activation structure — forcing equal groups
**fragments coherent expert clusters** and degrades routing. Prose's low τ is therefore a
**target-partition separability** limit (prose~others Jaccard 0.66–0.70 — prose shares ~⅔ of
its hot target experts with other domains), **not** a capacity/starvation problem. Balancing
cannot manufacture separability. This de-motivates the balance lever for the scale-up and
points instead to RQ5 (alternative/aggregate router source layer ℓ*) as the route to test
for prose.

## Results (accepted length τ, higher = better)

| domain | B3 dense | E1 hard-reuse | E2 soft-reuse | C1 from-scratch |
|---|---|---|---|---|
| math   | 0.847 | 0.928 | 0.915 | 0.904 |
| code   | 0.844 | 0.902 | **0.962** | 0.806 |
| chat   | **0.795** | 0.765 | 0.706 | 0.765 |
| prose  | 0.533 | 0.513 | 0.570 | **0.576** |
| **macro** | 0.755 | 0.777 | **0.788** | 0.763 |
| val_loss | 3.706 | 3.805 | 3.785 | **3.952** |

## Research questions

**RQ1 (main) — does reused-router MoE beat dense at matched active params?**
**Partially yes.** E2 (0.788) and E1 (0.777) both beat B3 dense (0.755) on macro-τ at
matched active FLOPs. **But the gains land on the low-entropy domains (math +0.08, code
+0.06–0.12), not on chat/prose** as the hypothesis predicted — E1 is actually *worse* than
dense on chat (−0.03) and prose (−0.02). The reused partition helps most where it is
sharpest (code/math), which is the opposite of the "biggest gains on chat/prose" claim.

**RQ2 (free lunch) — reused router vs from-scratch?**
**Weakly supported.** E2 soft (0.788) > C1 scratch (0.763) clearly; E1 hard (0.777) > C1
(0.763) narrowly (likely within noise at 12 prompts/domain). Stronger signal in
optimization: C1's val_loss (3.952) is markedly higher than E1/E2 (~3.80) at saturation —
the reused partition gives a better-conditioned objective even where τ gains are small.
Caveat: C1 matches/beats the reuse variants on chat and prose, so the reused signal is not
uniformly superior.

**RQ3 (interference / variance) — does routing reduce per-domain variance & OOD prose drop?**
**Not supported at this scale.** Per-domain spread widened: B3 range 0.31 → E2 range 0.39.
Routing increased dispersion rather than reducing dilution. Prose (OOD) *did* improve for the
adaptive variants (E2 0.570, C1 0.576 vs B3 0.533) but regressed for hard reuse (E1 0.513).

**RQ4 (hard vs soft reuse) — mechanism.**
**Soft wins.** E2 soft (0.788) > E1 hard (0.777) on macro, and critically the distilled/
adaptive router **recovers prose** (0.570) that the frozen hard map lost (0.513 < dense
0.533). Adaptivity matters exactly on the high-entropy domain where a single-token frozen
partition is noisiest — consistent with the §12 risk "single-token router signal noisy".

**RQ5 (signal source ℓ*)** — not tested (only ℓ*=39). Ablation pending.

**RQ6 (correctness / lossless)** — **confirmed by construction.** All variants share the
proven rejection sampler; `scripts/test_losslessness.py` shows accepted-token KL to p^t ≈
3e-5 for adversarial drafts. Routing only reshapes p^d; output distribution is unchanged.

## Honest caveats (why this is a gate, not a verdict)

1. **Scale:** 83k training tokens is ~1% of the spec's 1–1.5M-prompt target. Absolute τ is
   low (~0.75–0.79; DSpark production τ is 3.5–6). The draft is undertrained.
2. **Eval noise:** 12 prompts/domain — per-domain differences of ±0.03 may be within noise.
3. **C balance:** 7/16 draft groups under-served on this dump; starved experts for chat/prose
   partitions plausibly explain the missing high-entropy gains (§5.4 rebalancing not yet applied).
4. **τ definition** excludes the guaranteed correction token; production reporting would add +1.

## Verdict & next steps

- **Directional support** for the core thesis: a MoE draft routed by the target's collapsed
  router beats an equal-active-FLOP dense draft on macro-τ, and the **soft (distilled)** reuse
  is the best variant — it is the recommended mechanism, not the hard frozen map.
- **The headline hypothesis (biggest gains on chat/prose) is NOT yet confirmed** — at this
  scale the win is on code/math. Whether the prose/chat gain emerges is the key question for
  the scale-up.
- **To reach a verdict:** (1) scale the dump to ~1M prompts via vLLM-batched generation
  (§7.3); (2) apply balance-constrained `C` (§5.4) to stop starving chat/prose groups;
  (3) run the RQ5 ℓ* ablation and the K/k′/γ sweeps (§9); (4) re-measure with ≥100 eval
  prompts/domain for tighter error bars.
