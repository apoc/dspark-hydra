# Power analysis (exploratory, DSpark-faithful) - ref=C1_scratch_long_v2, gamma=5
tau = accepted length incl. bonus token (DSpark Sec4.1 fn4); norm = tau/(gamma+1)=6.
Contrast = paired per-prompt Delta (variant-ref); CI = cluster-bootstrap-by-prompt (prompts resampled,
conditional on the fixed decoding seeds [0, 1]). 5% line is a REFERENCE, not a gate.

## Mean tau + latency per variant
| variant | domain | n | tau | norm | t_anchor | t_draft | t_verify | L_offline ms/tok |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| C1_scratch_long_v2 | math | 50 | 1.912 | 0.319 | 304.4 | 14.3 | 314.1 | 333.29 |
| C1_scratch_long_v2 | code | 50 | 1.895 | 0.316 | 236.2 | 14.3 | 248.0 | 265.40 |
| C1_scratch_long_v2 | chat | 50 | 1.781 | 0.297 | 228.2 | 14.3 | 240.8 | 274.70 |
| C1_scratch_long_v2 | prose | 50 | 1.703 | 0.284 | 341.0 | 14.2 | 344.7 | 423.24 |
| C1_scratch_long_v2 | **macro** | - | **1.823** | 0.304 | | | | |
| E2_soft_long_v2 | math | 50 | 1.932 | 0.322 | 304.6 | 16.0 | 313.9 | 330.05 |
| E2_soft_long_v2 | code | 50 | 1.902 | 0.317 | 234.8 | 15.7 | 246.4 | 263.09 |
| E2_soft_long_v2 | chat | 50 | 1.816 | 0.303 | 227.4 | 15.8 | 239.6 | 269.10 |
| E2_soft_long_v2 | prose | 50 | 1.681 | 0.280 | 338.9 | 15.6 | 342.6 | 428.26 |
| E2_soft_long_v2 | **macro** | - | **1.833** | 0.305 | | | | |

## Paired contrasts vs C1_scratch_long_v2 (Delta = variant - C1_scratch_long_v2)
| variant | domain | mean Delta | rel% | 95% CI (abs) | CI lower rel% | >ref-line? |
|---|---|--:|--:|:--:|--:|:--:|
| E2_soft_long_v2 | math | +0.020 | +1.0% | [-0.020, +0.061] | -1.1% | no |
| E2_soft_long_v2 | code | +0.007 | +0.4% | [-0.032, +0.046] | -1.7% | no |
| E2_soft_long_v2 | chat | +0.035 | +2.0% | [+0.001, +0.070] | +0.0% | no |
| E2_soft_long_v2 | prose | -0.022 | -1.3% | [-0.081, +0.035] | -4.8% | no |
| E2_soft_long_v2 | **macro (pooled)** | +0.010 | +0.5% | [-0.013, +0.032] | -0.7% | no |

Flag: 'yes' = CI lower bound clears the reference line; '~' = point estimate clears but CI does not;
'no' = below. This is descriptive only (exploratory report; no accept/reject decision).
