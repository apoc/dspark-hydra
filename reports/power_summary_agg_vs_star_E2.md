# Power analysis (exploratory, DSpark-faithful) - ref=E2_soft, gamma=5
tau = accepted length incl. bonus token (DSpark Sec4.1 fn4); norm = tau/(gamma+1)=6.
Contrast = paired per-prompt Delta (variant-ref); CI = cluster-bootstrap-by-prompt (prompts resampled,
conditional on the fixed decoding seeds [0, 1]). 5% line is a REFERENCE, not a gate.

## Mean tau + latency per variant
| variant | domain | n | tau | norm | t_anchor | t_draft | t_verify | L_offline ms/tok |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| E2_soft | math | 50 | 1.897 | 0.316 | 307.6 | 15.9 | 316.2 | 339.36 |
| E2_soft | code | 50 | 1.854 | 0.309 | 240.8 | 15.9 | 252.2 | 276.59 |
| E2_soft | chat | 50 | 1.825 | 0.304 | 230.0 | 16.0 | 241.5 | 268.96 |
| E2_soft | prose | 50 | 1.591 | 0.265 | 334.8 | 15.4 | 338.7 | 440.89 |
| E2_soft | **macro** | - | **1.792** | 0.299 | | | | |
| E2_soft_agg | math | 50 | 1.907 | 0.318 | 305.5 | 15.8 | 314.7 | 335.98 |
| E2_soft_agg | code | 50 | 1.834 | 0.306 | 238.7 | 15.8 | 250.2 | 277.70 |
| E2_soft_agg | chat | 50 | 1.815 | 0.302 | 228.7 | 15.7 | 240.4 | 269.44 |
| E2_soft_agg | prose | 50 | 1.565 | 0.261 | 333.6 | 15.5 | 337.4 | 450.18 |
| E2_soft_agg | **macro** | - | **1.780** | 0.297 | | | | |

## Paired contrasts vs E2_soft (Delta = variant - E2_soft)
| variant | domain | mean Delta | rel% | 95% CI (abs) | CI lower rel% | >ref-line? |
|---|---|--:|--:|:--:|--:|:--:|
| E2_soft_agg | math | +0.011 | +0.6% | [-0.032, +0.053] | -1.7% | no |
| E2_soft_agg | code | -0.020 | -1.1% | [-0.056, +0.016] | -3.0% | no |
| E2_soft_agg | chat | -0.010 | -0.6% | [-0.044, +0.022] | -2.4% | no |
| E2_soft_agg | prose | -0.026 | -1.6% | [-0.082, +0.032] | -5.2% | no |
| E2_soft_agg | **macro (pooled)** | -0.012 | -0.6% | [-0.033, +0.010] | -1.9% | no |

Flag: 'yes' = CI lower bound clears the reference line; '~' = point estimate clears but CI does not;
'no' = below. This is descriptive only (exploratory report; no accept/reject decision).
