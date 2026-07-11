# Power analysis (exploratory, DSpark-faithful) - ref=B3_dense, gamma=5
tau = accepted length incl. bonus token (DSpark Sec4.1 fn4); norm = tau/(gamma+1)=6.
Contrast = paired per-prompt Delta (variant-ref); CI = cluster-bootstrap-by-prompt (prompts resampled,
conditional on the fixed decoding seeds [0, 1]). 5% line is a REFERENCE, not a gate.

## Mean tau + latency per variant
| variant | domain | n | tau | norm | t_anchor | t_draft | t_verify | L_offline ms/tok |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| B3_dense | math | 50 | 1.820 | 0.303 | 306.7 | 10.0 | 316.1 | 350.35 |
| B3_dense | code | 50 | 1.759 | 0.293 | 238.7 | 10.0 | 250.6 | 285.38 |
| B3_dense | chat | 50 | 1.845 | 0.307 | 228.0 | 10.0 | 240.5 | 261.92 |
| B3_dense | prose | 50 | 1.579 | 0.263 | 334.0 | 10.0 | 337.8 | 442.03 |
| B3_dense | **macro** | - | **1.751** | 0.292 | | | | |
| E1_hard_agg | math | 50 | 1.942 | 0.324 | 306.0 | 13.1 | 315.1 | 328.88 |
| E1_hard_agg | code | 50 | 1.829 | 0.305 | 239.8 | 13.0 | 251.7 | 277.94 |
| E1_hard_agg | chat | 50 | 1.827 | 0.304 | 228.3 | 13.0 | 240.7 | 265.89 |
| E1_hard_agg | prose | 50 | 1.517 | 0.253 | 335.2 | 13.3 | 339.2 | 461.60 |
| E1_hard_agg | **macro** | - | **1.779** | 0.296 | | | | |
| E2_soft_agg | math | 50 | 1.907 | 0.318 | 305.5 | 15.8 | 314.7 | 335.98 |
| E2_soft_agg | code | 50 | 1.834 | 0.306 | 238.7 | 15.8 | 250.2 | 277.70 |
| E2_soft_agg | chat | 50 | 1.815 | 0.302 | 228.7 | 15.7 | 240.4 | 269.44 |
| E2_soft_agg | prose | 50 | 1.565 | 0.261 | 333.6 | 15.5 | 337.4 | 450.18 |
| E2_soft_agg | **macro** | - | **1.780** | 0.297 | | | | |

## Paired contrasts vs B3_dense (Delta = variant - B3_dense)
| variant | domain | mean Delta | rel% | 95% CI (abs) | CI lower rel% | >ref-line? |
|---|---|--:|--:|:--:|--:|:--:|
| E1_hard_agg | math | +0.122 | +6.7% | [+0.078, +0.171] | +4.3% | ~ |
| E1_hard_agg | code | +0.070 | +4.0% | [+0.033, +0.106] | +1.9% | no |
| E1_hard_agg | chat | -0.018 | -1.0% | [-0.052, +0.015] | -2.8% | no |
| E1_hard_agg | prose | -0.062 | -3.9% | [-0.129, -0.001] | -8.2% | no |
| E1_hard_agg | **macro (pooled)** | +0.028 | +1.6% | [+0.002, +0.053] | +0.1% | no |
| E2_soft_agg | math | +0.087 | +4.8% | [+0.042, +0.132] | +2.3% | no |
| E2_soft_agg | code | +0.075 | +4.2% | [+0.038, +0.113] | +2.1% | no |
| E2_soft_agg | chat | -0.030 | -1.6% | [-0.071, +0.012] | -3.8% | no |
| E2_soft_agg | prose | -0.014 | -0.9% | [-0.071, +0.044] | -4.5% | no |
| E2_soft_agg | **macro (pooled)** | +0.030 | +1.7% | [+0.006, +0.054] | +0.3% | no |

Flag: 'yes' = CI lower bound clears the reference line; '~' = point estimate clears but CI does not;
'no' = below. This is descriptive only (exploratory report; no accept/reject decision).
