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
| E1_hard | math | 50 | 1.868 | 0.311 | 306.5 | 13.4 | 315.9 | 342.24 |
| E1_hard | code | 50 | 1.845 | 0.308 | 239.6 | 13.4 | 251.3 | 275.14 |
| E1_hard | chat | 50 | 1.837 | 0.306 | 227.1 | 13.2 | 239.7 | 263.20 |
| E1_hard | prose | 50 | 1.535 | 0.256 | 334.4 | 13.2 | 338.4 | 453.88 |
| E1_hard | **macro** | - | **1.771** | 0.295 | | | | |
| E2_soft | math | 50 | 1.897 | 0.316 | 307.6 | 15.9 | 316.2 | 339.36 |
| E2_soft | code | 50 | 1.854 | 0.309 | 240.8 | 15.9 | 252.2 | 276.59 |
| E2_soft | chat | 50 | 1.825 | 0.304 | 230.0 | 16.0 | 241.5 | 268.96 |
| E2_soft | prose | 50 | 1.591 | 0.265 | 334.8 | 15.4 | 338.7 | 440.89 |
| E2_soft | **macro** | - | **1.792** | 0.299 | | | | |
| C1_scratch | math | 50 | 1.875 | 0.313 | 306.4 | 14.4 | 315.5 | 341.96 |
| C1_scratch | code | 50 | 1.848 | 0.308 | 238.9 | 14.1 | 250.6 | 275.30 |
| C1_scratch | chat | 50 | 1.820 | 0.303 | 227.9 | 14.2 | 239.9 | 266.90 |
| C1_scratch | prose | 50 | 1.558 | 0.260 | 333.5 | 14.0 | 337.4 | 447.30 |
| C1_scratch | **macro** | - | **1.775** | 0.296 | | | | |

## Paired contrasts vs B3_dense (Delta = variant - B3_dense)
| variant | domain | mean Delta | rel% | 95% CI (abs) | CI lower rel% | >ref-line? |
|---|---|--:|--:|:--:|--:|:--:|
| E1_hard | math | +0.048 | +2.6% | [+0.013, +0.084] | +0.7% | no |
| E1_hard | code | +0.086 | +4.9% | [+0.043, +0.128] | +2.4% | no |
| E1_hard | chat | -0.007 | -0.4% | [-0.048, +0.034] | -2.6% | no |
| E1_hard | prose | -0.044 | -2.8% | [-0.099, +0.012] | -6.3% | no |
| E1_hard | **macro (pooled)** | +0.021 | +1.2% | [-0.003, +0.044] | -0.1% | no |
| E2_soft | math | +0.076 | +4.2% | [+0.038, +0.119] | +2.1% | no |
| E2_soft | code | +0.095 | +5.4% | [+0.059, +0.127] | +3.4% | ~ |
| E2_soft | chat | -0.019 | -1.1% | [-0.054, +0.015] | -2.9% | no |
| E2_soft | prose | +0.012 | +0.8% | [-0.040, +0.066] | -2.5% | no |
| E2_soft | **macro (pooled)** | +0.041 | +2.3% | [+0.019, +0.062] | +1.1% | no |
| C1_scratch | math | +0.055 | +3.0% | [+0.014, +0.097] | +0.8% | no |
| C1_scratch | code | +0.089 | +5.0% | [+0.052, +0.125] | +2.9% | ~ |
| C1_scratch | chat | -0.025 | -1.3% | [-0.068, +0.019] | -3.7% | no |
| C1_scratch | prose | -0.020 | -1.3% | [-0.089, +0.044] | -5.6% | no |
| C1_scratch | **macro (pooled)** | +0.025 | +1.4% | [-0.001, +0.049] | -0.0% | no |

Flag: 'yes' = CI lower bound clears the reference line; '~' = point estimate clears but CI does not;
'no' = below. This is descriptive only (exploratory report; no accept/reject decision).
