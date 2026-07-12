# Power analysis (exploratory, DSpark-faithful) - ref=B3_dense_v2, gamma=5
tau = accepted length incl. bonus token (DSpark Sec4.1 fn4); norm = tau/(gamma+1)=6.
Contrast = paired per-prompt Delta (variant-ref); CI = cluster-bootstrap-by-prompt (prompts resampled,
conditional on the fixed decoding seeds [0, 1]). 5% line is a REFERENCE, not a gate.

## Mean tau + latency per variant
| variant | domain | n | tau | norm | t_anchor | t_draft | t_verify | L_offline ms/tok |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| B3_dense_v2 | math | 50 | 1.876 | 0.313 | 303.8 | 9.9 | 313.1 | 335.84 |
| B3_dense_v2 | code | 50 | 1.870 | 0.312 | 234.8 | 9.8 | 246.9 | 264.50 |
| B3_dense_v2 | chat | 50 | 1.805 | 0.301 | 226.6 | 9.8 | 239.1 | 265.37 |
| B3_dense_v2 | prose | 50 | 1.639 | 0.273 | 339.1 | 9.9 | 342.7 | 433.18 |
| B3_dense_v2 | **macro** | - | **1.797** | 0.300 | | | | |
| E1_hard_v2 | math | 50 | 1.875 | 0.312 | 304.5 | 13.3 | 313.6 | 339.43 |
| E1_hard_v2 | code | 50 | 1.953 | 0.326 | 235.1 | 13.2 | 246.9 | 255.53 |
| E1_hard_v2 | chat | 50 | 1.812 | 0.302 | 228.6 | 13.4 | 240.0 | 269.36 |
| E1_hard_v2 | prose | 50 | 1.639 | 0.273 | 340.8 | 13.3 | 344.5 | 436.54 |
| E1_hard_v2 | **macro** | - | **1.820** | 0.303 | | | | |
| E2_soft_long_v2 | math | 50 | 1.932 | 0.322 | 304.6 | 16.0 | 313.9 | 330.05 |
| E2_soft_long_v2 | code | 50 | 1.902 | 0.317 | 234.8 | 15.7 | 246.4 | 263.09 |
| E2_soft_long_v2 | chat | 50 | 1.816 | 0.303 | 227.4 | 15.8 | 239.6 | 269.10 |
| E2_soft_long_v2 | prose | 50 | 1.681 | 0.280 | 338.9 | 15.6 | 342.6 | 428.26 |
| E2_soft_long_v2 | **macro** | - | **1.833** | 0.305 | | | | |
| C1_scratch_long_v2 | math | 50 | 1.912 | 0.319 | 304.4 | 14.3 | 314.1 | 333.29 |
| C1_scratch_long_v2 | code | 50 | 1.895 | 0.316 | 236.2 | 14.3 | 248.0 | 265.40 |
| C1_scratch_long_v2 | chat | 50 | 1.781 | 0.297 | 228.2 | 14.3 | 240.8 | 274.70 |
| C1_scratch_long_v2 | prose | 50 | 1.703 | 0.284 | 341.0 | 14.2 | 344.7 | 423.24 |
| C1_scratch_long_v2 | **macro** | - | **1.823** | 0.304 | | | | |

## Paired contrasts vs B3_dense_v2 (Delta = variant - B3_dense_v2)
| variant | domain | mean Delta | rel% | 95% CI (abs) | CI lower rel% | >ref-line? |
|---|---|--:|--:|:--:|--:|:--:|
| E1_hard_v2 | math | -0.001 | -0.1% | [-0.042, +0.040] | -2.3% | no |
| E1_hard_v2 | code | +0.083 | +4.4% | [+0.048, +0.120] | +2.6% | no |
| E1_hard_v2 | chat | +0.007 | +0.4% | [-0.025, +0.039] | -1.4% | no |
| E1_hard_v2 | prose | +0.000 | +0.0% | [-0.059, +0.059] | -3.6% | no |
| E1_hard_v2 | **macro (pooled)** | +0.022 | +1.2% | [-0.000, +0.044] | -0.0% | no |
| E2_soft_long_v2 | math | +0.056 | +3.0% | [+0.021, +0.091] | +1.1% | no |
| E2_soft_long_v2 | code | +0.032 | +1.7% | [-0.007, +0.070] | -0.3% | no |
| E2_soft_long_v2 | chat | +0.011 | +0.6% | [-0.021, +0.044] | -1.2% | no |
| E2_soft_long_v2 | prose | +0.042 | +2.6% | [-0.015, +0.103] | -0.9% | no |
| E2_soft_long_v2 | **macro (pooled)** | +0.035 | +2.0% | [+0.014, +0.057] | +0.8% | no |
| C1_scratch_long_v2 | math | +0.036 | +1.9% | [-0.004, +0.075] | -0.2% | no |
| C1_scratch_long_v2 | code | +0.025 | +1.3% | [-0.006, +0.057] | -0.3% | no |
| C1_scratch_long_v2 | chat | -0.024 | -1.3% | [-0.056, +0.008] | -3.1% | no |
| C1_scratch_long_v2 | prose | +0.065 | +3.9% | [+0.006, +0.127] | +0.4% | no |
| C1_scratch_long_v2 | **macro (pooled)** | +0.026 | +1.4% | [+0.004, +0.048] | +0.2% | no |

Flag: 'yes' = CI lower bound clears the reference line; '~' = point estimate clears but CI does not;
'no' = below. This is descriptive only (exploratory report; no accept/reject decision).
