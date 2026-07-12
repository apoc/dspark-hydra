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
| E2_soft_v2 | math | 50 | 1.844 | 0.307 | 303.7 | 15.6 | 312.7 | 344.93 |
| E2_soft_v2 | code | 50 | 1.784 | 0.297 | 235.4 | 15.5 | 246.9 | 282.01 |
| E2_soft_v2 | chat | 50 | 1.758 | 0.293 | 227.1 | 15.6 | 239.0 | 277.07 |
| E2_soft_v2 | prose | 50 | 1.576 | 0.263 | 340.0 | 15.7 | 343.6 | 454.30 |
| E2_soft_v2 | **macro** | - | **1.740** | 0.290 | | | | |
| C1_scratch_v2 | math | 50 | 1.795 | 0.299 | 304.2 | 13.7 | 313.0 | 353.72 |
| C1_scratch_v2 | code | 50 | 1.725 | 0.287 | 235.9 | 13.9 | 247.0 | 289.63 |
| C1_scratch_v2 | chat | 50 | 1.732 | 0.289 | 228.0 | 13.9 | 239.5 | 280.78 |
| C1_scratch_v2 | prose | 50 | 1.611 | 0.269 | 339.7 | 13.8 | 343.0 | 441.75 |
| C1_scratch_v2 | **macro** | - | **1.716** | 0.286 | | | | |

## Paired contrasts vs B3_dense_v2 (Delta = variant - B3_dense_v2)
| variant | domain | mean Delta | rel% | 95% CI (abs) | CI lower rel% | >ref-line? |
|---|---|--:|--:|:--:|--:|:--:|
| E1_hard_v2 | math | -0.001 | -0.1% | [-0.042, +0.040] | -2.3% | no |
| E1_hard_v2 | code | +0.083 | +4.4% | [+0.048, +0.120] | +2.6% | no |
| E1_hard_v2 | chat | +0.007 | +0.4% | [-0.025, +0.039] | -1.4% | no |
| E1_hard_v2 | prose | +0.000 | +0.0% | [-0.059, +0.059] | -3.6% | no |
| E1_hard_v2 | **macro (pooled)** | +0.022 | +1.2% | [-0.000, +0.044] | -0.0% | no |
| E2_soft_v2 | math | -0.032 | -1.7% | [-0.066, +0.001] | -3.5% | no |
| E2_soft_v2 | code | -0.087 | -4.6% | [-0.120, -0.053] | -6.4% | no |
| E2_soft_v2 | chat | -0.047 | -2.6% | [-0.079, -0.015] | -4.4% | no |
| E2_soft_v2 | prose | -0.062 | -3.8% | [-0.120, -0.004] | -7.3% | no |
| E2_soft_v2 | **macro (pooled)** | -0.057 | -3.2% | [-0.078, -0.036] | -4.3% | no |
| C1_scratch_v2 | math | -0.081 | -4.3% | [-0.117, -0.046] | -6.2% | no |
| C1_scratch_v2 | code | -0.146 | -7.8% | [-0.183, -0.108] | -9.8% | no |
| C1_scratch_v2 | chat | -0.073 | -4.1% | [-0.102, -0.043] | -5.6% | no |
| C1_scratch_v2 | prose | -0.027 | -1.7% | [-0.086, +0.032] | -5.2% | no |
| C1_scratch_v2 | **macro (pooled)** | -0.082 | -4.5% | [-0.103, -0.060] | -5.8% | no |

Flag: 'yes' = CI lower bound clears the reference line; '~' = point estimate clears but CI does not;
'no' = below. This is descriptive only (exploratory report; no accept/reject decision).
