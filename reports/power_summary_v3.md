# Power analysis (exploratory, DSpark-faithful) - ref=B3_dense_v3, gamma=5
tau = accepted length incl. bonus token (DSpark Sec4.1 fn4); norm = tau/(gamma+1)=6.
Contrast = paired per-prompt Delta (variant-ref); CI = cluster-bootstrap-by-prompt (prompts resampled,
conditional on the fixed decoding seeds [0, 1]). 5% line is a REFERENCE, not a gate.

## Mean tau + latency per variant
| variant | domain | n | tau | norm | t_anchor | t_draft | t_verify | L_offline ms/tok |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| B3_dense_v3 | math | 50 | 1.898 | 0.316 | 247.5 | 6.0 | 248.9 | 266.55 |
| B3_dense_v3 | code | 50 | 1.839 | 0.307 | 199.6 | 6.0 | 207.8 | 226.40 |
| B3_dense_v3 | chat | 50 | 1.874 | 0.312 | 191.8 | 5.9 | 200.6 | 215.26 |
| B3_dense_v3 | prose | 50 | 1.769 | 0.295 | 253.4 | 6.0 | 255.3 | 302.79 |
| B3_dense_v3 | **macro** | - | **1.845** | 0.308 | | | | |
| E1_hard_v3 | math | 50 | 1.878 | 0.313 | 250.0 | 12.8 | 251.6 | 276.05 |
| E1_hard_v3 | code | 50 | 1.839 | 0.307 | 202.9 | 12.7 | 211.1 | 234.04 |
| E1_hard_v3 | chat | 50 | 1.810 | 0.302 | 195.4 | 12.7 | 203.9 | 229.67 |
| E1_hard_v3 | prose | 50 | 1.632 | 0.272 | 255.2 | 12.7 | 257.2 | 332.80 |
| E1_hard_v3 | **macro** | - | **1.790** | 0.298 | | | | |
| E2_soft_v3 | math | 50 | 2.042 | 0.340 | 248.5 | 17.1 | 249.9 | 254.51 |
| E2_soft_v3 | code | 50 | 1.975 | 0.329 | 201.7 | 16.9 | 210.3 | 219.27 |
| E2_soft_v3 | chat | 50 | 1.963 | 0.327 | 193.3 | 17.0 | 202.1 | 212.02 |
| E2_soft_v3 | prose | 50 | 1.726 | 0.288 | 254.5 | 16.7 | 256.4 | 316.10 |
| E2_soft_v3 | **macro** | - | **1.926** | 0.321 | | | | |
| C1_scratch_v3 | math | 50 | 1.920 | 0.320 | 251.5 | 14.8 | 253.1 | 272.40 |
| C1_scratch_v3 | code | 50 | 1.871 | 0.312 | 202.7 | 14.7 | 211.3 | 230.88 |
| C1_scratch_v3 | chat | 50 | 1.824 | 0.304 | 195.1 | 14.6 | 204.2 | 228.79 |
| C1_scratch_v3 | prose | 50 | 1.703 | 0.284 | 256.5 | 14.3 | 258.6 | 323.02 |
| C1_scratch_v3 | **macro** | - | **1.829** | 0.305 | | | | |

## Paired contrasts vs B3_dense_v3 (Delta = variant - B3_dense_v3)
| variant | domain | mean Delta | rel% | 95% CI (abs) | CI lower rel% | >ref-line? |
|---|---|--:|--:|:--:|--:|:--:|
| E1_hard_v3 | math | -0.020 | -1.0% | [-0.055, +0.017] | -2.9% | no |
| E1_hard_v3 | code | -0.000 | -0.0% | [-0.034, +0.034] | -1.8% | no |
| E1_hard_v3 | chat | -0.064 | -3.4% | [-0.100, -0.029] | -5.3% | no |
| E1_hard_v3 | prose | -0.137 | -7.7% | [-0.199, -0.076] | -11.3% | no |
| E1_hard_v3 | **macro (pooled)** | -0.055 | -3.0% | [-0.079, -0.032] | -4.3% | no |
| E2_soft_v3 | math | +0.144 | +7.6% | [+0.106, +0.182] | +5.6% | yes |
| E2_soft_v3 | code | +0.136 | +7.4% | [+0.093, +0.180] | +5.0% | yes |
| E2_soft_v3 | chat | +0.089 | +4.8% | [+0.052, +0.127] | +2.8% | no |
| E2_soft_v3 | prose | -0.043 | -2.4% | [-0.115, +0.027] | -6.5% | no |
| E2_soft_v3 | **macro (pooled)** | +0.081 | +4.4% | [+0.055, +0.108] | +3.0% | no |
| C1_scratch_v3 | math | +0.022 | +1.1% | [-0.005, +0.049] | -0.3% | no |
| C1_scratch_v3 | code | +0.031 | +1.7% | [-0.000, +0.063] | -0.0% | no |
| C1_scratch_v3 | chat | -0.050 | -2.6% | [-0.087, -0.015] | -4.6% | no |
| C1_scratch_v3 | prose | -0.066 | -3.7% | [-0.138, +0.005] | -7.8% | no |
| C1_scratch_v3 | **macro (pooled)** | -0.016 | -0.8% | [-0.039, +0.008] | -2.1% | no |

Flag: 'yes' = CI lower bound clears the reference line; '~' = point estimate clears but CI does not;
'no' = below. This is descriptive only (exploratory report; no accept/reject decision).
