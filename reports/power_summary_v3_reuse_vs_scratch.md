# Power analysis (exploratory, DSpark-faithful) - ref=C1_scratch_v3, gamma=5
tau = accepted length incl. bonus token (DSpark Sec4.1 fn4); norm = tau/(gamma+1)=6.
Contrast = paired per-prompt Delta (variant-ref); CI = cluster-bootstrap-by-prompt (prompts resampled,
conditional on the fixed decoding seeds [0, 1]). 5% line is a REFERENCE, not a gate.

## Mean tau + latency per variant
| variant | domain | n | tau | norm | t_anchor | t_draft | t_verify | L_offline ms/tok |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| C1_scratch_v3 | math | 50 | 1.920 | 0.320 | 251.5 | 14.8 | 253.1 | 272.40 |
| C1_scratch_v3 | code | 50 | 1.871 | 0.312 | 202.7 | 14.7 | 211.3 | 230.88 |
| C1_scratch_v3 | chat | 50 | 1.824 | 0.304 | 195.1 | 14.6 | 204.2 | 228.79 |
| C1_scratch_v3 | prose | 50 | 1.703 | 0.284 | 256.5 | 14.3 | 258.6 | 323.02 |
| C1_scratch_v3 | **macro** | - | **1.829** | 0.305 | | | | |
| E2_soft_v3 | math | 50 | 2.042 | 0.340 | 248.5 | 17.1 | 249.9 | 254.51 |
| E2_soft_v3 | code | 50 | 1.975 | 0.329 | 201.7 | 16.9 | 210.3 | 219.27 |
| E2_soft_v3 | chat | 50 | 1.963 | 0.327 | 193.3 | 17.0 | 202.1 | 212.02 |
| E2_soft_v3 | prose | 50 | 1.726 | 0.288 | 254.5 | 16.7 | 256.4 | 316.10 |
| E2_soft_v3 | **macro** | - | **1.926** | 0.321 | | | | |

## Paired contrasts vs C1_scratch_v3 (Delta = variant - C1_scratch_v3)
| variant | domain | mean Delta | rel% | 95% CI (abs) | CI lower rel% | >ref-line? |
|---|---|--:|--:|:--:|--:|:--:|
| E2_soft_v3 | math | +0.122 | +6.4% | [+0.085, +0.160] | +4.4% | ~ |
| E2_soft_v3 | code | +0.104 | +5.6% | [+0.057, +0.151] | +3.1% | ~ |
| E2_soft_v3 | chat | +0.139 | +7.6% | [+0.105, +0.172] | +5.8% | yes |
| E2_soft_v3 | prose | +0.023 | +1.3% | [-0.056, +0.093] | -3.3% | no |
| E2_soft_v3 | **macro (pooled)** | +0.097 | +5.3% | [+0.070, +0.122] | +3.8% | ~ |

Flag: 'yes' = CI lower bound clears the reference line; '~' = point estimate clears but CI does not;
'no' = below. This is descriptive only (exploratory report; no accept/reject decision).
