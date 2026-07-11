# Power analysis (exploratory, DSpark-faithful) - ref=E1_hard, gamma=5
tau = accepted length incl. bonus token (DSpark Sec4.1 fn4); norm = tau/(gamma+1)=6.
Contrast = paired per-prompt Delta (variant-ref); CI = cluster-bootstrap-by-prompt (prompts resampled,
conditional on the fixed decoding seeds [0, 1]). 5% line is a REFERENCE, not a gate.

## Mean tau + latency per variant
| variant | domain | n | tau | norm | t_anchor | t_draft | t_verify | L_offline ms/tok |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| E1_hard | math | 50 | 1.868 | 0.311 | 306.5 | 13.4 | 315.9 | 342.24 |
| E1_hard | code | 50 | 1.845 | 0.308 | 239.6 | 13.4 | 251.3 | 275.14 |
| E1_hard | chat | 50 | 1.837 | 0.306 | 227.1 | 13.2 | 239.7 | 263.20 |
| E1_hard | prose | 50 | 1.535 | 0.256 | 334.4 | 13.2 | 338.4 | 453.88 |
| E1_hard | **macro** | - | **1.771** | 0.295 | | | | |
| E1_hard_agg | math | 50 | 1.942 | 0.324 | 306.0 | 13.1 | 315.1 | 328.88 |
| E1_hard_agg | code | 50 | 1.829 | 0.305 | 239.8 | 13.0 | 251.7 | 277.94 |
| E1_hard_agg | chat | 50 | 1.827 | 0.304 | 228.3 | 13.0 | 240.7 | 265.89 |
| E1_hard_agg | prose | 50 | 1.517 | 0.253 | 335.2 | 13.3 | 339.2 | 461.60 |
| E1_hard_agg | **macro** | - | **1.779** | 0.296 | | | | |

## Paired contrasts vs E1_hard (Delta = variant - E1_hard)
| variant | domain | mean Delta | rel% | 95% CI (abs) | CI lower rel% | >ref-line? |
|---|---|--:|--:|:--:|--:|:--:|
| E1_hard_agg | math | +0.074 | +4.0% | [+0.033, +0.118] | +1.8% | no |
| E1_hard_agg | code | -0.016 | -0.9% | [-0.052, +0.021] | -2.8% | no |
| E1_hard_agg | chat | -0.011 | -0.6% | [-0.043, +0.019] | -2.3% | no |
| E1_hard_agg | prose | -0.018 | -1.2% | [-0.074, +0.037] | -4.8% | no |
| E1_hard_agg | **macro (pooled)** | +0.007 | +0.4% | [-0.015, +0.029] | -0.9% | no |

Flag: 'yes' = CI lower bound clears the reference line; '~' = point estimate clears but CI does not;
'no' = below. This is descriptive only (exploratory report; no accept/reject decision).
