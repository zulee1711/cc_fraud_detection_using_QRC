# Equal-budget tuning at L = 15: TFIM (2 Trotter steps, XYZ-strings) vs ESN-45

Protocol in tuned.py. References (untuned, logistic, LSTM) from results/2_fraud at L = 15.

## Tuning (validation customer AP, mean over seeds 42-44)

| model | trials | untuned val AP | best val AP | best trial | best setting | at a bound |
|---|---|---|---|---|---|---|
| TFIM | 100 | 0.4152 | 0.4911 | 80 | h 0.0822, time 0.523, encoding_scaling 2.91, alpha 3.16 | none |
| ESN-45 | 100 | 0.3747 | 0.5901 | 44 | spectral_radius 0.825, leak_rate 0.256, input_scaling 2.16, alpha 0.000154 | alpha |

## Test, mean ± std over seeds 42-51

| model | customer AP | amount AP | overall AP | CP@100 | val customer AP |
|---|---|---|---|---|---|
| TFIM tuned | 0.485 ± 0.005 | 0.989 ± 0.001 | 0.711 ± 0.001 | 0.072 ± 0.000 | 0.493 ± 0.005 |
| TFIM tuned, 10⁵ shots | 0.474 ± 0.009 | 0.989 ± 0.002 | 0.708 ± 0.002 | 0.072 ± 0.000 | 0.486 ± 0.007 |
| TFIM tuned, 10⁴ shots | 0.447 ± 0.018 | 0.985 ± 0.002 | 0.700 ± 0.005 | 0.073 ± 0.001 | 0.456 ± 0.020 |
| ESN-45 tuned | 0.587 ± 0.010 | 0.979 ± 0.004 | 0.745 ± 0.003 | 0.072 ± 0.000 | 0.586 ± 0.010 |
| TFIM untuned | 0.429 ± 0.011 | 0.989 ± 0.002 | 0.703 ± 0.003 | 0.073 ± 0.001 | 0.420 ± 0.015 |
| TFIM untuned, 10⁵ shots | 0.359 ± 0.073 | 0.987 ± 0.001 | 0.681 ± 0.023 | 0.072 ± 0.001 | 0.350 ± 0.074 |
| ESN-45 untuned | 0.398 ± 0.019 | 0.989 ± 0.003 | 0.691 ± 0.005 | 0.072 ± 0.001 | 0.389 ± 0.019 |
| LSTM | 0.771 ± 0.004 | 0.997 ± 0.001 | 0.795 ± 0.001 | 0.070 ± 0.001 | 0.784 ± 0.002 |
| logistic (raw window) | 0.406 | 0.959 | 0.663 | 0.072 | 0.409 |

## Test, held-out seeds 45-51

| model | customer AP | amount AP | overall AP | CP@100 | val customer AP |
|---|---|---|---|---|---|
| TFIM tuned | 0.486 ± 0.004 | 0.989 ± 0.001 | 0.711 ± 0.001 | 0.072 ± 0.000 | 0.494 ± 0.003 |
| TFIM tuned, 10⁵ shots | 0.473 ± 0.010 | 0.989 ± 0.002 | 0.708 ± 0.003 | 0.072 ± 0.000 | 0.485 ± 0.007 |
| TFIM tuned, 10⁴ shots | 0.445 ± 0.021 | 0.986 ± 0.002 | 0.700 ± 0.006 | 0.072 ± 0.001 | 0.453 ± 0.023 |
| ESN-45 tuned | 0.586 ± 0.011 | 0.978 ± 0.005 | 0.745 ± 0.003 | 0.072 ± 0.000 | 0.585 ± 0.011 |
| TFIM untuned | 0.431 ± 0.012 | 0.989 ± 0.002 | 0.703 ± 0.003 | 0.073 ± 0.001 | 0.422 ± 0.015 |
| TFIM untuned, 10⁵ shots | 0.391 ± 0.034 | 0.987 ± 0.001 | 0.691 ± 0.010 | 0.072 ± 0.000 | 0.381 ± 0.041 |
| ESN-45 untuned | 0.406 ± 0.012 | 0.988 ± 0.004 | 0.693 ± 0.005 | 0.072 ± 0.001 | 0.395 ± 0.014 |
| LSTM | 0.769 ± 0.004 | 0.997 ± 0.001 | 0.795 ± 0.001 | 0.070 ± 0.001 | 0.784 ± 0.002 |
| logistic (raw window) | 0.406 | 0.959 | 0.663 | 0.072 | 0.409 |

## Paired differences in test customer AP, seeds 42-51

| a − b | mean diff | bootstrap 95% CI | paired t p | Wilcoxon p | a wins |
|---|---|---|---|---|---|
| TFIM tuned − ESN-45 tuned | -0.102 | [-0.108, -0.096] | 1.9e-10 | 0.002 | 0/10 |
| TFIM tuned, 10⁵ shots − ESN-45 tuned | -0.113 | [-0.119, -0.107] | 6.4e-11 | 0.002 | 0/10 |
| TFIM tuned − TFIM untuned | +0.056 | [+0.050, +0.062] | 6e-08 | 0.002 | 10/10 |
| ESN-45 tuned − ESN-45 untuned | +0.189 | [+0.175, +0.205] | 2.5e-09 | 0.002 | 10/10 |
| TFIM untuned − ESN-45 untuned | +0.031 | [+0.017, +0.046] | 0.003 | 0.0039 | 9/10 |
| TFIM tuned − logistic | +0.079 | | 1e-12 (1-sample) | | 10/10 |
| ESN-45 tuned − logistic | +0.181 | | 9.3e-13 (1-sample) | | 10/10 |

## Paired differences in test customer AP, held-out seeds 45-51

| a − b | mean diff | bootstrap 95% CI | paired t p | Wilcoxon p | a wins |
|---|---|---|---|---|---|
| TFIM tuned − ESN-45 tuned | -0.100 | [-0.107, -0.092] | 3.4e-07 | 0.016 | 0/7 |
| TFIM tuned, 10⁵ shots − ESN-45 tuned | -0.113 | [-0.121, -0.104] | 3.1e-07 | 0.016 | 0/7 |
| TFIM tuned − TFIM untuned | +0.055 | [+0.047, +0.065] | 2.8e-05 | 0.016 | 7/7 |
| ESN-45 tuned − ESN-45 untuned | +0.180 | [+0.168, +0.193] | 2.4e-07 | 0.016 | 7/7 |
| TFIM untuned − ESN-45 untuned | +0.025 | [+0.010, +0.040] | 0.025 | 0.031 | 6/7 |
| TFIM tuned − logistic | +0.081 | | 3.8e-09 (1-sample) | | 7/7 |
| ESN-45 tuned − logistic | +0.180 | | 1.1e-08 (1-sample) | | 7/7 |

Operational metrics at the report's operating threshold: operational.py -> operational.md.
