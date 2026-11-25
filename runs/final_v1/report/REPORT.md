# ReasonLite: frozen held-out evaluation

All four pre-existing checkpoints evaluated on the same 1,319 official GSM8K test questions. E1 was selected as the efficiency candidate using development validation only. No further model selection or tuning was based on test results.

| Arm | Accuracy (Wilson 95% CI) | Mean tokens | Correct / 1k tokens | Format rate |
|---|---:|---:|---:|---:|
| E0 | 38.82% (36.22%–41.48%) | 282.54 | 1.374 | 0.00% |
| E1 | 29.87% (27.46%–32.40%) | 119.28 | 2.504 | 94.92% |
| E2 | 30.25% (27.83%–32.78%) | 127.15 | 2.379 | 93.63% |
| E3 | 30.25% (27.83%–32.78%) | 117.80 | 2.568 | 93.78% |

## Paired tests

| Contrast | Delta (pp) | Exact McNemar p | Holm-adjusted p |
|---|---:|---:|---:|
| E0 → E1 | -8.95 | 1.078e-08 | 6.469e-08 |
| E0 → E2 | -8.57 | 4.008e-08 | 1.603e-07 |
| E0 → E3 | -8.57 | 2.317e-08 | 1.158e-07 |
| E1 → E2 | +0.38 | 0.7364 | 1 |
| E1 → E3 | +0.38 | 0.7163 | 1 |
| E2 → E3 | +0.00 | 1 | 1 |

## Limitations

- Single training seed; confidence intervals and paired tests quantify question-level uncertainty, not training-seed variability.
- Scorer v2 was revised on development outputs; numeric answer correctness is not proof of correct reasoning. GSM8K may overlap pretraining.
- Batched greedy inference can differ from batch-one generation. Calibration results are recorded separately; batched throughput is not single-request latency.
- GRPO arms are small exploratory pilots; KL disabled; E3 had a nonfinite gradient with finite saved adapter weights. Length shaping used retokenized lengths and a format bonus.
- Halo native training is separate and not claimed completed.
