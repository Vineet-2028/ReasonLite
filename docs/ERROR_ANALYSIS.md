# Error analysis and limitations

## Development bugs and recovery

1. JSONL parsing: `splitlines()` treated embedded Unicode separators as record breaks. Changed reader to file iteration and added a regression case. Frozen bytes unchanged.
2. Baseline originally scored 0/128 due to unsupported final-answer formats. Rescored stored outputs with explicit terminal answer support: 69/128. Original baseline and rescoring provenance preserved. Parser remains conservative and may undercount valid answers; no manual semantic audit of every output.
3. E3 step 28 recorded `NaN` gradient. Training reached 64 steps, adapter and metadata saved, strict JSON serialization failed on the log. Recovered from checkpoint logs without retraining, preserved NaN as JSON null with sidecar and original text. Saved tensors finite and changed; this does not prove every optimizer update was applied. Release exporter now preserves nonfinite logs explicitly; this maintenance patch has offline tests only.

## Observed GRPO behavior

| Arm | All wrong groups | All correct groups | Variable reward groups | Correct rollouts | Duration (s) | Nonfinite gradient steps |
|---|---:|---:|---:|---:|---:|---|
| E2 | 35 | 3 | 26/64 | 65/256 | 747.61 | none recorded |
| E3 | 31 | 5 | 29/64 | 73/256 | 701.34 | 28 |

Zero within-group reward variance gives no relative reward contrast (with beta=0). Sparse informative groups, small budgets and disabled KL are plausible limitations, not established causes of the test outcomes. E3 includes both length penalty and format bonus. More correct training rollouts does not establish greater test accuracy.

## Paired held-out transitions

| Contrast | Wrong → right | Right → wrong | Δ accuracy (pp) | Holm p |
|---|---:|---:|---:|---:|
| E0 → E1 | 153 | 271 | -8.95 | 6.469e-08 |
| E0 → E2 | 154 | 267 | -8.57 | 1.603e-07 |
| E0 → E3 | 147 | 260 | -8.57 | 1.158e-07 |
| E1 → E2 | 73 | 68 | +0.38 | 1 |
| E1 → E3 | 63 | 58 | +0.38 | 1 |
| E2 → E3 | 63 | 63 | +0.00 | 1 |

E0 outperformed all adapted checkpoints in accuracy after Holm correction. Neither RL arm significantly differs from E1. Non-significance does not establish equivalence. E1 retains the development-based candidate designation. E3's equal observed accuracy to E2 does not mean identical answers (63 transitions each way).

## Generalization and measurement threats

- One seed, one model size, one benchmark; no population-level or universal conclusion.
- Development scorer revision was retrospective; final scoring rule frozen. Numeric match may miss or overcredit reasoning; parser false negatives remain possible.
- Test questions existed in frozen archive but no test results used for tuning; potential benchmark contamination in pretraining unknown.
- Batch-size change affects outputs: calibration text agreement E0 15/16, E1 14/16, E2 15/16, E3 14/16; correctness agreement 16/16 except E3 15/16. Do not attribute validation-to-test changes exclusively to data shift.
- Adapted models have lower final batched questions/s despite fewer tokens. No claims about deployment latency or cost savings from output length alone.
- No hyperparameter sweep, QLoRA comparison, or independent seeds; no prompt-only concise baseline. No native Halo execution yet.

## Future experiments (separate from frozen study)

Predefine validation-only learning-rate/reward/budget comparisons, add concise-prompt control, independently repeat training seeds, isolate format and length rewards, test a second model and benchmark, verify exact sampled lengths, monitor mixed-precision overflow and compare KL settings. If optimizing after seeing this test, use a new untouched evaluation set for confirmatory claims. These are research extensions, not required to share the present pilot.
