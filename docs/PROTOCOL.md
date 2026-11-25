# Experimental protocol (executed study)

## Questions and arms

Can low-compute SFT improve output efficiency? Do either of two small GRPO pilots improve over that SFT checkpoint? E0 = pretrained Qwen instruct; E1 = LoRA SFT; E2 = E1 plus binary correctness reward; E3 = E1 plus correctness-gated length penalty and format bonus. Both RL arms begin from identical E1 adapter bytes. E1 was selected on development validation, not on final test outcomes. No model or reward was selected or retuned on the final test.

## Frozen data and environment

Model revision `7ae557604adf67be50417f59c2c2f167def9a775`; dataset `openai/gsm8k` main configuration revision `740312add88f781978c0658806c59bc2815b9866`. Python random seed 42 determines disjoint training-row IDs: 512 SFT, 256 RL, 128 validation, all 1,319 official test examples. GRPO uses the first 64 frozen RL examples in each arm. The manifest and protocol lock are the authoritative byte hashes. Test gold was included in frozen data for integrity but not fed into training; this is a procedural holdout, not cryptographically inaccessible to researchers. Potential pretraining overlap is unknown.

GPU Tesla T4; recorded evaluation stack Python 3.13.16, PyTorch 2.11.0+cu130, Transformers 4.51.3, PEFT 0.15.2; GRPO uses TRL 0.17.0. SFT metadata reports `bf16` on T4 (possibly emulated rather than native hardware support; do not claim native BF16 acceleration). The recorded environment is a snapshot, not a fully resolved transitive lock.

## SFT configuration

Completion-only masking; max length 768, LoRA r=8, alpha=16, dropout=0.05, q/k/v/o projections; learning rate 2e-4, one epoch, batch 2, gradient accumulation 8. All 512 examples retained, 32 nominal updates. SFT took 186.43 s, loss 0.49364, peak allocated 2716.85 MiB. Trainable 1,081,344 of 495,114,112 parameters. No 4-bit quantization; this is LoRA, not QLoRA.

## GRPO configuration

64 groups × four completions per arm, batch 4, accumulation 1, one epoch, LR 2e-5 constant, temperature 0.8, max prompt 512 and completion 256, clipping epsilon 0.2, `loss_type=bnpo`, scaled rewards, one iteration, beta=0 (no KL regularization), FP16 AMP, maximum gradient norm 1, gradient checkpointing. E2 correctness reward = 1 if exact parsed answer matches, otherwise 0. E3 for a correct answer = 1 − 0.15 × min(1, max(0,L−128)/128) + 0.05 × I(strict format); wrong answers = 0. L is decoded text re-tokenization, not exact sampled length. Thus E3 changes two reward terms simultaneously, not an isolated length ablation. The rollout scorer suppresses fallback parsing near the length cap. See stored configurations for full details.

## Evaluation and statistical analysis

Prompt asks for reasoning and final `#### <number>`. Greedy generation, maximum 384 new tokens. Validation used batch 1; final test batch 16, left padding, count through first EOS. Calibration used the same first 16 validation items for all models. Conservative numeric parser v2 was revised after inspecting development outputs (LaTeX boxed and explicit terminal prose); freeze before test. Numeric correctness is not verified reasoning. Strict format compliance is independent of correctness. Invalid means parser could not extract a supported answer. Correct / 1k tokens = 1000 × correct / total generated tokens. Time is synchronized batch-generation wall time, not end-to-end latency; questions/s is not comparable with batch-one latency.

Accuracy intervals: Wilson 95%; paired exact two-sided McNemar conditional binomial test on correctness transitions; Holm adjustment across six pairwise comparisons. One training seed, so these tests quantify item-level effects for the fixed checkpoints, not algorithmic superiority across training runs. No causal claims about individual hyperparameters.

Authoritative locks: [final protocol](../runs/final_protocol_v1.json), [manifest](../data/prepared/manifest.json), [raw report](../runs/final_v1/report/REPORT.md). Historical planning protocol is archived separately.
