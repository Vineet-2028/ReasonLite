# Application-ready project summary

Adapt the wording to your verified personal contribution and disclose AI assistance as required. ReasonLite is an empirical research project, not a peer-reviewed publication or a new base model.

> Implemented and ran a controlled small-LLM post-training study on a single Tesla T4 with completion-only LoRA SFT and two GRPO reward configurations. Compared four frozen Qwen2.5-0.5B-Instruct checkpoints across 5,276 predictions on the official GSM8K test set. SFT reduced generated tokens by 57.78% and improved correct answers per generated token by 1.82×, with an 8.95 percentage-point accuracy decrease. Recorded 512 RL rollouts, paired exact McNemar tests with Holm correction, sparse reward diagnostics, and recovery from a nonfinite training-log observation. The GRPO pilots did not significantly improve accuracy over SFT.

**Evidence for reviewers:** [README](../README.md), [technical report](RESEARCH_REPORT.md), [methodology](PROTOCOL.md), [raw results](../runs/final_v1/report/), [model card](MODEL_CARD.md), [limitations](ERROR_ANALYSIS.md). Use repository URL only after verifying reviewer access. Native Halo recipe is prepared, not run. Do not imply QLoRA or a hyperparameter sweep was performed.
