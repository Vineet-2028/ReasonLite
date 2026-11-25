# Third-party sources and sharing review

This package contains upstream benchmark text, tokenizer assets and adapters to a pretrained model. Keep it private pending your rights review. No license is assigned to the entire ReasonLite repository. Do not assume a repo-wide license overrides upstream terms.

- Qwen2.5-0.5B-Instruct: [model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [pinned revision license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE). Model card lists Apache-2.0. Tokenizer assets and adapters are provided with attribution; confirm and include applicable original license/notice texts before external redistribution.
- GSM8K: [dataset](https://huggingface.co/datasets/openai/gsm8k), [original upstream repository and license](https://github.com/openai/grade-school-math), [paper](https://arxiv.org/abs/2110.14168). Frozen JSONL files contain selected questions/solutions and transformed gold fields; confirm applicable license and include required notices before redistribution. Predictions can reproduce benchmark content.
- [TRL](https://github.com/huggingface/trl/tree/v0.17.0), [PEFT](https://github.com/huggingface/peft), [Transformers](https://github.com/huggingface/transformers), [PyTorch](https://github.com/pytorch/pytorch): installed dependencies, not vendored full packages.
- [WhiteCircle Halo](https://github.com/whitecircle/halo): separate recipe only; follow upstream license and supported environment. No claim that ReasonLite uses Halo-native training in the completed experiments.

Before public release verify all upstream notices and collaboration permissions, decide a license for original code, scan for secrets and private information, then choose what to publish. Private storage itself is not a blanket authorization to redistribute. Do not share the full package with reviewers until permitted. Where access is restricted, share approved summaries or obtain permissions first.
