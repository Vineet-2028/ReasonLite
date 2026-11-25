# ReasonLite — efficient post-training for mathematical reasoning

A controlled small-language-model research project. Question: does parameter-efficient supervised training, followed by correctness-first reward optimization, improve mathematical answer accuracy and token efficiency under a small GPU budget?

**Status:** v0.1 foundation. Offline tests passed during preparation; actual Qwen baseline, SFT, GRPO, Halo training, and model publication are not yet measured or completed. This is not a new foundation model or a claim of research novelty.

## Experimental plan

| ID | Treatment | Status |
|---|---|---|
| E0 | Qwen2.5-0.5B-Instruct, unchanged | Runnable baseline; not executed |
| E1 | Completion-only SFT, LoRA rank 8 | Training entry point; GPU verification pending |
| E2 | E1 + correctness-only GRPO | Planned |
| E3 | E1 + correctness-first, efficiency-shaped GRPO | Planned |

E2 and E3 must start from the **same** E1 checkpoint, use the same RL rows, generation budget, update count and seeds. The optional reward implementation is tested independently; no RL training is claimed. See `docs/PROTOCOL.md`.

## Run on a GPU

Use `ReasonLite_Colab.ipynb` (in this archive or separately downloaded). Upload this source archive to the notebook's initial upload cell. Select a GPU runtime before running. Kaggle alternative: enable internet and a GPU, upload this archive as a private dataset and extract it to `/kaggle/working`, then use the same CLI commands from the project root. Free compute is not guaranteed.

Python 3.10–3.12 and CUDA-enabled PyTorch >=2.6 recommended. A single T4 16 GB is the initial target, not a measured requirement. Versions in `requirements.txt` are a fixed API-compatible candidate stack, not a complete transitive lock or security audit. Use an isolated notebook/runtime and check package compatibility before training. Never load untrusted checkpoints.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m reasonlite.data --include-test
python -m reasonlite.evaluate --device cuda --limit 16 --out runs/e0_smoke
python -m reasonlite.evaluate --device cuda --limit 128 --out runs/e0_validation
```

Run only the smoke and full baseline today. The official test is stored for reproducibility but must not be inspected or evaluated until the protocol is frozen. The manifest pins Hub revisions, selected IDs are reproducible, and split files are checked by SHA-256. Do not rerun preparation over the same directory. Use a new directory for a changed protocol.

After E0 succeeds, run a two-update smoke SFT in a separate output directory, then the real training recipe:

```bash
CUDA_VISIBLE_DEVICES=0 python -m reasonlite.train_sft --out runs/sft_smoke --max-steps 2
CUDA_VISIBLE_DEVICES=0 python -m reasonlite.train_sft --out runs/sft
CUDA_VISIBLE_DEVICES=0 python -m reasonlite.evaluate --device cuda --adapter runs/sft/adapter --limit 128 --out runs/e1_validation
python -m reasonlite.compare runs/e0_validation runs/e1_validation --out runs/e0_vs_e1.json
```

Use the same GPU for E0/E1, including the same visible device. Evaluation is single-example greedy generation; timing includes generation and device synchronization, excludes loading and tokenization; peak allocated GPU memory includes model weights. Output token counts include generated stop tokens. A 384-token cap is the initial setting. If a smoke run reveals excessive truncation or a bug, correct the protocol before full validation and apply the new settings to all arms.

## Evidence and limitations

No accuracy, speedup, VRAM or training claims until the actual GPU results exist. Record the data manifest, metrics, predictions, training logs, package versions, adapter hash and commit. The dataset is public and may have appeared in the pretrained model's training corpus. Disjoint row IDs protect against direct experimental split reuse, not semantic duplicates or pretraining contamination. Numeric scoring does not validate reasoning steps or calibration. The synthetic tests verify the software, not mathematical ability.

Do not commit raw datasets, weight binaries or credentials. Review base model and dataset licenses and contributor ownership before publication. Third-party sources: [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [TRL SFT 0.17](https://huggingface.co/docs/trl/v0.17.0/sft_trainer), [TRL GRPO 0.17](https://huggingface.co/docs/trl/v0.17.0/grpo_trainer). No license is assigned to this project on the user's behalf.
