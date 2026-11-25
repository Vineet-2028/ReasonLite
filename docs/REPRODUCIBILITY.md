# Reproduction guide

## Fast reviewer audit (CPU only)

From repository root: `python -m unittest discover -s tests -v`, then `python -m scripts.audit_evidence`. Standard library only; no GPU, network or external dependency required. The audit validates exact hashes, saved tensor finiteness, row identities, per-answer scorer output, metrics, paired tests and rollout rewards. It verifies stored evidence, not independent model execution. `python -m reasonlite.final_report` regenerates CSV/Markdown/JSON/plots **in place**; use a disposable checkout if preserving release hashes. Matplotlib is optional for regeneration.

## GPU replay (fresh checkout; do not overwrite original evidence)

Use one CUDA-visible GPU, a compatible PyTorch build and the pinned direct dependencies in `requirements.txt`. Actual run Python 3.13.16 / torch 2.11.0+cu130. The direct versions are not a full lock. Fresh environments and GPU kernels may yield different outputs. Avoid installing Halo in this environment. Keep original evidence untouched. In Git Bash on Linux/Kaggle shell:

```bash
export CUDA_VISIBLE_DEVICES=0
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m reasonlite.evaluate --device cuda --limit 128 --out local_runs/e0_validation
python -m reasonlite.train_sft --out local_runs/sft_smoke --max-steps 2
python -m reasonlite.train_sft --out local_runs/e1_sft
python -m reasonlite.evaluate --device cuda --limit 128 --adapter local_runs/e1_sft/adapter --out local_runs/e1_validation
python -m reasonlite.train_grpo --parent runs/e1_sft_v2/adapter --out local_runs/e2 --mode correctness --prompts 64
python -m reasonlite.train_grpo --parent runs/e1_sft_v2/adapter --out local_runs/e3 --mode efficiency --prompts 64
```

GRPO explicitly expects the original E1 adapter checksum; a freshly trained E1 is not assumed byte-identical. A new protocol and explicit code change are necessary to use a different parent. Never overwrite original checkpoints. The release GRPO exporter records nonfinite diagnostics without corrupting JSON; its training configuration remains unchanged but GPU replay of the maintenance patch is not verified.

### Replay frozen test only (optional, not necessary for review)

Run on a compatible GPU with the same pinned environment; use an isolated checkout. The original calibration records are required by the frozen evaluator; they are already included. Start with calibration checks, then reproduce all four arms using the saved checkpoint and original protocol:

```bash
for arm in E0 E1 E2 E3; do
  python -m reasonlite.final_eval --arm "$arm" --split validation --out "local_runs/calibration/$arm" || exit 1
done
for arm in E0 E1 E2 E3; do
  python -m reasonlite.final_eval --arm "$arm" --split test --out "local_runs/test/$arm" || exit 1
done
```

The evaluator explicitly checks original calibration environment fields before final replay; hardware/software differences may cause it to refuse. Do not bypass checks or confuse new outputs with the original frozen run. Batched evaluation checkpoints are written per batch and safely reused with the same protocol/environment. New `local_runs/` is ignored by Git. No further tuning on the original test set.

## Notebooks

`notebooks/Review.ipynb` is CPU-only and runs after cloning. Historical Kaggle notebooks in `provenance/notebooks/` and root `ReasonLite_Colab.ipynb` capture the earlier workflow, including pre-repair code; they are **not** the recommended fresh-start workflow. Use the final source and commands above. `halo_native/README.md` explains the separate native Halo recipe, which needs a supported container/GPU and remains unexecuted.

To check every original package byte, run `python -m scripts.audit_evidence --strict-release` before editing. The default audit checks frozen code/data/evidence only, so truthful changes to README or contribution documentation do not break CI. Preserve the original release hash manifest as historical provenance.
