#!/usr/bin/env bash
set -euo pipefail
# Run on an H100/H200 Docker host, NOT Kaggle T4. Requires git, docker + NVIDIA Container Toolkit.
# Put this bundle in a directory with sufficient free disk. Run from that directory.
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p cache
if [ ! -d halo/.git ]; then git clone https://github.com/whitecircle/halo halo; fi
git -C halo checkout 3a21b4dbdb56cb840d42a59e98e40f967a1d2807
git -C halo rev-parse HEAD > halo_commit.txt
IMAGE=public.ecr.aws/whitecircle/halo:hopper-1.3.0rc1
docker pull "$IMAGE"
docker image inspect "$IMAGE" > halo_image.json
# Resolve and record model revision, then run official CLI in the pinned image.
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=128g \
 -e CUDA_VISIBLE_DEVICES=0 -e HF_HOME=/cache/hf -e HF_DATASETS_CACHE=/cache/datasets -e HALO_DATA_ROOT=/cache \
 -v "$HERE/halo:/workspace" -v "$HERE:/workspace/halo-smoke" -v "$HERE/cache:/cache" -w /workspace "$IMAGE" \
 bash -lc 'set -e; nvidia-smi; python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(including_emulation=False)"; halo launch --list; python /workspace/halo-smoke/prepare_model.py; halo launch sft /workspace/halo-smoke/config.yaml' 2>&1 | tee halo_run.log
# Record artifacts; a log alone is not proof of training.
find output -type f -name adapter_model.safetensors -print -exec sha256sum {} \; | tee adapter_checksums.txt
test -s adapter_checksums.txt
