#!/usr/bin/env bash

set -eo pipefail

ENV_NAME="ocr-env"
CHANNEL="https://download.pytorch.org/whl/cu128"

export HF_HOME="${HOME}/.hf_offline_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== Creating temporary install directory ==="
BUILD_ROOT="${HOME}/.tmpbuild"
mkdir -p "${BUILD_ROOT}"

TMP_WORKDIR="$(mktemp -d -p "${BUILD_ROOT}" torchbuild.XXXXXXXX)"
export TMPDIR="${TMP_WORKDIR}"
export PIP_CACHE_DIR="${TMP_WORKDIR}/pip-cache"
mkdir -p "$PIP_CACHE_DIR"

cleanup() {
	echo "=== Removing temporary install directory ==="
	rm -rf "${TMP_WORKDIR}" || true
}
trap cleanup EXIT


echo "=== Creating conda environment: $ENV_NAME ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV_NAME" python=3.12

echo "=== Activating environment ==="
conda activate "$ENV_NAME"

echo "=== Install CUDA toolkit ==="
conda install -y -c nvidia cuda-toolkit=12.8
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "=== Installing build prerequisites ==="
conda install -y -c conda-forge git-lfs gxx_linux-64 cmake make ninja
git lfs install

python -m pip install --upgrade pip setuptools wheel
pip install --no-cache-dir hatchling hatch-vcs pybind11[global]
pip install --no-cache-dir -U numpy

echo "=== Clearing cache ==="
conda clean -a -y || true
pip cache purge || true

echo "=== Install PyTorch ==="
pip install --no-cache-dir --default-timeout=600 torch torchvision torchaudio --index-url "$CHANNEL"

echo "=== Installing Nemotron OCR ==="

# Modify this if using A30 or A100 (8.0, not 7.0)
export TORCH_CUDA_ARCH_LIST="7.0"

if [ ! -d "nemotron-ocr-v1" ]; then
	echo "=== Cloning Nemotron Repository ==="
	git clone https://huggingface.co/nvidia/nemotron-ocr-v1
fi

pushd nemotron-ocr-v1 >/dev/null
git lfs pull || true
popd >/dev/null

pushd nemotron-ocr-v1/nemotron-ocr >/dev/null
pip install --no-cache-dir -v .
popd >/dev/null

echo "=== Installing dependencies ==="
pip install -r install_requirements.txt

echo "=== Setup complete ==="
