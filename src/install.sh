#!/usr/bin/env bash

set -eo pipefail

ENV_NAME="ocr-env"

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
conda create -y -n $ENV_NAME python=3.11

echo "=== Activating environment ==="
conda activate $ENV_NAME

echo "=== Install CUDA toolkit ==="
conda install -y -c nvidia cuda-toolkit=11.8
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "=== Clearing cache ==="
conda clean -a -y || true
pip cache purge || true

echo "=== Manually installing wheels ==="
pip install --no-cache-dir --default-timeout=600 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

echo "=== Installing model weights ==="
python -m pip install -U "huggingface-hub>=0.22" hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p .ds_ocr/models/DeepSeek-OCR
"$CONDA_PREFIX/bin/hf" download deepseek-ai/DeepSeek-OCR --local-dir .ds_ocr/models/DeepSeek-OCR

echo "=== Installing Markdown and PDF tools ==="
conda install -y -c conda-forge pandoc wkhtmltopdf

echo "=== Installing dependencies ==="
pip install -r install_requirements.txt

echo "=== Configuring offline settings ==="
conda env config vars set TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME="$PWD/.hf_cache"

echo "=== Setup complete ==="
