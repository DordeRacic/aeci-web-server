#!/usr/bin/env bash

set -eo pipefail

ENV_NAME="ocr-env"
CHANNEL="https://download.pytorch.org/whl/cu128"


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
export HF_HOME="${HOME}/.hf_offline_cache"

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

echo "=== Installing Huggingface ==="
pip install -U --no-cache-dir transformers datasets accelerate safetensors tokenizers sentencepiece
pip install -U --no-cache-dir "huggingface_hub[cli]"

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

echo "=== Configuring Nemotron and Huggingface to run locally ==="
MODEL_DIR="${HF_HOME}/models/nvidia_nemotron-ocr-v1"
mkdir -p "${MODEL_DIR}"

python - <<'PY'
import os, sys, time
from huggingface_hub import snapshot_download

repo_id = "nvidia/nemotron-ocr-v1"
local_dir = os.environ.get("MODEL_DIR")
snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False, resume_download=True)
PY

echo "=== Installing dependencies ==="
pip install -r install_requirements.txt


echo "=== Preparing offline configuration activation scripts==="
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" <<EOF
export HF_HOME="${HOME}/.hf_offline_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_ENABLE_TELEMETRY=0
export HF_HUB_DISABLE_HTTP=1
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${HF_HOME}/torch"
export TORCHHUB_DIR="${TORCH_HOME}/hub"
export PYTORCH_HUB_DIR="${TORCH_HOME}/hub"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
EOF
chmod +x "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh" <<EOF
unset HF_HOME
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
unset HF_HUB_ENABLE_TELEMETRY
unset HF_HUB_DISABLE_HTTP
unset TRANSFORMERS_CACHE
unset HF_DATASETS_CACHE
unset HF_HUB_CACHE
unset TORCH_HOME
unset TORCHHUB_DIR
unset PYTORCH_HUB_DIR
unset LD_LIBRARY_PATH
EOF
chmod +x "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh"

echo "=== Setup complete ==="
