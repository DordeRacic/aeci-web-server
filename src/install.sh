#!/usr/bin/env bash

set -e

ENV_NAME="ocr-env"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -y -n $ENV_NAME python=3.12

echo "=== Activating environment ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "=== Installing dependencies ==="
pip install -r install_requirements.txt
