#!/usr/bin/env bash

set -e

ENV_NAME='ocr-env'

echo "=== Deleting conda environment: $ENV_NAME ==="

source "$(conda info --base)/etc/profile.d/conda.sh"

conda remove -y --name $ENV_NAME --all

echo "=== Deleting downloaded models and cache ==="
rm -rf ./.hf_cache/
rm -rf ./.ds_ocr/

echo "=== Cleanup complete ==="
