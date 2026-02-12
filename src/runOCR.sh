#!/bin/bash

#SBATCH --partition=gpuq
#SBATCH --output=output.out
#SBATCH --error=output.err
#SBATCH --time=0-12:00:00

# Important line, do not remove. Solves a bug with C++ library used by OCR.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

module load cuda
srun python -u ocr.py ../../redacted-samples/Abaxis-Zoetis
