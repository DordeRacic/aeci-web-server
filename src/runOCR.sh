#!/bin/bash

#SBATCH --partition gpuq
#SBATCH --output=output.out
#SBATCH --error=output.err
#SBATCH --time=0-12:00:00

module load cuda
srun python -u ocr.py ../../redacted-sample/Abaxis-Zoetis
