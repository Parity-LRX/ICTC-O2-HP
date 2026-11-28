#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:08:00
#SBATCH --job-name=o2-parser-check
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python -u validate_parser.py
