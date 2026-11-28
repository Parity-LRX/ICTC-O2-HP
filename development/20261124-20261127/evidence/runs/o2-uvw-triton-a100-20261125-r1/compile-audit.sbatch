#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=o2-uvw-compile
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export TRITON_CACHE_DIR=/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/cpu-triton-cache
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python -u compile_audit.py
