#!/bin/bash
#SBATCH -p GPU-A100
#SBATCH --nodes=1
#SBATCH --nodelist=XHPC-A100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=o2-uvw-real-regression
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export O2_REPAIR_ROOT=/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-fp32-repair-a100-20261126-r1
export TRITON_CACHE_DIR=$O2_REPAIR_ROOT/triton-cache
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python -u qualification_launcher.py real_regression.py
