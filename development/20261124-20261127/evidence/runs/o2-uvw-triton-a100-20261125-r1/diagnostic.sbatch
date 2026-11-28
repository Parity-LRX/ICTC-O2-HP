#!/bin/bash
#SBATCH -p GPU-A100
#SBATCH --nodes=1
#SBATCH --nodelist=XHPC-A100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=o2-uvw-diagnostic
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 EQV3_DETERMINISTIC=1 TACE_USE_TF32=0
export EQX_USE_CUDA_GRAPH=0
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python -u controller.py
