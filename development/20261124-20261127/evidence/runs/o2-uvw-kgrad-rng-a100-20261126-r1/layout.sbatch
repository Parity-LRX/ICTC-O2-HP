#!/bin/bash
#SBATCH -p GPU-A100
#SBATCH --nodes=1
#SBATCH --nodelist=XHPC-A100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --job-name=o2-kgrad-layout-local
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export O2_KROOT=/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-kgrad-rng-a100-20261126-r1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export EQV3_DETERMINISTIC=1 TACE_USE_TF32=0 EQX_USE_CUDA_GRAPH=0
export TRITON_CACHE_DIR=$O2_KROOT/triton-cache
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python -u launch.py layout layout_screen.py
