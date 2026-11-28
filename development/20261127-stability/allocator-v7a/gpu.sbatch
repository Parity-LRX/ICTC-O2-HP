#!/bin/bash
#SBATCH -p GPU-A100
#SBATCH --nodes=1
#SBATCH --nodelist=XHPC-A100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=o2-angular-stability
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export O2_SWEEP_ROOT=/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-angular-stability-a100-20261127-r1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 EQV3_DETERMINISTIC=1 EQX_USE_CUDA_GRAPH=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-angular-stability-a100-20261127-r1/triton-cache
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python -u launch.py "$@"
if [[ "$1" == "inventory" ]]; then
  /public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python -u launch.py inventory calibrate.py --kind all
fi
