#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=o2-v3-canonical-convert
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 EQV3_DETERMINISTIC=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export O2_GENERAL_RUN_DIR="$PWD"
python_path=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python
"$python_path" prepare.py
"$python_path" -u "$PWD/snapshots/control-$SLURM_JOB_ID/convert.py"
