#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=o2-uvw-whole-amd-gate
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export O2_REPAIR_ROOT=/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-fp32-repair-a100-20261126-r1
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python -u qualification_launcher.py verify_audit.py
