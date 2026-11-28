#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=o2-angular-cause-audit
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export O2_CAUSE_ROOT=/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-angular-cause-20261127-r1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python -u analyze.py
