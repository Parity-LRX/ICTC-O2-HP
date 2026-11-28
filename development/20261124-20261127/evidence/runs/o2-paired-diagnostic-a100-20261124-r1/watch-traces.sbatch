#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:35:00
#SBATCH --job-name=o2-trace-distribution
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
rm -f WATCHER-STOP
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python -u watch_traces.py
