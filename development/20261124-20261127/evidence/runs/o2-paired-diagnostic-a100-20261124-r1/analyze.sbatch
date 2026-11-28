#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=o2-paired-analysis
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/prepared/mpl-cache-$SLURM_JOB_ID"
O2_DIAGNOSTIC_RESULTS="$(cat LATEST-RESULT.txt)"
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python -u analyze.py "$O2_DIAGNOSTIC_RESULTS" "$PWD/results/analysis-$SLURM_JOB_ID"
