#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=o2-v3-canonical-final
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export O2_GENERAL_RUN_DIR="$PWD"
python_path=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python
"$python_path" prepare.py
"$python_path" -u "$PWD/snapshots/control-$SLURM_JOB_ID/finalize.py"
