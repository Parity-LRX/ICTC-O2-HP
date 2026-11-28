#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --job-name=o2-uvw-trace-parse
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python -u parse_ready.py
