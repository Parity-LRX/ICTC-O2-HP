#!/bin/bash
#SBATCH --job-name=o2hp-eqx-report
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail
source "$PROJECT_ROOT/benchmarks/eqx/slurm/common_env.sh"
"$PYTHON_BIN" "$SCRIPT_ROOT/report_eqx.py" --raw "${O2HP_RAW_ROOT:-$O2HP_RUN_ROOT/raw}" --out "$O2HP_RUN_ROOT/results"
