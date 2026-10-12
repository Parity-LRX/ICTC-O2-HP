#!/bin/bash
#SBATCH --job-name=o2hp-eqx-scan
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-4%2
set -euo pipefail
: "${PROJECT_ROOT:?set checkout path}"
: "${O2_UPSTREAM_ROOT:?set the pinned external eqx checkout path}"
source "$PROJECT_ROOT/benchmarks/eqx/slurm/common_env.sh"
"$PYTHON_BIN" "$SCRIPT_ROOT/run_scan.py" --phase "${O2HP_SCAN_PHASE:-all}" --out "$O2HP_RUN_ROOT/raw"
