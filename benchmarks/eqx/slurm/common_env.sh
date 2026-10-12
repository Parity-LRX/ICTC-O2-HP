# Source this file only inside an sbatch allocation.
set -euo pipefail
: "${SLURM_JOB_ID:?requires sbatch}"
: "${PROJECT_ROOT:?set the absolute ICTC-O2-HP checkout path}"
: "${O2HP_RUN_ROOT:?set a dedicated output directory outside the checkout}"
SCRIPT_ROOT="$PROJECT_ROOT/benchmarks/eqx/scripts"
export O2HP_INSTALL_ROOT="${O2HP_INSTALL_ROOT:-$PROJECT_ROOT/src}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
CACHE_ROOT="$O2HP_RUN_ROOT/cache/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$O2HP_RUN_ROOT/logs" "$O2HP_RUN_ROOT/tmp/$SLURM_JOB_ID" \
  "$CACHE_ROOT/inductor" "$CACHE_ROOT/triton" "$CACHE_ROOT/generated" "$CACHE_ROOT/xdg" "$CACHE_ROOT/matplotlib"
export TMPDIR="$O2HP_RUN_ROOT/tmp/$SLURM_JOB_ID" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$O2HP_INSTALL_ROOT:$SCRIPT_ROOT${O2_UPSTREAM_ROOT:+:$O2_UPSTREAM_ROOT}"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor" TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg" MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export O2_GENERATED_FORWARD_PATH="$CACHE_ROOT/generated/eqx_forward.py"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export O2_LOW_ORDER_FUSED=auto O2_BACKWARD_INPUT_FUSED=auto O2_LOW_ORDER_FORWARD_BLOCK=256
export O2_TRITON_BLOCK=128 O2_WEIGHT_TILE=auto O2_WEIGHT_CHANNEL_TILE=64
export O2_WEIGHT_SCHEDULE=grouped O2_RUNTIME_EDGE_TILE=16 O2_WEIGHT_WARPS=4
export O2_FORWARD_PAIR=0 O2_FORWARD_SPLIT=0 O2_FORWARD_FUSED=0
export O2_FORWARD_DEFERRED=auto O2_FORWARD_WARPS=4
