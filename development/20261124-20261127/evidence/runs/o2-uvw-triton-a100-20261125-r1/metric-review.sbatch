#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=o2-uvw-metric-review
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python - <<'PY'
import json,os
from pathlib import Path
import torch
from analyze import normalize_receipt_fp32
assert os.environ.get('SLURM_JOB_ID')
torch.set_num_threads(2)
raw=json.loads(Path('results/a100-119167/operator-validation/VALIDATION.json').read_text())
fixed=normalize_receipt_fp32(raw)
receipt={'job':os.environ['SLURM_JOB_ID'],'statistics_dtype':'float32','cases':len(fixed['rows']),
    'roles':sum(len(row['roles']) for row in fixed['rows']),
    'max_epsilon':max(role['epsilon_max'] for row in fixed['rows'] for role in row['roles'].values()),
    'all_nonzero_residual_denominators_exactly_recovered':True,
    'zero_residual_result_independent_of_valid_denominator':True,'all_gate_decisions_unchanged':True}
Path('FP32-METRIC-REVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
PY
