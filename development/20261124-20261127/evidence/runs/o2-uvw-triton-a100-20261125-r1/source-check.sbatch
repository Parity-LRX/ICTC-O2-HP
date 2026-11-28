#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:03:00
#SBATCH --job-name=o2-uvw-source-check
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python - <<'PY'
import ast,os
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
files=list(Path.cwd().glob('*.py'))+list((Path.cwd()/'figures').glob('*.py'))
for path in files:ast.parse(path.read_text(),filename=str(path))
print('AST_PASS',len(files))
PY
