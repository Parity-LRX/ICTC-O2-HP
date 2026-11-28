#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=o2-uvw-fork
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
cd /public/home/sps-xia/bjxiao/o2-benchmark
test ! -e ICTC-O2-HP
git clone https://github.com/Parity-LRX/ICTC-O2-HP.git ICTC-O2-HP
cd ICTC-O2-HP
git switch -c codex/uvw-triton-v1
git rev-parse HEAD > ../runs/o2-uvw-triton-a100-20261125-r1/REMOTE-BASE.txt
git rm -r --quiet .
cp -a ../baselines/icml-o2-stage1-20261124-r1/payload/operator/upstream/official/. .
git add -A
git commit -m 'Restore sealed ICTC-O2-HP stage1 package as optimization base'
git tag snapshot/uvw-frozen-base-20261125
git rev-parse HEAD > ../runs/o2-uvw-triton-a100-20261125-r1/FORK-BASE.txt
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python - <<'PY'
import hashlib,json,os
from pathlib import Path
root=Path.cwd(); base=root.parent/'baselines/icml-o2-stage1-20261124-r1/payload/operator/upstream/official'
rows={}
for p in sorted(base.rglob('*')):
    if p.is_file():
        rel=str(p.relative_to(base));a=hashlib.sha256(p.read_bytes()).hexdigest();b=hashlib.sha256((root/rel).read_bytes()).hexdigest()
        assert a==b,rel
        rows[rel]=a
(root.parent/'runs/o2-uvw-triton-a100-20261125-r1/FORK-BASE-SEAL.json').write_text(json.dumps({'job':os.environ['SLURM_JOB_ID'],'baseline_wrapper':'e4763c76bf183499357e40cb39cc0934bb98779b','package_files':rows},indent=2)+'\n')
print('FROZEN_FORK_VERIFIED',len(rows),flush=True)
PY
