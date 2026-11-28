import ast,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;assert os.environ.get('SLURM_JOB_ID')
assert (ROOT/'CAPTURE-AUDIT.txt').exists()
out=ROOT/'snapshots'/('train-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
for name in ('common.py','candidate_kernels.py','train_screen.py','train_screen.sbatch','train_launcher.py'):
    p=ROOT/name
    if p.suffix=='.py':ast.parse(p.read_text())
    shutil.copy2(p,out/name)
files={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in out.iterdir()}
if (ROOT/'EXTRA-CANDIDATES.json').exists():
    p=ROOT/'EXTRA-CANDIDATES.json';shutil.copy2(p,out/p.name)
    files[p.name]=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
(ROOT/'TRAIN-PREFLIGHT.json').write_text(json.dumps({'snapshot':str(out),'files':files,'job':os.environ['SLURM_JOB_ID']},indent=2)+'\n')
print(json.dumps(files),flush=True)
