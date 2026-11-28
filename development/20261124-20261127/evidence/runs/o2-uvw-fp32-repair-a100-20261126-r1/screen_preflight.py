import ast,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;assert os.environ.get('SLURM_JOB_ID')
audit=Path((ROOT/'CAPTURE-AUDIT.txt').read_text().strip());assert json.loads((audit/'RESULT.json').read_text())['status']=='REAL_TRAINING_CAPTURE_AUDIT_PASS'
out=ROOT/'snapshots'/('screen-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
for name in ('common.py','candidate_kernels.py','local_screen.py','local_screen.sbatch','screen_launcher.py'):
    p=ROOT/name
    if p.suffix=='.py':ast.parse(p.read_text())
    shutil.copy2(p,out/name)
files={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in out.iterdir()}
(ROOT/'SCREEN-PREFLIGHT.json').write_text(json.dumps({'snapshot':str(out),'files':files,'job':os.environ['SLURM_JOB_ID']},indent=2)+'\n')
print(json.dumps(files),flush=True)
