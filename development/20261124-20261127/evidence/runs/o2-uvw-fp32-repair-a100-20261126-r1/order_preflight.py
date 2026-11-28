import ast,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
out=ROOT/'snapshots/order-r1';out.mkdir(parents=True)
for name in ('common.py','order_signature.py','order_audit.py','order.sbatch'):
    p=ROOT/name
    if p.suffix=='.py':ast.parse(p.read_text())
    shutil.copy2(p,out/name)
files={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in out.iterdir()}
(ROOT/'ORDER-PREFLIGHT.json').write_text(json.dumps({'snapshot':str(out),'files':files,'job':os.environ['SLURM_JOB_ID']},indent=2)+'\n')
print(json.dumps(files),flush=True)
