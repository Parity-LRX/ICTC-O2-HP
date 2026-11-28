import ast,hashlib,json,os,shutil,sys
from pathlib import Path
import torch
ROOT=Path(os.environ['O2_SWEEP_ROOT']);WS=ROOT.parents[1];assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
label=sys.argv[1];out=ROOT/'snapshots'/(label+'-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
if label=='full':
    audit=Path((ROOT/'SMOKE-AUDIT.txt').read_text().strip())
    assert json.loads((audit/'RESULT.json').read_text())['status']=='SMOKE_PASS'
if label=='measurement':
    audit=Path((ROOT/'GATE-AUDIT.txt').read_text().strip())
    assert json.loads((audit/'RESULT.json').read_text())['status']=='GATES_REVIEWED'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
models=json.loads((ROOT/'MODELS.json').read_text());assert models['status']=='ANGULAR_MODELS_PREPARED'
for row in models['rows']:
    assert sha(row['path'])==row['sha256'];p=ROOT/'models'/f'BILINEAR-L{row["lmax"]}.pt'
    k=torch.load(row['path'],map_location='cpu',weights_only=False)['kernels']['S01']
    if p.exists():assert torch.equal(torch.load(p,weights_only=True),k)
    else:torch.save(k,p)
for p in ROOT.glob('*.py'):ast.parse(p.read_text());shutil.copy2(p,out/p.name)
metrics=WS/'runs/o2-uvw-fp32-repair-a100-20261126-r1/snapshots/qualification-119423/control/metrics.py'
shutil.copy2(metrics,out/'metrics.py');shutil.copy2(ROOT/'CONTRACT.zh-CN.md',out/'CONTRACT.zh-CN.md')
receipt={'label':label,'snapshot':str(out),'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'models':models,'source':json.loads((ROOT/'SOURCE.json').read_text())}
(ROOT/(label.upper()+'-PREFLIGHT.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'label':label,'snapshot':str(out),'models':len(models['rows'])}),flush=True)
