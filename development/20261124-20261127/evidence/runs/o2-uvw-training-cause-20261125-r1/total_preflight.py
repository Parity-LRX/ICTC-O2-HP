import ast,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert os.environ.get('SLURM_JOB_ID')
out=ROOT/'snapshots'/('total-control-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
files={}
for name in ('total_probe.py','total.sbatch','total_launcher.py'):
    source=ROOT/name
    if source.suffix=='.py':ast.parse(source.read_text())
    shutil.copy2(source,out/name)
    files[name]=hashlib.file_digest((out/name).open('rb'),'sha256').hexdigest()
result={'snapshot':str(out),'files':files,'initial_preflight':json.loads((ROOT/'PREFLIGHT.json').read_text()),
    'authorization':'Same diagnosis; full-loss common cotangent control; additional allocation bounded at 5 minutes; total bound remains 20 GPU minutes.'}
(ROOT/'PREFLIGHT-TOTAL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
