import ast,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(os.environ['O2_TRAJECTORY_ROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
out=ROOT/'snapshots'/('control-'+os.environ['SLURM_JOB_ID']);out.mkdir()
for name in ('gpu.py','CONTRACT.zh-CN.md'):
    p=ROOT/name
    if p.suffix=='.py':ast.parse(p.read_text())
    shutil.copy2(p,out/name)
files={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in out.iterdir()}
repair=ROOT.parent/'o2-uvw-fp32-repair-a100-20261126-r1';qual=json.loads((repair/'QUALIFICATION-PREFLIGHT.json').read_text());dependencies={}
for name,digest in qual['files'].items():
    p=Path(qual['snapshot'])/name;assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==digest;dependencies[str(p)]=digest
control=ROOT.parent/'o2-uvw-triton-a100-20261125-r1/snapshots/preflight-119174'
for name in ('whole.py','bounded.py','diagnose.py','adapter/install.py'):
    p=control/name;dependencies[str(p)]=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
(ROOT/'PREFLIGHT.json').write_text(json.dumps({'snapshot':str(out),'files':files,'dependencies':dependencies},indent=2)+'\n');print(out,flush=True)
