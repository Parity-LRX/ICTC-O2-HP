"""AMD source seal before the first training operand capture."""
import ast,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[1];assert os.environ.get('SLURM_JOB_ID')
out=ROOT/'snapshots/capture-v1';out.mkdir(parents=True)
shutil.copytree(WS/'ICTC-O2-HP/src',out/'ictc-source')
(out/'control').mkdir()
for p in ROOT.glob('*.py'):
    ast.parse(p.read_text());shutil.copy2(p,out/'control'/p.name)
for pattern in ('*.sbatch','*.md'):
    for p in ROOT.glob(pattern):shutil.copy2(p,out/'control'/p.name)
files={str(p.relative_to(out)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in out.rglob('*') if p.is_file()}
old=WS/'runs/o2-uvw-triton-a100-20261125-r1/snapshots/preflight-119174/fork-src/ictc_o2_hp/uvw_triton.py'
assert files['ictc-source/ictc_o2_hp/uvw_triton.py']==hashlib.file_digest(old.open('rb'),'sha256').hexdigest()
receipt={'status':'CAPTURE_SOURCE_PREFLIGHT_PASS','job':os.environ['SLURM_JOB_ID'],'snapshot':str(out),'files':files}
(ROOT/'PREFLIGHT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
