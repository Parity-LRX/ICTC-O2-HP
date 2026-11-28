import ast,hashlib,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[1];assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
label=sys.argv[1];out=ROOT/'snapshots'/(label+'-'+os.environ['SLURM_JOB_ID']);control=out/'control';control.mkdir(parents=True)
for path in ROOT.glob('*.py'):ast.parse(path.read_text());shutil.copy2(path,control/path.name)
shutil.copy2(ROOT/'CONTRACT.zh-CN.md',control/'CONTRACT.zh-CN.md')
shutil.copy2(WS/'runs/o2-dd7a-angular-capacity-a100-20261126-r1/snapshots/full-120424/metrics.py',control/'metrics.py')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
source=json.loads((ROOT/'SOURCE.json').read_text())
for name,digest in source['files'].items():assert sha(WS/'ICTC-O2-HP/src'/name)==digest
receipt={'snapshot':str(out),'control':str(control),'source':source,'files':{p.name:sha(p) for p in control.iterdir() if p.is_file()},'job':os.environ['SLURM_JOB_ID'],'source_reused_with_all_hashes_verified':True}
(ROOT/(label.upper()+'-PREFLIGHT.json')).write_text(json.dumps(receipt,indent=2)+'\n');print('CONTROLS_FROZEN',label,out,flush=True)
