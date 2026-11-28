import ast,hashlib,json,os,shutil,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[1]
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
label=sys.argv[1];out=ROOT/'snapshots'/(label+'-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
src=out/'src';shutil.copytree(WS/'ICTC-O2-HP/src',src,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
control=out/'control';control.mkdir()
for path in ROOT.glob('*.py'):ast.parse(path.read_text());shutil.copy2(path,control/path.name)
shutil.copy2(ROOT/'CONTRACT.zh-CN.md',control/'CONTRACT.zh-CN.md')
metrics=WS/'runs/o2-dd7a-angular-capacity-a100-20261126-r1/snapshots/full-120424/metrics.py';shutil.copy2(metrics,control/'metrics.py')
for path in src.rglob('*.py'):ast.parse(path.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
archive=out/'source.tar.gz'
with tarfile.open(archive,'w:gz') as f:f.add(src,arcname='src')
head=(ROOT/'CANDIDATE-HEAD.txt').read_text().strip()
shutil.copy2(ROOT/'candidate-scaffold.bundle',out/'candidate.bundle')
source={'path':str(src),'package_sha256':sha(archive),'candidate_commit':head,'archive':str(archive),
        'files':{str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file()}}
(ROOT/'SOURCE.json').write_text(json.dumps(source,indent=2)+'\n')
receipt={'snapshot':str(out),'control':str(control),'source':source,'files':{p.name:sha(p) for p in control.iterdir() if p.is_file()},'job':os.environ['SLURM_JOB_ID']}
(ROOT/(label.upper()+'-PREFLIGHT.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print('FROZEN',label,head,out,flush=True)
