import ast,hashlib,json,os,shutil,sys
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';label=sys.argv[1]
out=ROOT/'snapshots'/(label+'-'+os.environ['SLURM_JOB_ID']);out.mkdir();(out/'control').mkdir()
for p in ROOT.glob('*.py'):ast.parse(p.read_text());shutil.copy2(p,out/'control'/p.name)
shutil.copy2(ROOT/'CONTRACT.zh-CN.md',out/'CONTRACT.zh-CN.md')
shutil.copytree(ROOT.parents[1]/'ICTC-O2-HP/src',out/'ictc-source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
files={str(p.relative_to(out)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in out.rglob('*') if p.is_file()}
receipt={'snapshot':str(out),'files':files,'operator_source_sha256':files['ictc-source/ictc_o2_hp/uvw_triton.py'],
    'candidate_commit':(ROOT/'CANDIDATE-HEAD.txt').read_text().strip(),'label':label,'job':os.environ['SLURM_JOB_ID']}
(ROOT/(label.upper()+'-PREFLIGHT.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
