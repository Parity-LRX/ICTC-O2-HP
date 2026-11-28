import ast,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path(os.environ['O2_HOT_ROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
snapshot=root/'snapshots'/('analysis-'+os.environ['SLURM_JOB_ID']);snapshot.mkdir(parents=True)
for p in root.glob('*.py'):ast.parse(p.read_text());shutil.copy2(p,snapshot/p.name)
shutil.copy2(root/'CONTRACT.zh-CN.md',snapshot/'CONTRACT.zh-CN.md')
files={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in snapshot.iterdir() if p.is_file()}
(root/'PREFLIGHT.json').write_text(json.dumps({'snapshot':str(snapshot),'files':files,'job':os.environ['SLURM_JOB_ID']},indent=2)+'\n')
subprocess.run([sys.executable,'-u',str(snapshot/'analyze.py')],check=True)
