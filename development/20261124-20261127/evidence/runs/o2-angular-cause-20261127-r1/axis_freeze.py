import ast,hashlib,json,os,shutil
from pathlib import Path
root=Path(__file__).resolve().parent;out=root/'snapshots'/('axis-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
files={}
for name in ('axis_probe.py','support.py'):
    path=root/name;ast.parse(path.read_text());shutil.copy2(path,out/name)
    with path.open('rb') as f:files[name]=hashlib.file_digest(f,'sha256').hexdigest()
(root/'AXIS-PREFLIGHT.json').write_text(json.dumps({'snapshot':str(out),'files':files},indent=2)+'\n')
