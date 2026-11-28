import ast,hashlib,json,os,shutil
from pathlib import Path
root=Path(__file__).resolve().parent;out=root/'snapshots'/('peak-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
files={}
for name in ('gpu_worker.py','support.py','peak_launch.py','peak.sbatch','first_compute.py'):
    path=root/name
    if path.suffix=='.py':ast.parse(path.read_text())
    shutil.copy2(path,out/name)
    with path.open('rb') as f:files[name]=hashlib.file_digest(f,'sha256').hexdigest()
receipt=json.dumps({'snapshot':str(out),'files':files},indent=2)+'\n'
(root/f'PEAK-PREFLIGHT-{os.environ["SLURM_JOB_ID"]}.json').write_text(receipt)
(root/'PEAK-PREFLIGHT.json').write_text(receipt)
