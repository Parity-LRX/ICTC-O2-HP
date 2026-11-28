import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(os.environ['O2_CAUSE_ROOT']);pre=json.loads((root/'PEAK-PREFLIGHT.json').read_text());source=Path(pre['snapshot'])
for name,digest in pre['files'].items():
    with (source/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
out=root/'results'/('peak-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
rows=[]
for variant in ('native-legacy','native-cache-cleared'):
    target=out/variant;target.mkdir()
    with (target/'process.log').open('w') as log:
        code=subprocess.call([sys.executable,'-u',str(source/'gpu_worker.py'),'oom','--variant',variant,'--output',str(target)],stdout=log,stderr=subprocess.STDOUT)
    rows.append({'variant':variant,'exit_code':code,'folder':str(target)})
(out/'MANIFEST.json').write_text(json.dumps(rows,indent=2)+'\n')
with (out/'first-compute.log').open('w') as log:
    subprocess.check_call([sys.executable,'-u',str(source/'first_compute.py'),'--output',str(out/'first-compute')],stdout=log,stderr=subprocess.STDOUT)
