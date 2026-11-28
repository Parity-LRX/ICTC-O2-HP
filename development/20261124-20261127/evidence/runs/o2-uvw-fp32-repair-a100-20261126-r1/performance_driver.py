"""Serial backend/mode/process matrix on one physical A100 allocation."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ['O2_REPAIR_ROOT']);receipt=json.loads((ROOT/'PERFORMANCE-PREFLIGHT.json').read_text());source=Path(receipt['snapshot'])
for name,digest in receipt['files'].items():assert hashlib.file_digest((source/name).open('rb'),'sha256').hexdigest()==digest
env=os.environ.copy();env['O2_REPAIR_SOURCE']=receipt['qualified_source']
out=ROOT/'results'/('performance-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
for mode in ('ef','train'):
    for repetition in range(3):
        order=('native','v1','candidate') if repetition%2==0 else ('candidate','v1','native')
        for backend in order:
            path=out/f'{mode}-{backend}-time-{repetition}'
            subprocess.run([sys.executable,'-u',str(source/'performance.py'),mode,backend,'time',str(path)],cwd=ROOT,env=env,check=True)
            print(path.name,flush=True)
    for backend in ('native','v1','candidate'):
        path=out/f'{mode}-{backend}-memory'
        subprocess.run([sys.executable,'-u',str(source/'performance.py'),mode,backend,'memory',str(path)],cwd=ROOT,env=env,check=True)
        print(path.name,flush=True)
    for backend in ('native','candidate'):
        path=out/f'{mode}-{backend}-trace'
        subprocess.run([sys.executable,'-u',str(source/'performance.py'),mode,backend,'trace',str(path)],cwd=ROOT,env=env,check=True)
        print(path.name,flush=True)
(ROOT/'PERFORMANCE-RESULT.txt').write_text(str(out)+'\n')
