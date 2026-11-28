"""Retain every role's error/ULP descriptors from the already captured full matrix."""
import json,os,subprocess,sys
from pathlib import Path
root=Path(os.environ['O2_SWEEP_ROOT']);control=Path(__file__).parent
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
gate=json.loads((root/'gate-audits/120876/RESULT.json').read_text())
ops=[r for r in gate['rows'] if r.get('identity',{}).get('level')=='operator']
nets=[r for r in gate['rows'] if r.get('level')=='network']
assert len(ops)==144 and all(r['status']=='PASS' for r in ops)
assert len(nets)==9 and all(r['infer']==r['train']=='PASS' for r in nets)
assert json.loads((root/'local-audits/120877/RESULT.json').read_text())['status']=='PASS'
assert Path((root/'LOCAL-RESULT.txt').read_text().strip()).name=='120870'
env={**os.environ,'O2_GATE_CAPTURE_POINTER':'GATE-CAPTURE-120870.txt'}
for name in ('gate_audit.py','local_audit.py'):
    subprocess.run([sys.executable,'-u',str(control/name)],env=env,check=True)
print('ALL_ROLE_DESCRIPTORS_RETAINED',os.environ['SLURM_JOB_ID'],flush=True)
