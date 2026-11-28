"""Run the complete registered matrix only after the affected-mode gates pass."""
import json,os,subprocess,sys
from pathlib import Path
root=Path(os.environ['O2_SWEEP_ROOT']);control=Path(__file__).parent
gate=json.loads((root/'gate-audits/120867/RESULT.json').read_text())
row,=[r for r in gate['rows'] if r.get('order')==7 and r.get('level')=='network']
assert row['infer']==row['train']=='PASS',row
assert json.loads((root/'local-audits/120868/RESULT.json').read_text())['status']=='PASS'
assert json.loads((root/'package-checks/120865/RESULT.json').read_text())['status']=='PASS'
candidate=json.loads((Path(gate['source'])/'network-L7-candidate/RESULT.json').read_text())
assert candidate['identity']['source_package_sha256']==json.loads(os.environ['O2_REPAIR_SOURCE_RECORD'])['package_sha256']
subprocess.run([sys.executable,'-u',str(control/'gate_driver.py')],check=True)
subprocess.run([sys.executable,'-u',str(control/'local_regress.py')],check=True)
