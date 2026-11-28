import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
gate=Path((ROOT/'GATE-AUDIT.txt').read_text().strip());data=json.loads((gate/'RESULT.json').read_text())
local=Path((ROOT/'LOCAL-AUDIT.txt').read_text().strip());regression=json.loads((local/'RESULT.json').read_text());assert regression['status']=='PASS'
network={r['order']:r for r in data['rows'] if r.get('level')=='network'};operators=[r for r in data['rows'] if 'key' in r]
assert set(network)==set(range(9)) and len(operators)==144
assert all(r['infer']==r['train']=='PASS' for r in network.values()),'Whole-network gates must pass before this measurement campaign'
assert all(r['status']=='PASS' for r in operators),'Operator gates must pass before measurement'
(ROOT/'MEASUREMENT-ELIGIBILITY.json').write_text(json.dumps({'status':'ALL_REQUESTED_MODES_ELIGIBLE','network_orders':sorted(network),'operator_cases':len(operators),'gate_audit':str(gate),'local_audit':str(local),'original_thresholds':True},indent=2)+'\n')
sys.argv=[str(ROOT/'control_freeze.py'),'measurement'];exec(compile((ROOT/'control_freeze.py').read_text(),str(ROOT/'control_freeze.py'),'exec'))
