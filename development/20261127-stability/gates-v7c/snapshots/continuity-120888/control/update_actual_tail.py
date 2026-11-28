"""Apply only independently identified, static exact tail plans."""
import json,os
from pathlib import Path
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
root=Path(__file__).resolve().parent;repo=root.parents[1]/'ICTC-O2-HP'
result=json.loads((Path((root/'ACTUAL-TAIL-PLANS.txt').read_text().strip())/'RESULT.json').read_text())
assert all(r['status']=='EXACT' for r in result['records'])
path=repo/'src/ictc_o2_hp/fp32_reduction_plans.json';data=json.loads(path.read_text());before={}
for key,plan in result['plans'].items():
    before[key]=data['plans'][key];data['plans'][key]=plan
path.write_text(json.dumps(data,indent=2)+'\n')
(root/'TAIL-PLAN-UPDATE.json').write_text(json.dumps({'before':before,'after':result['plans'],'source':str(root/'ACTUAL-TAIL-PLANS.txt')},indent=2)+'\n')
print('UPDATED_STATIC_TAIL_PLANS',len(before),flush=True)
