import gc,json,os
from pathlib import Path
from common import *
from metrics import full
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
source=Path((ROOT/'COMPLETE-PLANS.txt').read_text().strip());data=json.loads((source/'RESULT.json').read_text());assert data['status']=='COMPLETED';rows=[]
for row in data['records']:
    path=Path(row['arrays']);assert sha(path)==row['sha256'];v=torch.load(path,map_location='cpu',weights_only=False);m=full(v['candidate'],v['reference']);diff=(v['candidate']-v['reference']).abs().max();scale=max(v['candidate'].abs().max(),v['reference'].abs().max(),torch.tensor(1e-7));eps=float(diff/scale)
    rows.append({'key':row['key'],'status':'PASS' if all(r['passed'] for r in m.values()) and eps<=1e-5 else 'FAIL','metrics':m,'epsilon_max':eps});del v;gc.collect()
out=ROOT/'plan-audits'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True);save(out/'RESULT.json',{'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL','rows':rows,'source':str(source),'original_thresholds':True})
assert all(r['status']=='PASS' for r in rows)
save(WS/'ICTC-O2-HP/src/ictc_o2_hp/fp32_reduction_plans.json',{'plans':data['plans'],'status':'CALIBRATED_PLAN_REGISTRY','scope':'A100 FP32 actual observed channel layouts and all registered angular tails/strides; whole-training qualification is independent','source':str(source),'audit':str(out)})
(ROOT/'PLAN-AUDIT.txt').write_text(str(out)+'\n');print('PLANS_READY',len(data['plans']),len(rows),flush=True)
