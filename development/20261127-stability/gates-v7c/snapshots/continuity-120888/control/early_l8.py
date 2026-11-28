"""Early cross-allocation diagnostic; full same-policy audit remains separate."""
import gc,json,os
from pathlib import Path
from common import ROOT,torch,save
from metrics import full
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
out=ROOT/'early-l8-audits'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True)
reference=ROOT/'gate-captures/120845/network-L8-native'
candidate=ROOT/'gate-captures/120842/network-L8-candidate'
rows=[]
for name in ['ef.pt','joint-vjp.pt','step0.pt','step1.pt','step2.pt']:
    r=torch.load(reference/name,map_location='cpu',weights_only=False)
    a=torch.load(candidate/name,map_location='cpu',weights_only=False)
    metrics=full(a,r);row={'file':name,'roles':len(metrics),'failures':{k:v for k,v in metrics.items() if not v['passed']},
        'maximum_relative_l2':max((v.get('relative_l2',0) or 0) for v in metrics.values())}
    rows.append(row);save(out/'RESULT.json',{'rows':rows,'reference':str(reference),'candidate':str(candidate),'diagnostic_only':True})
    print(json.dumps(row),flush=True);del a,r,metrics;gc.collect()
