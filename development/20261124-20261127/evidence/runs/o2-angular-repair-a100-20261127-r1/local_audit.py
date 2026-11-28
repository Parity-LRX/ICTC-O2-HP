import json,os
from pathlib import Path
from common import *
from metrics import full
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
source=Path((ROOT/'LOCAL-RESULT.txt').read_text().strip());records=json.loads((source/'RESULT.json').read_text())['records'];rows=[]
for row in records:
    p=Path(row['path']);assert sha(p)==row['sha256'];v=torch.load(p,map_location='cpu',weights_only=False);m=full(v['candidate'],v['native']);eps=[]
    def epsilon(a,r):
        if isinstance(r,dict):
            for k in r:epsilon(a[k],r[k])
        elif isinstance(r,(list,tuple)):
            for x,y in zip(a,r):epsilon(x,y)
        elif isinstance(r,torch.Tensor):eps.append(float((a-r).abs().max()/max(float(a.abs().max()),float(r.abs().max()),1e-7)))
    epsilon(v['candidate'],v['native']);fail={k:r for k,r in m.items() if not r['passed']}
    rows.append({**row,'status':'PASS' if not fail and max(eps)<=1e-5 else 'NUMERIC_FAIL','failures':fail,'epsilon_max':max(eps),'roles':len(m),'maximum_ULP':max(r.get('ulp',{}).get('max_ulp',0) for r in m.values())})
out=ROOT/'local-audits'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True);save(out/'RESULT.json',{'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL','rows':rows});(ROOT/'LOCAL-AUDIT.txt').write_text(str(out)+'\n')
print('LOCAL_AUDIT',len(rows),sum(r['status']!='PASS' for r in rows),flush=True)
