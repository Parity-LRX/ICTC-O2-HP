import json,os
import torch
from context import ROOT,save
from metrics import full
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
source=ROOT/'results/qualification-a100-120073';rows=[]
for step in range(3):
    r=torch.load(source/f'legacy-native-step{step}.pt',map_location='cpu',weights_only=False)
    a=torch.load(source/f'legacy-candidate-step{step}.pt',map_location='cpu',weights_only=False)
    parts={k:full(a['training'][k],r['training'][k]) for k in ('gradients','updated')}
    rows.append({'step':step,'groups':{k:{'failed':{n:v for n,v in m.items() if not v['passed']},'max_R':max(v.get('relative_l2',0) or 0 for v in m.values()),
        'max_ulp':max(v.get('ulp',{}).get('max_ulp',0) for v in m.values()),'different_roles':[n for n,v in m.items() if v.get('max_abs',0)>0]} for k,m in parts.items()}})
save(ROOT/('EARLY-GRAD-'+os.environ['SLURM_JOB_ID']+'.json'),rows);print(json.dumps(rows),flush=True)
