import gc,json,os
from pathlib import Path
import torch
from context import ROOT,BASE,save,sha
from metrics import full
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
source=Path((ROOT/'QUICK-RESULT.txt').read_text().strip());m=json.loads((source/'MANIFEST.json').read_text());out=ROOT/'results'/('quick-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir();rows=[]
def load(name):
    assert sha(source/name)==m['files'][name];return torch.load(source/name,map_location='cpu',weights_only=False)
old=Path((BASE/'VERIFY-RESULT.txt').read_text().strip())
for step in range(3):
    r=load(f'legacy-native-step{step}.pt');a=load(f'legacy-candidate-step{step}.pt');v=full(a,r)
    original=torch.load(old/'native'/f'TRAINING-{step}.pt',map_location='cpu',weights_only=False)
    reference=full({k:r[k] for k in ('training','after_update_output')},original);assert all(x['passed'] and x.get('max_abs',0)==0 for x in reference.values())
    save(out/f'step{step}.json',v);bad={k:x for k,x in v.items() if not x['passed']}
    rows.append({'step':step+1,'roles':len(v),'failures':bad,'max_R':max((x.get('relative_l2',0) or 0) for x in v.values()),'max_ulp':max(x.get('ulp',{}).get('max_ulp',0) for x in v.values())})
    print(step+1,len(bad),rows[-1]['max_R'],flush=True);del r,a,original,v;gc.collect()
r={'status':'QUICK_THREE_STEP_PASS' if all(not v['failures'] for v in rows) else 'QUICK_THREE_STEP_FAIL','rows':rows,'package_sha256':m['package_sha256'],'operator_source_sha256':m['operator_source_sha256'],'scope':'early original legacy independent three-step screen; not full qualification/performance authorization'}
save(out/'RESULT.json',r);(ROOT/'QUICK-AUDIT.txt').write_text(str(out)+'\n');print(json.dumps({'status':r['status'],'rows':[{**{k:v for k,v in row.items() if k!='failures'},'FAIL':len(row['failures'])} for row in rows],'package_sha256':m['package_sha256']}),flush=True)
