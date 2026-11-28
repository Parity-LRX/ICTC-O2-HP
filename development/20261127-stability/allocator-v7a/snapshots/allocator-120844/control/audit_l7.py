"""Inspect the preserved L7 trajectory on a scheduler-managed CPU node."""
import json,os,sys,gc
from pathlib import Path
import torch
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
torch.set_num_threads(4)
root=Path(os.environ['O2_SWEEP_ROOT']);ws=root.parents[1]
source=ws/'runs/o2-angular-repair-a100-20261127-r1/gate-captures/120780'
name='blocks.2.ffn.so3_linear_2.bias'
def stats(a,r):
    a=a.float();r=r.float();d=a-r;den=r.norm()
    idx=int(d.abs().argmax())
    return {'norm':float(den),'difference_norm':float(d.norm()),
            'relative_l2':float(d.norm()/den) if den else None,
            'max_abs':float(d.abs().max()),'different':int((a!=r).sum()),
            'worst_index':idx,'candidate_value':float(a.reshape(-1)[idx]),'native_value':float(r.reshape(-1)[idx])}
rows=[]
for step in range(3):
    r=torch.load(source/f'network-L7-native/step{step}.pt',map_location='cpu',weights_only=False)
    a=torch.load(source/f'network-L7-candidate/step{step}.pt',map_location='cpu',weights_only=False)
    row={'step':step,'gradient':stats(a['training']['gradients'][name],r['training']['gradients'][name]),
         'updated':stats(a['training']['updated'][name],r['training']['updated'][name]),
         'loss':stats(a['training']['loss'],r['training']['loss']),
         'mixed_bias':stats(a['mixed']['mixed'][name],r['mixed']['mixed'][name]) if r['mixed']['mixed'][name] is not None else None}
    differences=[]
    for n,ref in r['training']['gradients'].items():
        got=a['training']['gradients'][n]
        if ref is not None and got is not None and not torch.equal(got,ref):differences.append({'name':n,**stats(got,ref)})
    row['gradient_differences']=differences;rows.append(row)
    print(json.dumps({k:v for k,v in row.items() if k!='gradient_differences'}),flush=True)
    del a,r;gc.collect()
out=root/'l7-audits'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True)
(out/'RESULT.json').write_text(json.dumps({'rows':rows,'source':str(source)},indent=2)+'\n')
