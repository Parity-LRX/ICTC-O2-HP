import json,os
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
def read(p):return json.loads(p.read_text())
label=os.environ.get('O2_NUMERIC_STAGE','')
audit=Path((ROOT/((label+'-' if label else '')+'FULL-AUDIT.txt')).read_text().strip());gate=read(audit/'RESULT.json');assert gate['status']=='FULL_QUALIFICATION_PASS'
def brief(m):
    return {'roles':len(m),'failures':sum(not v['passed'] for v in m.values()),
        'max_R':max((v.get('relative_l2',0) or 0 for v in m.values()),default=0),
        'max_abs':max((v.get('max_abs',0) for v in m.values()),default=0),
        'max_ulp':max((v.get('ulp',{}).get('max_ulp',0) for v in m.values()),default=0),
        'floating_elements':sum(v.get('ulp',{}).get('elements',0) for v in m.values()),
        'different_floating_elements':sum(v.get('ulp',{}).get('elements',0)-v.get('ulp',{}).get('equal_elements',0) for v in m.values())}
rows=[]
for mode in ('legacy','isolated','train-only'):
    for step in range(3):
        m=read(audit/f'{mode}-candidate-step{step}.json')
        groups={}
        for group,prefix in (('raw_gradients','/training/gradients/'),('updated_parameters','/training/updated/'),('optimizer','/training/optimizer/'),('EMA','/training/ema/'),('scheduler','/training/scheduler/'),('force_mixed','/mixed/mixed/'),('after_update','/after_update_output/')):
            groups[group]=brief({k:v for k,v in m.items() if k.startswith(prefix)})
        groups['raw_K']=brief({k:v for k,v in m.items() if k.startswith('/training/gradients/') and k.endswith('.K')})
        rows.append({'mode':mode,'step':step+1,'all':brief(m),'groups':groups})
result={'operator_source_sha256':gate['operator_source_sha256'],'package_sha256':gate.get('package_sha256'),'rows':rows,
    'E_F':brief(read(audit/'ef-candidate.json')),'joint_VJP':brief(read(audit/'vjp-candidate.json')),
    'local':gate['reports']['LOCAL'],'AMD_audit':str(audit),'RNG_phase_audited':True}
(ROOT/((label+'-' if label else '')+'NUMERICAL-SUMMARY.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
