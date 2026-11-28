"""Summarize original gates and ULP descriptions without adding a gate."""
import json,os
from pathlib import Path
ROOT=Path(os.environ['O2_REPAIR_ROOT'])
def read(path):return json.loads(path.read_text())
def save(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
    source=Path((ROOT/'REPLAY-AUDIT.txt').read_text().strip());result=read(source/'RESULT.json')
    summaries={};ulp={}
    for path in sorted(source.glob('candidate-*-METRICS.json')):
        if path.name.startswith('candidate-repeat-'):continue
        point=path.name[len('candidate-'):-len('-METRICS.json')];rows=read(path);summaries[point]={}
        for name,r in rows.items():
            fields=name.strip('/').split('/')
            group='/'.join(fields[:2]) if fields[0] in ('training','joint_vjp','force_mixed') else fields[0]
            s=summaries[point].setdefault(group,{'roles':0,'failed':0,'maximum_relative_l2':0,'maximum_absolute':0})
            s['roles']+=1;s['failed']+=int(not r['passed']);s['maximum_relative_l2']=max(s['maximum_relative_l2'],r.get('relative_l2',0) or 0)
            s['maximum_absolute']=max(s['maximum_absolute'],r.get('max_abs',0))
            u=r.get('ulp',{})
            if 'max_ulp' not in u:continue
            t=ulp.setdefault(point+'/'+group+'/'+u['value_dtype'],{'roles':0,'elements':0,'equal_elements':0,'within_one_ulp_elements':0,'max_ulp':0,'max_ulp_role':None})
            for key in ('elements','equal_elements','within_one_ulp_elements'):t[key]+=u[key]
            t['roles']+=1
            if u['max_ulp']>t['max_ulp']:t['max_ulp']=u['max_ulp'];t['max_ulp_role']=name
    local=read(Path((ROOT/'REGRESSION-AUDIT.txt').read_text().strip())/'RESULT.json')
    actual={}
    for row in local['results']:
        for name,r in row['roles'].items():
            t=actual.setdefault(name,{'roles':0,'nonzero_max_abs_roles':0,'epsilon_max':0,'max_ulp':0})
            t['roles']+=1;t['nonzero_max_abs_roles']+=int(r['max_abs']!=0);t['epsilon_max']=max(t['epsilon_max'],r['epsilon_max']);t['max_ulp']=max(t['max_ulp'],r['ulp']['max_ulp'])
    save(ROOT/'NUMERICAL-SUMMARY.json',{'status':result['status'],'same_state_groups':summaries,'ulp_descriptors':ulp,
        'actual_local_groups':actual,'ULP_is_not_an_acceptance_gate':True,'source':str(source),
        'independent_trajectory':'119415 FAIL retained; after first update full states differed; see REPLAY-CONTRACT.zh-CN.md'})
    print(json.dumps({'status':result['status'],'groups':summaries,'actual_local':actual}),flush=True)
if __name__=='__main__':main()
