"""AMD full-role local operator epsilon-max gates and ULP reporting."""
import json,os,platform
from pathlib import Path
import torch
from common import ROOT,save,sha
from metrics import ulp
def metric(a,r):
    assert a.shape==r.shape and a.dtype==r.dtype==torch.float32
    assert torch.isfinite(a).all() and torch.isfinite(r).all()
    d=a-r;scale=torch.maximum(a.abs().max(),r.abs().max()).clamp_min(1e-7)
    error=float(d.abs().max()/scale)
    return {'passed':error<=1e-5,'epsilon_max':error,'max_abs':float(d.abs().max()),'normalization_scale':float(scale),'ulp':ulp(a,r)}
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
    source=Path((ROOT/'REGRESSION-RESULT.txt').read_text().strip());record=json.loads((source/'REGRESSION.json').read_text())
    out=ROOT/'results'/('regression-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    results=[];failed=[]
    for index,row in enumerate(record['records']):
        path=Path(row['path']);assert sha(path)==row['sha256']
        v=torch.load(path,map_location='cpu',weights_only=False)
        a={k:metric(v['candidate'][k],r) for k,r in v['reference'].items()}
        repeat={k:metric(v['native_repeat'][k],r) for k,r in v['reference'].items()}
        assert all(m['passed'] for m in repeat.values())
        results.append({'backend':row['backend'],'site':row['site'],'phase':row['phase'],'roles':a,'native_repeat':repeat})
        failed.extend({'site':row['site'],'backend':row['backend'],'phase':row['phase'],'role':k,**m} for k,m in a.items() if not m['passed'])
        del v
        if index%16==0:print(index,len(failed),flush=True)
    save(out/'RESULT.json',{'status':'ACTUAL_LOCAL_REGRESSION_PASS' if not failed else 'ACTUAL_LOCAL_REGRESSION_FAIL',
        'node':platform.node(),'case_count':len(results),'roles_per_case':6,'failures':failed,
        'epsilon_max':max(m['epsilon_max'] for row in results for m in row['roles'].values()),'results':results,
        'source_sha256':sha(source/'REGRESSION.json'),'operator_source_sha256':record['operator_source_sha256']})
    (ROOT/'REGRESSION-AUDIT.txt').write_text(str(out)+'\n');print('cases',len(results),'failures',len(failed),flush=True)
if __name__=='__main__':main()
