import json,os,platform
from pathlib import Path
import torch
from common import ROOT,save,sha
from regression_audit import metric
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
    source=Path((ROOT/'GENERIC-RESULT.txt').read_text().strip());manifest=json.loads((source/'MANIFEST.json').read_text())
    results=[];failed=[]
    for row in manifest['records']:
        assert sha(Path(row['path']))==row['sha256'];v=torch.load(row['path'],map_location='cpu',weights_only=False)
        a={k:metric(v['candidate'][k],r) for k,r in v['reference'].items()}
        repeat={k:metric(v['native_repeat'][k],r) for k,r in v['reference'].items()}
        assert all(m['passed'] for m in repeat.values())
        failed.extend({'case':row['name'],'role':k,**m} for k,m in a.items() if not m['passed'])
        results.append({'case':row['name'],'roles':a,'native_repeat':repeat})
    out=ROOT/'results'/('generic-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    save(out/'RESULT.json',{'status':'GENERIC_REGRESSION_PASS' if not failed else 'GENERIC_REGRESSION_FAIL',
        'node':platform.node(),'case_count':len(results),'role_count':sum(len(v['roles']) for v in results),
        'live_weight_update_seen':manifest['live_weight_update_seen'],'failures':failed,'results':results,
        'operator_source_sha256':manifest['operator_source_sha256']})
    (ROOT/'GENERIC-AUDIT.txt').write_text(str(out)+'\n');print('failed',len(failed),flush=True)
if __name__=='__main__':main()
