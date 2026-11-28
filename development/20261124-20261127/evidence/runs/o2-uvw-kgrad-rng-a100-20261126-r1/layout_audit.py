import json,os,sys
from pathlib import Path
import torch
from context import ROOT,BASE,save,sha
sys.path.insert(0,str(BASE))
from regression_audit import metric
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
    source=Path((ROOT/'LAYOUT-RESULT.txt').read_text().strip());manifest=json.loads((source/'MANIFEST.json').read_text());rows=[]
    for r in manifest['records']:
        assert sha(Path(r['path']))==r['sha256'];v=torch.load(r['path'],map_location='cpu',weights_only=False)
        roles={k:metric(v['candidate'][k],t) for k,t in v['reference'].items()}
        rows.append({**r,'roles':roles,'direct_torch':metric(v['direct_torch'],v['reference']['weight_adjoint']),'swapped_torch':metric(v['swapped_torch'],v['reference']['weight_adjoint'])})
    result={'status':'LAYOUT_LOCAL_PASS' if all(v['passed'] for r in rows for v in r['roles'].values()) else 'LAYOUT_LOCAL_FAIL',
        'operator_source_sha256':manifest['operator_source_sha256'],'rows':rows}
    out=ROOT/'results'/('layout-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir();save(out/'RESULT.json',result);(ROOT/'LAYOUT-AUDIT.txt').write_text(str(out)+'\n')
    print(json.dumps({'status':result['status'],'rows':[{'shape':r['weight_shape'],'phase':r['phase'],'candidate_weight':r['roles']['weight_adjoint'],'direct_torch':r['direct_torch'],'swapped_torch':r['swapped_torch']} for r in rows]}),flush=True)
if __name__=='__main__':main()
