"""AMD audit of genuine training capture and all 10504-row operands."""
import json,os,platform
from pathlib import Path
import torch
from common import ROOT,OLD,save,sha,compare_tree
from metrics import full
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
    source=Path((ROOT/'CAPTURE-RESULT.txt').read_text().strip());capture=json.loads((source/'CAPTURE.json').read_text())
    out=ROOT/'results'/('capture-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    results={}
    for backend,oldname in (('native','native'),('v1','tiled')):
        rows=capture['backends'][backend];assert len(rows)==28
        for row in rows:
            assert sha(Path(row['path']))==row['sha256']
            v=torch.load(row['path'],map_location='cpu',weights_only=False)
            assert v['x'].shape[0]*v['x'].shape[1]==10504
            assert v['x'].dtype==v['K'].dtype==v['y'].dtype==torch.float32
            assert v['K'].shape[1]==v['x'].shape[-1]
            assert v['y'].shape==v['adjoints']['energy_forward_and_force']['g'].shape
            assert all(torch.isfinite(t).all() for t in (v['x'],v['K'],v['y']))
            for item in v['adjoints'].values():assert item['g'].dtype==torch.float32 and torch.isfinite(item['g']).all()
        a=torch.load(source/backend/'TRAINING.pt',map_location='cpu',weights_only=False)
        r=torch.load(OLD/f'results/a100-119167/train-{oldname}-gate/OBJECTS.pt',map_location='cpu',weights_only=False)
        metrics=full({k:a[k] for k in r if k in a},{k:r[k] for k in r if k in a})
        failed={k:v for k,v in metrics.items() if not v['passed']}
        assert not failed,(backend,failed)
        save(out/(backend+'-METRICS.json'),metrics)
        results[backend]={'site_count':len(rows),'actual_rows':10504,'capture_vs_original_failures':len(failed),
            'role_count':len(metrics),'site_phase_coverage':{row['name']:row['adjoint_phases'] for row in rows}}
        del a,r
    save(out/'RESULT.json',{'status':'REAL_TRAINING_CAPTURE_AUDIT_PASS','node':platform.node(),'results':results,
        'precision':'energy float64; other tensors/operations/statistics float32; integer ULP descriptors',
        'capture_sha256':sha(source/'CAPTURE.json')})
    (ROOT/'CAPTURE-AUDIT.txt').write_text(str(out)+'\n');print(json.dumps(results),flush=True)
if __name__=='__main__':main()
