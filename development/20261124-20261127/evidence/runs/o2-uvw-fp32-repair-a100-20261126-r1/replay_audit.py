"""AMD same-state gates at every point actually visited by either trajectory."""
import json,os,platform
from pathlib import Path
import torch
from common import ROOT,save,sha
from metrics import full
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
    source=Path((ROOT/'REPLAY-RESULT.txt').read_text().strip());manifest=json.loads((source/'MANIFEST.json').read_text())
    out=ROOT/'results'/('replay-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    summaries={};failures=[]
    for backend,ref_backend in (('native-repeat','native'),('candidate','native'),('candidate-repeat','candidate')):
        summaries[backend]={}
        for name,digest in manifest['cases'][backend].items():
            assert sha(source/backend/name)==digest and sha(source/ref_backend/name)==manifest['cases'][ref_backend][name]
            if name!='EF.pt':
                condition=name.replace('.pt','-CONDITION.json')
                a=json.loads((source/backend/condition).read_text());r=json.loads((source/ref_backend/condition).read_text())
                assert a==r,('Comparison does not start from identical state',backend,name)
            a=torch.load(source/backend/name,map_location='cpu',weights_only=False);r=torch.load(source/ref_backend/name,map_location='cpu',weights_only=False)
            metrics=full(a,r);failed={k:v for k,v in metrics.items() if not v['passed']}
            save(out/(backend+'-'+name.replace('.pt','-METRICS.json')),metrics)
            summaries[backend][name]={'passed':not failed,'role_count':len(metrics),'failures':failed,
                'maximum_relative_l2':max((v.get('relative_l2',0) or 0) for v in metrics.values())}
            failures.extend({'backend':backend,'point':name,'role':k,**v} for k,v in failed.items())
            del a,r;print(backend,name,len(failed),flush=True)
    result={'status':'WHOLE_TRAINING_GATE_PASS' if not failures else 'WHOLE_TRAINING_GATE_FAIL',
        'node':platform.node(),'input_state_conditions_identical':True,'points':manifest['points'],'summaries':summaries,
        'failures':failures,'operator_source_sha256':manifest['operator_source_sha256'],
        'contract':'original same-state gates; five states reached by both actual update trajectories; all objects retained',
        'independent_trajectory_comparison':'separate supplemental119415; later inputs/states differed; raw202 over-limit roles retained'}
    save(out/'RESULT.json',result);(ROOT/'REPLAY-AUDIT.txt').write_text(str(out)+'\n');(ROOT/'VERIFY-AUDIT.txt').write_text(str(out)+'\n')
    print(result['status'],len(failures),flush=True)
if __name__=='__main__':main()
