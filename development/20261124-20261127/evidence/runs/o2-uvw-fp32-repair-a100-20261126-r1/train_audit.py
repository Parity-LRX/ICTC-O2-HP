"""Independent AMD complete-array acceptance of actual-loss training candidates."""
import json,os,platform
from pathlib import Path
import torch
from common import ROOT,save,sha
from metrics import full
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
    source=Path((ROOT/'TRAIN-SCREEN.txt').read_text().strip());finish=json.loads((source/'FINISH.json').read_text())
    out=ROOT/'results'/('train-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    load=lambda case,name:torch.load(source/case/name,map_location='cpu',weights_only=False)
    ref=load('native','TRAINING.pt');mixed_ref=load('native','FORCE.pt');summaries={};passed=[]
    for case in finish['cases']:
        a=load(case,'TRAINING.pt');mixed=load(case,'FORCE.pt')
        metrics=full(a,ref);metrics.update(full(mixed,mixed_ref,'/force_mixed'))
        failed={k:v for k,v in metrics.items() if not v['passed']}
        grads={k:v for k,v in failed.items() if k.startswith('/training/gradients/')}
        higher={k:v for k,v in failed.items() if k.startswith('/force_mixed/mixed/')}
        summaries[case]={'passed':not failed,'failed_roles':len(failed),'parameter_gradient_failures':len(grads),
            'mixed_failures':len(higher),'maximum_relative_l2':max((v.get('relative_l2',0) or 0) for v in metrics.values()),
            'failures':failed,'configs':json.loads((source/case/'CASE.json').read_text())['forward_backward_configs']}
        save(out/(case+'-METRICS.json'),metrics)
        if not failed:passed.append(case)
        save(out/'RUNNING-SUMMARY.json',{k:{n:v for n,v in s.items() if n!='failures'} for k,s in summaries.items()})
        print(case,len(grads),len(higher),len(failed),flush=True)
        del a,mixed
    save(out/'RESULT.json',{'status':'AMD_REAL_LOSS_SCREEN_COMPLETED','node':platform.node(),'source':str(source),
        'passed_cases':passed,'cases':summaries,'gate':'original complete-object component bound and relative-L2; no fixed cotangents',
        'source_hashes':{str(p.relative_to(source)):sha(p) for p in source.rglob('*.pt')}})
    (ROOT/'TRAIN-AUDIT.txt').write_text(str(out)+'\n')
    print(json.dumps({k:{n:v for n,v in s.items() if n!='failures'} for k,s in summaries.items()}),flush=True)
if __name__=='__main__':main()
