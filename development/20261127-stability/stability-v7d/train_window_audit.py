"""Audit each completed training step early; preserve the full campaign audit."""
import argparse,gc,json,os,time,traceback
from pathlib import Path
from common import ROOT,torch,save,sha
from continuity import signature
from metrics import full
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
parser=argparse.ArgumentParser();parser.add_argument('--capture-job',required=True);parser.add_argument('--order',type=int,required=True)
args=parser.parse_args();capture=ROOT/'stability-captures'/args.capture_job
out=ROOT/'training-window-audits'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True)
source=json.loads(os.environ['O2_REPAIR_SOURCE_RECORD'])['package_sha256'];started=time.time();summaries=[]
def record(backend,name):
    folder=capture/f'L{args.order}-train-{backend}'
    while time.time()-started<2400:
        try:r=json.loads((folder/'RESULT.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(3);continue
        assert r['status'] in ('TRAINING','CAPTURED'),(backend,r)
        assert r['identity']['source_package_sha256']==source
        if name in r['files']:
            path=folder/name;assert sha(path)==r['files'][name]
            return torch.load(path,map_location='cpu',weights_only=False)
        time.sleep(3)
    raise TimeoutError((backend,name))
def compare(value,reference,label):
    values=full(value,reference);path=out/(label+'.json');save(path,values)
    failures={k:v for k,v in values.items() if not v['passed']}
    row={'label':label,'roles':len(values),'failures':failures,
         'max_relative_l2':max((v.get('relative_l2',0) or 0) for v in values.values()),
         'full_metrics':str(path),'sha256':sha(path)}
    summaries.append(row);print('EARLY_TRAIN_COMPARE',args.order,label,len(failures),row['max_relative_l2'],flush=True)
    assert not failures,('Original training gate exceeded',label,failures)
result={'order':args.order,'capture_job':args.capture_job,'scope':'Early training-window audit; complete 162-comparison campaign audit still required',
        'source_package_sha256':source,'original_thresholds':True,'summaries':summaries}
try:
    reference=record('native','INITIAL.pt')
    for backend in ('native-repeat','candidate'):compare(record(backend,'INITIAL.pt'),reference,'initial-'+backend)
    del reference;gc.collect()
    for step in range(20):
        reference=record('native',f'step{step}.pt');rng=record('native',f'rng{step}.pt')
        for backend in ('native-repeat','candidate'):
            value=record(backend,f'step{step}.pt');states=record(backend,f'rng{step}.pt')
            assert signature(states)==signature(rng)
            compare(value,reference,f'step{step}-'+backend);del value,states;gc.collect()
        del reference,rng;gc.collect()
    result['status']='PASS'
except AssertionError as error:
    result.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
except Exception as error:
    result.update(status='ERROR',error=repr(error),traceback=traceback.format_exc())
save(out/'RESULT.json',result)
if result['status']!='PASS':
    save(capture/'STOP.json',{'order':args.order,'audit':str(out),'status':result['status'],'phase':'training-window'})
print('EARLY_TRAIN_RESULT',args.order,result['status'],out,flush=True)
assert result['status']=='PASS'
