"""Independent full-array FP32 audits of continuous training and repeated inference."""
import argparse,gc,json,os,traceback
from pathlib import Path
from common import ROOT,torch,save,sha
from metrics import full
from continuity import signature
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);args=parser.parse_args()
manifest=json.loads(Path(args.manifest).read_text());order=manifest['order']
out=ROOT/'stability-audits'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True)
summaries=[];records={}
def record(folder):
    folder=Path(folder)
    if str(folder) not in records:
        r=json.loads((folder/'RESULT.json').read_text());assert r['status']=='CAPTURED',(folder,r['status'])
        assert r['identity']['source_package_sha256']==manifest['source_package_sha256']
        records[str(folder)]=r
    return records[str(folder)]
def load(folder,name):
    folder=Path(folder);r=record(folder);assert sha(folder/name)==r['files'][name]
    return torch.load(folder/name,map_location='cpu',weights_only=False)
def compare(a,r,label):
    values=full(a,r);file=out/(label+'.json');save(file,values)
    summary={'label':label,'roles':len(values),'failures':{k:v for k,v in values.items() if not v['passed']},
             'max_relative_l2':max((v.get('relative_l2',0) or 0) for v in values.values()),
             'max_ULP':max(v.get('ulp',{}).get('max_ulp',0) for v in values.values()),'full_metrics':str(file),'sha256':sha(file)}
    summaries.append(summary);print('STABILITY_COMPARE',order,label,len(summary['failures']),summary['max_relative_l2'],flush=True)
result={'order':order,'manifest':args.manifest,'summaries':summaries,'original_thresholds':True,'shared_true_loss_cotangents':False,
        'restored_inside_trajectory':False,'steps':20,'inference_calls_per_process':10,'fresh_inference_processes':3}
try:
    training=manifest['training']
    for backend,folder in training.items():
        r=record(folder);assert len(r['rows'])==20 and r['configuration']['steps']==20
        for previous,current in zip(r['rows'],r['rows'][1:]):
            assert previous['after_state']==current['before_state'],'Interrupted own training state'
            assert previous['after_rng']==current['before_rng'],'Interrupted own RNG'
    reference=load(training['native'],'INITIAL.pt')
    for backend in ('native-repeat','candidate'):
        compare(load(training[backend],'INITIAL.pt'),reference,'initial-state-'+backend)
    del reference;gc.collect()
    for step in range(20):
        reference=load(training['native'],f'step{step}.pt');native_rng=load(training['native'],f'rng{step}.pt')
        for backend in ('native-repeat','candidate'):
            candidate=load(training[backend],f'step{step}.pt');states=load(training[backend],f'rng{step}.pt')
            assert signature(states)==signature(native_rng),'Backend RNG mismatch'
            assert set(candidate['training']['gradients'])==set(reference['training']['gradients'])
            compare(candidate,reference,f'train-step{step}-{backend}');del candidate,states;gc.collect()
        del reference,native_rng;gc.collect()
    inference=manifest['inference']
    for repeat in range(3):
        group=inference[str(repeat)]
        for backend,folder in group.items():
            r=record(folder);assert len(r['rows'])==10 and r['parameters_unchanged']
            assert r['trained_state_sha256']==record(training[backend])['files']['WARM.pt']
        for call in range(10):
            reference=load(group['native'],f'call{call}.pt');native_rng=load(group['native'],f'rng{call}.pt')
            for backend in ('native-repeat','candidate'):
                value=load(group[backend],f'call{call}.pt');states=load(group[backend],f'rng{call}.pt')
                assert signature(states)==signature(native_rng),'Inference RNG mismatch'
                compare(value,reference,f'infer-R{repeat}-call{call}-{backend}');del value,states
            if repeat:
                for backend in ('native','native-repeat','candidate'):
                    compare(load(group[backend],f'call{call}.pt'),load(inference['0'][backend],f'call{call}.pt'),
                            f'fresh-R{repeat}-call{call}-{backend}')
            del reference,native_rng;gc.collect()
    result['status']='PASS' if all(not r['failures'] for r in summaries) else 'FAIL'
except Exception as error:
    result.update(status='ERROR',error=repr(error),traceback=traceback.format_exc())
save(out/'RESULT.json',result)
if result['status']!='PASS':save(Path(manifest['stop_path']),{'order':order,'audit':str(out),'status':result['status']})
print('STABILITY_AUDIT_RESULT',order,result['status'],out,flush=True)
assert result['status']=='PASS'
