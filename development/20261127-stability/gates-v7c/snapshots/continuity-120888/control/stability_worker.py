"""Bounded train-only continuity and repeated inference after each backend's own updates."""
import argparse,gc,json,os,traceback
from pathlib import Path
from common import *
from continuity import signature
def main(a):
    configure(True);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    record={'hardware':hardware(),'configuration':vars(a),'files':{},'rows':[]}
    def store(name,value):
        path=out/name;torch.save(value,path);record['files'][name]=sha(path)
    try:
        model=network(a.order,'native' if a.backend.startswith('native') else 'candidate',1)
        record['identity']=model.identity
        if a.mode=='train':
            whole.seed();initial=model.state();store('INITIAL.pt',initial)
            previous_state=signature(initial);previous_rng=None;del initial
            for step in range(a.steps):
                model.optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                before_rng=rng();before=signature(model.state())
                assert before==previous_state,'Training state changed between steps'
                if previous_rng is not None:assert signature(before_rng)==previous_rng,'RNG changed between steps'
                result=model.train_step(capture=True)
                assert torch.isfinite(result['loss']).all(),'Nonfinite true training loss'
                state=model.state();after=signature(state);after_rng=rng()
                store(f'step{step}.pt',{'training':result,'scheduler':cpu(model.trainer.scheduler.scheduler.state_dict())})
                store(f'rng{step}.pt',{'before':before_rng,'after':after_rng})
                record['rows'].append({'step':step,'before_state':before,'after_state':after,
                    'before_rng':signature(before_rng),'after_rng':signature(after_rng),
                    'loss':float(result['loss']),'allocated':torch.cuda.memory_allocated(),
                    'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated()})
                previous_state=after;previous_rng=signature(after_rng)
                save(out/'RESULT.json',{**record,'status':'TRAINING'});print('CONTINUOUS_STEP',a.order,a.backend,step,float(result['loss']),flush=True)
                if step==a.steps-1:store('WARM.pt',state)
                del state,result;gc.collect()
        else:
            warm=Path(a.warm);record['trained_state_sha256']=sha(warm)
            state=torch.load(warm,map_location='cpu',weights_only=False)
            model.model.load_state_dict(state['model'],strict=True);del state
            record['inference_parameter_role']='own raw model after 20 true-loss updates; not EMA-applied weights'
            whole.seed();parameters=signature(dict(model.model.named_parameters()))
            for call in range(a.calls):
                before=rng();value=cpu(model.infer());after=rng()
                store(f'call{call}.pt',value);store(f'rng{call}.pt',{'before':before,'after':after})
                del value;gc.collect()
                record['rows'].append({'call':call,'before_rng':signature(before),'after_rng':signature(after),
                    'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
                    'peak_allocated':torch.cuda.max_memory_allocated()})
                print('REPEATED_INFER',a.order,a.backend,a.repeat,call,flush=True)
            assert signature(dict(model.model.named_parameters()))==parameters,'Inference modified model parameters'
            assert model.original_ids=={n:id(p) for n,p in model.model.named_parameters()}
            record['parameters_unchanged']=True
        record['status']='CAPTURED'
    except Exception as error:
        record.update(status='OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else 'ERROR',error=repr(error),traceback=traceback.format_exc())
    save(out/'RESULT.json',record)
    print('STABILITY_WORKER',a.order,a.mode,a.backend,record['status'],flush=True)
    assert record['status']=='CAPTURED',record.get('error')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--order',type=int,required=True);parser.add_argument('--backend',required=True)
    parser.add_argument('--mode',choices=['train','infer'],required=True);parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--calls',type=int,default=10);parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--warm');parser.add_argument('--output',required=True);main(parser.parse_args())
