import argparse,gc,json,os,traceback
from pathlib import Path
from common import ROOT,OLD,torch,whole,configure,operator,network,cpu,rng,set_rng,save,sha,hardware,restore
def main(a):
    configure(True);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);record={'hardware':hardware(),'configuration':vars(a),'files':{}}
    def store(name,v):
        p=out/name;torch.save(v,p);record['files'][name]=sha(p)
    try:
        if a.level=='operator':
            fn,inputs,parameters,select,identity=operator(a.family,a.order,a.parity,a.size);record['identity']=identity
            active=inputs+parameters
            select('native')
            with torch.no_grad():g=(torch.randn_like(fn(*inputs))*.1).requires_grad_()
            directions=tuple(torch.randn_like(v)*.1 for v in active)
            for backend in ('native','native-repeat','candidate'):
                select('native' if backend.startswith('native') else 'candidate')
                if a.mode=='infer':
                    with torch.no_grad():y=fn(*inputs)
                    values={'output':cpu(y)};first=None
                else:
                    y=fn(*inputs);first=torch.autograd.grad(y,active,g,create_graph=a.size==16)
                    values={'output':cpu(y),'first':cpu(first)}
                if a.mode=='train' and a.size==16:
                    scalar=sum((v*d).sum() for v,d in zip(first,directions))
                    values['mixed']=cpu(torch.autograd.grad(scalar,active+(g,),allow_unused=True))
                store(backend+'.pt',values);del y,first,values;gc.collect();torch.cuda.empty_cache()
        else:
            model=network(a.order,a.backend,1);record['identity']=model.identity
            store('INITIAL.pt',model.state())
            initial_rng=rng();store('joint-vjp.pt',cpu(model.vjp()));set_rng(initial_rng)
            gc.collect();torch.cuda.empty_cache()
            whole.seed();store('ef.pt',cpu(model.infer()))
            # The EF diagnostic's RNG consumption is shared by protocol, not the true loss cotangent.
            record['phases']=[]
            for step in range(3):
                # Native _backward zeroes the same gradients before use. They
                # are not read by forward/mixed diagnostics; release them now.
                model.optimizer.zero_grad(set_to_none=True)
                gc.collect();torch.cuda.empty_cache()
                before=rng();mixed=model.force_mixed()
                record['phases'].append({'step':step,'phase':'mixed','allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated()})
                # Release only unreferenced diagnostic allocations, both backends.
                # Model, optimizer, EMA, scheduler and RNG continue uninterrupted.
                gc.collect();torch.cuda.empty_cache()
                training=model.train_step(capture=True)
                record['phases'].append({'step':step,'phase':'training','allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated()})
                after=cpu(model.forward(model.batch('train'),False))
                store(f'step{step}.pt',{'mixed':mixed,'training':training,'after_output':after,'scheduler':cpu(model.trainer.scheduler.scheduler.state_dict())})
                store(f'rng{step}.pt',{'before':before,'after':rng()})
                del mixed,training,after;gc.collect();torch.cuda.empty_cache();print('STEP',a.order,a.backend,step,flush=True)
            store('WARM.pt',model.state())
        record['status']='CAPTURED'
    except Exception as e:
        record.update(status='OOM' if isinstance(e,torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower() else 'ERROR',error=repr(e),traceback=traceback.format_exc())
    save(out/'RESULT.json',record);print(json.dumps({k:record[k] for k in ('status','configuration','error') if k in record}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--level',choices=['operator','network'],required=True);p.add_argument('--family',default='uvw');p.add_argument('--order',type=int,required=True);p.add_argument('--parity',default='canonical');p.add_argument('--size',type=int,default=16);p.add_argument('--backend',default='candidate');p.add_argument('--mode',default='train');p.add_argument('--output',required=True)
    main(p.parse_args())
