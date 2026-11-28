import argparse,gc,json,os,statistics,time,traceback
from pathlib import Path
from common import ROOT,torch,configure,operator,network,cpu,restore,rng,set_rng,hardware,save
def main(a):
    configure(True);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    stage='initialize';record={'job':os.environ['SLURM_JOB_ID'],'configuration':vars(a),'hardware':hardware()}
    try:
        if a.level=='operator':
            fn,inputs,parameters,select,identity=operator(a.family,a.order,a.parity,a.size);select('candidate')
            if a.mode=='infer':
                def call():
                    with torch.no_grad():return fn(*inputs)
            else:
                shape=(a.size,identity['output_width']) if a.family=='uvw' else tuple(inputs[0].shape)
                cot=torch.randn(shape,device='cuda',dtype=torch.float32)
                def call():
                    y=fn(*inputs);g=torch.autograd.grad(y,inputs+parameters,cot);return (y,)+g
            restore_state=lambda:None
            units=a.size;record['identity']=identity
        else:
            model=network(a.order,'candidate',a.size);state_path=ROOT/'gates'/f'network-L{a.order}'/('INFER.pt' if a.mode=='infer' else 'WARM.pt')
            state=torch.load(state_path,map_location='cpu',weights_only=False);restore_state=lambda:restore(model,state)
            restore_state();call=model.infer if a.mode=='infer' else model.train_step
            units=100*a.size;record['identity']=model.identity
            batch=model.batch('train')
            with torch.no_grad():
                graph=model.model.generate_graph(batch,enforce_max_neighbors_strictly=model.model.enforce_max_neighbors_strictly,use_pbc_single=model.model.use_pbc_single)
            record['actual_edges']=int(graph[0].shape[1]);assert record['actual_edges']==5252*a.size
            assert bool((batch.batch[graph[0][0]]==batch.batch[graph[0][1]]).all())
            del graph,batch
        record['units']=units;save(out/'START.json',record)
        stage='compile_and_warmup'
        warmups=1 if a.action=='capacity' else 2
        for _ in range(warmups):restore_state();value=call();del value
        torch.cuda.synchronize();restore_state();gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        stage='measurement';start=time.perf_counter();value=call();torch.cuda.synchronize();elapsed=time.perf_counter()-start
        peak={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
        leaves=[]
        def finite(v):
            if isinstance(v,torch.Tensor):leaves.extend(bool(torch.isfinite(part).all()) for part in v.detach().flatten().split(262144))
            elif isinstance(v,(tuple,list)):list(map(finite,v))
            elif isinstance(v,dict):list(map(finite,v.values()))
        finite(value);assert leaves and all(leaves)
        if a.level=='network' and a.mode=='train':
            active=[p.grad for p in model.model.parameters() if p.requires_grad]
            assert len(active)==record['identity']['active_parameters'] and all(g is not None and torch.isfinite(g).all() for g in active)
        record.update(status='PASS',elapsed_s=elapsed,throughput=units/elapsed,peak_bytes=peak,finite=True,warmups=warmups,
                      measurement='one synchronized call in fresh process after warmup; state restore excluded')
        del value
    except Exception as e:
        oom=isinstance(e,torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower()
        record.update(status='OOM' if oom else 'ERROR',stage=stage,error=repr(e),traceback=traceback.format_exc(),
                      failure_peak_bytes={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()})
    save(out/'RESULT.json',record);print(json.dumps({k:record[k] for k in ('status','configuration','stage','error','elapsed_s','peak_bytes') if k in record}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--level',choices=['operator','network'],required=True);p.add_argument('--family',default='uvw');p.add_argument('--order',type=int,required=True);p.add_argument('--parity',default='canonical');p.add_argument('--mode',choices=['infer','train'],required=True);p.add_argument('--size',type=int,required=True);p.add_argument('--action',choices=['timing','capacity'],default='timing');p.add_argument('--output',required=True)
    main(p.parse_args())
