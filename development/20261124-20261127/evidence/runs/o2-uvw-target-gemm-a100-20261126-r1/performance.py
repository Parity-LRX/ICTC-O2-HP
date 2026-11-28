"""Fresh-process synchronized timing/memory only for the AMD-qualified source."""
import argparse,gc,json,os,statistics,time
from pathlib import Path
import torch
from context import ROOT,SOURCE,configure,factory,common_state,restore,startup,save,sha,rng,rng_id
from ictc_o2_hp import uvw_triton as operator
from candidate_sources import qualified_sources,load_module
def main(args):
    configure();out=Path(args.output);out.mkdir(parents=True);expected=sha(SOURCE/'ictc_o2_hp/uvw_triton.py')
    gate=json.loads((Path((ROOT/'FULL-AUDIT.txt').read_text().strip())/'RESULT.json').read_text())
    assert gate['status']=='FULL_QUALIFICATION_PASS' and gate['operator_source_sha256']==expected
    sources=qualified_sources()
    model=factory('v3','native' if args.backend=='native' else 'candidate','checkpoint_ema','cuda');executed=expected
    if args.backend=='native':executed=sha(ROOT.parents[1]/'architectures/equiformerv3/source/canonical-cpu-control-c14109a12653/canonical.py')
    if args.backend in ('c05','mask'):
        r=sources[args.backend];module=load_module('performance_'+args.backend+'_operator',r['path'])
        operator.channel_matmul=module.channel_matmul;executed=r['sha256']
    state,receipt=common_state(args.mode);condition=startup(model,state,receipt,args.mode)
    condition.update(mode=args.mode,backend=args.backend,action=args.action,qualified_candidate_sha256=expected,
        qualified_package_sha256=gate['package_sha256'],
        executed_backend_source_sha256=executed,numerical_status='FULL_QUALIFICATION_PASS',
        source_AMD_gate=sources.get(args.backend,{}).get('gate','native reference'),
        timing_boundary='synchronized wall clock including graph/transfer and original full optimizer/EMA/scheduler step',
        common_state_restore_outside_measurement=True,training_loss='unchanged actual own native loss cotangents')
    call=model.infer if args.mode=='ef' else model.train_step
    restore(model,state);condition['RNG_at_measurement_start']=rng_id(rng());checked=call();values=checked.values() if isinstance(checked,dict) else (checked,)
    assert all(torch.isfinite(v).all() for v in values)
    if args.mode=='train':
        active=[p.grad for p in model.model.parameters() if p.grad is not None]
        assert len(active)==292 and all(torch.isfinite(v).all() for v in active);del active
    assert model.original_ids=={n:id(p) for n,p in model.model.named_parameters()}
    condition.update(finite_precheck=True,original_parameter_ids_unchanged=True,diagnostic_tensor_references_released_before_warmup=True)
    save(out/'CONDITION.json',condition);del checked,values
    for _ in range(3):restore(model,state);call()
    if args.action=='time':
        samples=[]
        for _ in range(5):
            restore(model,state);assert rng_id(rng())==condition['RNG_at_measurement_start'];torch.cuda.synchronize();started=time.perf_counter()
            call();torch.cuda.synchronize();samples.append(time.perf_counter()-started)
        result={'times_s':samples,'median_s':statistics.median(samples),'warmup':3,'samples':5}
    elif args.action=='memory':
        restore(model,state);gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        base={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()};call();torch.cuda.synchronize()
        result={'base_bytes':base,'peak_bytes':{'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()},
            'measurement':'fresh process warmed kernels; PyTorch allocator peak includes model/optimizer/EMA'}
    else:
        from diagnose import annotations
        restore(model,state);torch.cuda.synchronize()
        with annotations(model):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True,with_stack=False) as prof:
                with torch.profiler.record_function('qualified_'+args.mode):call()
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(out/'trace.json'));(out/'OPERATOR-TABLE.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=60))
        result={'trace_steps':1,'measurement':'one complete step from common state; kernel sum separate from wall'}
    save(out/'RESULT.json',{'status':'QUALIFIED_CANDIDATE_PAIRED_MEASUREMENT_COMPLETED',**result})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('ef','train'));p.add_argument('backend',choices=('native','c05','mask','candidate'))
    p.add_argument('action',choices=('time','memory','trace'));p.add_argument('output');main(p.parse_args())
