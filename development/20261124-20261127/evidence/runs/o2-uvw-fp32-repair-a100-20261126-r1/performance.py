"""Fresh-process paired whole-network timing/memory, only after AMD gates pass."""
import argparse,gc,importlib.util,json,os,statistics,sys,time
from pathlib import Path
import torch
from common import ROOT,SOURCE,OLD,configure,factory,common_state,restore,startup,save,sha
from ictc_o2_hp import uvw_triton as operator
def main(args):
    configure();out=Path(args.output);out.mkdir(parents=True)
    expected=sha(SOURCE/'ictc_o2_hp/uvw_triton.py')
    for pointer,status in (('REGRESSION-AUDIT.txt','ACTUAL_LOCAL_REGRESSION_PASS'),('GENERIC-AUDIT.txt','GENERIC_REGRESSION_PASS'),('VERIFY-AUDIT.txt','WHOLE_TRAINING_GATE_PASS')):
        gate=json.loads((Path((ROOT/pointer).read_text().strip())/'RESULT.json').read_text())
        assert gate['status']==status and gate['operator_source_sha256']==expected
    model=factory('v3','native' if args.backend=='native' else 'candidate','checkpoint_ema','cuda')
    executed_source=expected
    if args.backend=='native':
        executed_source=sha(ROOT.parents[1]/'architectures/equiformerv3/source/canonical-cpu-control-c14109a12653/canonical.py')
    if args.backend=='v1':
        path=OLD/'snapshots/preflight-119174/fork-src/ictc_o2_hp/uvw_triton.py'
        assert sha(path)=='e11374ef4ad0de741a93c61e7367853479dbe6b5644097ef57665e436d4a773f'
        spec=importlib.util.spec_from_file_location('performance_v1_operator',path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        operator.channel_matmul=module.channel_matmul
        executed_source=sha(path)
    state,receipt=common_state(args.mode)
    condition=startup(model,state,receipt,args.mode)
    condition.update(mode=args.mode,backend=args.backend,action=args.action,qualified_candidate_sha256=expected,
        executed_backend_source_sha256=executed_source,
        numerical_status='FROZEN_V1_EF_PASS_TRAIN_FAIL' if args.backend=='v1' else 'WHOLE_GATE_PASS',
        timing_boundary='synchronized wall clock including graph/transfer and original full optimizer/EMA/scheduler step',
        common_state_restore_outside_measurement=True,training_loss='unchanged actual own native loss cotangents')
    call=model.infer if args.mode=='ef' else model.train_step
    restore(model,state,receipt);checked=call()
    values=checked.values() if isinstance(checked,dict) else (checked,)
    assert all(torch.isfinite(v).all() for v in values)
    if args.mode=='train':
        active=[p.grad for p in model.model.parameters() if p.grad is not None]
        assert active and all(torch.isfinite(v).all() for v in active)
        condition['finite_precheck_gradient_tensors']=len(active)
        condition['finite_precheck_gradient_bytes']=sum(v.numel()*v.element_size() for v in active)
        del active
    assert model.original_ids=={n:id(p) for n,p in model.model.named_parameters()}
    condition['finite_precheck']=True;condition['original_parameter_ids_unchanged']=True
    condition['diagnostic_tensor_references_released_before_warmup']=True
    save(out/'CONDITION.json',condition);del checked,values
    for _ in range(3):restore(model,state,receipt);call()
    if args.action=='time':
        values=[]
        for _ in range(10):
            restore(model,state,receipt);torch.cuda.synchronize();started=time.perf_counter()
            call();torch.cuda.synchronize();values.append(time.perf_counter()-started)
        result={'times_s':values,'median_s':statistics.median(values),'warmup':3,'samples':10}
    elif args.action=='memory':
        restore(model,state,receipt);gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        base={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
        call();torch.cuda.synchronize()
        result={'base_bytes':base,'peak_bytes':{'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()},
            'measurement':'fresh process, warmed kernels, PyTorch allocator peak; includes resident common model/optimizer/EMA state'}
    else:
        from diagnose import annotations
        restore(model,state,receipt);torch.cuda.synchronize()
        with annotations(model):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,profile_memory=True,with_stack=False) as prof:
                with torch.profiler.record_function('qualified_'+args.mode):call()
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(out/'trace.json'))
        (out/'OPERATOR-TABLE.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=60))
        result={'trace_steps':1,'measurement':'single complete step from identical warmed common state; trace GPU kernel sums are separate from wall clock'}
    save(out/'RESULT.json',{'status':'QUALIFIED_CANDIDATE_PAIRED_MEASUREMENT_COMPLETED',**result})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('ef','train'));p.add_argument('backend',choices=('native','v1','candidate'))
    p.add_argument('action',choices=('time','memory','trace'));p.add_argument('output');main(p.parse_args())
