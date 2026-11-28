"""Paired diagnostics from identical saved states; numerical FAIL never gates profiling."""
import argparse
import ast
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch
import whole
from whole import ROOT,WS,NET,configure,cpu,factory,load,save,seed,sha

class OriginalV3(whole.V3):
    def __init__(self,device,recompute):
        sys.path[:0]=[str(NET/p) for p in ('fixture-api','upstream/src','upstream','env-vendor','deps')]
        native=load('original_native',NET/'fixture-api/whole_network.py');native.strict(torch)
        net=native.Network('native','cpu')
        checkpoint=torch.load(native.CHECKPOINT,map_location='cpu',weights_only=False)
        source=WS/'architectures/equiformerv3/source/canonical-cpu-control-c14109a12653/prior_audit.py'
        function=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='load_parameter_role')
        namespace={'torch':torch,'hashlib':hashlib,'json':json}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
        role=namespace['load_parameter_role'](net,checkpoint,'checkpoint_ema')
        self.model=net.model.to(device)
        if recompute:self.model.gradient_checkpointing_block_list=[1]*self.model.num_layers
        self.trainer=net.trainer;self.trainer.model=self.model
        self.trainer.cpu=device=='cpu';self.trainer.device=torch.device(device)
        from fairchem.core.modules.scheduler import CosineLRLambda
        schedule=self.trainer.config['optim']['scheduler_params']
        self.trainer.config['optim']['lr_lambda']=CosineLRLambda(**{k:schedule[k] for k in ('warmup_epochs','warmup_factor','epochs','lr_min_factor')})
        self.trainer.load_loss();self.trainer.load_optimizer();self.trainer.load_extras()
        for obj in list(self.trainer.normalizers.values())+list(self.trainer.elementrefs.values()):obj.to(device)
        self.net=net;self.device=device;self.optimizer=self.trainer.optimizer;self.ema=self.trainer.ema
        self.original_ids={n:id(p) for n,p in self.model.named_parameters()}
        self.frames=torch.load(ROOT/'prepared/FRAMES.pt',weights_only=True)
        self.patches=None
        self.identity={'architecture':'upstream_equiformerv3_unprojected','checkpoint':str(native.CHECKPOINT),
            'checkpoint_sha256':native.CHECKPOINT_SHA,'parameter_role':role,
            'memory_policy':'native nonreentrant block recomputation' if recompute else 'upstream checkpoint default',
            'gradient_checkpointing':self.model.gradient_checkpointing_block_list,
            'training_recipe':self.trainer.config['optim'],'loss':self.trainer.config['loss_functions'],
            'comparison':'absolute upstream architecture control; different mathematical model from canonical projection'}

def tensor_sha(value):
    value=value.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()

def parameters(model):
    rows={n:{'sha256':tensor_sha(p),'shape':list(p.shape),'dtype':str(p.dtype),'requires_grad':p.requires_grad}
          for n,p in model.model.named_parameters()}
    return {'sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),'tensors':rows,
            'count':len(rows),'numel':sum(p.numel() for p in model.model.parameters())}

def flags(model):
    assert torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0)
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert not torch.is_autocast_enabled('cuda') and torch.get_float32_matmul_precision()=='highest'
    assert not torch._inductor.config.triton.cudagraphs
    assert all(p.dtype==torch.float32 for p in model.model.parameters() if p.is_floating_point())
    properties=torch.cuda.get_device_properties(0)
    return {'node':os.environ['SLURM_JOB_NODELIST'],'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
       'gpu':properties.name,'gpu_uuid':str(properties.uuid),'gpu_memory_bytes':properties.total_memory,
       'torch':torch.__version__,'deterministic':torch.are_deterministic_algorithms_enabled(),
       'TF32':False,'AMP':False,'CUDA_Graph':False,'threads':torch.get_num_threads(),
       'env':{k:os.environ.get(k) for k in ('TACE_USE_EQX','TACE_USE_COMPILE','TACE_USE_CUE','TACE_USE_OEQ','TACE_USE_TF32','EQX_USE_CUDA_GRAPH')},
       'use_eqx_modules':{n:bool(m.use_eqx) for n,m in model.model.named_modules() if hasattr(m,'use_eqx')},
       'non_fp32_buffers':{n:str(b.dtype) for n,b in model.model.named_buffers() if b.is_floating_point() and b.dtype!=torch.float32}}

def state_identity(state):
    def tree(value):
        if isinstance(value,torch.Tensor):return {'dtype':str(value.dtype),'shape':list(value.shape),'sha256':tensor_sha(value)}
        if isinstance(value,dict):return {str(k):tree(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [tree(v) for v in value]
        if value is None or isinstance(value,(str,int,float,bool)):return value
        return repr(value)
    return {name:hashlib.sha256(json.dumps(tree(value),sort_keys=True).encode()).hexdigest() for name,value in state.items()}

def graph(model,mode):
    batch=model.batch(mode)
    if isinstance(batch,dict):
        fields={k:batch[k] for k in ('edge_index','edge_shifts','lattice','atomic_numbers','ptr') if k in batch}
        labels={k:batch[k] for k in ('energy','forces','stress','energy_weight','forces_weight','stress_weight') if k in batch}
        edges=batch['edge_index'].shape[1]
    else:
        fixed,hashes=model.net.topology(batch)
        fields={k:getattr(fixed,k) for k in ('edge_index','cell_offsets','neighbors','atomic_numbers','natoms')}
        labels={k:getattr(batch,k) for k in ('energy','forces','stress','fixed') if hasattr(batch,k)}
        edges=fixed.edge_index.shape[1]
    return {'edges':edges,'atoms':len(model.frames[mode]['z']),
            'graph':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':tensor_sha(v)} for k,v in fields.items()},
            'labels':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':tensor_sha(v)} for k,v in labels.items()}},cpu(fields)

def finite_check(model,mode,result):
    values=result.values() if isinstance(result,dict) else (result,)
    for value in values:assert torch.isfinite(value).all(),'Nonfinite output/loss'
    active=[];inactive=[]
    if mode=='train':
        for name,p in model.model.named_parameters():
            if p.grad is None:inactive.append(name)
            else:
                assert torch.isfinite(p.grad).all(),name
                active.append(name)
        assert active,'No active parameter gradients in a complete training step'
    return {'finite':True,'active_gradient_names':active,'inactive_gradient_names':inactive}

@contextlib.contextmanager
def annotations(model):
    """Trace-only annotations; no change to executed tensor operations."""
    hooks=[];saved=[]
    def method(obj,name,label):
        original=getattr(obj,name)
        def wrapper(*args,**kwargs):
            with torch.profiler.record_function(label):return original(*args,**kwargs)
        setattr(obj,name,wrapper);saved.append((obj,name,original))
    method(model,'batch','phase:graph_build_transfer')
    method(model,'forward','phase:model_forward')
    method(model,'loss','phase:loss')
    original_grad=torch.autograd.grad;original_backward=torch.Tensor.backward
    backward_depth=[0]
    def gradient(*args,**kwargs):
        if backward_depth[0]:return original_grad(*args,**kwargs)
        with torch.profiler.record_function('phase:force_from_energy'):return original_grad(*args,**kwargs)
    def backward(*args,**kwargs):
        backward_depth[0]+=1
        try:
            with torch.profiler.record_function('phase:loss_backward'):return original_backward(*args,**kwargs)
        finally:backward_depth[0]-=1
    torch.autograd.grad=gradient;torch.Tensor.backward=backward
    saved.extend([(torch.autograd,'grad',original_grad),(torch.Tensor,'backward',original_backward)])
    method(model.optimizer,'step','phase:optimizer')
    method(model.optimizer,'zero_grad','phase:zero_grad')
    if model.ema:method(model.ema,'update','phase:EMA')
    if hasattr(model,'trainer'):
        method(model.trainer.scheduler,'step','phase:scheduler')
        method(torch.nn.utils,'clip_grad_norm_','phase:clip')
    else:
        if model.clip_function is not None:method(model,'clip_function','phase:clip')
        else:method(torch.nn.utils,'clip_grad_norm_','phase:clip')
    def stage(name):
        name='.'+name
        if 'so2_m_linear' in name:return 'SO2_positive_m'
        if 'so2_linear_' in name:return 'SO2_linear'
        if name.endswith('.ece') or name.endswith('.contraction'):return 'local_O2_ECE'
        if name.endswith('.rad_func') or name.endswith('.edge_info'):return 'radial_MLP'
        if name.endswith('.radial_basis'):return 'radial_basis'
        if name.endswith('.ga.act'):return 'spherical_activation'
        if name.endswith('.ffn'):return 'FFN'
        if name.endswith('edge_degree_embedding'):return 'edge_degree_embedding'
        if name.endswith('node_embedding') or name.endswith('edge_embedding'):return 'embedding'
        if '.edge_updates.' in name and name.rsplit('.',1)[-1].isdigit():return 'edge_update'
        if '.interactions.' in name and name.rsplit('.',1)[-1].isdigit():return 'interaction_other'
        if '.products.' in name and name.rsplit('.',1)[-1].isdigit():return 'node_product'
        if name.endswith('.ga'):return 'attention_other'
        if name.endswith(('norm','norm_1','norm_2','alpha_norm','scatter_norm')):return 'normalization'
        if name.endswith(('energy_block','atomic_energy_layer','scale_shift')) or ('.energy_readouts.' in name and name.split('.energy_readouts.',1)[1].isdigit()):return 'readout'
        if name.endswith(('query_proj','key_proj','radial_proj')):return 'RRA_query_key_phase'
        if name.endswith('linear_coefs'):return 'edge_coefficients'
        if name.endswith(('linear_up','linear_down')):return 'local_linears'
        return None
    for name,module in model.model.named_modules():
        label=stage(name)
        if label is None or isinstance(module,torch.jit.ScriptModule):continue
        stack=[]
        def before(m,args,_stack=stack,_label=label,_name=name):
            entry=torch.profiler.record_function('stage:'+_label+'|'+_name);entry.__enter__();_stack.append(entry)
        def after(m,args,output,_stack=stack):
            if _stack:_stack.pop().__exit__(None,None,None)
        hooks.append(module.register_forward_pre_hook(before))
        hooks.append(module.register_forward_hook(after,always_call=True))
    if hasattr(model.model,'generate_graph'):method(model.model,'generate_graph','stage:graph_build_device|generate_graph')
    if hasattr(model.model,'_forward_edge'):method(model.model,'_forward_edge','stage:geometry_rotation|forward_edge')
    try:yield
    finally:
        for hook in hooks:hook.remove()
        for obj,name,original in reversed(saved):setattr(obj,name,original)

def main(args):
    os.environ['EQX_USE_CUDA_GRAPH']='0'
    configure();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    model=OriginalV3('cuda',args.backend.endswith('recompute')) if args.route=='v3-original' else factory(args.route,'candidate' if args.backend in ('ictc','ictc-eqx') else 'native','checkpoint_ema','cuda')
    if args.backend=='native-eqx-module':
        import types
        from adapter.install import Patches,_rra_messages
        route_patches=Patches()
        for name,module in model.model.named_modules():
            if type(module).__name__=='UvSO2Interaction' and hasattr(getattr(module,'rejector',None),'ece'):
                route_patches.set(module,'_compute_messages',types.MethodType(_rra_messages,module))
                route_patches.sites.append(name)
        assert len(route_patches.sites)==4
        model.diagnostic_route_patches=route_patches
        model.identity['diagnostic_native_route']='same unfused RRA module route as ICTC; ECE remains upstream native Eqx'
    if args.route=='tece-full' and args.backend in ('native-eqx-paired','ictc-eqx'):
        selected=[]
        for name,module in model.model.named_modules():
            if type(module).__name__=='EceO2Interaction':
                module.use_eqx=False;selected.append(name)
        assert len(selected)==2
        model.identity['diagnostic_native_route']={'EceO2Interaction_module_route':selected,'other_upstream_acceleration':'retained; same native/candidate'}
    hardware=flags(model)
    common=Path(args.common)
    call=model.infer if args.mode=='ef' else model.train_step
    if args.action=='prepare':
        # A native trajectory creates one common valid steady-training state.
        if args.mode=='train':
            for i in range(3):seed();finite_check(model,args.mode,call())
        state=model.state();torch.save(state,out/'START.pt')
        identity=parameters(model);g,fields=graph(model,args.mode);torch.save(fields,out/'GRAPH.pt')
        save(out/'CONDITION.json',{'route':args.route,'mode':args.mode,'identity':model.identity,
            'hardware':hardware,'parameters':identity,'graph':g,'start_sha256':sha(out/'START.pt'),
            'complete_state_identity':state_identity(state),
            'warm_state':'3 actual native training steps' if args.mode=='train' else 'exact initial checkpoint or declared random initialization',
            'source_checkpoint_unchanged':args.mode=='ef','seed':20261124})
        save(out/'RESULT.json',{'status':'COMMON_STATE_PREPARED'});return
    condition=json.loads((common/'CONDITION.json').read_text())
    assert sha(common/'START.pt')==condition['start_sha256']
    state=torch.load(common/'START.pt',map_location='cpu',weights_only=False,mmap=True)
    expected_state=state_identity(state)
    def restore():
        # PyTorch load_state_dict retains CPU Adam step tensors by reference.
        # This copy prevents measured updates from mutating the common template.
        model.restore(dict(state,optimizer=copy.deepcopy(state['optimizer'])))
    restore();identity=parameters(model)
    assert identity['sha256']==condition['parameters']['sha256'],'Paired starting parameters differ'
    restored_state=state_identity(model.state())
    assert restored_state==expected_state,('Complete restored state differs',restored_state,expected_state)
    g,_=graph(model,args.mode)
    assert g==condition['graph'],'Paired graphs or label tensors differ'
    coverage=[]
    if args.backend=='ictc-eqx':
        for runtime in model.patches.runtime:
            coverage.extend(row[-1] for row in getattr(runtime,'rows',[]))
            coverage.extend(getattr(runtime,'pairs',{}).values())
        assert coverage,'No candidate coupling runtime to audit'
        for pair in coverage:pair.collect_coverage=True;pair.kernel_calls=0
    seed();check=finite_check(model,args.mode,call())
    if coverage:
        check['ictc_forward_pair_calls']=sum(pair.kernel_calls for pair in coverage)
        assert check['ictc_forward_pair_calls']>0,'The accelerated upstream route bypassed ICTC'
        for pair in coverage:pair.collect_coverage=False
    restore()
    assert state_identity(state)==expected_state,'Common CPU template was mutated'
    save(out/'CONDITION.json',{'route':args.route,'mode':args.mode,'backend':args.backend,'action':args.action,
        'identity':model.identity,'hardware':hardware,'parameter_sha256':identity['sha256'],
        'common_state_sha256':condition['start_sha256'],'graph':g,'execution':check,
        'complete_state_identity':restored_state,'complete_restored_state_verified':True,
        'optimizer_restore':'independent deepcopy; no shared CPU Adam step tensors',
        'numerical_qualification':'DIAGNOSTIC_ONLY; previous FAIL/HOLD retained',
        'same_state_each_measured_step':True,'state_restore_outside_measurement':True})
    for _ in range(3):restore();call()
    restore();torch.cuda.synchronize()
    if args.action=='time':
        values=[]
        for _ in range(10):
            restore();torch.cuda.synchronize();start=time.perf_counter()
            call();torch.cuda.synchronize();values.append(time.perf_counter()-start)
        result={'status':'DIAGNOSTIC_COMPLETED','times_s':values,'median_s':statistics.median(values),
                'warmup':3,'measured':10,'measurement':'synchronized wall clock, full graph/transfer/forward and native full step'}
    elif args.action=='memory':
        restore();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        base={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
        call();torch.cuda.synchronize()
        result={'status':'DIAGNOSTIC_COMPLETED','base_bytes':base,
                'peak_bytes':{'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()},
                'measurement':'fresh process; warmed code; common starting model/optimizer/EMA state resident'}
    elif args.action=='trace':
        restore()
        with annotations(model):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,profile_memory=True,with_stack=False) as prof:
                with torch.profiler.record_function('diagnostic_'+args.mode):call()
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(out/'trace.json'))
        (out/'OPERATOR-TABLE.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=60))
        result={'status':'DIAGNOSTIC_COMPLETED','trace_steps':1,
                'measurement':'one complete step from the identical warmed common state; no optimizer cold initialization'}
    else:raise ValueError(args.action)
    assert state_identity(state)==expected_state,'Saved common state drifted during measurement'
    save(out/'RESULT.json',dict(result,job=os.environ['SLURM_JOB_ID'],common_template_unchanged=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('route','mode','backend','action','common','output'):parser.add_argument(name)
    arguments=parser.parse_args()
    try:main(arguments)
    except BaseException as error:
        out=Path(arguments.output);out.mkdir(parents=True,exist_ok=True)
        save(out/'ERROR.json',{'status':'EXECUTION_ERROR','type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        traceback.print_exc();raise
