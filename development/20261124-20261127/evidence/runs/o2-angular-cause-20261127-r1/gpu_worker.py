import argparse,ctypes,gc,json,os,time,traceback
from collections import deque
from support import *
from torch.utils._python_dispatch import TorchDispatchMode

def memory():
    return {'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
            'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
def meta(v):
    if isinstance(v,torch.Tensor):return {'shape':list(v.shape),'stride':list(v.stride()),'dtype':str(v.dtype),'bytes':v.numel()*v.element_size(),
                                         'storage_address':v.untyped_storage().data_ptr(),'storage_bytes':v.untyped_storage().nbytes()}
    if isinstance(v,(tuple,list)):return [meta(x) for x in v]
    if isinstance(v,dict):return {k:meta(x) for k,x in v.items()}
    return v if isinstance(v,(str,int,float,bool)) or v is None else str(type(v))

class Allocations(TorchDispatchMode):
    def __init__(self,out):self.out=out;self.phase='initialize';self.events=[];self.sequence=0
    def __torch_dispatch__(self,fn,types,args=(),kwargs=None):
        before=torch.cuda.memory_allocated();self.sequence+=1
        try:
            value=fn(*args,**(kwargs or {}))
        except Exception:
            node=torch._C._current_autograd_node()
            self.events.append({'sequence':self.sequence,'phase':self.phase,'op':str(fn),'arguments':meta(args),'memory':memory(),
                                'autograd_node':node.name() if node else None,'autograd_metadata':dict(node.metadata) if node else {},
                                'exception':traceback.format_exc(),'stack':traceback.format_stack(limit=20)})
            save(self.out/'FAILED-OP.json',self.events[-1])
            torch.cuda.memory._dump_snapshot(str(self.out/'AT-FAILURE-MEMORY.pickle'))
            raise
        after=torch.cuda.memory_allocated();delta=after-before
        if delta>=16*2**20:
            node=torch._C._current_autograd_node()
            self.events.append({'sequence':self.sequence,'phase':self.phase,'op':str(fn),'arguments':meta(args),'output':meta(value),
                                'allocation_delta':delta,'memory':memory(),'autograd_node':node.name() if node else None})
        return value

def oom(a,out):
    backend='candidate' if a.variant=='candidate-legacy' else 'native'
    tracker=Allocations(out);phases=[];record={}
    torch.cuda.memory._record_memory_history(max_entries=80000)
    try:
        with tracker:
            model=c.network(8,backend,1);phases.append({'phase':'model','memory':memory()})
            def annotate(name):
                def hook(module,args,value):
                    if isinstance(value,torch.Tensor) and value.grad_fn is not None:
                        value.grad_fn.metadata['cause_module']=name
                return hook
            for name,module in model.model.named_modules():
                if type(module).__name__=='RadialFunction':module.register_forward_hook(annotate(name))
            c.whole.seed();v=model.infer();del v;phases.append({'phase':'infer','memory':memory()})
            if a.variant!='native-train-only':
                tracker.phase='force_mixed';v=model.force_mixed();del v
                phases.append({'phase':'force_mixed','memory':memory()})
                torch.save(c.rng(),out/'before_train_rng.pt')
                if a.variant=='native-cache-cleared':
                    gc.collect();torch.cuda.empty_cache()
                    phases.append({'phase':'same_protocol_cache_cleared','memory':memory()})
            else:
                original=out.parent/'oom-native-legacy/before_train_rng.pt'
                if original.exists():c.set_rng(load(original));record['rng_source']=str(original)
                gc.collect();torch.cuda.empty_cache()
            tracker.phase='train_step';phases.append({'phase':'before_train','memory':memory()})
            v=model.train_step(capture=True);torch.cuda.synchronize()
            torch.save(v,out/'TRAINING.pt');record['status']='PASS'
    except Exception as e:
        record.update(status='OOM' if isinstance(e,torch.cuda.OutOfMemoryError) else 'ERROR',error=repr(e),traceback=traceback.format_exc())
    finally:
        record.update(phases=phases,final_memory=memory(),hardware=c.hardware(),large_allocations=tracker.events)
        torch.cuda.memory._dump_snapshot(str(out/'MEMORY.pickle'))
        save(out/'RESULT.json',record)

def grid(a,out):
    import ictc_o2_hp.bilinear_triton as original
    fixed=c.load_module('grid_transposed',Path(__file__).parent/'grid_transposed.py')
    driver=ctypes.CDLL('libcuda.so.1');assert driver.cuInit(0)==0
    limits=[]
    for attribute in (5,6,7):
        value=ctypes.c_int();assert driver.cuDeviceGetAttribute(ctypes.byref(value),attribute,0)==0;limits.append(value.value)
    records=[]
    for n in (128,65472,65535,65536,84032):
        torch.manual_seed(20261127);k=torch.randn(19,19,19,device='cuda')*.1
        x=torch.randn(n,19,32,device='cuda')*.1;y=torch.randn_like(x)*.1
        row={'n':n,'h':32,'original_main_grid':[(19+31)//32,(n//64*64*32+31)//32],'device_max_grid':limits}
        outputs={}
        for label,fn in (('original',original.product),('transposed_grid_diagnostic',fixed.product)):
            try:
                with torch.no_grad():z=fn(k,x,y)
                torch.cuda.synchronize();outputs[label]=z
                row[label]={'status':'PASS','memory':memory()}
            except Exception as e:row[label]={'status':'ERROR','error':repr(e)}
        if len(outputs)==2:row['same_arithmetic_grid_comparison']=metric(outputs['transposed_grid_diagnostic'],outputs['original'])
        with torch.no_grad():reference=c.bounded.product(k,x,y)
        row['native_reference_comparison']=metric(outputs['transposed_grid_diagnostic'],reference)
        records.append(row);save(out/'RESULT.json',{'status':'GRID_DIAGNOSIS','rows':records});print('GRID',row,flush=True)
        del k,x,y,outputs,z,reference;gc.collect();torch.cuda.empty_cache()

def choose(model,variant):
    model.select('candidate' if variant in ('candidate','channel-only') else 'native')
    namespace=model.model.blocks[0].ga.act.forward.__func__.__globals__
    namespace['product']=c.fused_product if variant in ('candidate','bilinear-only') else c.bounded.product

def local_trace(out):
    import ictc_o2_hp.uvw_triton as uvw
    import ictc_o2_hp.bilinear_triton as bil
    state={'phase':'infer','sequence':0,'rows':[],'saved':set()}
    oldmm=uvw._Matmul.forward;oldbil=bil.Bilinear.forward
    def add(kind,inputs,got,ref):
        state['sequence']+=1;m=metric(got.detach(),ref.detach())
        row={'sequence':state['sequence'],'phase':state['phase'],'kind':kind,'inputs':meta(inputs),'metric':m}
        state['rows'].append(row)
        key=(kind,state['phase'].split('-')[0])
        if m.get('different_elements') and key not in state['saved']:
            state['saved'].add(key);name=f'LOCAL-{kind}-{state["sequence"]}.pt'
            torch.save({'inputs':c.cpu(inputs),'candidate':c.cpu(got),'native_same_input':c.cpu(ref)},out/name);row['arrays']=name
    def mm(ctx,left,right):
        value=oldmm(ctx,left,right)
        with torch.no_grad():reference=left@right
        add('channel',(left,right),value,reference);return value
    def product(ctx,k,left,right):
        value=oldbil(ctx,k,left,right)
        with torch.no_grad():reference=c.bounded.product(k,left,right)
        add('bilinear',(k,left,right),value,reference);return value
    uvw._Matmul.forward=staticmethod(mm);bil.Bilinear.forward=staticmethod(product)
    return state

def ablation(a,out):
    model=c.network(a.order,'native',1);choose(model,a.variant)
    torch.save(model.state(),out/'INITIAL.pt')
    trace=local_trace(out) if a.variant=='candidate' else None
    c.whole.seed();torch.save(c.cpu(model.infer()),out/'ef.pt')
    phases=[]
    for step in range(3):
        if trace:trace['phase']=f'mixed-step{step}'
        before=c.rng();mixed=model.force_mixed()
        if trace:trace['phase']=f'train-step{step}'
        training=model.train_step(capture=True)
        if trace:trace['phase']=f'after-step{step}'
        after=c.cpu(model.forward(model.batch('train'),False))
        torch.save({'mixed':mixed,'training':training,'after_output':after,'scheduler':c.cpu(model.trainer.scheduler.scheduler.state_dict())},out/f'step{step}.pt')
        torch.save({'before':before,'after':c.rng()},out/f'rng{step}.pt')
        phases.append({'step':step,'memory':memory()});print('STEP',a.order,a.variant,step,flush=True)
        del mixed,training,after;gc.collect();torch.cuda.empty_cache()
        if trace:save(out/'LOCAL.json',trace['rows'])
    save(out/'RESULT.json',{'status':'CAPTURED','hardware':c.hardware(),'identity':model.identity,'phases':phases})

def optimizer(a,out):
    model=c.network(a.order,'native',1);named=dict(model.model.named_parameters());records=[]
    initial=load(SWEEP/f'gate-captures/120425/network-L{a.order}-native/INITIAL.pt')
    for backend in ('native','candidate'):
        previous=initial
        for step in range(3):
            value=load(SWEEP/f'gate-captures/120425/network-L{a.order}-{backend}/step{step}.pt');tr=value['training']
            c.restore(model,previous)
            for name,p in named.items():p.grad=tr['gradients'][name].cuda()
            norm=torch.nn.utils.clip_grad_norm_(model.model.parameters(),model.trainer.clip_grad_norm)
            model.optimizer.step();model.ema.update();model.trainer.scheduler.step()
            comparison=tree(c.cpu(named),tr['updated'])
            ema_comparison=tree(c.cpu(model.ema.state_dict()),tr['ema'])
            records.append({'backend':backend,'step':step,'clip_total_norm':float(norm),'updated':brief(comparison),'ema':brief(ema_comparison)})
            previous={'model':dict(initial['model'],**tr['updated']),'optimizer':tr['optimizer'],'ema':tr['ema'],'scheduler':value['scheduler']}
            save(out/'RESULT.json',{'status':'CUDA_OPTIMIZER_REPLAY','rows':records});print('OPTIMIZER',a.order,backend,step,records[-1],flush=True)
            del value,tr;gc.collect()

def main():
    p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--order',type=int,default=8);p.add_argument('--variant',default='native-legacy');p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);c.configure(True)
    try:globals()[a.action](a,out)
    except Exception as e:save(out/'RESULT.json',{'status':'OOM' if isinstance(e,torch.cuda.OutOfMemoryError) else 'ERROR','error':repr(e),'traceback':traceback.format_exc()})
if __name__=='__main__':main()
