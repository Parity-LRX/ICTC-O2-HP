"""Native CPU autograd checks of the graph-task predicate, including captures."""
import json,os
from pathlib import Path
import torch
ROOT=Path(os.environ['O2_KROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
records=[]
class Probe(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b):ctx.save_for_backward(a,b);return a@b
    @staticmethod
    def backward(ctx,g):
        a,b=ctx.saved_tensors
        mask=tuple(needed and node is not None and (value.is_leaf or torch._C._will_engine_execute_node(node)) for needed,value,(node,_) in zip(ctx.needs_input_grad,(a,b),ctx.next_functions))
        records.append({'case':case,'mask':mask,'requires_grad':ctx.needs_input_grad,'create_graph':torch.is_grad_enabled()})
        return g@b.T if mask[0] else None,a.T@g if mask[1] else None
torch.manual_seed(20261126)
for views in (False,True):
    for target in ('left','right','both','view-left','view-right','backward'):
        x=torch.randn(5,3,requires_grad=True);k=torch.randn(7,3,requires_grad=True)
        a=x.view_as(x) if views else x;b=k.T
        case=f'{views}-{target}';y=Probe.apply(a,b);ref=a@b;v=torch.randn_like(y)
        if target=='backward':
            y.backward(v,retain_graph=True);r=torch.autograd.grad(ref,(x,k),v);assert all(torch.equal(t.grad,g) for t,g in zip((x,k),r))
        else:
            args={'left':(x,),'right':(k,),'both':(x,k),'view-left':(a,),'view-right':(b,)}[target]
            got=torch.autograd.grad(y,args,v,create_graph=True);expected=torch.autograd.grad(ref,args,v,create_graph=True)
            assert all(torch.equal(p,q) for p,q in zip(got,expected))
            if target=='both':
                d=sum((g*t).sum() for g,t in zip(got,args));r=sum((g*t).sum() for g,t in zip(expected,args))
                gg=torch.autograd.grad(d,(x,k));rr=torch.autograd.grad(r,(x,k));assert all(torch.equal(p,q) for p,q in zip(gg,rr))
case='force-mixed-K'
x=torch.randn(4,3,requires_grad=True);k=torch.randn(7,3,requires_grad=True)
y=Probe.apply(x,k.T).square().sum();ref=(x@k.T).square().sum()
f=torch.autograd.grad(y,x,create_graph=True)[0];rf=torch.autograd.grad(ref,x,create_graph=True)[0]
assert torch.equal(f,rf)
g=torch.autograd.grad(f.square().sum(),(x,k));r=torch.autograd.grad(rf.square().sum(),(x,k))
assert all(torch.equal(p,q) for p,q in zip(g,r))
case='alias'
x=torch.randn(4,4,requires_grad=True);y=Probe.apply(x,x);r=x@x;v=torch.randn_like(y)
assert torch.equal(torch.autograd.grad(y,x,v)[0],torch.autograd.grad(r,x,v)[0])
result={'status':'GRAPH_TASK_MASK_CPU_PASS','records':records,'torch':torch.__version__,'scope':'predicate/capture and higher-order semantics only; not Triton or whole-network qualification'}
(ROOT/('MASK-PROBE-'+os.environ['SLURM_JOB_ID']+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
