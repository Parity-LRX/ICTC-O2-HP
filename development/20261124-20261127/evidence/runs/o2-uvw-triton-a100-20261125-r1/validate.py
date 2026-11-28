"""Full-array role-separated forward, VJP and mixed-direction diagnostics."""
import json,os,sys,time
from pathlib import Path
import torch
ROOT=Path(os.environ.get('O2_WHOLE_ROOT',Path(__file__).resolve().parent))
sys.path.insert(0,os.environ.get('O2_UVW_SOURCE',str(ROOT.parents[1]/'ICTC-O2-HP/src')))
from ictc_o2_hp.uvw_triton import canonical_linear
from ictc_o2_hp import UvwTensorProduct,ReferenceUvwTensorProduct
from whole import configure,save,seed

def metric(a,r):
    assert a.shape==r.shape
    assert torch.isfinite(a).all() and torch.isfinite(r).all()
    d=(a-r).float()
    scale=torch.maximum(a.float().abs().max(),r.float().abs().max()).clamp_min(1e-7)
    error=float(d.abs().max()/scale)
    return {'passed':error<=1e-5,'epsilon_max':error,'max_abs':float(d.abs().max()),
            'normalization_scale':float(scale),'statistics_dtype':'float32','elements':a.numel()}
def collect(fn,values,cot,directions):
    out=fn(*values)
    g=torch.autograd.grad(out,values,cot,create_graph=True)
    scalar=sum((a*b).sum() for a,b in zip(g,directions))
    second=torch.autograd.grad(scalar,(*values,cot),allow_unused=True)
    return [out,*g,*[v if v is not None else torch.zeros_like(target) for v,target in zip(second,(*values,cot))]]
def compare(name,fn,reference,values):
    out=reference(*values)
    cot=torch.randn_like(out,requires_grad=True)
    directions=tuple(torch.randn_like(x) for x in values)
    r=collect(reference,values,cot,directions)
    repeat=collect(reference,values,cot,directions)
    a=collect(fn,values,cot,directions)
    n=len(values)
    roles=['output']+[f'vjp_{i}' for i in range(n)]+[f'mixed_{i}' for i in range(n)]+['mixed_cotangent']
    rows={role:metric(x,y) for role,x,y in zip(roles,a,r)}
    native={role:metric(x,y) for role,x,y in zip(roles,repeat,r)}
    result={'name':name,'roles':rows,'native_repeat':native,
        'first_order_pass':all(rows[k]['passed'] for k in roles[:n+1]),
        'mixed_diagnostic_pass':all(rows[k]['passed'] for k in roles[n+1:]),
        'formal_new_high_order_gate':'HOLD_UNAPPROVED_NEW_OBJECT'}
    print(json.dumps(result),flush=True)
    return result
def main(out):
    configure();out.mkdir(parents=True,exist_ok=True);rows=[]
    seed()
    for w,v in ((256,1024),(192,768),(512,128),(384,96)):
        x=torch.randn(137,2,v,device='cuda',requires_grad=True)
        k=(torch.randn(w,v,device='cuda')*.02).requires_grad_()
        rows.append(compare(f'V3-{w}x{v}',canonical_linear,torch.nn.functional.linear,(x,k)))
    for parity in ('canonical','full'):
        for norm in ('none','element'):
            ref=ReferenceUvwTensorProduct(3,5,4,2,parity=parity,path_normalization=norm).cuda()
            candidate=UvwTensorProduct(3,5,4,2,parity=parity,path_normalization=norm).cuda()
            x=torch.randn(7,ref.input1_dimension,device='cuda',requires_grad=True)
            y=torch.randn(7,ref.input2_dimension,device='cuda',requires_grad=True)
            w=ref.weight.detach().clone().requires_grad_()
            rows.append(compare(f'UVW-{parity}-{norm}',candidate,ref,(x,y,w)))
    # Noncontiguous feature layout and an optimizer update must remain live.
    x=torch.randn(19,2,96,device='cuda').transpose(0,1).requires_grad_()
    k=torch.randn(64,96,device='cuda',requires_grad=True)
    rows.append(compare('strided-live-K',canonical_linear,torch.nn.functional.linear,(x,k)))
    before=canonical_linear(x,k).detach()
    with torch.no_grad():k.add_(.01)
    after=canonical_linear(x,k).detach()
    assert not torch.equal(before,after)
    receipt={'job':os.environ['SLURM_JOB_ID'],'gpu':torch.cuda.get_device_name(),
        'torch':torch.__version__,'rows':rows,'first_order_pass':all(r['first_order_pass'] for r in rows),
        'mixed_diagnostic_pass':all(r['mixed_diagnostic_pass'] for r in rows),
        'live_weight_update_seen':True,'new_high_order_formal_gate':'HOLD'}
    save(out/'VALIDATION.json',receipt)
    # First-order failure is recorded and must be examined before network use.
    assert receipt['first_order_pass'], 'First-order operator validation failed'
if __name__=='__main__':main(Path(sys.argv[1]))
