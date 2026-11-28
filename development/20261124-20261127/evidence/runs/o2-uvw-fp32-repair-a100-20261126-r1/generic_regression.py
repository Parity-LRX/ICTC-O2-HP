"""Retain original random/shared-UVW, full-parity, strided and live-K coverage."""
import gc,json,os
import torch
from common import ROOT,configure,cpu,save,sha,seed
from ictc_o2_hp import UvwTensorProduct,ReferenceUvwTensorProduct
from ictc_o2_hp.uvw_triton import canonical_linear
def collect(fn,values,cot,directions):
    y=fn(*values);g=torch.autograd.grad(y,values,cot,create_graph=True)
    scalar=sum((a*b).sum() for a,b in zip(g,directions))
    second=torch.autograd.grad(scalar,(*values,cot),allow_unused=True)
    return cpu([y,*g,*[v if v is not None else torch.zeros_like(t) for v,t in zip(second,(*values,cot))]])
def main():
    configure();seed();out=ROOT/'results'/('generic-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True);records=[]
    def compare(name,fn,reference,values):
        y=reference(*values);cot=torch.randn_like(y,requires_grad=True);directions=tuple(torch.randn_like(v) for v in values)
        roles=['output']+[f'vjp_{i}' for i in range(len(values))]+[f'mixed_{i}' for i in range(len(values))]+['mixed_cotangent']
        r=collect(reference,values,cot,directions);repeat=collect(reference,values,cot,directions);a=collect(fn,values,cot,directions)
        path=out/(name+'.pt');torch.save({'reference':dict(zip(roles,r)),'native_repeat':dict(zip(roles,repeat)),
            'candidate':dict(zip(roles,a))},path)
        records.append({'name':name,'path':str(path),'sha256':sha(path)});print(name,flush=True)
    for w,v in ((256,1024),(192,768),(512,128),(384,96)):
        x=torch.randn(137,2,v,device='cuda',requires_grad=True);k=(torch.randn(w,v,device='cuda')*.02).requires_grad_()
        compare(f'V3-{w}x{v}',canonical_linear,torch.nn.functional.linear,(x,k))
    for parity in ('canonical','full'):
        for norm in ('none','element'):
            ref=ReferenceUvwTensorProduct(3,5,4,2,parity=parity,path_normalization=norm).cuda()
            cand=UvwTensorProduct(3,5,4,2,parity=parity,path_normalization=norm).cuda()
            x=torch.randn(7,ref.input1_dimension,device='cuda',requires_grad=True)
            y=torch.randn(7,ref.input2_dimension,device='cuda',requires_grad=True);w=ref.weight.detach().clone().requires_grad_()
            compare(f'UVW-{parity}-{norm}',cand,ref,(x,y,w))
    x=torch.randn(19,2,96,device='cuda').transpose(0,1).requires_grad_();k=torch.randn(64,96,device='cuda',requires_grad=True)
    compare('strided-live-K',canonical_linear,torch.nn.functional.linear,(x,k))
    before=canonical_linear(x,k).detach()
    with torch.no_grad():k.add_(.01)
    after=canonical_linear(x,k).detach();assert not torch.equal(before,after)
    save(out/'MANIFEST.json',{'records':records,'live_weight_update_seen':True,
        'operator_source_sha256':sha(__import__('ictc_o2_hp.uvw_triton',fromlist=['__file__']).__file__)})
    (ROOT/'GENERIC-RESULT.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
