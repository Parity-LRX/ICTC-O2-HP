import argparse,gc,json,os
from pathlib import Path
from support import *
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True)
c.configure(True)
import ictc_o2_hp.uvw_triton as uvw
import ictc_o2_hp.bilinear_triton as bil
oldmm,oldbil=uvw._Matmul.forward,bil.Bilinear.forward
class FirstDifference(Exception):pass
rows=[]
for order in (2,4,6,7):
    model=c.network(order,'candidate',1)
    c.set_rng(load(SWEEP/f'gate-captures/120425/network-L{order}-candidate/rng0.pt')['before'])
    v=model.force_mixed();del v
    state={'site':None,'sequence':0,'phase':'original-step0-training-forward'};seen=set()
    hooks=[]
    for name,module in model.model.named_modules():
        if type(module).__name__=='CanonicalMLinear' or hasattr(module,'canonical_kernel'):
            hooks.append(module.register_forward_pre_hook(lambda m,args,n=name:state.update(site=n)))
    def record(kind,inputs,value,reference):
        state['sequence']+=1;m=metric(value.detach(),reference.detach())
        if m.get('different_elements',0) and kind not in seen:
            seen.add(kind)
            row={'order':order,**state,'kind':kind,'shapes':[list(x.shape) for x in inputs],'strides':[list(x.stride()) for x in inputs],'metric':m}
            torch.save({'inputs':c.cpu(inputs),'candidate':c.cpu(value),'native_same_input':c.cpu(reference)},out/f'L{order}-FIRST-{kind}.pt')
            if kind=='bilinear':
                k,left,right=inputs;n,_,h=left.shape;oa,la,lb=k.shape;count=min(64,n);plans=[]
                for parts in (1,2,3,4,5,6,9):
                    for four in (False,True):
                        result=left.new_empty((count,oa,h))
                        bil._bilinear[(bil.triton.cdiv(oa,32),bil.triton.cdiv(count*h,32))](k,left,right,result,oa,la,lb,h,n,*k.stride(),*left.stride(),*right.stride(),32,32,0,count,parts,four,num_warps=4)
                        plans.append({'parts':parts,'four_chains':four,'comparison':metric(result,reference[:count])})
                row['same_operands_fp32_reduction_plans_first64']=plans
            rows.append(row);save(out/'RESULT.json',{'status':'FIRST_DIFFERENCE_CAPTURED','rows':rows})
            if len(seen)==2:raise FirstDifference()
    def mm(ctx,left,right):
        value=oldmm(ctx,left,right)
        with torch.no_grad():ref=left@right
        record('channel',(left,right),value,ref);return value
    def product(ctx,k,left,right):
        value=oldbil(ctx,k,left,right)
        with torch.no_grad():ref=c.bounded.product(k,left,right)
        record('bilinear',(k,left,right),value,ref);return value
    uvw._Matmul.forward=staticmethod(mm);bil.Bilinear.forward=staticmethod(product)
    try:model.forward(model.batch('train'),True)
    except FirstDifference:pass
    finally:
        uvw._Matmul.forward=staticmethod(oldmm);bil.Bilinear.forward=staticmethod(oldbil)
        for h in hooks:h.remove()
    del model;gc.collect();torch.cuda.empty_cache()
print(json.dumps(rows),flush=True)
