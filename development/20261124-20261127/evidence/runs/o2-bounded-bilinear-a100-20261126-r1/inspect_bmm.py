"""Observe original einsum's actual primitive layout; no candidate math or timing."""
import copy,itertools,json,os
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from context import ROOT,configure,cpu,save,sha
import bounded
def main():
    configure();out=ROOT/'results'/('bmm-layout-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir();records=[];seen=set()
    source=Path((ROOT/'CAPTURE-RESULT.txt').read_text().strip());capture=json.loads((source/'CAPTURE.json').read_text())
    cases=list(capture['records'])
    # Recursive second derivatives reach all six coefficient-axis permutations,
    # including permutations not themselves unpacked by the first backward.
    for l in (19,25):
        template=next(r for r in cases if r['shapes']['k']==[l,l,l])
        for strides in itertools.permutations((l*l,l,1)):
            row=copy.deepcopy(template);row['strides']['k']=list(strides);cases.append(row)
    for row in cases:
        key=(tuple(row['shapes']['k']),tuple(row['strides']['k']),row['shapes']['a'][-1])
        if key in seen:continue
        seen.add(key);raw=torch.load(row['path'],map_location='cpu',weights_only=False);l=raw['k'].shape[0];h=raw['a'].shape[-1]
        k=torch.empty_strided(raw['k'].shape,row['strides']['k'],device='cuda').copy_(torch.arange(l**3,device='cuda',dtype=torch.float32).reshape(l,l,l))
        for n in (64,row['shapes']['a'][0]%64):
            if not n:continue
            a=raw['a'][:n].cuda();b=raw['b'][:n].cuda();found=[]
            class Observe(TorchDispatchMode):
                def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                    value=func(*args,**(kwargs or {}))
                    if func==torch.ops.aten.bmm.default:found.append((args[0].detach().clone(),args[1].detach().clone(),value.detach().clone(),list(args[0].stride()),list(args[1].stride())))
                    return value
            with Observe():y=bounded.product(k,a,b)
            assert len(found)==1
            left,right,value,ls,rs=found[0]
            assert left.shape==(1,l,l*l) and right.shape==(1,l*l,n*h)
            bc=k.reshape(l,l*l);cb=k.transpose(1,2).reshape(l,l*l)
            order='bc' if torch.equal(left[0],bc) else 'cb' if torch.equal(left[0],cb) else None
            assert order is not None
            outer=(a[:,:,None,:]*b[:,None,:,:]).permute(1,2,0,3)
            if order=='cb':outer=outer.transpose(0,1)
            assert torch.equal(right[0],outer.reshape(l*l,n*h))
            assert torch.equal(torch.bmm(left,right),value)
            records.append({'k_strides':row['strides']['k'],'L':l,'Nrows':n,'H':h,'order':order,'left_shape':list(left.shape),'left_stride':ls,'right_shape':list(right.shape),'right_stride':rs})
            print(records[-1],flush=True)
    save(out/'RESULT.json',{'status':'BMM_LAYOUT_OBSERVED','records':records,'scope':'original bounded/einsum lowering, synthetic angular coefficient labels, actual captured inputs; numerical diagnostic only'})
    (ROOT/'BMM-LAYOUT.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
