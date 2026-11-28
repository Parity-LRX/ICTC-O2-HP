"""Extend static native FP32 tail-plan identification beyond the old 32-part search."""
import gc,json,os
from pathlib import Path
from common import *
import ictc_o2_hp.fp32_plans as fp
configure(True)
source=Path((ROOT/'L7-LOCAL-CAPTURE.txt').read_text().strip())
out=ROOT/'tail-calibration'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True)
records=[];plans={}
for file in sorted(source.glob('operands-*.pt')):
    saved=torch.load(file,map_location='cpu',weights_only=False);meta=saved['metadata']
    if meta['kind']!='angular':continue
    values=[]
    for value,shape,stride,alignment in zip(saved['values'],meta['shapes'],meta['strides'],meta['pointer_mod_256']):
        size=1+sum((d-1)*s for d,s in zip(shape,stride));offset=alignment//4
        base=torch.empty(size+offset,device='cuda',dtype=torch.float32)
        values.append(base.as_strided(shape,stride,offset).copy_(value))
    k,a,b=values;n,d,h=a.shape;count=n%64;offset=n-count
    reference=bounded.product(k,a[offset:],b[offset:]);output=a.new_empty((n,d,h))
    key=fp.angular_key(k,a,b,count);best=None;closest=None;trials=0
    for parts in range(1,257):
        if best:break
        for four,swap in ((False,False),(True,False),(False,True),(True,True)):
            fp._angular[(fp.triton.cdiv(count*h,32),fp.triton.cdiv(d,32))](k,a,b,output,d,d,d,h,n,*k.stride(),*a.stride(),*b.stride(),offset,count,parts,four,swap,num_warps=4)
            actual=output[offset:];different=int((actual!=reference).sum());trials+=1
            score=float((actual-reference).norm()/reference.norm())
            candidate={'parts':parts,'four':four,'swap':swap}
            if closest is None or score<closest['relative_l2']:
                closest={'plan':candidate,'different_elements':different,'relative_l2':score}
            if not different:best=candidate;break
    row={'source':str(file),'key':key,'trials':trials,'plan':best,'closest':closest,'status':'EXACT' if best else 'UNMATCHED'}
    if best:plans[key]=best
    records.append(row);save(out/'RESULT.json',{'records':records,'plans':plans,'hardware':hardware()})
    print('ACTUAL_TAIL_PLAN',json.dumps(row),flush=True)
    del saved,values,k,a,b,reference,output,actual;gc.collect();torch.cuda.empty_cache()
(ROOT/'ACTUAL-TAIL-PLANS.txt').write_text(str(out)+'\n')
assert all(r['status']=='EXACT' for r in records)
