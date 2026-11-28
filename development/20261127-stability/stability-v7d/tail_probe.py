"""Test physical tail launch equivalence using the preserved actual operands."""
import gc,json,os
from pathlib import Path
from common import *
import ictc_o2_hp.fp32_plans as fp
configure(True)
source=Path((ROOT/'L7-LOCAL-CAPTURE.txt').read_text().strip())
out=ROOT/'tail-probes'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True)
records=[]
for file in sorted(source.glob('operands-*.pt')):
    saved=torch.load(file,map_location='cpu',weights_only=False);meta=saved['metadata']
    if meta['kind']!='angular':continue
    values=[]
    for value,shape,stride,alignment in zip(saved['values'],meta['shapes'],meta['strides'],meta['pointer_mod_256']):
        size=1+sum((d-1)*s for d,s in zip(shape,stride));offset=alignment//4
        base=torch.empty(size+offset,device='cuda',dtype=torch.float32)
        values.append(base.as_strided(shape,stride,offset).copy_(value))
    k,a,b=values;n,d,h=a.shape;main=n//64*64;regions=[]
    for start,count in [(0,main),(main,n-main)]:
        if not count:continue
        plan=fp.plans()[fp.angular_key(k,a,b,64 if count>=64 else count)]
        reference=bounded.product(k,a[start:start+count],b[start:start+count])
        expected=saved['reference'][start:start+count].cuda()
        for normalized in (False,True):
            aa=a[start:start+count] if normalized else a
            bb=b[start:start+count] if normalized else b
            result=a.new_empty((count,d,h)) if normalized else a.new_empty((n,d,h))
            fp._angular[(fp.triton.cdiv(count*h,32),fp.triton.cdiv(d,32))](
                k,aa,bb,result,d,d,d,h,count if normalized else n,*k.stride(),*aa.stride(),*bb.stride(),
                0 if normalized else start,count,plan['parts'],plan['four'],plan.get('swap',False),num_warps=4)
            actual=result if normalized else result[start:start+count]
            row={'source':str(file),'region':[start,count],'normalized':normalized,'plan':plan,
                 'reference_reproduced':torch.equal(reference,expected),'different_elements':int((actual!=reference).sum()),
                 'max_abs':float((actual-reference).abs().max())}
            regions.append(row);print(json.dumps(row),flush=True)
        del result,actual,reference,expected;gc.collect();torch.cuda.empty_cache()
    records.extend(regions);save(out/'RESULT.json',{'rows':records,'hardware':hardware()})
    del saved,values,k,a,b;gc.collect();torch.cuda.empty_cache()
(ROOT/'TAIL-PROBE.txt').write_text(str(out)+'\n')
