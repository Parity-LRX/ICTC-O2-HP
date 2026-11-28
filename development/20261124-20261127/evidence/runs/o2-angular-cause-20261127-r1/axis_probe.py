import json,os,gc
from pathlib import Path
from support import *
c.configure(True)
import ictc_o2_hp.bilinear_triton as bil
out=ROOT/'results'/('axis-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
first=ROOT/'results/peak-120617/first-compute';metadata=json.loads((first/'RESULT.json').read_text())['rows'];rows=[]
for order in (2,4,6,7):
    info=next(r for r in metadata if r['order']==order and r['kind']=='bilinear')
    data=load(first/f'L{order}-FIRST-bilinear.pt');inputs=[]
    for value,strides in zip(data['inputs'],info['strides']):
        rebuilt=torch.empty_strided(value.shape,strides,dtype=torch.float32,device='cuda');rebuilt.copy_(value);inputs.append(rebuilt)
    k,left,right=inputs;n,_,h=left.shape;oa,la,lb=k.shape;count=min(64,n)
    with torch.no_grad():native=c.bounded.product(k,left[:count],right[:count])
    saved=data['native_same_input'][:count].cuda()
    row={'order':order,'original_k_stride':list(k.stride()),'native_reconstruction':metric(native,saved),'plans':[]};outputs=[]
    for swap in (False,True):
        kk=k.permute(0,2,1) if swap else k;aa,bb=(right,left) if swap else (left,right)
        oa,la,lb=kk.shape
        for parts in (1,2,3,4,5,6,9):
            for four in (False,True):
                result=aa.new_empty((count,oa,h))
                bil._bilinear[(bil.triton.cdiv(oa,32),bil.triton.cdiv(count*h,32))](kk,aa,bb,result,oa,la,lb,h,n,*kk.stride(),*aa.stride(),*bb.stride(),32,32,0,count,parts,four,num_warps=4)
                row['plans'].append({'swapped_reduction_axes':swap,'parts':parts,'four_chains':four,'metric':metric(result,saved)})
                outputs.append(c.cpu(result))
    row['exact_plans']=[p for p in row['plans'] if p['metric']['different_elements']==0]
    torch.save({'native_saved':c.cpu(saved),'native_reconstructed':c.cpu(native),'plans':outputs},out/f'L{order}-PLAN-OUTPUTS.pt')
    rows.append(row);save(out/'RESULT.json',{'status':'FP32_AXIS_ORDER_PROBED','rows':rows});print('ORDER',order,'RECON',row['native_reconstruction'],'EXACT',row['exact_plans'],flush=True)
    del inputs,data,k,left,right,native,saved,result;gc.collect();torch.cuda.empty_cache()
(ROOT/'AXIS-RESULT.txt').write_text(str(out)+'\n')
