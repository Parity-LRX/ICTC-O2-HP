"""Complete nonexact plans under the unchanged original FP32 criteria."""
import gc,json,os,time
from pathlib import Path
from common import *
import ictc_o2_hp.fp32_plans as fp
def measure(value,reference):
    d=value-reference;den=reference.norm();rn=float(d.norm()/den) if den else 0. if not d.norm() else None
    scale=torch.maximum(torch.maximum(value.abs().max(),reference.abs().max()),value.new_tensor(1e-7))
    eps=float(d.abs().max()/scale);bad=int((d.abs()>2e-5+2e-5*reference.abs()).sum())
    return {'relative_l2':rn,'epsilon_max':eps,'component_fail_count':bad,'different_elements':int((value!=reference).sum()),
            'passed':bool(torch.isfinite(value).all()) and rn is not None and rn<=1e-5 and eps<=1e-5 and bad==0}
def main():
    configure(True);out=ROOT/'calibration-complete'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True)
    source=ROOT/'calibration/120771/RESULT.json';original=json.loads(source.read_text());plans=original['plans'].copy();records=[]
    todo=[r for r in original['records'] if r['status']=='UNMATCHED'];torch.manual_seed(20261128)
    for row in todo:
        kind=row['kind'];key=row['key'];candidates=[]
        if kind=='mm':
            meta=row['input'];left,right=[torch.empty_strided(shape,stride,device='cuda').normal_(0,.1) for shape,stride in zip(meta['shapes'],meta['strides'])]
            m,k=left.shape;n=right.shape[1];reference=left@right;output=torch.empty_like(reference)
            def launch(plan,full):
                grid=(fp.triton.cdiv(m,32),fp.triton.cdiv(n,32)) if full else (1,1)
                fp._mm[grid](left,right,output,m,n,k,*left.stride(),*right.stride(),plan['parts'],plan['four'],plan['block'],num_warps=4)
            for p in range(1,129):
                for four,block in ((False,16),(False,32),(True,16)):
                    trial={'parts':p,'four':four,'block':block};launch(trial,False)
                    score=float((output[:32,:32]-reference[:32,:32]).norm()/reference[:32,:32].norm());candidates.append((score,trial))
                    if score==0:break
                if candidates[-1][0]==0:break
        else:
            d,h,count=row['D'],row['H'],row['count'];ks=row['k_strides'];sa,sb=row['input_stride_scales']
            k=torch.empty_strided((d,d,d),ks,device='cuda').normal_(0,.1)
            left=torch.empty_strided((count,d,h),(d*h*sa,h*sa,1),device='cuda').normal_(0,.1)
            right=torch.empty_strided((count,d,h),(d*h*sb,h*sb,1),device='cuda').normal_(0,.1)
            reference=bounded.product(k,left,right);output=torch.empty_like(reference)
            def launch(plan,full):
                grid=(fp.triton.cdiv(count*h,32),fp.triton.cdiv(d,32)) if full else (1,1)
                fp._angular[grid](k,left,right,output,d,d,d,h,count,*k.stride(),*left.stride(),*right.stride(),0,count,plan['parts'],plan['four'],plan['swap'],num_warps=4)
            for p in range(1,33):
                for four,swap in ((False,False),(True,False),(False,True),(True,True)):
                    trial={'parts':p,'four':four,'swap':swap};launch(trial,False)
                    score=float((output[0,:32,:32]-reference[0,:32,:32]).norm()/reference[0,:32,:32].norm());candidates.append((score,trial))
                    if score==0:break
                if candidates[-1][0]==0:break
        candidates.sort(key=lambda x:x[0]);best=None;best_error=None
        for _,trial in candidates[:8]:
            launch(trial,True);error=measure(output,reference)
            if error['passed'] and (best_error is None or error['relative_l2']<best_error['relative_l2']):best=trial;best_error=error
            if error['different_elements']==0:break
        if best is not None:plans[key]=best
        entry={**row,'status':'EXACT' if best_error and best_error['different_elements']==0 else 'WITHIN_ORIGINAL_GATE' if best else 'NUMERIC_FAIL',
               'plan':best,'full_array_metric':best_error};records.append(entry)
        if best:
            launch(best,True);path=out/f'{len(records):04d}.pt';torch.save({'reference':cpu(reference),'candidate':cpu(output)},path);entry.update(arrays=str(path),sha256=sha(path))
        save(out/'RESULT.json',{'status':'COMPLETING','records':records,'plans':plans});print('COMPLETE',key,entry['status'],best,best_error,flush=True)
        del left,right,reference,output;gc.collect();torch.cuda.empty_cache()
    failures=[r for r in records if r['status']=='NUMERIC_FAIL'];save(out/'RESULT.json',{'status':'COMPLETED' if not failures else 'FAIL','records':records,'plans':plans,
               'original_calibration':str(source),'original_unmatched':len(todo),'failures':failures,'hardware':hardware(),'original_thresholds':True})
    (ROOT/'COMPLETE-PLANS.txt').write_text(str(out)+'\n');assert not failures
if __name__=='__main__':main()
