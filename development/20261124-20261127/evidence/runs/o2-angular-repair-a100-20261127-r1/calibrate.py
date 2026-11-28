"""Same-form native FP32 oracle; fit plans, then check complete arrays."""
import argparse,gc,itertools,json,os,time
from pathlib import Path
from common import *
import ictc_o2_hp.fp32_plans as fp
def run(a):
    configure(True);out=ROOT/'calibration'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True)
    records=[];plans={};started=time.time();torch.manual_seed(20261127)
    parts=list(range(1,17))+[20,24,28,32]
    def flush():
        save(out/'RESULT.json',{'status':'CALIBRATING','records':records,'plans':plans,'elapsed_s':time.time()-started,'hardware':hardware()})
    def fit_mm(left,right):
        m,k=left.shape;n=right.shape[1];reference=left@right;output=torch.empty_like(reference)
        fits=[];trials=0
        # Keep the exact same kernel in calibration and production.
        for count in parts:
            for four,block in ((False,16),(False,32),(True,16)):
                trials+=1;fp._mm[(1,1)](left,right,output,m,n,k,*left.stride(),*right.stride(),count,four,block,num_warps=4)
                if torch.equal(output[:32,:32],reference[:32,:32]):
                    fp._mm[(fp.triton.cdiv(m,32),fp.triton.cdiv(n,32))](left,right,output,m,n,k,*left.stride(),*right.stride(),count,four,block,num_warps=4)
                    if torch.equal(output,reference):return {'parts':count,'four':four,'block':block},trials,0
                    fits.append({'parts':count,'four':four,'block':block,'different':int((output!=reference).sum())})
        return None,trials,fits
    inventory=Path((ROOT/'INVENTORY-RESULT.txt').read_text().strip());shapes=json.loads((inventory/'RESULT.json').read_text())['shapes']
    if a.kind in ('all','mm'):
        for row in shapes:
            if row['kind']!='mm':continue
            values=[torch.empty_strided(shape,stride,device='cuda').normal_(0,.1) for shape,stride in zip(row['shapes'],row['strides'])]
            key=fp.mm_key(*values)
            if key in plans:continue
            plan,trials,detail=fit_mm(*values)
            record={'key':key,'kind':'mm','input':row,'plan':plan,'trials':trials,'whole_array_different':detail,'status':'EXACT' if plan else 'UNMATCHED'}
            if plan:plans[key]=plan
            records.append(record);flush();print('MM',key,record['status'],plan,flush=True)
            del values;gc.collect();torch.cuda.empty_cache()
    if a.kind in ('all','angular'):
        dims=[(l+1)**2 for l in range(1,9)]+[19]
        tails=[64,*range(4,64,4)] if not a.quick else [64,4,36,16]
        for d in dims:
            for h in ((32,) if d==19 else (32,512)):
                for count in tails:
                    logical_k=torch.randn(d,d,d,device='cuda')*.1
                    basea=torch.randn(count,d,h,device='cuda')*.1;baseb=torch.randn_like(basea)*.1
                    for ks in itertools.permutations((d*d,d,1)):
                        k=torch.empty_strided((d,d,d),ks,device='cuda').copy_(logical_k)
                        shared=None
                        for sa,sb in ((1,1),(1,2),(2,1),(2,2)):
                            left=torch.empty_strided((count,d,h),(d*h*sa,h*sa,1),device='cuda').copy_(basea)
                            right=torch.empty_strided((count,d,h),(d*h*sb,h*sb,1),device='cuda').copy_(baseb)
                            reference=bounded.product(k,left,right);output=torch.empty_like(reference);key=fp.angular_key(k,left,right,count)
                            def launch(plan,full):
                                grid=(fp.triton.cdiv(count*h,32),fp.triton.cdiv(d,32)) if full else (1,1)
                                fp._angular[grid](k,left,right,output,d,d,d,h,count,*k.stride(),*left.stride(),*right.stride(),0,count,plan['parts'],plan['four'],plan['swap'],num_warps=4)
                            plan=None;trials=0
                            if shared:
                                launch(shared,True)
                                if torch.equal(output,reference):plan=shared
                            if plan is None:
                                for p in parts:
                                    if plan:break
                                    for four,swap in ((False,False),(True,False),(False,True),(True,True)):
                                        trial={'parts':p,'four':four,'swap':swap};trials+=1;launch(trial,False)
                                        if torch.equal(output[0,:32,:32],reference[0,:32,:32]):
                                            launch(trial,True)
                                            if torch.equal(output,reference):plan=trial;break
                            record={'key':key,'kind':'angular','D':d,'H':h,'count':count,'k_strides':list(ks),'input_stride_scales':[sa,sb],
                                    'plan':plan,'trials':trials,'status':'EXACT' if plan else 'UNMATCHED','full_array_elements':reference.numel()}
                            if plan:plans[key]=plan;shared=plan
                            records.append(record)
                            if not plan:print('UNMATCHED_ANGULAR',key,flush=True)
                            del left,right,reference,output
                        flush()
                    print('ANGULAR',d,h,count,'rows',len(records),'unmatched',sum(r['status']=='UNMATCHED' for r in records),flush=True)
                    del logical_k,basea,baseb,k;gc.collect();torch.cuda.empty_cache()
    failure=[r for r in records if r['status']=='UNMATCHED']
    save(out/'RESULT.json',{'status':'PLANS_CALIBRATED' if not failure else 'UNMATCHED_PLANS','records':records,'plans':plans,'failures':failure,
                           'elapsed_s':time.time()-started,'hardware':hardware(),'precision':'FP32 native same-form oracle; exact full-array plan selection is diagnostic, original formal gates unchanged'})
    (ROOT/('CALIBRATION-'+a.kind.upper()+'.txt')).write_text(str(out)+'\n')
    assert not failure,f'{len(failure)} reduction plans unmatched'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['all','mm','angular'],default='all');p.add_argument('--quick',action='store_true');run(p.parse_args())
