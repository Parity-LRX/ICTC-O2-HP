import gc,itertools,json,os
from pathlib import Path
from common import *
def main():
    configure(True);out=ROOT/'local-regression'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True);records=[]
    for order in range(1,9):
        for h,n in ((32,16),(32,68),(32,100),(32,137),(512,100)):
            k=torch.load(ROOT/'models'/f'BILINEAR-L{order}.pt',map_location='cuda',weights_only=True)
            for perm in itertools.permutations(range(3)):
                kk=k.permute(perm);d=kk.shape[0]
                torch.manual_seed(20261127+order);a=(torch.randn(n,d,h,device='cuda')*.1).requires_grad_();b=(torch.randn_like(a)*.1).requires_grad_();g=(torch.randn_like(a)*.1).requires_grad_()
                directions=[torch.randn_like(a)*.1 for _ in range(2)]
                def evaluate(fn):
                    y=fn(kk,a,b);first=torch.autograd.grad(y,(a,b),g,create_graph=True)
                    mixed=torch.autograd.grad(sum((v*q).sum() for v,q in zip(first,directions)),(a,b,g),allow_unused=True)
                    return cpu({'output':y,'first':first,'mixed':mixed})
                reference=evaluate(bounded.product);candidate=evaluate(fused_product)
                case=out/f'L{order}-H{h}-N{n}-P{perm[0]}{perm[1]}{perm[2]}.pt';torch.save({'native':reference,'candidate':candidate},case)
                records.append({'path':str(case),'sha256':sha(case),'order':order,'H':h,'N':n,'permutation':list(perm),'k_stride':list(kk.stride())})
                del a,b,g,reference,candidate;gc.collect();torch.cuda.empty_cache()
            save(out/'RESULT.json',{'status':'LOCAL_CAPTURED','records':records,'hardware':hardware()});print('LOCAL',order,h,n,len(records),flush=True)
    (ROOT/'LOCAL-RESULT.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
