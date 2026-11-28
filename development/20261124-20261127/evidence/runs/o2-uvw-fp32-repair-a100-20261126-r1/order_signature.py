"""Diagnostic cancellation signatures expose native FP32 accumulation ordering."""
import json,os,sys
from pathlib import Path
import torch
from common import ROOT,configure,cpu,save,sha
sys.path.insert(0,str(ROOT/'snapshots/train-119347'))
from candidate_kernels import CONFIGS,matmul
def main():
    configure();source=Path((ROOT/'CAPTURE-RESULT.txt').read_text().strip())
    rows=json.loads((source/'CAPTURE.json').read_text())['backends']['native'][:4]
    out=ROOT/'results'/('order-signature-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True);manifest=[]
    chosen=('s32','p2k16','p2k32','p4k16','p4k32','i2k16','i4k16','h2k16','h4k16','even8','reverse8','pair8')
    for row in rows:
        v=torch.load(row['path'],map_location='cpu',weights_only=False)
        for role in ('forward','input_adjoint'):
            template=v['x'] if role=='forward' else v['adjoints']['energy_forward_and_force']['g']
            a=torch.zeros_like(template.cuda()).reshape(-1,template.shape[-1])
            b=torch.ones_like(v['K'].cuda().T if role=='forward' else v['K'].cuda())
            m,k=a.shape;r=torch.arange(m,device='cuda');group=r//k
            offsets=torch.tensor([0,1,4,8,16,32,64,128,256,512,768,1023],device='cuda')
            i=offsets[group%len(offsets)]%k;j=(i+(1<<(group%5)))%k;one=r%k
            a[r,i]=float(2**26);a[r,j]=-float(2**26);a[r,one]+=1
            valid=(one!=i)&(one!=j)&(i!=j)
            a.requires_grad_();b.requires_grad_()
            reference=(torch.nn.functional.linear(a.reshape(template.shape),b.T).reshape(m,-1)
                       if role=='forward' else torch.matmul(a,b)).detach()
            outputs={name:cpu(matmul(a,b,CONFIGS[name])) for name in chosen}
            name=f"{row['index']:02d}-{role}.pt";path=out/name
            torch.save({'a':cpu(a),'b':cpu(b),'valid_rows':cpu(valid),'reference':cpu(reference),'outputs':outputs},path)
            manifest.append({'site':row['name'],'role':role,'path':str(path),'sha256':sha(path),
                'a_shape':list(a.shape),'a_stride':list(a.stride()),'b_shape':list(b.shape),'b_stride':list(b.stride())})
            del a,b,reference,outputs
            torch.cuda.empty_cache();print(row['index'],role,flush=True)
    save(out/'MANIFEST.json',{'status':'ORDER_SIGNATURE_DIAGNOSTIC_ONLY','records':manifest,
        'construction':'one +2^26, one -2^26, one +1 per row; original GEMM shapes/layouts; native FP32 reference',
        'gpu':torch.cuda.get_device_name(),'kernel_source_sha256':sha(ROOT/'snapshots/train-119347/candidate_kernels.py')})
    (ROOT/'ORDER-SIGNATURE.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
