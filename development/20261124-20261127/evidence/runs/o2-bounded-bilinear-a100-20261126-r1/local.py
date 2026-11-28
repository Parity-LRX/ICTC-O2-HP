import gc,json,os
import torch
from context import ROOT,configure,cpu,save,sha,package_sha,seed
from ictc_o2_hp.bilinear_triton import product
import bounded
def collect(fn,k,a,b,u,da,db):
    a=a.detach().requires_grad_();b=b.detach().requires_grad_();u=u.detach().requires_grad_()
    y=fn(k,a,b);ga,gb=torch.autograd.grad(y,(a,b),u,create_graph=True)
    second=torch.autograd.grad((ga*da).sum()+(gb*db).sum(),(a,b,u))
    return cpu(dict(zip(('output','vjp_a','vjp_b','mixed_a','mixed_b','mixed_cotangent'),(y,ga,gb,*second))))
def main():
    configure();seed();out=ROOT/'results'/('local-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir();records=[]
    def check(name,k,a,b,u,identity):
        da=torch.randn_like(a);db=torch.randn_like(b)
        r=collect(bounded.product,k,a,b,u,da,db);repeat=collect(bounded.product,k,a,b,u,da,db);v=collect(product,k,a,b,u,da,db)
        path=out/f'{len(records):03d}.pt';torch.save({'reference':r,'native_repeat':repeat,'candidate':v},path)
        records.append({'name':name,'path':str(path),'sha256':sha(path),'identity':identity});print(name,flush=True)
        del r,repeat,v;gc.collect();torch.cuda.empty_cache()
    source=__import__('pathlib').Path((ROOT/'CAPTURE-RESULT.txt').read_text().strip())
    for index,row in enumerate(json.loads((source/'CAPTURE.json').read_text())['records']):
        assert sha(row['path'])==row['sha256'];values=torch.load(row['path'],map_location='cpu',weights_only=False);v={}
        for name,x in values.items():
            v[name]=torch.empty_strided(x.shape,row['strides'][name],device='cuda',dtype=torch.float32).copy_(x.cuda())
        check(f'actual-{index}',v['k'],v['a'],v['b'],v['g'],row);del v,values
    for l,n,h in ((19,1,3),(19,63,7),(19,64,32),(19,65,32),(25,17,5),(25,65,16)):
        k=torch.randn(l,l,l,device='cuda')*.03
        a=torch.randn(n,l,h*2,device='cuda')[...,::2];b=torch.randn(n,l,h*2,device='cuda')[...,1::2]
        u=torch.randn(n,l,h,device='cuda')
        check(f'generic-{l}-{n}-{h}',k,a,b,u,{'shape':[l,n,h],'strided':True})
    save(out/'MANIFEST.json',{'records':records,'package_sha256':package_sha(),'operator_source_sha256':sha(__import__('ictc_o2_hp.bilinear_triton',fromlist=['__file__']).__file__),'capture':str(source),'roles_per_case':6})
    (ROOT/'LOCAL-RESULT.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
