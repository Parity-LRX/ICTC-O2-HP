"""Full-array regression using captured 10504-row activations and actual adjoints."""
import argparse,gc,json,os
from pathlib import Path
import torch
from context import ROOT,BASE,configure,save,sha,cpu
from ictc_o2_hp.uvw_triton import canonical_linear
def collect(fn,x,k,g):
    x=x.detach().requires_grad_();k=k.detach().requires_grad_();g=g.detach().requires_grad_()
    y=fn(x,k);dx,dk=torch.autograd.grad(y,(x,k),g,create_graph=True)
    direction=(dx*x.detach()).sum()+(dk*k.detach()).sum()
    second=torch.autograd.grad(direction,(x,k,g),allow_unused=False)
    return cpu({'forward':y,'input_adjoint':dx,'weight_adjoint':dk,
        'mixed_input':second[0],'mixed_weight':second[1],'mixed_cotangent':second[2]})
def main():
    configure();source=Path((BASE/'CAPTURE-RESULT.txt').read_text().strip())
    capture=json.loads((source/'CAPTURE.json').read_text());out=ROOT/'results'/('regression-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    records=[]
    for backend,rows in capture['backends'].items():
        for row in rows:
            v=torch.load(row['path'],map_location='cpu',weights_only=False);x=v['x'].cuda();k=v['K'].cuda()
            assert x.shape[0]*x.shape[1]==10504
            for phase,item in v['adjoints'].items():
                g=item['g'].cuda()
                reference=collect(torch.nn.functional.linear,x,k,g)
                repeat=collect(torch.nn.functional.linear,x,k,g)
                candidate=collect(canonical_linear,x,k,g)
                path=out/(backend+'-'+f"{row['index']:02d}"+'-'+phase+'.pt')
                torch.save({'reference':reference,'native_repeat':repeat,'candidate':candidate},path)
                records.append({'backend':backend,'site':row['name'],'phase':phase,'path':str(path),'sha256':sha(path),
                    'input_shape':list(x.shape),'weight_shape':list(k.shape),'input_stride':list(x.stride())})
                del reference,repeat,candidate,g
            del x,k,v
            gc.collect();torch.cuda.empty_cache();print(backend,row['index'],flush=True)
    save(out/'REGRESSION.json',{'status':'ACTUAL_10504_ROW_REGRESSION_ARRAYS_CAPTURED','records':records,
        'operator_source_sha256':sha(__import__('ictc_o2_hp.uvw_triton',fromlist=['__file__']).__file__),
        'capture_source_sha256':sha(source/'CAPTURE.json'),'reference':'original FP32 F.linear autograd; actual x,K,force/loss cotangents'})
    (ROOT/'REGRESSION-RESULT.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
