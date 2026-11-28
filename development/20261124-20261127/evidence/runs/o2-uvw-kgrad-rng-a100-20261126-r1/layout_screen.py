"""Actual four channel shapes and both real adjoints; no performance claim."""
import gc,json,os,sys
import torch
from context import ROOT,BASE,SOURCE,configure,cpu,save,sha
sys.path.insert(0,str(BASE))
from real_regression import collect
from ictc_o2_hp.uvw_triton import canonical_linear
def main():
    configure();out=ROOT/'results'/('layout-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir()
    source=__import__('pathlib').Path((BASE/'CAPTURE-RESULT.txt').read_text().strip());capture=json.loads((source/'CAPTURE.json').read_text());records=[]
    for row in capture['backends']['native'][:4]:
        raw=torch.load(row['path'],map_location='cpu',weights_only=False);x=raw['x'].cuda();w=raw['K'].cuda()
        for phase,v in raw['adjoints'].items():
            g=v['g'].cuda();r=collect(torch.nn.functional.linear,x,w,g);a=collect(canonical_linear,x,w,g)
            direct=g.reshape(-1,g.shape[-1]).T@x.reshape(-1,x.shape[-1]);swapped=(x.reshape(-1,x.shape[-1]).T@g.reshape(-1,g.shape[-1])).T
            path=out/f'{row["index"]:02d}-{phase}.pt';torch.save({'reference':r,'candidate':a,'direct_torch':cpu(direct),'swapped_torch':cpu(swapped)},path)
            records.append({'path':str(path),'sha256':sha(path),'weight_shape':list(w.shape),'phase':phase,'site':row['name']})
            del g,r,a,direct,swapped
        del raw,x,w;gc.collect();torch.cuda.empty_cache()
    save(out/'MANIFEST.json',{'records':records,'operator_source_sha256':sha(SOURCE/'ictc_o2_hp/uvw_triton.py')});(ROOT/'LAYOUT-RESULT.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
