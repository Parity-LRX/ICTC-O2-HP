"""AMD comparison of accumulation signatures; never a model acceptance gate."""
import json,os
from pathlib import Path
import torch
from common import ROOT,save,sha
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
    source=Path((ROOT/'ORDER-SIGNATURE.txt').read_text().strip());manifest=json.loads((source/'MANIFEST.json').read_text())
    out=ROOT/'results'/('order-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True);records=[]
    for row in manifest['records']:
        assert sha(Path(row['path']))==row['sha256'];v=torch.load(row['path'],weights_only=False,map_location='cpu')
        r=v['reference'];mask=v['valid_rows']
        results={}
        for name,a in v['outputs'].items():
            bad=(a!=r)&mask[:,None]
            results[name]={'mismatch_elements':int(bad.sum()),'valid_elements':int(mask.sum())*r.shape[1],
                'mismatch_rows':int(bad.any(dim=1).sum()),'first_mismatch_indices':bad.nonzero()[:16].tolist()}
        records.append({'site':row['site'],'role':row['role'],'shape':list(r.shape),'results':results})
    save(out/'RESULT.json',{'status':'ORDER_SIGNATURE_ANALYZED_NOT_QUALIFICATION','records':records})
    (ROOT/'ORDER-AUDIT.txt').write_text(str(out)+'\n')
    print(json.dumps(records),flush=True)
if __name__=='__main__':main()
