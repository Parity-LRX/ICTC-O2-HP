import json,os
from pathlib import Path
import torch
from common import ROOT,save
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
source=Path((ROOT/'ORDER-SIGNATURE.txt').read_text().strip());audit=Path((ROOT/'ORDER-AUDIT.txt').read_text().strip())
manifest=json.loads((source/'MANIFEST.json').read_text());results=json.loads((audit/'RESULT.json').read_text())['records'];rows=[]
for meta,result in zip(manifest['records'],results):
    v=torch.load(meta['path'],weights_only=False,map_location='cpu');k=v['a'].shape[1];r=v['reference'][:,0]
    patterns=[]
    for group in range(min(8,r.numel()//k)):
        values=r[group*k:(group+1)*k]
        patterns.append({'group':group,'first64':''.join(str(int(x)) for x in values[:64]),'ones':int((values==1).sum())})
    ranked=sorted(result['results'].items(),key=lambda row:row[1]['mismatch_elements'])
    rows.append({'site':result['site'],'role':result['role'],'a_shape':meta['a_shape'],'a_stride':meta['a_stride'],
        'b_shape':meta['b_shape'],'b_stride':meta['b_stride'],'best':ranked[:5],'native_patterns':patterns})
save(ROOT/'ORDER-SYNTHESIS.json',rows);print(json.dumps(rows),flush=True)
