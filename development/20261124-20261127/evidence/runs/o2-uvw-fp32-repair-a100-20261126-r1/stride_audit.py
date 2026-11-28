import json,os
from pathlib import Path
import torch
from common import ROOT,save
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
source=Path((ROOT/'CAPTURE-RESULT.txt').read_text().strip());manifest=json.loads((source/'CAPTURE.json').read_text());rows=[]
for backend,sites in manifest['backends'].items():
    for site in sites:
        v=torch.load(site['path'],map_location='cpu',weights_only=False)
        rows.append({'backend':backend,'site':site['name'],'x_shape':list(v['x'].shape),
            'x_original_stride':v['x_original_stride'],'x_saved_stride':list(v['x'].stride()),
            'K_original_stride':v['K_original_stride'],'K_saved_stride':list(v['K'].stride()),
            'adjoints':{name:{'original_stride':a['stride'],'saved_stride':list(a['g'].stride()),'requires_grad_original':a['requires_grad']} for name,a in v['adjoints'].items()}})
save(ROOT/'STRIDE-AUDIT.json',rows);print(json.dumps(rows[:4]),flush=True)
