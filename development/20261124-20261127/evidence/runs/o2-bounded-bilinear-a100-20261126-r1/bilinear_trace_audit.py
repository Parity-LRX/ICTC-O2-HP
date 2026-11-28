import csv,json,os
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
shapes=json.loads(Path((ROOT/'TRACE-SHAPES-LATEST.txt').read_text().strip()).read_text())
audit=Path((ROOT/'PERFORMANCE-AUDIT.txt').read_text().strip());rows=[]
for v in shapes['rows']:
    selected=[]
    for c in v['counts']:
        d=c['input_dims']
        if c['name']=='aten::bmm' and isinstance(d,list) and len(d)>=2 and len(d[0])==3 and d[0][0]==1 and (d[0][1],d[0][2]) in ((19,361),(25,625)):
            selected.append(c)
    mode,backend=v['mode'],v['backend'];calls=sum(x['calls'] for x in selected)
    with (audit/f'traces/{mode}-{backend}-trace/KERNELS.csv').open() as f:kernels=list(csv.DictReader(f))
    fused=[k for k in kernels if k['kernel']=='_bilinear']
    if backend=='candidate':assert calls==0 and fused
    else:assert calls>0 and not fused
    rows.append({'mode':mode,'backend':backend,'old_angular_BMM_calls':calls,'fused_kernel_calls':sum(int(k['calls']) for k in fused),'fused_kernel_ms':sum(float(k['total_ms']) for k in fused),'old_angular_BMM_shapes':selected})
r={'status':'BILINEAR_FUSION_TRACE_PASS','rows':rows,'scope':'actual old 19x361/25x625 BMM calls eliminated; other network cat/bmm operations intentionally remain'}
(ROOT/'BILINEAR-TRACE-AUDIT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
