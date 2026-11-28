"""Read actual CPU trace shape records to identify channel-GEMM roles."""
import collections,hashlib,json,os
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
source=Path(os.environ.get('O2_TRACE_SOURCE',(ROOT/'PERFORMANCE-RESULT.txt').read_text().strip() if (ROOT/'PERFORMANCE-RESULT.txt').exists() else str(ROOT.parent/'o2-uvw-fp32-repair-a100-20261126-r1/results/performance-a100-119450')))
rows=[]
for mode in ('ef','train'):
    for backend in ('native','c05','mask','candidate'):
        p=source/f'{mode}-{backend}-trace/trace.json';data=json.loads(p.read_text());counter=collections.Counter()
        for e in data['traceEvents']:
            if e.get('name') in ('_Matmul','_MatmulBackward','aten::linear','aten::mm','aten::matmul','aten::bmm'):
                counter[(e['name'],json.dumps(e.get('args',{}).get('Input Dims'),sort_keys=True))]+=1
        rows.append({'mode':mode,'backend':backend,'counts':[{'name':k[0],'input_dims':json.loads(k[1]),'calls':v} for k,v in counter.items()]})
        del data
weight=[]
for row in rows:
    selected=[];length=63024 if row['mode']=='ef' else 10504
    for r in row['counts']:
        dims=r['input_dims']
        if r['name'] not in (('aten::mm','aten::bmm') if row['backend']=='native' else ('_Matmul',)) or not isinstance(dims,list) or len(dims)<2:continue
        a,b=dims[:2]
        if not isinstance(a,list) or not isinstance(b,list):continue
        if len(a)==len(b)==3 and a[0]==b[0]==1:a,b=a[1:],b[1:]
        if len(a)==len(b)==2 and a[1]==b[0]==length and (a[0],b[1]) in ((256,1024),(192,768),(512,128),(384,96),(1024,256),(768,192),(128,512),(96,384)):
            selected.append(r)
    weight.append({'mode':row['mode'],'backend':row['backend'],'long_K':length,'weight_GEMM_calls':sum(r['calls'] for r in selected),'records':selected})
r={'trace_source':str(source),'rows':rows,'weight_GEMM':weight,'script_sha256':hashlib.file_digest(Path(__file__).open('rb'),'sha256').hexdigest(),
    'scope':'actual CPU op dimensions and counts; each native mm/bmm or candidate _Matmul counted once; no standalone kernel time/correctness claim'}
out=ROOT/('TRACE-SHAPES-'+os.environ['SLURM_JOB_ID']+'.json');out.write_text(json.dumps(r,indent=2)+'\n');(ROOT/'TRACE-SHAPES-LATEST.txt').write_text(str(out)+'\n');print(json.dumps({'source':str(source),'weight_GEMM':weight}),flush=True)
