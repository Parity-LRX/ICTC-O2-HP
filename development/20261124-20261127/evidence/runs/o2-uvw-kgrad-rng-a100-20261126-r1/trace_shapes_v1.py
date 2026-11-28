"""Read actual CPU trace shape records to identify channel-GEMM roles."""
import collections,json,os
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
source=Path(os.environ.get('O2_TRACE_SOURCE',(ROOT/'PERFORMANCE-RESULT.txt').read_text().strip() if (ROOT/'PERFORMANCE-RESULT.txt').exists() else str(ROOT.parent/'o2-uvw-fp32-repair-a100-20261126-r1/results/performance-a100-119450')))
rows=[]
for mode in ('ef','train'):
    for backend in ('native','candidate'):
        p=source/f'{mode}-{backend}-trace/trace.json';data=json.loads(p.read_text());counter=collections.Counter()
        for e in data['traceEvents']:
            if e.get('name') in ('_Matmul','_MatmulBackward','aten::linear','aten::mm','aten::matmul'):
                counter[(e['name'],json.dumps(e.get('args',{}).get('Input Dims'),sort_keys=True))]+=1
        rows.append({'mode':mode,'backend':backend,'counts':[{'name':k[0],'input_dims':json.loads(k[1]),'calls':v} for k,v in counter.items()]})
        del data
r={'trace_source':str(source),'rows':rows,'scope':'CPU op input dimensions and counts; not standalone kernel cost or correctness evidence'}
out=ROOT/('TRACE-SHAPES-'+os.environ['SLURM_JOB_ID']+'.json');out.write_text(json.dumps(r,indent=2)+'\n');print(str(out),flush=True)
