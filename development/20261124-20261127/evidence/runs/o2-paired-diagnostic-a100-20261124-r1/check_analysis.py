import ast,json,os
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert os.environ.get('SLURM_JOB_ID')
ast.parse((ROOT/'analyze.py').read_text());print('ANALYSIS_SYNTAX_PASS',flush=True)
paths=list((ROOT/'results/a100-118576').glob('*-trace/trace.json'))
if paths:
    path=paths[0];obj=json.loads(path.read_text());events=obj['traceEvents']
    print('TRACE',path,'categories',dict(Counter(e.get('cat') for e in events)),flush=True)
    for cat in ('kernel','cuda_runtime','cpu_op','user_annotation','gpu_memcpy','gpu_memset'):
        subset=[e for e in events if e.get('cat')==cat]
        print('SCHEMA',cat,json.dumps(subset[0] if subset else {},ensure_ascii=False)[:1600],flush=True)
    print('TRITON_NAMES',Counter(e['name'] for e in events if e.get('cat')=='kernel' and ('_forward'==e['name'] or len(e['name'])<100)),flush=True)
