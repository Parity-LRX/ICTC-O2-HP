import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[1];assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
plans=json.loads((WS/'ICTC-O2-HP/src/ictc_o2_hp/fp32_reduction_plans.json').read_text())['plans']
inventory=Path((ROOT/'INVENTORY-RESULT.txt').read_text().strip());shapes=json.loads((inventory/'RESULT.json').read_text())['shapes'];rows=[]
for row in shapes:
    if row['kind']=='mm':
        left,right=row['shapes'];ls,rs=row['strides'];keys=['mm|'+','.join(map(str,(*left,right[1],*ls,*rs)))]
    else:
        k,a,b=row['shapes'];ks,astride,bstride=row['strides'];n=a[0];counts=([64] if n>=64 else [])+([n%64] if n%64 else [])
        keys=['angular|'+','.join(map(str,(*k,a[-1],count,*ks,*astride,*bstride))) for count in counts]
    rows.append({'input':row,'keys':keys,'missing':[key for key in keys if key not in plans]})
result={'status':'PASS' if all(not r['missing'] for r in rows) else 'MISSING','rows':rows,'actual_shapes':len(rows),'missing_keys':sorted({k for r in rows for k in r['missing']})}
(ROOT/'ACTUAL-COVERAGE.json').write_text(json.dumps(result,indent=2)+'\n');print('ACTUAL_COVERAGE',result['status'],len(rows),result['missing_keys'],flush=True)
