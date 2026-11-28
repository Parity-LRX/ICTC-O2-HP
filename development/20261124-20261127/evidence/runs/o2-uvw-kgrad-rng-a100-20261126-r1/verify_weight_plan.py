import json,os
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
plans=json.loads((ROOT/'WEIGHT-PLANS.json').read_text());summary=[]
for key,parts in (('256x1024',4),('192x768',3),('512x128',5),('384x96',9)):
    span=(10504+parts-1)//parts;groups=[]
    for p in range(parts):
        chains=[{'chain':[p*span+q for q in range(span) if q%32//8==lane and p*span+q<10504]} for lane in range(4)]
        groups.append({'add':[{'add':chains[:2]},{'add':chains[2:]}]})
    expected=groups[0]
    for group in groups[1:]:expected={'add':[expected,group]}
    assert expected==plans[key],key
    summary.append({'shape':key,'parts':parts,'span':span,'chain_rule':'four chains, k%32 in successive groups of8, ascending k',
        'partial_sum':'(c0+c1)+(c2+c3) explicit FP32 adds','global_sum':'ascending-part left fold','all10504_products_covered_once':True})
(ROOT/'WEIGHT-PLAN-VERIFIED.json').write_text(json.dumps({'status':'EXACT_ARITHMETIC_PLAN_MATCH','plans':summary},indent=2)+'\n');print(json.dumps(summary),flush=True)
