import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ['O2_SWEEP_ROOT']);CONTROL=Path(__file__).parent;out=ROOT/'inventory'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True)
rows=[];all_shapes={}
for order in [0,*range(1,9)]:
    target=out/f'L{order}';target.mkdir()
    with (target/'process.log').open('w') as log:
        result=subprocess.run([sys.executable,'-u',str(CONTROL/'inventory_worker.py'),'--order',str(order),'--output',str(target)],stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    rec=json.loads((target/'RESULT.json').read_text());assert result.returncode==0
    rows.append({'order':order,'folder':str(target),'status':rec['status']})
    for record in rec['rows']:
        key=json.dumps({k:record[k] for k in ('kind','shapes','strides')},sort_keys=True);all_shapes[key]=record
    (out/'RESULT.json').write_text(json.dumps({'rows':rows,'shapes':list(all_shapes.values())},indent=2)+'\n');print('INVENTORY',order,len(rec['rows']),len(all_shapes),flush=True)
(ROOT/'INVENTORY-RESULT.txt').write_text(str(out)+'\n')
