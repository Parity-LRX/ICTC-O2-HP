import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ['O2_CAUSE_ROOT']);CONTROL=Path(__file__).parent
OUT=ROOT/'results'/('gpu-'+os.environ['SLURM_JOB_ID']);OUT.mkdir(parents=True)
tasks=[('grid',8,'limits')]+[('oom',8,v) for v in ('native-legacy','native-train-only','candidate-legacy')]
tasks += [('optimizer',l,'replay') for l in (2,4,6,7)]
tasks += [('ablation',l,v) for l in (2,4,6,7) for v in ('native','candidate','channel-only','bilinear-only')]
rows=[]
for action,order,variant in tasks:
    name=f'{action}-{variant}' if action in ('grid','oom') else f'{action}-L{order}-{variant}'
    target=OUT/name;target.mkdir();cmd=[sys.executable,'-u',str(CONTROL/'gpu_worker.py'),action,'--order',str(order),'--variant',variant,'--output',str(target)]
    start=time.time()
    with (target/'process.log').open('w') as log:
        try:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=3600).returncode
        except subprocess.TimeoutExpired:code='TIMEOUT'
    result=json.loads((target/'RESULT.json').read_text()) if (target/'RESULT.json').exists() else {'status':'MISSING_RESULT'}
    rows.append({'action':action,'order':order,'variant':variant,'folder':str(target),'status':result['status'],'exit_code':code,'seconds':time.time()-start})
    (OUT/'MANIFEST.json').write_text(json.dumps({'rows':rows,'job':os.environ['SLURM_JOB_ID']},indent=2)+'\n');print(rows[-1],flush=True)
(ROOT/'GPU-RESULT.txt').write_text(str(OUT)+'\n')
