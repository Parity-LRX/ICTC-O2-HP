import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ['O2_SWEEP_ROOT']);CONTROL=Path(__file__).parent;SMOKE='--smoke' in sys.argv
OUT=ROOT/'gate-captures'/os.environ['SLURM_JOB_ID'];OUT.mkdir(parents=True)
rows=[]
def run(key,args,identity):
    folder=OUT/key;folder.mkdir(parents=True);start=time.time()
    command=[sys.executable,'-u',str(CONTROL/'gate_worker.py'),*args,'--output',str(folder)]
    with (folder/'process.log').open('w') as log:
        try:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            state=json.loads((folder/'RESULT.json').read_text()) if (folder/'RESULT.json').exists() else {'status':'PROCESS_ERROR','exit_code':result.returncode}
        except subprocess.TimeoutExpired:state={'status':'TIMEOUT','limit_s':1800}
    row={'key':key,'folder':str(folder),'identity':identity,'elapsed_s':time.time()-start,'status':state['status']};rows.append(row)
    (OUT/'PROGRESS.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
for order in ([1] if SMOKE else range(1,9)):
    for family,parities in (('uvw',['canonical'] if SMOKE else ['canonical','full']),('bilinear',['canonical'])):
        for parity in parities:
            for size in ([16] if SMOKE else [16,256,4096]):
                for mode in ('infer','train'):
                    identity={'level':'operator','family':family,'order':order,'parity':parity,'size':size,'mode':mode}
                    run(f'op-{family}-{parity}-L{order}-N{size}-{mode}',[v for k,x in identity.items() for v in ('--'+k,str(x))],identity)
for order in ([1] if SMOKE else [0,*range(1,9)]):
    for backend in ('native','native-repeat','candidate'):
        identity={'level':'network','order':order,'backend':backend}
        args=['--level','network','--order',str(order),'--backend','native' if backend.startswith('native') else 'candidate']
        run(f'network-L{order}-{backend}',args,identity)
(OUT/'MANIFEST.json').write_text(json.dumps({'smoke':SMOKE,'job':os.environ['SLURM_JOB_ID'],'control':str(CONTROL),'rows':rows},indent=2)+'\n')
(ROOT/('SMOKE-CAPTURE.txt' if SMOKE else 'GATE-CAPTURE.txt')).write_text(str(OUT)+'\n')
print('ALL_CAPTURE_ATTEMPTS_FINISHED',flush=True)
