import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ['O2_SWEEP_ROOT']);CONTROL=Path(__file__).parent;OUT=ROOT/'gate-captures'/os.environ['SLURM_JOB_ID'];OUT.mkdir(parents=True)
previous=Path((ROOT/'GATE-CAPTURE.txt').read_text().strip());rows=json.loads((previous/'MANIFEST.json').read_text())['rows'].copy()
for order in range(1,9):
    for family,parities in (('uvw',['canonical','full']),('bilinear',['canonical'])):
        for parity in parities:
            for size in (16,256,4096):
                for mode in ('infer','train'):
                    identity={'level':'operator','family':family,'order':order,'parity':parity,'size':size,'mode':mode};key=f'op-{family}-{parity}-L{order}-N{size}-{mode}'
                    folder=OUT/key;folder.mkdir();cmd=[sys.executable,'-u',str(CONTROL/'gate_worker.py'),*[v for k,x in identity.items() for v in ('--'+k,str(x))],'--output',str(folder)];start=time.time()
                    with (folder/'process.log').open('w') as log:
                        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                    rec=json.loads((folder/'RESULT.json').read_text());rows.append({'key':key,'folder':str(folder),'identity':identity,'elapsed_s':time.time()-start,'status':rec['status']})
                    (OUT/'PROGRESS.json').write_text(json.dumps(rows,indent=2)+'\n');print(key,rec['status'],flush=True)
(OUT/'MANIFEST.json').write_text(json.dumps({'job':os.environ['SLURM_JOB_ID'],'control':str(CONTROL),'reused_network_manifest':str(previous/'MANIFEST.json'),'rows':rows},indent=2)+'\n')
(ROOT/'EXTRA-CAPTURE.txt').write_text(str(OUT)+'\n')
subprocess.run([sys.executable,'-u',str(CONTROL/'local_regress.py')],check=True)
