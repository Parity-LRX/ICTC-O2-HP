import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ['O2_SWEEP_ROOT']);CONTROL=Path(__file__).parent;OUT=ROOT/'gate-captures'/os.environ['SLURM_JOB_ID'];OUT.mkdir(parents=True)
previous=Path((ROOT/'GATE-CAPTURE.txt').read_text().strip());rows=[r for r in json.loads((previous/'MANIFEST.json').read_text())['rows'] if r['identity'].get('order')!=8 or r['identity'].get('level')!='network']
for backend in ('native','native-repeat','candidate'):
    key=f'network-L8-{backend}';folder=OUT/key;folder.mkdir();identity={'level':'network','order':8,'backend':backend};start=time.time()
    cmd=[sys.executable,'-u',str(CONTROL/'gate_worker.py'),'--level','network','--order','8','--backend','native' if backend.startswith('native') else 'candidate','--output',str(folder)]
    with (folder/'process.log').open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    rec=json.loads((folder/'RESULT.json').read_text());rows.append({'key':key,'folder':str(folder),'identity':identity,'elapsed_s':time.time()-start,'status':rec['status']});print(key,rec['status'],flush=True)
(OUT/'MANIFEST.json').write_text(json.dumps({'job':os.environ['SLURM_JOB_ID'],'control':str(CONTROL),'previous_manifest_preserved':str(previous/'MANIFEST.json'),'rows':rows,'cache_policy':'native zero_grad moved before mixed/forward; no optimizer state restore'},indent=2)+'\n')
(ROOT/'GATE-CAPTURE.txt').write_text(str(OUT)+'\n')
