"""Serial GPU continuity campaign, gated by the complete independent AMD matrix."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(os.environ['O2_SWEEP_ROOT']);control=Path(__file__).parent
gate=json.loads((root/'gate-audits/120876/RESULT.json').read_text())
operators=[r for r in gate['rows'] if r.get('identity',{}).get('level')=='operator']
networks=[r for r in gate['rows'] if r.get('level')=='network']
assert len(operators)==144 and all(r['status']=='PASS' for r in operators)
assert len(networks)==9 and all(r['infer']==r['train']=='PASS' for r in networks)
local=json.loads((root/'local-audits/120877/RESULT.json').read_text())
assert local['status']=='PASS' and len(local['rows'])==240
expected=json.loads(os.environ['O2_REPAIR_SOURCE_RECORD'])['package_sha256']
captured=json.loads((Path(gate['source'])/'network-L7-candidate/RESULT.json').read_text())
assert captured['identity']['source_package_sha256']==expected
out=root/'stability-captures'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True)
groups=[];audits=[]
def run(order,backend,mode,repeat=0,warm=None):
    assert not (out/'STOP.json').exists(),'An independent audit reported a failure'
    key=f'L{order}-{mode}-{backend}'+(f'-R{repeat}' if mode=='infer' else '')
    folder=out/key;folder.mkdir();started=time.time()
    args=[sys.executable,'-u',str(control/'stability_worker.py'),'--order',str(order),'--backend',backend,'--mode',mode,
          '--repeat',str(repeat),'--steps','20','--calls','10','--output',str(folder)]
    if warm is not None:args.extend(['--warm',str(warm)])
    with (folder/'process.log').open('w') as log:
        process=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
    result=json.loads((folder/'RESULT.json').read_text()) if (folder/'RESULT.json').exists() else {'status':'PROCESS_ERROR'}
    print('STABILITY_CASE',key,result['status'],time.time()-started,flush=True)
    assert process.returncode==0 and result['status']=='CAPTURED',(key,result)
    return str(folder)
for order in [7,8,1,2,3,4,5,6]:
    training={b:run(order,b,'train') for b in ('native','native-repeat','candidate')}
    inference={str(rep):{b:run(order,b,'infer',rep,Path(training[b])/'WARM.pt')
                        for b in ('native','native-repeat','candidate')} for rep in range(3)}
    group={'order':order,'training':training,'inference':inference,'source_package_sha256':expected,
           'stop_path':str(out/'STOP.json'),'steps':20,'calls':10,'fresh_processes':3}
    manifest=out/f'L{order}-MANIFEST.json';manifest.write_text(json.dumps(group,indent=2)+'\n');groups.append(str(manifest))
    job=subprocess.check_output(['sbatch','--parsable','--mem=64G','--time=02:00:00','--export=ALL,CUDA_VISIBLE_DEVICES=',
                                 'cpu.sbatch','continuity','stability_audit.py','--manifest',str(manifest)],cwd=root,text=True).strip()
    audits.append({'order':order,'job':job});print('STABILITY_AUDIT_SUBMITTED',order,job,flush=True)
    (out/'RESULT.json').write_text(json.dumps({'status':'CAPTURED_AUDIT_PENDING','groups':groups,'audits':audits},indent=2)+'\n')
(root/'STABILITY-CAPTURE.txt').write_text(str(out)+'\n')
print('STABILITY_CAPTURE_FINISHED',out,flush=True)
