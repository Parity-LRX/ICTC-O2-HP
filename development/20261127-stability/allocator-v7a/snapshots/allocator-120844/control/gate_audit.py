import gc,json,os,pickle,shutil,sys
from pathlib import Path
from common import ROOT,torch,sha,save
from metrics import full
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
SMOKE='--smoke' in sys.argv;source=Path((ROOT/os.environ.get('O2_GATE_CAPTURE_POINTER','SMOKE-CAPTURE.txt' if SMOKE else 'GATE-CAPTURE.txt')).read_text().strip())
manifest=json.loads((source/'MANIFEST.json').read_text());out=ROOT/'gate-audits'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True)
rows=[]
def load(folder,name,rec):
    path=folder/name;assert sha(path)==rec['files'][name];return torch.load(path,map_location='cpu',weights_only=False)
def check(a,r):
    m=full(a,r);return {'passed':all(v['passed'] for v in m.values()),'failures':{k:v for k,v in m.items() if not v['passed']},
                     'roles':len(m),'max_R':max((v.get('relative_l2',0) or 0) for v in m.values()),'max_ulp':max(v.get('ulp',{}).get('max_ulp',0) for v in m.values())}
def rngid(v):
    import hashlib
    return {k:hashlib.sha256(x.numpy().tobytes() if isinstance(x,torch.Tensor) else pickle.dumps(x)).hexdigest() for k,x in v.items()}
for item in manifest['rows']:
    if item['identity']['level']!='operator':continue
    folder=Path(item['folder']);rec=json.loads((folder/'RESULT.json').read_text()) if (folder/'RESULT.json').exists() else {'status':item['status']}
    row=dict(item)
    if rec['status']=='CAPTURED':
        reference=load(folder,'native.pt',rec);repeat=load(folder,'native-repeat.pt',rec);candidate=load(folder,'candidate.pt',rec)
        row['repeat']=check(repeat,reference);row['comparison']=check(candidate,reference)
        eps=[]
        def epsilon(a,r):
            if isinstance(r,dict):
                for k in r:epsilon(a[k],r[k])
            elif isinstance(r,(list,tuple)):
                for aa,rr in zip(a,r):epsilon(aa,rr)
            elif isinstance(r,torch.Tensor):eps.append(float((a-r).abs().max()/max(float(a.abs().max()),float(r.abs().max()),1e-7)))
        epsilon(candidate,reference);row['epsilon_max']=max(eps);row['status']='PASS' if row['repeat']['passed'] and row['comparison']['passed'] and row['epsilon_max']<=1e-5 else 'NUMERIC_FAIL'
        del reference,repeat,candidate;gc.collect()
    rows.append(row);print(row['key'],row['status'],flush=True)
for order in ([1] if SMOKE else [0,*range(1,9)]):
    items={v['identity']['backend']:v for v in manifest['rows'] if v['identity']['level']=='network' and v['identity']['order']==order}
    records={b:(Path(v['folder']),json.loads((Path(v['folder'])/'RESULT.json').read_text())) for b,v in items.items() if (Path(v['folder'])/'RESULT.json').exists()}
    row={'level':'network','order':order,'infer':'NOT_CAPTURED','train':'NOT_CAPTURED','comparisons':{}}
    if len(records)==3 and all('ef.pt' in r['files'] for _,r in records.values()):
        f,r=records['native'];reference=load(f,'ef.pt',r);checks=[]
        for b in ('native-repeat','candidate'):
            f,r=records[b];x=check(load(f,'ef.pt',r),reference);row['comparisons']['ef-'+b]=x;checks.append(x['passed'])
        del reference;gc.collect()
        f,r=records['native'];reference=load(f,'joint-vjp.pt',r)
        for b in ('native-repeat','candidate'):
            f,r=records[b];x=check(load(f,'joint-vjp.pt',r),reference);row['comparisons']['joint-vjp-'+b]=x;checks.append(x['passed'])
        row['infer']='PASS' if all(checks) else 'NUMERIC_FAIL';del reference;gc.collect()
        if row['infer']=='PASS' and not SMOKE:
            warm=ROOT/'gates'/f'network-L{order}';warm.mkdir(parents=True,exist_ok=True)
            f,r=records['native'];assert sha(f/'INITIAL.pt')==r['files']['INITIAL.pt'];shutil.copy2(f/'INITIAL.pt',warm/'INFER.pt')
    if len(records)==3 and all(r['status']=='CAPTURED' for _,r in records.values()):
        passed=row['infer']=='PASS';previous={}
        for step in range(3):
            f,r=records['native'];reference=load(f,f'step{step}.pt',r);native_rng=load(f,f'rng{step}.pt',r)
            for b in ('native-repeat','candidate'):
                f,r=records[b];v=load(f,f'step{step}.pt',r);x=check(v,reference);row['comparisons'][f'step{step}-{b}']=x;passed &= x['passed']
                states=load(f,f'rng{step}.pt',r);assert rngid(states['before'])==rngid(native_rng['before']) and rngid(states['after'])==rngid(native_rng['after'])
                if b in previous:assert rngid(previous[b])==rngid(states['before'])
                previous[b]=states['after'];del v;gc.collect()
            del reference;gc.collect()
        row['train']='PASS' if passed else 'NUMERIC_FAIL'
        if passed and not SMOKE:
            warm=ROOT/'gates'/f'network-L{order}';warm.mkdir(parents=True,exist_ok=True)
            f,r=records['native'];assert sha(f/'WARM.pt')==r['files']['WARM.pt'];shutil.copy2(f/'WARM.pt',warm/'WARM.pt')
    row['execution_status']={b:r['status'] for b,(_,r) in records.items()};rows.append(row);print('NETWORK',order,row['infer'],row['train'],flush=True)
save(out/'RESULT.json',{'status':'SMOKE_PASS' if SMOKE and all((r.get('status')=='PASS') if 'key' in r else r['infer']==r['train']=='PASS' for r in rows) else 'GATES_REVIEWED',
                     'source':str(source),'rows':rows,'job':os.environ['SLURM_JOB_ID'],'original_thresholds':True,'shared_true_loss_cotangents':False,'restored_inside_three_steps':False})
(ROOT/('SMOKE-AUDIT.txt' if SMOKE else 'GATE-AUDIT.txt')).write_text(str(out)+'\n')
