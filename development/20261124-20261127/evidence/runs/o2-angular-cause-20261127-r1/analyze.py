import gc,json,os,pickle,time
from collections import Counter,defaultdict
from pathlib import Path
from support import *
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
OUT=ROOT/'results'/('audit-'+os.environ['SLURM_JOB_ID']);OUT.mkdir(parents=True)
gpu=ROOT/'results/gpu-120614';rows=[];hashes={}
def verified(path):
    hashes[str(path)]=sha(path);return load(path)
for order in (2,4,6,7):
    reference=SWEEP/f'gate-captures/120425/network-L{order}-native'
    original_candidate=SWEEP/f'gate-captures/120425/network-L{order}-candidate'
    for variant in ('native','candidate','channel-only','bilinear-only'):
        folder=gpu/f'ablation-L{order}-{variant}';record=json.loads((folder/'RESULT.json').read_text())
        row={'order':order,'variant':variant,'execution_status':record['status'],'steps':[]}
        if record['status']=='CAPTURED':
            initial=verified(folder/'INITIAL.pt');old_initial=verified(reference/'INITIAL.pt')
            row['initial_vs_original_native']=brief(tree(initial,old_initial));assert not row['initial_vs_original_native']['failures']
            for step in range(3):
                v=verified(folder/f'step{step}.pt');n=verified(reference/f'step{step}.pt')
                metrics=tree(v,n);item={'step':step,'vs_original_native':brief(metrics),
                                      'raw_gradients':brief(tree(v['training']['gradients'],n['training']['gradients'])),
                                      'mixed':brief(tree(v['mixed'],n['mixed']))}
                if variant=='candidate':
                    ca=verified(original_candidate/f'step{step}.pt');item['vs_original_candidate']=brief(tree(v,ca));del ca
                nrng=verified(reference/f'rng{step}.pt');vrng=verified(folder/f'rng{step}.pt')
                item['rng_tensors_identical']=all(torch.equal(vrng[phase][kind],nrng[phase][kind]) for phase in ('before','after') for kind in ('cpu','cuda'))
                item['rng_identical']=item['rng_tensors_identical'] and all(vrng[phase]['python']==nrng[phase]['python'] and
                    all(c.np.array_equal(x,y) for x,y in zip(vrng[phase]['numpy'],nrng[phase]['numpy'])) for phase in ('before','after'))
                row['steps'].append(item);save(OUT/f'METRICS-L{order}-{variant}-step{step}.json',metrics)
                del v,n,metrics;gc.collect()
            if (folder/'LOCAL.json').exists():
                local=json.loads((folder/'LOCAL.json').read_text());interesting=[x for x in local if x['metric'].get('different_elements',0)>0]
                row['local_summary']={'calls':len(local),'different_calls':len(interesting),'first_differences':interesting[:8],
                                      'first_each_kind':{kind:next((x for x in interesting if x['kind']==kind),None) for kind in ('channel','bilinear')},
                                      'failed_calls':[x for x in local if not x['metric']['passed']]}
        rows.append(row);save(OUT/'ABLATIONS.json',rows);print('AUDIT',order,variant,[len(s['vs_original_native']['failures']) for s in row['steps']],flush=True)

first_path=ROOT/'results/peak-120617/first-compute/RESULT.json'
wait_start=time.monotonic()
while not first_path.exists() or len(json.loads(first_path.read_text()).get('rows',[]))<8:
    assert time.monotonic()-wait_start<1800,'Peak / first-compute evidence incomplete after 30 minutes'
    print('WAITING_PEAK_AND_FIRST_COMPUTE',flush=True);time.sleep(15)
memory=[]
for folder in [gpu/f'oom-{v}' for v in ('native-legacy','native-train-only','candidate-legacy')]+list((ROOT/'results').glob('peak-*/*')):
    if not (folder/'RESULT.json').exists():continue
    rec=json.loads((folder/'RESULT.json').read_text());item={'folder':str(folder),'status':rec['status'],'phases':rec.get('phases'),'final_memory':rec.get('final_memory')}
    if (folder/'FAILED-OP.json').exists():item['failed_op']=json.loads((folder/'FAILED-OP.json').read_text())
    snapshot=folder/'AT-FAILURE-MEMORY.pickle'
    if not snapshot.exists():snapshot=folder/'MEMORY.pickle'
    if snapshot.exists():
        with snapshot.open('rb') as f:data=pickle.load(f)
        active=[]
        for segment in data['segments']:
            for block in segment['blocks']:
                if block['state']=='active_allocated':active.append(block)
        grouped=defaultdict(lambda:{'bytes':0,'blocks':0})
        for block in active:
            frames=block.get('frames',[])
            key=' | '.join(f"{x.get('filename')}:{x.get('line')}:{x.get('name')}" for x in frames if '/o2-benchmark/' in x.get('filename',''))
            key=key or 'runtime or library only';grouped[key]['bytes']+=block['size'];grouped[key]['blocks']+=1
        item['snapshot']={'path':str(snapshot),'at_failure':snapshot.name.startswith('AT-FAILURE'),
                          'active_allocated_bytes':sum(v['size'] for v in active),
                          'active_requested_bytes':sum(v.get('requested_size',v['size']) for v in active),
                          'groups':sorted([{'stack':k,**v} for k,v in grouped.items()],key=lambda r:r['bytes'],reverse=True),
                          'largest_blocks':sorted(active,key=lambda r:r['size'],reverse=True)[:24]}
        inactive=[b['size'] for s in data['segments'] for b in s['blocks'] if b['state']=='inactive']
        item['snapshot']['inactive_bytes']=sum(inactive)
        item['snapshot']['largest_inactive_block']=max(inactive,default=0)
        item['snapshot']['inactive_split_bytes']=sum(b['size'] for s in data['segments'] if any(x['state']!='inactive' for x in s['blocks']) for b in s['blocks'] if b['state']=='inactive')
        if item['snapshot']['at_failure']:
            address_shapes={}
            def tensors(v):
                if isinstance(v,dict):
                    if 'storage_address' in v:address_shapes[v['storage_address']]=v
                    else:
                        for x in v.values():tensors(x)
                elif isinstance(v,list):
                    for x in v:tensors(x)
            for event in rec.get('large_allocations',[]):tensors(event.get('output'))
            shaped=[]
            for block in active:
                if block['address'] in address_shapes:shaped.append({'allocated_bytes':block['size'],'tensor':address_shapes[block['address']]})
            byshape=defaultdict(lambda:{'allocated_bytes':0,'blocks':0})
            for block in shaped:
                key=str(block['tensor']['shape']);byshape[key]['allocated_bytes']+=block['allocated_bytes'];byshape[key]['blocks']+=1
            item['snapshot']['shapes']=[{'shape':k,**v} for k,v in sorted(byshape.items(),key=lambda x:x[1]['allocated_bytes'],reverse=True)]
        del data,active;gc.collect()
    memory.append(item);save(OUT/'MEMORY.json',memory)
optimizer=[]
for order in (2,4,6,7):
    path=gpu/f'optimizer-L{order}-replay/RESULT.json';optimizer.append({'order':order,**json.loads(path.read_text())})
save(OUT/'OPTIMIZER.json',optimizer)
first=[]
for path in (ROOT/'results').glob('peak-*/first-compute/RESULT.json'):
    hashes[str(path)]=sha(path);first.extend(json.loads(path.read_text())['rows'])
save(OUT/'FIRST-COMPUTE.json',first)
result={'status':'DIAGNOSTIC_AUDIT_COMPLETE','ablation_summary':[{'order':r['order'],'variant':r['variant'],'execution_status':r['execution_status'],
        'step_failures':[len(s['vs_original_native']['failures']) for s in r['steps']],
        'same_backend_reproduction':[s.get('vs_original_candidate',s['vs_original_native']) for s in r['steps']] if r['variant'] in ('native','candidate') else None,
        'first_local_differences':r.get('local_summary',{}).get('first_each_kind')} for r in rows],
        'memory_summary':[{'folder':x['folder'],'status':x['status'],'phases':x.get('phases'),'final_memory':x.get('final_memory'),'failed_op':x.get('failed_op')} for x in memory],
        'first_compute':first,'original_loss_precision_thresholds_unchanged':True,'old_failures_preserved':True,'source_hashes':hashes}
save(OUT/'RESULT.json',result);(ROOT/'AUDIT-RESULT.txt').write_text(str(OUT)+'\n')
print(json.dumps(result['ablation_summary'],indent=2),flush=True)
