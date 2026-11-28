import gc,json,os
from support import *
from metrics import ulp
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
out=ROOT/'results'/('supplement-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
axis=Path((ROOT/'AXIS-RESULT.txt').read_text().strip());rec=json.loads((axis/'RESULT.json').read_text());axis_rows=[]
for row in rec['rows']:
    arrays=load(axis/f'L{row["order"]}-PLAN-OUTPUTS.pt');checks=[]
    assert len(arrays['plans'])==len(row['plans'])
    for plan,value in zip(row['plans'],arrays['plans']):
        m=metric(value,arrays['native_saved'])
        assert all(m[k]==plan['metric'][k] for k in ('passed','component_fail_count','different_elements'))
        checks.append({**plan,'metric':m,'gpu_metric':plan['metric'],'ulp':ulp(value,arrays['native_saved'])})
    axis_rows.append({'order':row['order'],'native_reconstruction':metric(arrays['native_reconstructed'],arrays['native_saved']),
                      'plans':checks,'exact_plans':[r for r in checks if r['metric']['different_elements']==0]})
    del arrays;gc.collect()
save(out/'AXIS-AMD.json',{'status':'FULL_PLAN_OUTPUTS_RECOMPUTED','source':str(axis),'rows':axis_rows})
legacy=ROOT/'results/peak-120617/native-legacy';cleared=ROOT/'results/peak-120617/native-cache-cleared'
rn,rc=load(legacy/'before_train_rng.pt'),load(cleared/'before_train_rng.pt')
assert all(torch.equal(rn[k],rc[k]) for k in ('cpu','cuda'))
assert rn['python']==rc['python'] and all(c.np.array_equal(x,y) for x,y in zip(rn['numpy'],rc['numpy']))
reference=load(cleared/'TRAINING.pt');comparisons={}
for label,path in [('native_train_only',ROOT/'results/gpu-120614/oom-native-train-only/TRAINING.pt'),
                   ('candidate_same_legacy',ROOT/'results/gpu-120614/oom-candidate-legacy/TRAINING.pt')]:
    value=load(path);m=tree(value,reference);save(out/f'L8-{label}-METRICS.json',m);comparisons[label]=brief(m);del value;gc.collect()
save(out/'L8-CACHE-CONTROL.json',{'same_original_protocol_before_train_rng':True,'old_oom_status_preserved':True,
     'control':'native original protocol with gc.collect + empty_cache only before train; weights/loss/RNG/derivatives unchanged',
     'comparisons':comparisons})
save(out/'RESULT.json',{'status':'SUPPLEMENT_AUDITED','axis_source':str(axis),'axis_rows':axis_rows,'L8':comparisons})
(ROOT/'SUPPLEMENT-RESULT.txt').write_text(str(out)+'\n')
print('AXIS',[(r['order'],len(r['exact_plans'])) for r in axis_rows],flush=True);print('L8',comparisons,flush=True)
