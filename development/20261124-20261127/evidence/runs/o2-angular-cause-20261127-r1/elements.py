import gc,json,os
from support import *
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
cpu=json.loads((ROOT/'results/cpu-120612/RESULT.json').read_text());rows=[]
for case in cpu['rows']:
    order=case['order'];obj=cpu_model(order);initial=load(SWEEP/f'gate-captures/120425/network-L{order}-native/INITIAL.pt')
    ids={name:pid for names,group in zip(group_names(obj),initial['optimizer']['param_groups']) for name,pid in zip(names,group['params'])}
    for step in (1,2):
        values={b:load(SWEEP/f'gate-captures/120425/network-L{order}-{b}/step{step}.pt')['training'] for b in ('native','candidate')}
        for name in case['targets']:
            n,ca=values['native']['updated'][name],values['candidate']['updated'][name];d=ca-n
            indices=d.abs().flatten().argsort(descending=True)[:5];row={'order':order,'step':step,'name':name,'metric':metric(ca,n),'elements':[]}
            for i in indices:
                i=int(i);r={'index':i,'difference':float(d.flatten()[i])}
                for backend in ('native','candidate'):
                    v=values[backend];state=v['optimizer']['state'][ids[name]]
                    group=next(g for g in v['optimizer']['param_groups'] if ids[name] in g['params'])
                    m=state['exp_avg'].flatten()[i];v2=state['exp_avg_sq'].flatten()[i];t=float(state['step'])
                    corrected_m=m/(1-group['betas'][0]**t);corrected_rms=(v2/(1-group['betas'][1]**t)).sqrt()
                    r[backend]={'raw_gradient':float(v['gradients'][name].flatten()[i]),'m':float(m),'v':float(v2),
                                'mhat':float(corrected_m),'sqrt_vhat':float(corrected_rms),'eps':group['eps'],
                                'normalized_adam_direction':float(corrected_m/(corrected_rms+group['eps'])),
                                'updated':float(v['updated'][name].flatten()[i])}
                row['elements'].append(r)
            rows.append(row)
        del values;gc.collect()
    del obj,initial;gc.collect()
save(ROOT/'results/ELEMENTS.json',{'rows':rows,'statistics_dtype':'float32','purpose':'per-component native Adam sensitivity; not a changed optimizer'})
for r in rows:
    if not r['metric']['passed']:print(r['order'],r['step'],r['name'],r['elements'][0],flush=True)
