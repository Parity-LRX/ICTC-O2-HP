import gc,json,os
from pathlib import Path
from support import *
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
torch.set_num_threads(4)
OUT=ROOT/'results'/('cpu-'+os.environ['SLURM_JOB_ID']);OUT.mkdir(parents=True)
old=json.loads((SWEEP/'gate-audits/120428/RESULT.json').read_text())
receipts={};rows=[]
for order in (2,4,6,7):
    audit=next(v for v in old['rows'] if v.get('level')=='network' and v['order']==order)
    targets=sorted({key.split('/training/updated/')[1] for step,v in audit['comparisons'].items() for key in v.get('failures',{}) if key.startswith('/training/updated/')})
    obj=cpu_model(order);groups=group_names(obj);nameids={}
    initial=load(SWEEP/f'gate-captures/120425/network-L{order}-native/INITIAL.pt')
    for names,group in zip(groups,initial['optimizer']['param_groups']):
        assert len(names)==len(group['params'])
        nameids.update(zip(names,group['params']))
    named=dict(obj.model.named_parameters());row={'order':order,'targets':targets,'steps':[],'optimizer_groups':[{k:v for k,v in g.items() if k!='params'} for g in initial['optimizer']['param_groups']]}
    histories={b:initial for b in ('native','candidate')}
    for step in range(3):
        values={}
        for backend in ('native','candidate'):
            folder=SWEEP/f'gate-captures/120425/network-L{order}-{backend}'
            rec=json.loads((folder/'RESULT.json').read_text());path=folder/f'step{step}.pt'
            actual=sha(path);assert actual==rec['files'][path.name];receipts[str(path)]=actual
            values[backend]=load(path)
        n,ca=(values[b] for b in ('native','candidate'))
        details={'step_zero_based':step,'full_record':brief(tree(ca,n)),
                 'raw_gradients':brief(tree(ca['training']['gradients'],n['training']['gradients'])),
                 'mixed':brief(tree(ca['mixed'],n['mixed'])),'targets':{}}
        for name in targets:
            info={'initial':{'max_abs':float(initial['model'][name].abs().max()),'norm':float(initial['model'][name].norm())},'raw_gradient_difference':metric(ca['training']['gradients'][name],n['training']['gradients'][name]),'updated_difference':metric(ca['training']['updated'][name],n['training']['updated'][name])}
            for backend in ('native','candidate'):
                tr=values[backend]['training'];pid=nameids[name];state=tr['optimizer']['state'][pid]
                group=next(g for g in tr['optimizer']['param_groups'] if pid in g['params'])
                grad=tr['gradients'][name];prior=histories[backend]
                info[backend]={'gradient_norm':float(grad.norm()),'gradient_max_abs':float(grad.abs().max()),
                               'updated_norm':float(tr['updated'][name].norm()),'first_moment_norm':float(state['exp_avg'].norm()),
                               'second_moment_max':float(state['exp_avg_sq'].max()),'step':float(state['step']),
                               'lr_used':next(g['lr'] for g in prior['optimizer']['param_groups'] if pid in g['params']),
                               'eps':group['eps'],'beta1':group['betas'][0],'beta2':group['betas'][1]}
            details['targets'][name]=info
        # CPU replay is an auxiliary control; exact CUDA replay follows separately.
        for backend in ('native','candidate'):
            prior=histories[backend];obj.model.load_state_dict(prior['model'],strict=True)
            obj.optimizer.load_state_dict(prior['optimizer']);obj.ema.load_state_dict(prior['ema'])
            for name,p in named.items():p.grad=values[backend]['training']['gradients'][name].clone()
            norm=torch.nn.utils.clip_grad_norm_(obj.model.parameters(),100)
            obj.optimizer.step();obj.ema.update()
            details[backend+'_cpu_replay']={'clip_total_norm':float(norm),'targets':{name:metric(named[name].detach(),values[backend]['training']['updated'][name]) for name in targets}}
            histories[backend]={'model':dict(initial['model'],**values[backend]['training']['updated']),
                                'optimizer':values[backend]['training']['optimizer'],'ema':values[backend]['training']['ema']}
        row['steps'].append(details);save(OUT/'PROGRESS.json',rows+[row]);print('ORDER_STEP',order,step,details['full_record']['failures'].keys(),flush=True)
        del n,ca,values;gc.collect()
    rows.append(row);del obj,initial,histories,named;gc.collect()
save(OUT/'RESULT.json',{'status':'ARCHIVED_ARRAYS_ANALYZED','rows':rows,'verified_source_hashes':receipts,'precision':'FP32 statistics; original thresholds'})
(ROOT/'CPU-RESULT.txt').write_text(str(OUT)+'\n')
