"""Original three-step schedule, phase RNG, counterfactuals and optimizer controls."""
import copy,gc,hashlib,json,os,pickle,random,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(os.environ['O2_TRAJECTORY_ROOT']);WS=ROOT.parents[1];REPAIR=WS/'runs/o2-uvw-fp32-repair-a100-20261126-r1'
os.environ['O2_REPAIR_ROOT']=str(REPAIR);os.environ['O2_REPAIR_SOURCE']=str(REPAIR/'snapshots/qualification-119423/ictc-source')
sys.path.insert(0,str(REPAIR/'snapshots/qualification-119423/control'))
from common import configure,factory,common_state,startup,cpu,save,sha,state_identity
def rng():return {'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate()}
def rng_id(r):return {k:hashlib.sha256(v.numpy().tobytes() if isinstance(v,torch.Tensor) else pickle.dumps(v)).hexdigest() for k,v in r.items()}
def set_rng(r):
    torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state(r['cuda']);np.random.set_state(r['numpy']);random.setstate(r['python'])
def main():
    configure();out=ROOT/'results'/('a100-'+os.environ['SLURM_JOB_ID']);out.mkdir()
    model=factory('v3','candidate','checkpoint_ema','cuda');initial,receipt=common_state('train');ef,efreceipt=common_state('ef')
    save(out/'STARTUP.json',startup(model,initial,receipt,'train'))
    original=[(m,f) for m,n,f in model.patches.saved if n=='forward'];candidate=[m.forward for m,_ in original]
    def select(case):
        for (m,f),new in zip(original,candidate):object.__setattr__(m,'forward',f if case.startswith('native') else new)
    def restore(state,r=None):
        model.restore(dict(state,optimizer=copy.deepcopy(state['optimizer'])))
        assert state_identity(model.state())==state_identity(state)
        if r is not None:set_rng(r);assert rng_id(rng())==rng_id(r)
    save(out/'TRAINING-FLAGS.json',{'scaler':str(model.trainer.scaler),'shared_parameters':bool(getattr(model.model,'shared_parameters',[])),
        'dropout':{n:{'type':type(m).__name__,'p':getattr(m,'p',None),'drop_prob':getattr(m,'drop_prob',None)} for n,m in model.model.named_modules() if 'Drop' in type(m).__name__},
        'rng_capture':'CPU torch, current CUDA generator, NumPy and Python; before each actual phase',
        'original_parameter_ids_unchanged':model.original_ids=={n:id(p) for n,p in model.model.named_parameters()}})
    phase='';frames=[];rot=model.model._init_edge_rot_mat
    def rotation(edge):
        before=rng_id(rng());value=rot(edge);v=cpu(value)
        frames.append({'phase':phase,'rng':before,'frame_sha256':hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest(),'shape':list(value.shape)})
        return value
    model.model._init_edge_rot_mat=rotation
    manifest={};states={};random_states={}
    for case in ('native','candidate','native-repeat'):
        select(case);restore(ef);phase=case+'-ef';model.infer();restore(initial);phase=case+'-vjp';model.vjp();restore(initial)
        for step in range(3):
            key=f'{case}-step{step}';state=model.state();states[(case,step)]=state
            mixed_rng=rng();phase=key+'-mixed';mixed=model.force_mixed()
            training_rng=rng();phase=key+'-train';training=model.train_step(capture=True);training['scheduler']=cpu(model.trainer.scheduler.scheduler.state_dict())
            after_rng=rng();phase=key+'-after';after=cpu(model.forward(model.batch('train'),False))
            random_states[(case,step)]={'mixed':mixed_rng,'train':training_rng,'after':after_rng}
            path=out/(key+'.pt');torch.save({'mixed':mixed,'training':training,'after_update_output':after},path);manifest[path.name]=sha(path)
            torch.save({'state':state,'rng':random_states[(case,step)]},out/(key+'-INPUT.pt'))
            save(out/(key+'-CONDITION.json'),{'state':state_identity(state),'rng':{k:rng_id(v) for k,v in random_states[(case,step)].items()}})
            del mixed,training,after;gc.collect();torch.cuda.empty_cache();print(key,flush=True)
    # Reproduce the initial five-point helper's different RNG phase with no weight update.
    select('native');restore(initial);phase='native-vjp-before-mixed';model.vjp();different_rng=rng();phase='native-same-theta-replay-RNG';mixed=model.force_mixed()
    path=out/'native-same-theta-replay-RNG.pt';torch.save({'mixed':mixed,'rng':different_rng},path);manifest[path.name]=sha(path);del mixed
    # At each original phase compare both backends at each trajectory's actual state.
    for step in range(3):
        for state_branch in ('native','candidate'):
            state=states[(state_branch,step)]
            for engine in ('native','candidate'):
                select(engine);key=f'cross-step{step}-{engine}-at-{state_branch}'
                restore(state,random_states[('native',step)]['mixed']);phase=key+'-mixed';mixed=model.force_mixed()
                restore(state,random_states[('native',step)]['train']);phase=key+'-train';training=model.train_step(capture=True);training['scheduler']=cpu(model.trainer.scheduler.scheduler.state_dict())
                restore_after_rng=random_states[('native',step)]['after'];set_rng(restore_after_rng);phase=key+'-after';after=cpu(model.forward(model.batch('train'),False))
                path=out/(key+'.pt');torch.save({'mixed':mixed,'training':training,'after_update_output':after},path);manifest[path.name]=sha(path)
                del mixed,training,after;gc.collect();torch.cuda.empty_cache();print(key,flush=True)
    # Pure native optimizer with the exact saved raw gradient arrays; no model backward.
    assert not model.trainer.scaler and not getattr(model.model,'shared_parameters',[])
    select('native')
    for label,branch in (('native-grads','native'),('native-grads-repeat','native'),('candidate-grads','candidate')):
        restore(initial);raw=torch.load(out/(branch+'-step0.pt'),map_location='cpu',weights_only=False)['training']['gradients']
        model.optimizer.zero_grad(set_to_none=True)
        for name,p in model.model.named_parameters():p.grad=None if raw[name] is None else raw[name].to(p.device).clone()
        gradnorm=torch.nn.utils.clip_grad_norm_(model.model.parameters(),max_norm=model.trainer.clip_grad_norm)
        model.optimizer.step();model.ema.update();model.trainer.scheduler.step()
        path=out/('optimizer-'+label+'.pt');torch.save({'state':model.state(),'grad_norm':cpu(gradnorm)},path);manifest[path.name]=sha(path)
        del raw;gc.collect();torch.cuda.empty_cache()
    # Actual native loss cotangents and same-cotangent causal controls, not a repair.
    targets=[('coordinates',None)]+[(n,p) for n,p in model.model.named_parameters() if p.requires_grad]
    for step in (1,2):
        native_full_cot=None;native_force_cot=None
        for branch in ('native','candidate'):
            select('native');state=states[(branch,step)];key=f'cot-step{step}-at-{branch}'
            restore(state,random_states[('native',step)]['mixed']);phase=key+'-force'
            batch=model.batch('train');batch.pos.requires_grad_(True);pred=model.forward(batch,True)
            actual_targets=(batch.pos,)+tuple(p for _,p in targets[1:]);names=[n for n,_ in targets]
            all_losses=model.trainer.loss_functions
            try:
                model.trainer.loss_functions=[row for row in all_losses if row[0]=='forces'];loss=model.loss(pred,batch)
            finally:model.trainer.loss_functions=all_losses
            cot=torch.autograd.grad(loss,pred['forces'],retain_graph=True)[0]
            if native_force_cot is None:native_force_cot=cot.detach().clone()
            own=torch.autograd.grad(pred['forces'],actual_targets,cot.detach(),retain_graph=True,allow_unused=True)
            shared=torch.autograd.grad(pred['forces'],actual_targets,native_force_cot,allow_unused=True)
            force={'output':cpu(pred['forces']),'cotangent':cpu(cot),'own':cpu(dict(zip(names,own))),'shared':cpu(dict(zip(names,shared)))}
            del pred,cot,own,shared,loss
            restore(state,random_states[('native',step)]['train']);phase=key+'-total'
            batch=model.batch('train');pred=model.forward(batch,True);loss=model.loss(pred,batch)
            keys=list(pred);cot=torch.autograd.grad(loss,tuple(pred[k] for k in keys),retain_graph=True)
            if native_full_cot is None:native_full_cot={k:v.detach().clone() for k,v in zip(keys,cot)}
            params=tuple(p for _,p in targets[1:]);param_names=[n for n,_ in targets[1:]]
            own=torch.autograd.grad(tuple(pred[k] for k in keys),params,tuple(v.detach() for v in cot),retain_graph=True,allow_unused=True)
            shared=torch.autograd.grad(tuple(pred[k] for k in keys),params,tuple(native_full_cot[k] for k in keys),allow_unused=True)
            total={'output':cpu(pred),'cotangent':cpu(dict(zip(keys,cot))),'own':cpu(dict(zip(param_names,own))),'shared':cpu(dict(zip(param_names,shared)))}
            path=out/(key+'.pt');torch.save({'force':force,'total':total},path);manifest[path.name]=sha(path)
            del pred,cot,own,shared,loss,force,total;gc.collect();torch.cuda.empty_cache()
    assert model.original_ids=={n:id(p) for n,p in model.model.named_parameters()}
    save(out/'FRAMES.json',frames);save(out/'MANIFEST.json',{'files':manifest,'operator_source_sha256':sha(Path(os.environ['O2_REPAIR_SOURCE'])/'ictc_o2_hp/uvw_triton.py'),
        'source':'frozen a80f0dc, native loss and optimizer unchanged; original phase-specific RNG replayed',
        'state_and_rng_equalities':{str(step):{part:rng_id(random_states[('native',step)][part])==rng_id(random_states[('candidate',step)][part]) for part in ('mixed','train','after')} for step in range(3)}})
    (ROOT/'GPU-RESULT.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
