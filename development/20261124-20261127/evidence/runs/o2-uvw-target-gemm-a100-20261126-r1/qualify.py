"""Original independent trajectories and RNG-isolated diagnostic controls."""
import collections,gc,hashlib,json,os
import torch
from context import package_sha
from context import ROOT,SOURCE,configure,factory,common_state,startup,cpu,save,sha,state_identity,rng,rng_id,set_rng,restore,switches
def main():
    configure();out=ROOT/'results'/('qualification-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir()
    model=factory('v3','candidate','checkpoint_ema','cuda');initial,receipt=common_state('train');ef,efr=common_state('ef')
    select=switches(model);save(out/'STARTUP.json',startup(model,initial,receipt,'train'))
    phase='';frames=[];original=model.model._init_edge_rot_mat;files={};checks={};counts=collections.defaultdict(lambda:collections.Counter())
    from ictc_o2_hp import uvw_triton as operator
    matmul_forward=operator._Matmul.forward;target=operator._target_input_grads
    weight_shapes={(256,1024),(192,768),(512,128),(384,96)}
    def observed_forward(ctx,left,right):
        m,k=left.shape;n=right.shape[1]
        if k in (10504,63024) and (m,n) in weight_shapes:counts[phase]['weight_K'+str(k)]+=1
        return matmul_forward(ctx,left,right)
    def observed_target(ctx,left,right):
        result=target(ctx,left,right)
        if ctx.needs_input_grad[1] and not result[1] and left.shape[0] in (10504,63024) and tuple(reversed(right.shape)) in weight_shapes:
            counts[phase]['pruned_K_outputs']+=1
        return result
    operator._Matmul.forward=staticmethod(observed_forward);operator._target_input_grads=observed_target
    def rotation(edge):
        before=rng_id(rng());value=original(edge);v=cpu(value)
        frames.append({'phase':phase,'rng':before,'frame_sha256':hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()})
        return value
    def store(name,value):
        path=out/name;torch.save(value,path);files[name]=sha(path)
    model.model._init_edge_rot_mat=rotation
    for backend in ('native','native-repeat','candidate','candidate-repeat'):
        select(backend)
        restore(model,ef);phase=backend+'-ef';before=rng();store(phase+'.pt',cpu(model.infer()));store(phase+'-RNG.pt',before)
        restore(model,initial);phase=backend+'-vjp';before=rng();store(phase+'.pt',model.vjp());store(phase+'-RNG.pt',before)
        for mode in ('legacy','isolated','train-only'):
            # Legacy begins with the original restore after VJP. No state reset within a trajectory.
            restore(model,initial)
            for step in range(3):
                key=f'{mode}-{backend}-step{step}';before_state=state_identity(model.state());phase_rng={};arrays={}
                if mode!='train-only':
                    phase_rng['mixed']=rng();phase=key+'-mixed';arrays['mixed']=model.force_mixed()
                    assert state_identity(model.state())==before_state
                    if mode=='isolated':set_rng(phase_rng['mixed'])
                phase_rng['train']=rng();phase=key+'-train';arrays['training']=model.train_step(capture=True)
                arrays['training']['scheduler']=cpu(model.trainer.scheduler.scheduler.state_dict())
                after_state=state_identity(model.state());phase_rng['after']=rng()
                if mode!='train-only':
                    phase=key+'-after';arrays['after_update_output']=cpu(model.forward(model.batch('train'),False))
                    assert state_identity(model.state())==after_state
                    if mode=='isolated':set_rng(phase_rng['after'])
                store(key+'.pt',arrays);store(key+'-RNG.pt',phase_rng)
                checks[key]={'before':before_state,'after':after_state,'rng':{k:rng_id(v) for k,v in phase_rng.items()},
                    'rng_at_next_step':rng_id(rng()),'parameter_ids_unchanged':model.original_ids=={n:id(p) for n,p in model.model.named_parameters()}}
                assert checks[key]['parameter_ids_unchanged']
                del arrays;gc.collect();torch.cuda.empty_cache();print(key,flush=True)
    save(out/'FRAMES.json',frames);save(out/'CONDITIONS.json',checks)
    save(out/'ADJOINT-CALLS.json',{k:dict(v) for k,v in counts.items()})
    save(out/'MANIFEST.json',{'files':files,'package_sha256':package_sha(),'operator_source_sha256':sha(SOURCE/'ictc_o2_hp/uvw_triton.py'),
        'adjoint_calls_sha256':sha(out/'ADJOINT-CALLS.json'),
        'backends':['native','native-repeat','candidate','candidate-repeat'],'modes':['legacy','isolated','train-only'],
        'steps':3,'shared_cotangents':False,'no_model_optimizer_restore_inside_trajectory':True,
        'diagnostics':'legacy exact original order; isolated forks RNG only; train-only omits diagnostics',
        'precision':'original energy FP64, all other model math/derivatives/accumulation FP32'})
    (ROOT/'QUALIFICATION-RESULT.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
