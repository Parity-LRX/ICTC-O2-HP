"""Phase-specific RNG and transparent diagnostics; no restores inside training trajectory."""
import gc,json,os
import torch
from context import ROOT,SOURCE,configure,factory,common_state,startup,cpu,save,sha,state_identity,rng,rng_id,set_rng,restore,switches
def main():
    configure();out=ROOT/'results'/('protocol-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir()
    model=factory('v3','candidate','checkpoint_ema','cuda');initial,receipt=common_state('train');ef,efr=common_state('ef')
    select=switches(model);save(out/'STARTUP.json',startup(model,initial,receipt,'train'))
    phase='';frames=[];original=model.model._init_edge_rot_mat
    def rotation(edge):
        before=rng_id(rng());value=original(edge);v=cpu(value)
        import hashlib
        frames.append({'phase':phase,'rng':before,'frame_sha256':hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()})
        return value
    model.model._init_edge_rot_mat=rotation;manifest={};checks={}
    for mode in ('legacy','isolated','train-only'):
        for backend in ('native','candidate'):
            select(backend);restore(model,ef);phase=mode+'-'+backend+'-ef';model.infer();restore(model,initial)
            if mode=='legacy':phase=mode+'-'+backend+'-vjp';model.vjp();restore(model,initial)
            for step in range(3):
                key=f'{mode}-{backend}-step{step}';before=state_identity(model.state());phase_rng={};mixed=None
                if mode!='train-only':
                    phase_rng['mixed']=rng();phase=key+'-mixed';mixed=model.force_mixed()
                    assert state_identity(model.state())==before
                    if mode=='isolated':set_rng(phase_rng['mixed'])
                phase_rng['train']=rng();phase=key+'-train';training=model.train_step(capture=True);training['scheduler']=cpu(model.trainer.scheduler.scheduler.state_dict())
                after_state=state_identity(model.state());phase_rng['after']=rng();after=None
                if mode!='train-only':
                    phase=key+'-after';after=cpu(model.forward(model.batch('train'),False));assert state_identity(model.state())==after_state
                    if mode=='isolated':set_rng(phase_rng['after'])
                arrays={'training':training}
                if mode!='train-only':arrays.update(mixed=mixed,after_update_output=after)
                path=out/(key+'.pt');torch.save(arrays,path);manifest[path.name]=sha(path)
                torch.save(phase_rng,out/(key+'-RNG.pt'))
                checks[key]={'before':before,'after':after_state,'rng':{k:rng_id(v) for k,v in phase_rng.items()},
                    'rng_at_next_step':rng_id(rng()),'parameter_ids_unchanged':model.original_ids=={n:id(p) for n,p in model.model.named_parameters()}}
                assert checks[key]['parameter_ids_unchanged']
                del arrays,training,mixed,after;gc.collect();torch.cuda.empty_cache();print(key,flush=True)
    save(out/'FRAMES.json',frames);save(out/'CONDITIONS.json',checks)
    save(out/'MANIFEST.json',{'files':manifest,'operator_source_sha256':sha(SOURCE/'ictc_o2_hp/uvw_triton.py'),
        'diagnostics':'legacy unchanged; isolated uses actual training RNG fork, restores RNG only; train-only omits all diagnostics',
        'no_model_optimizer_restore_inside_trajectory':True,'precision':'original energy FP64, all other math/derivatives FP32'})
    (ROOT/'PROTOCOL-RESULT.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
