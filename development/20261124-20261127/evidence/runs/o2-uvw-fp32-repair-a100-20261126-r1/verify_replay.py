"""Replay every actual native/candidate updated state under both execution backends."""
import copy,gc,json,os
from pathlib import Path
import torch,triton
from common import ROOT,SOURCE,configure,factory,common_state,startup,cpu,save,sha,seed,state_identity,parameters,graph
def main():
    configure();out=ROOT/'results'/('replay-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    model=factory('v3','candidate','checkpoint_ema','cuda');initial,receipt=common_state('train');ef_state,ef_receipt=common_state('ef')
    old=Path((ROOT/'VERIFY-RESULT.txt').read_text().strip())
    points=[('initial',initial,state_identity(initial))]
    for branch in ('native','candidate'):
        for step in (0,1):
            p=torch.load(old/branch/f'TRAINING-{step}.pt',map_location='cpu',weights_only=False)['training']
            state={'model':dict(initial['model'],**p['updated']),'optimizer':p['optimizer'],'ema':p['ema'],'scheduler':p['scheduler']}
            expected=json.loads((old/branch/f'STEP-{step}.json').read_text())['after_state_identity']
            assert state_identity(state)==expected,('State reconstruction differs',branch,step)
            points.append((branch+f'-update-{step+1}',state,expected))
    originals=[(m,f) for m,n,f in model.patches.saved if n=='forward'];candidate=[m.forward for m,_ in originals]
    def select(backend):
        for (m,f),new in zip(originals,candidate):object.__setattr__(m,'forward',f if backend.startswith('native') else new)
    def restore(state,expected):
        model.restore(dict(state,optimizer=copy.deepcopy(state['optimizer'])))
        assert state_identity(model.state())==expected
        assert model.original_ids=={n:id(p) for n,p in model.model.named_parameters()}
    environment=startup(model,initial,receipt,'train');environment.update(triton=triton.__version__,cuda_runtime=torch.version.cuda)
    save(out/'STARTUP.json',environment);manifest={}
    for backend in ('native','native-repeat','candidate','candidate-repeat'):
        select(backend);folder=out/backend;folder.mkdir();manifest[backend]={}
        restore(ef_state,state_identity(ef_state));ef=cpu(model.infer());torch.save(ef,folder/'EF.pt');del ef
        manifest[backend]['EF.pt']=sha(folder/'EF.pt')
    for point,state,expected in points:
        for backend in ('native','native-repeat','candidate','candidate-repeat'):
            select(backend);restore(state,expected);g,_=graph(model,'train');assert g==receipt['graph']
            before={'state_identity':expected,'parameters':parameters(model),'graph':g}
            folder=out/backend;vjp=model.vjp();mixed=model.force_mixed();training=model.train_step(capture=True)
            training['scheduler']=cpu(model.trainer.scheduler.scheduler.state_dict())
            after=cpu(model.forward(model.batch('train'),False))
            arrays={'joint_vjp':vjp,'force_mixed':mixed,'training':training,'after_update_output':after}
            name=point+'.pt';torch.save(arrays,folder/name);manifest[backend][name]=sha(folder/name)
            save(folder/(point+'-CONDITION.json'),before)
            assert model.original_ids=={n:id(p) for n,p in model.model.named_parameters()}
            del vjp,mixed,training,after,arrays;gc.collect();torch.cuda.empty_cache()
            print(point,backend,flush=True)
    save(out/'MANIFEST.json',{'status':'BOTH_ACTUAL_STATE_TRAJECTORIES_REPLAYED','cases':manifest,
        'points':[name for name,_,_ in points],'input_state_identity_verified':True,
        'operator_source_sha256':sha(SOURCE/'ictc_o2_hp/uvw_triton.py'),
        'independent_trajectory_source':str(old),'contract':'Same-state complete-object gates; actual own native losses'})
    (ROOT/'REPLAY-RESULT.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
