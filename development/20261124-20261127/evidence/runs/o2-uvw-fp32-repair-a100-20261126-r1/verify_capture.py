"""Paired E/F, full VJP, mixed derivatives and three native optimizer updates."""
import gc,json,os
import torch
import triton
from common import ROOT,SOURCE,configure,factory,common_state,restore,startup,cpu,save,sha,state_identity
def main():
    configure();out=ROOT/'results'/('verify-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    model=factory('v3','candidate','checkpoint_ema','cuda')
    originals=[(m,f) for m,n,f in model.patches.saved if n=='forward'];candidate=[m.forward for m,_ in originals]
    assert len(originals)==28
    train_state,train_receipt=common_state('train');ef_state,ef_receipt=common_state('ef')
    environment=startup(model,train_state,train_receipt,'train')
    environment.update(triton=triton.__version__,cuda_runtime=torch.version.cuda)
    save(out/'STARTUP.json',environment)
    manifests={}
    for case in ('native','native-repeat','candidate','candidate-repeat'):
        for (m,f),new in zip(originals,candidate):object.__setattr__(m,'forward',f if case.startswith('native') else new)
        folder=out/case;folder.mkdir();files=[]
        save(folder/'EF-CONDITION.json',startup(model,ef_state,ef_receipt,'ef'))
        ef=cpu(model.infer());torch.save(ef,folder/'EF.pt');files.append('EF.pt');del ef
        save(folder/'TRAIN-CONDITION.json',startup(model,train_state,train_receipt,'train'))
        vjp=model.vjp();torch.save(vjp,folder/'JOINT-VJP.pt');files.append('JOINT-VJP.pt');del vjp
        restore(model,train_state,train_receipt)
        for step in range(3):
            before_identity=state_identity(model.state())
            mixed=model.force_mixed();name=f'MIXED-{step}.pt';torch.save(mixed,folder/name);files.append(name);del mixed
            training=model.train_step(capture=True)
            training['scheduler']=cpu(model.trainer.scheduler.scheduler.state_dict())
            after=cpu(model.forward(model.batch('train'),False))
            name=f'TRAINING-{step}.pt';torch.save({'training':training,'after_update_output':after},folder/name);files.append(name)
            save(folder/f'STEP-{step}.json',{'before_state_identity':before_identity,'after_state_identity':state_identity(model.state()),
                'parameter_ids_unchanged':model.original_ids=={n:id(p) for n,p in model.model.named_parameters()}})
            assert model.original_ids=={n:id(p) for n,p in model.model.named_parameters()}
            del training,after
            gc.collect();torch.cuda.empty_cache();print(case,step,flush=True)
        manifests[case]={name:sha(folder/name) for name in files}
        assert state_identity(train_state)==json.loads((out/'STARTUP.json').read_text())['state_identity']
    save(out/'MANIFEST.json',{'status':'WHOLE_VERIFICATION_ARRAYS_CAPTURED','cases':manifests,
        'operator_source_sha256':sha(SOURCE/'ictc_o2_hp/uvw_triton.py'),
        'scope':'600-atom E/F; fixed 100-atom full VJP and three consecutive native full-loss training/mixed/update steps',
        'loss':'unchanged native per_atom_mae20 + l2mae20 + mae5; actual own cotangents'})
    (ROOT/'VERIFY-RESULT.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
