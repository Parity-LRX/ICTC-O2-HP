"""Real native full-loss training and mixed derivatives for every FP32 candidate."""
import gc,json,os
import torch
from common import ROOT,configure,factory,common_state,restore,startup,cpu,save,sha,state_identity
from candidate_kernels import CONFIGS,channel_matmul
from ictc_o2_hp import uvw_triton as operator
def main():
    configure();out=ROOT/'results'/('train-screen-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    assert (ROOT/'LOCAL-SCREEN.txt').exists()
    model=factory('v3','candidate','checkpoint_ema','cuda');state,receipt=common_state('train')
    save(out/'STARTUP.json',startup(model,state,receipt,'train'))
    originals=[(m,f) for m,n,f in model.patches.saved if n=='forward'];current=[m.forward for m,_ in originals]
    assert len(originals)==28;v1_mm=operator.channel_matmul
    cases={'native':None,'v1':None,**{n:(c,c) for n,c in CONFIGS.items()}}
    extra=__import__('pathlib').Path(__file__).with_name('EXTRA-CANDIDATES.json')
    if extra.exists():
        cases={**{'native':None,'v1':None},**{name:tuple(tuple(c) for c in pair) for name,pair in json.loads(extra.read_text()).items()}}
    source_sha=sha(__file__);kernel_sha=sha(__import__('candidate_kernels').__file__)
    for case,pair in cases.items():
        folder=out/case;folder.mkdir()
        for (m,f),new in zip(originals,current):object.__setattr__(m,'forward',f if case=='native' else new)
        operator.channel_matmul=v1_mm if pair is None else channel_matmul(*pair)
        restore(model,state,receipt)
        step=model.train_step(capture=True)
        after=cpu(model.forward(model.batch('train'),False))
        torch.save({'training':step,'after_update_output':after},folder/'TRAINING.pt');del step,after
        restore(model,state,receipt)
        mixed=model.force_mixed();torch.save(mixed,folder/'FORCE.pt');del mixed
        save(folder/'CASE.json',{'case':case,'forward_backward_configs':pair,'source_sha256':source_sha,
            'kernel_source_sha256':kernel_sha,'state_identity':state_identity(state),
            'parameter_ids_unchanged':model.original_ids=={n:id(p) for n,p in model.model.named_parameters()}})
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();print(case,flush=True)
    (ROOT/'TRAIN-SCREEN.txt').write_text(str(out)+'\n')
    save(out/'FINISH.json',{'status':'REAL_LOSS_CANDIDATE_ARRAYS_CAPTURED','cases':list(cases)})
if __name__=='__main__':main()
