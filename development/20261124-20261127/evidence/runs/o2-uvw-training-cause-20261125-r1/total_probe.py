"""Full original three-term loss: real backward versus common-output-cotangent control."""
import copy,gc,json,os,sys,time
from pathlib import Path
import torch
ROOT=Path(os.environ['O2_CAUSE_ROOT'])
initial=json.loads((ROOT/'PREFLIGHT.json').read_text());sys.path.insert(0,initial['snapshot'])
from probe import configure,factory,cpu,save,sha,operator,triton_mm,OLD,state_identity,parameters,graph,flags
def main():
    configure();job=os.environ['SLURM_JOB_ID'];out=ROOT/'results'/('total-a100-'+job);out.mkdir(parents=True)
    model=factory('v3','candidate','checkpoint_ema','cuda')
    original=[(m,f) for m,n,f in model.patches.saved if n=='forward'];candidate=[m.forward for m,_ in original]
    common=Path(json.loads((OLD/'PREFLIGHT.json').read_text())['common_states']['train'])
    condition=json.loads((common/'CONDITION.json').read_text());assert sha(common/'START.pt')==condition['start_sha256']
    state=torch.load(common/'START.pt',map_location='cpu',weights_only=False);expected=state_identity(state)
    def restore():
        model.restore(dict(state,optimizer=copy.deepcopy(state['optimizer'])))
        assert state_identity(model.state())==expected
    restore();assert parameters(model)['sha256']==condition['parameters']['sha256']
    current_graph,_=graph(model,'train');assert current_graph==condition['graph']
    save(out/'STARTUP.json',{'job':job,'hardware':flags(model),'complete_state_identity':expected,
        'common_state_sha256':condition['start_sha256'],'graph':current_graph,'maximum_gpu_seconds':300})
    native_cot=None
    for case in ('native','tiled'):
        for (m,f),new in zip(original,candidate):object.__setattr__(m,'forward',f if case=='native' else new)
        operator.channel_matmul=triton_mm
        for mode in ('direct','own_linear','shared_linear'):
            restore();batch=model.batch('train');prediction=model.forward(batch,True);loss=model.loss(prediction,batch)
            cot=torch.autograd.grad(loss,tuple(prediction.values()),retain_graph=True,allow_unused=True)
            cot={k:(v.detach() if v is not None else torch.zeros_like(prediction[k])) for k,v in zip(prediction,cot)}
            if native_cot is None:native_cot={k:v.clone() for k,v in cot.items()}
            selected=cot if mode=='own_linear' else native_cot
            objective=loss if mode=='direct' else sum((prediction[k].float()*selected[k].float()).sum() for k in prediction)
            raw={};hooks=[]
            for n,p in model.model.named_parameters():
                if p.requires_grad:hooks.append(p.register_hook(lambda g,n=n:raw.__setitem__(n,cpu(g))))
            model.trainer._backward(objective)
            for h in hooks:h.remove()
            gradients={n:raw.get(n) for n,p in model.model.named_parameters() if p.requires_grad}
            torch.save({'loss':cpu(loss),'output':cpu(prediction),'cotangent':cpu(cot),
                'native_cotangent':cpu(native_cot),'gradients':gradients},out/(case+'-'+mode+'.pt'))
            assert model.original_ids=={n:id(p) for n,p in model.model.named_parameters()}
            del raw,gradients,objective,loss,cot,prediction,batch
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();print(case,mode,flush=True)
    save(out/'FINISH.json',{'job':job,'status':'FULL_LOSS_COTANGENT_CONTROL_COMPLETED','state_identity':expected,
        'parameters':condition['parameters'],'graph':condition['graph'],'loss_recipe':model.identity['loss'],
        'note':'Linearized loss is causal diagnostic only, not a training candidate or qualification.'})
    (ROOT/'TOTAL-RESULT.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
