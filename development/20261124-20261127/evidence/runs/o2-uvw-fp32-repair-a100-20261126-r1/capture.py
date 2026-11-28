"""Capture all 28 actual trained channel inputs, original K and force adjoints."""
import gc,json,os,types
import torch
from common import ROOT,SOURCE,configure,factory,common_state,restore,startup,cpu,save,sha
from ictc_o2_hp.uvw_triton import canonical_linear
def main():
    configure();out=ROOT/'results'/('capture-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    model=factory('v3','native','checkpoint_ema','cuda');state,receipt=common_state('train')
    save(out/'STARTUP.json',startup(model,state,receipt,'train'))
    sites=[(n,m) for n,m in model.model.named_modules() if type(m).__name__=='CanonicalMLinear'];assert len(sites)==28
    original=[m.forward for _,m in sites];manifest={}
    for backend in ('native','v1'):
        restore(model,state,receipt);records={};phase=['energy_forward_and_force']
        for name,module in sites:
            def forward(current,x_m,concat_outputs=True,_name=name):
                x=torch.einsum('iab,ebi->eai',current.C_in,x_m)
                y=torch.nn.functional.linear(x,current.K) if backend=='native' else canonical_linear(x,current.K)
                if _name not in records:
                    assert x.shape[0]*x.shape[1]==10504
                    records[_name]={'x':cpu(x),'K':cpu(current.K),'y':cpu(y),
                        'x_original_stride':list(x.stride()),'K_original_stride':list(current.K.stride()),
                        'parameter_id':id(current.K),'m':current.m,'adjoints':{},'hook_calls':{}}
                if y.requires_grad:
                    def adjoint(g,_site=_name):
                        label=phase[0];entry=records[_site]
                        entry['hook_calls'][label]=entry['hook_calls'].get(label,0)+1
                        if label not in entry['adjoints']:
                            entry['adjoints'][label]={'g':cpu(g),'stride':list(g.stride()),'requires_grad':g.requires_grad}
                        return g
                    y.register_hook(adjoint)
                y=torch.einsum('oab,eao->ebo',current.C_out,y)
                result=(y[:,:1],y[:,1:2])
                return (torch.cat(result,1) if backend=='native' else y) if concat_outputs else result
            object.__setattr__(module,'forward',types.MethodType(forward,module))
        batch=model.batch('train');output=model.forward(batch,True)
        phase[0]='loss_backward';loss=model.loss(output,batch);raw={};hooks=[]
        for n,p in model.model.named_parameters():
            if p.requires_grad:hooks.append(p.register_hook(lambda g,n=n:raw.__setitem__(n,cpu(g))))
        model.trainer._backward(loss)
        for hook in hooks:hook.remove()
        model.trainer.scheduler.step()
        arrays={'output':cpu(output),'loss':cpu(loss),'gradients':raw,'updated':cpu(dict(model.model.named_parameters())),
            'optimizer':cpu(model.optimizer.state_dict()),'ema':cpu(model.ema.state_dict()),
            'scheduler':cpu(model.trainer.scheduler.scheduler.state_dict())}
        folder=out/backend;folder.mkdir()
        torch.save(arrays,folder/'TRAINING.pt');rows=[]
        for index,(name,module) in enumerate(sites):
            record=records[name];assert record['parameter_id']==id(module.K)
            assert 'energy_forward_and_force' in record['adjoints'],name
            path=folder/(f'site-{index:02d}.pt');torch.save(record,path)
            rows.append({'index':index,'name':name,'path':str(path),'sha256':sha(path),
                'x_shape':list(record['x'].shape),'K_shape':list(record['K'].shape),'m':module.m,
                'adjoint_phases':list(record['adjoints']),'hook_calls':record['hook_calls']})
        manifest[backend]=rows;save(folder/'SITES.json',rows)
        assert model.original_ids=={n:id(p) for n,p in model.model.named_parameters()}
        del records,arrays,output,loss,raw,batch
        gc.collect();torch.cuda.empty_cache();print(backend,len(rows),flush=True)
    for (_,m),f in zip(sites,original):object.__setattr__(m,'forward',f)
    save(out/'CAPTURE.json',{'status':'ACTUAL_TRAINING_28_SITES_CAPTURED','rows':10504,'backends':manifest,
        'source_sha256':sha(SOURCE/'ictc_o2_hp/uvw_triton.py')})
    (ROOT/'CAPTURE-RESULT.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
