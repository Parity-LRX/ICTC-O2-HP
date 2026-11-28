"""Controlled FP32 backend swaps, exact loss cotangents and real force-loss VJPs."""
import copy,gc,hashlib,json,os,sys,time,types
from pathlib import Path
import torch

ROOT=Path(os.environ.get('O2_CAUSE_ROOT',Path(__file__).resolve().parent));WS=ROOT.parents[1]
OLD=WS/'runs/o2-uvw-triton-a100-20261125-r1';CONTROL=OLD/'snapshots/preflight-119174'
os.environ.update(O2_WHOLE_ROOT=str(OLD),O2_UVW_TILED='1',O2_UVW_SOURCE=str(CONTROL/'fork-src'))
sys.path[:0]=[str(CONTROL),str(CONTROL/'fork-src')]
from whole import configure,factory,cpu,save,sha,seed
from diagnose import flags,parameters,graph,state_identity
from ictc_o2_hp import uvw_triton as operator

native_mm=lambda a,b:torch.matmul(a,b)
triton_mm=operator._Matmul.apply

def hybrid(name,forward,backward):
    class Hybrid(torch.autograd.Function):
        @staticmethod
        def forward(ctx,a,b):
            if forward=='triton':return operator._Matmul.forward(ctx,a,b)
            ctx.save_for_backward(a,b)
            return native_mm(a,b)
        @staticmethod
        def backward(ctx,g):
            a,b=ctx.saved_tensors
            return (backward(g,b.T) if ctx.needs_input_grad[0] else None,
                    backward(a.T,g) if ctx.needs_input_grad[1] else None)
    Hybrid.__name__=name
    return Hybrid.apply

TN=hybrid('TritonForwardNativeAdjoints','triton',native_mm)
NT=hybrid('NativeForwardTritonAdjoints','native',triton_mm)
TTN=hybrid('TritonForwardFirstNativeSecond','triton',TN)
NNT=hybrid('NativeForwardFirstTritonSecond','native',NT)
CASES=('native','native-layout','native-flat','tiled','T-N','N-T','T-T-N','N-N-T')
OPS={'native-flat':native_mm,'tiled':triton_mm,'T-N':TN,'N-T':NT,'T-T-N':TTN,'N-N-T':NNT}

def main():
    configure();job=os.environ['SLURM_JOB_ID'];out=ROOT/'results'/('a100-'+job);out.mkdir(parents=True)
    preflight=json.loads((ROOT/'PREFLIGHT.json').read_text());control=Path(preflight['snapshot'])
    for name,digest in preflight['files'].items():assert sha(control/name)==digest,name
    model=factory('v3','candidate','checkpoint_ema','cuda')
    common=Path(json.loads((OLD/'PREFLIGHT.json').read_text())['common_states']['train'])
    condition=json.loads((common/'CONDITION.json').read_text());assert sha(common/'START.pt')==condition['start_sha256']
    state=torch.load(common/'START.pt',map_location='cpu',weights_only=False)
    expected=state_identity(state)
    def restore():
        model.restore(dict(state,optimizer=copy.deepcopy(state['optimizer'])))
        assert state_identity(model.state())==expected
    restore();assert parameters(model)['sha256']==condition['parameters']['sha256']
    g,_=graph(model,'train');assert g==condition['graph']
    save(out/'STARTUP.json',{'job':job,'hardware':flags(model),'candidate_commit':'df12a3a85ec3b66c9e4909dca0c90a86a12b1516',
        'common_state_sha256':condition['start_sha256'],'complete_state_identity':expected,'graph':g,
        'loss_recipe':model.trainer.config['loss_functions'],'cases':CASES,'maximum_gpu_h':20/60})
    originals=[(obj,value) for obj,name,value in model.patches.saved if name=='forward']
    current=[(obj,obj.forward) for obj,_ in originals]
    assert len(originals)==28
    def layout_forward(module,x,concat_outputs=True):
        x=torch.einsum('iab,ebi->eai',module.C_in,x)
        y=torch.nn.functional.linear(x,module.K)
        y=torch.einsum('oab,eao->ebo',module.C_out,y)
        return y if concat_outputs else (y[:,:1],y[:,1:2])
    def select(case):
        for (module,original),(_,candidate) in zip(originals,current):
            forward=original if case=='native' else (types.MethodType(layout_forward,module) if case=='native-layout' else candidate)
            object.__setattr__(module,'forward',forward)
        operator.channel_matmul=OPS.get(case,native_mm)
    named=[(n,p) for n,p in model.model.named_parameters() if p.requires_grad]
    progress=[];native_cot=None
    for case in CASES:
        started=time.time();select(case);restore()
        layouts={};layout_hooks=[];loss_inputs={};loss_hooks=[]
        for name,module in model.model.named_modules():
            if type(module).__name__!='CanonicalMLinear':continue
            def note(m,args,result,_name=name):
                if _name in layouts:return
                value=result if isinstance(result,torch.Tensor) else result[0]
                layouts[_name]={'input_shape':list(args[0].shape),'input_stride':list(args[0].stride()),
                    'output_shape':list(value.shape),'output_stride':list(value.stride()),'output_contiguous':value.is_contiguous()}
            layout_hooks.append(module.register_forward_hook(note))
        for label,info in model.trainer.loss_functions:
            def loss_note(module,args,_label=label):
                if _label not in loss_inputs:loss_inputs[_label]={'pred':cpu(args[0]),'target':cpu(args[1]),'module':str(module)}
            loss_hooks.append(info['fn'].register_forward_pre_hook(loss_note))
        captured=model.train_step(capture=True)
        total={k:captured[k] for k in ('output','loss','gradients')}
        del captured
        for hook in layout_hooks+loss_hooks:hook.remove()
        restore()
        # First replicate the exact old helper independently of the decomposition graph.
        direct_original=model.force_mixed()
        restore()
        batch=model.batch('train');batch.pos.requires_grad_(True)
        prediction=model.forward(batch,True)
        old_losses=model.trainer.loss_functions
        try:
            model.trainer.loss_functions=[row for row in old_losses if row[0]=='forces']
            force_loss=model.loss(prediction,batch)
        finally:model.trainer.loss_functions=old_losses
        targets=(batch.pos,)+tuple(p for _,p in named);names=['coordinates']+[n for n,_ in named]
        direct=torch.autograd.grad(force_loss,targets,retain_graph=True,allow_unused=True)
        cot=torch.autograd.grad(force_loss,prediction['forces'],retain_graph=True)[0].detach()
        if native_cot is None:native_cot=cot.detach().clone()
        own=torch.autograd.grad(prediction['forces'],targets,grad_outputs=cot,retain_graph=True,allow_unused=True)
        shared=torch.autograd.grad(prediction['forces'],targets,grad_outputs=native_cot,allow_unused=True)
        result={'total':total,'force_original':cpu(direct_original),'force_direct':cpu(dict(zip(names,direct))),
            'force_self_vjp':cpu(dict(zip(names,own))),'force_shared_vjp':cpu(dict(zip(names,shared))),
            'force_output':cpu(prediction['forces']),'force_loss':cpu(force_loss),'self_cotangent':cpu(cot),
            'native_cotangent':cpu(native_cot),'loss_inputs':loss_inputs,'layouts':layouts}
        torch.save(result,out/(case+'.pt'))
        save(out/(case+'-CONDITION.json'),{'case':case,'complete_state_identity':expected,
            'parameter_sha256':condition['parameters']['sha256'],'graph':g,
            'original_parameter_ids_unchanged':model.original_ids=={n:id(p) for n,p in model.model.named_parameters()}})
        del result,total,direct_original,direct,own,shared,prediction,force_loss,batch,cot
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        row={'case':case,'elapsed_s':time.time()-started};progress.append(row);save(out/'PROGRESS.json',progress);print(json.dumps(row),flush=True)
    assert state_identity(state)==expected,'Common template mutated'
    save(out/'FINISH.json',{'job':job,'status':'CONTROLLED_DIAGNOSTICS_COMPLETED','cases':list(CASES)})
    (ROOT/'GPU-RESULT.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
