"""Serial, separate-process whole-network qualification and conditional profiling."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

import torch

ROOT = Path(os.environ.get('O2_WHOLE_ROOT', Path(__file__).resolve().parent))
WS = ROOT.parents[1]
NET = WS/'baselines/icml-o2-stage1-20261124-r1/payload/network'

def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str)+'\n')

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def seed():
    import numpy as np
    random.seed(20261124);np.random.seed(20261124);torch.manual_seed(20261124)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(20261124)

def configure():
    assert os.environ.get('SLURM_JOB_ID'), 'Scheduler allocation required'
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch._inductor.config.triton.cudagraphs=False
    torch.use_deterministic_algorithms(True)
    seed()

def metric(a,r):
    if a is None or r is None:
        return {'passed':a is r,'same_activity':a is r}
    assert a.shape==r.shape and a.dtype==r.dtype, (a.shape,r.shape,a.dtype,r.dtype)
    assert torch.isfinite(a).all() and torch.isfinite(r).all(), 'Nonfinite result'
    d=(a-r).float();b=r.float(); n=b.norm(); dn=d.norm()
    rel=float(dn/n) if n else (0.0 if not dn else None)
    bad=int((d.abs()>2e-5+2e-5*b.abs()).sum())
    return {'passed':bad==0 and rel is not None and rel<=1e-5,
            'component_fail_count':bad,'max_abs':float(d.abs().max()),'relative_l2':rel,
            'statistics_dtype':'float32','value_dtype':str(a.dtype),'reference_norm':float(n)}

def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [cpu(v) for v in value]
    return value

class V3:
    def __init__(self,variant,role,device):
        sys.path[:0]=[str(NET/p) for p in ('fixture-api','upstream/src','upstream','env-vendor','deps')]
        canonical_path=WS/'architectures/equiformerv3/source/canonical-cpu-control-c14109a12653/canonical.py'
        canonical=load('whole_canonical',canonical_path)
        from bounded import product
        canonical.product=product
        native=load('whole_native',NET/'fixture-api/whole_network.py')
        native.strict(torch)
        net=native.Network('native','cpu')
        converted_root=Path((WS/'runs/o2-v3-canonical-cpu-20261124-r1/CONVERT-RESULT.txt').read_text().splitlines()[0])
        row=json.loads((converted_root/'RESULT.json').read_text())['roles'][role]
        assert sha(row['checkpoint'])==row['sha256']
        payload=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
        assert payload['adapter_sha256']==sha(canonical_path)
        canonical.install(net.model,payload['basis'],payload['kernels'])
        net.model.load_state_dict(payload['state_dict'],strict=True)
        for name,p in net.model.named_parameters():
            assert hashlib.sha256(p.detach().contiguous().numpy().tobytes()).hexdigest()==row['new_parameter_hashes'][name],name
        self.model=net.model.to(device)
        self.model.gradient_checkpointing_block_list=[1]*self.model.num_layers
        self.trainer=net.trainer; self.trainer.model=self.model
        self.trainer.cpu=(device=='cpu');self.trainer.device=torch.device(device)
        # Inference-only checkpoint loader skips these native training helpers.
        schedule=self.trainer.config['optim'].get('scheduler_params',{})
        if schedule.get('lambda_type')=='cosine':
            from fairchem.core.modules.scheduler import CosineLRLambda
            self.trainer.config['optim']['lr_lambda']=CosineLRLambda(**{k:schedule[k] for k in
                   ('warmup_epochs','warmup_factor','epochs','lr_min_factor')})
        self.trainer.load_loss();self.trainer.load_optimizer();self.trainer.load_extras()
        for obj in list(self.trainer.normalizers.values())+list(self.trainer.elementrefs.values()):
            obj.to(device)
        self.net=net;self.device=device
        self.optimizer=self.trainer.optimizer;self.ema=self.trainer.ema
        self.identity={'architecture':'equiformerv3-ictc-o2-canonical-min-v1','role':role,
                       'checkpoint':row['checkpoint'],'checkpoint_sha256':row['sha256'],
                       'training_recipe':self.trainer.config['optim'],
                       'loss':self.trainer.config['loss_functions'],
                       'energy_label':'uncorrected_total_energy','stress_label':'raw * -0.1 * ase.units.GPa',
                       'batch_size':1,'ef_atoms':600,'train_atoms':100,'strict_reload':True}
        self.identity['spherical_execution']='bounded exact bilinear canonical kernel; original dense 600-atom reference OOM retained separately'
        self.identity['memory_policy']='native nonreentrant block recomputation for all seven blocks; same both backends'
        self.patches=None
        if variant=='candidate':
            from adapter.install import install_v3_canonical
            count=sum(type(m).__name__=='CanonicalMLinear' for m in self.model.modules())
            self.patches=install_v3_canonical(self.model,expected_sites=count)
            self.identity['replacement_sites']=self.patches.sites
        self.original_ids={n:id(p) for n,p in self.model.named_parameters()}
        self.frames=torch.load(ROOT/'prepared/FRAMES.pt',weights_only=True)

    def batch(self,mode):
        from ase import Atoms,units
        frame=self.frames[mode]
        atom=Atoms(numbers=frame['z'].numpy(),positions=frame['pos'].numpy(),cell=frame['cell'].numpy(),pbc=frame['pbc'])
        data=self.net.calc.a2g.convert(atom);data.sid='mp-1245152-0-2-'+mode
        if mode=='train':
            data.energy=frame['uncorrected_total_energy'].reshape(1).float()
            data.forces=frame['forces'].float()
            data.stress=(frame['stress']*(-0.1*units.GPa)).reshape(1,3,3).float()
        return self.net.collate([data],otf_graph=True).to(self.device)

    def forward(self,batch,training):
        self.model.train(training)
        output=self.trainer._forward(batch)
        assert output['energy'].dtype==torch.float64
        assert output['forces'].dtype==torch.float32
        if 'stress' in output:assert output['stress'].dtype==torch.float32
        return output

    def infer(self):
        batch=self.batch('ef')
        with torch.no_grad():
            out=self.forward(batch,False)
            return {k:self.trainer._denorm_preds(k,v,batch) for k,v in out.items()}

    def loss(self,out,batch):
        # Native loss formulas/weights and normalizers; all non-energy statistics FP32.
        return self.trainer._compute_loss({k:v.float() for k,v in out.items()},batch)

    def train_step(self,capture=False):
        batch=self.batch('train');out=self.forward(batch,True)
        loss=self.loss(out,batch)
        raw={};hooks=[]
        if capture:
            for name,p in self.model.named_parameters():
                if p.requires_grad:
                    hooks.append(p.register_hook(lambda g,n=name: raw.__setitem__(n,cpu(g))))
        self.trainer._backward(loss)
        for hook in hooks:hook.remove()
        self.trainer.scheduler.step()
        assert self.original_ids=={n:id(p) for n,p in self.model.named_parameters()}
        if capture:
            return {'output':cpu(out),'loss':cpu(loss),'gradients':{n:raw.get(n) for n,p in self.model.named_parameters() if p.requires_grad},
                    'updated':cpu(dict(self.model.named_parameters())),
                    'ema':cpu(self.ema.state_dict()) if self.ema else None,
                    'optimizer':cpu(self.optimizer.state_dict())}
        return loss.detach()

    def vjp(self):
        batch=self.batch('train');batch.pos.requires_grad_(True)
        out=self.forward(batch,True)
        seed()
        cot={k:torch.randn_like(v)/max(1,v.numel())**.5 for k,v in out.items()}
        value=sum((out[k]*cot[k]).sum() for k in out)
        named=[(n,p) for n,p in self.model.named_parameters() if p.requires_grad]
        gradients=torch.autograd.grad(value,(batch.pos,)+tuple(p for n,p in named),allow_unused=True)
        return {'output':cpu(out),'cotangent':cpu(cot),'vjp':cpu(dict(zip(['coordinates']+[n for n,p in named],gradients)))}

    def force_mixed(self):
        batch=self.batch('train');batch.pos.requires_grad_(True)
        out=self.forward(batch,True)
        original=self.trainer.loss_functions
        try:
            self.trainer.loss_functions=[row for row in original if row[0]=='forces']
            assert len(self.trainer.loss_functions)==1
            loss=self.loss(out,batch)
        finally:self.trainer.loss_functions=original
        named=[(n,p) for n,p in self.model.named_parameters() if p.requires_grad]
        gradients=torch.autograd.grad(loss,(batch.pos,)+tuple(p for n,p in named),allow_unused=True)
        return {'loss':cpu(loss),'force':cpu(out['forces']),
                'mixed':cpu(dict(zip(['coordinates']+[n for n,p in named],gradients)))}

    def state(self):
        return cpu({'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
                    'ema':self.ema.state_dict() if self.ema else None,'scheduler':self.trainer.scheduler.scheduler.state_dict()})

    def restore(self,state):
        self.model.load_state_dict(state['model'],strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        if self.ema:self.ema.load_state_dict(state['ema'])
        self.trainer.scheduler.scheduler.load_state_dict(state['scheduler'])
        self.optimizer.zero_grad(set_to_none=True)
        seed()

def factory(route,variant,role,device):
    if route=='v3':return V3(variant,role,device)
    from tece import TECE
    return TECE(route,variant,role,device)

def compare_tree(a,r,prefix=''):
    metrics={}
    if isinstance(r,dict):
        assert set(a)==set(r),(prefix,set(a)^set(r))
        for k in r:metrics.update(compare_tree(a[k],r[k],prefix+'/'+str(k)))
    elif isinstance(r,(list,tuple)):
        assert len(a)==len(r)
        for i,(aa,rr) in enumerate(zip(a,r)):metrics.update(compare_tree(aa,rr,prefix+'/'+str(i)))
    elif isinstance(r,torch.Tensor) or r is None:
        if isinstance(r,torch.Tensor) and not r.is_floating_point():
            metrics[prefix]={'passed':torch.equal(a,r)}
        else:metrics[prefix]=metric(a,r)
    else:metrics[prefix]={'passed':a==r}
    return metrics

def worker(args):
    configure();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    device='cpu' if args.action in ('prepare','labels') else 'cuda'
    if device=='cuda':
        assert torch.cuda.device_count()==1
        assert 'A100' in torch.cuda.get_device_name(0)
    model=factory(args.route,args.variant,args.role,device)
    save(out/'IDENTITY.json',model.identity)
    if args.action=='labels':
        batch=model.batch('train')
        if isinstance(batch,dict):
            fields={k:v for k,v in batch.items() if k in ('energy','forces','stress','energy_weight','forces_weight','stress_weight','entropy')}
        else:fields={k:getattr(batch,k) for k in ('energy','forces','stress','fixed','natoms')}
        torch.save(cpu(fields),out/'LABELS.pt')
        save(out/'RESULT.json',{'status':'NATIVE_LABEL_ARRAYS_AUDITED','fields':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in fields.items()}})
        return
    if args.action=='prepare':
        torch.save(model.state(),out/'INITIAL.pt');return
    initial=model.state()
    if args.action=='qualify':
        if hasattr(model,'validate_memory_policy') and args.mode=='ef_only':model.validate_memory_policy(out)
        if hasattr(model,'validate_local'):model.validate_local(out)
        seed()
        if args.mode=='force_only':
            force=model.force_mixed()
            torch.save({'force_loss':force},out/'ARRAYS.pt')
            save(out/'RESULT.json',{'status':'ISOLATED_NATIVE_FORCE_LOSS_MIXED_ARRAYS_CAPTURED'})
            return
        if args.mode=='ef_only':
            seed();ef=cpu(model.infer())
            torch.save({'ef':ef},out/'ARRAYS.pt')
            save(out/'RESULT.json',{'status':'WHOLE_EF_ARRAYS_CAPTURED','job':os.environ['SLURM_JOB_ID']})
            return
        if args.mode=='train_only':
            force=model.force_mixed();torch.save(force,out/'FORCE-LOSS.pt');model.restore(initial)
            vjp=model.vjp();torch.save(vjp,out/'JOINT-VJP.pt');model.restore(initial)
            step=model.train_step(capture=True);torch.save(step,out/'TRAINING.pt')
            next_out=cpu(model.forward(model.batch('train'),False))
            torch.save({'force_loss':force,'joint_vjp':vjp,'training':step,'after_update_output':next_out},out/'ARRAYS.pt')
            save(out/'RESULT.json',{'status':'WHOLE_TRAIN_ARRAYS_CAPTURED','job':os.environ['SLURM_JOB_ID']})
            return
        seed();ef=cpu(model.infer());torch.save(ef,out/'EF.pt');model.restore(initial)
        force=model.force_mixed();torch.save(force,out/'FORCE-LOSS.pt');model.restore(initial)
        vjp=model.vjp();torch.save(vjp,out/'JOINT-VJP.pt');model.restore(initial)
        step=model.train_step(capture=True)
        torch.save(step,out/'TRAINING.pt')
        model.restore(initial)
        seed();post=cpu(model.infer())
        record={'ef':ef,'force_loss':force,'joint_vjp':vjp,'training':step,'restored_ef':post}
        torch.save(record,out/'ARRAYS.pt')
        save(out/'RESULT.json',{'status':'WHOLE_ARRAYS_CAPTURED','job':os.environ['SLURM_JOB_ID'],
                               'parameters':len(list(model.model.parameters())),'max_allocated':torch.cuda.max_memory_allocated()})
        return
    mode=args.mode
    call=model.infer if mode=='ef' else model.train_step
    for _ in range(3):call()
    model.restore(initial);torch.cuda.synchronize()
    if args.action=='profile':
        times=[]
        for i in range(10):
            torch.cuda.synchronize();start=time.perf_counter();call();torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
        model.restore(initial);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        call();torch.cuda.synchronize()
        memory={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
        model.restore(initial)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,profile_memory=True,with_stack=False) as prof:
            with torch.profiler.record_function('whole_'+mode):call()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(out/'trace.json'))
        (out/'HOTSPOTS.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=50))
        save(out/'RESULT.json',{'status':'PROFILE_COMPLETED','job':os.environ['SLURM_JOB_ID'],
                'mode':mode,'times_s':times,'median_s':statistics.median(times),'memory_bytes':memory,
                'timing_boundary':'native graph construction/transfer + E/F or complete native loss/backward/optimizer/EMA/scheduler',
                'warmup':3,'measured':10,'trace_steps':1,'cuda_graph':False})

def main():
    p=argparse.ArgumentParser();p.add_argument('route');p.add_argument('action');p.add_argument('variant');p.add_argument('output')
    p.add_argument('--role',default='checkpoint_ema');p.add_argument('--mode',default='ef');args=p.parse_args()
    try:worker(args)
    except BaseException as e:
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        save(out/'ERROR.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
        traceback.print_exc();raise

if __name__=='__main__':main()
