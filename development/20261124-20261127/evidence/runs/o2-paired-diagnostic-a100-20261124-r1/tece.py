"""Pinned TECE factories and native data/loss/optimizer protocols."""
import ast
import copy
import importlib
import inspect
import json
import logging
from pathlib import Path
import sys
import types
from functools import wraps
from typing import Dict, List, Any, Callable

import torch
from whole import ROOT,WS,cpu,seed,sha,load

def extracted(path,names,namespace):
    tree=ast.parse(path.read_text())
    body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    assert len(body)==len(names)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

class TECE:
    def __init__(self,route,variant,role,device):
        if (ROOT/'vendor').exists():sys.path.insert(0,str(ROOT/'vendor'))
        full=route=='tece-full';self.full=full;self.device=device
        revision='5b428ee36ffd' if full else '8593380e6cbe'
        source=WS/('architectures/tace/source/upstream-'+revision)
        self.source=source
        sys.path.insert(0,str(source))
        from tace.models import TensorModel
        self.frames=torch.load(ROOT/'prepared/FRAMES.pt',weights_only=True)
        seed()
        if full:
            from tace.models import TECE as NativeTECE
            from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
            from tace.dataset.neighbour_list import get_neighborhood
            frame=self.frames['train'];numbers=sorted(set(frame['z'].tolist()))
            edges,shifts,pbc,cell=get_neighborhood(positions=frame['pos'].numpy(),cutoff=5.,
                    pbc=(True,True,True),lattice=frame['cell'].numpy(),max_neighbors=None,backend='ase')
            stats={'atomic_numbers':numbers,'avg_num_neighbors':len(edges[0])/len(frame['z']),
                   'atomic_energy':{z:0. for z in numbers},
                   'rms_forces':{z:float(frame['forces'][frame['z']==z].square().mean().sqrt()) for z in numbers}}
            config=copy.deepcopy(DEFAULT_MODEL_CONFIG)
            config.update(cutoff=5.,max_neighbors=None,Lmax=3,lmax=3,mmax=3,num_channel=64,num_layers=2,
                          parity=True,statistics=[stats],target_property=['energy','forces'])
            config['node_embedding']['type']='tensor'
            config['radial_basis'].update(hidden=[64,64,64,1024],apply_cutoff=False)
            config['atomic_basis'].update(type='ece_o2',correlation=3,algorithm='recursive',use_asymmetric_contraction=False)
            config['product_basis'].update(type=None,correlation=3)
            self.model=TensorModel(NativeTECE(**config))
            self.identity={'architecture':'TECE-O2-ECE-parity-full','source':str(source),'seed':20261124,
                           'checkpoint':'fixed_random_initialization_structural_only',
                           'config':config,'statistics':stats,'loss_properties':['energy','forces'],
                           'loss_weights':[1.,10.],'optimizer':'Adam lr=.01 amsgrad=True weight_decay=0',
                           'ema_decay':.99,'scope':'Implementation/performance only; no prediction accuracy claim.'}
            self.targets=['energy','forces'];self.loss_weights=[1.,10.]
        else:
            path=ROOT/'downloads/TECE-OAM-RRA-1.0.pt'
            assert (ROOT/'downloads/TECE-COMPLETE.txt').exists(),'Official pretrained checkpoint not completely downloaded'
            assert path.stat().st_size==890523407
            assert sha(path)=='9f36562582d931347c3904f763e820edcaf5f27c3f13beb6776e49fcc7de38bb'
            package=torch.load(path,map_location='cpu',weights_only=False)
            if isinstance(package,torch.nn.Module):
                self.model=package
                config=self.model.readout_fn.model_config
            else:
                # Use exact upstream factory functions, without importing optional Trainer dependencies.
                from tace.utils.env import acceleration_enabled
                ns={'torch':torch,'Dict':Dict,'List':List,'Any':Any,'importlib':importlib,
                    'deep_convert':lambda x:x,'acceleration_enabled':acceleration_enabled}
                extracted(source/'tace/lightning/torch_model.py',{'_prune_removed_keys','select_wrapper','create_model'},ns)
                state=package['state_dict'];config=copy.deepcopy(package['cfg'])
                assert isinstance(config,dict)
                state_ns={'torch':torch,'Dict':Dict,'copy':copy}
                extracted(source/'tace/lightning/lit_model.py',{'saved_radial_config','_load_model_state_dict'},state_ns)
                config=state_ns['saved_radial_config'](config,state)
                self.model=ns['create_model'](config,package['statistics'],package['target_property'],package.get('embedding_property',[]),prune_removed_keys=True)
                state_ns['_load_model_state_dict'](self.model,state,True,False)
            self.targets=['energy','forces','stress'];self.loss_weights=[1.,5.,.5]
            self.model.reset_target_property(self.targets)
            self.identity={'architecture':'TECE-OAM-RRA-1.0','source':str(source),'checkpoint':str(path),
                  'checkpoint_sha256':sha(path),'package_type':str(type(package)),
                  'config':config,'loss_properties':self.targets,'loss_weights':self.loss_weights,
                  'optimizer':'AdamW lr=.001 weight_decay=1e-8; current official E/F/S recipe',
                  'scope':'Pretrained-weight implementation/performance; no independent task accuracy claim.'}
            import tace.models.adapter as adapter_module
            if hasattr(adapter_module,'DATA_DTYPE'):adapter_module.DATA_DTYPE=torch.float32
        # Explicit common precision: learned/geometry FP32; energy outputs and sums FP64.
        self.model.to(device)
        energy_buffers={}
        for name,module in self.model.named_modules():
            if type(module).__name__ in ('AtomicEnergy','OneHotToAtomicEnergy','ScaleShift'):
                for key,buffer in module._buffers.items():
                    if key in ('atomic_energy','scale','shift') and isinstance(buffer,torch.Tensor):
                        energy_buffers[(name,key)]=buffer.detach().clone()
        for p in self.model.parameters():
            if p.is_floating_point():p.data=p.data.to(device=device,dtype=torch.float32)
        for module in self.model.modules():
            for name,buffer in list(module._buffers.items()):
                if isinstance(buffer,torch.Tensor) and buffer.is_floating_point():
                    module._buffers[name]=buffer.to(device=device,dtype=torch.float32)
        for (name,key),buffer in energy_buffers.items():
            self.model.get_submodule(name)._buffers[key]=buffer.to(device=device,dtype=torch.float64)
        readout=self.model.readout_fn
        if full:
            readout.atomic_energy_layer.register_forward_pre_hook(lambda m,inputs:(inputs[0].to(torch.float64),)+inputs[1:])
        for module in readout.energy_readouts:
            module.register_forward_hook(lambda m,i,o:o.to(torch.float64))
        readout.atomic_energy_layer.register_forward_hook(lambda m,i,o:o.to(torch.float64))
        self.model.eval()
        self.memory_saved=[]
        if not full:
            from torch.utils.checkpoint import checkpoint
            representation=self.model.readout_fn.representation
            modules=[representation.node_embedding]
            for group in ('edge_updates','interactions','products'):
                modules.extend(getattr(representation,group))
            for module in modules:
                original=module.forward
                def recompute(current,*args,_original=original,**kwargs):
                    return checkpoint(_original,*args,use_reentrant=False,**kwargs)
                wrapped=types.MethodType(recompute,module)
                object.__setattr__(module,'forward',wrapped)
                self.memory_saved.append((module,original,wrapped))
            self.identity['memory_policy']='nonreentrant per-embedding/edge-update/interaction/product recomputation, same both backends; original 600-atom native OOM retained'
        self.original_ids={n:id(p) for n,p in self.model.named_parameters()}
        self.patches=None
        if variant=='candidate':
            from adapter.install import install_tece_full,install_tece_canonical
            if full:
                sites=sum(type(m).__name__=='EceO2Interaction' for m in self.model.modules())
                self.patches=install_tece_full(self.model,expected_sites=sites)
            else:
                sites=sum(type(m).__name__=='UvSO2Interaction' and hasattr(getattr(m,'rejector',None),'ece') for m in self.model.modules())
                self.patches=install_tece_canonical(self.model,expected_sites=sites)
            self.identity['replacement_sites']=self.patches.sites
        # Execute the exact upstream optimizer construction on a module containing
        # the same model prefix as Lightning, retaining native parameter groups.
        proxy=torch.nn.Module();proxy.add_module('model',self.model)
        proxy.cfg={'optimizer':({'_target_':'torch.optim.Adam','lr':.01,'amsgrad':True,'weight_decay':0.}
                   if full else {'_target_':'torch.optim.AdamW','lr':.001,'weight_decay':1e-8})}
        tree=ast.parse((source/'tace/lightning/lit_model.py').read_text())
        function=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='configure_optimizers')
        namespace={'torch':torch,'copy':copy,'importlib':importlib,'inspect':inspect,'logging':logging}
        self.clip_function=None
        if not full:
            policy=load('native_optimizer_policy',source/'tace/lightning/optimizer.py')
            namespace['get_optimizer_param_groups']=policy.get_optimizer_param_groups
            self.clip_function=policy.clip_gradient_norm
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(source/'tace/lightning/lit_model.py'),'exec'),namespace)
        self.optimizer=namespace['configure_optimizers'](proxy)['optimizer']
        self.identity['optimizer_parameter_groups']=[{'count':len(group['params']),'weight_decay':group['weight_decay']} for group in self.optimizer.param_groups]
        self.ema_decay=.99 if full else .999
        from tace.utils.ema import ExponentialMovingAverage
        self.ema=ExponentialMovingAverage(list(self.model.parameters()),self.ema_decay,True)
        self.ema.to(device)
        self.identity['ema_implementation']='native tace.utils.ema.ExponentialMovingAverage, use_num_updates=True'
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=.8 if full else .5,patience=50 if full else 25)
        self.identity.update(batch_size=1,ef_atoms=600,train_atoms=100,
                  energy_label='uncorrected_total_energy',scheduler_interval='validation epoch; no per-batch step',
                  gradient_clip_val=10.,gradient_clip_algorithm='norm',gradient_clip_norm_type=2.)

    def batch(self,mode):
        from ase import Atoms,units
        from torch_geometric.data import Batch
        from tace.dataset.graph import from_atoms
        from tace.dataset.quantity import KeySpecification
        frame=self.frames[mode]
        atom=Atoms(numbers=frame['z'].numpy(),positions=frame['pos'].numpy(),cell=frame['cell'].numpy(),pbc=frame['pbc'])
        if mode=='train':
            atom.info['energy']=float(frame['uncorrected_total_energy'])
            atom.arrays['forces']=frame['forces'].numpy()
            atom.info['stress']=(frame['stress']*(-.1*units.GPa)).numpy()
        keyspec=KeySpecification.from_defaults()
        data=from_atoms(element=self.model.get_torch_element(),atoms=atom,cutoff=self.model.get_cutoff(),
            max_neighbors=self.model.get_max_neighbors(),keyspec=keyspec,target_property=self.targets,
            embedding_property=[],training=True,neighborlist_backend='ase')
        batch=Batch.from_data_list([data]).to_dict()
        if mode=='train':
            assert torch.equal(batch['forces'].float(),frame['forces']), 'Native force labels changed or were masked out'
            assert bool((batch['forces_weight']==1).all()) and bool((batch['energy_weight']==1).all())
            if 'stress' in self.targets:assert bool((batch['stress_weight']==1).all())
        for name,value in list(batch.items()):
            if isinstance(value,torch.Tensor):
                if value.is_floating_point():value=value.float()
                batch[name]=value.to(self.device)
        return batch

    def validate_local(self,folder):
        if self.full or self.patches is None:return
        records=[]
        for module,name,native in self.patches.saved:
            if name!='forward' or not hasattr(module,'irreps_in1'):continue
            values=(torch.randn(17,module.irreps_in1.dim,device=self.device),
                    torch.randn(17,module.irreps_in2.dim,device=self.device),
                    torch.randn(17,module.weight_numel,device=self.device))
            cot=torch.randn_like(native(*values))
            def collect(fn):
                variables=tuple(v.detach().clone().requires_grad_() for v in values)
                output=fn(*variables)
                gradients=torch.autograd.grad(output,variables,cot)
                return (output.detach(),)+tuple(x.detach() for x in gradients)
            reference=collect(native);repeat=collect(native);candidate=collect(module.forward)
            metrics=[]
            for a,r in zip(candidate+repeat,reference+reference):
                assert torch.isfinite(a).all() and torch.isfinite(r).all()
                d=(a-r).float();denom=torch.maximum(torch.maximum(a.abs().max(),r.abs().max()),r.new_tensor(1e-7))
                epsilon=float(d.abs().max()/denom)
                metrics.append({'epsilon_max':epsilon,'passed':epsilon<=1e-5})
            index=len(records)
            torch.save(cpu({'inputs':values,'cotangent':cot,'reference':reference,'repeat':repeat,'candidate':candidate}),folder/f'ACTUAL-ECE-{index}.pt')
            records.append({'irreps':str(module.irreps_in1),'weight_numel':module.weight_numel,'metrics':metrics})
        assert records
        (folder/'ACTUAL-ECE-LOCAL.json').write_text(json.dumps(records,indent=2))
        assert all(x['passed'] for row in records for x in row['metrics']),'Actual-checkpoint ECE forward/full VJP failed'
        seed()

    def validate_memory_policy(self,folder):
        if self.full or self.patches is not None:return
        from whole import compare_tree
        for module,original,wrapped in self.memory_saved:object.__setattr__(module,'forward',original)
        seed();reference=cpu(self.forward(self.batch('train'),False))
        for module,original,wrapped in self.memory_saved:object.__setattr__(module,'forward',wrapped)
        seed();candidate=cpu(self.forward(self.batch('train'),False))
        metrics=compare_tree(candidate,reference)
        (folder/'MEMORY-POLICY-GATE.json').write_text(json.dumps(metrics,indent=2))
        torch.save({'reference':reference,'recomputed':candidate},folder/'MEMORY-POLICY-ARRAYS.pt')
        assert all(row['passed'] for row in metrics.values()),'Native recomputation E/F/stress mismatch on the 100-atom validation frame'
        seed()

    def forward(self,batch,training):
        self.model.train(training)
        out=self.model(batch)
        out={k:out[k] for k in self.targets}
        assert out['energy'].dtype==torch.float64,out['energy'].dtype
        assert out['forces'].dtype==torch.float32,out['forces'].dtype
        if 'stress' in out:assert out['stress'].dtype==torch.float32
        return out

    def infer(self):return self.forward(self.batch('ef'),False)

    def loss(self,out,batch):
        if 'tace.utils.loss' not in sys.modules:
            package=types.ModuleType('tace.utils.loss')
            package.__path__=[str(self.source/'tace/utils/loss')]
            sys.modules['tace.utils.loss']=package
        if not self.full:
            common=importlib.import_module('tace.utils.loss.common')
            common.DATA_DTYPE=torch.float32
        if not self.full and 'tace.utils.loss.registry' not in sys.modules:
            registry=types.ModuleType('tace.utils.loss.registry')
            registry.__dict__.update(torch=torch,Callable=Callable,wraps=wraps,loss_inputs=common.loss_inputs,LOSS_FN={})
            extracted(self.source/'tace/utils/loss/registry.py',{'register_loss'},registry.__dict__)
            sys.modules['tace.utils.loss.registry']=registry
        from tace.utils.loss.mse_fn import mse_energy_per_atom,mse_forces,mse_stress
        out={k:v.float() for k,v in out.items()}
        functions=[mse_energy_per_atom,mse_forces]+([] if self.full else [mse_stress])
        return sum(w*fn(out,batch) for w,fn in zip(self.loss_weights,functions))

    def train_step(self,capture=False):
        batch=self.batch('train');out=self.forward(batch,True);loss=self.loss(out,batch)
        assert loss.dtype==torch.float32,loss.dtype
        self.optimizer.zero_grad();loss.backward()
        gradients=cpu({n:p.grad for n,p in self.model.named_parameters()}) if capture else None
        clip_parameters=[p for group in self.optimizer.param_groups for p in group['params']]
        if self.clip_function is None:torch.nn.utils.clip_grad_norm_(clip_parameters,10.)
        else:self.clip_function(clip_parameters,10.)
        self.optimizer.step()
        self.ema.update()
        assert self.original_ids=={n:id(p) for n,p in self.model.named_parameters()}
        if capture:return {'output':cpu(out),'loss':cpu(loss),'gradients':gradients,'updated':cpu(dict(self.model.named_parameters())),
                          'ema':cpu(self.ema.state_dict()),'optimizer':cpu(self.optimizer.state_dict())}
        return loss.detach()

    def vjp(self):
        batch=self.batch('train');batch['positions'].requires_grad_(True)
        out=self.forward(batch,True);seed()
        cot={k:torch.randn_like(v)/max(1,v.numel())**.5 for k,v in out.items()}
        value=sum((out[k]*cot[k]).sum() for k in out)
        named=[(n,p) for n,p in self.model.named_parameters() if p.requires_grad]
        gradients=torch.autograd.grad(value,(batch['positions'],)+tuple(p for n,p in named),allow_unused=True)
        return {'output':cpu(out),'cotangent':cpu(cot),'vjp':cpu(dict(zip(['coordinates']+[n for n,p in named],gradients)))}

    def force_mixed(self):
        batch=self.batch('train');batch['positions'].requires_grad_(True)
        out=self.forward(batch,True)
        # Import exact native component without optional Trainer dependencies.
        self.loss(out,batch)
        from tace.utils.loss.mse_fn import mse_forces
        loss=self.loss_weights[1]*mse_forces({k:v.float() for k,v in out.items()},batch)
        assert loss.dtype==torch.float32,loss.dtype
        named=[(n,p) for n,p in self.model.named_parameters() if p.requires_grad]
        gradients=torch.autograd.grad(loss,(batch['positions'],)+tuple(p for n,p in named),allow_unused=True)
        return {'loss':cpu(loss),'force':cpu(out['forces']),
                'mixed':cpu(dict(zip(['coordinates']+[n for n,p in named],gradients)))}

    def state(self):return cpu({'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'ema':self.ema.state_dict(),'scheduler':self.scheduler.state_dict()})

    def restore(self,state):
        self.model.load_state_dict(state['model'],strict=True);self.optimizer.load_state_dict(state['optimizer'])
        self.ema.load_state_dict(state['ema']);self.ema.to(self.device);self.scheduler.load_state_dict(state['scheduler'])
        self.optimizer.zero_grad(set_to_none=True);seed()
