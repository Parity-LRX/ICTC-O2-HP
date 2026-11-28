import copy,gc,hashlib,importlib.util,json,logging,os,random,sys,types
from pathlib import Path
ROOT=Path(os.environ['O2_SWEEP_ROOT']);WS=ROOT.parents[1]
SOURCE_RECORD=json.loads(os.environ['O2_REPAIR_SOURCE_RECORD']) if 'O2_REPAIR_SOURCE_RECORD' in os.environ else json.loads((ROOT/'SOURCE.json').read_text())
SOURCE=Path(SOURCE_RECORD['path'])
OLD=WS/'runs/o2-uvw-triton-a100-20261125-r1';CONTROL=OLD/'snapshots/preflight-119174'
os.environ.update(O2_WHOLE_ROOT=str(OLD),O2_UVW_TILED='1',O2_UVW_SOURCE=str(SOURCE),NVIDIA_TF32_OVERRIDE='0',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',EQV3_DETERMINISTIC='1',EQX_USE_CUDA_GRAPH='0')
sys.path[:0]=[str(SOURCE),str(CONTROL)]
import numpy as np
import torch
import whole,bounded
from ictc_o2_hp.bilinear_triton import product as fused_product
def configure(gpu=False):
    logging.getLogger().setLevel(logging.WARNING);whole.configure()
    if gpu:
        assert torch.cuda.device_count()==1
        props=torch.cuda.get_device_properties(0)
        assert 'A100' in props.name and 38*2**30<props.total_memory<42*2**30
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False,default=str)+'\n')
cpu=whole.cpu
metric=whole.metric
compare_tree=whole.compare_tree
def rng():return {'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate()}
def set_rng(s):
    torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state(s['cuda']);np.random.set_state(s['numpy']);random.setstate(s['python'])
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def native_batch(self,mode):
    from ase import Atoms,units
    frame=self.frames['train'];samples=[]
    for i in range(self.copies):
        atom=Atoms(numbers=frame['z'].numpy(),positions=frame['pos'].numpy(),cell=frame['cell'].numpy(),pbc=frame['pbc'])
        data=self.net.calc.a2g.convert(atom);data.sid=f'angular-{self.order}-{i}'
        if mode=='train':
            data.energy=frame['uncorrected_total_energy'].reshape(1).float();data.forces=frame['forces'].float()
            data.stress=(frame['stress']*(-0.1*units.GPa)).reshape(1,3,3).float()
        samples.append(data)
    return self.net.collate(samples,otf_graph=True).to(self.device)
def network(order,backend='candidate',copies=1):
    obj=whole.V3('native','checkpoint_ema','cpu');obj.order=order;obj.copies=copies
    if order!=0:
        artifact=ROOT/'models'/f'L{order}.pt';p=torch.load(artifact,map_location='cpu',weights_only=False)
        ref=load_module(f'angular_reference_L{order}',ROOT/'models'/f'canonical-L{order}.py')
        assert sha(ROOT/'models'/f'canonical-L{order}.py')==p['reference_source_sha256']
        model=type(obj.model)(**p['kwargs']);ref.install(model,p['basis'],p['kernels']);model.load_state_dict(p['state_dict'],strict=True)
        from rotation_compat import init_edge_rot_mat
        model._init_edge_rot_mat=lambda edge:init_edge_rot_mat(edge,use_rotation_mask=True)
        obj.model=model;obj.trainer.model=model;obj.net.model=model;del p
        obj.identity.update(architecture=f'equiformerv3-canonical-structural-L{order}-M{order}',source_checkpoint_accuracy=False)
    obj.model=obj.model.cuda();obj.device='cuda';obj.trainer.model=obj.model;obj.net.model=obj.model
    obj.trainer.cpu=False;obj.trainer.device=torch.device('cuda')
    obj.trainer.load_optimizer();obj.trainer.load_extras();obj.optimizer=obj.trainer.optimizer;obj.ema=obj.trainer.ema
    for v in list(obj.trainer.normalizers.values())+list(obj.trainer.elementrefs.values()):v.to('cuda')
    obj.original_ids={n:id(p) for n,p in obj.model.named_parameters()};obj.batch=types.MethodType(native_batch,obj)
    obj.identity.update(lmax=obj.model.lmax,mmax=obj.model.mmax,copies=copies,atoms=100*copies,
                        active_parameters=sum(p.requires_grad for p in obj.model.parameters()),
                        source_package_sha256=SOURCE_RECORD['package_sha256'])
    namespace=obj.model.blocks[0].ga.act.forward.__func__.__globals__
    originals=[(m,m.forward) for m in obj.model.modules() if type(m).__name__=='CanonicalMLinear']
    from adapter.install import install_v3_canonical
    obj.patches=install_v3_canonical(obj.model,expected_sites=len(originals));fused=[m.forward for m,_ in originals]
    def select(which):
        for (m,old),new in zip(originals,fused):object.__setattr__(m,'forward',old if which=='native' else new)
        namespace['product']=bounded.product if which=='native' else fused_product
    obj.select=select;select(backend)
    return obj
def restore(obj,state):obj.restore(copy.deepcopy(state))
def hardware():
    p=torch.cuda.get_device_properties(0);free,total=torch.cuda.mem_get_info()
    return {'name':p.name,'total_bytes':p.total_memory,'uuid':str(p.uuid),'free_bytes':free,'runtime_total_bytes':total,
            'node':os.environ.get('SLURMD_NODENAME'),'torch':torch.__version__,'single_visible_device':torch.cuda.device_count()==1,
            'TF32':False,'AMP':False,'CUDA_Graph':False,'threads':torch.get_num_threads(),
            'allocator_configuration':os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}
def operator(family,order,parity,rows):
    torch.manual_seed(20261126+order)
    if family=='uvw':
        from ictc_o2_hp.uvw_o2 import UvwTensorProduct,ReferenceUvwTensorProduct
        model=UvwTensorProduct(32,32,32,order,parity=parity).cuda().float()
        x=(torch.randn(rows,model.input1_dimension,device='cuda')*.1).requires_grad_()
        y=(torch.randn(rows,model.input2_dimension,device='cuda')*.1).requires_grad_()
        native=types.MethodType(ReferenceUvwTensorProduct._contract,model);candidate=model._contract
        def select(which):object.__setattr__(model,'_contract',native if which=='native' else candidate)
        return model,(x,y),tuple(model.parameters()),select,{'family':'uvw','m_max':order,'parity':parity,'U':32,'V':32,'W':32,'rows':rows,'paths':len(model.paths),'parameters':model.weight.numel(),'input_widths':[model.input1_dimension,model.input2_dimension],'output_width':model.representation_dimension}
    k=torch.load(ROOT/'models'/f'BILINEAR-L{order}.pt',map_location='cuda',weights_only=True)
    d=k.shape[0];x=(torch.randn(rows,d,32,device='cuda')*.1).requires_grad_();y=(torch.randn_like(x)*.1).requires_grad_()
    state={'fn':fused_product}
    def fn(a,b):return state['fn'](k,a,b)
    def select(which):state['fn']=bounded.product if which=='native' else fused_product
    return fn,(x,y),(),select,{'family':'bilinear','l_max':order,'m_max':order,'angular_dimension':d,'H':32,'rows':rows,'coefficient_trainable':False}
