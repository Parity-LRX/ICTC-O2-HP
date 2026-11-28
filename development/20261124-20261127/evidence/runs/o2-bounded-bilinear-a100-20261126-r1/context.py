import copy,hashlib,os,pickle,random,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(os.environ['O2_KROOT']);WS=ROOT.parents[1];BASE=WS/'runs/o2-uvw-fp32-repair-a100-20261126-r1'
PREVIOUS=WS/'runs/o2-uvw-target-gemm-a100-20261126-r1'
SOURCE=Path(os.environ.get('O2_REPAIR_SOURCE',WS/'ICTC-O2-HP/src'))
def package_sha(source=SOURCE):
    h=hashlib.sha256()
    for p in sorted(source.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.cu','.cubin','.ptx','.json'):
            h.update(str(p.relative_to(source)).encode()+b'\0'+hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()
os.environ['O2_REPAIR_ROOT']=str(BASE);os.environ['O2_REPAIR_SOURCE']=str(SOURCE)
sys.path.insert(0,str(BASE/'snapshots/qualification-119423/control'))
from common import configure,factory as original_factory,common_state,startup,cpu,save,sha,state_identity,seed
def factory(route,variant,role,device):
    model=original_factory(route,variant,role,device)
    import bounded
    model._bilinear_globals=model.model.blocks[0].ga.act.forward.__func__.__globals__
    model._bilinear_reference=bounded.product
    from ictc_o2_hp.bilinear_triton import product
    model._bilinear_candidate=product
    model._bilinear_globals['product']=product if variant=='candidate' else bounded.product
    return model
def rng():return {'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate()}
def rng_id(r):return {k:hashlib.sha256(v.numpy().tobytes() if isinstance(v,torch.Tensor) else pickle.dumps(v)).hexdigest() for k,v in r.items()}
def set_rng(r):
    torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state(r['cuda']);np.random.set_state(r['numpy']);random.setstate(r['python'])
def restore(model,state):
    model.restore(dict(state,optimizer=copy.deepcopy(state['optimizer'])))
    assert state_identity(model.state())==state_identity(state)
def switches(model):
    old=[(m,f) for m,n,f in model.patches.saved if n=='forward'];new=[m.forward for m,_ in old];assert len(old)==28
    def select(backend):
        native=backend.startswith('native')
        for (m,f),candidate in zip(old,new):object.__setattr__(m,'forward',f if native else candidate)
        model._bilinear_globals['product']=model._bilinear_reference if native or backend.startswith('baseline') else model._bilinear_candidate
    return select
