import gc, hashlib, importlib.util, json, os, sys
from pathlib import Path
ROOT=Path(os.environ.get('O2_CAUSE_ROOT',Path(__file__).resolve().parent))
WS=ROOT.parents[1]
SWEEP=WS/'runs/o2-dd7a-angular-capacity-a100-20261126-r1'
os.environ['O2_SWEEP_ROOT']=str(SWEEP)
sys.path.insert(0,str(SWEEP/'snapshots/full-120424'))
import common as c
torch=c.torch
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(path):return torch.load(path,map_location='cpu',weights_only=False)
def metric(a,r):
    if a is None or r is None:return {'passed':a is r}
    if not a.is_floating_point():return {'passed':torch.equal(a,r)}
    d=(a-r).float();norm=r.float().norm();dn=d.norm()
    rel=float(dn/norm) if norm else (0.0 if not dn else None)
    bad=int((d.abs()>2e-5+2e-5*r.float().abs()).sum())
    return {'passed':bool(torch.isfinite(a).all() and torch.isfinite(r).all()) and bad==0 and rel is not None and rel<=1e-5,
            'component_fail_count':bad,'max_abs':float(d.abs().max()),'relative_l2':rel,
            'reference_norm':float(norm),'different_elements':int((a!=r).sum()),'elements':a.numel()}
def tree(a,r,prefix=''):
    if isinstance(r,dict):
        out={}
        for k in r:out.update(tree(a[k],r[k],prefix+'/'+str(k)))
        return out
    if isinstance(r,(tuple,list)):
        out={}
        for i,(aa,rr) in enumerate(zip(a,r)):out.update(tree(aa,rr,prefix+'/'+str(i)))
        return out
    return {prefix:metric(a,r)} if isinstance(r,torch.Tensor) or r is None else {prefix:{'passed':a==r}}
def brief(rows):
    return {'roles':len(rows),'failures':{k:v for k,v in rows.items() if not v['passed']},
            'different_roles':sum(v.get('different_elements',0)>0 for v in rows.values()),
            'maximum_relative_l2':max((v.get('relative_l2',0) or 0) for v in rows.values())}
def cpu_model(order):
    obj=c.whole.V3('native','checkpoint_ema','cpu')
    if order:
        p=load(SWEEP/'models'/f'L{order}.pt')
        ref=c.load_module(f'cause_cpu_L{order}',SWEEP/'models'/f'canonical-L{order}.py')
        model=type(obj.model)(**p['kwargs']);ref.install(model,p['basis'],p['kernels'])
        model.load_state_dict(p['state_dict'],strict=True)
        obj.model=model;obj.trainer.model=model;obj.net.model=model
    obj.trainer.load_optimizer();obj.trainer.load_extras()
    obj.optimizer=obj.trainer.optimizer;obj.ema=obj.trainer.ema
    return obj
def group_names(obj):
    names={id(p):n for n,p in obj.model.named_parameters()}
    return [[names[id(p)] for p in g['params']] for g in obj.optimizer.param_groups]
