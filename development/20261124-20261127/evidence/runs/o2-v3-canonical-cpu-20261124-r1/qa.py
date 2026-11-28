"""Actual energy/force mixed VJPs and arbitrary-update structural checks; no optimizer."""
import os

import torch

from common import start,finish,save,parameter_hashes
from canonical import sites,native_action,modes
from audit_support import load_models,fixture,execute,metric,apply_angular


def energy_force(model,positions,atomic,edges,shifts,mode,force_cotangent):
    coords=positions.clone().requires_grad_()
    energy=execute(model,('ALL29',None,None),coords,None,atomic,edges,shifts,mode)
    assert energy.dtype==torch.float64
    named=list(model.named_parameters())
    first=torch.autograd.grad(energy,(coords,)+tuple(p for _,p in named),create_graph=True,allow_unused=True)
    force=-first[0]
    mixed=torch.autograd.grad(force,tuple(p for _,p in named),force_cotangent,allow_unused=True)
    return {'energy':energy.detach(),'force':force.detach(),
        'energy_parameter_vjp':[v.detach() if v is not None else None for v in first[1:]],
        'force_parameter_mixed':[v.detach() if v is not None else None for v in mixed],
        'parameter_names':[n for n,_ in named]}


def compare(a,r,gate):
    result={k:metric(a[k],r[k],gate) for k in ('energy','force')}
    for category in ('energy_parameter_vjp','force_parameter_mixed'):
        result[category]={n:metric(x,y,gate) for n,x,y in zip(r['parameter_names'],a[category],r[category])}
    return result


def main():
    output,net,original,prior=start('qa')
    models,identities=load_models(net,original,prior)
    positions,atomic,edges,shifts=fixture()
    from e3nn import o3
    transforms=(('rotation_z_0.523',o3.matrix_z(torch.tensor(.523))),
                ('mirror_x',torch.diag(torch.tensor([-1.,1.,1.]))))
    roles={}
    for role,model in models.items():
        before=parameter_hashes(model)
        torch.manual_seed(20261124)
        cot=torch.randn_like(positions);cot=cot/cot.norm()
        reference=energy_force(model,positions,atomic,edges,shifts,'reference',cot)
        candidate=energy_force(model,positions,atomic,edges,shifts,'candidate',cot)
        repeated=energy_force(model,positions,atomic,edges,shifts,'reference',cot)
        result={'implementation':compare(candidate,reference,'network'),
            'native_repeat':compare(repeated,reference,'network'),
            'reference_energy':float(reference['energy']),'reference_force':reference['force'].tolist(),
            'force_dtype':str(reference['force'].dtype),'energy_dtype':str(reference['energy'].dtype),
            'symmetry':{}}
        for name,q in transforms:
            moved=energy_force(model,positions@q.T,atomic,edges,shifts@q.T,'reference',cot@q.T)
            expected=dict(reference);expected['force']=reference['force']@q.T
            result['symmetry'][name]=compare(moved,expected,'diagnostic')
        # Deterministic in-memory perturbation proves constraints are not a property
        # only of initialization. No optimizer, loss fitting, saved weight change.
        saved={n:p.detach().clone() for n,p in model.named_parameters() if n.endswith('.K')}
        with torch.no_grad():
            torch.manual_seed(20261125)
            for name,p in model.named_parameters():
                if name in saved:p.add_(torch.randn_like(p)*1e-4)
            updates=[]
            modules=dict(model.named_modules())
            modes(model,linear='packed',sphere='direct')
            for label,path,_ in sites():
                if not label.startswith('A'):continue
                module=modules[path]
                x=torch.randn(2,19,module.num_in_channels)*.1
                y=module(x);y=(y,) if isinstance(y,torch.Tensor) else y
                for name,q in (('rotation_y',o3.matrix_y(torch.tensor(.713))),('mirror_x',transforms[1][1])):
                    d=native_action(q,2,True)
                    z=module(apply_angular(d,x));z=(z,) if isinstance(z,torch.Tensor) else z
                    expected=[apply_angular(d,y[0])]+list(y[1:])
                    updates.append({'site_id':label,'transform':name,
                        'outputs':[metric(a,b,'diagnostic') for a,b in zip(z,expected)]})
            for name,p in model.named_parameters():
                if name in saved:p.copy_(saved[name])
        assert parameter_hashes(model)==before
        result['parameter_perturbation']={'updated_K_tensors':len(saved),'optimizer_steps':0,
            'restored_all_values_exactly':True,'checks':updates}
        save(output/(role+'-QA.json'),result);roles[role]=result
        print({'role':role,'energy':result['reference_energy'],
               'force_max_abs':float(reference['force'].abs().max()),'restored':True},flush=True)
    finish('qa',output,{'status':'REAL_ENERGY_FORCE_MIXED_VJP_AND_UPDATE_STRUCTURE_RECORDED',
        'job_id':os.environ['SLURM_JOB_ID'],'roles':roles,'checkpoint_identities':identities,
        'scope':'Fixed directed graph with Cartesian PBC shifts; no dynamic graph/stress/public-data/training qualification',
        'gpu_jobs':0,'gpu_hours':0,'optimizer_steps':0})


if __name__=='__main__':main()
