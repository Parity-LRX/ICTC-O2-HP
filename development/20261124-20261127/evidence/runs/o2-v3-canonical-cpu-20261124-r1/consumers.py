"""Original producer/consumer boundaries, mixed coordinate derivatives and energy readout."""
import os

import torch

from common import start,finish,save,parameter_hashes
from canonical import sites,native_action
from audit_support import (load_models,fixture,execute,consumer_parameters,collect,
                           comparison,transformed,apply_angular)


def main():
    output,net,checkpoint,prior=start('consumers')
    models,identities=load_models(net,checkpoint,prior)
    from e3nn import o3
    transformations=(('rotation_x_0.417',o3.matrix_x(torch.tensor(.417))),
                     ('mirror_x',torch.diag(torch.tensor([-1.,1.,1.]))))
    positions,atomic,edges,shifts=fixture()
    all_roles={}
    for role,model in models.items():
        before=parameter_hashes(model);rows=[]
        for seq,case in enumerate(sites()+[('ALL29',None,None)]):
            label,path,index=case
            torch.manual_seed(20261124+(index or 0));hidden=torch.randn(4,25,128)*.1
            use_hidden=label not in ('E00','ALL29')
            inputs=(positions,hidden) if use_hidden else (positions,)
            named=consumer_parameters(model,case)
            def fn(implementation,shift=shifts):
                return lambda *args:execute(model,case,args[0],args[1] if use_hidden else None,
                                            atomic,edges,shift,implementation)
            seed=20261124+seq
            reference=collect(fn('reference'),inputs,named,seed,coordinate_only=True)
            repeat=collect(fn('reference'),inputs,named,seed,coordinate_only=True)
            candidate=collect(fn('candidate'),inputs,named,seed,coordinate_only=True)
            row={'site_id':label,'module':path,'consumer_scope':'embedding/block with next block/norm, terminal energy readout where applicable',
                'native_repeat':comparison(repeat,reference,'network'),
                'implementation':comparison(candidate,reference,'network'),'symmetry':{},
                'output_dtype':str(reference['output'][0].dtype),
                'real_energy_probe':label=='ALL29' or index==6,
                'active_parameter_roles':len(named)}
            if row['real_energy_probe']:
                # Real model E/F readout; derivatives are not an optimizer/training step.
                row['energy']=float(reference['output'][0])
                row['energy_vjp_coordinate_gradient_norm']=float(reference['full_vjp'][0].norm())
            for transform_name,q in transformations:
                d=native_action(q,4,False)
                input_actions=[lambda x,q=q:x@q.T]+([lambda x,d=d:apply_angular(d,x)] if use_hidden else [])
                output_actions=[(lambda x:x) if row['real_energy_probe'] else (lambda x,d=d:apply_angular(d,x))]
                values=tuple(fn_(v) for fn_,v in zip(input_actions,inputs))
                cot=tuple(fn_(v) for fn_,v in zip(output_actions,reference['cotangent']))
                directions=tuple(fn_(v) for fn_,v in zip(input_actions,reference['directions']))
                changed=collect(fn('reference',shifts@q.T),values,named,seed,cot,directions,coordinate_only=True)
                row['symmetry'][transform_name]=comparison(changed,transformed(reference,input_actions,output_actions),symmetry=True)
            save(output/(role+'-'+label+'.json'),row);rows.append(row)
            print({'role':role,'site':label,'forward':row['implementation']['output_pass'],
                'full_vjp':row['implementation']['full_vjp_pass'],'mixed':row['implementation']['mixed_pass'],
                'output_dtype':row['output_dtype']},flush=True)
        assert parameter_hashes(model)==before
        all_roles[role]=rows
    finish('consumers',output,{'status':'ALL29_CONSUMER_COORDINATE_DERIVATIVES_AND_READOUT_RECORDED',
        'job_id':os.environ['SLURM_JOB_ID'],'roles':all_roles,'checkpoint_identities':identities,
        'fixture':{'positions':positions.tolist(),'atomic_numbers':atomic.tolist(),'edges':edges.tolist(),
            'periodic_cartesian_shifts':shifts.tolist(),'fixed_graph':True,'dynamic_neighbor_rebuild_qualified':False},
        'symmetry_gate':'HOLD_THRESHOLD_UNAPPROVED','optimizer_steps':0,'gpu_jobs':0,'gpu_hours':0})


if __name__=='__main__':main()
