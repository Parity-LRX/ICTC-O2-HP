"""29 local canonical subgraphs: full output, all VJPs and mixed directional products."""
import os

import torch

from common import start,finish,save,parameter_hashes
from canonical import sites,modes,native_action
from audit_support import (load_models,fixture,geometry,embedding_explicit,collect,comparison,
                           transformed,apply_angular,metric)


def main():
    output,net,checkpoint,prior=start('local')
    models,identities=load_models(net,checkpoint,prior)
    from e3nn import o3
    qrot=o3.matrix_y(torch.tensor(.713))
    qmirror=torch.diag(torch.tensor([-1.,1.,1.]))
    all_roles={}
    for role,model in models.items():
        before=parameter_hashes(model);rows=[]
        modules=dict(model.named_modules())
        coords,atomic,edges,shifts=fixture()
        for seq,case in enumerate(sites()):
            label,path,index=case;module=modules[path]
            named=[(path+'.'+n,p) for n,p in module.named_parameters()]
            torch.manual_seed(20261124+seq)
            if label=='E00':inputs=(coords,)
            elif label.startswith('S'):inputs=(torch.randn(2,19,64)*.1,torch.randn(2,96)*.1)
            elif label.startswith('F'):inputs=(torch.randn(2,25,128)*.1,)
            else:inputs=(torch.randn(2,19,module.num_in_channels)*.1,)
            def fn(implementation,shift=shifts):
                def run(*args):
                    modes(model,linear='expanded',sphere='factorized')
                    if implementation=='candidate':modes(model,linear='expanded',sphere='factorized',only=(path,'packed','direct')) if label!='E00' else None
                    if label=='E00':
                        radial,envelope=geometry(model,args[0],edges,shift)
                        if implementation=='candidate':return embedding_explicit(model,atomic,radial,edges,envelope)
                        return module(atomic,radial,edges,envelope)
                    return module(*args)
                return run
            reference=collect(fn('reference'),inputs,named,20261124+seq)
            repeat=collect(fn('reference'),inputs,named,20261124+seq)
            candidate=collect(fn('candidate'),inputs,named,20261124+seq)
            row={'site_id':label,'module':path,'native_repeat':comparison(repeat,reference),
                 'implementation':comparison(candidate,reference),'symmetry':{}}
            for transform_name,q in (('rotation_y_0.713',qrot),('mirror_x',qmirror)):
                d=native_action(q,4 if label in ('E00',) or label.startswith('F') else 2,not(label=='E00' or label.startswith('F')))
                if label=='E00':
                    input_actions=[lambda x,q=q:x@q.T];output_actions=[lambda x,d=d:apply_angular(d,x)]
                else:
                    input_actions=[lambda x,d=d:apply_angular(d,x)]+([lambda x:x] if label.startswith('S') else [])
                    output_actions=[lambda x,d=d:apply_angular(d,x)]
                    if label.startswith('A') and int(label[1:])%2:output_actions.append(lambda x:x)
                values=tuple(fn_(v) for fn_,v in zip(input_actions,inputs))
                cot=tuple(fn_(v) for fn_,v in zip(output_actions,reference['cotangent']))
                directions=tuple(fn_(v) for fn_,v in zip(input_actions,reference['directions']))
                changed=collect(fn('reference',shifts@q.T),values,named,20261124+seq,cot,directions)
                expected=transformed(reference,input_actions,output_actions)
                row['symmetry'][transform_name]=comparison(changed,expected,symmetry=True)
            save(output/(role+'-'+label+'.json'),row);rows.append(row)
            print({'role':role,'site':label,'forward':row['implementation']['output_pass'],
                'full_vjp':row['implementation']['full_vjp_pass'],'mixed':row['implementation']['mixed_pass']},flush=True)
        assert parameter_hashes(model)==before
        all_roles[role]=rows
    finish('local',output,{'status':'ALL29_LOCAL_CANONICAL_CPU_CHECKS_RECORDED',
        'job_id':os.environ['SLURM_JOB_ID'],'roles':all_roles,'checkpoint_identities':identities,
        'symmetry_gate':'HOLD_THRESHOLD_UNAPPROVED','gpu_jobs':0,'gpu_hours':0,'training_steps':0})


if __name__=='__main__':main()
