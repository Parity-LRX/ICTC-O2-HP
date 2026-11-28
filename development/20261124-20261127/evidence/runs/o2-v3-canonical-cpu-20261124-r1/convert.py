"""Calibrate actual bases, project 14 modules, and emit independent CPU checkpoints."""
import copy
import os

import torch

from common import start,finish,save,sha,parameter_hashes,SOURCE,ROOT
from canonical import MODEL_ID,calibrate,install,CanonicalMLinear


def main():
    output,net,checkpoint,prior=start('convert')
    basis,records,actions=calibrate()
    torch.save({'basis':basis,'actions':actions},output/'CALIBRATION.pt')
    role_models={};identities={};old_hashes={}
    for role in ('checkpoint_original','checkpoint_ema'):
        identities[role]=prior.load_parameter_role(net,checkpoint,role)
        old_hashes[role]=parameter_hashes(net.model)
        role_models[role]=copy.deepcopy(net.model)
    results={}
    for role,model in role_models.items():
        old_parameters={n:p.detach().clone() for n,p in model.named_parameters()}
        migration,kernels,coefficients=install(model,basis)
        assert len(migration)==28
        copied={n:bool(torch.equal(p,old_parameters[n])) for n,p in model.named_parameters() if n in old_parameters}
        assert all(copied.values())
        module_records=[]
        torch.manual_seed(20261124)
        for name,module in model.named_modules():
            if not isinstance(module,CanonicalMLinear):continue
            x=torch.randn(2,2,module.in_features)
            with torch.no_grad():
                a=module(x)
                module.execution='expanded';b=module(x);module.execution='packed'
                aa,bb=module.expanded_weights()
                module_records.append({'module':name,'packed_expanded_max_abs':float((a-b).abs().max()),
                    'packed_expanded_relative_l2':float((a-b).norm()/b.norm()),
                    'raw_projected_A_norm':float(aa.norm()),'raw_projected_B_norm':float(bb.norm())})
        state={n:p.detach().clone() for n,p in model.state_dict().items()}
        payload={'schema':'ictc-v3-canonical-o2-v1','model_id':MODEL_ID,
            'state_dict':state,'basis':basis,'kernels':kernels,'source_role':role,
            'original_checkpoint_sha256':'429ccded98163122e7ba588d78e2441653f37f3e091e106c432807fe373c8f98',
            'config':copy.deepcopy(checkpoint['config']),
            'normalizers':copy.deepcopy(checkpoint.get('normalizers',{})),
            'elementrefs':copy.deepcopy(checkpoint.get('elementrefs',{})),
            'adapter_sha256':sha(SOURCE/'canonical.py'),'source_parameter_hashes':old_hashes[role],
            'source_role_identity':identities[role],'migration':migration,
            'coefficient_definition':'Gaunt structure projected from frozen grid; normalization scalars frozen in coefficient receipts',
            'optimizer':None,'training_steps':0,'gpu_hours':0}
        temporary=output/(role+'-canonical.pt')
        torch.save(payload,temporary)
        digest=sha(temporary)
        destination=output/(role+'-canonical-'+digest[:16]+'.pt')
        temporary.rename(destination)
        # Independent model object; strict load rejects missing or unexpected keys.
        restored=copy.deepcopy(role_models[role])
        loaded=torch.load(destination,map_location='cpu',weights_only=False)
        restored.load_state_dict(loaded['state_dict'],strict=True)
        assert parameter_hashes(restored)==parameter_hashes(model)
        results[role]={'checkpoint':str(destination),'sha256':digest,'source_identity':identities[role],
            'source_parameters':len(old_parameters),'unchanged_copied_parameters':len(copied),
            'projected_parameter_tensors':len(migration),'migration':migration,
            'coefficient_diagnostics':coefficients,'projection_execution':module_records,
            'strict_reload_all_parameters_equal':True,'new_parameter_hashes':parameter_hashes(model)}
        save(output/(role+'-CONVERSION.json'),results[role])
        print({'role':role,'checkpoint':str(destination),'projected_tensors':len(migration),'copied_tensors':len(copied)},flush=True)
    assert parameter_hashes(net.model)==old_hashes['checkpoint_ema']
    result={'status':'CANONICAL_CPU_CHECKPOINTS_CONSTRUCTED_NOT_YET_QUALIFIED',
        'job_id':os.environ['SLURM_JOB_ID'],'model_id':MODEL_ID,'calibration':records,'roles':results,
        'representation_numeric_gate':'UNAPPROVED: residuals are diagnostics, not an invented FP32 acceptance threshold',
        'gpu_jobs':0,'gpu_hours':0,'training_steps':0}
    finish('convert',output,result)


if __name__=='__main__':main()
