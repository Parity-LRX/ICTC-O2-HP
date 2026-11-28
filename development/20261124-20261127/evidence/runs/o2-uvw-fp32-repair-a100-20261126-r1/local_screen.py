"""Numerical selection only; formal acceptance will independently run on AMD."""
import gc,json,os
from pathlib import Path
import torch
from common import ROOT,configure,save,sha
from candidate_kernels import CONFIGS,matmul
def descriptor(a,r):
    d=(a-r).float();scale=torch.maximum(a.abs().max(),r.abs().max()).clamp_min(1e-7)
    return {'epsilon_max':float(d.abs().max()/scale),'relative_l2':float(d.norm()/r.norm()),'max_abs':float(d.abs().max())}
def main():
    configure();source=Path((ROOT/'CAPTURE-RESULT.txt').read_text().strip())
    audit=Path((ROOT/'CAPTURE-AUDIT.txt').read_text().strip());assert json.loads((audit/'RESULT.json').read_text())['status']=='REAL_TRAINING_CAPTURE_AUDIT_PASS'
    capture=json.loads((source/'CAPTURE.json').read_text());out=ROOT/'results'/('local-screen-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    data=[]
    for backend,rows in capture['backends'].items():
        for row in rows:
            record=torch.load(row['path'],map_location='cpu',weights_only=False)
            x=record['x'].cuda().requires_grad_();w=record['K'].cuda().requires_grad_()
            g=record['adjoints']['energy_forward_and_force']['g'].cuda()
            y=torch.nn.functional.linear(x,w)
            dx,dw=torch.autograd.grad(y,(x,w),g)
            data.append((backend,row['name'],x.detach().reshape(-1,x.shape[-1]),w.detach(),g.reshape(-1,g.shape[-1]),
                (y.detach().reshape(-1,y.shape[-1]),dx.detach().reshape(-1,dx.shape[-1]),dw.detach())))
    results={}
    for label,config in CONFIGS.items():
        values=[]
        try:
            for backend,name,x,w,g,refs in data:
                actual=(matmul(x,w.T,config),matmul(g,w,config),matmul(x.T,g,config).T)
                values.append({'backend':backend,'site':name,'weight_shape':list(w.shape),
                    'roles':{role:descriptor(a,r) for role,a,r in zip(('forward','input_adjoint','weight_adjoint'),actual,refs)}})
            results[label]={'status':'SELECTION_MEASURED','config':config,'sites':values,
                'maxima':{role:{key:max(v['roles'][role][key] for v in values) for key in ('epsilon_max','relative_l2','max_abs')}
                    for role in ('forward','input_adjoint','weight_adjoint')}}
            results[label]['by_shape']={str(shape):{role:{key:max(v['roles'][role][key] for v in values if v['weight_shape']==shape)
                for key in ('epsilon_max','relative_l2','max_abs')} for role in ('forward','input_adjoint','weight_adjoint')}
                for shape in ([256,1024],[192,768],[512,128],[384,96])}
        except Exception as error:
            results[label]={'status':'UNSUPPORTED','config':config,'error':repr(error)}
        save(out/'SELECTION.json',results);print(json.dumps({label:{k:v for k,v in results[label].items() if k!='sites'}}),flush=True)
    save(out/'CONDITION.json',{'status':'NUMERICAL_SELECTION_ONLY_NOT_ACCEPTANCE','actual_sites':len(data),'rows':10504,
        'capture_manifest_sha256':sha(source/'CAPTURE.json'),'kernel_source_sha256':sha(Path(__file__).with_name('candidate_kernels.py')),
        'FP32':True,'TF32':False,'AMP':False,'CUDA_Graph':False,'gpu':torch.cuda.get_device_name()})
    (ROOT/'LOCAL-SCREEN.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
