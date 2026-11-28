"""Distinguish differentiable force-adjoint execution from ordinary first backward."""
import json,os
from pathlib import Path
import torch
from common import ROOT,configure,cpu,save
configure();source=Path((ROOT/'CAPTURE-RESULT.txt').read_text().strip());rows=json.loads((source/'CAPTURE.json').read_text())['backends']['native'][:4]
out=ROOT/'results'/('adjoint-reference-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True);results=[]
def diff(a,r):
    d=(a-r).float();return {'max_abs':float(d.abs().max()),'relative_l2':float(d.norm()/r.norm()),'stride':list(a.stride())}
for row in rows:
    v=torch.load(row['path'],weights_only=False,map_location='cpu');x=v['x'].cuda().requires_grad_();k=v['K'].cuda().requires_grad_()
    g=v['adjoints']['energy_forward_and_force']['g'].cuda().requires_grad_();y=torch.nn.functional.linear(x,k)
    true=torch.autograd.grad(y,x,g,create_graph=True,retain_graph=True)[0]
    false=torch.autograd.grad(y,x,g,create_graph=False,retain_graph=True)[0]
    both=torch.autograd.grad(y,(x,k),g,create_graph=True,retain_graph=True)[0]
    flat_y=x.reshape(-1,x.shape[-1])@k.T
    flat=torch.autograd.grad(flat_y,x,g.reshape(-1,g.shape[-1]),create_graph=True)[0]
    direct=(g.reshape(-1,g.shape[-1])@k).reshape(x.shape)
    transposed=(k.T@g.reshape(-1,g.shape[-1]).T).T.reshape(x.shape)
    result={'site':row['name'],'x_stride':list(x.stride()),'g_stride':list(g.stride()),'y_grad_fn':str(y.grad_fn),
        'true_stride':list(true.stride()),'ordinary_vs_differentiable':diff(false,true),'both_vs_input_only':diff(both,true),
        'flat_vs_differentiable':diff(flat,true),'direct_vs_differentiable':diff(direct,true),'transpose_vs_differentiable':diff(transposed,true)}
    torch.save(cpu({'true':true,'false':false,'both':both,'flat':flat,'direct':direct,'transposed':transposed}),out/f"{row['index']:02d}.pt")
    results.append(result)
save(out/'RESULT.json',results);(ROOT/'ADJOINT-REFERENCE.txt').write_text(str(out)+'\n');print(json.dumps(results),flush=True)
