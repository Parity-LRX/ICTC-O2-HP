"""Full-array CPU derivative, covariance and independent checkpoint loading helpers."""
import copy
import json

import torch

from common import ROOT,SOURCE,sha,parameter_hashes
from canonical import install,modes


def load_models(net,original,prior):
    folder=ROOT.joinpath('CONVERT-RESULT.txt').read_text().splitlines()[0]
    converted=json.loads(__import__('pathlib').Path(folder,'RESULT.json').read_text())
    models={};identities={}
    for role in ('checkpoint_original','checkpoint_ema'):
        prior.load_parameter_role(net,original,role)
        row=converted['roles'][role]
        path=__import__('pathlib').Path(row['checkpoint'])
        assert sha(path)==row['sha256']
        payload=torch.load(path,map_location='cpu',weights_only=False)
        assert payload['adapter_sha256']==sha(SOURCE/'canonical.py')
        assert payload['source_parameter_hashes']==parameter_hashes(net.model)
        model=copy.deepcopy(net.model)
        install(model,payload['basis'],payload['kernels'])
        model.load_state_dict(payload['state_dict'],strict=True)
        assert parameter_hashes(model)==row['new_parameter_hashes']
        model.eval()
        models[role]=model
        identities[role]={'checkpoint':str(path),'sha256':row['sha256'],
            'strict_load_new_process_all_parameters_equal':True,'source_role':role,
            'new_parameter_tensors':len(list(model.parameters()))}
    return models,identities


def fixture():
    positions=torch.tensor([[-.4,.2,.7],[.8,1.3,-.5],[-1.1,.7,-.2],[.3,-.8,.4]])
    atomic=torch.tensor([14,8,6,1],dtype=torch.long)
    pairs=[(i,j) for i in range(4) for j in range(4) if i!=j]+[(0,1),(1,0),(0,2),(2,0)]
    edges=torch.tensor(pairs,dtype=torch.long).T.contiguous()
    shifts=torch.zeros(len(pairs),3)
    shifts[-2,1]=3.3;shifts[-1,1]=-3.3
    return positions,atomic,edges,shifts


def geometry(model,coords,edges,shifts):
    model.device=coords.device;model.dtype=coords.dtype;model.batch_size=1
    vectors=coords[edges[0]]-coords[edges[1]]+shifts
    distances=vectors.square().sum(-1).sqrt()
    torch.manual_seed(20261124)
    radial,envelope=model._forward_edge(distances,vectors)
    return radial,envelope


def embedding_explicit(model,atomic,radial,edges,envelope):
    module=model.edge_degree_embedding
    descriptor=torch.cat((radial,module.source_embedding(atomic[edges[0]]),
                          module.target_embedding(atomic[edges[1]])),1)
    m0=(module.rad_func(descriptor)*envelope).reshape(len(edges[0]),5,128)
    value=torch.einsum('eal,elc->eac',model.so3_rotation.wigner_inv[:,:,:5],m0)
    return torch.zeros(len(atomic),25,128,dtype=torch.float32).index_add(0,edges[1],value)/module.rescale_factor


def metric(candidate,reference,gate='o2'):
    if candidate is None or reference is None:
        return {'same_activity':candidate is reference,'pass':candidate is reference}
    a,r=candidate.detach(),reference.detach()
    assert a.shape==r.shape and a.dtype==r.dtype
    assert a.dtype in (torch.float32,torch.float64)
    assert torch.isfinite(a).all() and torch.isfinite(r).all()
    d=(a-r).float();rf=r.float();af=a.float()
    nr,nd=rf.norm(),d.norm()
    rel=float(nd/nr) if nr else (0. if not nd else None)
    max_abs=float(d.abs().max())
    denominator=torch.maximum(torch.maximum(af.abs().max(),rf.abs().max()),rf.new_tensor(1e-7))
    epsilon=float(d.abs().max()/denominator)
    components=int((d.abs()>2e-5+2e-5*rf.abs()).sum())
    result={'max_abs':max_abs,'relative_l2':rel,'epsilon_max':epsilon,
        'component_fail_count':components,'reference_norm':float(nr),'finite':True,
        'statistics_dtype':'float32','value_dtype':str(a.dtype)}
    if gate=='o2':result['pass']=epsilon<=1e-5
    elif gate=='network':result['pass']=components==0 and rel is not None and rel<=1e-5
    else:result['status']='HOLD_THRESHOLD_UNAPPROVED'
    return result


def flatten(value):
    return (value,) if isinstance(value,torch.Tensor) else tuple(value)


def collect(fn,inputs,named,seed,cotangent=None,directions=None,coordinate_only=False):
    variables=tuple(x.detach().clone().requires_grad_() for x in inputs)
    parameters=tuple(p for _,p in named)
    output=flatten(fn(*variables))
    if cotangent is None:
        torch.manual_seed(seed)
        cotangent=tuple(torch.randn_like(v)/torch.sqrt(v.new_tensor(float(v.numel()))) for v in output)
    if directions is None:
        torch.manual_seed(seed+1)
        directions=tuple(torch.randn_like(v) for v in variables)
        directions=tuple(v/v.norm() for v in directions)
        if coordinate_only:directions=(directions[0],)+tuple(torch.zeros_like(v) for v in directions[1:])
    all_variables=variables+parameters
    first=torch.autograd.grad(output,all_variables,cotangent,create_graph=True,allow_unused=True)
    tangent=sum((g*d).sum() for g,d in zip(first[:len(variables)],directions) if g is not None)
    mixed=torch.autograd.grad(tangent,all_variables,allow_unused=True)
    return {'output':[x.detach() for x in output],
        'full_vjp':[x.detach() if x is not None else None for x in first],
        'mixed':[x.detach() if x is not None else None for x in mixed],
        'labels':[f'input_{i}' for i in range(len(variables))]+[n for n,_ in named],
        'cotangent':tuple(x.detach() for x in cotangent),'directions':directions}


def comparison(candidate,reference,first_gate='o2',symmetry=False):
    gate='diagnostic' if symmetry else first_gate
    result={'output':[metric(a,r,gate) for a,r in zip(candidate['output'],reference['output'])]}
    for category in ('full_vjp','mixed'):
        category_gate='diagnostic' if symmetry else ('network' if category=='mixed' else first_gate)
        result[category]={n:metric(a,r,category_gate) for n,a,r in zip(reference['labels'],candidate[category],reference[category])}
    if symmetry:result['status']='HOLD_THRESHOLD_UNAPPROVED'
    else:
        result['output_pass']=all(v['pass'] for v in result['output'])
        result['full_vjp_pass']=all(v['pass'] for v in result['full_vjp'].values())
        result['mixed_pass']=all(v['pass'] for v in result['mixed'].values())
        result['all_pass']=all(result[k] for k in ('output_pass','full_vjp_pass','mixed_pass'))
    return result


def transformed(record,input_actions,output_actions):
    result={k:v for k,v in record.items()}
    result['output']=[fn(v) for fn,v in zip(output_actions,record['output'])]
    for category in ('full_vjp','mixed'):
        values=list(record[category])
        for i,fn in enumerate(input_actions):
            if values[i] is not None:values[i]=fn(values[i])
        result[category]=values
    return result


def apply_angular(action,value):
    return torch.einsum('ab,nbc->nac',action,value)


def execute(model,case,coords,hidden,atomic,edges,shifts,implementation):
    label,path,index=case
    modes(model,linear='expanded',sphere='factorized')
    if implementation=='candidate':
        if label=='ALL29':modes(model,linear='packed',sphere='direct')
        elif path!='edge_degree_embedding':modes(model,linear='expanded',sphere='factorized',only=(path,'packed','direct'))
    radial,envelope=geometry(model,coords,edges,shifts)
    if label in ('E00','ALL29'):
        if label=='E00' and implementation=='candidate':
            node=embedding_explicit(model,atomic,radial,edges,envelope)
            node=node+torch.nn.functional.pad(model.sphere_embedding(atomic)[:,None,:],(0,0,0,24))
        else:node=model._forward_embedding(atomic,radial,edges,envelope)
    else:node=hidden
    if label=='ALL29':indices=range(7)
    elif label=='E00':indices=(0,)
    else:indices=range(index,min(index+2,7))
    for i in indices:
        node=model.blocks[i](node,atomic[edges[0]],atomic[edges[1]],radial,edges,envelope,
                             batch=torch.zeros(len(atomic),dtype=torch.long))
    if label=='ALL29' or index==6:
        node=model.norm(node)
        atom=model.energy_block(node[:,:1].reshape(len(atomic),128)).to(torch.float64)
        return atom.sum()/model.avg_num_nodes
    next_index=1 if label=='E00' else index+2
    if next_index<7:node=model.blocks[next_index].norm_1(node)
    return node


def consumer_parameters(model,case):
    label,_,index=case
    if label=='ALL29':return list(model.named_parameters())
    if label=='E00':prefixes=('edge_degree_embedding.','sphere_embedding.','blocks.0.','blocks.1.norm_1.')
    else:
        prefixes=tuple(f'blocks.{i}.' for i in range(index,min(index+2,7)))
        if index==6:prefixes+=('norm.','energy_block.')
        elif index+2<7:prefixes+=(f'blocks.{index+2}.norm_1.',)
    return [(n,p) for n,p in model.named_parameters() if n.startswith(prefixes)]
