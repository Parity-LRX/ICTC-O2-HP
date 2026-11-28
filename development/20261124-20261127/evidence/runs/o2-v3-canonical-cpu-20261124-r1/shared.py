"""Frozen native subgraphs and explicit smooth geometry/parameter-dependent gauge."""
import torch


def fixture(collinear=False):
    position = torch.tensor([[0.,0.,0.],[0.,1.6,0.]]) if collinear else torch.tensor([[-.4,.2,.7],[.8,1.3,-.5]])
    atomic = torch.tensor([14,8],dtype=torch.long)
    edges = torch.tensor([[0,1],[1,0]],dtype=torch.long)
    return position,atomic,edges


def geometry(net,coords,edges):
    # The native model forward normally initializes these runtime-only fields.
    net.model.device=coords.device
    net.model.dtype=coords.dtype
    net.model.batch_size=1
    vectors = coords[edges[0]]-coords[edges[1]]
    distance = vectors.square().sum(-1).sqrt()
    torch.manual_seed(20261124)
    radial,envelope = net.model._forward_edge(distance,vectors)
    return vectors,radial,envelope


def radial_raw(net,ga,coords,atomic,edges):
    vectors = coords[edges[0]]-coords[edges[1]]
    distance = vectors.square().sum(-1).sqrt()
    rbf = net.model.distance_expansion(distance)
    source = ga.source_embedding(atomic[edges[0]])
    target = ga.target_embedding(atomic[edges[1]])
    values = ga.rad_func.net(torch.cat((rbf,source,target),1)).reshape(len(distance),5,256)
    return values,vectors


def local_radial(raw):
    degrees = torch.cat((torch.arange(5),torch.arange(1,5),torch.arange(1,5),torch.arange(2,5),torch.arange(2,5)))
    return raw.index_select(1,degrees)


def value_chain(ga,x,weights,envelope):
    value,extra = ga.so2_linear_1(x*weights)
    scalar = extra[:,ga.split_m0_channels_list[0]:]
    value = ga.so2_linear_2(ga.act(value,scalar))
    alpha = extra[:,:ga.split_m0_channels_list[0]].view(len(x),ga.num_heads,ga.attn_alpha_channels)
    alpha = ga.alpha_dropout(ga.alpha_act(ga.alpha_norm(alpha)))
    logits = torch.einsum('bik,ik->bi',alpha,ga.alpha_dot)
    # Each edge in this fixture has its own complete incoming segment.
    receiver = torch.arange(len(x),dtype=torch.long)
    alpha = ga.attn_softmax(logits,receiver,num_nodes=len(x),exp_rescale=envelope)*envelope
    alpha = ga.attn_weights_dropout(alpha.view(len(x),1,ga.num_heads,1))
    return (value.view(len(x),19,ga.num_heads,ga.attn_value_channels)*alpha).reshape(len(x),19,128)


def complex_block(jac,m,cout,cin):
    offset,count = (5,4) if m==1 else (13,3)
    r,i = torch.arange(offset,offset+count),torch.arange(offset+count,offset+2*count)
    def b(rows,cols):
        return jac.index_select(0,rows).index_select(2,cols).reshape(count*cout,count*cin)
    rr,ri,ir,ii = b(r,r),b(r,i),b(i,r),b(i,i)
    linear = torch.complex((rr+ii)*.5,(ir-ri)*.5)
    anti = torch.complex((rr-ii)*.5,(ir+ri)*.5)
    return linear,anti


def matrix(module,m):
    layer=module.so2_m_linear[m-1]
    a,b=layer.fc.weight.detach().split(layer.out_features,0)
    return torch.complex(a,b)


def reflection(unitary,z):
    return unitary@(unitary.T@z.conj())


def eta_for(net,producer,coords,atomic,edges):
    vectors=coords[edges[0]]-coords[edges[1]]
    distance=vectors.square().sum(-1).sqrt()
    rbf=net.model.distance_expansion(distance)
    description=torch.cat((rbf,producer.source_embedding(atomic[edges[0]]),producer.target_embedding(atomic[edges[1]])),1)
    raw=producer.rad_func.net(description)
    weights=torch.linspace(.5,1.5,raw.shape[-1])
    q=(raw*weights).mean()
    direction=vectors[0]/distance[0]
    return .125*torch.tanh(q+.125*direction[0]+.0625*direction[2])


def shear(values,eta,inverse=False,m_primary=True):
    """Apply U or U^-1. Rank-one N has disjoint a/b supports, hence N^2=0."""
    sign=-1. if inverse else 1.
    if values.ndim==2:
        flat=values
        average=flat[:,1:].sum(-1,keepdim=True)/torch.sqrt(flat.new_tensor(float(flat.shape[-1]-1)))
        first=flat[:,:1]+sign*eta*average
        return torch.cat((first,flat[:,1:]),-1)
    from experimental.models.equiformer_v3.so3 import CoefficientMappingModule
    q,c=values.shape[1:]
    maximum=2 if q==19 else 4
    if not m_primary:
        p=CoefficientMappingModule(4,maximum).to_m
        values=torch.einsum('ab,nbc->nac',p,values)
    chunks=[]
    offset=0
    for m in range(maximum+1):
        count=5-m
        if m==0:
            part=values[:,:count].reshape(len(values),-1)
            average=part[:,1:].sum(-1,keepdim=True)/torch.sqrt(part.new_tensor(float(part.shape[-1]-1)))
            chunks.append(torch.cat((part[:,:1]+sign*eta*average,part[:,1:]),-1).reshape(len(values),count,c))
            offset+=count
        else:
            real=values[:,offset:offset+count].reshape(len(values),-1)
            imag=values[:,offset+count:offset+2*count].reshape(len(values),-1)
            scale=torch.sqrt(real.new_tensor(float(real.shape[-1]-1)))
            real_first=real[:,:1]-sign*eta*imag[:,1:].sum(-1,keepdim=True)/scale
            imag_first=imag[:,:1]+sign*eta*real[:,1:].sum(-1,keepdim=True)/scale
            chunks.extend((torch.cat((real_first,real[:,1:]),-1).reshape(len(values),count,c),
                           torch.cat((imag_first,imag[:,1:]),-1).reshape(len(values),count,c)))
            offset+=2*count
    result=torch.cat(chunks,1)
    if not m_primary:
        result=torch.einsum('ba,nbc->nac',p,result)
    return result


def metric(candidate,reference):
    if candidate is None or reference is None:
        return {'same_activity':candidate is reference,'pass':candidate is reference}
    a,r=candidate.detach(),reference.detach()
    assert a.dtype==r.dtype==torch.float32 and a.shape==r.shape
    assert torch.isfinite(a).all() and torch.isfinite(r).all()
    delta=a-r;nr=r.norm();nd=delta.norm()
    ratio=float(nd/nr) if nr else (0. if not nd else None)
    failures=int((delta.abs()>2e-5+2e-5*r.abs()).sum())
    return {'max_abs':float(delta.abs().max()),'relative_l2':ratio,'component_fail_count':failures,
            'pass':failures==0 and ratio is not None and ratio<=1e-5,'reference_norm':float(nr)}
