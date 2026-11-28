"""CPU-only canonical O2 architecture. No frozen source or checkpoint is modified."""
import types

import torch
import torch.nn.functional as F

MODEL_ID = 'equiformerv3-ictc-o2-canonical-min-v1'


def angular_layout(maximum, primary):
    from experimental.models.equiformer_v3.so3 import CoefficientMappingModule
    mapping = CoefficientMappingModule(4, maximum)
    full = CoefficientMappingModule(4, 4)
    ids = full.coefficient_idx(4, maximum)
    projection = torch.zeros(len(ids), 25)
    projection[:, ids] = mapping.to_m if primary else torch.eye(len(ids))
    return projection, ids


def native_action(matrix, maximum, primary):
    """Natural parity action in actual frozen Wigner/m-major conventions."""
    from experimental.models.equiformer_v3.so3 import SO3Rotation
    det = torch.linalg.det(matrix)
    proper = matrix * det
    rotation = SO3Rotation(4, 4, use_rotation_mask=True)
    rotation.set_wigner(proper[None])
    full_projection, _ = angular_layout(4, True)
    degree_parity = torch.cat([torch.full((2*l+1,), float((-1)**l)) for l in range(5)])
    parity = degree_parity if float(det) < 0 else torch.ones(25)
    global_action = full_projection.T @ rotation.wigner[0] @ torch.diag(parity)
    projection, _ = angular_layout(maximum, primary)
    return projection @ global_action @ projection.T


def calibrate():
    from e3nn import o3
    angle = torch.tensor(.371)
    mirror = torch.diag(torch.tensor([-1., 1., 1.]))
    rotation = o3.matrix_y(angle)
    ds = native_action(mirror, 4, True)
    dr = native_action(rotation, 4, True)
    basis, records = {}, []
    candidates = (torch.eye(2), torch.tensor([[0.,-1.],[1.,0.]]),
                  torch.diag(torch.tensor([1.,-1.])), torch.tensor([[0.,1.],[1.,0.]]))
    offset = 5
    for m in range(1, 5):
        count = 5-m
        target = torch.tensor([[torch.cos(m*angle),-torch.sin(m*angle)],
                               [torch.sin(m*angle),torch.cos(m*angle)]])
        for j, l in enumerate(range(m, 5)):
            ids = torch.tensor([offset+j, offset+count+j])
            s, r = ds[ids][:,ids], dr[ids][:,ids]
            errors = [(c@s@c.T-candidates[2]).square().sum()+(c@r@c.T-target).square().sum() for c in candidates]
            selected = int(torch.stack(errors).argmin())
            c = candidates[selected].clone()
            basis[f'{l},{m}'] = c
            records.append({'l':l,'m':m,'candidate':selected,'C':c.tolist(),
                'reflection_max_abs':float((c@s@c.T-candidates[2]).abs().max()),
                'rotation_max_abs':float((c@r@c.T-target).abs().max())})
        offset += 2*count
    return basis, records, {'reflection':ds,'rotation':dr,'angle':angle}


class CanonicalMLinear(torch.nn.Module):
    def __init__(self, old, basis):
        super().__init__()
        self.m, self.in_features, self.out_features = old.m, old.in_features, old.out_features
        count = 5-self.m
        cin, cout = old.in_features//count, old.out_features//count
        c = torch.stack([basis[f'{l},{self.m}'] for l in range(self.m,5)])
        self.register_buffer('C_in',c.repeat_interleave(cin,0))
        self.register_buffer('C_out',c.repeat_interleave(cout,0))
        a,b = old.fc.weight.detach().chunk(2,0)
        t = torch.stack((torch.stack((a,-b)),torch.stack((b,a))))
        tc = torch.einsum('oar,rsoi,ibs->aboi',self.C_out,t,self.C_in)
        self.K = torch.nn.Parameter(((tc[0,0]+tc[1,1])*.5).contiguous())
        self.execution = 'packed'
        self.projection_diagnostic = {
            'old_shape':list(old.fc.weight.shape),'new_shape':list(self.K.shape),
            'old_parameter_elements':old.fc.weight.numel(),'new_parameter_elements':self.K.numel(),
            'canonical_discarded_B_norm':float(((tc[1,0]-tc[0,1])*.5).norm()),
            'canonical_A_norm':float(self.K.detach().norm()),
            'canonical_diagonal_mismatch':float((tc[0,0]-tc[1,1]).abs().max())}

    def expanded_weights(self):
        q = torch.einsum('oar,ias->oirs',self.C_out,self.C_in)
        a = ((q[:,:,0,0]+q[:,:,1,1])*.5)*self.K
        b = ((q[:,:,1,0]-q[:,:,0,1])*.5)*self.K
        return a,b

    def forward(self,x_m,concat_outputs=True):
        if self.execution == 'expanded':
            a,b = self.expanded_weights()
            z = F.linear(x_m,torch.cat((a,b),0))
            real,imag = z.split(self.out_features,-1)
            result = (real[:,:1]-imag[:,1:2],real[:,1:2]+imag[:,:1])
        else:
            x = torch.einsum('iab,ebi->eai',self.C_in,x_m)
            y = F.linear(x,self.K)
            y = torch.einsum('oab,eao->ebo',self.C_out,y)
            result = (y[:,:1],y[:,1:2])
        return torch.cat(result,1) if concat_outputs else result


def frozen_kernel(grid):
    return torch.einsum('aq,qb,qc->abc',grid.from_grid_mat.detach(),
                         grid.to_grid_mat.detach(),grid.to_grid_mat.detach())


def natural_kernel(grid):
    """Gaunt invariant structure; scalar normalizations calibrated to the frozen grid.

    The fitted scalars define a new, explicitly recorded reference. They do not
    assert bitwise equivalence to the finite-grid baseline.
    """
    from e3nn import o3
    old = frozen_kernel(grid)
    projection, ids = angular_layout(grid.mmax,False)
    # Build l-major restricted blocks before the fixed m-major permutation.
    if grid.use_m_primary:
        from experimental.models.equiformer_v3.so3 import CoefficientMappingModule
        p = CoefficientMappingModule(4,grid.mmax).to_m
        old_l = torch.einsum('ia,abc,bj,ck->ijk',p.T,old,p,p)
    else:
        p = torch.eye(len(ids));old_l=old
    result = torch.zeros_like(old_l)
    degree_ids = {l:torch.nonzero((ids>=l*l)&(ids<(l+1)**2)).flatten() for l in range(5)}
    rows=[]
    for lo in range(5):
        for l1 in range(5):
            for l2 in range(5):
                if (lo+l1+l2)%2 or not abs(l1-l2)<=lo<=l1+l2:continue
                io,i1,i2 = (degree_ids[l] for l in (lo,l1,l2))
                mo,m1,m2 = (ids[ii]-l*l for ii,l in zip((io,i1,i2),(lo,l1,l2)))
                cg = o3.wigner_3j(lo,l1,l2,dtype=torch.float32)
                cg = cg[mo][:,m1][:,:,m2]
                old_block=old_l[io[:,None,None],i1[None,:,None],i2[None,None,:]]
                denom=cg.square().sum()
                if not denom:continue
                scalar=(cg*old_block).sum()/denom
                block=scalar*cg
                result[io[:,None,None],i1[None,:,None],i2[None,None,:]]=block
                rows.append({'lout':lo,'l1':l1,'l2':l2,'normalization':float(scalar),
                    'fit_max_abs':float((block-old_block).abs().max())})
    k=torch.einsum('ai,ijk,bj,ck->abc',p,result,p,p)
    return k,old,rows


def product(k,a,b,mode='direct'):
    if mode=='factorized':
        intermediate=torch.einsum('abc,nck->nabk',k,b)
        return torch.einsum('nabk,nbk->nak',intermediate,a)
    return torch.einsum('abc,nbk,nck->nak',k,a,b)


def sphere_forward(self,inputs,scalars):
    if self.canonical_execution=='frozen_grid':return self.canonical_original_forward(inputs,scalars)
    c=inputs.shape[-1]
    scalars=scalars.reshape(len(inputs),1,-1)
    scalar=self.scalar_act(scalars[:,:,:c])
    gate=self.gate_act(scalars[:,:,c:])
    a,b=inputs.chunk(2,-1)
    value=product(self.canonical_kernel,a,b,self.canonical_execution)*gate
    return torch.cat((value[:,:1]+scalar,value[:,1:]),1)


def ffn_forward(self,inputs):
    if self.canonical_execution=='frozen_grid':return self.canonical_original_forward(inputs)
    scalar=self.scalar_mlp(inputs[:,:1])
    x=self.so3_linear_1(inputs)
    gm=self.grid_mlp
    gate=gm.gate_act(gm.gating_linear(inputs[:,:1]))
    a,b=F.linear(x,gm.grid_linear_1.weight).chunk(2,-1)
    value=product(self.canonical_kernel,a*gate,b,self.canonical_execution)
    value=F.linear(value,gm.grid_linear_2.weight)
    value=torch.cat((value[:,:1]+scalar,value[:,1:]),1)
    return self.so3_linear_2(value)


def install(model,basis,kernels=None):
    migrations=[];coefficients={};diagnostics={}
    for i,block in enumerate(model.blocks):
        for attr in ('so2_linear_1','so2_linear_2'):
            layer=getattr(block.ga,attr)
            for j,old in enumerate(layer.so2_m_linear):
                new=CanonicalMLinear(old,basis)
                migrations.append({'old_key':f'blocks.{i}.ga.{attr}.so2_m_linear.{j}.fc.weight',
                    'new_key':f'blocks.{i}.ga.{attr}.so2_m_linear.{j}.K',**new.projection_diagnostic})
                layer.so2_m_linear[j]=new
        for family,module,grid,forward in (('S',block.ga.act,block.ga.act.so3_grid,sphere_forward),
                                           ('F',block.ffn,block.ffn.so3_grid,ffn_forward)):
            key=family+str(i+1).zfill(2)
            if kernels is None:
                k,old,norms=natural_kernel(grid)
                diagnostics[key]={'frozen_grid_max_abs_difference':float((k-old).abs().max()),
                    'frozen_grid_relative_l2':float((k-old).norm()/old.norm()),'normalizations':norms,
                    'definition':'Gaunt invariant structure with frozen-grid least-squares scalar normalization; new model reference'}
            else:k=kernels[key]
            coefficients[key]=k.clone()
            module.register_buffer('canonical_kernel',k.clone())
            module.canonical_original_forward=module.forward
            module.forward=types.MethodType(forward,module)
            module.canonical_execution='direct'
    model.canonical_model_id=MODEL_ID
    return migrations,coefficients,diagnostics


def modes(model,linear='packed',sphere='direct',only=None):
    for name,module in model.named_modules():
        if isinstance(module,CanonicalMLinear):module.execution=linear
        if hasattr(module,'canonical_execution'):module.canonical_execution=sphere
    if only:
        path,linear_mode,sphere_mode=only
        module=dict(model.named_modules())[path]
        if hasattr(module,'so2_m_linear'):
            for child in module.so2_m_linear:child.execution=linear_mode
        elif hasattr(module,'canonical_execution'):module.canonical_execution=sphere_mode


def sites():
    result=[('E00','edge_degree_embedding',None)]
    for i in range(7):
        result.extend([(f'A{2*i+1:02}',f'blocks.{i}.ga.so2_linear_1',i),
                       (f'S{i+1:02}',f'blocks.{i}.ga.act',i),
                       (f'A{2*i+2:02}',f'blocks.{i}.ga.so2_linear_2',i),
                       (f'F{i+1:02}',f'blocks.{i}.ffn',i)])
    return result
