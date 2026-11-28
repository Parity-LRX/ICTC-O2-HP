"""Exact bilinear canonical kernel with bounded temporary tensor residency."""
import torch

class Bilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,k,a,b):
        assert not k.requires_grad,'This helper supports the frozen angular coefficient buffer only'
        ctx.save_for_backward(k,a,b)
        # Same einsum and complete arrays, independently batch rows; no graph detach in adjoint.
        values=[torch.einsum('abc,nbk,nck->nak',k,a[start:start+64],b[start:start+64])
                for start in range(0,len(a),64)]
        return torch.cat(values,0)

    @staticmethod
    def backward(ctx,g):
        k,a,b=ctx.saved_tensors
        da=Bilinear.apply(k.permute(1,0,2),g,b) if ctx.needs_input_grad[1] else None
        db=Bilinear.apply(k.permute(2,0,1),g,a) if ctx.needs_input_grad[2] else None
        return None,da,db

def product(k,a,b,mode='direct'):
    assert mode=='direct'
    return Bilinear.apply(k,a,b)

def validate():
    from whole import metric,seed
    seed();k=torch.randn(19,19,19);a=torch.randn(137,19,8,requires_grad=True);b=torch.randn_like(a,requires_grad=True)
    u=torch.randn_like(a);direction=torch.randn_like(a)
    def collect(fn):
        out=fn(k,a,b)
        first=torch.autograd.grad(out,(a,b),u,create_graph=True)
        second=torch.autograd.grad(sum((x*direction).sum() for x in first),(a,b))
        return (out,)+first+second
    ref=collect(lambda k,a,b:torch.einsum('abc,nbk,nck->nak',k,a,b))
    got=collect(product)
    rows=[metric(x.detach(),y.detach()) for x,y in zip(got,ref)]
    assert all(x['passed'] for x in rows),rows
    return rows
