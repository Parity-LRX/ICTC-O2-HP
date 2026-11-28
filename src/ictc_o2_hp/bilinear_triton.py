"""FP32 bounded angular bilinear contraction with recursively differentiable VJPs."""
import torch
import triton
import triton.language as tl

@triton.jit
def _add(a,b):
    return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[a,b],dtype=tl.float32,is_pure=True,pack=1)

@triton.jit
def _piece(K,A,B,rows,cols,base,start,acc,
           OA:tl.constexpr,LA:tl.constexpr,LB:tl.constexpr,H:tl.constexpr,N:tl.constexpr,
           K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,
           A0:tl.constexpr,A1:tl.constexpr,A2:tl.constexpr,
           B0:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,SPAN:tl.constexpr,WIDTH:tl.constexpr,LANE:tl.constexpr):
    local=start+LANE*WIDTH+tl.arange(0,16).to(tl.int64)
    q=base+local
    valid=(tl.arange(0,16)<WIDTH)&(local<SPAN)&(q<LA*LB)
    ia=q//LB;ib=q%LB
    coeff=tl.load(K+rows[:,None]*K0+ia[None,:]*K1+ib[None,:]*K2,(rows[:,None]<OA)&valid[None,:],0)
    n=cols//H;h=cols%H
    aa=tl.load(A+n[None,:]*A0+ia[:,None]*A1+h[None,:]*A2,valid[:,None]&(n[None,:]<N),0)
    bb=tl.load(B+n[None,:]*B0+ib[:,None]*B1+h[None,:]*B2,valid[:,None]&(n[None,:]<N),0)
    outer=tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[aa,bb],dtype=tl.float32,is_pure=True,pack=1)
    return tl.dot(coeff,outer,acc,input_precision='ieee')

@triton.jit
def _region(K,A,B,rows,cols,OA:tl.constexpr,LA:tl.constexpr,LB:tl.constexpr,H:tl.constexpr,N:tl.constexpr,
              K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,A0:tl.constexpr,A1:tl.constexpr,A2:tl.constexpr,
              B0:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,PARTS:tl.constexpr,FOUR:tl.constexpr):
    span=tl.cdiv(LA*LB,PARTS)
    value=tl.full((BM,BN),0,tl.float32)
    for part in tl.static_range(PARTS):
        base=tl.full((),part*span,tl.int64)
        c0=tl.full((BM,BN),0,tl.float32)
        if FOUR:
            c1=tl.full((BM,BN),0,tl.float32);c2=tl.full((BM,BN),0,tl.float32);c3=tl.full((BM,BN),0,tl.float32)
            for block in range(tl.cdiv(span,32)):
                start=tl.cast(block,tl.int64)*32
                c0=_piece(K,A,B,rows,cols,base,start,c0,OA,LA,LB,H,N,K0,K1,K2,A0,A1,A2,B0,B1,B2,span,8,0)
                c1=_piece(K,A,B,rows,cols,base,start,c1,OA,LA,LB,H,N,K0,K1,K2,A0,A1,A2,B0,B1,B2,span,8,1)
                c2=_piece(K,A,B,rows,cols,base,start,c2,OA,LA,LB,H,N,K0,K1,K2,A0,A1,A2,B0,B1,B2,span,8,2)
                c3=_piece(K,A,B,rows,cols,base,start,c3,OA,LA,LB,H,N,K0,K1,K2,A0,A1,A2,B0,B1,B2,span,8,3)
            c0=_add(_add(c0,c1),_add(c2,c3))
        else:
            for block in range(tl.cdiv(span,16)):
                start=tl.cast(block,tl.int64)*16
                c0=_piece(K,A,B,rows,cols,base,start,c0,OA,LA,LB,H,N,K0,K1,K2,A0,A1,A2,B0,B1,B2,span,16,0)
        if part==0:value=c0
        else:value=_add(value,c0)
    return value

@triton.jit
def _bilinear(K,A,B,Y,OA:tl.constexpr,LA:tl.constexpr,LB:tl.constexpr,H:tl.constexpr,N:tl.constexpr,
              K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,A0:tl.constexpr,A1:tl.constexpr,A2:tl.constexpr,
              B0:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,OFFSET:tl.constexpr,COUNT:tl.constexpr,PARTS:tl.constexpr,FOUR:tl.constexpr):
    rows=tl.program_id(1).to(tl.int64)*BM+tl.arange(0,BM).to(tl.int64)
    start=tl.full((),OFFSET*H,tl.int64)+tl.program_id(0).to(tl.int64)*BN
    cols=start+tl.arange(0,BN).to(tl.int64)
    value=_region(K,A,B,rows,cols,OA,LA,LB,H,N,K0,K1,K2,A0,A1,A2,B0,B1,B2,BM,BN,PARTS,FOUR)
    n=cols//H;h=cols%H
    tl.store(Y+(n[None,:]*OA+rows[:,None])*H+h[None,:],value,(rows[:,None]<OA)&(cols[None,:]<(OFFSET+COUNT)*H)&(n[None,:]<N))

class Bilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,k,a,b):
        if k.requires_grad:raise ValueError('Angular coefficients are the original fixed buffer; active model weights are not frozen')
        if k.ndim!=3 or a.ndim!=3 or b.ndim!=3 or a.shape[0]!=b.shape[0] or a.shape[2]!=b.shape[2] or k.shape[1:]!=(a.shape[1],b.shape[1]):raise ValueError('Incompatible angular bilinear shapes')
        if not all(v.is_cuda and v.dtype==torch.float32 and v.device==a.device for v in (k,a,b)):raise TypeError('CUDA FP32 operands required')
        ctx.save_for_backward(k,a,b)
        n,_,h=a.shape;oa,la,lb=k.shape
        y=a.new_empty((n,oa,h))
        if n and h:
            # At most two grid launches, no row loop, chunk list or concatenation.
            # Keep native arithmetic boundaries without charging main blocks for
            # a tail variant's larger shared-memory allocation.
            main_parts=1;main_four=True;tail_parts=1;tail_four=True
            column=(k.stride()==(1,la*lb,la))
            if (oa,la,lb,h)==(19,19,19,32):
                main_parts=4 if column else 1
                tail_parts=({4:1,24:5} if column else {4:3,24:4}).get(n%64,1)
            elif (oa,la,lb,h)==(25,25,25,512):
                main_four=column;tail_four=False
                tail_parts=1 if column else {36:3,24:2}.get(n%64,1)
            main=n//64*64
            from .fp32_plans import angular
            if main:
                if not angular(k,a,b,y,0,main):
                    _bilinear[(triton.cdiv(main*h,32),triton.cdiv(oa,32))](k,a,b,y,oa,la,lb,h,n,*k.stride(),*a.stride(),*b.stride(),32,32,0,main,main_parts,main_four,num_warps=4)
            if main<n:
                if not angular(k,a,b,y,main,n-main):
                    _bilinear[(triton.cdiv((n-main)*h,32),triton.cdiv(oa,32))](k,a,b,y,oa,la,lb,h,n,*k.stride(),*a.stride(),*b.stride(),32,32,main,n-main,tail_parts,tail_four,num_warps=4)
        return y
    @staticmethod
    def backward(ctx,g):
        k,a,b=ctx.saved_tensors
        da=Bilinear.apply(k.permute(1,0,2),g,b) if ctx.needs_input_grad[1] else None
        db=Bilinear.apply(k.permute(2,0,1),g,a) if ctx.needs_input_grad[2] else None
        return None,da,db

def product(k,a,b,mode='direct'):
    if mode!='direct':raise ValueError('Only the original direct canonical bilinear is supported')
    return Bilinear.apply(k,a,b)
