"""Candidate eight-product chains; offline compilation before GPU qualification."""
import triton
import triton.language as tl
@triton.jit
def _add_rn(a,b):
    return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[a,b],dtype=tl.float32,is_pure=True,pack=1)
@triton.jit
def _chain8(A,B,rows,cols,base,start,acc,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,SPAN:tl.constexpr,
    SAM:tl.constexpr,SAK:tl.constexpr,SBK:tl.constexpr,SBN:tl.constexpr,LANE:tl.constexpr):
    for inner in tl.static_range(8):
        local=start+8*LANE+inner;k=base+local;valid=(local<SPAN)&(k<K)
        a=tl.load(A+rows*SAM+k*SAK,(rows<M)&valid,0)
        b=tl.load(B+k*SBK+cols*SBN,(cols<N)&valid,0)
        acc=tl.fma(a[:,None],b[None,:],acc)
    return acc
@triton.jit
def _weight_fma8(A,B,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
    SAM:tl.constexpr,SAK:tl.constexpr,SBK:tl.constexpr,SBN:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,PARTS:tl.constexpr):
    rows=tl.program_id(0).to(tl.int64)*BM+tl.arange(0,BM).to(tl.int64)
    cols=tl.program_id(1).to(tl.int64)*BN+tl.arange(0,BN).to(tl.int64)
    part=tl.program_id(2).to(tl.int64);span=tl.cdiv(K,PARTS);base=part*span
    c0=tl.full((BM,BN),0,tl.float32);c1=tl.full((BM,BN),0,tl.float32)
    c2=tl.full((BM,BN),0,tl.float32);c3=tl.full((BM,BN),0,tl.float32)
    for start in range(tl.cdiv(span,32)):
        q=tl.cast(start,tl.int64)*32
        c0=_chain8(A,B,rows,cols,base,q,c0,M,N,K,span,SAM,SAK,SBK,SBN,0)
        c1=_chain8(A,B,rows,cols,base,q,c1,M,N,K,span,SAM,SAK,SBK,SBN,1)
        c2=_chain8(A,B,rows,cols,base,q,c2,M,N,K,span,SAM,SAK,SBK,SBN,2)
        c3=_chain8(A,B,rows,cols,base,q,c3,M,N,K,span,SAM,SAK,SBK,SBN,3)
    value=_add_rn(_add_rn(c0,c1),_add_rn(c2,c3))
    tl.store(P+(part*M+rows[:,None])*N+cols[None,:],value,(rows[:,None]<M)&(cols[None,:]<N))
