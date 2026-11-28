"""Bounded FP32 accumulation experiments; all derivative products stay in Triton."""
import torch
import triton
import triton.language as tl

CONFIGS={
    's32':(32,32,32,4,1),
    's16':(32,32,16,4,1),
    's64':(32,32,64,4,1),
    'n64':(32,64,32,4,1),
    'm64':(64,32,32,4,1),
    'w8':(32,32,32,8,1),
    'p2k16':(32,32,16,4,2),
    'p2k32':(32,32,32,4,2),
    'p4k16':(32,32,16,4,4),
    'p4k32':(32,32,32,4,4),
    'i2k16':(32,32,16,4,2,1),
    'i4k16':(32,32,16,4,4,1),
    'i2k32':(32,32,32,4,2,1),
    'i4k32':(32,32,32,4,4,1),
    'h2k16':(32,32,16,4,2,2),
    'h4k16':(32,32,16,4,4,2),
    'even8':(32,32,32,4,1,3),
    'reverse8':(32,32,32,4,1,4),
    'pair8':(32,32,32,4,1,5),
}
for tree in range(1,15):CONFIGS['h4t'+str(tree)]=(32,32,16,4,4,2,tree)
for tree in range(5):CONFIGS['h8t'+str(tree)]=(32,32,16,4,8,2,tree)
CONFIGS['p8k16']=(32,32,16,4,8)
CONFIGS['i8k16']=(32,32,16,4,8,1)

@triton.jit
def _dot_step(A,B,rows,cols,k,acc,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
              SAM:tl.constexpr,SAK:tl.constexpr,SBK:tl.constexpr,SBN:tl.constexpr):
    a=tl.load(A+rows[:,None]*SAM+k[None,:]*SAK,(rows[:,None]<M)&(k[None,:]<K),0)
    b=tl.load(B+k[:,None]*SBK+cols[None,:]*SBN,(k[:,None]<K)&(cols[None,:]<N),0)
    return tl.dot(a,b,acc,input_precision='ieee')

@triton.jit
def _mm(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
        SAM:tl.constexpr,SAK:tl.constexpr,SBK:tl.constexpr,SBN:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,PARTS:tl.constexpr,MODE:tl.constexpr,TREE:tl.constexpr):
    rows=tl.program_id(0).to(tl.int64)*BM+tl.arange(0,BM).to(tl.int64)
    cols=tl.program_id(1).to(tl.int64)*BN+tl.arange(0,BN).to(tl.int64)
    kk=tl.arange(0,BK).to(tl.int64)
    if MODE==3:kk=(kk//8)*8+(kk%4)*2+(kk%8)//4
    if MODE==4:kk=(kk//8)*8+7-(kk%8)
    if MODE==5:kk=(kk//8)*8+(kk%2)+(kk%8)//4*2+((kk//2)%2)*4
    c0=tl.full((BM,BN),0,tl.float32)
    if PARTS>=2:c1=tl.full((BM,BN),0,tl.float32)
    if PARTS>=4:
        c2=tl.full((BM,BN),0,tl.float32);c3=tl.full((BM,BN),0,tl.float32)
    if PARTS==8:
        c4=tl.full((BM,BN),0,tl.float32);c5=tl.full((BM,BN),0,tl.float32)
        c6=tl.full((BM,BN),0,tl.float32);c7=tl.full((BM,BN),0,tl.float32)
    for tile in range(tl.cdiv(K,BK*PARTS)):
        k=tl.cast(tile,tl.int64)*BK*PARTS+kk
        jump=BK
        if MODE==1:
            k=tl.cast(tile,tl.int64)*BK*PARTS+kk*PARTS
            jump=1
        if MODE==2:
            k=tl.cast(tile,tl.int64)*BK+kk
            jump=tl.cdiv(K,BK*PARTS)*BK
        c0=_dot_step(A,B,rows,cols,k,c0,M,N,K,SAM,SAK,SBK,SBN)
        if PARTS>=2:c1=_dot_step(A,B,rows,cols,k+jump,c1,M,N,K,SAM,SAK,SBK,SBN)
        if PARTS>=4:
            c2=_dot_step(A,B,rows,cols,k+2*jump,c2,M,N,K,SAM,SAK,SBK,SBN)
            c3=_dot_step(A,B,rows,cols,k+3*jump,c3,M,N,K,SAM,SAK,SBK,SBN)
        if PARTS==8:
            c4=_dot_step(A,B,rows,cols,k+4*jump,c4,M,N,K,SAM,SAK,SBK,SBN)
            c5=_dot_step(A,B,rows,cols,k+5*jump,c5,M,N,K,SAM,SAK,SBK,SBN)
            c6=_dot_step(A,B,rows,cols,k+6*jump,c6,M,N,K,SAM,SAK,SBK,SBN)
            c7=_dot_step(A,B,rows,cols,k+7*jump,c7,M,N,K,SAM,SAK,SBK,SBN)
    if PARTS==2:c0=c0+c1
    if PARTS==4:
        if TREE==0:c0=(c0+c1)+(c2+c3)
        elif TREE==1:c0=(c0+c2)+(c1+c3)
        elif TREE==2:c0=(c0+c3)+(c1+c2)
        elif TREE==3:c0=((c0+c1)+c2)+c3
        elif TREE==4:c0=((c0+c1)+c3)+c2
        elif TREE==5:c0=((c0+c2)+c1)+c3
        elif TREE==6:c0=((c0+c2)+c3)+c1
        elif TREE==7:c0=((c0+c3)+c1)+c2
        elif TREE==8:c0=((c0+c3)+c2)+c1
        elif TREE==9:c0=((c1+c2)+c0)+c3
        elif TREE==10:c0=((c1+c2)+c3)+c0
        elif TREE==11:c0=((c1+c3)+c0)+c2
        elif TREE==12:c0=((c1+c3)+c2)+c0
        elif TREE==13:c0=((c2+c3)+c0)+c1
        else:c0=((c2+c3)+c1)+c0
    if PARTS==8:
        if TREE==0:c0=((c0+c1)+(c2+c3))+((c4+c5)+(c6+c7))
        elif TREE==1:c0=((((((c0+c1)+c2)+c3)+c4)+c5)+c6)+c7
        elif TREE==2:c0=((c0+c4)+(c2+c6))+((c1+c5)+(c3+c7))
        elif TREE==3:c0=((((((c0+c4)+c2)+c6)+c1)+c5)+c3)+c7
        else:c0=(((c0+c1)+(c2+c3))+(c4+c5))+(c6+c7)
    tl.store(C+rows[:,None]*N+cols[None,:],c0,(rows[:,None]<M)&(cols[None,:]<N))

def matmul(a,b,config):
    assert a.is_cuda and a.dtype==b.dtype==torch.float32
    assert a.ndim==b.ndim==2 and a.shape[1]==b.shape[0]
    if config[0]==0:
        plan=config[1];n,k=b.shape[1],a.shape[1]
        recipes={(256,1024):CONFIGS['h2k16'],(192,768):CONFIGS['h4k16']}
        if plan>=6:recipes[(192,768)]=CONFIGS['h4t3']
        if plan in (2,4,5):recipes[(128,512)]=CONFIGS['h4k16']
        if plan in (2,3):recipes[(96,384)]=CONFIGS['p4k32']
        if plan==5:recipes[(96,384)]=CONFIGS['h4k16']
        if plan in (7,8,9):recipes[(128,512)]=CONFIGS['h8t2']
        if plan in (7,10,11,12,13):recipes[(96,384)]=CONFIGS['h4t1']
        if plan==8:recipes[(96,384)]=CONFIGS['h8t2']
        if plan==11:recipes[(128,512)]=CONFIGS['h4t1']
        if plan==12:recipes[(128,512)]=CONFIGS['h8t1']
        if plan==13:recipes[(128,512)]=CONFIGS['h8t0']
        config=recipes.get((n,k),CONFIGS['s32'])
    m,k=a.shape;n=b.shape[1];out=a.new_empty((m,n));bm,bn,bk,warps,parts=config[:5]
    mode=config[5] if len(config)>5 else 0
    tree=config[6] if len(config)>6 else 0
    if m and n:
        _mm[(triton.cdiv(m,bm),triton.cdiv(n,bn))](a,b,out,m,n,k,*a.stride(),*b.stride(),bm,bn,bk,parts,mode,tree,num_warps=warps)
    return out

class Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b,forward_config,backward_config):
        ctx.save_for_backward(a,b);ctx.backward_config=backward_config
        return matmul(a,b,forward_config)
    @staticmethod
    def backward(ctx,g):
        a,b=ctx.saved_tensors;c=ctx.backward_config
        da=Matmul.apply(g,b.T,c,c) if ctx.needs_input_grad[0] else None
        db=Matmul.apply(a.T,g,c,c) if ctx.needs_input_grad[1] else None
        return da,db,None,None

def channel_matmul(forward_config,backward_config=None):
    backward_config=backward_config or forward_config
    def call(a,b):return Matmul.apply(a,b,forward_config,backward_config)
    return call
