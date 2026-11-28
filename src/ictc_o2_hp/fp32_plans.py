"""Measured FP32 reduction plans; no native computation in the execution path."""
import json
from functools import lru_cache
from pathlib import Path
import torch
import triton
import triton.language as tl

@triton.jit
def _add(a,b):
    return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[a,b],dtype=tl.float32,is_pure=True,pack=1)

@triton.jit
def _mm_piece(A,B,rows,cols,q,base,acc,M,N,K,span,AM,AK,BK,BN):
    kk=base+q
    av=tl.load(A+rows[:,None]*AM+kk[None,:]*AK,(rows[:,None]<M)&(q[None,:]<span)&(kk[None,:]<K),0)
    bv=tl.load(B+kk[:,None]*BK+cols[None,:]*BN,(q[:,None]<span)&(kk[:,None]<K)&(cols[None,:]<N),0)
    return tl.dot(av,bv,acc,input_precision='ieee')

@triton.jit
def _mm(A,B,Y,M,N,K,AM,AK,BK,BN,PARTS,FOUR:tl.constexpr,BLOCK:tl.constexpr):
    rows=tl.program_id(0).to(tl.int64)*32+tl.arange(0,32).to(tl.int64)
    cols=tl.program_id(1).to(tl.int64)*32+tl.arange(0,32).to(tl.int64)
    span=tl.cdiv(K,PARTS);value=tl.full((32,32),0,tl.float32)
    for part in range(PARTS):
        base=part*span;c0=tl.full((32,32),0,tl.float32)
        if FOUR:
            c1=tl.full((32,32),0,tl.float32);c2=tl.full((32,32),0,tl.float32);c3=tl.full((32,32),0,tl.float32)
            for start in range(tl.cdiv(span,32)):
                q=start*32+tl.arange(0,16).to(tl.int64)
                # Mask the padded eight entries before each dot, preserving the
                # four native eight-product chains rather than a reassociation.
                q0=tl.where(tl.arange(0,16)<8,q,span)
                q1=tl.where(tl.arange(0,16)<8,q+8,span)
                q2=tl.where(tl.arange(0,16)<8,q+16,span)
                q3=tl.where(tl.arange(0,16)<8,q+24,span)
                c0=_mm_piece(A,B,rows,cols,q0,base,c0,M,N,K,span,AM,AK,BK,BN)
                c1=_mm_piece(A,B,rows,cols,q1,base,c1,M,N,K,span,AM,AK,BK,BN)
                c2=_mm_piece(A,B,rows,cols,q2,base,c2,M,N,K,span,AM,AK,BK,BN)
                c3=_mm_piece(A,B,rows,cols,q3,base,c3,M,N,K,span,AM,AK,BK,BN)
            c0=_add(_add(c0,c1),_add(c2,c3))
        else:
            for start in range(tl.cdiv(span,BLOCK)):
                q=start*BLOCK+tl.arange(0,BLOCK).to(tl.int64)
                c0=_mm_piece(A,B,rows,cols,q,base,c0,M,N,K,span,AM,AK,BK,BN)
        if part==0:value=c0
        else:value=_add(value,c0)
    tl.store(Y+rows[:,None]*N+cols[None,:],value,(rows[:,None]<M)&(cols[None,:]<N))

@triton.jit
def _angular_piece(K,A,B,rows,cols,q,base,acc,D,LA,LB,H,N,span,K0,K1,K2,A0,A1,A2,B0,B1,B2,SWAP:tl.constexpr):
    qq=base+q;valid=(q<span)&(qq<LA*LB)
    if SWAP:ia=qq%LA;ib=qq//LA
    else:ia=qq//LB;ib=qq%LB
    coeff=tl.load(K+rows[:,None]*K0+ia[None,:]*K1+ib[None,:]*K2,(rows[:,None]<D)&valid[None,:],0)
    nn=cols//H;hh=cols%H
    av=tl.load(A+nn[None,:]*A0+ia[:,None]*A1+hh[None,:]*A2,valid[:,None]&(nn[None,:]<N),0)
    bv=tl.load(B+nn[None,:]*B0+ib[:,None]*B1+hh[None,:]*B2,valid[:,None]&(nn[None,:]<N),0)
    product=tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[av,bv],dtype=tl.float32,is_pure=True,pack=1)
    return tl.dot(coeff,product,acc,input_precision='ieee')

@triton.jit
def _angular(K,A,B,Y,D,LA,LB,H,N,K0,K1,K2,A0,A1,A2,B0,B1,B2,OFFSET,COUNT,PARTS,FOUR:tl.constexpr,SWAP:tl.constexpr):
    rows=tl.program_id(1).to(tl.int64)*32+tl.arange(0,32).to(tl.int64)
    cols=OFFSET*H+tl.program_id(0).to(tl.int64)*32+tl.arange(0,32).to(tl.int64)
    span=tl.cdiv(LA*LB,PARTS);value=tl.full((32,32),0,tl.float32)
    for part in range(PARTS):
        base=part*span;c0=tl.full((32,32),0,tl.float32)
        if FOUR:
            c1=tl.full((32,32),0,tl.float32);c2=tl.full((32,32),0,tl.float32);c3=tl.full((32,32),0,tl.float32)
            for start in range(tl.cdiv(span,32)):
                q=start*32+tl.arange(0,16).to(tl.int64)
                q0=tl.where(tl.arange(0,16)<8,q,span);q1=tl.where(tl.arange(0,16)<8,q+8,span)
                q2=tl.where(tl.arange(0,16)<8,q+16,span);q3=tl.where(tl.arange(0,16)<8,q+24,span)
                c0=_angular_piece(K,A,B,rows,cols,q0,base,c0,D,LA,LB,H,N,span,K0,K1,K2,A0,A1,A2,B0,B1,B2,SWAP)
                c1=_angular_piece(K,A,B,rows,cols,q1,base,c1,D,LA,LB,H,N,span,K0,K1,K2,A0,A1,A2,B0,B1,B2,SWAP)
                c2=_angular_piece(K,A,B,rows,cols,q2,base,c2,D,LA,LB,H,N,span,K0,K1,K2,A0,A1,A2,B0,B1,B2,SWAP)
                c3=_angular_piece(K,A,B,rows,cols,q3,base,c3,D,LA,LB,H,N,span,K0,K1,K2,A0,A1,A2,B0,B1,B2,SWAP)
            c0=_add(_add(c0,c1),_add(c2,c3))
        else:
            for start in range(tl.cdiv(span,16)):
                q=start*16+tl.arange(0,16).to(tl.int64)
                c0=_angular_piece(K,A,B,rows,cols,q,base,c0,D,LA,LB,H,N,span,K0,K1,K2,A0,A1,A2,B0,B1,B2,SWAP)
        if part==0:value=c0
        else:value=_add(value,c0)
    nn=cols//H;hh=cols%H
    tl.store(Y+(nn[None,:]*D+rows[:,None])*H+hh[None,:],value,(rows[:,None]<D)&(cols[None,:]<(OFFSET+COUNT)*H)&(nn[None,:]<N))

def mm_key(left,right):
    return 'mm|'+','.join(map(str,(*left.shape,right.shape[1],*left.stride(),*right.stride())))

def angular_key(k,a,b,count):
    return 'angular|'+','.join(map(str,(*k.shape,a.shape[-1],count,*k.stride(),*a.stride(),*b.stride())))

@lru_cache(None)
def plans():
    path=Path(__file__).with_name('fp32_reduction_plans.json')
    return json.loads(path.read_text())['plans'] if path.exists() else {}

def channel(left,right,out):
    plan=plans().get(mm_key(left,right))
    if plan is None:return False
    m,k=left.shape;n=right.shape[1]
    _mm[(triton.cdiv(m,32),triton.cdiv(n,32))](left,right,out,m,n,k,*left.stride(),*right.stride(),plan['parts'],plan['four'],plan.get('block',16),num_warps=4)
    return True

def angular(k,a,b,out,offset,count):
    plan=plans().get(angular_key(k,a,b,64 if count>=64 else count))
    if plan is None:return False
    n,_,h=a.shape;d,la,lb=k.shape
    _angular[(triton.cdiv(count*h,32),triton.cdiv(d,32))](k,a,b,out,d,la,lb,h,n,*k.stride(),*a.stride(),*b.stride(),offset,count,plan['parts'],plan['four'],plan.get('swap',False),num_warps=4)
    return True
