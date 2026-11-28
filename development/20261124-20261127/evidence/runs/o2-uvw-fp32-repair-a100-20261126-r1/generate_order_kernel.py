"""AMD code generation from independently measured FP32 contraction trees."""
import ast,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';sys.setrecursionlimit(10000)
source=Path((ROOT/'REDUCTION-INFERENCE.txt').read_text().strip());manifest=json.loads((source/'RESULT.json').read_text())
header='''"""Generated FP32 FMA/add schedules for the frozen A100 NN adjoint shapes.
Indices describe arithmetic order only, independently of model parameters/data.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _add_rn(a,b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;",constraints="=f,f,f",args=[a,b],dtype=tl.float32,is_pure=True,pack=1)

@triton.jit
def _chain(A,B,P,rows,cols,seed,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
           SAM:tl.constexpr,SAK:tl.constexpr,SBK:tl.constexpr,SBN:tl.constexpr,
           BASE:tl.constexpr,LENGTH:tl.constexpr):
    acc=seed
    for chunk in range(tl.cdiv(LENGTH,16)):
        q=chunk*16+tl.arange(0,16)
        k=tl.load(P+BASE+q,q<LENGTH,0).to(tl.int64)
        a=tl.load(A+rows[:,None]*SAM+k[None,:]*SAK,(rows[:,None]<M)&(q[None,:]<LENGTH)&(k[None,:]<K),0)
        b=tl.load(B+k[:,None]*SBK+cols[None,:]*SBN,(q[:,None]<LENGTH)&(k[:,None]<K)&(cols[None,:]<N),0)
        acc=tl.dot(a,b,acc,input_precision="ieee")
    return acc

'''
parts=[header];tables={};summaries=[]
for entry in manifest['records']:
    payload=json.loads(Path(entry['path']).read_text());m,n,k=payload['shape_mnk'];tree=payload['tree']
    indices=[];lines=[];counter=[0];chains=[]
    def fresh():
        label='v'+str(counter[0]);counter[0]+=1;return label
    def chain(sequence,seed):
        offset=len(indices);indices.extend(sequence);indices.extend([0]*((-len(sequence))%16))
        name=fresh();lines.append(f'    {name}=_chain(A,B,P,rows,cols,{seed},M,N,K,SAM,SAK,SBK,SBN,{offset},{len(sequence)})')
        chains.append(sequence);return name
    def emit(node):
        if isinstance(node,int):return chain([node],'zero')
        if 'fma' in node:
            sequence=[]
            while isinstance(node,dict) and 'fma' in node:
                leaf,node=node['fma'];assert isinstance(leaf,int);sequence.append(leaf)
            sequence.reverse()
            if isinstance(node,int):return chain([node]+sequence,'zero')
            return chain(sequence,emit(node))
        left,right=node['add'];left=emit(left);right=emit(right);name=fresh()
        lines.append(f'    {name}=_add_rn({left},{right})');return name
    result=emit(tree)
    assert sorted(v for c in chains for v in c)==list(range(k)),(k,n,'Missing or repeated contraction terms')
    name=f'_nn_k{k}_n{n}'
    prefix=f'''@triton.jit
def {name}(A,B,P,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,SAM:tl.constexpr,SAK:tl.constexpr,SBK:tl.constexpr,SBN:tl.constexpr):
    rows=tl.program_id(0).to(tl.int64)*32+tl.arange(0,32).to(tl.int64)
    cols=tl.program_id(1).to(tl.int64)*32+tl.arange(0,32).to(tl.int64)
    zero=tl.full((32,32),0,tl.float32)
'''
    parts.append(prefix+'\n'.join(lines)+f'\n    tl.store(C+rows[:,None]*N+cols[None,:],{result},(rows[:,None]<M)&(cols[None,:]<N))\n\n')
    tables[(k,n)]=indices
    summaries.append({'k':k,'n':n,'chains':len(chains),'chain_lengths':[len(c) for c in chains],
        'chain_indices':chains,'tree_sha256':hashlib.file_digest(Path(entry['path']).open('rb'),'sha256').hexdigest()})
parts.append('_INDICES='+repr(tables)+'\n_CACHE={}\n_KERNELS={(512,128):_nn_k512_n128,(384,96):_nn_k384_n96}\n')
parts.append('''
def matmul(a,b):
    m,k=a.shape;n=b.shape[1];key=(k,n)
    assert key in _KERNELS and a.dtype==b.dtype==torch.float32 and a.is_cuda
    cache_key=(a.device.index,k,n)
    if cache_key not in _CACHE:_CACHE[cache_key]=torch.tensor(_INDICES[key],dtype=torch.int64,device=a.device)
    out=a.new_empty((m,n))
    if m and n:_KERNELS[key][(triton.cdiv(m,32),triton.cdiv(n,32))](a,b,_CACHE[cache_key],out,m,n,k,*a.stride(),*b.stride(),num_warps=4)
    return out
''')
code=''.join(parts);ast.parse(code);path=ROOT/'native_order_kernel.py';path.write_text(code)
(ROOT/'NATIVE-ORDER-GENERATION.json').write_text(json.dumps({'status':'INDEX_COVERAGE_AND_SYNTAX_PASS','source':str(source),
    'kernel_source_sha256':hashlib.sha256(code.encode()).hexdigest(),'shapes':summaries},indent=2)+'\n')
print(json.dumps(summaries),flush=True)
