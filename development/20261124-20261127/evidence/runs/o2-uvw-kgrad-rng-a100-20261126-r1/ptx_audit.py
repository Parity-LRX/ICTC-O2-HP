"""Offline sm80 FP32 arithmetic audit, including all four long-K shapes."""
import hashlib,importlib.util,json,os,re,sys
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
r=json.loads((ROOT/'FULL-PREFLIGHT.json').read_text());path=Path(r['snapshot'])/'ictc-source/ictc_o2_hp/uvw_triton.py'
assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==r['operator_source_sha256']
spec=importlib.util.spec_from_file_location('accepted_kgrad_audit',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
out=ROOT/'results'/('ptx-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir();rows=[]
def compile_one(name,kernel,pointers,constants,need_fma,need_add):
    signature={**{k:'*fp32' for k in pointers},**{k:'constexpr' for k in constants}}
    compiled=triton.compile(ASTSource(kernel,signature=signature,constexprs=constants),target=GPUTarget('cuda',80,32),options={'num_warps':4})
    ptx=compiled.asm['ptx'];forbidden=re.findall(r'\b(?:mma|wmma|wgmma)\.|\.(?:tf32|bf16|f16)\b',ptx)
    assert not forbidden
    if need_fma:assert 'fma.rn.f32' in ptx
    if need_add:assert 'add.rn.f32' in ptx
    (out/(name+'.ptx')).write_text(ptx)
    rows.append({'case':name,'constants':constants,'PTX_sha256':hashlib.sha256(ptx.encode()).hexdigest(),'FP32_fma':need_fma,'explicit_FP32_add':need_add,'forbidden_instructions':forbidden,'shared_bytes':compiled.metadata.shared})
for name,m,n,k,am,ak,bk,bn,parts in (
    ('forward-p2',10504,256,1024,1024,1,1,1024,2),('forward-p4',10504,192,768,768,1,1,768,4),
    ('input-p6',10504,128,512,512,1,128,1,6),('input-p3',10504,96,384,384,1,96,1,3),('generic',274,1024,256,256,1,1024,1,1)):
    c=dict(M=m,N=n,K=k,SAM=am,SAK=ak,SBK=bk,SBN=bn,BM=32,BN=32,BK=16 if parts>1 else 32,PARTS=parts)
    compile_one(name,module._uvw_mm,('A','B','C'),c,True,parts>1)
for m,n,p in ((256,1024,4),(192,768,3),(512,128,5),(384,96,9)):
    c=dict(M=m,N=n,K=10504,SAM=1,SAK=m,SBK=n,SBN=1,BM=32,BN=32,PARTS=p)
    compile_one(f'weight-{m}-{n}',module._uvw_weight_partials,('A','B','P'),c,True,True)
    compile_one(f'reduce-{m}-{n}',module._uvw_weight_reduce,('P','C'),dict(SIZE=m*n,PARTS=p,BLOCK=256),False,True)
result={'status':'FP32_SM80_PTX_AUDIT_PASS','operator_source_sha256':r['operator_source_sha256'],'scope':'offline arithmetic audit; no timing/performance evidence','rows':rows}
(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');(ROOT/'PTX-AUDIT.txt').write_text(str(out)+'\n');print(json.dumps(result),flush=True)
