"""Compile the accepted source for sm80 on AMD and audit arithmetic instructions."""
import hashlib,importlib.util,json,os,re,sys
from pathlib import Path
ROOT=Path(os.environ['O2_REPAIR_ROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
qual=json.loads((ROOT/'QUALIFICATION-PREFLIGHT.json').read_text());path=Path(qual['snapshot'])/'ictc-source/ictc_o2_hp/uvw_triton.py'
assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==qual['selected']['operator_source_sha256']
spec=importlib.util.spec_from_file_location('accepted_uvw_precision_audit',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
out=ROOT/'results'/('ptx-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir();rows=[]
for name,m,n,k,am,ak,bk,bn,parts in (
        ('forward-p2',10504,256,1024,1024,1,1,1024,2),
        ('forward-p4',10504,192,768,768,1,1,768,4),
        ('input-p6',10504,128,512,512,1,128,1,6),
        ('input-p3',10504,96,384,384,1,96,1,3),
        ('input-p1',10504,1024,256,256,1,1024,1,1),
        ('weight-p1',1024,256,10504,1,1024,256,1,1)):
    constants=dict(M=m,N=n,K=k,SAM=am,SAK=ak,SBK=bk,SBN=bn,BM=32,BN=32,BK=16 if parts>1 else 32,PARTS=parts)
    signature={'A':'*fp32','B':'*fp32','C':'*fp32',**{key:'constexpr' for key in constants}}
    compiled=triton.compile(ASTSource(module._uvw_mm,signature=signature,constexprs=constants),target=GPUTarget('cuda',80,32),options={'num_warps':4})
    ptx=compiled.asm['ptx'];forbidden=re.findall(r'\b(?:mma|wmma|wgmma)\.|\.(?:tf32|bf16|f16)\b',ptx)
    assert not forbidden and 'fma.rn.f32' in ptx
    if parts>1:assert 'add.rn.f32' in ptx
    (out/(name+'.ptx')).write_text(ptx)
    rows.append({'case':name,'constants':constants,'PTX_sha256':hashlib.sha256(ptx.encode()).hexdigest(),
        'forbidden_reduced_precision_instructions':forbidden,'contains_fp32_fma':True,'explicit_fp32_add_required':parts>1,'shared_bytes':compiled.metadata.shared})
result={'status':'ACCEPTED_SOURCE_SM80_PTX_AUDIT_PASS','operator_source_sha256':qual['selected']['operator_source_sha256'],
    'scope':'offline source compilation; arithmetic audit only, not GPU execution/performance evidence','rows':rows,'triton':triton.__version__}
(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');(ROOT/'PTX-AUDIT.txt').write_text(str(out)+'\n');print(json.dumps(result),flush=True)
