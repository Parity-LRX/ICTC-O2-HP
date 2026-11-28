import importlib.util,json,os,re,sys
from pathlib import Path
from context import ROOT,sha,save
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
label=os.environ.get('O2_PTX_LABEL','FULL')
pre=json.loads((ROOT/(label+'-PREFLIGHT.json')).read_text());path=Path(pre['snapshot'])/'ictc-source/ictc_o2_hp/bilinear_triton.py';assert sha(path)==pre['operator_source_sha256']
spec=importlib.util.spec_from_file_location('bilinear_audit',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
out=ROOT/'results'/('ptx-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir();rows=[]
cases=[]
for l,n,h in ((19,5252,32),(19,31512,32),(25,100,512),(25,600,512)):
    for column in (False,True):
        ks=(1,l*l,l) if column else (l*l,l,1)
        main_parts=4 if l==19 and column else 1
        main_four=l==19 or column
        tail_parts=(1 if n==5252 else 5) if l==19 and column else (3 if n==5252 else 4) if l==19 else 1 if column else (3 if n==100 else 2)
        for offset,count,parts,four in ((0,n//64*64,main_parts,main_four),(n//64*64,n%64,tail_parts,l==19)):
            cases.append((l,n,h,ks,offset,count,parts,four))
for l,n,h,ks,offset,count,parts,four in cases:
        c=dict(OA=l,LA=l,LB=l,H=h,N=n,K0=ks[0],K1=ks[1],K2=ks[2],A0=l*h*2,A1=h*2,A2=1,B0=l*h*2,B1=h*2,B2=1,BM=32,BN=32,OFFSET=offset,COUNT=count,PARTS=parts,FOUR=four)
        signature={**{k:'*fp32' for k in ('K','A','B','Y')},**{k:'constexpr' for k in c}}
        v=triton.compile(ASTSource(m._bilinear,signature=signature,constexprs=c),target=GPUTarget('cuda',80,32),options={'num_warps':4})
        ptx=v.asm['ptx'];assert 'mul.rn.f32' in ptx and 'fma.rn.f32' in ptx
        if parts>1 or four:assert 'add.rn.f32' in ptx
        assert not re.search(r'\b(?:mma|wmma|wgmma)\.|\.(?:tf32|bf16|f16|f64)\b',ptx)
        p=out/f'L{l}-{len(rows)}.ptx';p.write_text(ptx);rows.append({'shape':[l,n,h],'strides':ks,'constants':c,'ptx_sha256':sha(p),'shared':v.metadata.shared,'warps':4})
r={'status':'FP32_SM80_PTX_AUDIT_PASS','operator_source_sha256':pre['operator_source_sha256'],'rows':rows,'scope':'offline FP32 arithmetic audit, not a performance measurement'}
save(out/'RESULT.json',r);(ROOT/('PTX-AUDIT.txt' if label=='FULL' else label+'-PTX-AUDIT.txt')).write_text(str(out)+'\n');print(json.dumps(r),flush=True)
