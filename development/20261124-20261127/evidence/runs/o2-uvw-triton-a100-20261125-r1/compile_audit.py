"""Offline sm80 compilation and PTX precision audit on a scheduled CPU node."""
import hashlib,json,os,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'ICTC-O2-HP/src'))
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from ictc_o2_hp.uvw_triton import _uvw_mm
def main():
    assert os.environ.get('SLURM_JOB_ID')
    out=ROOT/'results'/('compile-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    rows=[]
    for name,m,n,k,am,ak,bk,bn in (
            ('forward',63024,256,1024,1024,1,1,1024),
            ('input-adjoint',63024,1024,256,256,1,1024,1),
            ('weight-adjoint',1024,256,274,1,1024,256,1),
            ('uvw-projection',14,12,5,5,1,12,1)):
        constants=dict(M=m,N=n,K=k,SAM=am,SAK=ak,SBK=bk,SBN=bn,BM=32,BN=32,BK=32)
        signature={'A':'*fp32','B':'*fp32','C':'*fp32',**{key:'constexpr' for key in constants}}
        source=ASTSource(_uvw_mm,signature=signature,constexprs=constants)
        kernel=triton.compile(source,target=GPUTarget('cuda',80,32),options={'num_warps':4})
        ptx=kernel.asm['ptx'];(out/(name+'.ptx')).write_text(ptx)
        forbidden=re.findall(r'\b(?:mma|wmma)\.|\.(?:tf32|bf16|f16)\b',ptx)
        assert not forbidden,(name,forbidden[:5])
        row={'variant':name,'triton':triton.__version__,'target':'sm80','forbidden_reduced_precision_instructions':forbidden,
             'contains_fp32_fma':'fma.rn.f32' in ptx,'PTX_sha256':hashlib.sha256(ptx.encode()).hexdigest(),
             'registers':'NOT_MEASURED_OFFLINE','shared_bytes':kernel.metadata.shared}
        rows.append(row);print(json.dumps(row),flush=True)
    receipt={'job':os.environ['SLURM_JOB_ID'],'status':'OFFLINE_SM80_COMPILED_AND_PTX_AUDITED',
        'scope':'compilation only; no GPU execution/performance/numerical claim','rows':rows}
    (ROOT/'COMPILE-AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
