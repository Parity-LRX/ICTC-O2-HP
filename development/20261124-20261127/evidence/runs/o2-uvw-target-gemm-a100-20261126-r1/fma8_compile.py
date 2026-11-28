import hashlib,json,os,re,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from fma8_kernels import _weight_fma8
out=ROOT/'results'/('fma8-compile-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir();rows=[]
assembler=Path(triton.__file__).parent/'backends/nvidia/bin/ptxas'
for bm,bn,warps in ((16,32,4),(16,64,4),(16,64,8),(16,128,8),(32,32,8),(32,64,8)):
    name=f'{bm}x{bn}-w{warps}';c=dict(M=256,N=1024,K=10504,SAM=1,SAK=256,SBK=1024,SBN=1,BM=bm,BN=bn,PARTS=4)
    signature={'A':'*fp32','B':'*fp32','P':'*fp32',**{k:'constexpr' for k in c}}
    compiled=triton.compile(ASTSource(_weight_fma8,signature=signature,constexprs=c),target=GPUTarget('cuda',80,32),options={'num_warps':warps})
    ptx=compiled.asm['ptx'];assert 'fma.rn.f32' in ptx and 'add.rn.f32' in ptx
    assert not re.search(r'\b(?:mma|wmma|wgmma)\.|\.(?:tf32|bf16|f16)\b',ptx)
    p=out/(name+'.ptx');p.write_text(ptx)
    info=subprocess.run([str(assembler),'-v','--gpu-name=sm_80',str(p),'-o',str(out/(name+'.cubin'))],capture_output=True,text=True,check=True)
    (out/(name+'-ptxas.txt')).write_text(info.stdout+info.stderr)
    rows.append({'tile':[bm,bn],'warps':warps,'shared_bytes':compiled.metadata.shared,'fma_instructions_in_PTX':ptx.count('fma.rn.f32'),
        'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),'ptxas':info.stderr})
r={'status':'OFFLINE_FP32_FMA8_COMPILE_PASS','no_GPU_execution_or_performance_claim':True,'rows':rows}
(out/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n');(ROOT/'FMA8-COMPILE.txt').write_text(str(out)+'\n');print(json.dumps(r),flush=True)
