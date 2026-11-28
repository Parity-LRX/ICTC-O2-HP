"""Offline CPU-AMD NVRTC build; no CUDA driver or GPU execution."""
import ctypes as C,hashlib,json,os,re,subprocess
from pathlib import Path
import triton
ROOT=Path(os.environ['O2_KROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';WS=ROOT.parents[1]
src=WS/'ICTC-O2-HP/src/ictc_o2_hp/kernels/weight_fma8.cu';code=src.read_bytes()
lib=Path(triton.__file__).parents[1]/'nvidia/cuda_nvrtc/lib/libnvrtc.so.12';nv=C.CDLL(str(lib))
nv.nvrtcCreateProgram.argtypes=[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p]
nv.nvrtcCompileProgram.argtypes=[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)]
for name in ('nvrtcGetProgramLogSize','nvrtcGetPTXSize'):getattr(nv,name).argtypes=[C.c_void_p,C.POINTER(C.c_size_t)]
for name in ('nvrtcGetProgramLog','nvrtcGetPTX'):getattr(nv,name).argtypes=[C.c_void_p,C.c_void_p]
nv.nvrtcDestroyProgram.argtypes=[C.POINTER(C.c_void_p)]
out=ROOT/'results'/('cuda-build-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir();rows=[]
assembler=Path(triton.__file__).parent/'backends/nvidia/bin/ptxas'
major=C.c_int();minor=C.c_int();assert nv.nvrtcVersion(C.byref(major),C.byref(minor))==0
for tile,unroll in ((32,8),(64,8),(64,4)):
    stem=f'weight-fma8-sm80-n{tile}-u{unroll}';program=C.c_void_p()
    assert nv.nvrtcCreateProgram(C.byref(program),code,b'weight_fma8.cu',0,None,None)==0
    opts=[b'--gpu-architecture=compute_80',b'--std=c++14',b'--ftz=false',b'--fmad=false',f'-DTILE_N={tile}'.encode(),f'-DK_UNROLL={unroll}'.encode()]
    status=nv.nvrtcCompileProgram(program,len(opts),(C.c_char_p*len(opts))(*opts))
    length=C.c_size_t();assert nv.nvrtcGetProgramLogSize(program,C.byref(length))==0
    log=C.create_string_buffer(length.value);assert nv.nvrtcGetProgramLog(program,log)==0
    (out/(stem+'-nvrtc.txt')).write_bytes(log.value);assert status==0,log.value.decode()
    assert nv.nvrtcGetPTXSize(program,C.byref(length))==0
    ptx=C.create_string_buffer(length.value);assert nv.nvrtcGetPTX(program,ptx)==0;nv.nvrtcDestroyProgram(C.byref(program))
    text=ptx.value.decode();assert 'fma.rn.f32' in text and 'add.rn.f32' in text
    assert not re.search(r'\b(?:mma|wmma|wgmma)\.|\.(?:tf32|bf16|f16|f64)\b',text)
    p=out/(stem+'.ptx');p.write_text(text)
    result=subprocess.run([str(assembler),'-v','--gpu-name=sm_80',str(p),'-o',str(out/(stem+'.cubin'))],capture_output=True,text=True,check=True)
    (out/(stem+'-ptxas.txt')).write_text(result.stdout+result.stderr)
    rows.append({'stem':stem,'tile':[32,tile],'warps':4,'unroll':unroll,'options':[v.decode() for v in opts],
        'source_sha256':hashlib.sha256(code).hexdigest(),'ptx_sha256':hashlib.sha256(ptx.value).hexdigest(),
        'cubin_sha256':hashlib.file_digest((out/(stem+'.cubin')).open('rb'),'sha256').hexdigest(),'ptxas':result.stderr})
r={'status':'OFFLINE_CUDA_FMA8_BUILD_PASS','NVRTC':[major.value,minor.value],'NVRTC_library':str(lib),'rows':rows,'scope':'offline precision/resource audit; no GPU performance claim'}
(out/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n');(ROOT/'CUDA-BUILD.txt').write_text(str(out)+'\n');print(json.dumps(r),flush=True)
