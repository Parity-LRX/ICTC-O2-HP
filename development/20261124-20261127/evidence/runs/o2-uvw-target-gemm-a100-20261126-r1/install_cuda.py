import hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
build=Path((ROOT/'CUDA-BUILD.txt').read_text().strip());record=json.loads((build/'RESULT.json').read_text())
assert record['status']=='OFFLINE_CUDA_FMA8_BUILD_PASS'
row=next(r for r in record['rows'] if r['tile']==[32,64] and r['unroll']==8)
folder=ROOT.parents[1]/'ICTC-O2-HP/src/ictc_o2_hp/kernels'
assert hashlib.file_digest((folder/'weight_fma8.cu').open('rb'),'sha256').hexdigest()==row['source_sha256']
assert hashlib.file_digest((build/(row['stem']+'.cubin')).open('rb'),'sha256').hexdigest()==row['cubin_sha256']
for ext in ('.cubin','.ptx'):shutil.copy2(build/(row['stem']+ext),folder/('weight_fma8_sm80'+ext))
meta={**row,'NVRTC':record['NVRTC'],'build_receipt':str(build/'RESULT.json'),'precision':'FP32 RN FMA/add, FTZ off; no TF32/AMP/half/FP64/Tensor Core',
    'layout':'A[256,10504] strides[1,256], B[10504,1024] contiguous and 16-byte aligned; output partials[4,256,1024]',
    'domain':'SM80; unsupported domains use the existing Triton implementation; no runtime compilation'}
(folder/'weight_fma8_sm80.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
