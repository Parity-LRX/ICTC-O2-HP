import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);label,script=sys.argv[1:3];r=json.loads((ROOT/(label.upper()+'-PREFLIGHT.json')).read_text());source=Path(r['snapshot'])
for name,digest in r['files'].items():assert hashlib.file_digest((source/name).open('rb'),'sha256').hexdigest()==digest
env=os.environ.copy();env['O2_REPAIR_SOURCE']=r.get('qualified_source',str(source/'ictc-source'))
env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
env.update(NVIDIA_TF32_OVERRIDE='0',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',EQV3_DETERMINISTIC='1',EQX_USE_CUDA_GRAPH='0')
sys.exit(subprocess.call([sys.executable,'-u',str(source/'control'/script),*sys.argv[3:]],cwd=ROOT,env=env))
