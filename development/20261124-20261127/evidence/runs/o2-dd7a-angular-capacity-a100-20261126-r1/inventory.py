import ast,hashlib,inspect,json,os,shutil,sys
from pathlib import Path
ROOT=Path(os.environ['O2_SWEEP_ROOT']);WS=ROOT.parents[1]
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
capsule=ROOT/'snapshots'/('preparation-'+os.environ['SLURM_JOB_ID']);capsule.mkdir(parents=True)
for p in ROOT.glob('*.py'):ast.parse(p.read_text());shutil.copy2(p,capsule/p.name)
shutil.copy2(ROOT/'CONTRACT.zh-CN.md',capsule/'CONTRACT.zh-CN.md')
shutil.copytree(WS/'ICTC-O2-HP/src',capsule/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
h=hashlib.sha256()
for p in sorted((capsule/'src').rglob('*')):
    if p.is_file() and p.suffix in ('.py','.cu','.cubin','.ptx','.json'):
        h.update(str(p.relative_to(capsule/'src')).encode()+b'\0'+bytes.fromhex(sha(p)))
seal=json.loads((WS/'runs/o2-bounded-bilinear-a100-20261126-r1/FINAL-SNAPSHOT.json').read_text())
assert h.hexdigest()==seal['package_sha256'] and (ROOT/'CANDIDATE-HEAD.txt').read_text().strip()==seal['candidate_commit']
old=WS/'runs/o2-uvw-triton-a100-20261125-r1';control=old/'snapshots/preflight-119174'
os.environ.update(O2_WHOLE_ROOT=str(old),O2_UVW_TILED='1',O2_UVW_SOURCE=str(capsule/'src'),NVIDIA_TF32_OVERRIDE='0',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',EQV3_DETERMINISTIC='1',EQX_USE_CUDA_GRAPH='0')
sys.path[:0]=[str(control),str(capsule/'src')]
import torch
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
from whole import V3
model=V3('native','checkpoint_ema','cpu')
checkpoint=model.net.calc.trainer.config
native_config_path=WS/'baselines/icml-o2-stage1-20261124-r1/payload/network/fixture-api/checkpoint/omat24-mptrj-salex_gradient.pt'
original=torch.load(native_config_path,map_location='cpu',weights_only=False)['config']
torch.save(original,ROOT/'BASE-CONFIG.pt')
inventory={'candidate_commit':seal['candidate_commit'],'package_sha256':h.hexdigest(),'snapshot':str(capsule),
 'model_class':type(model.model).__name__,'model_module':type(model.model).__module__,
 'model_signature':str(inspect.signature(type(model.model))),
 'config_model':original.get('model'),'config_model_attributes':original.get('model_attributes'),
 'trainer_config_keys':list(checkpoint),'checkpoint':str(native_config_path),'checkpoint_sha256':sha(native_config_path),
 'frame_fields':{mode:{k:({'shape':list(v.shape),'dtype':str(v.dtype)} if isinstance(v,torch.Tensor) else str(v)) for k,v in f.items()} for mode,f in model.frames.items()},
 'train_recipe':model.identity['training_recipe'],'loss':model.identity['loss'],
 'active_parameters':sum(p.requires_grad for p in model.model.parameters()),'job':os.environ['SLURM_JOB_ID']}
(ROOT/'INVENTORY.json').write_text(json.dumps(inventory,indent=2,default=str)+'\n')
(ROOT/'SOURCE.json').write_text(json.dumps({'path':str(capsule/'src'),'package_sha256':h.hexdigest(),'files':{str(p.relative_to(capsule)):sha(p) for p in capsule.rglob('*') if p.is_file()}},indent=2)+'\n')
print(json.dumps(inventory,default=str),flush=True)
