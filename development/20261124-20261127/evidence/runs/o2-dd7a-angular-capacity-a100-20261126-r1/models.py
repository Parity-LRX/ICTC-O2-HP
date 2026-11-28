"""Prepare independently initialized angular architectures; keep dd7a backend sealed."""
import ast,gc,hashlib,importlib.util,json,os,shutil,sys
from pathlib import Path
ROOT=Path(os.environ['O2_SWEEP_ROOT']);WS=ROOT.parents[1]
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
source=json.loads((ROOT/'SOURCE.json').read_text());SRC=Path(source['path'])
OLD=WS/'runs/o2-uvw-triton-a100-20261125-r1';CONTROL=OLD/'snapshots/preflight-119174'
os.environ.update(O2_WHOLE_ROOT=str(OLD),O2_UVW_TILED='1',O2_UVW_SOURCE=str(SRC),NVIDIA_TF32_OVERRIDE='0',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',EQV3_DETERMINISTIC='1',EQX_USE_CUDA_GRAPH='0')
sys.path[:0]=[str(CONTROL),str(SRC)]
import torch
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
from whole import V3
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=ROOT/'models';out.mkdir(exist_ok=True);capsule=ROOT/'snapshots'/('models-'+os.environ['SLURM_JOB_ID']);capsule.mkdir(parents=True)
for p in ROOT.glob('*.py'):ast.parse(p.read_text());shutil.copy2(p,capsule/p.name)
shutil.copy2(ROOT/'CONTRACT.zh-CN.md',capsule/'CONTRACT.zh-CN.md')
original=WS/'architectures/equiformerv3/source/canonical-cpu-control-c14109a12653/canonical.py'
text=original.read_text()
replacements={
 'CoefficientMappingModule(4, 4)':'CoefficientMappingModule(L, L)',
 'CoefficientMappingModule(4,':'CoefficientMappingModule(L,',
 'full.coefficient_idx(4,':'full.coefficient_idx(L,',
 'len(ids), 25)':'len(ids), (L+1)**2)',
 'SO3Rotation(4, 4,':'SO3Rotation(L, L,',
 'angular_layout(4, True)':'angular_layout(L, True)',
 'torch.ones(25)':'torch.ones((L+1)**2)',
 'native_action(mirror, 4, True)':'native_action(mirror, L, True)',
 'native_action(rotation, 4, True)':'native_action(rotation, L, True)',
 'range(5)':'range(L+1)', 'range(1, 5)':'range(1, L+1)',
 'range(m, 5)':'range(m, L+1)', 'range(self.m,5)':'range(self.m,L+1)',
 'offset = 5':'offset = L+1', 'count = 5-m':'count = L+1-m',
 'count = 5-self.m':'count = L+1-self.m'}
changes={}
for old,new in replacements.items():
    assert old in text,old
    changes[old]={'replacement':new,'occurrences':text.count(old)};text=text.replace(old,new)
save=lambda p,v:Path(p).write_text(json.dumps(v,indent=2,default=str)+'\n')
save(out/'REFERENCE-GENERALIZATION.json',{'original':str(original),'original_sha256':sha(original),'replacements':changes,'purpose':'only construct independent L=1..8 references; backend dd7a unchanged'})
template=V3('native','checkpoint_ema','cpu');cls=type(template.model)
base=torch.load(ROOT/'BASE-CONFIG.pt',map_location='cpu',weights_only=False)['model'];rows=[]
for l in range(1,9):
    print('PREPARE_L',l,flush=True)
    path=out/f'canonical-L{l}.py';path.write_text(f'L={l}\n'+text)
    spec=importlib.util.spec_from_file_location(f'sweep_canonical_{l}',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    kwargs={k:v for k,v in base.items() if k!='name'}
    kwargs.update(lmax=l,mmax=l,attn_grid_resolution_list=[20,20],ffn_grid_resolution_list=[20,20],gradient_checkpointing_block_list=[1]*7)
    torch.manual_seed(20261126+l);model=cls(**kwargs)
    basis,calibration,actions=module.calibrate()
    ks,_,diag_s=module.natural_kernel(model.blocks[0].ga.act.so3_grid)
    kf,_,diag_f=module.natural_kernel(model.blocks[0].ffn.so3_grid)
    kernels={f'{kind}{i+1:02}':k for i in range(7) for kind,k in (('S',ks),('F',kf))}
    migrations,_,_=module.install(model,basis,kernels)
    assert len(migrations)==14*l
    target=out/f'L{l}.pt'
    torch.save({'state_dict':model.state_dict(),'kwargs':kwargs,'basis':basis,'kernels':kernels,
                'calibration':calibration,'normalizations':{'S':diag_s,'F':diag_f},'seed':20261126+l,
                'reference_source_sha256':sha(path)},target)
    row={'lmax':l,'mmax':l,'path':str(target),'sha256':sha(target),'reference_source':str(path),'reference_source_sha256':sha(path),
         'parameters':sum(p.numel() for p in model.parameters()),'active_parameter_tensors':sum(p.requires_grad for p in model.parameters()),
         'positive_m_sites':len(migrations),'S_shape':list(ks.shape),'F_shape':list(kf.shape),
         'calibration_max_rotation':max(r['rotation_max_abs'] for r in calibration),'calibration_max_reflection':max(r['reflection_max_abs'] for r in calibration),
         'kwargs':kwargs,'seed':20261126+l,'initialization':'independent fixed-seed structural model; not original checkpoint accuracy'}
    rows.append(row);save(out/'PROGRESS.json',rows);print(json.dumps(row),flush=True)
    del model,basis,actions,ks,kf,kernels,module;gc.collect()
save(ROOT/'MODELS.json',{'status':'ANGULAR_MODELS_PREPARED','job':os.environ['SLURM_JOB_ID'],'rows':rows,'source_package_sha256':source['package_sha256'],
 'anchor':{'lmax':4,'mmax':2,'factory':'original V3 checkpoint_ema unchanged','gate':'original dd7a full audit 120335'}})
