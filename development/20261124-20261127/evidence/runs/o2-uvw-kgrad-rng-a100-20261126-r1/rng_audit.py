"""Supplement full-array gates with saved RNG bytes and all EF/VJP frame phases."""
import json,os
from pathlib import Path
import torch
from context import ROOT,rng_id,save,sha
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
source=Path((ROOT/'QUALIFICATION-RESULT.txt').read_text().strip());manifest=json.loads((source/'MANIFEST.json').read_text())
conditions=json.loads((source/'CONDITIONS.json').read_text());frames=json.loads((source/'FRAMES.json').read_text());files=0;transitions=[]
for name,digest in manifest['files'].items():
    if name.endswith('-RNG.pt'):assert sha(source/name)==digest;files+=1
for role in ('ef','vjp'):
    reference=torch.load(source/f'native-{role}-RNG.pt',map_location='cpu',weights_only=False)
    expected=[(f['rng'],f['frame_sha256']) for f in frames if f['phase']==f'native-{role}']
    assert expected
    for backend in manifest['backends']:
        actual=torch.load(source/f'{backend}-{role}-RNG.pt',map_location='cpu',weights_only=False)
        assert rng_id(actual)==rng_id(reference)
        assert [(f['rng'],f['frame_sha256']) for f in frames if f['phase']==f'{backend}-{role}']==expected
for mode in manifest['modes']:
    for backend in manifest['backends']:
        for step in range(3):
            key=f'{mode}-{backend}-step{step}';r=torch.load(source/(key+'-RNG.pt'),map_location='cpu',weights_only=False)
            assert {k:rng_id(v) for k,v in r.items()}==conditions[key]['rng']
            if step:
                prior=conditions[f'{mode}-{backend}-step{step-1}'];current=conditions[key]
                assert current['before']==prior['after']
                assert current['rng']['train' if mode=='train-only' else 'mixed']==prior['rng_at_next_step']
                transitions.append({'mode':mode,'backend':backend,'step':step+1,'model_optimizer_EMA_scheduler_continuous':True,'RNG_continuous':True})
out=ROOT/'results'/('rng-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir()
r={'status':'RNG_PHASE_AND_CONTINUITY_PASS','operator_source_sha256':manifest['operator_source_sha256'],'saved_RNG_files_verified':files,
    'EF_and_joint_VJP_actual_RNG_and_frames_equal':True,'transitions':transitions,'conditions_sha256':sha(source/'CONDITIONS.json'),'frames_sha256':sha(source/'FRAMES.json')}
save(out/'RESULT.json',r);(ROOT/'RNG-AUDIT.txt').write_text(str(out)+'\n');print(json.dumps(r),flush=True)
