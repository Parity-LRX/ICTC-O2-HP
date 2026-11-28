import json,os
from pathlib import Path
ROOT=Path(os.environ['O2_TRAJECTORY_ROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
def read(p):return json.loads(p.read_text())
audit=Path((ROOT/'GPU-AUDIT.txt').read_text().strip());a=read(audit/'RESULT.json');d=read(audit/'DECOMPOSITION.json')
saved=read(Path((ROOT/'ANALYSIS-RESULT.txt').read_text().strip())/'RESULT.json')
rows=[]
for step,e in a['cotangent_controls'].items():
    rows.append({'step_one_based':int(step)+1,
        'mixed_phase_force_cotangent_R':e['force']['cotangents']['force']['relative_l2'],
        'training_phase_force_cotangent_R':e['total']['cotangents']['/forces']['relative_l2'],
        'force_own_failures':e['force']['own']['failed'],'force_shared_failures':e['force']['shared_native_cotangent']['failed'],
        'force_shared_max_R':e['force']['shared_native_cotangent']['maximum_relative_l2'],
        'total_own_failures':e['total']['own']['failed'],'total_shared_failures':e['total']['shared_native_cotangent']['failed'],
        'total_shared_max_R':e['total']['shared_native_cotangent']['maximum_relative_l2'],
        'energy_cotangent_max_abs':e['total']['cotangents']['/energy']['max_abs'],
        'stress_cotangent_max_abs':e['total']['cotangents']['/stress']['max_abs'],
        'native_traversal':e['native_traversal'],'candidate_traversal':e['candidate_traversal']})
updates=[]
for r in saved['trajectory']:
    u=r['updated_parameters'];updates.append({'step_one_based':r['step_zero_based']+1,**{k:v for k,v in u.items() if k!='different'},
        'max_ulp':max(v['ulp']['max_ulp'] for v in u['different'].values()),
        'gradient_different_tensors':r['raw_gradient']['different_tensors'],'gradient_max_R':r['raw_gradient']['summary']['maximum_relative_l2']})
result={'borderline_state_pass_role':[r for r in d if r['native_state_only']['passed']],
    'cotangents':rows,'updated_parameters':updates,
    'current_same_state_max_R':max(r['maximum_relative_l2'] for r in a['same_state_actual_RNG'].values()),
    'decomposition':a['decomposition']}
(ROOT/'DETAILS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
