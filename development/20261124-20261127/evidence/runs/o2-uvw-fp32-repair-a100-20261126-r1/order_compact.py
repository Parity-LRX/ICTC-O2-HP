import ast,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent;assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
rows=json.loads((ROOT/'ORDER-SYNTHESIS.json').read_text())
tree=ast.parse((ROOT/'snapshots/train-119347/candidate_kernels.py').read_text())
configs=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIGS' for t in n.targets))
lines=[];plan={}
for row in rows:
    best=row['best'][0];n,k=row['b_shape'][1],row['a_shape'][1]
    plan[f'{n},{k}']=configs[best[0]]
    lines.append({'role':row['role'],'shape_mnk':[row['a_shape'][0],n,k],
        'top':[[name,item['mismatch_rows']] for name,item in row['best']],
        'native_first_groups':[p['first64'] for p in row['native_patterns'][:3]]})
(ROOT/'ORDER-COMPACT.json').write_text(json.dumps(lines,indent=2)+'\n')
(ROOT/'SHAPE-PLAN.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(lines),flush=True)
