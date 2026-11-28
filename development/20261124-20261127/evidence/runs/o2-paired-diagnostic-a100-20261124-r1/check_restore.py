import ast,copy,json,os
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
assert os.environ.get('SLURM_JOB_ID')
for name in ('diagnose.py','controller.py','analyze.py'):ast.parse((ROOT/name).read_text())
p=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.AdamW([p])
for _ in range(3):p.grad=torch.ones_like(p);optimizer.step()
state=copy.deepcopy(optimizer.state_dict());key=next(iter(state['state']))
optimizer.load_state_dict(state);p.grad=torch.ones_like(p);optimizer.step()
alias=state['state'][key]['step'].item()
assert alias==4,'Expected PyTorch CPU state alias receipt'
safe=copy.deepcopy(optimizer.state_dict());before=safe['state'][key]['step'].item()
optimizer.load_state_dict(copy.deepcopy(safe));p.grad=torch.ones_like(p);optimizer.step()
assert safe['state'][key]['step'].item()==before
result={'status':'RESTORE_CONTROL_CHECK_PASS','job':os.environ['SLURM_JOB_ID'],'naive_template_step_after_update':alias,
        'cloned_optimizer_template_unchanged':True,'scope':'tiny CPU AdamW alias proof; production full-state hashes recorded in GPU diagnostics'}
(ROOT/'prepared/RESTORE-CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
