import gc,json,os
from pathlib import Path
import torch
from continuity import signature
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
root=Path(__file__).resolve().parent
sample=root/'gate-captures/120866/network-L7-native/INITIAL.pt'
first=torch.load(sample,map_location='cpu',weights_only=False);expected=signature(first);del first;gc.collect()
second=torch.load(sample,map_location='cpu',weights_only=False);assert signature(second)==expected
parameter=next(v for v in second['model'].values() if isinstance(v,torch.Tensor) and v.is_floating_point() and v.numel())
parameter.reshape(-1)[0]=torch.nextafter(parameter.reshape(-1)[0],parameter.new_tensor(float('inf')))
assert signature(second)!=expected
record={'status':'PASS','roundtrip_state_signature':expected,'detects_one_ulp_parameter_change':True}
(root/'CONTINUITY-CHECK.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
