import importlib.metadata,json,os,platform,sys
from pathlib import Path
import numpy,torch,triton
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
root=Path(os.environ['O2_KROOT'])
r={'python':sys.version,'executable':sys.executable,'node':platform.node(),'platform':platform.platform(),'torch':torch.__version__,
    'torch_CUDA_runtime':torch.version.cuda,'triton':triton.__version__,'numpy':numpy.__version__,
    'installed_versions':{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},
    'scope':'environment metadata captured on AMD; actual GPU/device UUID/precision flags in each measurement CONDITION.json'}
(root/'ENVIRONMENT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='installed_versions'}),flush=True)
