"""AMD summary of complete-real-array numerical selection statistics."""
import json,os
from pathlib import Path
from common import ROOT,save
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
source=Path((ROOT/'LOCAL-SCREEN.txt').read_text().strip());data=json.loads((source/'SELECTION.json').read_text())
groups={}
for shape in ('[256, 1024]','[192, 768]','[512, 128]','[384, 96]'):
    groups[shape]={}
    for role in ('forward','input_adjoint','weight_adjoint'):
        rows=[{'name':name,'config':v['config'],**v['by_shape'][shape][role]} for name,v in data.items() if v['status']=='SELECTION_MEASURED']
        groups[shape][role]=sorted(rows,key=lambda r:(r['relative_l2'],r['epsilon_max']))[:8]
save(ROOT/'LOCAL-RANKING.json',{'source':str(source),'status':'SELECTION_SUMMARY_NOT_ACCEPTANCE','groups':groups})
print(json.dumps(groups),flush=True)
