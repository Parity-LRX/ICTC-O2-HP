"""Inspect the saved element responsible for the largest local ULP distance."""
import json,os
from pathlib import Path
import torch
ROOT=Path(os.environ['O2_REPAIR_ROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(2)
audit=json.loads((Path((ROOT/'REGRESSION-AUDIT.txt').read_text().strip())/'RESULT.json').read_text())
index,role=max(((i,k) for i,row in enumerate(audit['results']) for k in row['roles']),key=lambda v:audit['results'][v[0]]['roles'][v[1]]['ulp']['max_ulp'])
source=Path((ROOT/'REGRESSION-RESULT.txt').read_text().strip());record=json.loads((source/'REGRESSION.json').read_text())['records'][index]
v=torch.load(record['path'],map_location='cpu',weights_only=False);a=v['candidate'][role];r=v['reference'][role]
def ordered(x):
    value=x.contiguous().view(torch.int32).to(torch.int64);return torch.where(value<0,-(value&0x7fffffff),value)
distance=(ordered(a)-ordered(r)).abs();flat=int(distance.flatten().argmax());assert int(distance.flatten()[flat])==audit['results'][index]['roles'][role]['ulp']['max_ulp']
result={'backend_capture':record['backend'],'site':record['site'],'phase':record['phase'],'role':role,'flat_index':flat,
    'candidate_fp32':float(a.flatten()[flat]),'reference_fp32':float(r.flatten()[flat]),
    'absolute_difference_fp32':float((a.flatten()[flat]-r.flatten()[flat]).abs()),'ulp_distance':int(distance.flatten()[flat]),
    'whole_role':audit['results'][index]['roles'][role],'raw_source':record['path'],
    'interpretation':'ULP distance across zero is not a relative error bound; original full-array epsilon_max gate remains unchanged'}
(ROOT/'ULP-EXAMPLE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
