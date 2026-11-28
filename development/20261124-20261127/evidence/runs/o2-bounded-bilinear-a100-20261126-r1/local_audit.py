import gc,json,os,platform
from pathlib import Path
import torch
from context import ROOT,save,sha
from metrics import full
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
source=Path((ROOT/'LOCAL-RESULT.txt').read_text().strip());manifest=json.loads((source/'MANIFEST.json').read_text())
out=ROOT/'results'/('local-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir();rows=[]
for row in manifest['records']:
    assert sha(row['path'])==row['sha256'];v=torch.load(row['path'],map_location='cpu',weights_only=False)
    m=full(v['candidate'],v['reference']);repeat=full(v['native_repeat'],v['reference'])
    assert all(x['passed'] for x in repeat.values())
    eps={k:float((v['candidate'][k]-r).abs().max()/max(float(r.abs().max()),float(v['candidate'][k].abs().max()),1e-7)) for k,r in v['reference'].items()}
    rows.append({'identity':row,'roles':m,'epsilon_max':eps,'passed':all(x['passed'] for x in m.values()) and all(e<=1e-5 for e in eps.values())})
    print(row['name'],rows[-1]['passed'],max(eps.values()),flush=True);del v;gc.collect()
save(out/'CASES.json',rows)
r={'status':'LOCAL_BILINEAR_PASS' if all(v['passed'] for v in rows) else 'LOCAL_BILINEAR_FAIL','cases':len(rows),'roles':len(rows)*6,'failures':[v['identity']['name'] for v in rows if not v['passed']],'epsilon_max':max(e for v in rows for e in v['epsilon_max'].values()),'max_ulp':max(x.get('ulp',{}).get('max_ulp',0) for v in rows for x in v['roles'].values()),'operator_source_sha256':manifest['operator_source_sha256'],'package_sha256':manifest['package_sha256'],'node':platform.node()}
r['groups']={}
for group,prefix in (('actual','actual-'),('generic','generic-')):
    selected=[v for v in rows if v['identity']['name'].startswith(prefix)]
    r['groups'][group]={'cases':len(selected),'roles':len(selected)*6,'epsilon_max':max(e for v in selected for e in v['epsilon_max'].values()),'max_R':max((x.get('relative_l2',0) or 0) for v in selected for x in v['roles'].values()),'max_ulp':max(x.get('ulp',{}).get('max_ulp',0) for v in selected for x in v['roles'].values())}
save(out/'RESULT.json',r);(ROOT/'LOCAL-AUDIT.txt').write_text(str(out)+'\n');print(json.dumps(r),flush=True)
