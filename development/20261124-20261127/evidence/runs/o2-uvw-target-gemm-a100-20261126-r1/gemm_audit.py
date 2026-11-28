import json,os
from pathlib import Path
import torch
from context import ROOT,save,sha
from regression_audit import metric
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
source=Path((ROOT/'GEMM-RESULT.txt').read_text().strip());record=json.loads((source/'RESULT.json').read_text())
gate=json.loads((Path((ROOT/'FULL-AUDIT.txt').read_text().strip())/'RESULT.json').read_text())
assert gate['status']=='FULL_QUALIFICATION_PASS' and record['operator_source_sha256']==gate['operator_source_sha256']
if 'package_sha256' in record:
    assert record['package_sha256']==gate['package_sha256']
else:
    from context import package_sha
    prepath=ROOT/'SHARED32MICRO-PREFLIGHT.json';pre=json.loads(prepath.read_text())
    for name,digest in pre['files'].items():assert sha(Path(pre['snapshot'])/name)==digest
    assert package_sha(Path(pre['snapshot'])/'ictc-source')==gate['package_sha256']
    record['AMD_derived_package_binding']={'preflight':str(prepath),'sha256':sha(prepath),'package_sha256':gate['package_sha256'],'scope':'derived from frozen launch capsule, not an original GPU manifest field'}
for r in record['rows']:
    assert sha(Path(r['path']))==r['sha256'];v=torch.load(r['path'],map_location='cpu',weights_only=False)
    r['metrics']={k:metric(a,v['native']) for k,a in v.items() if k!='native'}
    assert all(m['passed'] for k,m in r['metrics'].items() if k!='a80')
    r['c05_over_candidate']=r['timings']['c05']['median_s']/r['timings']['candidate']['median_s']
    r['a80_over_candidate_diagnostic']=r['timings']['a80']['median_s']/r['timings']['candidate']['median_s']
    del v
out=ROOT/'results'/('gemm-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir();record['status']='QUALIFIED_GEMM_MEASUREMENT_AUDIT_PASS'
save(out/'RESULT.json',record);(ROOT/'GEMM-AUDIT.txt').write_text(str(out)+'\n')
print(json.dumps({'status':record['status'],'rows':[{'shape':r['shape'],'phase':r['phase'],'ms':{k:1000*v['median_s'] for k,v in r['timings'].items()},'ratio_c05':r['c05_over_candidate'],'ratio_a80_diagnostic':r['a80_over_candidate_diagnostic'],'candidate_ulp':r['metrics']['candidate']['ulp']} for r in record['rows']]}),flush=True)
