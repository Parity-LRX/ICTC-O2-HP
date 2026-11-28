"""Bind the actual frozen measurement launches to the accepted CUDA package."""
import json,os
from pathlib import Path
from context import ROOT,sha,package_sha,save
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
def read(p):return json.loads(Path(p).read_text())
gate=read(Path((ROOT/'FULL-AUDIT.txt').read_text().strip())/'RESULT.json');expected=gate['package_sha256'];rows=[]
for label in ('SHARED32MICRO','PERFORMANCE'):
    p=ROOT/(label+'-PREFLIGHT.json');r=read(p);folder=Path(r['snapshot'])
    for name,digest in r['files'].items():assert sha(folder/name)==digest,(label,name)
    source=Path(r.get('qualified_source',str(folder/'ictc-source')))
    assert package_sha(source)==expected
    if label=='PERFORMANCE':assert r['qualified_backends']['candidate']['package_sha256']==expected
    rows.append({'measurement':label,'preflight':str(p),'preflight_sha256':sha(p),'snapshot':str(folder),'executed_source':str(source),'package_sha256':expected,'launch_control_sha256':r['files']['control/launch.py']})
save(ROOT/'MEASUREMENT-SOURCE-BINDINGS.json',{'status':'FROZEN_MEASUREMENT_PACKAGE_BINDINGS_PASS','AMD_gate':str(Path((ROOT/'FULL-AUDIT.txt').read_text().strip())/'RESULT.json'),'package_sha256':expected,'rows':rows,'scope':'source/binary binding from frozen launch capsules; does not add absent fields to historical GPU records'})
print('FROZEN_MEASUREMENT_PACKAGE_BINDINGS_PASS',flush=True)
