"""Verify the protected 8324 package and the newly qualified bilinear package."""
import json
from pathlib import Path
from context import ROOT,PREVIOUS,SOURCE,WS,sha,package_sha
def read(p):return json.loads(Path(p).read_text())
def qualified_sources():
    old=read(PREVIOUS/'FULL-PREFLIGHT.json');oldgate=read(Path((PREVIOUS/'FULL-AUDIT.txt').read_text().strip())/'RESULT.json')
    assert oldgate['status']=='FULL_QUALIFICATION_PASS' and oldgate['package_sha256']==package_sha(Path(old['snapshot'])/'ictc-source')
    pre=read(ROOT/'FULL-PREFLIGHT.json');gate=read(Path((ROOT/'FULL-AUDIT.txt').read_text().strip())/'RESULT.json');folder=Path(pre['snapshot'])/'ictc-source'
    assert gate['status']=='FULL_QUALIFICATION_PASS' and gate['package_sha256']==package_sha(folder)
    assert gate['operator_source_sha256']==sha(folder/'ictc_o2_hp/bilinear_triton.py')==pre['operator_source_sha256']
    for name,digest in old['files'].items():
        if name.startswith('ictc-source/'):assert sha(folder/Path(name).relative_to('ictc-source'))==digest
    reference=WS/'runs/o2-uvw-triton-a100-20261125-r1/snapshots/preflight-119174/bounded.py'
    return {'baseline':{'path':str(reference),'sha256':sha(reference),'channel_source_sha256':oldgate['operator_source_sha256'],'package_sha256':oldgate['package_sha256'],'gate':str(Path((PREVIOUS/'FULL-AUDIT.txt').read_text().strip())/'RESULT.json'),'commit':old['candidate_commit']},
        'candidate':{'path':str(folder/'ictc_o2_hp/bilinear_triton.py'),'sha256':gate['operator_source_sha256'],'package_sha256':gate['package_sha256'],'gate':str(Path((ROOT/'FULL-AUDIT.txt').read_text().strip())/'RESULT.json'),'commit':pre['candidate_commit']}}
