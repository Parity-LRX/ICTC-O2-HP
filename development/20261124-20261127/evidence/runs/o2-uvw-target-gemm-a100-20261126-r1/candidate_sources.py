"""Resolve only frozen sources with successful whole-trajectory AMD receipts."""
import importlib.util,json,sys
from pathlib import Path
from context import ROOT,WS,sha,package_sha
PREVIOUS=WS/'runs/o2-uvw-kgrad-rng-a100-20261126-r1'
def read(p):return json.loads(Path(p).read_text())
def qualified_sources():
    records={}
    for name,root,label,pointer in (('c05',PREVIOUS,'FULL','FULL-AUDIT.txt'),('mask',ROOT,'MASK','MASK-FULL-AUDIT.txt'),('candidate',ROOT,'FULL','FULL-AUDIT.txt')):
        pre=read(root/(label+'-PREFLIGHT.json'));gate=read(Path((root/pointer).read_text().strip())/'RESULT.json')
        path=Path(pre['snapshot'])/'ictc-source/ictc_o2_hp/uvw_triton.py';digest=sha(path)
        assert gate['status']=='FULL_QUALIFICATION_PASS' and digest==gate['operator_source_sha256']==pre['operator_source_sha256']
        if gate.get('package_sha256'):
            assert package_sha(Path(pre['snapshot'])/'ictc-source')==gate['package_sha256']
        for relative,expected in pre['files'].items():
            if relative.startswith('ictc-source/'):
                assert sha(Path(pre['snapshot'])/relative)==expected
        if name=='c05':assert digest=='355b26cae3721f6e16c6e2e8d1efd14aa1ff85ede566f2d1bcf76c02873bd944'
        records[name]={'path':path,'sha256':digest,'gate':str(Path((root/pointer).read_text().strip())/'RESULT.json'),'commit':pre['candidate_commit'],'package_sha256':gate.get('package_sha256')}
    return records
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
