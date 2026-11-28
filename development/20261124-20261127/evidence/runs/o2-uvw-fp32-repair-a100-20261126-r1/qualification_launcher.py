import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ['O2_REPAIR_ROOT']);receipt=json.loads((ROOT/'QUALIFICATION-PREFLIGHT.json').read_text());source=Path(receipt['snapshot'])
for name,digest in receipt['files'].items():assert hashlib.file_digest((source/name).open('rb'),'sha256').hexdigest()==digest
env=os.environ.copy();env['O2_REPAIR_SOURCE']=str(source/'ictc-source')
assert sys.argv[1] in ('real_regression.py','regression_audit.py','generic_regression.py','generic_audit.py','verify_capture.py','verify_audit.py','verify_replay.py','replay_audit.py')
if sys.argv[1] in ('verify_capture.py','verify_replay.py'):
    for pointer,status in (('REGRESSION-AUDIT.txt','ACTUAL_LOCAL_REGRESSION_PASS'),('GENERIC-AUDIT.txt','GENERIC_REGRESSION_PASS')):
        gate=json.loads((Path((ROOT/pointer).read_text().strip())/'RESULT.json').read_text())
        assert gate['status']==status,(pointer,gate['status'])
        assert gate['operator_source_sha256']==receipt['selected']['operator_source_sha256']
sys.exit(subprocess.call([sys.executable,'-u',str(source/'control'/sys.argv[1])],cwd=ROOT,env=env))
