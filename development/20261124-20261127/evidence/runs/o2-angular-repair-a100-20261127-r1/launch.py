import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(os.environ['O2_SWEEP_ROOT']);label,script=sys.argv[1:3];pre=json.loads((root/(label.upper()+'-PREFLIGHT.json')).read_text());control=Path(pre['control'])
os.environ['O2_REPAIR_SOURCE_RECORD']=json.dumps(pre['source'])
for name,digest in pre['files'].items():
    with (control/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
sys.exit(subprocess.call([sys.executable,'-u',str(control/script),*sys.argv[3:]]))
