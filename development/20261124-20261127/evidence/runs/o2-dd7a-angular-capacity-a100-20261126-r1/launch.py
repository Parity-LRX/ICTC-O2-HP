import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(os.environ['O2_SWEEP_ROOT']);label,script=sys.argv[1:3]
pre=json.loads((root/(label.upper()+'-PREFLIGHT.json')).read_text());source=Path(pre['snapshot'])
for name,digest in pre['files'].items():
    with (source/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
os.environ['O2_CONTROL_SNAPSHOT']=str(source)
sys.exit(subprocess.call([sys.executable,'-u',str(source/script),*sys.argv[3:]],cwd=root,env=os.environ.copy()))
