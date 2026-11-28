import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(os.environ['O2_CAUSE_ROOT']);pre=json.loads((root/'AXIS-PREFLIGHT.json').read_text());source=Path(pre['snapshot'])
for name,digest in pre['files'].items():
    with (source/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
sys.exit(subprocess.call([sys.executable,'-u',str(source/'axis_probe.py')]))
