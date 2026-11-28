import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ['O2_TRAJECTORY_ROOT']);r=json.loads((ROOT/'PREFLIGHT.json').read_text());source=Path(r['snapshot'])
for name,digest in r['files'].items():assert hashlib.file_digest((source/name).open('rb'),'sha256').hexdigest()==digest
for name,digest in r['dependencies'].items():assert hashlib.file_digest(Path(name).open('rb'),'sha256').hexdigest()==digest
sys.exit(subprocess.call([sys.executable,'-u',str(source/'gpu.py')],cwd=ROOT))
