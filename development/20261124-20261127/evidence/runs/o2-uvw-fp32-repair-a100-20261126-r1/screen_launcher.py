import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ['O2_REPAIR_ROOT']);receipt=json.loads((ROOT/'SCREEN-PREFLIGHT.json').read_text());source=Path(receipt['snapshot'])
for name,digest in receipt['files'].items():assert hashlib.file_digest((source/name).open('rb'),'sha256').hexdigest()==digest
sys.exit(subprocess.call([sys.executable,'-u',str(source/'local_screen.py')],cwd=ROOT,env=os.environ.copy()))
