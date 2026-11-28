import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ['O2_CAUSE_ROOT']);receipt=json.loads((ROOT/'PREFLIGHT-TOTAL.json').read_text())
sealed=Path(receipt['snapshot'])
for name,digest in receipt['files'].items():assert hashlib.file_digest((sealed/name).open('rb'),'sha256').hexdigest()==digest
sys.exit(subprocess.call([sys.executable,'-u',str(sealed/'total_probe.py')],cwd=ROOT,env=os.environ.copy()))
