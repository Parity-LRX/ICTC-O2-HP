"""Execute the CPU-sealed diagnostic, not a mutable working source."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
receipt=json.loads((root/'PREFLIGHT.json').read_text());snapshot=Path(receipt['snapshot'])
for name,digest in receipt['files'].items():
    with (snapshot/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest,name
raise SystemExit(subprocess.run([sys.executable,'-u',str(snapshot/'probe.py')],cwd=root,env=os.environ.copy()).returncode)
