import subprocess,sys
from pathlib import Path
for name in ('inspect_bmm.py','infer_bmm.py'):
    subprocess.run([sys.executable,'-u',str(Path(__file__).parent/name)],check=True)
