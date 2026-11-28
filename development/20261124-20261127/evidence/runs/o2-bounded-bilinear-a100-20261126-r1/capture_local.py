import subprocess,sys
from pathlib import Path
for name in ('capture.py','local.py'):
    subprocess.run([sys.executable,'-u',str(Path(__file__).parent/name)],check=True)
