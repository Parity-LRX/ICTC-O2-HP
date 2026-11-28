import subprocess,sys
from pathlib import Path
for name in ('report.py','documents.py','seal.py'):
    subprocess.run([sys.executable,'-u',str(Path(__file__).parent/name)],check=True)
