import subprocess,sys
from pathlib import Path
for name in ('measurement_binding.py','report.py','documents.py','seal.py'):
    subprocess.run([sys.executable,'-u',str(Path(__file__).parent/name)],check=True)
