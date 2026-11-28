import subprocess,sys
from pathlib import Path
for name in ('rng_audit.py','numerical_summary.py','ptx_audit.py'):
    subprocess.run([sys.executable,'-u',str(Path(__file__).parent/name)],check=True)
