import os,subprocess,sys
from pathlib import Path
for script in ('real_regression.py','generic_regression.py','qualify.py'):
    subprocess.run([sys.executable,'-u',str(Path(__file__).parent/script)],check=True)
