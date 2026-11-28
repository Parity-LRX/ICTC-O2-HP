import subprocess,sys
from pathlib import Path
control=Path(__file__).parent
subprocess.run([sys.executable,'-u',str(control/'local_regress.py'),'--tail-only'],check=True)
subprocess.run([sys.executable,'-u',str(control/'gate_driver.py'),'--quick','--orders','7'],check=True)
