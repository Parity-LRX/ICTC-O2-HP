import json,os,subprocess,sys
from pathlib import Path
root=Path(os.environ['O2_KROOT'])
pre=json.loads((root/'PERFORMANCE-PREFLIGHT.json').read_text())
control=Path(pre['snapshot'])/'control'
for name in ('performance_audit.py','trace_shapes.py','figure.py','environment.py'):
    subprocess.run([sys.executable,'-u',str(control/name)],check=True)
