import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);pre=json.loads((ROOT/'PERFORMANCE-PREFLIGHT.json').read_text());control=Path(pre['snapshot'])/'control'
for name in ('performance_audit.py','trace_shapes.py','bilinear_trace_audit.py','figure.py','environment.py'):
    subprocess.run([sys.executable,'-u',str(control/name)],check=True)
