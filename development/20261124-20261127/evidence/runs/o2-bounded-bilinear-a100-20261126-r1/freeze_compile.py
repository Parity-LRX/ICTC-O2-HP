import os,subprocess,sys
from pathlib import Path
folder=Path(__file__).parent
subprocess.run([sys.executable,'-u',str(folder/'preflight.py'),sys.argv[1]],check=True)
env=os.environ.copy();env['O2_PTX_LABEL']=sys.argv[1].upper()
subprocess.run([sys.executable,'-u',str(folder/'ptx_audit.py')],check=True,env=env)
