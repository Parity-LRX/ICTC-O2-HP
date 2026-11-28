import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
for pointer,status in (('LOCAL-AUDIT.txt','LOCAL_BILINEAR_PASS'),):
    path=Path((ROOT/pointer).read_text().strip());assert json.loads((path/'RESULT.json').read_text())['status']==status
if sys.argv[1]=='full':
    path=Path((ROOT/'QUICK-AUDIT.txt').read_text().strip());assert json.loads((path/'RESULT.json').read_text())['status']=='QUICK_THREE_STEP_PASS'
subprocess.run([sys.executable,'-u',str(Path(__file__).parent/'preflight.py'),sys.argv[1]],check=True)
