"""Select only a numerically qualified candidate that improves the target GEMM."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
g=json.loads((Path((ROOT/'GEMM-AUDIT.txt').read_text().strip())/'RESULT.json').read_text())
assert g['status']=='QUALIFIED_GEMM_MEASUREMENT_AUDIT_PASS'
wide=[r for r in g['rows'] if r['shape']==[256,1024]];assert len(wide)==2
assert all(r['c05_over_candidate']>1 for r in wide),'Target GEMM still regresses; keep the rejected snapshot and continue optimization'
(ROOT/'SELECTED.json').write_text(json.dumps({'status':'SELECTED_AFTER_FULL_AMD_AND_GEMM_AUDIT','GEMM_audit':(ROOT/'GEMM-AUDIT.txt').read_text().strip(),'candidate_commit':(ROOT/'CANDIDATE-HEAD.txt').read_text().strip(),'phases':[{k:r[k] for k in ('shape','phase','c05_over_candidate')} for r in wide]},indent=2)+'\n')
subprocess.run([sys.executable,'-u',str(Path(__file__).parent/'performance_preflight.py')],check=True)
