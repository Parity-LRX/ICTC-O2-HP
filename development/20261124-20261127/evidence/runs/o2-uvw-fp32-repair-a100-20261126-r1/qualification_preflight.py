"""AMD freezes the selected package and both GPU/AMD qualification controls."""
import ast,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[1];assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
selected=json.loads((ROOT/'SELECTED.json').read_text())
out=ROOT/'snapshots'/('qualification-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
shutil.copytree(WS/'ICTC-O2-HP/src',out/'ictc-source')
(out/'control').mkdir()
for name in ('common.py','metrics.py','real_regression.py','regression_audit.py','generic_regression.py','generic_audit.py','verify_capture.py','verify_audit.py','verify_replay.py','replay_audit.py'):
    p=ROOT/name;ast.parse(p.read_text());shutil.copy2(p,out/'control'/name)
files={str(p.relative_to(out)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in out.rglob('*') if p.is_file()}
assert files['ictc-source/ictc_o2_hp/uvw_triton.py']==selected['operator_source_sha256']
receipt={'snapshot':str(out),'files':files,'selected':selected,'job':os.environ['SLURM_JOB_ID']}
(ROOT/'QUALIFICATION-PREFLIGHT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
