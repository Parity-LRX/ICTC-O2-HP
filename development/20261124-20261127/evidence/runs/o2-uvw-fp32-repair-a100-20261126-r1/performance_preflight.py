import ast,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
qual=json.loads((ROOT/'QUALIFICATION-PREFLIGHT.json').read_text());gate=Path((ROOT/'VERIFY-AUDIT.txt').read_text().strip())
assert json.loads((gate/'RESULT.json').read_text())['status']=='WHOLE_TRAINING_GATE_PASS'
out=ROOT/'snapshots'/('performance-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
for name in ('common.py','performance.py','performance_driver.py','performance_audit.py'):
    p=ROOT/name;ast.parse(p.read_text());shutil.copy2(p,out/name)
parser=ROOT.parents[1]/'runs/o2-paired-diagnostic-a100-20261124-r1/analyze.py'
ast.parse(parser.read_text());shutil.copy2(parser,out/'trace_parser.py')
files={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in out.iterdir()}
receipt={'snapshot':str(out),'files':files,'qualified_source':str(Path(qual['snapshot'])/'ictc-source'),'job':os.environ['SLURM_JOB_ID'],
    'operator_source_sha256':qual['selected']['operator_source_sha256']}
(ROOT/'PERFORMANCE-PREFLIGHT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
