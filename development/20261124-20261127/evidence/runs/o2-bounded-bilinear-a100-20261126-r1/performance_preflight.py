import ast,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
from candidate_sources import qualified_sources
sources=qualified_sources()
qual=json.loads((ROOT/'FULL-PREFLIGHT.json').read_text());gate=json.loads((Path((ROOT/'FULL-AUDIT.txt').read_text().strip())/'RESULT.json').read_text())
assert gate['status']=='FULL_QUALIFICATION_PASS' and gate['operator_source_sha256']==qual['operator_source_sha256']
out=ROOT/'snapshots'/('performance-'+os.environ['SLURM_JOB_ID']);out.mkdir();(out/'control').mkdir()
shutil.copy2(ROOT/'CONTRACT.zh-CN.md',out/'CONTRACT.zh-CN.md')
for p in ROOT.glob('*.py'):ast.parse(p.read_text());shutil.copy2(p,out/'control'/p.name)
parser=ROOT.parents[1]/'runs/o2-paired-diagnostic-a100-20261124-r1/analyze.py';shutil.copy2(parser,out/'control/trace_parser.py')
files={str(p.relative_to(out)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in out.rglob('*') if p.is_file()}
r={'snapshot':str(out),'files':files,'qualified_source':str(Path(qual['snapshot'])/'ictc-source'),'operator_source_sha256':gate['operator_source_sha256'],'AMD_gate':str(Path((ROOT/'FULL-AUDIT.txt').read_text().strip())/'RESULT.json'),
    'qualified_backends':{k:{n:str(v) for n,v in r.items()} for k,r in sources.items()}}
(ROOT/'PERFORMANCE-PREFLIGHT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
