import ast,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
out=ROOT/'snapshots'/('control-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
for path in ROOT.iterdir():
    if path.suffix in ('.py','.md','.sbatch'):
        if path.suffix=='.py':ast.parse(path.read_text())
        shutil.copy2(path,out/path.name)
WS=ROOT.parents[1];source=WS/'runs/o2-dd7a-angular-capacity-a100-20261126-r1/snapshots/preparation-120410/src/ictc_o2_hp/bilinear_triton.py'
text=source.read_text();text=text.replace('tl.program_id(0)','tl.program_id(GRID_SWAP_TEMP)').replace('tl.program_id(1)','tl.program_id(0)').replace('tl.program_id(GRID_SWAP_TEMP)','tl.program_id(1)')
text=text.replace('(triton.cdiv(oa,32),triton.cdiv(main*h,32))','(triton.cdiv(main*h,32),triton.cdiv(oa,32))')
text=text.replace('(triton.cdiv(oa,32),triton.cdiv((n-main)*h,32))','(triton.cdiv((n-main)*h,32),triton.cdiv(oa,32))')
ast.parse(text);(out/'grid_transposed.py').write_text(text)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
files={p.name:sha(p) for p in out.iterdir() if p.is_file()}
(ROOT/'PREFLIGHT.json').write_text(json.dumps({'snapshot':str(out),'files':files,'source_bilinear_sha256':sha(source),'job':os.environ['SLURM_JOB_ID']},indent=2)+'\n')
print('FROZEN',out,flush=True)
