"""Freeze causal-probe source without changing the measured candidate or old run."""
import ast,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    assert os.environ.get('SLURM_JOB_ID')
    snapshot=ROOT/'snapshots'/('control-'+os.environ['SLURM_JOB_ID']);snapshot.mkdir(parents=True)
    for p in list(ROOT.glob('*.py'))+list(ROOT.glob('*.sbatch'))+list(ROOT.glob('*.md')):
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        shutil.copy2(p,snapshot/p.name)
    old=WS/'runs/o2-uvw-triton-a100-20261125-r1';sealed=old/'snapshots/preflight-119174'
    receipt=json.loads((old/'PREFLIGHT.json').read_text())
    for name in ('whole.py','bounded.py','adapter/install.py','fork-src/ictc_o2_hp/uvw_triton.py'):
        assert sha(sealed/name)==receipt['files'][name],name
    assert sha(WS/'ICTC-O2-HP/src/ictc_o2_hp/uvw_triton.py')==sha(sealed/'fork-src/ictc_o2_hp/uvw_triton.py')
    files={str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file()}
    result={'status':'SOURCE_PREFLIGHT_PASS','job':os.environ['SLURM_JOB_ID'],'snapshot':str(snapshot),'files':files,
        'candidate_commit':'df12a3a85ec3b66c9e4909dca0c90a86a12b1516','old_snapshot_verified':str(sealed),
        'old_gate_json_sha256':sha(old/'results/analysis-119186/train-tiled-GATE.json')}
    (ROOT/'PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
