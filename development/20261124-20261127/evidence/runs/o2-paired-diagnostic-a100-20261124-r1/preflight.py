import ast,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[1]
assert os.environ.get('SLURM_JOB_ID')
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
syntax={str(p.relative_to(ROOT)):'PASS' for p in list(ROOT.glob('*.py'))+list((ROOT/'adapter').glob('*.py')) if ast.parse(p.read_text())}
assert sha(ROOT/'prepared/FRAMES.pt')=='a9e246559540440a5be65f8f5a32fb7ee38362d0e91e2528c5af88afffa2352d'
assert sha(ROOT/'downloads/TECE-OAM-RRA-1.0.pt')=='9f36562582d931347c3904f763e820edcaf5f27c3f13beb6776e49fcc7de38bb'
reference=json.loads((WS/'runs/o2-three-route-exec-20261124-r1/results/preflight-118331/SOURCE-SEALS.json').read_text())
for role in ('operator','v3_canonical','tece_canonical','tece_full'):
    row=reference[role]
    for name,expected in row['files'].items():
        path=Path(row['root'])/name
        if 'sha256' in expected:assert sha(path)==expected['sha256'],str(path)
        else:assert path.is_symlink() and os.readlink(path)==expected['symlink']
result={'status':'CPU_PREFLIGHT_PASS','job':os.environ['SLURM_JOB_ID'],'syntax':syntax,'frozen_sources':'UNCHANGED','input_sha256':sha(ROOT/'prepared/FRAMES.pt'),
        'TECE_checkpoint_sha256':sha(ROOT/'downloads/TECE-OAM-RRA-1.0.pt')}
(ROOT/'PREFLIGHT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
