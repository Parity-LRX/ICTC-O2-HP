"""Scheduler-only source seal, inherited-source audit and recoverable snapshots."""
import ast, hashlib, json, os, shutil, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WS=ROOT.parents[1]
FORK=WS/'ICTC-O2-HP'
BASE=WS/'baselines/icml-o2-stage1-20261124-r1/payload/operator/upstream/official'
RUNTIME=WS/'runs/o2-v3-canonical-cpu-20261124-r1/tools/git'
def git(path,*args):
    environment=os.environ.copy();environment['GIT_EXEC_PATH']=str(ROOT/'tools')
    return subprocess.check_output([str(RUNTIME/'lib/ld-linux-x86-64.so.2'),'--library-path',str(RUNTIME/'lib'),str(RUNTIME/'bin/git'),
        '-c','core.hooksPath=/dev/null','-c','commit.gpgsign=false','-c','gc.auto=0','-c','pack.threads=2','-C',str(path),*args],text=True,env=environment).strip()
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    assert os.environ.get('SLURM_JOB_ID')
    inherited={}
    for p in BASE.rglob('*'):
        if p.is_file():
            rel=p.relative_to(BASE)
            if str(rel)=='src/ictc_o2_hp/__init__.py':continue
            assert sha(p)==sha(FORK/rel),str(rel)
            inherited[str(rel)]=sha(p)
    for tree in (ROOT,FORK/'src'):
        for p in tree.rglob('*.py'):
            if 'snapshots' not in p.parts:ast.parse(p.read_text(),filename=str(p))
    snapshot=ROOT/'snapshots'/('preflight-'+os.environ['SLURM_JOB_ID']);snapshot.mkdir(parents=True)
    for p in list(ROOT.glob('*.py'))+list(ROOT.glob('*.sbatch'))+list(ROOT.glob('*.md')):
        shutil.copy2(p,snapshot/p.name)
    shutil.copytree(ROOT/'adapter',snapshot/'adapter')
    shutil.copytree(FORK/'src',snapshot/'fork-src')
    commit=git(FORK,'rev-parse','HEAD')
    assert not git(FORK,'status','--porcelain'),'Commit implementation before preflight'
    # Preserve the complete object database directly; compute nodes do not ship
    # Git's child executables required by bundle/pack-objects.
    shutil.copytree(FORK/'.git',snapshot/'candidate.git')
    seal={str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file()}
    old=WS/'runs/o2-paired-diagnostic-a100-20261124-r1'
    previous=Path((old/'LATEST-RESULT.txt').read_text().strip())
    progress=json.loads((previous/'PROGRESS.json').read_text())
    common={mode:next(r['folder'] for r in progress if r['route']=='v3' and r['mode']==mode and r['action']=='prepare' and r['exit_code']==0) for mode in ('ef','train')}
    receipt={'status':'SOURCE_PREFLIGHT_PASS','job':os.environ['SLURM_JOB_ID'],'candidate_commit':commit,
        'snapshot':str(snapshot),'files':seal,'baseline_wrapper':'e4763c76bf183499357e40cb39cc0934bb98779b',
        'remote_base':(ROOT/'REMOTE-BASE.txt').read_text().strip(),'inherited_files_verified':inherited,
        'common_states':common,'frames_sha256':sha(ROOT/'prepared/FRAMES.pt')}
    (ROOT/'PREFLIGHT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('files','inherited_files_verified')}),flush=True)
if __name__=='__main__':main()
