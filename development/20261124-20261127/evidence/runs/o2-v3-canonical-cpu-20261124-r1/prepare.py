"""Freeze CPU audit inputs and independently recheck the protected baseline."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(snapshot, args):
    runtime = ROOT / 'tools/git'
    command = [str(runtime/'lib/ld-linux-x86-64.so.2'), '--library-path', str(runtime/'lib'),
               str(runtime/'bin/git'), '-c', 'core.hooksPath=/dev/null', '-c', 'commit.gpgsign=false',
               '-c', 'core.fsmonitor=false', '-c', 'gc.auto=0', '-c', 'maintenance.auto=false',
               '-C', str(snapshot)]
    return subprocess.check_output(command + args, text=True).strip()


def main():
    assert os.environ.get('SLURM_JOB_ID')
    base = WS/'baselines/icml-o2-stage1-20261124-r1'
    manifest = json.loads((base/'BASELINE-MANIFEST.json').read_text())
    files = links = 0
    for name, entry in manifest.items():
        path = base/'payload'/name
        if entry['type'] == 'file':
            assert not path.is_symlink() and sha(path) == entry['sha256'], name
            files += 1
        elif entry['type'] == 'symlink':
            assert path.is_symlink() and os.readlink(path) == entry['target'], name
            links += 1
    snapshot = ROOT/'snapshots'/('control-'+os.environ['SLURM_JOB_ID'])
    snapshot.mkdir(parents=True, exist_ok=False)
    sources = list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+list(ROOT.glob('*.sbatch'))
    for source in sources:
        if source.suffix == '.py':
            ast.parse(source.read_text(), filename=str(source))
        shutil.copy2(source, snapshot/source.name)
    for directory in ('architectures', 'analysis/equiformerv3-optimization-plan-20261124-r1'):
        for source in (WS/directory).rglob('*'):
            if source.is_file() and source.suffix in ('.md', '.tsv'):
                target = snapshot/'workspace-inputs'/source.relative_to(WS)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    # Known corrected role loader, frozen by the prior valid CPU audit.
    helper = WS/'runs/o2-parity-map-20261124-r1/snapshots/control-cpu-117872/audit_mapping.py'
    shutil.copy2(helper, snapshot/'prior_audit.py')
    seal = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*') if p.is_file()}
    (snapshot/'SEAL.json').write_text(json.dumps({'files': seal, 'baseline_files': files,
        'baseline_links': links, 'baseline_manifest_sha256': sha(base/'BASELINE-MANIFEST.json'),
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu_jobs': 0, 'gpu_hours': 0}, indent=2)+'\n')
    for args in (['init', '--quiet', '--template='], ['config', 'user.name', 'O2 snapshot'],
                 ['config', 'user.email', 'o2-snapshot@localhost'], ['add', '.'],
                 ['commit', '--quiet', '-m', 'Freeze canonical O2 CPU conversion and validation inputs']):
        git(snapshot, args)
    commit = git(snapshot, ['rev-parse', 'HEAD'])
    (ROOT/'CONTROL-SNAPSHOT.txt').write_text(str(snapshot)+'\n'+commit+'\n')
    print(json.dumps({'input_snapshot': str(snapshot), 'commit': commit, 'baseline_files': files}))


if __name__ == '__main__':
    main()
