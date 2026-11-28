"""Shared CPU provenance and frozen native model loader."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(os.environ['O2_GENERAL_RUN_DIR']).resolve()
WS = ROOT.parents[1]
SOURCE = Path(__file__).resolve().parent
NETWORK = WS/'baselines/icml-o2-stage1-20261124-r1/payload/network'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def start(stage):
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm CPU allocation required'
    output = ROOT/'results'/(stage+'-'+os.environ['SLURM_JOB_ID'])
    output.mkdir(parents=True, exist_ok=False)
    seal = json.loads((SOURCE/'SEAL.json').read_text())
    assert seal['job_id'] == os.environ['SLURM_JOB_ID'], 'Use this job own control snapshot'
    for name, digest in seal['files'].items():
        assert sha(SOURCE/name) == digest, name
    for name in ('deps', 'env-vendor', 'upstream/src', 'upstream', 'fixture-api'):
        sys.path.insert(0, str(NETWORK/name))
    prior = load('general_prior_'+stage, SOURCE/'prior_audit.py')
    native = load('general_native_'+stage, NETWORK/'fixture-api/whole_network.py')
    native.strict(torch)
    torch.set_num_threads(2)
    assert sha(native.CHECKPOINT) == native.CHECKPOINT_SHA
    net = native.Network('native', 'cpu')
    checkpoint = torch.load(native.CHECKPOINT, map_location='cpu', weights_only=False)
    save(output/'IDENTITY.json', {'stage': stage, 'job': os.environ['SLURM_JOB_ID'],
        'source_snapshot': str(SOURCE), 'source_seal_sha256': sha(SOURCE/'SEAL.json'),
        'checkpoint_sha256': native.CHECKPOINT_SHA, 'torch': torch.__version__,
        'dtype': 'float32 / complex64 only', 'gpu_jobs': 0, 'gpu_hours': 0})
    return output, net, checkpoint, prior


def parameter_hashes(model):
    return {name: hashlib.sha256(p.detach().contiguous().numpy().tobytes()).hexdigest()
            for name, p in model.named_parameters()}


def difference(a, b):
    assert a.dtype in (torch.float32, torch.complex64) and b.dtype == a.dtype
    assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    d = a-b
    nr, nd = b.norm(), d.norm()
    return {'max_abs': float(d.abs().max()), 'relative_l2': float(nd/nr) if nr else (0. if not nd else None)}


def finish(stage, output, result):
    save(output/'RESULT.json', result)
    target = ROOT/'snapshots'/('result-'+stage+'-'+os.environ['SLURM_JOB_ID'])
    target.mkdir(parents=True, exist_ok=False)
    shutil.copytree(output, target/'results')
    shutil.copytree(SOURCE, target/'control', ignore=shutil.ignore_patterns('.git'))
    save(target/'RESULT-SEAL.json', {str(p.relative_to(target)): sha(p)
                                   for p in target.rglob('*') if p.is_file()})
    prepare = load('general_prepare_'+stage, ROOT/'prepare.py')
    for args in (['init', '--quiet', '--template='], ['config', 'user.name', 'O2 snapshot'],
                 ['config', 'user.email', 'o2-snapshot@localhost'], ['add', '.'],
                 ['commit', '--quiet', '-m', 'Freeze general O2 '+stage+' CPU evidence']):
        prepare.git(target, args)
    commit = prepare.git(target, ['rev-parse', 'HEAD'])
    (ROOT/(stage.upper()+'-SNAPSHOT.txt')).write_text(str(target)+'\n'+commit+'\n')
    (ROOT/(stage.upper()+'-RESULT.txt')).write_text(str(output)+'\n')
    print(json.dumps({'stage': stage, 'status': result['status'], 'result': str(output),
                      'snapshot': str(target), 'commit': commit, 'gpu_hours': 0}), flush=True)
