"""Publish bounded control/receipt snapshots from a scheduler-managed CPU node."""
import hashlib,json,os,shutil,sys
from pathlib import Path
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
root=Path(__file__).resolve().parent;repo=root.parents[1]/'ICTC-O2-HP'
out=repo/'development/20261127-stability'/sys.argv[1]
files=[root/'CONTRACT.zh-CN.md',root/'PROGRESS.zh-CN.md',root/'gpu.sbatch',root/'cpu.sbatch',root/'SOURCE.json']
if sys.argv[1]=='allocator-v7a':
    files += sorted((root/'snapshots/allocator-120844/control').glob('*.py'))
    files += [root/'INITIAL-PREFLIGHT.json',root/'ALLOCATOR-PREFLIGHT.json',root/'gate-audits/120846/RESULT.json']
else:
    files += [(Path(x) if Path(x).is_absolute() else root/Path(x)).resolve() for x in sys.argv[2:]]
for file in files:
    assert file.is_file(),file
    file.relative_to(root)
out.mkdir(parents=True)
manifest=[]
for file in files:
    rel=file.relative_to(root);target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
    manifest.append({'file':str(rel),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size})
(out/'MANIFEST.json').write_text(json.dumps({'files':manifest,'scheduler_job':os.environ['SLURM_JOB_ID']},indent=2)+'\n')
print('SNAPSHOT',out,len(manifest),flush=True)
