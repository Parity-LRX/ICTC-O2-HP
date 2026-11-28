import csv,hashlib,json,os,shutil,subprocess,tarfile,tempfile
from pathlib import Path
from support import *
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
audit=Path((ROOT/'AUDIT-RESULT.txt').read_text().strip());result=json.loads((audit/'RESULT.json').read_text())
assert result['status']=='DIAGNOSTIC_AUDIT_COMPLETE'
for row in result['ablation_summary']:
    assert row['execution_status']=='CAPTURED'
    if row['same_backend_reproduction']:
        assert all(not v['failures'] and v['different_roles']==0 for v in row['same_backend_reproduction'])
optimizer=json.loads((audit/'OPTIMIZER.json').read_text())
assert all(x['status']=='CUDA_OPTIMIZER_REPLAY' for x in optimizer)
assert all(not r['updated']['failures'] and r['updated']['different_roles']==0 and not r['ema']['failures'] and r['ema']['different_roles']==0 for case in optimizer for r in case['rows'])
source=json.loads((SWEEP/'SOURCE.json').read_text());source_root=Path(source['path']);checked={}
for name,digest in source['files'].items():
    if name.startswith('src/'):
        path=source_root.parent/name;actual=sha(path);assert actual==digest;checked[str(path)]=actual
save(ROOT/'PRESERVED-SOURCE-CHECK.json',{'candidate':'dd7a3db136c1d2946aa0e41b87a167d8f940b122','files':checked})
accounting=(ROOT/'ACCOUNTING-RECEIPT.tsv').read_text()
(ROOT/'ACCOUNTING.tsv').write_text(accounting)
jobs=[]
for line in accounting.splitlines():
    if not line.strip():continue
    jid,state,elapsed,code,node,*_=line.split('|');assert state=='COMPLETED'
    jobs.append({'job':jid,'state':state,'elapsed_seconds':int(elapsed),'gpus':1,'nodes':1,'node':node,'GPUh':int(elapsed)/3600,'exit_code':code})
save(ROOT/'GPU-ACCOUNTING.json',{'allocations':jobs,'GPUh':sum(v['GPUh'] for v in jobs),'serial':True,'old_120441_not_double_counted':True})
raw=[]
for path in (ROOT/'results').rglob('*'):
    if path.is_file():raw.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha(path)})
save(ROOT/'RAW-RESULT-MANIFEST.json',{'files':raw,'note':'Large arrays and allocator snapshots remain at these paths; control archive is not a full raw-array archive.'})
files=[]
for path in ROOT.rglob('*'):
    if not path.is_file() or 'triton-cache' in path.parts or path.suffix in ('.pt','.pickle','.gz') or path.name=='FINAL-SNAPSHOT.json':continue
    if path.suffix in ('.py','.md','.json','.sbatch','.txt','.tsv') or 'snapshots' in path.parts:files.append(path)
archive=ROOT/'diagnostic-controls-and-receipts.tar.gz'
with tarfile.open(archive,'w:gz') as f:
    for path in sorted(files):f.add(path,arcname=str(path.relative_to(ROOT)))
with tempfile.TemporaryDirectory(prefix='o2-cause-verify-',dir=ROOT) as tmp:
    with tarfile.open(archive) as f:f.extractall(tmp,filter='data')
    for path in files:assert sha(Path(tmp)/path.relative_to(ROOT))==sha(path)
save(ROOT/'FINAL-SNAPSHOT.json',{'status':'DIAGNOSTIC_EVIDENCE_SEALED','archive':str(archive),'archive_sha256':sha(archive),'archive_members':len(files),
     'independent_unpack_and_hash_check':True,'raw_results_manifest':str(ROOT/'RAW-RESULT-MANIFEST.json'),'audit':str(audit),
     'source_preserved':str(ROOT/'PRESERVED-SOURCE-CHECK.json'),'candidate_unchanged':'dd7a3db136c1d2946aa0e41b87a167d8f940b122',
     'GPUh':sum(v['GPUh'] for v in jobs),'no_new_backend_qualification':True})
print('SEALED',archive,len(files),flush=True)
