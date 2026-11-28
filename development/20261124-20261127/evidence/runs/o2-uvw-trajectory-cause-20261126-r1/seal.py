"""AMD source/receipt seal, raw hashes, independent archive unpack."""
import ast,hashlib,json,os,shutil,tarfile
from pathlib import Path
ROOT=Path(os.environ['O2_TRAJECTORY_ROOT']);WS=ROOT.parents[1];assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
def read(p):return json.loads(p.read_text())
def save(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
audit=Path((ROOT/'GPU-AUDIT.txt').read_text().strip());a=read(audit/'RESULT.json');gpu=Path((ROOT/'GPU-RESULT.txt').read_text().strip())
assert a['status']=='PHASE_MATCHED_CAUSAL_AUDIT_COMPLETE' and all(r['failed']==0 for r in a['same_state_actual_RNG'].values())
assert sum(r['failed'] for r in a['independent'].values())==202
assert all(r['failed']==0 and r['maximum_absolute']==0 for r in a['reproduction'].values())
assert all(a['input_state_byte_hashes_match_old'].values())
assert a['decomposition']['state_only_failures']==201 and a['decomposition']['max_context_R']==a['decomposition']['max_closure_R']==0
for p in ROOT.glob('*.py'):ast.parse(p.read_text())
rows=[line.split('|') for line in (ROOT/'ACCOUNTING.tsv').read_text().splitlines() if line];row=next(r for r in rows if r[0]=='120028')
assert row[2]=='COMPLETED' and row[3]=='0:0' and row[6]=='1' and row[7]=='XHPC-A100'
accounting={'job':'120028','nodes':1,'gpus':1,'elapsed_s':int(row[4]),'gpu_h':int(row[4])/3600,
    'method':'actual Slurm ElapsedRaw x one requested/visible GPU; all cases serial','CPU_receipts':'ACCOUNTING.tsv',
    'project_cap_gpu_h':500,'project_total_not_recomputed':True,'previous_run_gpu_h':0.9208333333333334,'append_once_by_job_id':True}
save(ROOT/'GPU-ACCOUNTING.json',accounting)
target=ROOT/'snapshots'/('delivery-'+os.environ['SLURM_JOB_ID']);target.mkdir();source=target/'source';source.mkdir()
controls=source/'control';controls.mkdir()
for p in ROOT.iterdir():
    if p.is_file() and p.suffix in ('.py','.md','.json','.tsv','.txt','.sbatch') and p.name!='FINAL-SNAPSHOT.json':shutil.copy2(p,controls/p.name)
shutil.copytree(audit,source/'causal-audit')
shutil.copytree(Path((ROOT/'ANALYSIS-RESULT.txt').read_text().strip()),source/'saved-array-analysis')
preflight=read(ROOT/'PREFLIGHT.json');shutil.copytree(preflight['snapshot'],source/'executed-gpu-source')
repair=ROOT.parent/'o2-uvw-fp32-repair-a100-20261126-r1';qual=read(repair/'QUALIFICATION-PREFLIGHT.json')
shutil.copytree(qual['snapshot'],source/'accepted-reference-and-candidate')
for name,digest in preflight['dependencies'].items():assert sha(Path(name))==digest
logs=source/'logs';logs.mkdir()
for p in (ROOT/'logs').glob('*.log'):
    if os.environ['SLURM_JOB_ID'] not in p.name:shutil.copy2(p,logs/p.name)
docs=['AGENTS.md','HANDOFF.md','OPERATOR_STATUS.md','architectures/equiformerv3/README.zh-CN.md','architectures/equiformerv3/TRAJECTORY-CAUSE-20261126-r1.zh-CN.md']
for name in docs:
    dest=source/'workspace-docs'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(WS/name,dest)
raw={str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (ROOT/'results').rglob('*') if p.is_file()}
save(target/'RAW-MANIFEST.json',raw)
parent=read(repair/'FINAL-SNAPSHOT.json');assert sha(Path(parent['archive']))==parent['archive_sha256']
save(source/'DEPENDENCIES.json',{'prior_snapshot_receipt':str(repair/'FINAL-SNAPSHOT.json'),'prior_archive':parent['archive'],
    'prior_archive_sha256':parent['archive_sha256'],'raw_manifest':str(target/'RAW-MANIFEST.json'),
    'raw_manifest_sha256':sha(target/'RAW-MANIFEST.json'),
    'raw_scope':'all current run result files rehashed on AMD; large tensors/RNG inputs remain at original paths; older immutable dependencies are in previous sealed run'})
manifest={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()};save(target/'SOURCE-MANIFEST.json',manifest)
archive=target/'SOURCE-AND-RECEIPTS.tar.gz'
with tarfile.open(archive,'w:gz') as tf:tf.add(source,arcname='source')
unpack=target/'independent-unpack';unpack.mkdir()
with tarfile.open(archive) as tf:tf.extractall(unpack,filter='data')
found={str(p.relative_to(unpack/'source')):sha(p) for p in (unpack/'source').rglob('*') if p.is_file()};assert found==manifest
receipt={'status':'DIAGNOSIS_SEALED_WITH_INDEPENDENT_UNPACK_QA','job':os.environ['SLURM_JOB_ID'],'source_files_verified':len(manifest),
    'raw_files_rehashed':len(raw),'archive':str(archive),'archive_sha256':sha(archive),'snapshot':str(target),
    'raw_manifest':str(target/'RAW-MANIFEST.json'),'raw_manifest_sha256':sha(target/'RAW-MANIFEST.json'),
    'candidate_commit_unchanged':'a80f0dc9c8ed71ddd513ba12a579733ed1580429','candidate_source_sha256':a['operator_source_sha256'],
    'diagnosis':'28 K FP32 gradient differences seed independent update divergence; native state-only recovers201; one boundary sum; original phase RNG now matched',
    'trajectory_status':'202 FAIL retained; repair plan only','gpu_accounting':accounting,'archive_scope':'source and receipts; raw tensors external and fully hashed'}
save(ROOT/'FINAL-SNAPSHOT.json',receipt);save(target/'FINAL-SNAPSHOT.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)
