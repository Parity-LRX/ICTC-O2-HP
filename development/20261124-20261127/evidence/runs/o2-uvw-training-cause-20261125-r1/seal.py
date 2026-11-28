"""Scheduled integrity review, raw-object seals and independently unpacked delivery."""
import ast,hashlib,json,os,shutil,subprocess,tarfile,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[1]
assert os.environ.get('SLURM_JOB_ID')
def read(path):return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
analysis=Path((ROOT/'ANALYSIS-RESULT.txt').read_text().strip());synthesis=Path((ROOT/'SYNTHESIS-RESULT.txt').read_text().strip())
summary=read(analysis/'SUMMARY.json');total=read(synthesis/'TOTAL-LOSS-CONTROL.json')
assert summary['tiled']['gradients']['failed']==4
assert summary['tiled']['force_original']['failed']==50
assert summary['tiled']['force_original']['failed_parameter_roles']==49
assert summary['tiled']['force_original']['component_fail_count']==3
assert summary['tiled']['force_shared_vjp']['failed']==0
assert summary['T-T-N']['force_original']['failed']==51
assert summary['N-N-T']['gradients']['failed']==0 and summary['N-N-T']['force_original']['failed']==0
assert len(total['tiled']['direct_vs_native_direct']['failed'])==4
assert not total['tiled']['shared_vs_native_direct']['failed']
assert total['tiled']['direct_vs_own']['maximum_relative_l2']==0
for label in ('energy','stress'):assert total['tiled']['loss_cotangents'][label]['bitwise_equal']
gpu_roots=[Path((ROOT/name).read_text().strip()) for name in ('GPU-RESULT.txt','TOTAL-RESULT.txt')]
startups=[read(p/'STARTUP.json') for p in gpu_roots]
assert startups[0]['complete_state_identity']==startups[1]['complete_state_identity']
assert startups[0]['graph']==startups[1]['graph']
for s in startups:
    assert s['hardware']['gpu']=='NVIDIA A100-SXM4-40GB'
    assert s['hardware']['deterministic'] and s['hardware']['threads']==4
    assert not any(s['hardware'][k] for k in ('TF32','AMP','CUDA_Graph'))
assert startups[0]['hardware']['gpu_uuid']==startups[1]['hardware']['gpu_uuid']
fork=WS/'ICTC-O2-HP'
# Compute node has no git executable; use the login control-plane read-only receipts.
head=(ROOT/'FORK-HEAD.txt').read_text().strip()
status=(ROOT/'FORK-STATUS.txt').read_text()
assert head=='df12a3a85ec3b66c9e4909dca0c90a86a12b1516' and not status
for p in ROOT.glob('*.py'):ast.parse(p.read_text())
for receipt in (read(ROOT/'PREFLIGHT.json'),read(ROOT/'PREFLIGHT-TOTAL.json')):
    for name,digest in receipt['files'].items():assert sha(Path(receipt['snapshot'])/name)==digest
raw={str(p.relative_to(ROOT)):sha(p) for base in gpu_roots for p in base.glob('*.pt')}
save(ROOT/'RAW-OBJECTS-SHA256.json',raw)
for manifest_path,key in ((analysis/'MANIFEST.json','sources'),(synthesis/'MANIFEST.json','total_objects_sha256')):
    manifest=read(manifest_path);base=gpu_roots[0 if key=='sources' else 1]
    for name,digest in manifest[key].items():assert sha(base/name)==digest
review={'status':'DIAGNOSTIC_EVIDENCE_REVIEW_PASS','formal_training_status':'UNCHANGED_FAIL',
    'fork_commit':head,'fork_clean':not bool(status),'old_failed_roles':54,'reproduced_failed_roles':54,
    'force_mixed_same_cotangent_failures':0,'full_loss_same_cotangent_failures':0,
    'same_physical_gpu':startups[0]['hardware']['gpu_uuid'],'all_source_ast_parse_pass':True,
    'raw_object_count':len(raw),'raw_objects_manifest':'RAW-OBJECTS-SHA256.json'}
save(ROOT/'REVIEW.json',review)
delivery=ROOT/'snapshots'/('delivery-'+os.environ['SLURM_JOB_ID']);delivery.mkdir(parents=True)
files={}
for p in ROOT.rglob('*'):
    if not p.is_file():continue
    relative=p.relative_to(ROOT)
    if any(part in ('triton-cache','__pycache__') for part in relative.parts):continue
    if relative.parts[0]=='snapshots' or p.suffix=='.pt' or p.name=='FINAL-SNAPSHOT.json':continue
    if p.suffix not in ('.py','.sbatch','.md','.json','.csv','.txt','.log'):continue
    files['run/'+str(relative)]=p
for p in (WS/'AGENTS.md',WS/'HANDOFF.md',WS/'OPERATOR_STATUS.md',
    WS/'architectures/equiformerv3/README.zh-CN.md',WS/'architectures/equiformerv3/TRAINING-FAIL-CAUSE-20261126-r1.zh-CN.md'):
    files['workspace_docs/'+str(p.relative_to(WS))]=p
for key in ('PREFLIGHT.json','PREFLIGHT-TOTAL.json'):
    receipt=read(ROOT/key);base=Path(receipt['snapshot'])
    for p in base.rglob('*'):
        if p.is_file():files['sealed_controls/'+base.name+'/'+str(p.relative_to(base))]=p
digests={name:sha(p) for name,p in files.items()};save(delivery/'FILES-SHA256.json',digests)
archive=delivery/'SOURCE-AND-RECEIPTS.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for name,p in sorted(files.items()):tar.add(p,arcname=name,recursive=False)
with tempfile.TemporaryDirectory(dir=delivery,prefix='unpack-review-') as temp:
    base=Path(temp)
    with tarfile.open(archive) as tar:
        assert {m.name for m in tar.getmembers()}==set(digests)
        for member in tar.getmembers():assert (base/member.name).resolve().is_relative_to(base.resolve())
        tar.extractall(base,filter='data')
    for name,digest in digests.items():assert sha(base/name)==digest
result={**review,'job':os.environ['SLURM_JOB_ID'],'archive':str(archive),'archive_sha256':sha(archive),
    'archive_files':len(digests),'independent_unpack_member_hash_pass':True,
    'raw_objects_embedded':False,'raw_object_location':'results/a100-119324 and results/total-a100-119328',
    'source_of_model_weights_inputs':'Original sealed119167/preflight119174 and common-state receipt referenced in PREFLIGHT.json; unchanged.'}
save(ROOT/'FINAL-SNAPSHOT.json',result);print(json.dumps(result),flush=True)
