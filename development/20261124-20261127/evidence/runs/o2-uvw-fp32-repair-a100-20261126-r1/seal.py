"""AMD delivery seal: recoverable Git/source, receipts, stage-verified raw hashes."""
import ast,hashlib,json,os,shutil,tarfile
from pathlib import Path
ROOT=Path(os.environ['O2_REPAIR_ROOT']);WS=ROOT.parents[1]
def read(path):return json.loads(path.read_text())
def save(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
    selected=read(ROOT/'SELECTED.json');qual=read(ROOT/'QUALIFICATION-PREFLIGHT.json');performance=read(ROOT/'PERFORMANCE-PREFLIGHT.json')
    gates={}
    for pointer,status in (('REGRESSION-AUDIT.txt','ACTUAL_LOCAL_REGRESSION_PASS'),('GENERIC-AUDIT.txt','GENERIC_REGRESSION_PASS'),('REPLAY-AUDIT.txt','WHOLE_TRAINING_GATE_PASS'),('PERFORMANCE-AUDIT.txt','PERFORMANCE_AUDIT_PASS'),('PTX-AUDIT.txt','ACCEPTED_SOURCE_SM80_PTX_AUDIT_PASS')):
        path=Path((ROOT/pointer).read_text().strip());r=read(path/'RESULT.json')
        assert r['status']==status and r['operator_source_sha256']==selected['operator_source_sha256'];gates[pointer]={'path':str(path),'status':status,'receipt_sha256':sha(path/'RESULT.json')}
    fork=WS/'ICTC-O2-HP';assert (ROOT/'CANDIDATE-HEAD.txt').read_text().strip()==selected['commit']
    assert sha(fork/'src/ictc_o2_hp/uvw_triton.py')==selected['operator_source_sha256']
    for name,digest in qual['files'].items():assert sha(Path(qual['snapshot'])/name)==digest
    for name,digest in performance['files'].items():assert sha(Path(performance['snapshot'])/name)==digest
    for path in ROOT.glob('*.py'):ast.parse(path.read_text())
    allocations=[]
    final_performance=Path((ROOT/'PERFORMANCE-RESULT.txt').read_text().strip()).name.rsplit('-',1)[1]
    gpu_ids=set(('119339 119343 119345 119348 119351 119359 119362 119364 119371 119381 119391 119406 119412 119414 119436 119444 '+final_performance).split())
    for line in (ROOT/'ACCOUNTING.tsv').read_text().splitlines():
        fields=line.split('|')
        if fields[0] not in gpu_ids:continue
        job,name,state,exitcode,elapsed,tres,nodes,node=fields[:8]
        assert int(nodes)==1 and node=='XHPC-A100'
        if job=='119444':assert state.startswith('CANCELLED')
        else:assert state=='COMPLETED' and exitcode=='0:0'
        allocations.append({'job':job,'name':name,'state':state,'elapsed_s':int(elapsed),'nodes':1,'gpus':1,'gpu_h':int(elapsed)/3600})
    assert {r['job'] for r in allocations}==gpu_ids
    accounting={'allocations':allocations,'total_gpu_h':sum(r['gpu_h'] for r in allocations),'maximum_concurrent_nodes':1,
        'gpus_per_allocation':1,'method':'Slurm actual ElapsedRaw, explicit --gres=gpu:1 and actual device_count=1; serial allocations',
        'includes_failed_numerical_candidates':True,'project_cap_gpu_h':500,'append_by_unique_job_id':True,
        'project_total_not_recomputed':True,'earlier_ledger':'runs/o2-hp-approved-20261123-r1/provenance/BUDGET-AUDIT.json'}
    save(ROOT/'GPU-ACCOUNTING.json',accounting)
    target=ROOT/'snapshots'/('delivery-'+os.environ['SLURM_JOB_ID']);target.mkdir();source=target/'source';source.mkdir()
    shutil.copytree(fork,source/'ICTC-O2-HP',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(qual['snapshot'],source/'qualified-source')
    shutil.copytree(performance['snapshot'],source/'performance-source')
    controls=source/'controls';controls.mkdir()
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ('.py','.md','.json','.txt','.tsv','.sbatch') and p.name!='FINAL-SNAPSHOT.json':shutil.copy2(p,controls/p.name)
    logs=source/'logs';logs.mkdir()
    for p in (ROOT/'logs').glob('*.log'):
        if os.environ['SLURM_JOB_ID'] not in p.name:shutil.copy2(p,logs/p.name)
    old=ROOT.parent/'o2-uvw-triton-a100-20261125-r1';shutil.copytree(old/'snapshots/preflight-119174',source/'frozen-old-control')
    for pointer in gates:
        path=Path((ROOT/pointer).read_text().strip());destination=source/'receipts'/path.name;destination.mkdir(parents=True)
        for p in path.iterdir():
            if p.is_file() and p.suffix in ('.json','.csv','.txt','.md','.py','.ptx'):shutil.copy2(p,destination/p.name)
        if (path/'traces').is_dir():shutil.copytree(path/'traces',destination/'traces',ignore=shutil.ignore_patterns('KERNEL-EVENTS.csv'))
    supplemental=ROOT/'results/verify-audit-amd-119415';shutil.copytree(supplemental,source/'receipts'/supplemental.name)
    docs=['AGENTS.md','HANDOFF.md','OPERATOR_STATUS.md','architectures/equiformerv3/README.zh-CN.md','architectures/equiformerv3/UVW-FP32-REPAIR-20261126-r1.zh-CN.md']
    for name in docs:
        p=WS/name;destination=source/'workspace-docs'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination)
    shutil.copytree(ROOT/'figures',source/'figures')
    # Reuse hashes already checked against full arrays by each AMD acceptance job.
    # This avoids pretending the final light archive contains the large tensor payloads.
    raw={}
    def record(path,digest,verified_by):
        path=Path(path);assert path.is_file()
        raw[str(path)]={'bytes':path.stat().st_size,'sha256':digest,'content_verified_by':verified_by,'final_seal_rehashed':False}
    capture=Path((ROOT/'CAPTURE-RESULT.txt').read_text().strip());c=read(capture/'CAPTURE.json')
    for rows in c['backends'].values():
        for r in rows:record(r['path'],r['sha256'],'AMD119340')
    regression=Path((ROOT/'REGRESSION-RESULT.txt').read_text().strip())
    for r in read(regression/'REGRESSION.json')['records']:record(r['path'],r['sha256'],'AMD119411')
    generic=Path((ROOT/'GENERIC-RESULT.txt').read_text().strip())
    for r in read(generic/'MANIFEST.json')['records']:record(r['path'],r['sha256'],'AMD119413')
    for pointer,verified in (('REPLAY-RESULT.txt','AMD119437'),('VERIFY-RESULT.txt','AMD119415')):
        path=Path((ROOT/pointer).read_text().strip())
        for backend,rows in read(path/'MANIFEST.json')['cases'].items():
            for name,digest in rows.items():record(path/backend/name,digest,verified)
    for p in (ROOT/'results').rglob('*'):
        if p.is_file() and p.suffix in ('.json','.csv','.txt') and p.stat().st_size<16*2**20:
            record(p,sha(p),'final AMD seal');raw[str(p)]['final_seal_rehashed']=True
    performance_audit=Path((ROOT/'PERFORMANCE-AUDIT.txt').read_text().strip())
    for p in (performance_audit/'traces').glob('*/SUMMARY.json'):
        r=read(p);record(r['fingerprint']['path'],r['trace_sha256'],'AMD performance audit')
    save(target/'RAW-RESULT-MANIFEST.json',raw)
    dependencies={}
    for mode,folder in read(old/'PREFLIGHT.json')['common_states'].items():
        path=Path(folder);c=read(path/'CONDITION.json');checkpoint=Path(c['identity']['checkpoint'])
        assert sha(path/'START.pt')==c['start_sha256'] and sha(checkpoint)==c['identity']['checkpoint_sha256']
        dependencies[mode]={'common_state':str(path/'START.pt'),'common_state_sha256':c['start_sha256'],
            'condition':str(path/'CONDITION.json'),'condition_sha256':sha(path/'CONDITION.json'),
            'checkpoint':str(checkpoint),'checkpoint_sha256':c['identity']['checkpoint_sha256']}
    save(source/'EXTERNAL-DEPENDENCIES.json',{'old_control':str(old/'snapshots/preflight-119174'),
        'old_preflight':str(old/'PREFLIGHT.json'),'old_preflight_sha256':sha(old/'PREFLIGHT.json'),'immutable_inputs':dependencies,
        'raw_manifest':str(target/'RAW-RESULT-MANIFEST.json'),'raw_manifest_sha256':sha(target/'RAW-RESULT-MANIFEST.json'),
        'scope':'recoverable Git/source/receipts archived; large raw tensors and immutable common states/checkpoints retained at original paths; failed exploratory payloads retained separately, not claimed fully hashed here'})
    manifest={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()};save(target/'SOURCE-MANIFEST.json',manifest)
    archive=target/'SOURCE-AND-RECEIPTS.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:tf.add(source,arcname='source')
    unpack=target/'independent-unpack';unpack.mkdir()
    with tarfile.open(archive) as tf:tf.extractall(unpack,filter='data')
    found={str(p.relative_to(unpack/'source')):sha(p) for p in (unpack/'source').rglob('*') if p.is_file()}
    assert found==manifest
    receipt={'status':'SOURCE_AND_RECEIPTS_SEALED_WITH_INDEPENDENT_UNPACK_QA','job':os.environ['SLURM_JOB_ID'],
        'candidate_commit':selected['commit'],'operator_source_sha256':selected['operator_source_sha256'],
        'snapshot':str(target),'archive':str(archive),'archive_sha256':sha(archive),'source_files_verified':len(manifest),
        'raw_manifest':str(target/'RAW-RESULT-MANIFEST.json'),'raw_manifest_sha256':sha(target/'RAW-RESULT-MANIFEST.json'),
        'raw_content_verification':'stage AMD jobs as recorded; small receipts rehashed during final seal; no claim of final reread of all large arrays',
        'gates':gates,'gpu_accounting':accounting,'independent_trajectory_fail_retained':str(supplemental),
        'pushed':False}
    save(ROOT/'FINAL-SNAPSHOT.json',receipt);save(target/'FINAL-SNAPSHOT.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('gates','gpu_accounting')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
