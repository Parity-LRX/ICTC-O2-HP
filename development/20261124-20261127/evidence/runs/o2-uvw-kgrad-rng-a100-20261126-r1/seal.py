"""Recoverable sources/receipts plus explicit large-array references."""
import ast,hashlib,json,os,shutil,tarfile
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);WS=ROOT.parents[1];assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
def read(p):return json.loads(p.read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def main():
    gate_names={'PROTOCOL-AUDIT.txt':'PROTOCOL_ISOLATION_PASS','FULL-AUDIT.txt':'FULL_QUALIFICATION_PASS','RNG-AUDIT.txt':'RNG_PHASE_AND_CONTINUITY_PASS','PTX-AUDIT.txt':'FP32_SM80_PTX_AUDIT_PASS','PERFORMANCE-AUDIT.txt':'PERFORMANCE_AUDIT_PASS'}
    gates={};qualified=read(ROOT/'FULL-PREFLIGHT.json');operator=qualified['operator_source_sha256']
    for pointer,status in gate_names.items():
        path=Path((ROOT/pointer).read_text().strip());r=read(path/'RESULT.json');assert r['status']==status
        if pointer!='PROTOCOL-AUDIT.txt':assert r['operator_source_sha256']==operator
        gates[pointer]={'path':str(path),'receipt_sha256':sha(path/'RESULT.json'),'status':status}
    assert sha(WS/'ICTC-O2-HP/src/ictc_o2_hp/uvw_triton.py')==operator
    assert not (ROOT/'GIT-STATUS.txt').read_text().strip()
    trajectory=Path((ROOT/'QUALIFICATION-RESULT.txt').read_text().strip());conditions=read(trajectory/'CONDITIONS.json');continuity=[]
    for mode in ('legacy','isolated','train-only'):
        for backend in ('native','native-repeat','candidate','candidate-repeat'):
            for step in (1,2):
                before=conditions[f'{mode}-{backend}-step{step}'];prior=conditions[f'{mode}-{backend}-step{step-1}']
                assert before['before']==prior['after']
                assert before['rng']['train' if mode=='train-only' else 'mixed']==prior['rng_at_next_step']
                continuity.append({'mode':mode,'backend':backend,'step':step+1,'state_continuous':True,'RNG_continuous':True})
    save(ROOT/'CONTINUITY-AUDIT.json',{'status':'ALL_24_STATE_AND_RNG_TRANSITIONS_CONTINUOUS','rows':continuity,'conditions_sha256':sha(trajectory/'CONDITIONS.json')})
    for p in ROOT.glob('*.py'):ast.parse(p.read_text())
    stage_paths=set()
    for receipt_path in ROOT.glob('*-PREFLIGHT.json'):
        r=read(receipt_path);stage_paths.add(Path(r['snapshot']))
        for name,digest in r['files'].items():assert sha(Path(r['snapshot'])/name)==digest
    performance=Path((ROOT/'PERFORMANCE-RESULT.txt').read_text().strip());perf_job=performance.name.rsplit('-',1)[1]
    gpu_ids=set(('120046 120048 120056 120063 120068 120070 120073 120088 120091 '+perf_job).split());allocations=[]
    for line in (ROOT/'ACCOUNTING.tsv').read_text().splitlines():
        fields=line.split('|');job,name,state,exitcode,elapsed,tres,nodes,node=fields[:8]
        if job not in gpu_ids:continue
        assert node=='XHPC-A100' and int(nodes)==1
        assert state in ('COMPLETED','FAILED')
        allocations.append({'job':job,'name':name,'state':state,'exit_code':exitcode,'elapsed_s':int(elapsed),'start':fields[8],'end':fields[9],'nodes':1,'gpus':1,'GPUh':int(elapsed)/3600})
    assert {r['job'] for r in allocations}==gpu_ids
    ordered=sorted(allocations,key=lambda r:r['start'])
    assert all(a['end']<=b['start'] for a,b in zip(ordered,ordered[1:]))
    accounting={'allocations':allocations,'total_GPUh':sum(r['GPUh'] for r in allocations),'maximum_nodes':1,'gpus_per_allocation':1,
        'includes_failed_startups':True,'actual_scheduler_intervals_serial':True,
        'GPU_count_evidence':'each submitted with --gres=gpu:1; network startup asserts one visible CUDA device; local code uses only current CUDA device; site SlurmDB does not record GPU in ReqTRES/AllocTRES',
        'scheduler_TRES_record':str(ROOT/'SCHEDULER-GRES-RECORD.txt'),'project_budget_GPUh':500,'project_total_not_recomputed':True,'append_by_unique_job_ID':True,
        'prior_ledger':'runs/o2-hp-approved-20261123-r1/provenance/BUDGET-AUDIT.json'}
    save(ROOT/'GPU-ACCOUNTING.json',accounting)
    target=ROOT/'snapshots'/('delivery-'+os.environ['SLURM_JOB_ID']);target.mkdir();source=target/'source';source.mkdir()
    shutil.copytree(WS/'ICTC-O2-HP',source/'ICTC-O2-HP',ignore=shutil.ignore_patterns('__pycache__'))
    controls=source/'controls';controls.mkdir()
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ('.py','.md','.json','.txt','.tsv') and p.name!='FINAL-SNAPSHOT.json':shutil.copy2(p,controls/p.name)
    for p in sorted(stage_paths):shutil.copytree(p,source/'stage-snapshots'/p.name)
    shutil.copytree(ROOT/'figures',source/'figures')
    logs=source/'logs';logs.mkdir()
    for p in (ROOT/'logs').glob('*.log'):
        if os.environ['SLURM_JOB_ID'] not in p.name:shutil.copy2(p,logs/p.name)
    for g in gates.values():
        path=Path(g['path']);destination=source/'receipts'/path.name;destination.mkdir(parents=True)
        for p in path.iterdir():
            if p.is_file() and p.suffix in ('.json','.csv','.txt','.ptx'):shutil.copy2(p,destination/p.name)
        if (path/'traces').is_dir():shutil.copytree(path/'traces',destination/'traces',ignore=shutil.ignore_patterns('KERNEL-EVENTS.csv'))
    failed=ROOT/'results/qualification-a100-120073';destination=source/'receipts/rejected-v3b';destination.mkdir(parents=True)
    for name in ('MANIFEST.json','CONDITIONS.json','FRAMES.json','STARTUP.json'):shutil.copy2(failed/name,destination/name)
    base=WS/'runs/o2-uvw-fp32-repair-a100-20261126-r1';old=WS/'runs/o2-uvw-triton-a100-20261125-r1'
    shutil.copytree(old/'snapshots/preflight-119174',source/'external-controls/old-frozen-control')
    shutil.copytree(base/'snapshots/qualification-119423/control',source/'external-controls/v2-qualified-control')
    docs=['AGENTS.md','HANDOFF.md','OPERATOR_STATUS.md','architectures/equiformerv3/README.zh-CN.md','architectures/equiformerv3/KGRAD-RNG-REPAIR-20261126-r1.zh-CN.md']
    for name in docs:
        p=WS/name;destination=source/'workspace-docs'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination)
    raw={}
    def record(path,digest,verification):
        p=Path(path);assert p.is_file();raw[str(p)]={'bytes':p.stat().st_size,'sha256':digest,'verified_by':verification,'final_seal_rehashed':False}
    for kind,manifest in (('REGRESSION','REGRESSION.json'),('GENERIC','MANIFEST.json')):
        folder=Path((ROOT/(kind+'-RESULT.txt')).read_text().strip())
        for row in read(folder/manifest)['records']:record(row['path'],row['sha256'],gates['FULL-AUDIT.txt']['path'])
    folder=Path((ROOT/'QUALIFICATION-RESULT.txt').read_text().strip())
    for name,digest in read(folder/'MANIFEST.json')['files'].items():record(folder/name,digest,gates['FULL-AUDIT.txt']['path'])
    for name,digest in read(failed/'MANIFEST.json')['files'].items():record(failed/name,digest,'GPU manifest; AMD120084 checked original three-step raw gradients/updates only; full AMD120076 stopped after known failure')
    for pointer in ('PROTOCOL-RESULT.txt','LAYOUT-RESULT.txt'):
        folder=Path((ROOT/pointer).read_text().strip());m=read(folder/'MANIFEST.json')
        if pointer=='LAYOUT-RESULT.txt':
            for row in m['records']:record(row['path'],row['sha256'],(ROOT/'LAYOUT-AUDIT.txt').read_text().strip())
        else:
            for name,digest in m['files'].items():record(folder/name,digest,'AMD120049')
    capture=Path((base/'CAPTURE-RESULT.txt').read_text().strip())
    for rows in read(capture/'CAPTURE.json')['backends'].values():
        for r in rows:record(r['path'],r['sha256'],'prior immutable AMD119340 capture')
    for p in (ROOT/'results').rglob('*'):
        if p.is_file() and p.suffix in ('.json','.txt','.csv') and p.stat().st_size<16*2**20:
            record(p,sha(p),'final AMD seal');raw[str(p)]['final_seal_rehashed']=True
    for p in Path(gates['PERFORMANCE-AUDIT.txt']['path']).glob('traces/*/SUMMARY.json'):
        r=read(p);record(r['fingerprint']['path'],r['trace_sha256'],gates['PERFORMANCE-AUDIT.txt']['path'])
    micro=Path((ROOT/'GEMM-RESULT.txt').read_text().strip())
    for r in read(micro/'RESULT.json')['rows']:record(r['path'],r['sha256'],gates['PERFORMANCE-AUDIT.txt']['path'])
    save(target/'RAW-RESULT-MANIFEST.json',raw)
    deps={}
    for mode,folder in read(old/'PREFLIGHT.json')['common_states'].items():
        folder=Path(folder);c=read(folder/'CONDITION.json');checkpoint=Path(c['identity']['checkpoint'])
        assert sha(folder/'START.pt')==c['start_sha256'] and sha(checkpoint)==c['identity']['checkpoint_sha256']
        deps[mode]={'state':str(folder/'START.pt'),'state_sha256':c['start_sha256'],'checkpoint':str(checkpoint),'checkpoint_sha256':c['identity']['checkpoint_sha256']}
    deps['frames']={'path':str(old/'prepared/FRAMES.pt'),'sha256':sha(old/'prepared/FRAMES.pt')}
    save(source/'EXTERNAL-DEPENDENCIES.json',{'immutable_inputs':deps,'old_control':str(old/'snapshots/preflight-119174'),'base_v2_control':str(base/'snapshots/qualification-119423/control'),
        'raw_manifest':str(target/'RAW-RESULT-MANIFEST.json'),'raw_manifest_sha256':sha(target/'RAW-RESULT-MANIFEST.json'),
        'scope':'Git/source/receipts included; large tensors and original native model/environment remain at hashed workspace paths; stage AMD hash verification reused explicitly'})
    manifest={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()};save(target/'SOURCE-MANIFEST.json',manifest)
    archive=target/'SOURCE-AND-RECEIPTS.tar.gz'
    with tarfile.open(archive,'w:gz') as t:t.add(source,arcname='source')
    unpack=target/'independent-unpack';unpack.mkdir()
    with tarfile.open(archive) as t:t.extractall(unpack,filter='data')
    found={str(p.relative_to(unpack/'source')):sha(p) for p in (unpack/'source').rglob('*') if p.is_file()};assert found==manifest
    result={'status':'SEALED_WITH_INDEPENDENT_UNPACK_QA','job':os.environ['SLURM_JOB_ID'],'candidate_commit':(ROOT/'CANDIDATE-HEAD.txt').read_text().strip(),
        'operator_source_sha256':operator,'snapshot':str(target),'archive':str(archive),'archive_sha256':sha(archive),'source_files':len(manifest),
        'RAW_manifest':str(target/'RAW-RESULT-MANIFEST.json'),'RAW_manifest_sha256':sha(target/'RAW-RESULT-MANIFEST.json'),'gates':gates,'accounting':accounting,
        'old_202_FAIL_preserved':str(base/'results/verify-audit-amd-119415'),'new_original_independent_three_steps':'PASS','shared_cotangents':False,'pushed':False}
    save(ROOT/'FINAL-SNAPSHOT.json',result);save(target/'FINAL-SNAPSHOT.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
