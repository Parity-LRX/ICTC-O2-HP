"""Recoverable source/receipt archive with explicit large-array references."""
import ast,hashlib,json,os,shutil,tarfile
from pathlib import Path
from context import ROOT,WS,BASE,PREVIOUS,package_sha
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def main():
    pre=read(ROOT/'FULL-PREFLIGHT.json');operator=pre['operator_source_sha256'];gates={}
    for name,status in {'LOCAL-AUDIT':'LOCAL_BILINEAR_PASS','FULL-AUDIT':'FULL_QUALIFICATION_PASS','RNG-AUDIT':'RNG_PHASE_AND_CONTINUITY_PASS','PTX-AUDIT':'FP32_SM80_PTX_AUDIT_PASS','PERFORMANCE-AUDIT':'PERFORMANCE_AUDIT_PASS'}.items():
        folder=Path((ROOT/(name+'.txt')).read_text().strip());r=read(folder/'RESULT.json');assert r['status']==status and r['operator_source_sha256']==operator
        gates[name]={'path':str(folder),'status':status,'sha256':sha(folder/'RESULT.json')}
    package=read(Path(gates['FULL-AUDIT']['path'])/'RESULT.json')['package_sha256']
    assert package_sha(WS/'ICTC-O2-HP/src')==package and not (ROOT/'GIT-STATUS.txt').read_text().strip()
    baseline=read(ROOT/'BASELINE-8324-SNAPSHOT.json');assert sha(baseline['archive'])==baseline['archive_sha256']
    assert read(ROOT/'PLAN-VALIDATION.json')['status']=='ALL_NATIVE_PLAN_LEAVES_AND_MERGES_MATCH'
    assert read(ROOT/'BILINEAR-TRACE-AUDIT.json')['status']=='BILINEAR_FUSION_TRACE_PASS'
    for p in ROOT.glob('*.py'):ast.parse(p.read_text())
    stages=set()
    for p in ROOT.glob('*-PREFLIGHT.json'):
        r=read(p);folder=Path(r['snapshot']);stages.add(folder)
        for name,digest in r['files'].items():assert sha(folder/name)==digest,(p,name)
    trajectory=Path((ROOT/'QUALIFICATION-RESULT.txt').read_text().strip());conditions=read(trajectory/'CONDITIONS.json');continuity=[]
    for mode in ('legacy','isolated','train-only'):
        for backend in ('native','native-repeat','candidate','candidate-repeat'):
            for step in (1,2):
                now=conditions[f'{mode}-{backend}-step{step}'];before=conditions[f'{mode}-{backend}-step{step-1}']
                assert now['before']==before['after'] and now['rng']['train' if mode=='train-only' else 'mixed']==before['rng_at_next_step']
                continuity.append({'mode':mode,'backend':backend,'step':step+1})
    save(ROOT/'CONTINUITY-AUDIT.json',{'status':'ALL_24_STATE_AND_RNG_TRANSITIONS_CONTINUOUS','rows':continuity})
    expected=set((ROOT/'GPU-JOB-IDS.txt').read_text().split());allocations=[]
    for line in (ROOT/'SACCT-ALL.tsv').read_text().splitlines():
        f=line.split('|')
        if f[0] not in expected:continue
        job,name,state,exitcode,elapsed,tres,nodes,node=f[:8];seconds=int(elapsed)
        active=node=='XHPC-A100' and seconds>0
        assert (active and int(nodes)==1) or (seconds==0 and state.startswith('CANCELLED')),(job,f)
        allocations.append({'job':job,'name':name,'state':state,'exit_code':exitcode,'elapsed_s':seconds,'start':f[8],'end':f[9],'nodes':1 if active else 0,'gpus':1 if active else 0,'requested_gpus':1,'GPUh':seconds/3600 if active else 0})
    assert {r['job'] for r in allocations}==expected
    ordered=sorted((r for r in allocations if r['gpus']),key=lambda r:r['start']);assert all(a['end']<=b['start'] for a,b in zip(ordered,ordered[1:]))
    accounting={'allocations':allocations,'total_GPUh':sum(r['GPUh'] for r in allocations),'actual_allocated_intervals_serial':True,'maximum_nodes':1,'gpus_per_active_allocation':1,'includes_failed_and_unallocated_cancelled_jobs':True,'GPU_count_evidence':'explicit --gres=gpu:1; code uses current CUDA device; performance asserts one visible device; SlurmDB omits GPU TRES','project_budget_GPUh':500,'append_by_unique_job_ID':True,'project_total_not_recomputed':True,'prior_ledger':str(PREVIOUS/'GPU-ACCOUNTING.json')}
    save(ROOT/'GPU-ACCOUNTING.json',accounting)
    target=ROOT/'snapshots'/('delivery-'+os.environ['SLURM_JOB_ID']);target.mkdir();source=target/'source';source.mkdir()
    shutil.copytree(WS/'ICTC-O2-HP',source/'ICTC-O2-HP',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    controls=source/'controls';controls.mkdir()
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ('.py','.md','.json','.txt','.tsv','.cu','.ptx','.cubin') and p.name!='FINAL-SNAPSHOT.json':shutil.copy2(p,controls/p.name)
    for p in sorted(stages):shutil.copytree(p,source/'stage-snapshots'/p.name)
    shutil.copytree(ROOT/'figures',source/'figures')
    for p in (ROOT/'logs').glob('*.log'):
        if os.environ['SLURM_JOB_ID'] not in p.name:
            d=source/'logs'/p.name;d.parent.mkdir(exist_ok=True);shutil.copy2(p,d)
    raw={}
    for folder in (ROOT/'results').iterdir():
        if not folder.is_dir():continue
        for name in ('MANIFEST.json','CAPTURE.json'):
            path=folder/name
            if not path.exists():continue
            m=read(path)
            for row in m.get('records',[]):
                if 'path' in row and 'sha256' in row:
                    p=Path(row['path']);assert p.is_file();raw[str(p)]={'bytes':p.stat().st_size,'sha256':row['sha256'],'verified_by':'stage manifest and archived AMD receipts; no final tensor rehash'}
            for name,digest in m.get('files',{}).items():
                p=folder/name;assert p.is_file();raw[str(p)]={'bytes':p.stat().st_size,'sha256':digest,'verified_by':'stage manifest and archived AMD receipts; no final tensor rehash'}
    for p in (ROOT/'results').rglob('*'):
        if not p.is_file():continue
        if p.suffix in ('.json','.txt','.csv','.ptx','.cubin') and p.stat().st_size<16*2**20:
            d=source/'receipts'/p.relative_to(ROOT/'results');d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
            raw[str(p)]={'bytes':p.stat().st_size,'sha256':sha(p),'verified_by':'final AMD seal'}
        if p.suffix=='.pt' and str(p) not in raw:raw[str(p)]={'bytes':p.stat().st_size,'sha256':sha(p),'verified_by':'final AMD seal only; incomplete stage has no numerical qualification'}
    for p in Path(gates['PERFORMANCE-AUDIT']['path']).glob('traces/*/SUMMARY.json'):
        r=read(p);q=Path(r['fingerprint']['path']);raw[str(q)]={'bytes':q.stat().st_size,'sha256':r['trace_sha256'],'verified_by':gates['PERFORMANCE-AUDIT']['path']}
    save(target/'RAW-RESULT-MANIFEST.json',raw)
    old=WS/'runs/o2-uvw-triton-a100-20261125-r1';shutil.copytree(old/'snapshots/preflight-119174',source/'external-controls/old-frozen-control')
    shutil.copytree(BASE/'snapshots/qualification-119423/control',source/'external-controls/qualified-v2-control')
    canonical=WS/'architectures/equiformerv3/source/canonical-cpu-control-c14109a12653/canonical.py';shutil.copy2(canonical,source/'external-controls/canonical.py')
    deps={}
    for mode,folder in read(old/'PREFLIGHT.json')['common_states'].items():
        folder=Path(folder);r=read(folder/'CONDITION.json');checkpoint=Path(r['identity']['checkpoint'])
        assert sha(folder/'START.pt')==r['start_sha256'] and sha(checkpoint)==r['identity']['checkpoint_sha256']
        deps[mode]={'state':str(folder/'START.pt'),'state_sha256':r['start_sha256'],'checkpoint':str(checkpoint),'checkpoint_sha256':r['identity']['checkpoint_sha256']}
    deps['frames']={'path':str(old/'prepared/FRAMES.pt'),'sha256':sha(old/'prepared/FRAMES.pt')}
    save(source/'EXTERNAL-DEPENDENCIES.json',{'immutable_inputs':deps,'canonical_source':str(canonical),'canonical_source_sha256':sha(canonical),'raw_manifest':str(target/'RAW-RESULT-MANIFEST.json'),'raw_manifest_sha256':sha(target/'RAW-RESULT-MANIFEST.json'),'scope':'Git, fused sources/plan, control snapshots, receipts and figure included; large tensors/traces/checkpoint/framework remain at hashed workspace paths'})
    for name in ('AGENTS.md','HANDOFF.md','OPERATOR_STATUS.md','architectures/equiformerv3/README.zh-CN.md','architectures/equiformerv3/BOUNDED-BILINEAR-20261126-r1.zh-CN.md'):
        d=source/'workspace-docs'/name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(WS/name,d)
    manifest={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()};save(target/'SOURCE-MANIFEST.json',manifest)
    archive=target/'SOURCE-AND-RECEIPTS.tar.gz'
    with tarfile.open(archive,'w:gz') as t:t.add(source,arcname='source')
    unpack=target/'independent-unpack';unpack.mkdir()
    with tarfile.open(archive) as t:t.extractall(unpack,filter='data')
    assert {str(p.relative_to(unpack/'source')):sha(p) for p in (unpack/'source').rglob('*') if p.is_file()}==manifest
    r={'status':'SEALED_WITH_INDEPENDENT_UNPACK_QA','candidate_commit':(ROOT/'CANDIDATE-HEAD.txt').read_text().strip(),'operator_source_sha256':operator,'package_sha256':package,'job':os.environ['SLURM_JOB_ID'],'snapshot':str(target),'archive':str(archive),'archive_sha256':sha(archive),'source_files':len(manifest),'RAW_manifest':str(target/'RAW-RESULT-MANIFEST.json'),'RAW_manifest_sha256':sha(target/'RAW-RESULT-MANIFEST.json'),'gates':gates,'accounting':accounting,'protected_8324_archive_sha256':baseline['archive_sha256'],'historical_and_new_FAIL_unchanged':True,'shared_loss_cotangents':False,'pushed':False}
    save(ROOT/'FINAL-SNAPSHOT.json',r);save(target/'FINAL-SNAPSHOT.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':main()
