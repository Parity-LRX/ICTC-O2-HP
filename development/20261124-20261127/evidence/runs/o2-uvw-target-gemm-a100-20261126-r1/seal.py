"""Recoverable source, binary and receipt seal; large arrays stay explicitly referenced."""
import ast,hashlib,json,os,shutil,tarfile
from pathlib import Path
from context import ROOT,WS,BASE,package_sha
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
def read(p):return json.loads(Path(p).read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    pre=read(ROOT/'FULL-PREFLIGHT.json');operator=pre['operator_source_sha256'];gates={}
    for pointer,status in {'FULL-AUDIT':'FULL_QUALIFICATION_PASS','RNG-AUDIT':'RNG_PHASE_AND_CONTINUITY_PASS','PTX-AUDIT':'FP32_SM80_PTX_AUDIT_PASS','GEMM-AUDIT':'QUALIFIED_GEMM_MEASUREMENT_AUDIT_PASS','PERFORMANCE-AUDIT':'PERFORMANCE_AUDIT_PASS'}.items():
        folder=Path((ROOT/(pointer+'.txt')).read_text().strip());r=read(folder/'RESULT.json')
        assert r['status']==status and r['operator_source_sha256']==operator
        gates[pointer]={'path':str(folder),'status':status,'sha256':sha(folder/'RESULT.json')}
    package=read(Path(gates['FULL-AUDIT']['path'])/'RESULT.json')['package_sha256']
    assert package_sha(WS/'ICTC-O2-HP/src')==package
    assert not (ROOT/'GIT-STATUS.txt').read_text().strip()
    prior=read(ROOT/'C05-BASELINE-SNAPSHOT.json');assert sha(prior['archive'])==prior['archive_sha256']
    for p in ROOT.glob('*.py'):ast.parse(p.read_text())
    stages=set()
    for p in ROOT.glob('*-PREFLIGHT.json'):
        r=read(p);folder=Path(r['snapshot']);stages.add(folder)
        for name,digest in r['files'].items():assert sha(folder/name)==digest,(p,name)
    trajectory=Path((ROOT/'QUALIFICATION-RESULT.txt').read_text().strip());c=read(trajectory/'CONDITIONS.json');continuity=[]
    for mode in ('legacy','isolated','train-only'):
        for backend in ('native','native-repeat','candidate','candidate-repeat'):
            for step in (1,2):
                now=c[f'{mode}-{backend}-step{step}'];before=c[f'{mode}-{backend}-step{step-1}']
                assert now['before']==before['after']
                assert now['rng']['train' if mode=='train-only' else 'mixed']==before['rng_at_next_step']
                continuity.append({'mode':mode,'backend':backend,'step':step+1})
    save(ROOT/'CONTINUITY-AUDIT.json',{'status':'ALL_24_STATE_AND_RNG_TRANSITIONS_CONTINUOUS','rows':continuity})
    expected={'120158','120168','120188','120247','120253'}
    expected.add(Path((ROOT/'QUALIFICATION-RESULT.txt').read_text().strip()).name.rsplit('-',1)[1])
    for name in ('GEMM-RESULT.txt','PERFORMANCE-RESULT.txt'):expected.add(Path((ROOT/name).read_text().strip()).name.rsplit('-',1)[1])
    allocations=[]
    for line in (ROOT/'SACCT-ALL.tsv').read_text().splitlines():
        f=line.split('|')
        if f[0] not in expected:continue
        job,name,state,exitcode,elapsed,tres,nodes,node=f[:8]
        assert int(nodes)==1 and node=='XHPC-A100' and state in ('COMPLETED','FAILED','CANCELLED')
        allocations.append({'job':job,'name':name,'state':state,'exit_code':exitcode,'elapsed_s':int(elapsed),'start':f[8],'end':f[9],'nodes':1,'gpus':1,'GPUh':int(elapsed)/3600})
    assert {v['job'] for v in allocations}==expected
    ordered=sorted(allocations,key=lambda r:r['start']);assert all(a['end']<=b['start'] for a,b in zip(ordered,ordered[1:]))
    accounting={'allocations':allocations,'total_GPUh':sum(r['GPUh'] for r in allocations),'actual_scheduler_intervals_serial':True,'maximum_nodes':1,'gpus_per_allocation':1,'project_budget_GPUh':500,
        'GPU_count_evidence':'explicit --gres=gpu:1 and runtime single-device assertions; site SlurmDB omits GPU TRES','append_by_unique_job_ID':True,'project_total_not_recomputed':True}
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
    for p in (ROOT/'results').rglob('*'):
        if not p.is_file():continue
        if p.suffix in ('.json','.txt','.csv','.ptx','.cubin') and p.stat().st_size<16*2**20:
            d=source/'receipts'/p.relative_to(ROOT/'results');d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
            raw[str(p)]={'bytes':p.stat().st_size,'sha256':sha(p),'verified_by':'final AMD seal'}
    for kind,name in (('REGRESSION','REGRESSION.json'),('GENERIC','MANIFEST.json')):
        for row in read(Path((ROOT/(kind+'-RESULT.txt')).read_text().strip())/name)['records']:
            p=Path(row['path']);assert p.is_file();raw[str(p)]={'bytes':p.stat().st_size,'sha256':row['sha256'],'verified_by':gates['FULL-AUDIT']['path']}
    for name,digest in read(trajectory/'MANIFEST.json')['files'].items():
        p=trajectory/name;raw[str(p)]={'bytes':p.stat().st_size,'sha256':digest,'verified_by':gates['FULL-AUDIT']['path']}
    micro=Path((ROOT/'GEMM-RESULT.txt').read_text().strip())
    for r in read(micro/'RESULT.json')['rows']:
        p=Path(r['path']);raw[str(p)]={'bytes':p.stat().st_size,'sha256':r['sha256'],'verified_by':gates['GEMM-AUDIT']['path']}
    for p in Path(gates['PERFORMANCE-AUDIT']['path']).glob('traces/*/SUMMARY.json'):
        r=read(p);q=Path(r['fingerprint']['path']);raw[str(q)]={'bytes':q.stat().st_size,'sha256':r['trace_sha256'],'verified_by':gates['PERFORMANCE-AUDIT']['path']}
    save(target/'RAW-RESULT-MANIFEST.json',raw)
    old=WS/'runs/o2-uvw-triton-a100-20261125-r1'
    shutil.copytree(old/'snapshots/preflight-119174',source/'external-controls/old-frozen-control')
    shutil.copytree(BASE/'snapshots/qualification-119423/control',source/'external-controls/qualified-v2-control')
    deps={}
    for mode,folder in read(old/'PREFLIGHT.json')['common_states'].items():
        folder=Path(folder);r=read(folder/'CONDITION.json');checkpoint=Path(r['identity']['checkpoint'])
        assert sha(folder/'START.pt')==r['start_sha256'] and sha(checkpoint)==r['identity']['checkpoint_sha256']
        deps[mode]={'state':str(folder/'START.pt'),'state_sha256':r['start_sha256'],'checkpoint':str(checkpoint),'checkpoint_sha256':r['identity']['checkpoint_sha256']}
    deps['frames']={'path':str(old/'prepared/FRAMES.pt'),'sha256':sha(old/'prepared/FRAMES.pt')}
    save(source/'EXTERNAL-DEPENDENCIES.json',{'immutable_inputs':deps,'raw_manifest':str(target/'RAW-RESULT-MANIFEST.json'),'raw_manifest_sha256':sha(target/'RAW-RESULT-MANIFEST.json'),'scope':'Sources, binary, Git and receipts included; large tensors and installed framework/environment remain at hashed workspace paths'})
    for name in ('AGENTS.md','HANDOFF.md','OPERATOR_STATUS.md','architectures/equiformerv3/README.zh-CN.md','architectures/equiformerv3/TARGET-FMA8-20261126-r1.zh-CN.md'):
        d=source/'workspace-docs'/name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(WS/name,d)
    manifest={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()};save(target/'SOURCE-MANIFEST.json',manifest)
    archive=target/'SOURCE-AND-RECEIPTS.tar.gz'
    with tarfile.open(archive,'w:gz') as t:t.add(source,arcname='source')
    unpack=target/'independent-unpack';unpack.mkdir()
    with tarfile.open(archive) as t:t.extractall(unpack,filter='data')
    assert {str(p.relative_to(unpack/'source')):sha(p) for p in (unpack/'source').rglob('*') if p.is_file()}==manifest
    result={'status':'SEALED_WITH_INDEPENDENT_UNPACK_QA','candidate_commit':(ROOT/'CANDIDATE-HEAD.txt').read_text().strip(),'operator_source_sha256':operator,'package_sha256':package,'job':os.environ['SLURM_JOB_ID'],'snapshot':str(target),'archive':str(archive),'archive_sha256':sha(archive),'source_files':len(manifest),'RAW_manifest':str(target/'RAW-RESULT-MANIFEST.json'),'RAW_manifest_sha256':sha(target/'RAW-RESULT-MANIFEST.json'),'gates':gates,'accounting':accounting,'c05_preserved':prior,'historical_FAIL_unchanged':True,'shared_cotangents':False,'pushed':False}
    save(ROOT/'FINAL-SNAPSHOT.json',result);save(target/'FINAL-SNAPSHOT.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
