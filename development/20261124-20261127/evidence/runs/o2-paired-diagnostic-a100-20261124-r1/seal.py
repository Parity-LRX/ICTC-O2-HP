"""CPU delivery: validate paired receipts, account GPU allocations, seal artifacts."""
import csv,hashlib,importlib.metadata,json,os,re,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[1]
assert os.environ.get('SLURM_JOB_ID')
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
analysis=ROOT/'results/analysis-118587'
status=read(analysis/'ANALYSIS.json');checks=read(analysis/'PAIRED-CHECKS.json')
assert status['paired_pass']==6 and all(c['paired_conditions_pass'] for c in checks)
with (analysis/'MEASUREMENTS.csv').open() as f:measurements=list(csv.DictReader(f))
primary=[r for r in measurements if r['route'] in ('v3','tece-canonical','tece-full') and r['backend'] in ('native','ictc')]
assert len(primary)==12 and all(r['median_s'] and r['peak_allocated_GiB'] and r['kernel_calls'] for r in primary)
fast_checks=read(analysis/'FAST-PAIRED-CHECKS.json')
for extension in ('.json','.csv'):shutil.copy2(analysis/('FAST-PAIRED-CHECKS'+extension),analysis/('FAST-PAIRED-CHECKS-initial'+extension))
for row in fast_checks:
    row['status']='MATCHED_COMPLETE' if row['paired_pass'] else 'UNSUPPORTED_PACKED_WIGNER_MODULE_INTERFACE'
    if row['successful_actions']!=6:
        for key in list(row):
            if key.endswith('_identical'):row[key]=None
        row['coverage_calls']=None
save(analysis/'FAST-PAIRED-CHECKS.json',fast_checks)
with (analysis/'FAST-PAIRED-CHECKS.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(fast_checks[0]));writer.writeheader();writer.writerows(fast_checks)
allocation_ids=('118576','118586','118594','118630')
allocations=[]
with (ROOT/'GPU-ACCOUNTING.txt').open() as f:
    for row in csv.DictReader(f,delimiter='|'):
        if row['JobIDRaw'] not in allocation_ids:continue
        assert row['NodeList']=='XHPC-A100' and int(row['NNodes'])==1
        allocations.append(dict(row,gpus=1,gpu_h=int(row['ElapsedRaw'])/3600))
assert len(allocations)==len(allocation_ids)
ordered=sorted(allocations,key=lambda r:r['Start'])
assert all(a['End']<=b['Start'] for a,b in zip(ordered,ordered[1:])),'GPU allocations overlap'
gpu_h=sum(r['gpu_h'] for r in allocations)
for allocation in allocation_ids:
    seals=read(ROOT/'results'/('a100-'+allocation)/'SOURCE-SEAL.json')
    for name,expected in seals.items():assert sha(ROOT/'snapshots'/('control-'+allocation)/name)==expected
graph_audit=[]
for row in read(analysis/'ALL-RECEIPTS.json'):
    if row['action']!='trace' or row['status']!='DIAGNOSTIC_COMPLETED':continue
    trace=Path(row['folder'])/'trace.json'
    with trace.open('rb') as f:payload=f.read()
    calls=re.findall(rb'"name"\s*:\s*"((?:cuda|cu)(?:GraphLaunch|GraphInstantiate|StreamBeginCapture|StreamEndCapture)[^"]*)"',payload)
    graph_audit.append({'route':row['route'],'mode':row['mode'],'backend':row['backend'],'capture_or_graph_launch_calls':len(calls),'trace':str(trace)})
    assert not calls,('CUDA Graph observed in a forbidden-Graph diagnostic',str(trace),calls[:3])
    del payload
with (ROOT/'TRACE-CUDA-GRAPH-AUDIT.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(graph_audit[0]));writer.writeheader();writer.writerows(graph_audit)
receipt={'status':'PAIRED_DIAGNOSTIC_DELIVERED','cpu_job':os.environ['SLURM_JOB_ID'],'analysis':str(analysis),
    'main_workloads':12,'paired_modes':6,'paired_conditions_pass':6,'gpu_allocations':allocations,'gpu_h_this_run':gpu_h,
    'extra_accelerated_paired_modes_complete':sum(r['paired_pass'] for r in fast_checks),'extra_accelerated_paired_modes_unavailable':sum(not r['paired_pass'] for r in fast_checks),
    'unavailable_scope':'TECE canonical module route with accelerated packed Wigner fails before ECE; native/candidate same RuntimeError',
    'one_physical_node':'XHPC-A100','gpus_per_allocation':1,'serial_allocations_verified':True,
    'qualification':'DIAGNOSTIC_ONLY; historical FAIL/HOLD retained; no new numerical acceptance',
    'next_steps':'STOP pending user distribution and condition review',
    'prior_accounting':'previous whole-network run 0.671667 GPUh; stage1-local upper 0.015833 GPUh; archived upper123.370278; not a refreshed global balance'}
save(ROOT/'FINAL-RECEIPT.json',receipt)
environment={'python':sys.version,'executable':sys.executable,'packages':{}}
for name in ('torch','triton','e3nn','torch-geometric','torch-scatter','numpy','scipy','ase','matplotlib','torch-ema','omegaconf'):
    try:environment['packages'][name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:environment['packages'][name]='not installed as distribution; see pinned source/vendor'
save(ROOT/'ENVIRONMENT.json',environment)
lines=['# 三路线原生/ICTC诊断热点：第一步完成','',
       '已补齐V3 canonical、TECE canonical、TECE full parity的E/F推理与完整训练，以及每组Torch原生对照。主矩阵六组配对条件检查通过；数值FAIL/HOLD保留，未启动第二、第三步。','',
       f"本轮{len(allocations)}次allocation均为XHPC-A100单节点单GPU串行，共{gpu_h:.6f} GPUh（包含停止及所有异常）。",'',
       '|路线|模式|同步墙钟中位数 Torch原生 / ICTC (s)|峰值allocated Torch原生 / ICTC (GiB)|kernel时间和 Torch原生 / ICTC (ms)|',
       '|---|---|---:|---:|---:|']
index={(r['route'],r['mode'],r['backend']):r for r in primary}
for route in ('v3','tece-canonical','tece-full'):
    for mode in ('ef','train'):
        native=index[(route,mode,'native')];ictc=index[(route,mode,'ictc')]
        pairs=[' / '.join(f"{float(r[k]):.3f}" for r in (native,ictc)) for k in ('median_s','peak_allocated_GiB','kernel_ms')]
        lines.append(f"|{route}|{mode}|{pairs[0]}|{pairs[1]}|{pairs[2]}|")
relative='results/analysis-118587'
lines+=['','推理600原子，训练100原子，均batch=1。训练逐样本恢复同一个原生3步预热后的完整状态，恢复不计时。time/memory/trace为独立进程，3次代码预热、10次耗时采样、1次trace。kernel时间和按实际GPU事件统计，不能当同步墙钟；memcpy/memset单列。显存包含统一harness常驻对象。','',
        f'![kernel分布]({relative}/kernel-distribution.png)','',f'![耗时与显存]({relative}/time-memory.png)','',
        f'[详细报告、原始V3与可选Eqx对照、完整分布入口]({relative}/REPORT.zh-CN.md) · [可搜索全kernel表]({relative}/inventory.html) · [原始10次耗时]({relative}/RAW-TIMES.csv) · [配对条件CSV]({relative}/PAIRED-CHECKS.csv) · [完整原生配置]({relative}/MODEL-CONFIGS.json)','',
        f'![原生加速及相同ECE切口]({relative}/native-fast-comparison.png)','',
        '另列上游原生融合Eqx路径、保留Eqx的原生module route与相同module route的ICTC。主表的纯Torch数学参考不代替上游融合后端的整网性能参考；见详细报告及FAST-PAIRED-CHECKS。','',
        '额外同加速切口：full parity推理/训练两组条件一致并确认实际ICTC调用；canonical两组在ECE之前遇到packed Wigner与dense bmm接口不兼容，标为不可运行。未完成项的比较字段留空，不表示已测得不一致，也不填零耗时/吞吐。','',
        '本轮按公共状态回放。恢复本身不计时；派生权重布局缓存失效后若在call内重建，重建成本计入。本轮不声称持续缓存部署时的正式速度。全部kernel按实际事件完整统计；前向模块stage无法覆盖全部反向线程，未归因比例在CSV单列，不将其猜测为geometry。','',
        '[执行条件](CONDITIONS.zh-CN.md) · [资格边界](QUALIFICATION-BOUNDARY.zh-CN.md) · [实际GPU账目](GPU-ACCOUNTING.txt) · [实际trace禁Graph核查](TRACE-CUDA-GRAPH-AUDIT.csv) · [最终回执](FINAL-RECEIPT.json)','',
        '同路线内严格配对；不同路线保留原生架构/图/loss/optimizer/stress差异，跨架构只列绝对成本。V3 canonical是独立投影模型，原始上游V3另列绝对成本；TECE full为固定随机初始化结构性能。共享节点未锁时钟，配对核对物理GPU UUID。','',
        '完成分布后停止，等待用户审阅条件和分布，再决定后续工作。']
(ROOT/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
delivery=ROOT/'snapshots'/('delivery-'+os.environ['SLURM_JOB_ID']);delivery.mkdir(exist_ok=False)
selected=[]
selected.extend(p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.sbatch','.md','.json','.txt','.csv'))
selected.extend(p for p in (ROOT/'prepared').iterdir() if p.is_file() and p.suffix in ('.pt','.json'))
selected.extend(p for p in (ROOT/'logs').iterdir() if p.is_file() and not p.name.endswith('-'+os.environ['SLURM_JOB_ID']+'.log'))
selected.extend(p for p in analysis.rglob('*') if p.is_file())
for allocation in allocation_ids:
    selected.extend(p for p in (ROOT/'snapshots'/('control-'+allocation)).rglob('*') if p.is_file())
    selected.extend(p for p in (ROOT/'results'/('a100-'+allocation)).rglob('*') if p.is_file() and p.name in
        ('SOURCE-SEAL.json','STARTUP.json','RESUME.json','PROGRESS.json','FINISH.json','CONDITION.json','RESULT.json','ERROR.json','START.pt','GRAPH.pt','process.log','trace.json','OPERATOR-TABLE.txt'))
for source in sorted(set(selected)):
    target=delivery/'run'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
    if source.stat().st_size<=4*1024*1024:shutil.copy2(source,target)
    else:os.link(source,target)
for source in (ROOT/'vendor').resolve().rglob('*'):
    if source.is_file():
        target=delivery/'vendor'/source.relative_to((ROOT/'vendor').resolve());target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
checkpoints={}
for row in read(analysis/'MODEL-CONFIGS.json'):
    identity=row['identity'];source=Path(identity.get('checkpoint',''))
    if not source.is_file() or str(source) in checkpoints:continue
    observed=sha(source);assert observed==identity['checkpoint_sha256']
    target=delivery/'checkpoints'/(observed[:12]+'-'+source.name);target.parent.mkdir(parents=True,exist_ok=True);os.link(source.resolve(),target)
    checkpoints[str(source)]={'sha256':observed,'snapshot':str(target.relative_to(delivery))}
save(delivery/'CHECKPOINTS.json',checkpoints)
reference=read(WS/'runs/o2-three-route-exec-20261124-r1/results/preflight-118331/SOURCE-SEALS.json')
for role in ('operator','v3_canonical','tece_canonical','tece_full'):
    row=reference[role]
    for name,expected in row['files'].items():
        source=Path(row['root'])/name;target=delivery/'dependencies'/role/name;target.parent.mkdir(parents=True,exist_ok=True)
        if 'sha256' in expected:
            assert sha(source)==expected['sha256'];os.link(source,target)
        else:
            assert source.is_symlink() and os.readlink(source)==expected['symlink'];target.symlink_to(expected['symlink'])
manifest={}
for p in sorted(delivery.rglob('*')):
    name=str(p.relative_to(delivery))
    if p.is_symlink():manifest[name]={'symlink':os.readlink(p)}
    elif p.is_file():manifest[name]={'size':p.stat().st_size,'sha256':sha(p)}
save(delivery/'SHA-MANIFEST.json',manifest)
final={'snapshot':str(delivery),'payload_files':len(manifest),'manifest_sha256':sha(delivery/'SHA-MANIFEST.json'),
       'gpu_h_this_run':gpu_h,'main_workloads':12,'paired_conditions_pass':6}
save(ROOT/'FINAL-SNAPSHOT.json',final);print(json.dumps(final),flush=True)
