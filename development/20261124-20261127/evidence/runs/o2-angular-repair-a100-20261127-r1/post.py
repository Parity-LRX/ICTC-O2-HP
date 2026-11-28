"""Automatic AMD median audit, complete coverage, figures and recoverable receipts."""
import csv,hashlib,json,math,os,shutil,statistics,subprocess,tarfile
from pathlib import Path
ROOT=Path(os.environ['O2_SWEEP_ROOT']);WS=ROOT.parents[1]
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
def read(p,default=None):return json.loads(Path(p).read_text()) if Path(p).exists() else default
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False,default=str)+'\n')
def csvwrite(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
submissions=read(ROOT/'SUBMISSIONS.json');job=str(submissions['measurement']);DATA=ROOT/'measurements'/job
candidate=read(ROOT/'SOURCE.json')['candidate_commit'];short=candidate[:7]
records=read(DATA/'MEASUREMENTS.json',[]);capacities=read(DATA/'CAPACITY.json',[]);manifest=read(DATA/'MANIFEST.json',{})
rows=[];inputs={};errors=[];devices=set()
for item in records:
    config=item['configuration'];row={**config,'action':item['action'],'status':item['status']}
    trials=item.get('trials',[])
    if trials:
        assert len(trials)==3
        for t in trials:
            path=Path(t['folder'])/'RESULT.json'
            if path.exists():inputs[str(path)]=sha(path)
            if t['status']=='PASS':devices.add(t['result']['hardware']['uuid'])
    if item['status']=='PASS':
        values=[t['result']['elapsed_s'] for t in trials];assert len(values)==3 and all(v>0 for v in values)
        median=statistics.median(values);assert median==item['median_s']
        row.update(median_s=median,throughput=statistics.median(t['result']['throughput'] for t in trials),
                   allocated_GiB=statistics.median(t['result']['peak_bytes']['allocated'] for t in trials)/2**30,
                   reserved_GiB=statistics.median(t['result']['peak_bytes']['reserved'] for t in trials)/2**30,
                   run0_s=values[0],run1_s=values[1],run2_s=values[2])
    rows.append(row)
assert len(devices)<=1,devices
csvwrite(ROOT/'MEASUREMENTS.csv',rows)
capacity_rows=[]
for c in capacities:
    row={**c['configuration'],**{k:v for k,v in c.items() if k!='configuration'}}
    if row['status']=='STABLE_CAPACITY_3OF3':
        assert row['maximum_stable_size']<row['OOM_upper_size'] and row['OOM_upper_size']-row['maximum_stable_size']<=row['resolution']
        row['resident_units']=row['maximum_stable_size']*(100 if row['level']=='network' else 1)
        if row['level']=='network':row['resident_edges']=5252*row['maximum_stable_size']
    capacity_rows.append(row)
csvwrite(ROOT/'CAPACITY.csv',capacity_rows)
families=[('uvw','canonical'),('uvw','full'),('bilinear','canonical')]
def find(level,family,parity,order,mode,size):
    return next((r for r in rows if r['action']=='timing' and r['level']==level and r.get('family')==family and r.get('parity')==parity and r['order']==order and r['mode']==mode and r['size']==size),None)
def cap(level,family,parity,order,mode):
    return next((r for r in capacity_rows if r['level']==level and r.get('family')==family and r.get('parity')==parity and r['order']==order and r['mode']==mode),None)
coverage=[]
for family,parity in families:
    for order in range(1,9):
        for mode in ('infer','train'):
            for size in (256,4096):
                r=find('operator',family,parity,order,mode,size);coverage.append({'level':'operator','family':family,'parity':parity,'order':order,'mode':mode,'size':size,'status':r['status'] if r else 'NOT_RUN'})
for order in [0,*range(1,9)]:
    for mode in ('infer','train'):
        r=find('network',None,None,order,mode,1);coverage.append({'level':'network','order':order,'mode':mode,'size':1,'status':r['status'] if r else 'NOT_RUN'})
csvwrite(ROOT/'COVERAGE.csv',coverage);save(ROOT/'INPUT-MANIFEST.json',inputs)
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
figures=ROOT/'figures';figures.mkdir(exist_ok=True)
styles=[('uvw','canonical','#4477AA','o'),('uvw','full','#CC6677','^'),('bilinear','canonical','#228833','s')]
with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'}):
    for level in ('operator','network'):
        fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
        for i,mode in enumerate(('infer','train')):
            series=styles if level=='operator' else [(None,None,'#4477AA','o')]
            for family,parity,color,marker in series:
                label=(family+' / '+parity) if level=='operator' else 'L=M; fixed-seed structural model'
                xs=[[],[],[]];ys=[[],[],[]]
                for order in range(1,9):
                    for col,size in ((0,4096 if level=='operator' else 1),(1,256 if level=='operator' else 1)):
                        r=find(level,family,parity,order,mode,size)
                        if r and r['status']=='PASS':
                            xs[col].append(order);ys[col].append(r['throughput'] if col==0 else 1000*r['median_s'])
                            for j in range(3):
                                raw=(size if level=='operator' else 100)/r[f'run{j}_s'] if col==0 else 1000*r[f'run{j}_s']
                                axes[i,col].scatter(order+(j-1)*.055,raw,s=9,color=color,alpha=.4)
                        else:
                            xs[col].append(order);ys[col].append(float('nan'))
                            axes[i,col].plot([order],[.025],marker='x',color=color,transform=axes[i,col].get_xaxis_transform())
                    c=cap(level,family,parity,order,mode)
                    if c and c['status']=='STABLE_CAPACITY_3OF3':
                        xs[2].append(order);ys[2].append(c['resident_units'])
                        upper=c['OOM_upper_size']*(100 if level=='network' else 1)
                        axes[i,2].vlines(order,c['resident_units'],upper,color=color,lw=1.2,alpha=.7)
                        axes[i,2].scatter([order],[upper],marker='v',s=24,facecolors='none',edgecolors=color)
                    else:xs[2].append(order);ys[2].append(float('nan'))
                for col in range(3):axes[i,col].plot(xs[col],ys[col],color=color,marker=marker,label=label)
            if level=='network':
                for col in range(3):
                    r=find(level,None,None,0,mode,1) if col<2 else cap(level,None,None,0,mode)
                    if r and r['status'] in ('PASS','STABLE_CAPACITY_3OF3'):
                        value=r['throughput'] if col==0 else 1000*r['median_s'] if col==1 else r['resident_units']
                        axes[i,col].scatter([4],[value],marker='*',s=150,facecolors='none',edgecolors='#CC6677',label='Original L4/M2 checkpoint anchor')
            for col,ax in enumerate(axes[i]):
                ax.set(xticks=list(range(1,9)),xlim=(.7,8.3),xlabel='m_max' if level=='operator' else 'l_max = m_max')
                if any(math.isfinite(v) and v>0 for line in ax.lines for v in line.get_ydata()):
                    ax.set_yscale('log',base=10)
                    ax.text(.98,.02,'log10 y',ha='right',transform=ax.transAxes,fontsize=8)
                ax.grid(alpha=.18);ax.set_title(mode+' | '+('throughput' if col==0 else 'latency' if col==1 else 'stable capacity, 3/3 fit'))
                ax.set_ylabel(('rows/s' if level=='operator' else 'atoms/s at 100 atoms') if col==0 else 'ms' if col==1 else 'resident rows' if level=='operator' else 'resident atoms, 100-atom batches')
                ax.text(-.1,1.03,chr(65+3*i+col),transform=ax.transAxes,fontweight='bold')
        axes[0,0].legend(fontsize=8,frameon=False)
        fig.suptitle(short+' | '+level+' angular sweep | A100 40G, one GPU, serial\n3 fresh processes; dots: runs, line: median; capacity triangles: next OOM (not CI); gaps: unavailable',fontsize=12)
        for ext in ('png','pdf','svg'):fig.savefig(figures/(level+'-angular-throughput-latency-capacity.'+ext),dpi=160,facecolor='white')
        plt.close(fig)
status='COMPLETE_WITH_RECORDED_HOLDS' if manifest.get('status')=='ALL_MEASUREMENT_ATTEMPTS_COMPLETE' else 'INCOMPLETE_EXECUTION'
doc=[f'# {short}修复后角向阶数、延迟/吞吐与A100 40G容量','',f'状态：{status}。候选{candidate}；原dd7a3db/8324及全部旧FAIL/封存保持。每个成功配置均有三个独立进程并重新核对中位数。AMD完整导数及各自真实loss三步通过后才计量，验收见GATE-AUDIT.txt和LOCAL-AUDIT.txt。',
 '', '算子：UVW U=V=W=32、canonical/full；Gaunt双线性H=32。推理为forward，训练为forward+完整输入/权重VJP，所需混合二阶另验。固定N256测延迟、N4096测吞吐。它们是不同算子方程，只比较绝对成本。',
 '', '整网：7层/128通道、l=m=1..8，固定seed独立结构模型和统一20×20系数生成网格；不是原checkpoint任务准确率。原l4/m2 checkpoint单列星标。固定100原子真实帧、原图/PBC/重复边/标签，推理含E/F/stress，训练含原loss、clip/AdamW/EMA/scheduler及真实力损失混合二阶。',
 '', '精度和原门不变：能量输出/求和FP64，其他数学/累加/导数/误差FP32，禁TF32/AMP/half/CUDA Graph。数值失败、OOM、构建错误、超时及未执行分别保留，未给缺失项填零。详见COVERAGE.csv与AMD GATE-AUDIT回执。',
 '', '容量指数搜索+二分，每个探测规模三个新进程，最大3/3成功规模及下一实际OOM上界在CAPACITY.csv。整网按100原子一图精确搜索；算子保留实际行数分辨率，通常约初始OOM上界1/32。容量是完整执行边界，不是所有N的新数值资格证明；不同batch容量边界吞吐另列，不与固定N混淆。',
 '', '计时两次预热，容量一次预热，正式调用前在计量外恢复公共状态并清除无引用缓存；真实三步验收内不恢复状态/共享loss余切。图上点为三个进程值，中位数折线不代表独立科学样本置信区间。纵轴为log10比例尺，仅显示正测量值；不可用配置留缺口，原因分类见表。容量竖线和空心下三角表示3/3稳定规模至下一实际OOM的搜索区间，不是统计误差条。',
 '', f'![算子阶数扫描]({figures/"operator-angular-throughput-latency-capacity.png"})',
 '', f'![整网阶数扫描]({figures/"network-angular-throughput-latency-capacity.png"})',
 '', '|层级|族/宇称|阶数|模式|N或图数|状态|中位延迟ms|吞吐/s|allocated GiB|reserved GiB|','|---|---|---:|---|---:|---|---:|---:|---:|---:|']
for c in coverage:
    r=find(c['level'],c.get('family'),c.get('parity'),c['order'],c['mode'],c['size']) or c
    fmt=lambda k,scale=1:f'{r[k]*scale:.6g}' if k in r else '—'
    doc.append(f'|{c["level"]}|{c.get("family", "network")}/{c.get("parity", "canonical")}|{c["order"] if c["order"] else "4,m2 anchor"}|{c["mode"]}|{c["size"]}|{r["status"]}|{fmt("median_s",1000)}|{fmt("throughput")}|{fmt("allocated_GiB")}|{fmt("reserved_GiB")}|')
doc+=['','全部原始三次值见MEASUREMENTS.csv；每个试次RESULT/日志路径和原始结果均在measurements/。每个容量点上下界、失败状态及边界吞吐见CAPACITY.csv。',
 '', '## 流程参考','', '制图遵循scientific-visualization技能：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. [当前arXiv记录](https://arxiv.org/abs/2609.00065)。2026-11-27核对当前v2，无期刊版本。该引用说明制图流程来源，不支持本工作区性能数字。内部分析图，不声明投稿规范或自动可访问性认证。']
(ROOT/'REPORT.zh-CN.md').write_text('\n'.join(doc)+'\n')
gpu_ids=[str(v) for v in submissions['GPU_jobs']]
account=(ROOT/'ACCOUNTING-RECEIPT.tsv').read_text();alloc=[]
(ROOT/'SACCT-GPU.tsv').write_text(account)
if account:
    for line in account.splitlines():
        f=line.split('|')
        if f[0] in gpu_ids:alloc.append({'job':f[0],'state':f[1],'exit_code':f[2],'elapsed_s':int(f[3]),'nodes':f[4],'node':f[5],'start':f[6],'end':f[7],'GPUh':int(f[3])/3600})
save(ROOT/'GPU-ACCOUNTING.json',{'allocations':alloc,'total_GPUh':sum(r['GPUh'] for r in alloc),'one_GPU_per_allocation':True,'project_budget_GPUh':500,'prior_ledger':str(WS/'runs/o2-angular-cause-20261127-r1/GPU-ACCOUNTING.json'),'scheduler_receipt_from_login_control_plane':True})
delivery=ROOT/'snapshots'/('delivery-'+os.environ['SLURM_JOB_ID']);delivery.mkdir(parents=True);payload=delivery/'source';payload.mkdir()
for p in ROOT.iterdir():
    if p.is_file() and p.suffix in ('.py','.md','.json','.csv','.txt','.tsv'):shutil.copy2(p,payload/p.name)
shutil.copytree(figures,payload/'figures')
for p in (ROOT/'models').glob('canonical-L*.py'):
    q=payload/'model-references';q.mkdir(exist_ok=True);shutil.copy2(p,q/p.name)
files={str(p.relative_to(payload)):sha(p) for p in payload.rglob('*') if p.is_file()}
archive=delivery/'SOURCE-REPORT-RECEIPTS.tar.gz'
with tarfile.open(archive,'w:gz') as t:t.add(payload,arcname='source')
unpack=delivery/'independent-unpack';unpack.mkdir()
with tarfile.open(archive) as t:t.extractall(unpack,filter='data')
assert {str(p.relative_to(unpack/'source')):sha(p) for p in (unpack/'source').rglob('*') if p.is_file()}==files
final={'status':status,'candidate_commit':candidate,'AMD_job':os.environ['SLURM_JOB_ID'],'measurement_job':job,'measurements':len(rows),'capacity_targets':len(capacity_rows),'coverage_status_counts':{s:sum(r['status']==s for r in coverage) for s in set(r['status'] for r in coverage)},'archive':str(archive),'archive_sha256':sha(archive),'independent_unpack_QA':True,'source_files':files,'large_inputs':'model/qualification tensors and raw trial directories stay at hashed workspace paths recorded in SOURCE/MODELS/gate/measurement manifests','GPUh':sum(r['GPUh'] for r in alloc)}
save(ROOT/'FINAL-SNAPSHOT.json',final);print(json.dumps({k:v for k,v in final.items() if k!='source_files'}),flush=True)
