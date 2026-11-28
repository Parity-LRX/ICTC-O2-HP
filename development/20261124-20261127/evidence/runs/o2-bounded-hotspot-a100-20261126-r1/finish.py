"""Generate the absolute hotspot figure and evidence-bounded Chinese report."""
import ast,collections,csv,hashlib,json,math,os,shutil
from pathlib import Path
ROOT=Path(os.environ['O2_HOT_ROOT']);WS=ROOT.parents[1];OLD=WS/'runs/o2-bounded-bilinear-a100-20261126-r1'
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
DATA=Path((ROOT/'ANALYSIS-RESULT.txt').read_text().strip());RESULT=json.loads((DATA/'RESULT.json').read_text())
assert RESULT['status']=='REUSED_TRACE_HOTSPOT_ANALYSIS_PASS'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvrows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
capsule=ROOT/'snapshots'/('delivery-'+os.environ['SLURM_JOB_ID']);capsule.mkdir(parents=True)
for p in ROOT.glob('*.py'):ast.parse(p.read_text());shutil.copy2(p,capsule/p.name)
shutil.copy2(ROOT/'CONTRACT.zh-CN.md',capsule/'CONTRACT.zh-CN.md')
AUDIT=Path(RESULT['source_audit']);old=read(AUDIT/'RESULT.json');work={(r['mode'],r['backend']):r for r in old['workloads']}
assert sha(AUDIT/'RESULT.json')==RESULT['inputs'][str(AUDIT/'RESULT.json')]
categories=['GEMM/GEMV','ICTC channel GEMM','ICTC bilinear','ICTC weight gradient','copy/layout','elementwise','index/scatter/gather','fill/zero','reduction/norm','unclassified kernel']
detail=[]
for mode in ('ef','train'):
    for backend in ('native','candidate'):
        amounts=collections.defaultdict(float);calls=collections.Counter()
        for r in RESULT['families']:
            if (r['mode'],r['backend'])!=(mode,backend) or r['family']=='ICTC GPU':continue
            amounts[r['family']]+=r['total_ms'];calls[r['family']]+=r['calls']
        for r in RESULT['ICTC_kernels']:
            if (r['mode'],r['backend'])!=(mode,backend):continue
            name=r['kernel'];label='ICTC bilinear' if name=='_bilinear' else 'ICTC channel GEMM' if name=='_uvw_mm' else 'ICTC weight gradient'
            amounts[label]+=float(r['total_ms']);calls[label]+=int(r['calls'])
        total=work[mode,backend]['kernel_ms'];assert math.isclose(sum(amounts.values()),total,rel_tol=1e-10)
        assert sum(calls.values())==work[mode,backend]['kernel_calls']
        detail.extend({'mode':mode,'backend':backend,'category':k,'kernel_calls':calls[k],'total_ms':amounts[k],'percent':100*amounts[k]/total} for k in categories)
csvwrite(ROOT/'HOTSPOTS.csv',detail)
grouped=collections.defaultdict(lambda:[0,0.0]);roles=collections.defaultdict(lambda:[0,0.0])
for r in csvrows(DATA/'MATRIX-SHAPES.csv'):
    key=(r['mode'],r['backend'],r['operator'],r['input_dims']);grouped[key][0]+=int(r['kernel_calls']);grouped[key][1]+=float(r['total_ms'])
    if r['operator']!='aten::bmm':continue
    dims=json.loads(r['input_dims']);edges=31512 if r['mode']=='ef' else 5252
    label='other BMM shapes'
    if isinstance(dims,list) and len(dims)>=2 and len(dims[0])==len(dims[1])==3:
        a,b=dims[:2]
        if a[0]==b[0]==edges and any(v in (19,25) for v in a[1:]+b[1:]):label='edge rotation shape family'
        elif a[0]==b[0] and 2 in a[1:] and 2 in b[1:] and edges in a[1:]+b[1:]:label='canonical 2x2 basis shape family'
        elif a[0]==b[0]==1 and (a[1:] in ([19,361],[25,625]) or b[1:] in ([19,361],[25,625])):label='old angular bilinear BMM'
    roles[(r['mode'],r['backend'],label)][0]+=int(r['kernel_calls']);roles[(r['mode'],r['backend'],label)][1]+=float(r['total_ms'])
matrix=[{'mode':k[0],'backend':k[1],'operator':k[2],'input_dims':k[3],'kernel_calls':v[0],'total_ms':v[1]} for k,v in sorted(grouped.items(),key=lambda kv:-kv[1][1])]
role_rows=[{'mode':k[0],'backend':k[1],'shape_family':k[2],'kernel_calls':v[0],'total_ms':v[1]} for k,v in sorted(roles.items())]
csvwrite(ROOT/'MATRIX-SHAPES-MERGED.csv',matrix);csvwrite(ROOT/'BMM-SHAPE-FAMILIES.csv',role_rows)

os.environ['MPLCONFIGDIR']=str(ROOT/'mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
figures=ROOT/'figures';figures.mkdir(exist_ok=True)
with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'}):
    fig,axes=plt.subplots(2,2,figsize=(17.5,11),layout='constrained')
    phase_keys=[['model_forward','force_from_energy','graph_build_transfer','unattributed'],['model_forward','force_from_energy','loss_backward','loss','clip','optimizer','EMA','graph_build_transfer']]
    for i,mode in enumerate(('ef','train')):
        for col in (0,1):
            ax=axes[i,col];labels=categories if col==0 else phase_keys[i];maximum=0
            for b,backend in enumerate(('native','candidate')):
                values=[next((r['total_ms'] for r in (detail if col==0 else RESULT['phases']) if r['mode']==mode and r['backend']==backend and r['category' if col==0 else 'phase']==key),0) for key in labels]
                maximum=max(maximum,max(values));offset=-.19 if b==0 else .19;y=np.arange(len(labels))+offset
                ax.barh(y,values,height=.34,color='#4477AA' if b==0 else '#228833',hatch='///' if b==0 else '',edgecolor='#333333',linewidth=.5,label='Native Torch' if b==0 else 'dd7a3db')
                for yy,value in zip(y,values):ax.text(value+1,yy,f'{value:.1f}',va='center',fontsize=8)
            ax.set(yticks=np.arange(len(labels)),yticklabels=labels,xlabel='Summed actual GPU kernel duration (ms)',xlim=(0,maximum*1.18),ylim=(len(labels)-.5,-.5))
            ax.grid(axis='x',alpha=.16);ax.set_title(('E/F: 600 atoms, 31,512 edges' if mode=='ef' else 'Training: 100 atoms, 5,252 edges')+(' | kernel categories' if col==0 else ' | execution phases'))
            ax.text(-.10,1.03,chr(65+i*2+col),transform=ax.transAxes,fontweight='bold',fontsize=13)
    fig.suptitle('New fused candidate vs native Torch | same V3 canonical model, checkpoint and inputs\nOne restored-state trace per mode/backend; durations are separate from steady-state wall latency',fontsize=13)
    axes[0,0].legend(loc='lower right',frameon=False)
    for ext in ('png','pdf','svg'):fig.savefig(figures/('NATIVE-VS-DD7A-HOTSPOTS.'+ext),dpi=160,facecolor='white')
    plt.close(fig)
from PIL import Image
with Image.open(figures/'NATIVE-VS-DD7A-HOTSPOTS.png') as im:image_metadata={'pixels':list(im.size),'dpi':list(im.info['dpi']),'mode':im.mode}
caption='Internal analysis figure, publisher choices pending. All actual cat=kernel X durations included. Bars start at zero; native hatching and candidate solid fill encode backend redundantly. All categories/phases retained, including explicit zero categories. ICTC is split by exact kernel names. Different E/F and training sizes retain separate axes; not cross-workload speed comparisons. One trace each; no uncertainty interval is inferred. Native GEMM includes computations now assigned to custom ICTC categories. Latency uses 15 calls (3 fresh processes x 5); these are benchmark repeats, not independent scientific samples. No filtering or smoothing. Full tables supplied.'
(figures/'CAPTION.txt').write_text(caption+'\n')

comparison={r['mode']:r for r in RESULT['comparison']};c=work['ef','candidate'];t=work['train','candidate']
lines=['# dd7a3db热点与原生Torch延迟/显存对照','',
 '候选dd7a3db；复用A100120352、AMD120353已验收计量和纯CPU分析120390。本轮不修改实现或旧FAIL，新增GPUh=0。',
 '', '原生对照为同一`equiformerv3-ictc-o2-canonical-min-v1`的Torch执行，采用原bounded64角向收缩及七块非重入checkpoint策略。checkpoint、原292活动权重、图/PBC/重复边、RNG、loss、clip/AdamW/EMA/scheduler和精度按原回执配对；不把不同模型/未投影上游加速实现当作同一分母。',
 '', '## 延迟和显存','',
 '| 模式 | 原生Torch s | dd7a3db s | 加速比 | 耗时变化 |', '|---|---:|---:|---:|---:|']
for mode,r in comparison.items():lines.append(f'| {mode} | {r["native_s"]:.6f} | {r["candidate_s"]:.6f} | {r["speedup"]:.4f}× | {r["latency_change_percent"]:+.2f}% |')
lines+=['','每格为3个新进程各3预热/5同步调用，共15样本的中位数；包含建图/传输、求力和完整原生更新。公共状态恢复在计时外；它仅定义计量条件，完整各自真实损失连续三步已在原run独立验收。E/F600原子31512边，训练100原子5252边，二者工作量不同。原15个样本、进程中位数和范围保留在原WORKLOADS/RAW-TIMINGS与本轮TRACE-METADATA。',
 '', '| 模式 | 原生allocated GiB | 候选allocated GiB | 差值MiB | 原生reserved GiB | 候选reserved GiB | 差值MiB |','|---|---:|---:|---:|---:|---:|---:|']
for mode,r in comparison.items():lines.append(f'| {mode} | {r["native_allocated_bytes"]/2**30:.6f} | {r["candidate_allocated_bytes"]/2**30:.6f} | {r["allocated_delta_MiB"]:+.5f} | {r["native_reserved_bytes"]/2**30:.6f} | {r["candidate_reserved_bytes"]/2**30:.6f} | {r["reserved_delta_MiB"]:+.0f} |')
lines+=['', 'allocated峰值没有改善：E/F仅增加56KiB，训练增加12.45020MiB（+0.10730%）。reserved分别少66/44MiB，反映此次分配器保留量；本轮未测最大容量。memory使用独立新进程，不把profile内存统计替代此表。',
 '', '## 当前热点：绝对GPU时间','', '| 类别 | EF原生ms | EF候选ms | 候选占比 | 训练原生ms | 训练候选ms | 候选占比 |','|---|---:|---:|---:|---:|---:|---:|']
def value(mode,backend,label):return next(r for r in detail if (r['mode'],r['backend'],r['category'])==(mode,backend,label))
for label in categories:
    en=value('ef','native',label);ec=value('ef','candidate',label);tn=value('train','native',label);tc=value('train','candidate',label)
    lines.append(f'| {label} | {en["total_ms"]:.3f} | {ec["total_ms"]:.3f} | {ec["percent"]:.2f}% | {tn["total_ms"]:.3f} | {tc["total_ms"]:.3f} | {tc["percent"]:.2f}% |')
lines+=['', '分母为每份trace全部实际GPU kernel时间和，不是同步墙钟。原生Torch的部分GEMM、copy和逐元素计算被融合后转入ICTC类别，所以ICTC占比升到约25%不能解释成整体变慢或“绝大多数时间仍在Triton”。',
 '', f'候选E/F kernel合计{c["kernel_ms"]:.3f}ms，训练{t["kernel_ms"]:.3f}ms。最大单类仍是GEMM/GEMV（39.42%/38.49%）；通道核、双线性核、权重伴随分别如表。训练权重伴随归约合计约19.52ms，其中weight_fma8约9.30ms，仅占候选训练kernel合计约1.17%，应按整网份额确定优化优先级。',
 '', f'![以绝对毫秒比较原生Torch与新候选的类别及阶段耗时]({figures/"NATIVE-VS-DD7A-HOTSPOTS.png"})',
 '', '## 阶段与矩阵形状','', '| 模式 | 阶段 | 原生ms | 候选ms | 候选占比 |','|---|---|---:|---:|---:|']
for mode,keys in zip(('ef','train'),phase_keys):
    for key in keys:
        a=next((r for r in RESULT['phases'] if (r['mode'],r['backend'],r['phase'])==(mode,'native',key)),{'total_ms':0})
        b=next((r for r in RESULT['phases'] if (r['mode'],r['backend'],r['phase'])==(mode,'candidate',key)),{'total_ms':0,'percent':0})
        lines.append(f'| {mode} | {key} | {a["total_ms"]:.3f} | {b["total_ms"]:.3f} | {b["percent"]:.2f}% |')
lines+=['', 'E/F主要在从能量求力（71.69%）；训练loss反向59.95%、求力27.87%。optimizer/EMA/clip合计不足1%，当前主要计算瓶颈在完整求导链。',
 '', '| 模式 | BMM形状族（含伴随） | 候选ms | 调用kernel数 |','|---|---|---:|---:|']
for r in role_rows:
    if r['backend']=='candidate':lines.append(f'| {r["mode"]} | {r["shape_family"]} | {r["total_ms"]:.3f} | {r["kernel_calls"]} |')
lines+=['', '上表按实际输入形状分组，不将所有未映射模块的反向耗时猜测为geometry。边旋转形状族与原执行源码`so3.py:359/366`的Wigner正/逆旋转、`input_block.py:96`的初始嵌入旋转相符；2×2形状族与实际adapter/install.py:68/85的C_in/C_out换基相符。它们及各自伴随仍是Torch路径，属于进一步融合的候选。',
 '', '| 模式 | 主要矩阵调用 | 输入形状 | GPU合计ms | kernel数 |','|---|---|---|---:|---:|']
for mode in ('ef','train'):
    selected=[r for r in matrix if r['mode']==mode and r['backend']=='candidate'][:7]
    for r in selected:lines.append(f'| {mode} | {r["operator"]} | `{r["input_dims"]}` | {r["total_ms"]:.3f} | {r["kernel_calls"]} |')
lines+=['', '形状表合并同形状的各相位/模块，计数为关联的GPU kernel数。逐相位/模块版本保留在MATRIX-SHAPES.csv；不是把Bilinear的两次网格发射当作两次数学调用。模块stage前向映射未覆盖所有反向，覆盖率保留在TRACE-METADATA，phase与kernel类别则独立完整统计。',
 '', '## 显存峰值为何没有下降','',
 'profile_memory事件显示：原生与候选的E/F峰值都出现在求力中的index_add_/index_put_边聚合，输入边消息形状[31512,25,128]。训练峰值都出现在loss_backward的BmmBackward0，原始输入形状[5252,5,128]×[5252,128,25]，与输入嵌入旋转反向边界相符。',
 '', '这些是触发profile峰值的算子和边界：E/F触发分配约0.5MB、训练约2.6MB，不能把全部7.59/11.34GiB归到该分配。共同存活的消息、伴随及高阶计算图仍在；双线性临时量的消除没有降低整网全局峰值。完整张量存活期尚未逐项分解。profile进程的峰值数值保留在PROFILE-MEMORY-PEAKS.json，独立memory进程的正式峰值以上表为准。',
 '', '## 墙钟与trace边界','', '| 模式 | 后端 | kernel和ms | kernel并集ms | 首末kernel间隔ms | profile包含ms | 独立计时中位ms |','|---|---|---:|---:|---:|---:|---:|']
for r in RESULT['timing_boundaries']:lines.append(f'| {r["mode"]} | {r["backend"]} | {r["kernel_sum_ms"]:.1f} | {r["kernel_union_ms"]:.1f} | {r["first_to_last_kernel_ms"]:.1f} | {r["profile_CPU_annotation_ms"]:.1f} | {r["independent_steady_median_ms"]:.1f} |')
lines+=['', '原生/候选E/F kernel次数88,968→11,478，训练65,072→29,267；kernel总时间降幅约9.04%/14.35%，小于独立墙钟约49.63%的降幅。结合kernel次数和profile中发射间隙大幅减少，支持本次收益包含显著Python调度/细碎发射开销消除。profile受记录开销影响，不能把上表间隔相减直接解释成稳态CPU耗时，更不能用kernel份额分摊独立墙钟。',
 '', '后续提速应优先检查边旋转及其伴随、C_in/C_out换基、布局copy与通道计算之间的融合边界；显存优化应沿已定位的边聚合/输入嵌入旋转反向峰值，检查边消息及其伴随的存活期和分块消费。保留完整混合二阶、原FP32归约语义及每次独立三步门控。本轮止于分析，未启动下一实现。',
 '', '## 数据与制图方法','',
 '原始回执、trace SHA、CPU external-id关联、完整CSV与本轮脚本已保留；每个类别/相位的calls和ms均回算至原始kernel总量。原候选封存及8324/历史FAIL不改。图为内部报告，零基线、颜色加纹理区分后端、不隐藏0值、无过滤/平滑；一份trace不产生统计置信区间。',
 '', '制图遵循本地主机scientific-visualization技能，参考Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065；2026-11-26核对当前v2，无期刊版本信息。[当前arXiv记录](https://arxiv.org/abs/2609.00065)。这项引用说明流程来源，不支撑本工作区的性能数字。']
(ROOT/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
source_paths=[WS/'baselines/icml-o2-stage1-20261124-r1/payload/network/upstream/experimental/models/equiformer_v3'/n for n in ('so3.py','input_block.py')]
source_paths += [WS/'runs/o2-uvw-triton-a100-20261125-r1/snapshots/preflight-119174/adapter/install.py',WS/'architectures/equiformerv3/source/canonical-cpu-control-c14109a12653/canonical.py']
sources={str(p):sha(p) for p in source_paths}
save(ROOT/'FIGURE-MANIFEST.json',{'figure_script_sha256':sha(__file__),'matplotlib':matplotlib.__version__,'image':image_metadata,'size_inches':[17.5,11],'caption':caption,'outputs':{p.name:sha(p) for p in figures.iterdir() if p.is_file()},'source_data':str(DATA),'shape_source_hashes':sources})
arch=WS/'architectures/equiformerv3/HOTSPOT-BOUNDED-20261126-r1.zh-CN.md'
arch.write_text('# dd7a3db热点与原生框架对照\n\n新候选相对同canonical模型Torch，E/F/训练均约1.985×；allocated未减少，reserved分别少66/44MiB。最大kernel类别仍为GEMM/GEMV（39.42%/38.49%），训练主要为loss反向。\n\n[完整热点、延迟/显存及峰值位置报告](../../runs/o2-bounded-hotspot-a100-20261126-r1/REPORT.zh-CN.md)。复用A100120352/AMD120353，纯CPU新分析，新增GPUh=0；候选/原loss/optimizer/精度/阈值及旧FAIL未改。\n')
path=WS/'architectures/equiformerv3/README.zh-CN.md';text=path.read_text();marker='HOTSPOT-BOUNDED-20261126-r1.zh-CN.md'
if marker not in text:
    first,_,rest=text.partition('\n');path.write_text(first+'\n\n最新热点：[dd7a3db与原生Torch的延迟/显存及绝对耗时分解]('+marker+')。\n'+rest)
path=WS/'HANDOFF.md';text=path.read_text();marker='runs/o2-bounded-hotspot-a100-20261126-r1/REPORT.zh-CN.md'
if marker not in text:
    first,_,rest=text.partition('\n');path.write_text(first+'\n\n2026-11-26新增[dd7a3db热点与原生Torch延迟/显存对照]('+marker+')：复用120352/120353，纯CPU分析120390。两模式约1.985×，allocated未改善；最大kernel类别仍GEMM/GEMV，训练loss反向约60%。原候选封存/8324及旧FAIL不改，新增GPUh=0，无新实现。\n'+rest)
prior_seal=read(OLD/'FINAL-SNAPSHOT.json');assert sha(prior_seal['archive'])==prior_seal['archive_sha256']
files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ('logs','mpl-cache')) and p.name!='DELIVERY.json'}
save(ROOT/'DELIVERY.json',{'status':'HOTSPOT_REPORT_AND_FIGURE_COMPLETE','candidate_commit':RESULT['candidate_commit'],'analysis_job':'120390','delivery_job':os.environ['SLURM_JOB_ID'],'new_GPUh':0,'source_A100_job':'120352','source_AMD_job':'120353','prior_candidate_archive_sha256':prior_seal['archive_sha256'],'source_hashes':sources,'files':files})
print(json.dumps({'status':'HOTSPOT_REPORT_AND_FIGURE_COMPLETE','report':str(ROOT/'REPORT.zh-CN.md'),'figure':str(figures/'NATIVE-VS-DD7A-HOTSPOTS.png'),'comparison':RESULT['comparison'],'BMM_shapes':role_rows}),flush=True)
