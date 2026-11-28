"""CPU-only paired diagnostic receipts, exact GPU kernel inventories and figures."""
import argparse,bisect,csv,gc,hashlib,html,json,math,os,re,statistics
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ROUTES=('v3','tece-canonical','tece-full')
TITLES={'v3':'V3 canonical','tece-canonical':'TECE canonical','tece-full':'TECE full parity'}
def read(p):return json.loads(p.read_text())
def save(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=str)+'\n')
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def csvwrite(path,rows,fields=None):
    if fields is None:fields=list(rows[0]) if rows else ['status']
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def external(e):
    args=e.get('args',{})
    return args.get('External id',args.get('external id'))
def correlation(e):
    args=e.get('args',{})
    return args.get('correlation',args.get('Correlation ID'))

class Intervals:
    def __init__(self,events):
        self.events=sorted(events,key=lambda e:e['ts']);self.starts=[e['ts'] for e in self.events]
        self.max_duration=max((e.get('dur',0) for e in events),default=0)
    def at(self,t):
        right=bisect.bisect_right(self.starts,t);left=bisect.bisect_left(self.starts,t-self.max_duration)
        candidates=[e for e in self.events[left:right] if e['ts']<=t<=e['ts']+e.get('dur',0)]
        return min(candidates,key=lambda e:e.get('dur',0)) if candidates else None

ICTC_NAMES={'_forward','_input_grad','_weight_grad','_weight_grad_partials','_weight_grad_reduce',
            '_weight_shared','_weight_unshared','_reduce_weight',
            '_fused_forward','_fused_left','_right','_weights','_joint',
            '_forward_uvu','_input_grad_uvu','_weight_grad_uvu','_uvu_forward','_uvu_backward'}
def family(name):
    # Explicit name-based categories, not claims about architecture stages.
    if name in ICTC_NAMES or re.search(r'(ictc|_hp_uvu|_uvu_|uvu_forward|uvu_backward)',name,re.I):return 'ICTC Triton'
    if re.search(r'(gemm|gemv|cutlass|cublas|mma|sgemm|s884|s168|ampere_s)',name,re.I):return 'GEMM/GEMV'
    if re.search(r'(index|scatter|gather|select|embedding)',name,re.I):return 'index/scatter/gather'
    if re.search(r'(reduce|reduction|scan|softmax|norm)',name,re.I):return 'reduction/norm'
    if re.search(r'(copy|transpose|permute)',name,re.I):return 'copy/layout'
    if re.search(r'(fill|zero)',name,re.I):return 'fill/zero'
    if re.search(r'(elementwise|vectorized|pointwise|unrolled_elementwise)',name,re.I):return 'elementwise'
    return 'unclassified kernel'

def parse_trace(folder,target):
    target.mkdir(parents=True,exist_ok=True)
    trace=folder/'trace.json';fingerprint={'path':str(trace),'size':trace.stat().st_size,'mtime_ns':trace.stat().st_mtime_ns,'parser_sha256':digest(Path(__file__))}
    cached=target/'SUMMARY.json'
    if cached.exists():
        previous=read(cached)
        if previous.get('fingerprint')==fingerprint and previous.get('classification_version')==2:return previous
    obj=read(trace);events=obj['traceEvents'];counts=Counter(e.get('cat','') for e in events)
    kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
    runtime={correlation(e):e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and correlation(e) is not None}
    cpu={external(e):e for e in events if e.get('cat')=='cpu_op' and external(e) is not None}
    phases=Intervals([e for e in events if e.get('name','').startswith('phase:') and 'dur' in e])
    stages_by_tid=defaultdict(list)
    for e in events:
        if e.get('name','').startswith('stage:') and 'dur' in e:stages_by_tid[e.get('tid')].append(e)
    stages={tid:Intervals(rows) for tid,rows in stages_by_tid.items()}
    totals=defaultdict(lambda:[0,0.0,0.0]);byphase=defaultdict(lambda:[0,0.0]);bystage=defaultdict(lambda:[0,0.0]);byfamily=defaultdict(lambda:[0,0.0])
    byphasefamily=defaultdict(lambda:[0,0.0]);byoperator=defaultdict(lambda:[0,0.0]);mapped=Counter();inventory=[]
    for event in kernels:
        name=event['name'];duration=event['dur']/1000;cat=family(name)
        launch=runtime.get(correlation(event));operation=cpu.get(external(event))
        anchor=launch or operation
        phase=phases.at(anchor['ts']) if anchor else None
        stage=stages[anchor.get('tid')].at(anchor['ts']) if anchor and anchor.get('tid') in stages else None
        if stage is None and operation and operation.get('tid') in stages:stage=stages[operation['tid']].at(operation['ts'])
        phase=phase['name'][6:] if phase else 'unattributed'
        stage=stage['name'][6:] if stage else 'unattributed'
        operator=operation.get('name','unattributed') if operation else 'unattributed'
        totals[name][0]+=1;totals[name][1]+=duration;totals[name][2]=max(totals[name][2],duration)
        for table,key in ((byphase,phase),(bystage,stage),(byfamily,cat),(byphasefamily,(phase,cat)),(byoperator,operator)):
            table[key][0]+=1;table[key][1]+=duration
        if anchor:mapped['launch_calls']+=1;mapped['launch_ms']+=duration
        if phase!='unattributed':mapped['phase_calls']+=1;mapped['phase_ms']+=duration
        if stage!='unattributed':mapped['stage_calls']+=1;mapped['stage_ms']+=duration
        inventory.append({'kernel':name,'phase':phase,'stage':stage,'operator':operator,'family':cat,'duration_ms':duration,
                          'start_us':event['ts'],'correlation':correlation(event),'external_id':external(event)})
    total=sum(e['dur']/1000 for e in kernels)
    rows=[{'kernel':n,'family':family(n),'calls':v[0],'total_ms':v[1],'average_us':v[1]*1000/v[0],'max_us':v[2]*1000,'percent':100*v[1]/total} for n,v in sorted(totals.items(),key=lambda item:-item[1][1])]
    csvwrite(target/'KERNELS.csv',rows);csvwrite(target/'KERNEL-EVENTS.csv',inventory)
    def grouped(table,key):
        data=[{key:n,'calls':v[0],'total_ms':v[1],'percent':100*v[1]/total} for n,v in sorted(table.items(),key=lambda item:-item[1][1])]
        assert sum(r['calls'] for r in data)==len(kernels)
        assert math.isclose(sum(r['total_ms'] for r in data),total,rel_tol=1e-10)
        csvwrite(target/(key.upper()+'.csv'),data);return data
    phasesummary=grouped(byphase,'phase');stagesummary=grouped(bystage,'stage');familysummary=grouped(byfamily,'family');grouped(byoperator,'operator')
    csvwrite(target/'PHASE-FAMILY.csv',[{'phase':k[0],'family':k[1],'calls':v[0],'total_ms':v[1],'percent':v[1]*100/total} for k,v in byphasefamily.items()])
    transfers=[e for e in events if e.get('cat') in ('gpu_memcpy','gpu_memset') and e.get('ph')=='X']
    transfer_groups=defaultdict(lambda:[0,0.0])
    for event in transfers:transfer_groups[(event['cat'],event['name'])][0]+=1;transfer_groups[(event['cat'],event['name'])][1]+=event['dur']/1000
    transferrows=[{'category':k[0],'name':k[1],'calls':v[0],'total_ms':v[1]} for k,v in transfer_groups.items()]
    csvwrite(target/'TRANSFERS.csv',transferrows,fields=['category','name','calls','total_ms'])
    annotations=[{'name':e['name'],'cpu_inclusive_ms':e['dur']/1000} for e in events if e.get('name','').startswith(('phase:','diagnostic_')) and 'dur' in e]
    csvwrite(target/'CPU-INCLUSIVE-ANNOTATIONS.csv',annotations)
    summary={'fingerprint':fingerprint,'classification_version':2,'trace_sha256':digest(trace),'event_categories':dict(counts),'kernel_calls':len(kernels),'kernel_ms':total,
             'kernel_time_definition':'sum of actual cat=kernel X event durations; may overlap across streams; distinct from wall time',
             'mapped':dict(mapped),'phase_coverage_percent':mapped['phase_ms']*100/total,'stage_coverage_percent':mapped['stage_ms']*100/total,
             'phase':phasesummary,'family':familysummary,'stage':stagesummary,'top_kernels':rows[:20],'transfers':transferrows}
    save(cached,summary);del obj,events,kernels,inventory;gc.collect();return summary

def paired_checks(records):
    checks=[]
    for route in ROUTES:
        for mode in ('ef','train'):
            selected=[r for r in records if r['route']==route and r['mode']==mode and r['backend'] in ('native','ictc')]
            conditions=[r['condition'] for r in selected if r.get('condition')]
            row={'route':route,'mode':mode,'completed_actions':sum(r['status']=='DIAGNOSTIC_COMPLETED' for r in selected),'expected_actions':6}
            for key in ('parameter_sha256','common_state_sha256','graph'):
                values=[json.dumps(c[key],sort_keys=True) for c in conditions];row[key+'_identical']=len(values)==6 and len(set(values))==1
            for key in ('gpu_uuid','node','torch','deterministic','TF32','AMP','CUDA_Graph','threads'):
                values=[json.dumps(c['hardware'][key],sort_keys=True) for c in conditions];row[key+'_identical']=len(values)==6 and len(set(values))==1
            for key in ('same_state_each_measured_step','state_restore_outside_measurement'):
                row[key]=len(conditions)==6 and all(c.get(key) is True for c in conditions)
            for key in ('memory_policy','spherical_execution'):
                values=[json.dumps(c['identity'].get(key),sort_keys=True) for c in conditions];row[key+'_identical']=len(values)==6 and len(set(values))==1
            values=[json.dumps(c['hardware']['non_fp32_buffers'],sort_keys=True) for c in conditions]
            row['buffer_dtypes_identical']=len(values)==6 and len(set(values))==1
            if mode=='train':
                values=[json.dumps(c.get('complete_state_identity'),sort_keys=True) for c in conditions]
                row['complete_restored_state_identical']=len(values)==6 and len(set(values))==1 and all(c.get('complete_restored_state_verified') for c in conditions)
                row['common_template_unchanged']=len(selected)==6 and all(r['result'].get('common_template_unchanged') for r in selected)
                values=[json.dumps(c['execution']['active_gradient_names'],sort_keys=True) for c in conditions]
                row['active_gradient_roles_identical']=len(values)==6 and len(set(values))==1
            else:
                row['complete_restored_state_identical']=True
                row['common_template_unchanged']=True
                row['active_gradient_roles_identical']=True
            row['paired_conditions_pass']=row['completed_actions']==6 and all(v for k,v in row.items() if k.endswith('_identical') or k in ('same_state_each_measured_step','state_restore_outside_measurement','common_template_unchanged'))
            checks.append(row)
    return checks

def figures(workloads,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    colors={'ICTC Triton':'#CC6677','GEMM/GEMV':'#4477AA','index/scatter/gather':'#228833','reduction/norm':'#AA3377',
            'copy/layout':'#66CCEE','fill/zero':'#CCBB44','elementwise':'#BBBBBB','unclassified kernel':'#332288'}
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    for i,route in enumerate(ROUTES):
        for j,mode in enumerate(('ef','train')):
            ax=axes[i,j];ax.set_title(TITLES[route]+' / '+('E/F inference (600 atoms)' if mode=='ef' else 'full training step (100 atoms)'))
            for y,backend in enumerate(('native','ictc')):
                row=workloads.get((route,mode,backend));left=0
                if not row or not row.get('trace'):ax.text(0,y,'missing',va='center');continue
                families={r['family']:r['percent'] for r in row['trace']['family']}
                for cat,color in colors.items():
                    width=families.get(cat,0);ax.barh(y,width,left=left,color=color,height=.55);left+=width
                ax.text(101,y,f"{row['trace']['kernel_ms']:.0f} ms; {row['trace']['kernel_calls']:,} calls",va='center',fontsize=8)
            ax.set_yticks([0,1],['Native\nTorch','ICTC\n(Torch rest)']);ax.invert_yaxis();ax.set_xlim(0,149);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('% of summed GPU kernel time')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c,label=n) for n,c in colors.items()],loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('Torch reference vs ICTC with Torch rest | one restored-state diagnostic trace\nCategories use explicit kernel-name rules; original accelerated controls are shown separately',fontsize=12)
    fig.savefig(out/'kernel-distribution.png',dpi=170);fig.savefig(out/'kernel-distribution.svg');plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained')
    for i,route in enumerate(ROUTES):
        for j,mode in enumerate(('ef','train')):
            ax=axes[i,j];selected=[workloads.get((route,mode,b),{}) for b in ('native','ictc')];top=0
            for x,row in enumerate(selected):
                if 'time' not in row:continue
                samples=row['time']['times_s'];ax.bar(x,statistics.median(samples),width=.5,color=['#4477AA','#CC6677'][x],alpha=.65)
                top=max(top,max(samples))
                ax.scatter(np.linspace(x-.12,x+.12,len(samples)),samples,s=10,color='black',zorder=3)
                peak=row.get('memory',{}).get('peak_bytes',{}).get('allocated')
                label=f"{statistics.median(samples):.3f} s"+(f"\n{peak/2**30:.2f} GiB peak" if peak is not None else '')
                ax.annotate(label,(x,max(samples)),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
            ax.set_title(TITLES[route]+' / '+mode);ax.set_xticks([0,1],['Native Torch','ICTC\n(Torch rest)']);ax.set_ylabel('Synchronized wall time (s)');ax.set_ylim(0,top*1.32 if top else 1)
    fig.suptitle('Ten repeats from one identical common state | bars: median; dots: raw steps\nPeak allocated memory collected separately in a fresh process; no cross-architecture speed ratios',fontsize=11)
    fig.savefig(out/'time-memory.png',dpi=170);fig.savefig(out/'time-memory.svg');plt.close(fig)

def main(args):
    assert os.environ.get('SLURM_JOB_ID'),'CPU scheduler allocation required'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);results=Path(args.results)
    progress=read(results/'PROGRESS.json');records=[];workloads={};model_configs=[]
    for row in progress:
        if row['action']=='prepare':
            folder=Path(row['folder'])
            if (folder/'CONDITION.json').exists():
                c=read(folder/'CONDITION.json')
                model_configs.append({'route':row['route'],'mode':row['mode'],'backend':row['backend'],
                    'identity':c['identity'],'parameter_numel':c['parameters']['numel'],'parameter_sha256':c['parameters']['sha256'],
                    'graph':c['graph'],'hardware':c['hardware'],'common_state_sha256':c['start_sha256'],'folder':str(folder)})
            continue
        folder=Path(row['folder']);result=read(folder/'RESULT.json') if (folder/'RESULT.json').exists() else {}
        condition=read(folder/'CONDITION.json') if (folder/'CONDITION.json').exists() else None
        item=dict(row,status=result.get('status','EXECUTION_ERROR'),condition=condition,result=result)
        key=(row['route'],row['mode'],row['backend']);workload=workloads.setdefault(key,{'route':key[0],'mode':key[1],'backend':key[2]})
        if item['status']=='DIAGNOSTIC_COMPLETED':
            if row['action']=='trace':workload['trace']=parse_trace(folder,out/'traces'/folder.name)
            else:workload[row['action']]=result
            workload['condition']=condition
        records.append(item)
    checks=paired_checks(records);save(out/'PAIRED-CHECKS.json',checks)
    csvwrite(out/'PAIRED-CHECKS.csv',checks,list(dict.fromkeys(k for r in checks for k in r)))
    save(out/'MODEL-CONFIGS.json',model_configs)
    rows=[];times=[]
    for key,row in workloads.items():
        condition=row.get('condition',{});graph=condition.get('graph',{})
        data={'route':key[0],'mode':key[1],'backend':key[2],'atoms':graph.get('atoms'),'directed_edges':graph.get('edges')}
        timing=row.get('time');memory=row.get('memory');trace=row.get('trace')
        data.update(median_s=timing['median_s'] if timing else None,min_s=min(timing['times_s']) if timing else None,max_s=max(timing['times_s']) if timing else None,
                    peak_allocated_GiB=memory['peak_bytes']['allocated']/2**30 if memory else None,peak_reserved_GiB=memory['peak_bytes']['reserved']/2**30 if memory else None,
                    base_allocated_GiB=memory['base_bytes']['allocated']/2**30 if memory else None,kernel_calls=trace['kernel_calls'] if trace else None,kernel_ms=trace['kernel_ms'] if trace else None,
                    phase_coverage_percent=trace['phase_coverage_percent'] if trace else None,stage_coverage_percent=trace['stage_coverage_percent'] if trace else None)
        rows.append(data)
        if timing:
            for i,t in enumerate(timing['times_s']):times.append({'route':key[0],'mode':key[1],'backend':key[2],'repeat':i+1,'wall_s':t})
    csvwrite(out/'MEASUREMENTS.csv',rows);csvwrite(out/'RAW-TIMES.csv',times)
    save(out/'ALL-RECEIPTS.json',records);save(out/'WORKLOADS.json',[dict(v,key=list(k)) for k,v in workloads.items()])
    figures(workloads,out)
    doc=['# 三路线诊断热点与配对检查','',f"GPU结果：`{results}`；CPU分析作业 `{os.environ['SLURM_JOB_ID']}`。",'',
         '本报告仅用于查看当前实现的分布；旧FAIL/HOLD保留，第二、第三步未启动。原生/ICTC为每一路线内部的配对；原始V3是不同模型的绝对成本参考。','',
         '## 配对条件','',f"六组完整采集及参数/公共状态/图标签/硬件/精度/线程一致检查：{sum(r['paired_conditions_pass'] for r in checks)}/6。详见 [PAIRED-CHECKS.csv](PAIRED-CHECKS.csv)。",'',
         '训练每次恢复同一个原生3步预热状态，恢复在计时之外。time/memory/trace为独立进程，各3次代码预热。耗时为10次同步墙钟中位数，含建图/传输；训练含真实E/F（各原生要求的stress）损失、反向、clip、optimizer和EMA，scheduler保留原生间隔。原生及候选均使用能量FP64、其他计算FP32，无TF32/AMP/CUDA Graph。','',
         '三条路线以同一公开Al构型为输入，batch=1，但保留各自原生图规则、架构宽度/层数、loss和optimizer。V3与TECE canonical计算stress，TECE full保留E/F配方。各原生完整config、loss权重、参数量、内存重算策略和实际flags见[MODEL-CONFIGS.json](MODEL-CONFIGS.json)。同路线内逐项配对，不把不同架构或不同原子数强行称同一计算任务。','',
         '显卡为Slurm共享节点上独立分配的一卡；同一配对核对物理GPU UUID。没有锁定时钟或独占整节点，10次采样是同一帧/状态的重复测量，不是独立任务精度实验或多节点统计。','',
         '## 测量','', '|路线|模式|后端|原子/有向边|中位耗时 s|峰值allocated/reserved GiB|kernel时间和 ms / 次数|','|---|---|---|---:|---:|---:|---:|']
    def fmt(v,d=3):return '缺失' if v is None else f'{v:.{d}f}'
    for row in rows:
        doc.append(f"|{row['route']}|{row['mode']}|{row['backend']}|{row['atoms']}/{row['directed_edges']}|{fmt(row['median_s'])}|{fmt(row['peak_allocated_GiB'])}/{fmt(row['peak_reserved_GiB'])}|{fmt(row['kernel_ms'])} / {row['kernel_calls']}|")
    doc+=['','kernel时间和是实际cat=kernel事件时长之和，可能跨stream重叠，不能当同步墙钟。memcpy/memset单列；CPU算子包含时间不并入kernel分母。峰值显存包含harness常驻模型/EMA/优化器，不声称最小部署显存或最大容量。原始时间见[RAW-TIMES.csv](RAW-TIMES.csv)。','','![kernel分布](kernel-distribution.png)','','![耗时和显存](time-memory.png)','','## 完整分布入口','']
    for key,row in workloads.items():
        trace=row.get('trace')
        if not trace:continue
        name='-'.join(key)+'-trace';relative='traces/'+name
        doc += [f"### {' / '.join(key)}",'',f"kernel总时长 {trace['kernel_ms']:.3f} ms，{trace['kernel_calls']}次；launch阶段映射覆盖 {trace['phase_coverage_percent']:.2f}%，前向模块stage映射覆盖 {trace['stage_coverage_percent']:.2f}%。未映射保留unattributed，不猜测为geometry。",'',
                f"[完整kernel CSV]({relative}/KERNELS.csv) · [逐事件]({relative}/KERNEL-EVENTS.csv) · [阶段]({relative}/PHASE.csv) · [模块stage]({relative}/STAGE.csv) · [memcpy/memset]({relative}/TRANSFERS.csv)",'', '|核（前5，名称截断仅用于展示）|调用|总ms|占比|','|---|---:|---:|---:|']
        for kernel in trace['top_kernels'][:5]:doc.append(f"|`{kernel['kernel'][:115].replace('|','/')}`|{kernel['calls']}|{kernel['total_ms']:.3f}|{kernel['percent']:.2f}%|")
        doc.append('')
    errors=[r for r in records if r['exit_code']]
    doc+=['## 执行异常与边界','']
    doc += [f"- `{r['route']}/{r['mode']}/{r['backend']}/{r['action']}`：exit={r['exit_code']}；{r.get('error')}；原始日志 `{r['folder']}/process.log`。" for r in errors] or ['本批已列工作负载无执行异常。']
    doc+=['','TECE full为固定随机初始化结构性能。V3 canonical为独立投影模型，不能对未投影上游模型声明同checkpoint等价加速。不同架构保留各自原生邻居规则、宽度/层数、loss、stress及optimizer，不跨架构算倍率。所有结果需连同[执行条件](../../CONDITIONS.zh-CN.md)和[ALL-RECEIPTS.json](ALL-RECEIPTS.json)审阅。','',
          '阶段归因通过GPU kernel correlation到CPU launch及trace注释，不以CPU包含时间直接加和。自动微分后端线程无可映射前向模块时显式未归因；逐核分布和阶段分布分别保留。名称类别规则见analyze.py，不把名称类别当数学子图身份。']
    (out/'REPORT.zh-CN.md').write_text('\n'.join(doc)+'\n')
    # Offline searchable full tables; no external scripts or network dependencies.
    body=['<!doctype html><meta charset="utf-8"><title>O2 paired diagnostic</title><style>body{font:14px sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:6px}td:first-child{overflow-wrap:anywhere;max-width:850px}input{width:70%;padding:8px}</style>',
          '<h1>Paired diagnostic kernel inventory</h1><p>Percent denominator: all actual GPU kernels within this single trace. CPU operators and memory transfers excluded.</p><input id="q" placeholder="Filter route, mode, backend, kernel name" oninput="document.querySelectorAll(\'tbody tr\').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase()))"><table><thead><tr><th>route / mode / backend / full kernel name</th><th>calls</th><th>ms</th><th>%</th></tr></thead><tbody>']
    for key,row in workloads.items():
        if 'trace' not in row:continue
        with (out/'traces'/('-'.join(key)+'-trace')/'KERNELS.csv').open() as stream:
            for k in csv.DictReader(stream):body.append('<tr><td>'+html.escape(' / '.join(key)+' / '+k['kernel'])+'</td><td>'+k['calls']+'</td><td>'+f"{float(k['total_ms']):.4f}"+'</td><td>'+f"{float(k['percent']):.3f}"+'</td></tr>')
    (out/'inventory.html').write_text('\n'.join(body+['</tbody></table>']))
    save(out/'ANALYSIS.json',{'status':'ANALYSIS_COMPLETED','job':os.environ['SLURM_JOB_ID'],'paired_pass':sum(r['paired_conditions_pass'] for r in checks),'workloads':len(workloads),'records':len(records),'errors':len(errors)})
    print(json.dumps(read(out/'ANALYSIS.json')),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('results');parser.add_argument('output');main(parser.parse_args())
