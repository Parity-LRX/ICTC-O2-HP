"""Report original fused execution and a matched ECE module cut separately."""
import csv,json,statistics
from pathlib import Path
def run(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    records=json.loads((out/'ALL-RECEIPTS.json').read_text())
    workloads={tuple(r['key']):r for r in json.loads((out/'WORKLOADS.json').read_text())}
    checks=[]
    for route in ('tece-canonical','tece-full'):
        for mode in ('ef','train'):
            reference='native-eqx-module' if route=='tece-canonical' else 'native-eqx-paired'
            selected=[r for r in records if r['route']==route and r['mode']==mode and r['backend'] in (reference,'ictc-eqx')]
            conditions=[r['condition'] for r in selected if r['status']=='DIAGNOSTIC_COMPLETED']
            row={'route':route,'mode':mode,'successful_actions':len(conditions),'expected_actions':6}
            for key in ('parameter_sha256','common_state_sha256','graph','complete_state_identity'):
                values=[json.dumps(c.get(key),sort_keys=True) for c in conditions];row[key+'_identical']=len(values)==6 and len(set(values))==1
            for key in ('node','gpu_uuid','torch','threads','deterministic','TF32','AMP','CUDA_Graph','use_eqx_modules','env','non_fp32_buffers'):
                values=[json.dumps(c['hardware'][key],sort_keys=True) for c in conditions];row[key+'_identical']=len(values)==6 and len(set(values))==1
            values=[json.dumps(c['execution']['active_gradient_names'],sort_keys=True) for c in conditions]
            row['active_gradient_roles_identical']=len(values)==6 and len(set(values))==1
            row['paired_pass']=len(conditions)==6 and all(v for k,v in row.items() if k.endswith('_identical'))
            row['coverage_calls']=max((c['execution'].get('ictc_forward_pair_calls',0) for c in conditions),default=0)
            checks.append(row)
    with (out/'FAST-PAIRED-CHECKS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
    (out/'FAST-PAIRED-CHECKS.json').write_text(json.dumps(checks,indent=2))
    colors={'ICTC Triton':'#CC6677','GEMM/GEMV':'#4477AA','index/scatter/gather':'#228833','reduction/norm':'#AA3377','copy/layout':'#66CCEE','fill/zero':'#CCBB44','elementwise':'#BBBBBB','unclassified kernel':'#332288'}
    labels=('Native fused','Native module','ICTC module')
    table=['\n## 原生加速路径及相同ECE切口对照\n',
           '三种执行都使用同一公开帧、完整公共状态、FP32/FP64约定与原生重算设置，禁用CUDA Graph。TECE canonical的原生Eqx采用融合RRA路径，现有ICTC接入使用RRA module route（fused=False）；full parity的同切口双方将两处EceO2Interaction选择原源码module分支并保留其他加速。另测相同module route的原生参考，区分拆开融合路径的成本与ECE后端差异。完整原生融合路径仍保留为整网性能参考。canonical采用该版本equivariantx，full采用历史版本eqx，各固定在对应源码中。',
           '\n|路线|模式|后端|中位秒|allocated / reserved GiB|kernel ms|\n|---|---|---|---:|---:|---:|']
    for route,filename in (('tece-canonical','native-fast-comparison'),('tece-full','native-fast-full-comparison')):
        fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
        reference='native-eqx-module' if route=='tece-canonical' else 'native-eqx-paired'
        backends=('native-eqx',reference,'ictc-eqx')
        for i,mode in enumerate(('ef','train')):
            top=0
            for y,(backend,label) in enumerate(zip(backends,labels)):
                row=workloads.get((route,mode,backend),{})
                if not row.get('trace') or not row.get('time'):
                    axes[i,0].text(0,y,'Unavailable: '+label,va='center')
                    axes[i,1].text(y,.05,'Unavailable',ha='center',rotation=90,transform=axes[i,1].get_xaxis_transform(),color='#666666');continue
                left=0;families={r['family']:r['percent'] for r in row['trace']['family']}
                for name,color in colors.items():
                    width=families.get(name,0);axes[i,0].barh(y,width,left=left,height=.55,color=color);left+=width
                values=row['time']['times_s'];median=statistics.median(values)
                top=max(top,max(values))
                axes[i,1].bar(y,median,width=.55,color=('#4477AA','#66CCEE','#CC6677')[y]);axes[i,1].scatter([y]*len(values),values,s=9,color='black',zorder=3)
                peak=row.get('memory',{}).get('peak_bytes',{}).get('allocated')
                axes[i,1].annotate(f'{median:.3f} s'+(f'\n{peak/2**30:.2f} GiB' if peak else ''),(y,max(values)),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
            axes[i,0].set_yticks(range(3),labels);axes[i,0].set_ylim(2.48,-.48);axes[i,0].set_xlim(0,100);axes[i,0].set_xlabel('% of summed GPU kernel time');axes[i,0].set_title(route+' / '+mode)
            axes[i,1].set_xticks(range(3),['Native\nfused','Native\nmodule','ICTC\nmodule']);axes[i,1].set_xlim(-.55,2.55);axes[i,1].set_ylim(0,top*1.35 if top else 1);axes[i,1].set_ylabel('Synchronized wall time (s)');axes[i,1].set_title('Same restored state; median and ten raw steps')
        fig.legend(handles=[Patch(color=c,label=n) for n,c in colors.items()],loc='outside lower center',ncol=4,frameon=False,fontsize=8)
        fig.suptitle('Upstream fused execution and matched ECE module cut\nOriginal fused and module execution are listed separately; no qualified speedup claim',fontsize=12)
        fig.savefig(out/(filename+'.png'),dpi=180);fig.savefig(out/(filename+'.svg'));plt.close(fig)
    for key,row in workloads.items():
        if key[2] not in ('native-eqx','native-eqx-module','native-eqx-paired','ictc-eqx') or not row.get('time'):continue
        memory=row.get('memory',{}).get('peak_bytes',{});trace=row.get('trace',{})
        memtext=f"{memory['allocated']/2**30:.3f} / {memory['reserved']/2**30:.3f}" if 'allocated' in memory else 'unavailable'
        table.append(f"|{key[0]}|{key[1]}|{key[2]}|{row['time']['median_s']:.6f}|{memtext}|{trace.get('kernel_ms','unavailable')}|")
    table+=['','![canonical原生加速及同切口对照](native-fast-comparison.png)','','![full parity原生加速及同切口对照](native-fast-full-comparison.png)','',
            'TECE canonical的加速module参考及ICTC在ECE之前的align步骤均报batch1 must be a 3D tensor。原生融合路径使用packed Wigner，而当前fused=False分支调用dense bmm；两组标为不可运行，不据其缺失值报速度或显存。主矩阵的原生Torch/ICTC及原生融合Eqx均有完整采集。该加速切口接口修复留待分布审阅后的下一步决定。','',
            '[加速路径配对检查](FAST-PAIRED-CHECKS.csv)。只在完整六项采集均成功且状态/图/精度/实际后端/硬件一致时记paired_pass；失败或缺失保留原始状态。候选预检另记录实际ICTC pair调用次数，避免上游融合路径绕过接入。','',
            '本轮为状态回放诊断：每个样本恢复公共状态，恢复本身不计时；恢复引起派生权重缓存失效时，若缓存在call内重建，其成本计入该样本。这里的代码预热不代表部署时所有权重布局缓存持续有效。生产缓存边界待用户审阅后决定，不在本轮据此报正式部署加速。']
    with (out/'REPORT.zh-CN.md').open('a') as f:f.write('\n'.join(table)+'\n')
    print(json.dumps({'fast_pair_pass':sum(r['paired_pass'] for r in checks),'fast_pairs':checks}),flush=True)
