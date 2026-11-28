"""Independent view of non-overlapping launch-attributed GPU execution phases."""
import json,os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/analysis-118587'
assert os.environ.get('SLURM_JOB_ID')
import fast_review
fast_review.run(OUT)
rows=json.loads((OUT/'WORKLOADS.json').read_text());lookup={tuple(r['key']):r for r in rows}
colors={'graph_build_transfer':'#999933','model_forward':'#4477AA','force_from_energy':'#66CCEE','loss':'#DDCC77',
        'loss_backward':'#CC6677','clip':'#AA4499','optimizer':'#228833','EMA':'#44AA99','scheduler':'#882255',
        'zero_grad':'#999999','unattributed':'#222222'}
titles={'v3':'V3 canonical','tece-canonical':'TECE canonical','tece-full':'TECE full parity'}
plt.rcParams.update({'font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(3,2,figsize=(11,8.5),layout='constrained')
for i,route in enumerate(titles):
    for j,mode in enumerate(('ef','train')):
        ax=axes[i,j];ax.set_title(titles[route]+' / '+mode)
        for y,backend in enumerate(('native','ictc')):
            phases={r['phase']:r['percent'] for r in lookup[(route,mode,backend)]['trace']['phase']};left=0
            for name,color in colors.items():
                width=phases.get(name,0);ax.barh(y,width,left=left,color=color,height=.58);left+=width
                if width>=14:ax.text(left-width/2,y,f'{width:.1f}%',ha='center',va='center',color='black' if name in ('force_from_energy','loss','EMA','graph_build_transfer','zero_grad') else 'white',fontsize=8)
            assert abs(left-100)<1e-6
        ax.set_xlim(0,100);ax.set_yticks([0,1],['Native','ICTC']);ax.invert_yaxis();ax.set_xlabel('% of summed GPU kernel time');ax.set_xticks([0,25,50,75,100])
fig.legend(handles=[Patch(color=c,label=n) for n,c in colors.items()],loc='outside lower center',ncol=4,frameon=False,fontsize=8)
fig.suptitle('Execution phases from CPU launch correlation | one full warmed-state trace\nNested force/backward phases are exclusive; unmapped kernels remain explicit',fontsize=11)
fig.savefig(OUT/'execution-phases.png',dpi=180);fig.savefig(OUT/'execution-phases.svg');plt.close(fig)
with (OUT/'REPORT.zh-CN.md').open('a') as f:
    f.write('\n## 执行阶段分布\n\n下图按GPU kernel的CPU launch所属阶段统计，分母仍为该trace所有kernel时间之和；嵌套的求力阶段与模型前向分开，loss反向中的重算保留在对应阶段。CPU包含时间未参与百分比分母。graph_build_transfer对应batch构建与传输；模型内部的几何和设备建图归入model_forward，其模块stage另见CSV。\n\n![执行阶段](execution-phases.png)\n')
print(json.dumps({'status':'PHASE_PLOT_COMPLETED','job':os.environ['SLURM_JOB_ID']}),flush=True)
