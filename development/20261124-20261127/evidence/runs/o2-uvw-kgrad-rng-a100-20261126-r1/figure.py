"""Provisional internal figure: all timing samples, allocator peaks, kernel sums."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
os.environ['MPLCONFIGDIR']=str(ROOT/'figure-mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
def read(path):return json.loads(path.read_text())
audit=Path((ROOT/'PERFORMANCE-AUDIT.txt').read_text().strip());result=read(audit/'RESULT.json');assert result['status']=='PERFORMANCE_AUDIT_PASS'
data=Path(result['source']);out=ROOT/'figures';out.mkdir(exist_ok=True)
palette={'ICTC Triton':'#CC6677','GEMM/GEMV':'#4477AA','index/scatter/gather':'#228833','reduction/norm':'#AA3377',
    'copy/layout':'#66CCEE','fill/zero':'#CCBB44','elementwise':'#BBBBBB','unclassified kernel':'#332288'}
hatches=['','///','...','xxx','\\\\','+++','---','ooo'];backend_colors=['#4477AA','#999999','#CC6677']
with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'}):
    fig,axes=plt.subplots(2,3,figsize=(16.5,8.6),layout='constrained',gridspec_kw={'width_ratios':[1,1,1.35]})
    memory_top=max(r['reserved_GiB'] for r in read(audit/'WORKLOADS.json'))*1.20
    for i,mode in enumerate(('ef','train')):
        label='E/F: 600 atoms, 31,512 edges' if mode=='ef' else 'Full training: 100 atoms, 5,252 edges'
        rows={r['backend']:r for r in read(audit/'WORKLOADS.json') if r['mode']==mode}
        names=['Native\nTorch','UVW v2'+('\nFAIL (diagnostic)' if mode=='train' else ''),'UVW v3\nK-gradient repair']
        time_ax,memory_ax,kernel_ax=axes[i];maximum=0
        for j,b in enumerate(('native','v2','candidate')):
            row=rows[b];median=row['median_s'];backend_top=0
            time_ax.bar(j,median,width=.6,color=backend_colors[j],alpha=.65,hatch='///' if b=='v2' else '',edgecolor='#333333',linewidth=.6)
            for repeat,marker in enumerate(('o','^','s')):
                samples=read(data/f'{mode}-{b}-time-{repeat}'/'RESULT.json')['times_s'];maximum=max(maximum,max(samples))
                backend_top=max(backend_top,max(samples))
                time_ax.scatter(j+np.linspace(-.20,.20,len(samples)),samples,s=13,marker=marker,color='black',alpha=.65,zorder=3)
            time_ax.annotate(f'{median:.4f} s',(j,backend_top),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9,fontweight='bold')
            memory_ax.bar(j,row['reserved_GiB'],width=.66,facecolor='none',edgecolor='#333333',linestyle='--',linewidth=1.0)
            memory_ax.bar(j,row['allocated_GiB'],width=.48,color=backend_colors[j],hatch='///' if b=='v2' else '',edgecolor='#333333',linewidth=.6)
            memory_ax.annotate(f'{row["allocated_GiB"]:.3f}',(j,row['allocated_GiB']),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
        time_ax.set(xticks=range(3),xticklabels=names,ylabel='Synchronized wall time (s)',ylim=(0,maximum*1.20),title=label)
        memory_ax.set(xticks=range(3),xticklabels=names,ylabel='Peak allocator memory (GiB)',ylim=(0,memory_top),title='Allocated fill / reserved dashed outline')
        time_ax.grid(axis='y',alpha=.15);memory_ax.grid(axis='y',alpha=.15)
        for j,b in enumerate(('native','candidate')):
            trace=read(audit/'traces'/f'{mode}-{b}-trace'/'SUMMARY.json');families={r['family']:r['percent'] for r in trace['family']};left=0
            for k,(name,color) in enumerate(palette.items()):
                value=families.get(name,0)
                kernel_ax.barh(j,value,left=left,color=color,height=.48,hatch=hatches[k],edgecolor='white',linewidth=.4)
                if name=='ICTC Triton' and value>2:kernel_ax.text(left+value/2,j,f'{value:.1f}%',ha='center',va='center',color='black',fontsize=9)
                left+=value
            kernel_ax.text(0,j+.40,f'{trace["kernel_ms"]:.1f} ms total, {trace["kernel_calls"]:,} calls',va='center',fontsize=9)
        kernel_ax.set(yticks=[0,1],yticklabels=['Native Torch','UVW v3'],xlim=(0,100),ylim=(1.65,-.50),
            xlabel='% of summed GPU kernel time',title='One restored-state trace (separate from wall time)')
        kernel_ax.set_xticks([0,25,50,75,100]);kernel_ax.spines['left'].set_visible(False)
        for j,ax in enumerate(axes[i]):ax.text(-.14,1.04,chr(65+i*3+j),transform=ax.transAxes,fontweight='bold',fontsize=13)
    fig.suptitle('V3 canonical | fixed checkpoint, inputs and complete starting state | one A100, serial execution\nEnergy output/reduction FP64; other math/derivatives FP32; TF32/AMP/CUDA Graph off',fontsize=13)
    handles=[Patch(facecolor=color,hatch=hatches[k],label=name,edgecolor='#555555',linewidth=.4) for k,(name,color) in enumerate(palette.items())]
    fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False,fontsize=9)
    for extension in ('png','pdf','svg'):fig.savefig(out/('UVW-KGRAD-RNG-20261126-r1.'+extension),dpi=160,facecolor='white')
    plt.close(fig)
description=('Internal diagnostic figure, not a journal submission layout. A/D show medians and all 15 calls (3 fresh processes x 5; o,^,s identify processes). '
    'Calls are benchmark repetitions, not independent scientific samples. B/E show separately measured allocated and reserved peaks. '
    'C/F normalize all actual cat=kernel X event durations; they are not wall-clock shares. V2 independent training remains FAIL and its cost is diagnostic only. '
    'No observations filtered or smoothed; complete raw data and timing boundaries are in the paired audit. No cross-architecture comparison.')
(out/'CAPTION.txt').write_text(description+'\n')
manifest={'data':str(audit),'RESULT_sha256':hashlib.file_digest((audit/'RESULT.json').open('rb'),'sha256').hexdigest(),
    'script_sha256':hashlib.file_digest(Path(__file__).open('rb'),'sha256').hexdigest(),'matplotlib':matplotlib.__version__,
    'dimensions_inches':[16.5,8.6],'dpi':160,'destination':'provisional internal report; publisher-specific choices not requested',
    'transformations':['all raw timing calls; 15-call median','allocator bytes / 2**30','sum kernel durations normalized per trace'],
    'caption':description,'outputs':{p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in out.glob('UVW-KGRAD-RNG-20261126-r1.*')}}
(out/'FIGURE-MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
