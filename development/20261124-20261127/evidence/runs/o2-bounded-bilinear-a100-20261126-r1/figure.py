import hashlib,json,os
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';os.environ['MPLCONFIGDIR']=str(ROOT/'figure-mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
def read(p):return json.loads(p.read_text())
audit=Path((ROOT/'PERFORMANCE-AUDIT.txt').read_text().strip());r=read(audit/'RESULT.json');assert r['status']=='PERFORMANCE_AUDIT_PASS'
data=Path(r['source']);out=ROOT/'figures';out.mkdir(exist_ok=True)
backends=('native','baseline','candidate');labels=['Native\nTorch','8324\nbaseline','Bilinear\nfused'];colors=['#4477AA','#999999','#228833','#CC6677']
palette={'ICTC GPU':'#CC6677','GEMM/GEMV':'#4477AA','index/scatter/gather':'#228833','reduction/norm':'#AA3377','copy/layout':'#66CCEE','fill/zero':'#CCBB44','elementwise':'#BBBBBB','unclassified kernel':'#332288'}
hatches=['','///','...','xxx','\\','+++','---','ooo']
with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'}):
    fig,axes=plt.subplots(2,3,figsize=(18.5,10.5),layout='constrained',gridspec_kw={'width_ratios':[1,1,1.6]})
    memory_top=max(v['reserved_GiB'] for v in r['workloads'])*1.18
    for i,mode in enumerate(('ef','train')):
        by={v['backend']:v for v in r['workloads'] if v['mode']==mode};ta,ma,ka=axes[i];maximum=0
        for j,b in enumerate(backends):
            v=by[b];top=0;ta.bar(j,v['median_s'],width=.62,color=colors[j],alpha=.65,edgecolor='#333333',linewidth=.6)
            for repeat,marker in enumerate(('o','^','s')):
                samples=read(data/f'{mode}-{b}-time-{repeat}/RESULT.json')['times_s'];top=max(top,max(samples));maximum=max(maximum,top)
                ta.scatter(j+np.linspace(-.18,.18,len(samples)),samples,s=13,marker=marker,color='black',alpha=.65,zorder=3)
            ta.annotate(f'{v["median_s"]:.4f} s',(j,top),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9,fontweight='bold')
            ma.bar(j,v['reserved_GiB'],width=.66,facecolor='none',edgecolor='#333333',linestyle='--',linewidth=1)
            ma.bar(j,v['allocated_GiB'],width=.46,color=colors[j],edgecolor='#333333',linewidth=.6)
            ma.annotate(f'{v["allocated_GiB"]:.3f}',(j,v['allocated_GiB']),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
            trace=read(audit/f'traces/{mode}-{b}-trace/SUMMARY.json');families={x['family']:x['percent'] for x in trace['family']};left=0
            for k,(name,color) in enumerate(palette.items()):
                value=families.get(name,0);ka.barh(j,value,left=left,color=color,height=.45,hatch=hatches[k],edgecolor='white',linewidth=.4)
                if name=='ICTC GPU' and value>2:ka.text(left+value/2,j,f'{value:.1f}%',ha='center',va='center',fontsize=9)
                left+=value
            ka.text(0,j+.34,f'{trace["kernel_ms"]:.1f} ms, {trace["kernel_calls"]:,} calls',va='center',fontsize=9)
        title='E/F: 600 atoms, 31,512 edges' if mode=='ef' else 'Full training: 100 atoms, 5,252 edges'
        ta.set(xticks=range(3),xticklabels=labels,ylabel='Synchronized wall time (s)',ylim=(0,maximum*1.18),title=title)
        ma.set(xticks=range(3),xticklabels=labels,ylabel='Peak allocator memory (GiB)',ylim=(0,memory_top),title='Allocated fill / reserved dashed outline')
        for ax in (ta,ma):ax.grid(axis='y',alpha=.15)
        ka.set(yticks=range(3),yticklabels=['Native Torch','8324','Bilinear fused'],xlim=(0,100),ylim=(2.6,-.5),xlabel='% of summed GPU kernel time',title='Separate restored-state trace')
        ka.set_xticks([0,25,50,75,100]);ka.spines['left'].set_visible(False)
        for j,ax in enumerate(axes[i]):ax.text(-.12,1.04,chr(65+3*i+j),transform=ax.transAxes,fontweight='bold',fontsize=13)
    fig.suptitle('V3 canonical | same checkpoint, inputs, complete starting state and RNG | one A100, serial\nEnergy output/reduction FP64; other math/derivatives FP32; TF32/AMP/CUDA Graph off',fontsize=13)
    fig.legend(handles=[Patch(facecolor=c,hatch=hatches[k],label=n,edgecolor='#555555',linewidth=.4) for k,(n,c) in enumerate(palette.items())],loc='outside lower center',ncol=4,frameon=False,fontsize=9)
    for ext in ('png','pdf','svg'):fig.savefig(out/('BOUNDED-BILINEAR-20261126-r1.'+ext),dpi=160,facecolor='white')
    plt.close(fig)
caption='Internal report, not publisher layout. All 15 calls (3 fresh processes x 5; marker denotes process) and their median are shown. Benchmark calls are not independent scientific replicates. Memory peaks use separate processes. Kernel shares normalize all actual GPU kernels per trace and are separate from wall time. No filtering/smoothing; complete source data retained. Historical FAIL is preserved separately.'
(out/'CAPTION.txt').write_text(caption+'\n')
m={'audit':str(audit),'audit_sha256':hashlib.file_digest((audit/'RESULT.json').open('rb'),'sha256').hexdigest(),'script_sha256':hashlib.file_digest(Path(__file__).open('rb'),'sha256').hexdigest(),'matplotlib':matplotlib.__version__,'dimensions_inches':[18.5,10.5],'dpi':160,'caption':caption,'outputs':{p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in out.glob('BOUNDED-BILINEAR-20261126-r1.*')}}
(out/'FIGURE-MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m),flush=True)
