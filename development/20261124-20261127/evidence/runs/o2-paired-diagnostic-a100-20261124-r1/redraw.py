import hashlib,json,os,runpy,shutil
from pathlib import Path
import analyze
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/analysis-118587'
assert os.environ.get('SLURM_JOB_ID')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
prior=OUT/'plot-v1';prior.mkdir(exist_ok=False)
for p in list(OUT.glob('*.png'))+list(OUT.glob('*.svg')):shutil.copy2(p,prior/p.name)
data_sha=sha(OUT/'WORKLOADS.json')
rows=json.loads((OUT/'WORKLOADS.json').read_text());workloads={tuple(r['key']):r for r in rows}
analyze.figures(workloads,OUT)
report=(OUT/'REPORT.zh-CN.md').read_text().split('\n## 原生加速路径及相同ECE切口对照')[0]
(OUT/'REPORT.zh-CN.md').write_text(report)
runpy.run_path(str(ROOT/'plot_phases.py'),run_name='__main__')
assert sha(OUT/'WORKLOADS.json')==data_sha
(OUT/'PLOT-RECEIPT.json').write_text(json.dumps({'job':os.environ['SLURM_JOB_ID'],'data_unchanged_sha256':data_sha,
    'changes':'explicit Torch reference labels; prevent annotation/title overlap; no measurement changes',
    'calculation_source':str(OUT/'ANALYSIS-SOURCE.py'),'calculation_source_sha256':sha(OUT/'ANALYSIS-SOURCE.py'),
    'plot_source_sha256':{name:sha(ROOT/name) for name in ('analyze.py','fast_review.py','plot_phases.py','redraw.py')}},indent=2))
print('PLOT_LAYOUT_REVIEW_COMPLETED; DATA_UNCHANGED',flush=True)
