"""Parse completed traces on a scheduled CPU node while GPU capture continues."""
import json,os,time,traceback
from pathlib import Path
import analyze
ROOT=Path(__file__).resolve().parent
assert os.environ.get('SLURM_JOB_ID')
target=ROOT/'results/analysis-118587/traces';done=set();errors={}
deadline=time.monotonic()+90*60
while time.monotonic()<deadline:
    for allocation in ('a100-118576','a100-118586','a100-118594','a100-118630'):
        for path in (ROOT/'results'/allocation).glob('*-trace/RESULT.json'):
            folder=path.parent
            if str(folder) in done:continue
            try:
                result=analyze.parse_trace(folder,target/folder.name)
                done.add(str(folder));print(json.dumps({'parsed':str(folder),'kernels':result['kernel_calls'],'kernel_ms':result['kernel_ms']}),flush=True)
            except Exception as error:
                errors[str(folder)]=str(error);print(traceback.format_exc(),flush=True)
    if (ROOT/'FAST-FINISH.txt').exists() or (ROOT/'WATCHER-STOP').exists():break
    time.sleep(20)
(ROOT/'prepared/TRACE-WATCHER.json').write_text(json.dumps({'job':os.environ['SLURM_JOB_ID'],'parsed':sorted(done),'errors':errors},indent=2))
