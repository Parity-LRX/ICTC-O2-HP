"""Parse immutable completed traces on CPU while the single GPU keeps collecting."""
import importlib.util,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    assert os.environ.get('SLURM_JOB_ID')
    spec=importlib.util.spec_from_file_location('parser',ROOT.parents[1]/'runs/o2-paired-diagnostic-a100-20261124-r1/analyze.py')
    parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser);parser.ICTC_NAMES.add('_uvw_mm')
    gpu=ROOT/'results/a100-119167';done={}
    labels=[m+'-'+b for b in ('native','tiled','ictc') for m in ('ef','train')]
    while True:
        for label in labels:
            if label in done:continue
            folder=gpu/(label+'-trace');receipt=folder/'RESULT.json'
            if not receipt.exists():continue
            try:result=json.loads(receipt.read_text())
            except json.JSONDecodeError:continue
            if result.get('status')!='DIAGNOSTIC_COMPLETED':continue
            summary=parser.parse_trace(folder,ROOT/'trace-cache'/label)
            done[label]={'kernel_ms':summary['kernel_ms'],'kernel_calls':summary['kernel_calls']}
            print(json.dumps({'completed_trace':label,**done[label]}),flush=True)
            (ROOT/'TRACE-PARSE-PROGRESS.json').write_text(json.dumps(done,indent=2)+'\n')
        if len(done)==len(labels):break
        if (gpu/'FINISH.json').exists():break
        time.sleep(10)
    (ROOT/'TRACE-PARSE-FINISH.json').write_text(json.dumps({'job':os.environ['SLURM_JOB_ID'],'parsed':done,'expected':6},indent=2)+'\n')
if __name__=='__main__':main()
