"""One A100 allocation; independent serial processes for all paired captures."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def save(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def main():
    job=os.environ['SLURM_JOB_ID'];out=ROOT/'results'/('a100-'+job);out.mkdir(parents=True)
    preflight=json.loads((ROOT/'PREFLIGHT.json').read_text())
    snapshot=Path(preflight['snapshot'])
    for name,digest in preflight['files'].items():
        with (snapshot/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,name
    env=os.environ.copy();env.update(O2_WHOLE_ROOT=str(ROOT),O2_CANDIDATE_COMMIT=preflight['candidate_commit'],
        O2_UVW_SOURCE=str(snapshot/'fork-src'),
        TRITON_CACHE_DIR=str(out/'triton-cache'),MPLCONFIGDIR=str(out/'mpl-cache'),
        TACE_USE_EQX='0',TACE_USE_COMPILE='0',TACE_USE_CUE='0',TACE_USE_OEQ='0',EQX_USE_CUDA_GRAPH='0')
    save(out/'STARTUP.json',{'job':job,'start_unix':time.time(),'node':os.environ['SLURM_JOB_NODELIST'],
        'nodes':1,'gpus':1,'preflight':str(ROOT/'PREFLIGHT.json'),'commit':preflight['candidate_commit']})
    progress=[]
    def run(label,command,overrides=None):
        folder=out/label;folder.mkdir(exist_ok=True)
        environment=env.copy();environment.update(overrides or {})
        started=time.time()
        with (folder/'process.log').open('w') as log:
            try:code=subprocess.run(command+ [str(folder)],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=1500).returncode
            except subprocess.TimeoutExpired:code=124
        row={'label':label,'folder':str(folder),'exit_code':code,'elapsed_s':time.time()-started}
        progress.append(row);save(out/'PROGRESS.json',progress);print(json.dumps(row),flush=True)
        return code
    code=run('operator-validation',[sys.executable,'-u',str(snapshot/'validate.py')])
    if code:
        save(out/'FINISH.json',{'status':'STOPPED_OPERATOR_VALIDATION','job':job});return
    # Full role-separated network checks, independent from diagnostic captures.
    for mode in ('ef','train'):
        for backend in ('native','tiled','ictc'):
            run(f'{mode}-{backend}-gate',[sys.executable,'-u',str(snapshot/'network_check.py'),mode,backend,preflight['common_states'][mode]],
                {'O2_UVW_TILED':'1' if backend=='tiled' else '0'})
    for backend in ('native','tiled','ictc'):
        for mode in ('ef','train'):
            for action in ('time','memory','trace'):
                run(f'{mode}-{backend}-{action}',[sys.executable,'-u',str(snapshot/'diagnose.py'),'v3',mode,
                    'ictc' if backend=='tiled' else backend,action,preflight['common_states'][mode]],
                    {'O2_UVW_TILED':'1' if backend=='tiled' else '0'})
    save(out/'FINISH.json',{'status':'CAPTURES_FINISHED','job':job,'end_unix':time.time()})
    (ROOT/'LATEST-RESULT.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
