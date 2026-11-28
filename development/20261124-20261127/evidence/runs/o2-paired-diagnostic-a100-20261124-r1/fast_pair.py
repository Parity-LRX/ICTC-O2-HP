"""Matched upstream acceleration and RRA module-route controls, no new kernels."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;JOB=os.environ['SLURM_JOB_ID']
OUT=ROOT/'results'/('a100-'+JOB);CONTROL=ROOT/'snapshots'/('control-'+JOB)
def save(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def main():
    OUT.mkdir(exist_ok=False);CONTROL.mkdir(exist_ok=False)
    for name in ('diagnose.py','fast_pair.py','whole.py','tece.py','bounded.py','fast-pair.sbatch','CONDITIONS.zh-CN.md'):shutil.copy2(ROOT/name,CONTROL/name)
    shutil.copytree(ROOT/'adapter',CONTROL/'adapter')
    seals={}
    for p in CONTROL.rglob('*'):
        if p.is_file():
            with p.open('rb') as f:seals[str(p.relative_to(CONTROL))]=hashlib.file_digest(f,'sha256').hexdigest()
    save(OUT/'SOURCE-SEAL.json',seals)
    previous=Path((ROOT/'LATEST-RESULT.txt').read_text().strip())
    progress=json.loads((previous/'PROGRESS.json').read_text());save(OUT/'RESUME.json',{'previous':str(previous),'inherited_rows':len(progress)})
    save(OUT/'STARTUP.json',{'job':JOB,'node':os.environ['SLURM_JOB_NODELIST'],'nodes':1,'gpus':1,'start_unix':time.time(),'purpose':'matched native acceleration and module-route diagnostic only'})
    environment=os.environ.copy();environment.update(O2_WHOLE_ROOT=str(ROOT),TACE_USE_EQX='1',TACE_USE_COMPILE='0',TACE_USE_CUE='0',TACE_USE_OEQ='0',EQX_USE_CUDA_GRAPH='0')
    environment.update(TRITON_CACHE_DIR=str(OUT/'triton-cache'),TORCHINDUCTOR_CACHE_DIR=str(OUT/'inductor-cache'),MPLCONFIGDIR=str(OUT/'mpl-cache'))
    def run(route,mode,backend,action,common):
        folder=OUT/('-'.join((route,mode,backend,action)));folder.mkdir()
        command=[sys.executable,'-u',str(CONTROL/'diagnose.py'),route,mode,backend,action,str(common),str(folder)]
        started=time.time()
        with (folder/'process.log').open('w') as log:
            try:code=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=1200).returncode
            except subprocess.TimeoutExpired:code=124
        error=json.loads((folder/'ERROR.json').read_text()) if (folder/'ERROR.json').exists() else None
        row={'route':route,'mode':mode,'backend':backend,'action':action,'folder':str(folder),'exit_code':code,'elapsed_s':time.time()-started,'error':error['message'] if error else None}
        progress.append(row);save(OUT/'PROGRESS.json',progress);print(json.dumps(row),flush=True);return code
    for route in ('tece-canonical','tece-full'):
        for mode in ('ef','train'):
            successful=[r for r in progress if r['route']==route and r['mode']==mode and r['backend']=='native-eqx' and r['exit_code']==0]
            if len(successful)!=3:
                print(json.dumps({'skip_fast_pair':route+'/'+mode,'reason':'upstream accelerated complete measurement unavailable; primary Torch/ICTC pair already preserved'}),flush=True);continue
            common=Path(next(r['folder'] for r in progress if r['route']==route and r['mode']==mode and r['backend']=='native' and r['action']=='prepare'))
            reference='native-eqx-module' if route=='tece-canonical' else 'native-eqx-paired'
            for backend in (reference,'ictc-eqx'):
                for action in ('time','memory','trace'):
                    code=run(route,mode,backend,action,common)
                    if code:break
    save(OUT/'FINISH.json',{'job':JOB,'end_unix':time.time(),'tasks':len(progress)})
    (ROOT/'LATEST-RESULT.txt').write_text(str(OUT)+'\n')
    (ROOT/'FAST-FINISH.txt').write_text(str(OUT)+'\n')
if __name__=='__main__':main()
