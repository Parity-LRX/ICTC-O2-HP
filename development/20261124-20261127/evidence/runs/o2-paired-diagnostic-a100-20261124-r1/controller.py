"""One allocation, one GPU, all backends and measurements strictly serial."""
import hashlib,json,os
from pathlib import Path
import shutil,subprocess,sys,time

ROOT=Path(__file__).resolve().parent
JOB=os.environ['SLURM_JOB_ID'];OUT=ROOT/'results'/('a100-'+JOB)
CONTROL=ROOT/'snapshots'/('control-'+JOB)
def save(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str)+'\n')
def main():
    OUT.mkdir(parents=True,exist_ok=False);CONTROL.mkdir(parents=True,exist_ok=False)
    for name in ('diagnose.py','controller.py','whole.py','tece.py','bounded.py','diagnostic.sbatch','CONDITIONS.zh-CN.md'):
        shutil.copy2(ROOT/name,CONTROL/name)
    shutil.copytree(ROOT/'adapter',CONTROL/'adapter')
    seals={}
    for path in CONTROL.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:seals[str(path.relative_to(CONTROL))]=hashlib.file_digest(stream,'sha256').hexdigest()
    save(OUT/'SOURCE-SEAL.json',seals)
    save(OUT/'STARTUP.json',{'job':JOB,'node':os.environ['SLURM_JOB_NODELIST'],'nodes':1,'gpus':1,
          'start_unix':time.time(),'numerical_gate':'not applied to diagnostic profiling','maximum_gpu_h':3})
    environment=os.environ.copy();environment['O2_WHOLE_ROOT']=str(ROOT)
    environment['TRITON_CACHE_DIR']=str(OUT/'triton-cache')
    environment['TORCHINDUCTOR_CACHE_DIR']=str(OUT/'inductor-cache')
    environment['MPLCONFIGDIR']=str(OUT/'mpl-cache')
    environment.update(TACE_USE_EQX='0',TACE_USE_COMPILE='0',TACE_USE_CUE='0',TACE_USE_OEQ='0',EQX_USE_CUDA_GRAPH='0')
    progress=[];inherited={}
    resume=ROOT/'RESUME-FROM.txt'
    if resume.exists():
        previous=Path(resume.read_text().strip())
        # Import only fully completed primary pairs. The first allocation's E/F
        # has no optimizer update; old partial training is never inherited.
        rows=json.loads((previous/'PROGRESS.json').read_text())
        selected=[]
        for key in ((r,m) for r in ('v3','tece-canonical','tece-full') for m in ('ef','train')):
            subset=[r for r in rows if (r['route'],r['mode'])==key and r['exit_code']==0]
            if len(subset)!=7:continue
            if key[1]=='train':
                conditions=[json.loads((Path(r['folder'])/'CONDITION.json').read_text()) for r in subset if r['action']!='prepare']
                assert all(c.get('complete_restored_state_verified') for c in conditions)
            selected.extend(subset);inherited[key]=Path(next(r['folder'] for r in subset if r['action']=='prepare'))
        assert len(inherited)==6,'Final controls require all six complete primary pairs'
        progress.extend(selected)
        save(OUT/'RESUME.json',{'previous':str(previous),'inherited_rows':selected,'optimizer_alias_affects_inherited_EF':False})
    def run(route,mode,backend,action,common):
        folder=OUT/(route+'-'+mode+'-'+backend+'-'+action);folder.mkdir(exist_ok=False)
        env=environment.copy()
        if backend=='native-eqx':env['TACE_USE_EQX']='1'
        command=[sys.executable,'-u',str(CONTROL/'diagnose.py'),route,mode,backend,action,str(common),str(folder)]
        started=time.time()
        with (folder/'process.log').open('w') as log:
            try:code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200).returncode
            except subprocess.TimeoutExpired:code=124
        error=json.loads((folder/'ERROR.json').read_text()) if (folder/'ERROR.json').exists() else None
        row={'route':route,'mode':mode,'backend':backend,'action':action,'folder':str(folder),'exit_code':code,
             'elapsed_s':time.time()-started,'error':error['message'] if error else None}
        progress.append(row);save(OUT/'PROGRESS.json',progress);print(json.dumps(row),flush=True)
        return folder,code
    common_states=dict(inherited)
    for route in ('v3','tece-canonical','tece-full'):
        for mode in ('ef','train'):
            if (route,mode) in inherited:continue
            common,code=run(route,mode,'native','prepare',OUT/'unused')
            if code:continue
            common_states[(route,mode)]=common
            for backend in ('native','ictc'):
                for action in ('time','memory','trace'):run(route,mode,backend,action,common)
    # Absolute upstream V3 controls retain original, unprojected parameters and grid.
    for mode in ('ef','train'):
        for backend in ('original-default','original-recompute'):
            common,code=run('v3-original',mode,backend,'prepare',OUT/'unused')
            if code:continue
            failures=[]
            for action in ('time','memory','trace'):
                folder,code=run('v3-original',mode,backend,action,common);failures.append(code)
            if not any(failures):break
    # Optional upstream accelerated controls use the same common states, with no fallback.
    for route in ('tece-canonical','tece-full'):
        for mode in ('ef','train'):
            if (route,mode) not in common_states:continue
            for action in ('time','memory','trace'):
                folder,code=run(route,mode,'native-eqx',action,common_states[(route,mode)])
                if code:break
    save(OUT/'FINISH.json',{'job':JOB,'end_unix':time.time(),'tasks':len(progress)})
    (ROOT/'LATEST-RESULT.txt').write_text(str(OUT)+'\n')
    (ROOT/'WATCHER-STOP').write_text('All GPU capture and controls completed in '+JOB+'\n')

if __name__=='__main__':main()
