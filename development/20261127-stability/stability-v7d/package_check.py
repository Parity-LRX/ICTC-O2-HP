"""Check that a built wheel carries the FP32 execution plans and imports on CPU."""
import hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
root=Path(__file__).resolve().parent;repo=root.parents[1]/'ICTC-O2-HP'
out=root/'package-checks'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True)
with (out/'build.log').open('w') as log:
    subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--no-index',
                    '--wheel-dir',str(out),str(repo)],stdout=log,stderr=subprocess.STDOUT,check=True)
wheel,=out.glob('*.whl')
with zipfile.ZipFile(wheel) as archive:
    for name in ['bilinear_plans.json','fp32_reduction_plans.json']:
        assert archive.read('ictc_o2_hp/'+name)==(repo/'src/ictc_o2_hp'/name).read_bytes()
    archive.extractall(out/'installed')
code="import sys;sys.path.insert(0,sys.argv[1]);from ictc_o2_hp.fp32_plans import plans;assert plans()['angular|64,64,64,32,4,1,4096,64,4096,64,1,2048,32,1']['parts']==54;print('INSTALLED_PLAN_IMPORT_PASS',len(plans()))"
result=subprocess.check_output([sys.executable,'-c',code,str(out/'installed')],cwd=out,text=True)
record={'status':'PASS','wheel':str(wheel),'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'import':result.strip()}
(out/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n');(root/'PACKAGE-CHECK.txt').write_text(str(out)+'\n')
print(json.dumps(record),flush=True)
