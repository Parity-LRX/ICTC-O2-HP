"""AMD complete-object gates, actual own losses and independent state transitions."""
import gc,json,os,platform
from pathlib import Path
import torch
from context import ROOT,BASE,PREVIOUS,SOURCE,save,sha
from metrics import full
def read(p):return json.loads(Path(p).read_text())
def summary(v):return {'role_count':len(v),'failures':{k:x for k,x in v.items() if not x['passed']},'max_relative_l2':max((x.get('relative_l2',0) or 0) for x in v.values()),'max_abs':max(x.get('max_abs',0) for x in v.values()),'max_ulp':max(x.get('ulp',{}).get('max_ulp',0) for x in v.values())}
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
    out=ROOT/'results'/('full-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir();records={}
    local_path=Path((ROOT/'LOCAL-AUDIT.txt').read_text().strip());local=read(local_path/'RESULT.json');assert local['status']=='LOCAL_BILINEAR_PASS'
    source=Path((ROOT/'QUALIFICATION-RESULT.txt').read_text().strip());manifest=read(source/'MANIFEST.json')
    assert local['operator_source_sha256']==manifest['operator_source_sha256'] and local['package_sha256']==manifest['package_sha256']
    prior=read(PREVIOUS/'FULL-PREFLIGHT.json');baseline=Path(prior['snapshot'])/'ictc-source'
    qualified=Path(json.loads((ROOT/'FULL-PREFLIGHT.json').read_text())['snapshot'])/'ictc-source'
    for name,digest in prior['files'].items():
        if name.startswith('ictc-source/'):
            assert sha(qualified/Path(name).relative_to('ictc-source'))==digest,(name,'8324 source changed')
    assert manifest['channel_source_sha256']=='8b2af4ac04906f2ed8ef8e6f040417b4aa93df84f6b897d02b7adea4b28ba585'
    c=read(source/'CONDITIONS.json');frames=read(source/'FRAMES.json');counts=read(source/'ADJOINT-CALLS.json')
    assert sha(source/'ADJOINT-CALLS.json')==manifest['adjoint_calls_sha256']
    for backend in ('candidate','candidate-repeat'):
        assert counts[backend+'-ef'].get('weight_K63024',0)==0 and counts[backend+'-ef']['pruned_K_outputs']==28
        for mode in manifest['modes']:
            for step in range(3):
                key=f'{mode}-{backend}-step{step}-train';assert counts[key]['weight_K10504']==56 and counts[key]['pruned_K_outputs']==28
                assert counts[key]['fused_bilinear_calls']>0
    def load(name):
        assert sha(source/name)==manifest['files'][name];return torch.load(source/name,map_location='cpu',weights_only=False)
    def compare(name,a,r):
        v=full(a,r);save(out/(name+'.json'),v);records[name]=summary(v)
    for role in ('ef','vjp'):
        r=load('native-'+role+'.pt')
        for backend in ('native-repeat','candidate','candidate-repeat'):compare(role+'-'+backend,load(backend+'-'+role+'.pt'),r)
        del r;gc.collect()
    for mode in manifest['modes']:
        for step in range(3):
            key=f'{mode}-native-step{step}';r=load(key+'.pt')
            for backend in ('native-repeat','candidate','candidate-repeat'):
                other=f'{mode}-{backend}-step{step}';compare(other,load(other+'.pt'),r)
                assert c[other]['rng']==c[key]['rng'] and c[other]['rng_at_next_step']==c[key]['rng_at_next_step']
                a=[(v['rng'],v['frame_sha256']) for v in frames if v['phase'].startswith(other+'-')]
                b=[(v['rng'],v['frame_sha256']) for v in frames if v['phase'].startswith(key+'-')];assert a==b
            del r;gc.collect()
    for backend in manifest['backends']:
        for step in range(3):
            key=f'isolated-{backend}-step{step}';other=f'train-only-{backend}-step{step}'
            a=load(key+'.pt');b=load(other+'.pt');compare('isolation-'+backend+'-'+str(step),a['training'],b['training'])
            assert c[key]['rng']['train']==c[other]['rng']['train'] and c[key]['rng_at_next_step']==c[other]['rng_at_next_step']
            del a,b;gc.collect()
    old=Path((BASE/'VERIFY-RESULT.txt').read_text().strip())
    for step in range(3):
        v=load(f'legacy-native-step{step}.pt');prior=torch.load(old/'native'/f'TRAINING-{step}.pt',map_location='cpu',weights_only=False)
        compare('original-native-'+str(step),{k:v[k] for k in ('training','after_update_output')},prior)
        compare('original-native-mixed-'+str(step),v['mixed'],torch.load(old/'native'/f'MIXED-{step}.pt',map_location='cpu',weights_only=False));del v,prior;gc.collect()
    reports={'LOCAL':{'receipt':str(local_path),'cases':local['cases'],'roles':local['roles'],'failures':len(local['failures']),'epsilon_max':local['epsilon_max'],'max_ulp':local['max_ulp']},'WHOLE':{'records':records,'failures':sum(len(v['failures']) for v in records.values()),'roles':sum(v['role_count'] for v in records.values()),'max_ulp':max(v['max_ulp'] for v in records.values())}}
    r={'status':'FULL_QUALIFICATION_PASS' if all(v['failures']==0 for v in reports.values()) else 'FULL_QUALIFICATION_FAIL','reports':reports,'package_sha256':manifest['package_sha256'],'operator_source_sha256':manifest['operator_source_sha256'],'channel_source_sha256':manifest['channel_source_sha256'],'independent_legacy_three_steps':True,'actual_own_losses':True,'RNG_and_rotation_phases_match':True,'historical_FAIL_unchanged':True,'node':platform.node(),'adjoint_calls':counts}
    save(out/'RESULT.json',r);(ROOT/'FULL-AUDIT.txt').write_text(str(out)+'\n');print(json.dumps(r),flush=True)
if __name__=='__main__':main()
