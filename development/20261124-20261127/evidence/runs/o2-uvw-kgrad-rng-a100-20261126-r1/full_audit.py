"""AMD original full-array gates; ULP is descriptive, never an added gate."""
import gc,json,os,platform
from pathlib import Path
import torch
from context import ROOT,BASE,save,sha
from metrics import full
from regression_audit import metric
def summary(m):
    return {'role_count':len(m),'failures':{k:v for k,v in m.items() if not v['passed']},
        'max_relative_l2':max(v.get('relative_l2',0) or 0 for v in m.values()),
        'max_abs':max(v.get('max_abs',0) for v in m.values()),
        'max_ulp':max(v.get('ulp',{}).get('max_ulp',0) for v in m.values())}
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
    out=ROOT/'results'/('full-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir();reports={};operator=set()
    for kind,manifest_name in (('REGRESSION','REGRESSION.json'),('GENERIC','MANIFEST.json')):
        source=Path((ROOT/(kind+'-RESULT.txt')).read_text().strip());manifest=json.loads((source/manifest_name).read_text());rows=[]
        operator.add(manifest['operator_source_sha256'])
        for index,row in enumerate(manifest['records']):
            assert sha(Path(row['path']))==row['sha256'];v=torch.load(row['path'],map_location='cpu',weights_only=False)
            a={k:metric(v['candidate'][k],r) for k,r in v['reference'].items()};repeat={k:metric(v['native_repeat'][k],r) for k,r in v['reference'].items()}
            assert all(x['passed'] for x in repeat.values());rows.append({'case':index,'identity':row,'roles':a,'native_repeat':repeat})
            del v;gc.collect()
        save(out/(kind+'.json'),rows)
        reports[kind]={'cases':len(rows),'roles':sum(len(r['roles']) for r in rows),'failures':sum(not v['passed'] for r in rows for v in r['roles'].values()),
            'epsilon_max':max(v['epsilon_max'] for r in rows for v in r['roles'].values()),'max_ulp':max(v['ulp'].get('max_ulp',0) for r in rows for v in r['roles'].values()),
            'weight_ulp_max':max((v['ulp']['max_ulp'] for r in rows for k,v in r['roles'].items() if k=='weight_adjoint'),default=0)}
    source=Path((ROOT/'QUALIFICATION-RESULT.txt').read_text().strip());manifest=json.loads((source/'MANIFEST.json').read_text());operator.add(manifest['operator_source_sha256'])
    conditions=json.loads((source/'CONDITIONS.json').read_text());frames=json.loads((source/'FRAMES.json').read_text());records={}
    def load(name):
        assert sha(source/name)==manifest['files'][name];return torch.load(source/name,map_location='cpu',weights_only=False)
    def compare(name,a,r):
        m=full(a,r);save(out/(name+'.json'),m);records[name]=summary(m)
    for role in ('ef','vjp'):
        r=load('native-'+role+'.pt')
        for backend in ('native-repeat','candidate','candidate-repeat'):compare(role+'-'+backend,load(backend+'-'+role+'.pt'),r)
        del r;gc.collect()
    for mode in manifest['modes']:
        for step in range(3):
            key=f'{mode}-native-step{step}';r=load(key+'.pt')
            for backend in ('native-repeat','candidate','candidate-repeat'):
                other=f'{mode}-{backend}-step{step}';compare(other,load(other+'.pt'),r)
                assert conditions[other]['rng']==conditions[key]['rng']
                assert conditions[other]['rng_at_next_step']==conditions[key]['rng_at_next_step']
                a=[(v['rng'],v['frame_sha256']) for v in frames if v['phase'].startswith(other+'-')]
                b=[(v['rng'],v['frame_sha256']) for v in frames if v['phase'].startswith(key+'-')]
                assert a==b,(key,other,'rotation phase mismatch')
            del r;gc.collect()
    for backend in manifest['backends']:
        for step in range(3):
            key=f'isolated-{backend}-step{step}';other=f'train-only-{backend}-step{step}'
            a=load(key+'.pt');b=load(other+'.pt');compare('isolation-'+backend+'-'+str(step),a['training'],b['training'])
            assert conditions[key]['rng']['train']==conditions[other]['rng']['train']
            assert conditions[key]['rng_at_next_step']==conditions[other]['rng_at_next_step']
            del a,b;gc.collect()
    # The native legacy control must reproduce the original failed experiment, not a reseeded replacement.
    old=Path((BASE/'VERIFY-RESULT.txt').read_text().strip())
    for step in range(3):
        v=load(f'legacy-native-step{step}.pt');prior=torch.load(old/'native'/f'TRAINING-{step}.pt',map_location='cpu',weights_only=False)
        compare('original-native-'+str(step),{k:v[k] for k in ('training','after_update_output')},prior)
        compare('original-native-mixed-'+str(step),v['mixed'],torch.load(old/'native'/f'MIXED-{step}.pt',map_location='cpu',weights_only=False))
        del v,prior;gc.collect()
    reports['WHOLE']={'records':records,'failures':sum(len(v['failures']) for v in records.values()),
        'roles':sum(v['role_count'] for v in records.values()),'max_ulp':max(v['max_ulp'] for v in records.values())}
    assert len(operator)==1
    passed=all(v['failures']==0 for v in reports.values())
    result={'status':'FULL_QUALIFICATION_PASS' if passed else 'FULL_QUALIFICATION_FAIL','node':platform.node(),
        'reports':reports,'operator_source_sha256':operator.pop(),'independent_legacy_three_steps':True,'actual_own_losses':True,
        'RNG_and_rotation_phases_match':True,'diagnostic_isolation_control':True,'historical_202_FAIL_unchanged':True}
    save(out/'RESULT.json',result);(ROOT/'FULL-AUDIT.txt').write_text(str(out)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
