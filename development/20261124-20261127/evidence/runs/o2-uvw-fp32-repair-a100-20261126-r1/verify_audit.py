"""AMD acceptance requires every E/F, derivative, update and repetition role."""
import json,os,platform
from pathlib import Path
import torch
from common import ROOT,save,sha
from metrics import full
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
    source=Path((ROOT/'VERIFY-RESULT.txt').read_text().strip());manifest=json.loads((source/'MANIFEST.json').read_text())
    out=ROOT/'results'/('verify-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    cases={};all_failed=[]
    for case,ref_case in (('native-repeat','native'),('candidate','native'),('candidate-repeat','candidate')):
        records={};role_count=0
        for name,digest in manifest['cases'][case].items():
            assert sha(source/case/name)==digest
            assert sha(source/ref_case/name)==manifest['cases'][ref_case][name]
            a=torch.load(source/case/name,map_location='cpu',weights_only=False)
            r=torch.load(source/ref_case/name,map_location='cpu',weights_only=False)
            metrics=full(a,r);failed={k:v for k,v in metrics.items() if not v['passed']};role_count+=len(metrics)
            records[name]={'passed':not failed,'role_count':len(metrics),'failures':failed,
                'maximum_relative_l2':max((v.get('relative_l2',0) or 0) for v in metrics.values())}
            all_failed.extend({'case':case,'object':name,'role':k,**v} for k,v in failed.items())
            save(out/(case+'-'+name.replace('.pt','-METRICS.json')),metrics)
            del a,r
            print(case,name,len(failed),flush=True)
        cases[case]={'passed':all(v['passed'] for v in records.values()),'role_count':role_count,'records':records}
    for case in manifest['cases']:
        for step in range(3):assert json.loads((source/case/f'STEP-{step}.json').read_text())['parameter_ids_unchanged']
        ef=json.loads((source/case/'EF-CONDITION.json').read_text());tr=json.loads((source/case/'TRAIN-CONDITION.json').read_text())
        native_ef=json.loads((source/'native/EF-CONDITION.json').read_text());native_tr=json.loads((source/'native/TRAIN-CONDITION.json').read_text())
        for label,row,reference in (('ef',ef,native_ef),('train',tr,native_tr)):
            assert row['state_identity']==reference['state_identity'] and row['graph']==reference['graph']
            assert row['hardware']['gpu_uuid']==reference['hardware']['gpu_uuid']
    result={'status':'WHOLE_TRAINING_GATE_PASS' if not all_failed else 'WHOLE_TRAINING_GATE_FAIL',
        'node':platform.node(),'cases':cases,'failures':all_failed,'operator_source_sha256':manifest['operator_source_sha256'],
        'manifest_sha256':sha(source/'MANIFEST.json'),'gate':'unchanged component bound plus complete-object relative-L2<=1e-5; same own native losses'}
    save(out/'RESULT.json',result);(ROOT/'VERIFY-AUDIT.txt').write_text(str(out)+'\n')
    print(result['status'],'failed',len(all_failed),flush=True)
if __name__=='__main__':main()
