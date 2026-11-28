import gc,json,os
from pathlib import Path
import torch
from context import ROOT,BASE,save,sha
from metrics import full
def brief(m):return {'roles':len(m),'failed':{k:v for k,v in m.items() if not v['passed']},'max_R':max(v.get('relative_l2',0) or 0 for v in m.values()),'max_abs':max(v.get('max_abs',0) for v in m.values())}
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
    source=Path((ROOT/'PROTOCOL-RESULT.txt').read_text().strip());manifest=json.loads((source/'MANIFEST.json').read_text());conditions=json.loads((source/'CONDITIONS.json').read_text())
    out=ROOT/'results'/('protocol-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir();records={}
    old=Path((BASE/'VERIFY-RESULT.txt').read_text().strip())
    for backend in ('native','candidate'):
        for step in range(3):
            objects={}
            for mode in ('legacy','isolated','train-only'):
                name=f'{mode}-{backend}-step{step}.pt';assert sha(source/name)==manifest['files'][name]
                objects[mode]=torch.load(source/name,map_location='cpu',weights_only=False)
            m=full(objects['isolated']['training'],objects['train-only']['training']);records[f'isolation-{backend}-{step}']=brief(m);save(out/f'ISOLATION-{backend}-{step}.json',m)
            for part in ('train','after'):
                assert conditions[f'isolated-{backend}-step{step}']['rng'][part]==conditions[f'train-only-{backend}-step{step}']['rng'][part]
            assert conditions[f'isolated-{backend}-step{step}']['rng_at_next_step']==conditions[f'train-only-{backend}-step{step}']['rng_at_next_step']
            original=torch.load(old/backend/f'TRAINING-{step}.pt',map_location='cpu',weights_only=False)
            m=full({k:objects['legacy'][k] for k in ('training','after_update_output')},original);records[f'legacy-{backend}-{step}']=brief(m)
            oldmixed=torch.load(old/backend/f'MIXED-{step}.pt',map_location='cpu',weights_only=False)
            m=full(objects['legacy']['mixed'],oldmixed);records[f'legacy-mixed-{backend}-{step}']=brief(m)
            del objects,original,oldmixed;gc.collect()
    result={'status':'PROTOCOL_ISOLATION_PASS' if all(not r['failed'] for r in records.values()) else 'PROTOCOL_ISOLATION_FAIL',
        'records':records,'actual_rng_equal':True,'operator_source_sha256':manifest['operator_source_sha256'],'baseline_expected_legacy_FAIL_preserved':True}
    save(out/'RESULT.json',result);(ROOT/'PROTOCOL-AUDIT.txt').write_text(str(out)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
