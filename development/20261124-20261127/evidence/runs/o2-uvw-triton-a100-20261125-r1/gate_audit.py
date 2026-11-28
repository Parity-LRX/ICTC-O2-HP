"""Scheduled CPU review of completed network captures while GPU collection continues."""
import json,os,sys
from pathlib import Path
import torch
from whole import compare_tree,save
ROOT=Path(__file__).resolve().parent
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(2)
    mode=sys.argv[1];gpu=ROOT/'results/a100-119167';out=ROOT/'results'/('gate-audit-'+os.environ['SLURM_JOB_ID']);out.mkdir()
    ref=torch.load(gpu/(mode+'-native-gate')/'OBJECTS.pt',map_location='cpu',weights_only=False)
    summary={}
    for backend in ('native','tiled','ictc'):
        folder=gpu/(mode+'-'+backend+'-gate')
        if not (folder/'OBJECTS.pt').exists():continue
        actual=torch.load(folder/'OBJECTS.pt',map_location='cpu',weights_only=False)
        repeat=torch.load(folder/'REPEAT.pt',map_location='cpu',weights_only=False)
        metrics=compare_tree(actual,ref);self_repeat=compare_tree(repeat,actual)
        failed={k:v for k,v in metrics.items() if not v['passed']}
        rfailed={k:v for k,v in self_repeat.items() if not v['passed']}
        save(out/(backend+'-METRICS.json'),{'roles':metrics,'failed':failed,'self_repeat_failed':rfailed})
        summary[backend]={'all_roles_pass':not failed,'failed_roles':len(failed),'self_repeat_failed':len(rfailed),
            'failures':dict(list(failed.items())[:8])}
    save(out/'SUMMARY.json',{'mode':mode,'summary':summary});print(json.dumps({'mode':mode,'summary':summary}),flush=True)
if __name__=='__main__':main()
