"""Scheduled CPU audit of frozen complete-array training receipts and objects."""
import csv,hashlib,json,os,sys
from collections import Counter
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[1]
OLD=WS/'runs/o2-uvw-triton-a100-20261125-r1'
sys.path.insert(0,str(OLD/'snapshots/preflight-119174'))
os.environ['O2_WHOLE_ROOT']=str(OLD)
from whole import metric,save
def read(p):return json.loads(p.read_text())
def csvwrite(path,rows):
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4)
    out=ROOT/'results'/('cpu-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    gate=read(OLD/'results/analysis-119186/train-tiled-GATE.json');failed=gate['failed']
    ref=torch.load(OLD/'results/a100-119167/train-native-gate/OBJECTS.pt',map_location='cpu',weights_only=False)
    cand=torch.load(OLD/'results/a100-119167/train-tiled-gate/OBJECTS.pt',map_location='cpu',weights_only=False)
    def value(tree,path):
        for part in path.strip('/').split('/'):tree=tree[part]
        return tree
    rows=[];components=[]
    for name,item in failed.items():
        a=value(cand,name).float();r=value(ref,name).float();d=a-r
        budget=2e-5+2e-5*r.abs();bad=d.abs()>budget
        row={'role':name,'shape':str(list(r.shape)),**item,
             'epsilon_max':float(d.abs().max()/torch.maximum(a.abs().max(),r.abs().max()).clamp_min(1e-7)),
             'max_component_budget_ratio':float((d.abs()/budget).max())}
        rows.append(row)
        for index in bad.nonzero().tolist():
            idx=tuple(index);components.append({'role':name,'index':str(index),'reference':float(r[idx]),
                'candidate':float(a[idx]),'difference':float(d[idx]),'budget':float(budget[idx]),'ratio':float(d[idx].abs()/budget[idx])})
    csvwrite(out/'FAILED-ROLES.csv',rows)
    if components:csvwrite(out/'FAILED-COMPONENTS.csv',components)
    forward={key:metric(cand['output'][key],ref['output'][key]) for key in ref['output']}
    forward['loss']=metric(cand['loss'],ref['loss'])
    kgrad={n:metric(cand['gradients'][n],r) for n,r in ref['gradients'].items() if n.endswith('.K')}
    kmixed={n:metric(cand['force_mixed']['mixed'][n],r) for n,r in ref['force_mixed']['mixed'].items() if n.endswith('.K')}
    cancellation={}
    for path in failed:
        if not path.startswith('/gradients/'):continue
        name=path[len('/gradients/'):]
        g=ref['gradients'][name].float();f=ref['force_mixed']['mixed'][name].float();other=g-f
        cancellation[name]={'total_norm':float(g.norm()),'force_norm':float(f.norm()),'other_residual_norm':float(other.norm()),
            'cancellation_proxy':float((f.norm()+other.norm())/g.norm()),
            'force_total_cosine':float((f*g).sum()/(f.norm()*g.norm())),
            'scope':'other = saved full gradient minus saved force-only gradient; floating-point proxy, not separately evaluated E/stress gradients'}
    summary={'source':'frozen119167/119186','failed_roles':len(rows),'groups':dict(Counter(n.split('/')[1] for n in failed)),
        'coordinate_coordinate_failed':sum(n.endswith('/coordinates') for n in failed),
        'coordinate_parameter_failed':sum(n.startswith('/force_mixed/mixed/') and not n.endswith('/coordinates') for n in failed),
        'component_failed_roles':sum(r['component_fail_count']>0 for r in rows),'failed_components':len(components),
        'training_forward':forward,'K_gradient_roles':len(kgrad),'K_gradient_failed':{k:v for k,v in kgrad.items() if not v['passed']},
        'K_mixed_roles':len(kmixed),'K_mixed_failed':{k:v for k,v in kmixed.items() if not v['passed']},'cancellation_proxy':cancellation,
        'relative_l2_max':max(r['relative_l2'] for r in rows),'reference_norm_min':min(r['reference_norm'] for r in rows),
        'block_counts':dict(Counter(n.split('blocks.')[1].split('.')[0] if 'blocks.' in n else 'outside_blocks' for n in failed))}
    save(out/'SUMMARY.json',summary)
    (ROOT/'CPU-RESULT.txt').write_text(str(out)+'\n')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
