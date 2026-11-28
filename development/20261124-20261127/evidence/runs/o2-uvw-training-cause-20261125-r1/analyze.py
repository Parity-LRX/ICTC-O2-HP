"""Scheduled, complete-array analysis of controlled backend ablations."""
import csv,json,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[1]
OLD=WS/'runs/o2-uvw-triton-a100-20261125-r1'
os.environ['O2_WHOLE_ROOT']=str(OLD)
sys.path.insert(0,str(OLD/'snapshots/preflight-119174'))
from whole import metric,save,sha
CASES=('native','native-layout','native-flat','tiled','T-N','N-T','T-T-N','N-N-T')
def csvwrite(path,rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def dictionary(a,r):
    assert a.keys()==r.keys()
    return {k:metric(a[k],r[k]) for k in r}
def summary(metrics):
    failed={k:v for k,v in metrics.items() if not v['passed']}
    values=[v for v in metrics.values() if 'relative_l2' in v]
    return {'roles':len(metrics),'failed':len(failed),'failed_parameter_roles':sum(k!='coordinates' for k in failed),
        'failed_coordinate_roles':int('coordinates' in failed),'component_fail_count':sum(v.get('component_fail_count',0) for v in metrics.values()),
        'maximum_relative_l2':max((v['relative_l2'] for v in values if v['relative_l2'] is not None),default=0),
        'failed_metrics':failed}
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4)
    source=Path((ROOT/'GPU-RESULT.txt').read_text().strip());assert (source/'FINISH.json').exists()
    out=ROOT/'results'/('analysis-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    load=lambda case:torch.load(source/(case+'.pt'),map_location='cpu',weights_only=False)
    ref=load('native');results={};allrows=[];decomp=[];old_comparisons={};layout_changes={};residuals={}
    old_fail=json.loads((OLD/'results/analysis-119186/train-tiled-GATE.json').read_text())['failed']
    focus={p.split('/mixed/')[-1] for p in old_fail if '/mixed/' in p}
    for case in CASES:
        a=ref if case=='native' else load(case)
        groups={'gradients':dictionary(a['total']['gradients'],ref['total']['gradients']),
            'force_original':dictionary(a['force_original']['mixed'],ref['force_original']['mixed']),
            'force_direct':dictionary(a['force_direct'],ref['force_direct']),
            'force_self_vjp':dictionary(a['force_self_vjp'],ref['force_self_vjp']),
            'force_shared_vjp':dictionary(a['force_shared_vjp'],ref['force_shared_vjp']),
            'direct_vs_original':dictionary(a['force_direct'],a['force_original']['mixed']),
            'direct_vs_self':dictionary(a['force_direct'],a['force_self_vjp'])}
        for group,metrics in groups.items():
            for role,item in metrics.items():allrows.append({'case':case,'group':group,'role':role,
                'passed':item['passed'],'relative_l2':item.get('relative_l2'),'max_abs':item.get('max_abs'),
                'component_fail_count':item.get('component_fail_count'),'reference_norm':item.get('reference_norm')})
        result={g:summary(m) for g,m in groups.items()}
        result['output']={k:metric(a['total']['output'][k],ref['total']['output'][k]) for k in ref['total']['output']}
        result['loss']=metric(a['total']['loss'],ref['total']['loss'])
        result['force_cotangent']=metric(a['self_cotangent'],ref['self_cotangent'])
        result['force_output_bitwise_native']=torch.equal(a['force_output'],ref['force_output'])
        result['force_original_vs_fresh_force']=metric(a['force_original']['force'],a['force_output'])
        result['force_cotangent_norm']=float(a['self_cotangent'].norm())
        force_inputs=a['loss_inputs']['forces'];err=force_inputs['pred']-force_inputs['target'];lengths=err.norm(dim=-1)
        residuals[case]={'n':lengths.numel(),'min':float(lengths.min()),'max':float(lengths.max()),
            'quantiles':{str(q):float(torch.quantile(lengths,q)) for q in (0.0,0.01,0.1,0.5,1.0)},
            'error_change_max_abs':float((err-(ref['loss_inputs']['forces']['pred']-ref['loss_inputs']['forces']['target'])).abs().max()),
            'energy_mae_sign_flips':int(((a['loss_inputs']['energy']['pred']-a['loss_inputs']['energy']['target']).sign()!=
                (ref['loss_inputs']['energy']['pred']-ref['loss_inputs']['energy']['target']).sign()).sum()),
            'stress_mae_sign_flips':int(((a['loss_inputs']['stress']['pred']-a['loss_inputs']['stress']['target']).sign()!=
                (ref['loss_inputs']['stress']['pred']-ref['loss_inputs']['stress']['target']).sign()).sum())}
        layout_changes[case]={n:{'native':ref['layouts'][n],'case':v} for n,v in a['layouts'].items() if v!=ref['layouts'][n]}
        for role in sorted(focus):
            d=a['force_direct'][role]-ref['force_direct'][role]
            fixed=a['force_shared_vjp'][role]-ref['force_shared_vjp'][role]
            cot=a['force_self_vjp'][role]-a['force_shared_vjp'][role]
            traversal=a['force_direct'][role]-a['force_self_vjp'][role]
            native_traversal=ref['force_shared_vjp'][role]-ref['force_direct'][role]
            dn=d.norm();rn=ref['force_direct'][role].norm();d2=(d*d).sum()
            row={'case':case,'role':role,'reference_norm':float(rn),'direct_difference_norm':float(dn)}
            for label,t in [('fixed_native_cotangent',fixed),('cotangent_change',cot),('candidate_traversal',traversal),('native_traversal',native_traversal)]:
                row[label+'_norm']=float(t.norm());row[label+'_relative_l2']=float(t.norm()/rn) if rn else None
                row[label+'_projection_on_direct_error']=float((t*d).sum()/d2) if d2 else 0
            row['telescoping_residual_norm']=float((d-fixed-cot-traversal-native_traversal).norm())
            decomp.append(row)
        if case in ('native','tiled'):
            old=torch.load(OLD/('results/a100-119167/train-'+case+'-gate/OBJECTS.pt'),map_location='cpu',weights_only=False)
            old_comparisons[case]={'output':{k:metric(a['total']['output'][k],old['output'][k]) for k in old['output']},
                'gradients':summary(dictionary(a['total']['gradients'],old['gradients'])),
                'force_original':summary(dictionary(a['force_original']['mixed'],old['force_mixed']['mixed']))}
            del old
        results[case]=result
        if case!='native':del a
    controls={}
    for label,l,r in [('TTN_vs_tiled','T-T-N','tiled'),('NNT_vs_nativeflat','N-N-T','native-flat')]:
        a=load(l);b=load(r)
        controls[label]={'force_bitwise_equal':torch.equal(a['force_output'],b['force_output']),
            'force_metric':metric(a['force_output'],b['force_output']),
            'energy_bitwise_equal':torch.equal(a['total']['output']['energy'],b['total']['output']['energy']),
            'cotangent_bitwise_equal':torch.equal(a['self_cotangent'],b['self_cotangent'])};del a,b
    csvwrite(out/'ALL-METRICS.csv',allrows);csvwrite(out/'ERROR-DECOMPOSITION.csv',decomp)
    save(out/'ABLATIONS.json',results);save(out/'LOSS-RESIDUALS.json',residuals);save(out/'LAYOUT-CHANGES.json',layout_changes)
    save(out/'REPRODUCTION.json',old_comparisons);save(out/'STAGED-CONTROLS.json',controls)
    compact={case:{g:{k:v for k,v in data[g].items() if k!='failed_metrics'} for g in
        ('gradients','force_original','force_direct','force_self_vjp','force_shared_vjp','direct_vs_original','direct_vs_self')} for case,data in results.items()}
    save(out/'SUMMARY.json',compact)
    save(out/'MANIFEST.json',{'source':str(source),'sources':{p.name:sha(p) for p in source.glob('*.pt')},
        'analysis_source_sha256':sha(Path(__file__)),'statistics_dtype':'float32','complete_object_metrics':True})
    (ROOT/'ANALYSIS-RESULT.txt').write_text(str(out)+'\n');print(json.dumps(compact),flush=True)
if __name__=='__main__':main()
