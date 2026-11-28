"""Scheduled synthesis; distinguish actual loss from linearized diagnostic."""
import csv,json,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[1];OLD=WS/'runs/o2-uvw-triton-a100-20261125-r1'
os.environ['O2_WHOLE_ROOT']=str(OLD);sys.path.insert(0,str(OLD/'snapshots/preflight-119174'))
from whole import metric,save,sha
def read(path):return json.loads(path.read_text())
def dictionary(a,r):return {k:metric(a[k],r[k]) for k in r}
def compact(m):
    return {'failed':{k:v for k,v in m.items() if not v['passed']},
        'maximum_relative_l2':max((v.get('relative_l2',0) or 0) for v in m.values())}
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4)
    source=Path((ROOT/'GPU-RESULT.txt').read_text().strip());analysis=Path((ROOT/'ANALYSIS-RESULT.txt').read_text().strip())
    total=Path((ROOT/'TOTAL-RESULT.txt').read_text().strip());out=ROOT/'results'/('synthesis-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    records={};ref=torch.load(total/'native-direct.pt',weights_only=False,map_location='cpu')
    for case in ('native','tiled'):
        direct=torch.load(total/(case+'-direct.pt'),weights_only=False,map_location='cpu')
        own=torch.load(total/(case+'-own_linear.pt'),weights_only=False,map_location='cpu')
        shared=torch.load(total/(case+'-shared_linear.pt'),weights_only=False,map_location='cpu')
        records[case]={'direct_vs_native_direct':compact(dictionary(direct['gradients'],ref['gradients'])),
            'own_vs_native_direct':compact(dictionary(own['gradients'],ref['gradients'])),
            'shared_vs_native_direct':compact(dictionary(shared['gradients'],ref['gradients'])),
            'direct_vs_own':compact(dictionary(direct['gradients'],own['gradients'])),
            'loss_cotangents':{k:{**metric(direct['cotangent'][k],ref['cotangent'][k]),
                'bitwise_equal':torch.equal(direct['cotangent'][k],ref['cotangent'][k])} for k in ref['cotangent']}}
        if case=='tiled':
            before=read(OLD/'results/analysis-119186/train-tiled-GATE.json')['failed']
            focus=[p[len('/gradients/'):] for p in before if p.startswith('/gradients/')]
            records[case]['original_four']={n:{'direct':metric(direct['gradients'][n],ref['gradients'][n]),
                'shared':metric(shared['gradients'][n],ref['gradients'][n])} for n in focus}
            first=torch.load(source/'tiled.pt',map_location='cpu',weights_only=False)
            records[case]['direct_vs_first_probe']=compact(dictionary(direct['gradients'],first['total']['gradients']))
            del first
        del direct,own,shared
    raw=torch.load(source/'tiled.pt',map_location='cpu',weights_only=False)
    reference=torch.load(source/'native.pt',map_location='cpu',weights_only=False)
    cot=raw['self_cotangent'];r=reference['self_cotangent'];d=cot-r;atom_error=d.norm(dim=-1)
    indices=torch.argsort(atom_error,descending=True)[:10]
    residual=reference['loss_inputs']['forces']['pred']-reference['loss_inputs']['forces']['target']
    cot_info={'metric':metric(cot,r),'reference_norm':float(r.norm()),'max_atom_relative_l2':float((atom_error/r.norm(dim=-1)).max()),
        'top_atoms':[{'atom_index':int(i),'cotangent_error_norm':float(atom_error[i]),'reference_residual_norm':float(residual[i].norm())} for i in indices]}
    # All arithmetic below stays in FP32, including summaries over saved scalar rows.
    rows=[row for row in csv.DictReader((analysis/'ERROR-DECOMPOSITION.csv').open()) if row['case']=='tiled']
    fields={}
    for field in ('fixed_native_cotangent_projection_on_direct_error','cotangent_change_projection_on_direct_error',
                  'candidate_traversal_norm','native_traversal_norm','fixed_native_cotangent_relative_l2'):
        values=torch.tensor([float(row[field]) for row in rows],dtype=torch.float32)
        fields[field]={'min':float(values.min()),'median':float(values.median()),'max':float(values.max())}
    save(out/'TOTAL-LOSS-CONTROL.json',records);save(out/'COTANGENT-SENSITIVITY.json',cot_info)
    save(out/'DECOMPOSITION-SUMMARY.json',fields)
    first=read(analysis/'ABLATIONS.json')
    lines=['| 后端（前向/一阶/二阶） | 完整参数梯度FAIL | force-only二阶FAIL | 同原生力余切后FAIL | 力余切relative-L2 |',
        '|---|---:|---:|---:|---:|']
    labels={'native':'N/N/N，原模块','native-layout':'N/N/N，adapter布局','native-flat':'N/N/N，flatten matmul',
        'tiled':'T/T/T，当前候选','T-N':'T/N/N','N-T':'N/T/T','T-T-N':'T/T/N','N-N-T':'N/N/T'}
    for case,a in first.items():lines.append(f"| {labels[case]} | {a['gradients']['failed']} | {a['force_original']['failed']} | {a['force_shared_vjp']['failed']} | {a['force_cotangent']['relative_l2']:.6g} |")
    (out/'ABLATION-TABLE.md').write_text('\n'.join(lines)+'\n')
    save(out/'MANIFEST.json',{'total_source':str(total),'analysis_source':str(analysis),'synthesis_source_sha256':sha(Path(__file__)),
        'total_objects_sha256':{p.name:sha(p) for p in total.glob('*.pt')}})
    (ROOT/'SYNTHESIS-RESULT.txt').write_text(str(out)+'\n')
    print(json.dumps({'total_control':records,'cotangent':cot_info,'decomposition':fields}),flush=True)
if __name__=='__main__':main()
