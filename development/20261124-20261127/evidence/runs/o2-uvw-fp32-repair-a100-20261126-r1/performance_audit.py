"""AMD pairing audit, all raw timing samples and complete kernel inventories."""
import csv,hashlib,importlib.util,json,os,statistics
from pathlib import Path
ROOT=Path(os.environ['O2_REPAIR_ROOT']);WS=ROOT.parents[1]
def read(path):return json.loads(path.read_text())
def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
    source=Path((ROOT/'PERFORMANCE-RESULT.txt').read_text().strip())
    out=ROOT/'results'/('performance-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir()
    preflight=read(ROOT/'PERFORMANCE-PREFLIGHT.json')
    parser_path=Path(preflight['snapshot'])/'trace_parser.py'
    assert sha(parser_path)==preflight['files']['trace_parser.py']
    spec=importlib.util.spec_from_file_location('frozen_trace_parser',parser_path)
    parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser);parser.ICTC_NAMES.add('_uvw_mm')
    save(out/'CLASSIFICATION.json',{'parser':str(parser_path),'sha256':sha(parser_path),'added_ICTC_names':['_uvw_mm'],
        'denominator':'all actual cat=kernel X event durations; separate from wall clock'})
    rows=[];all_conditions={};pairs={};trace_rows=[];raw_samples=[]
    for mode in ('ef','train'):
        conditions=[]
        for backend in ('native','v1','candidate'):
            samples=[];medians=[]
            for repeat in range(3):
                path=source/f'{mode}-{backend}-time-{repeat}';r=read(path/'RESULT.json');c=read(path/'CONDITION.json')
                assert r['status']=='QUALIFIED_CANDIDATE_PAIRED_MEASUREMENT_COMPLETED' and len(r['times_s'])==10
                assert all(v>0 for v in r['times_s']);samples.extend(r['times_s']);medians.append(r['median_s']);conditions.append(c)
                all_conditions[path.name]=c
                raw_samples.extend({'mode':mode,'backend':backend,'process_repeat':repeat,'sample':i,'seconds':v} for i,v in enumerate(r['times_s']))
            path=source/f'{mode}-{backend}-memory';r=read(path/'RESULT.json');c=read(path/'CONDITION.json')
            assert r['status']=='QUALIFIED_CANDIDATE_PAIRED_MEASUREMENT_COMPLETED';conditions.append(c);all_conditions[path.name]=c
            q=statistics.quantiles(samples,n=4,method='inclusive')
            row={'mode':mode,'backend':backend,'median_s':statistics.median(samples),'process_medians_s':medians,
                'minimum_s':min(samples),'maximum_s':max(samples),'q25_s':q[0],'q75_s':q[2],
                'samples':len(samples),'allocated_GiB':r['peak_bytes']['allocated']/2**30,'reserved_GiB':r['peak_bytes']['reserved']/2**30,
                'memory_peak_bytes':r['peak_bytes'],'base_bytes':r['base_bytes'],
                'numerical_status':c['numerical_status']}
            if backend!='v1':
                path=source/f'{mode}-{backend}-trace';c=read(path/'CONDITION.json');conditions.append(c);all_conditions[path.name]=c
                assert read(path/'RESULT.json')['status']=='QUALIFIED_CANDIDATE_PAIRED_MEASUREMENT_COMPLETED'
                trace=parser.parse_trace(path,out/'traces'/path.name)
                # This experiment forbids CUDA Graph; inspect actual runtime names as well as flags.
                obj=read(path/'trace.json')
                assert not any(e.get('cat') in ('cuda_runtime','cuda_driver') and 'GraphLaunch' in e.get('name','') for e in obj['traceEvents'])
                del obj
                hp=sum(v['total_ms'] for v in trace['family'] if v['family']=='ICTC Triton')
                row.update(kernel_ms=trace['kernel_ms'],kernel_calls=trace['kernel_calls'],ICTC_ms=hp,ICTC_percent=100*hp/trace['kernel_ms'])
                trace_rows.append({'mode':mode,'backend':backend,**row})
            rows.append(row)
        checks={}
        for key in ('state_identity','parameters','graph','common_state_sha256','qualified_candidate_sha256','timing_boundary'):
            values=[json.dumps(c[key],sort_keys=True) for c in conditions];checks[key]=len(conditions)==14 and len(set(values))==1
        for key in ('gpu_uuid','node','torch','deterministic','TF32','AMP','CUDA_Graph','threads','non_fp32_buffers'):
            values=[json.dumps(c['hardware'][key],sort_keys=True) for c in conditions];checks[key]=len(set(values))==1
        checks['finite_prechecks']=all(c['finite_precheck'] for c in conditions)
        checks['parameter_identity']=all(c['original_parameter_ids_unchanged'] for c in conditions)
        checks['restore_excluded']=all(c['common_state_restore_outside_measurement'] for c in conditions)
        pairs[mode]={'passed':all(checks.values()),'conditions':len(conditions),'checks':checks}
        assert pairs[mode]['passed'],pairs[mode]
    comparisons=[]
    for mode in ('ef','train'):
        by={r['backend']:r for r in rows if r['mode']==mode};n,v,c=(by[b] for b in ('native','v1','candidate'))
        comparisons.append({'mode':mode,'native_s':n['median_s'],'v1_s':v['median_s'],'candidate_s':c['median_s'],
            'native_over_candidate':n['median_s']/c['median_s'],'v1_over_candidate_diagnostic':v['median_s']/c['median_s'],
            'candidate_time_change_vs_native_percent':100*(c['median_s']/n['median_s']-1),
            'candidate_time_change_vs_v1_percent':100*(c['median_s']/v['median_s']-1),
            'native_allocated_GiB':n['allocated_GiB'],'v1_allocated_GiB':v['allocated_GiB'],'candidate_allocated_GiB':c['allocated_GiB'],
            'candidate_reserved_GiB':c['reserved_GiB'],'v1_train_status':'FAIL retained; timing comparison diagnostic only'})
    save(out/'WORKLOADS.json',rows);save(out/'PAIRED-AUDIT.json',pairs);save(out/'ALL-CONDITIONS.json',all_conditions)
    save(out/'COMPARISON.json',comparisons)
    with (out/'RAW-TIMINGS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(raw_samples[0]));w.writeheader();w.writerows(raw_samples)
    result={'status':'PERFORMANCE_AUDIT_PASS','source':str(source),'paired':pairs,'comparison':comparisons,
        'operator_source_sha256':read(ROOT/'SELECTED.json')['operator_source_sha256'],'traces':trace_rows}
    save(out/'RESULT.json',result);(ROOT/'PERFORMANCE-AUDIT.txt').write_text(str(out)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
