"""All four qualified backends, all samples and all actual GPU kernels."""
import csv,hashlib,importlib.util,json,os,statistics
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);WS=ROOT.parents[1]
def read(p):return json.loads(Path(p).read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
    source=Path((ROOT/'PERFORMANCE-RESULT.txt').read_text().strip());pre=read(ROOT/'PERFORMANCE-PREFLIGHT.json')
    out=ROOT/'results'/('performance-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir()
    parser_path=Path(pre['snapshot'])/'control/trace_parser.py';assert sha(parser_path)==pre['files']['control/trace_parser.py']
    spec=importlib.util.spec_from_file_location('target_frozen_parser',parser_path);parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    names=['_uvw_mm','_uvw_weight_partials','_uvw_weight_reduce','weight_fma8'];parser.ICTC_NAMES.update(names)
    original_family=parser.family
    parser.family=lambda name:'ICTC GPU' if 'weight_fma8' in name or original_family(name)=='ICTC Triton' else original_family(name)
    save(out/'CLASSIFICATION.json',{'parser_sha256':sha(parser_path),'added_ICTC_names':names,'denominator':'all actual cat=kernel durations; not wall clock'})
    rows=[];pairs={};all_conditions={};raw=[]
    for mode in ('ef','train'):
        conditions=[]
        for backend in ('native','c05','mask','candidate'):
            times=[];medians=[]
            for repeat in range(3):
                folder=source/f'{mode}-{backend}-time-{repeat}';r=read(folder/'RESULT.json');c=read(folder/'CONDITION.json')
                assert r['status']=='QUALIFIED_CANDIDATE_PAIRED_MEASUREMENT_COMPLETED' and len(r['times_s'])==5
                assert all(t>0 for t in r['times_s']);times+=r['times_s'];medians.append(r['median_s']);conditions.append(c);all_conditions[folder.name]=c
                raw.extend({'mode':mode,'backend':backend,'process':repeat,'call':i,'seconds':t} for i,t in enumerate(r['times_s']))
            folder=source/f'{mode}-{backend}-memory';m=read(folder/'RESULT.json');c=read(folder/'CONDITION.json');conditions.append(c);all_conditions[folder.name]=c
            assert m['status']=='QUALIFIED_CANDIDATE_PAIRED_MEASUREMENT_COMPLETED'
            q=statistics.quantiles(times,n=4,method='inclusive')
            row={'mode':mode,'backend':backend,'median_s':statistics.median(times),'process_medians_s':medians,'q25_s':q[0],'q75_s':q[2],
                'minimum_s':min(times),'maximum_s':max(times),'calls':len(times),'allocated_GiB':m['peak_bytes']['allocated']/2**30,
                'reserved_GiB':m['peak_bytes']['reserved']/2**30,'peak_bytes':m['peak_bytes'],'base_bytes':m['base_bytes']}
            folder=source/f'{mode}-{backend}-trace';c=read(folder/'CONDITION.json');conditions.append(c);all_conditions[folder.name]=c
            assert read(folder/'RESULT.json')['status']=='QUALIFIED_CANDIDATE_PAIRED_MEASUREMENT_COMPLETED'
            trace=parser.parse_trace(folder,out/'traces'/folder.name)
            obj=read(folder/'trace.json');assert not any(e.get('cat') in ('cuda_runtime','cuda_driver') and 'GraphLaunch' in e.get('name','') for e in obj['traceEvents']);del obj
            hp=sum(v['total_ms'] for v in trace['family'] if v['family']=='ICTC GPU')
            row.update(kernel_ms=trace['kernel_ms'],kernel_calls=trace['kernel_calls'],ICTC_ms=hp,ICTC_percent=100*hp/trace['kernel_ms']);rows.append(row)
        checks={}
        for k in ('state_identity','parameters','graph','common_state_sha256','qualified_candidate_sha256','qualified_package_sha256','timing_boundary','RNG_at_measurement_start'):
            checks[k]=len(conditions)==20 and len({json.dumps(c[k],sort_keys=True) for c in conditions})==1
        for k in ('gpu_uuid','node','torch','deterministic','TF32','AMP','CUDA_Graph','threads','non_fp32_buffers'):
            checks[k]=len({json.dumps(c['hardware'][k],sort_keys=True) for c in conditions})==1
        checks['finite_precheck']=all(c['finite_precheck'] for c in conditions);checks['parameter_identity']=all(c['original_parameter_ids_unchanged'] for c in conditions)
        checks['restore_excluded']=all(c['common_state_restore_outside_measurement'] for c in conditions)
        for c in conditions:
            if c['backend']!='native':assert c['executed_backend_source_sha256']==pre['qualified_backends'][c['backend']]['sha256']
        pairs[mode]={'passed':all(checks.values()),'conditions':len(conditions),'checks':checks};assert pairs[mode]['passed'],pairs[mode]
    comparisons=[]
    for mode in ('ef','train'):
        by={r['backend']:r for r in rows if r['mode']==mode}
        for backend in ('c05','mask','candidate'):
            r=by[backend]
            comparisons.append({'mode':mode,'backend':backend,'median_s':r['median_s'],
                'time_change_vs_native_percent':100*(r['median_s']/by['native']['median_s']-1),
                'time_change_vs_c05_percent':100*(r['median_s']/by['c05']['median_s']-1),
                'time_change_vs_mask_percent':100*(r['median_s']/by['mask']['median_s']-1),
                'allocated_change_vs_c05_bytes':r['peak_bytes']['allocated']-by['c05']['peak_bytes']['allocated'],
                'reserved_change_vs_c05_bytes':r['peak_bytes']['reserved']-by['c05']['peak_bytes']['reserved']})
    save(out/'WORKLOADS.json',rows);save(out/'PAIRED-AUDIT.json',pairs);save(out/'ALL-CONDITIONS.json',all_conditions);save(out/'COMPARISON.json',comparisons)
    with (out/'RAW-TIMINGS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
    result={'status':'PERFORMANCE_AUDIT_PASS','source':str(source),'paired':pairs,'comparison':comparisons,'operator_source_sha256':pre['operator_source_sha256'],'package_sha256':pre['qualified_backends']['candidate']['package_sha256'],'workloads':rows}
    save(out/'RESULT.json',result);(ROOT/'PERFORMANCE-AUDIT.txt').write_text(str(out)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
