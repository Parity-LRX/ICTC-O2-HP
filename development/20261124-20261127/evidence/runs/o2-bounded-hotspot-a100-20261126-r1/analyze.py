"""CPU-only analysis of immutable, already qualified A100 receipts."""
import ast,bisect,collections,csv,gc,hashlib,json,math,os,shutil,statistics
from pathlib import Path

ROOT=Path(os.environ['O2_HOT_ROOT']);WS=ROOT.parents[1]
OLD=WS/'runs/o2-bounded-bilinear-a100-20261126-r1'
AUDIT=Path((OLD/'PERFORMANCE-AUDIT.txt').read_text().strip())
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
OUT=ROOT/'results'/os.environ['SLURM_JOB_ID'];OUT.mkdir(parents=True)
inputs={}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):
    p=Path(p);inputs[str(p)]=sha(p);return json.loads(p.read_text())
def save(name,v):(OUT/name).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['status']);w.writeheader();w.writerows(rows)
def external(e):return e.get('args',{}).get('External id',e.get('args',{}).get('external id'))
def union_ms(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
        else:merged.append([start,end])
    return sum(b-a for a,b in merged)/1000

seal=read(OLD/'FINAL-SNAPSHOT.json');audit=read(AUDIT/'RESULT.json')
assert seal['status']=='SEALED_WITH_INDEPENDENT_UNPACK_QA' and audit['status']=='PERFORMANCE_AUDIT_PASS'
assert seal['candidate_commit']=='dd7a3db136c1d2946aa0e41b87a167d8f940b122'
assert seal['operator_source_sha256']==audit['operator_source_sha256']
assert inputs[str(AUDIT/'RESULT.json')]==seal['gates']['PERFORMANCE-AUDIT']['sha256']
assert all(audit['paired'][m]['passed'] for m in ('ef','train'))
source=Path(audit['source']);work={(r['mode'],r['backend']):r for r in audit['workloads']}
comparison=[];families=[];phases=[];custom=[];operators=[];interval_rows=[];matrix=[];memory=[];metadata=[]
for mode in ('ef','train'):
    native=work[mode,'native'];candidate=work[mode,'candidate']
    row={'mode':mode,'native_s':native['median_s'],'candidate_s':candidate['median_s'],
         'speedup':native['median_s']/candidate['median_s'],
         'latency_change_percent':100*(candidate['median_s']/native['median_s']-1)}
    for kind in ('allocated','reserved'):
        a=native['peak_bytes'][kind];b=candidate['peak_bytes'][kind]
        row.update({f'native_{kind}_bytes':a,f'candidate_{kind}_bytes':b,f'{kind}_delta_bytes':b-a,
                    f'{kind}_delta_MiB':(b-a)/2**20,f'{kind}_delta_percent':100*(b/a-1)})
    comparison.append(row)
    for backend in ('native','candidate'):
        prefix={'mode':mode,'backend':backend};folder=AUDIT/f'traces/{mode}-{backend}-trace'
        s=read(folder/'SUMMARY.json');k_ms=s['kernel_ms'];n=work[mode,backend]
        assert math.isclose(k_ms,n['kernel_ms'],rel_tol=1e-12)
        for group,name in (('family','families'),('phase','phases')):
            assert sum(r['calls'] for r in s[group])==s['kernel_calls']
            assert math.isclose(sum(r['total_ms'] for r in s[group]),k_ms,rel_tol=1e-10)
            destination=families if group=='family' else phases
            destination.extend({**prefix,**r} for r in s[group])
        for filename,destination in (('KERNELS.csv',custom),('OPERATOR.csv',operators)):
            path=folder/filename;inputs[str(path)]=sha(path)
            with path.open() as f:
                for r in csv.DictReader(f):
                    if filename=='KERNELS.csv' and r['family']!='ICTC GPU':continue
                    destination.append({**prefix,**r})
        trace_path=source/f'{mode}-{backend}-trace/trace.json';data=read(trace_path)
        assert inputs[str(trace_path)]==s['trace_sha256']
        events=data['traceEvents'];cpu={str(external(e)):e for e in events if e.get('cat')=='cpu_op' and external(e) is not None}
        kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
        assert len(kernels)==s['kernel_calls']
        kernel_intervals=[(e['ts'],e['ts']+e['dur']) for e in kernels]
        device_intervals=[(e['ts'],e['ts']+e['dur']) for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and e.get('ph')=='X']
        annotations=[e for e in events if e.get('name')=='qualified_'+mode and e.get('cat')=='user_annotation' and 'dur' in e]
        interval_rows.append({**prefix,'kernel_sum_ms':k_ms,'kernel_union_ms':union_ms(kernel_intervals),
            'device_activity_union_ms':union_ms(device_intervals),
            'first_to_last_kernel_ms':(max(b for a,b in kernel_intervals)-min(a for a,b in kernel_intervals))/1000,
            'profile_CPU_annotation_ms':annotations[0]['dur']/1000 if len(annotations)==1 else None,
            'independent_steady_median_ms':1000*n['median_s']})
        grouped=collections.defaultdict(lambda:[0,0.0]);path=folder/'KERNEL-EVENTS.csv';inputs[str(path)]=sha(path)
        with path.open() as f:
            for r in csv.DictReader(f):
                op=cpu.get(r['external_id']);dims=op.get('args',{}).get('Input Dims') if op else None
                if r['operator'] not in ('aten::bmm','aten::mm','aten::addmm','_Matmul','Bilinear'):continue
                key=(r['operator'],json.dumps(dims,separators=(',',':')),r['phase'],r['stage'])
                grouped[key][0]+=1;grouped[key][1]+=float(r['duration_ms'])
        matrix.extend({**prefix,'operator':key[0],'input_dims':key[1],'phase':key[2],'stage':key[3],
                       'kernel_calls':v[0],'total_ms':v[1],'percent':100*v[1]/k_ms}
                      for key,v in sorted(grouped.items(),key=lambda item:-item[1][1]))
        gpu_memory=[e for e in events if e.get('name')=='[memory]' and e.get('args',{}).get('Device Type')==1 and 'Total Allocated' in e.get('args',{})]
        if gpu_memory:
            peak=max(gpu_memory,key=lambda e:e['args']['Total Allocated']);t=peak['ts'];tid=peak.get('tid')
            owners=sorted((e for e in events if e.get('cat')=='cpu_op' and e.get('tid')==tid and 'dur' in e and e['ts']<=t<=e['ts']+e['dur']),key=lambda e:e['dur'])
            phase=[e['name'][6:] for e in events if e.get('name','').startswith('phase:') and e.get('cat')=='user_annotation' and 'dur' in e and e['ts']<=t<=e['ts']+e['dur']]
            memory.append({**prefix,'profile_peak_allocated_bytes':peak['args']['Total Allocated'],
                'profile_peak_reserved_bytes':peak['args'].get('Total Reserved'),'phase':phase,
                'trigger_allocation_bytes':peak['args'].get('Bytes'),'trigger_args':peak['args'],
                'CPU_owners':[{'name':e['name'],'input_dims':e.get('args',{}).get('Input Dims')} for e in owners[:8]],
                'scope':'profile-memory event ownership only; not independent memory-process peak'})
        samples=[]
        for process in range(3):
            r=read(source/f'{mode}-{backend}-time-{process}/RESULT.json');assert len(r['times_s'])==5
            samples+=r['times_s']
        assert statistics.median(samples)==n['median_s']
        metadata.append({**prefix,'stage_coverage_percent':s['stage_coverage_percent'],
            'phase_coverage_percent':s['phase_coverage_percent'],'kernel_calls':s['kernel_calls'],
            'kernel_ms':k_ms,'all_15_times_s':samples})
        print(json.dumps({**prefix,'kernel_ms':k_ms,'top_matrix':matrix[-len(grouped):][:5],'memory':memory[-1:]},ensure_ascii=False),flush=True)
        del data,events,cpu,kernels,gpu_memory;gc.collect()

for name,rows in (('COMPARISON.csv',comparison),('FAMILY.csv',families),('PHASE.csv',phases),('ICTC-KERNELS.csv',custom),
                  ('OPERATORS.csv',operators),('MATRIX-SHAPES.csv',matrix),('TIMING-BOUNDARIES.csv',interval_rows)):
    table(name,rows)
save('PROFILE-MEMORY-PEAKS.json',memory);save('TRACE-METADATA.json',metadata)
save('RESULT.json',{'status':'REUSED_TRACE_HOTSPOT_ANALYSIS_PASS','candidate_commit':seal['candidate_commit'],
    'operator_source_sha256':seal['operator_source_sha256'],'source_audit':str(AUDIT),'source_GPU_job':'120352',
    'source_AMD_job':'120353','AMD_job':os.environ['SLURM_JOB_ID'],'new_GPUh':0,
    'comparison':comparison,'families':families,'phases':phases,'ICTC_kernels':custom,'timing_boundaries':interval_rows,
    'profile_memory':memory,'coverage':metadata,'inputs':inputs})
save('INPUT-MANIFEST.json',inputs)
control=OUT/'control';control.mkdir();shutil.copy2(__file__,control/'analyze.py');shutil.copy2(ROOT/'CONTRACT.zh-CN.md',control/'CONTRACT.zh-CN.md')
(ROOT/'ANALYSIS-RESULT.txt').write_text(str(OUT)+'\n');print('REUSED_TRACE_HOTSPOT_ANALYSIS_PASS',flush=True)
