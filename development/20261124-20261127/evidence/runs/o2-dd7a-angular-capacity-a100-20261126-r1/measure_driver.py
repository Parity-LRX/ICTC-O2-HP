"""Serial fresh-process triples; exact graph capacity and bounded row capacity."""
import json,os,signal,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ['O2_SWEEP_ROOT']);CONTROL=Path(__file__).parent
gate_path=Path((ROOT/'GATE-AUDIT.txt').read_text().strip());gates=json.loads((gate_path/'RESULT.json').read_text());assert gates['status']=='GATES_REVIEWED'
OUT=ROOT/'measurements'/os.environ['SLURM_JOB_ID'];OUT.mkdir(parents=True)
points={r['key']:r for r in gates['rows'] if 'key' in r};networks={r['order']:r for r in gates['rows'] if r.get('level')=='network'}
records=[];capacity=[];cache={}
def save(name,value):
    path=OUT/name;temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def triple(config,action='timing'):
    key='-'.join(str(config.get(k,'none')) for k in ('level','family','parity','order','mode','size'))
    ck=(key,action)
    if ck in cache:return cache[ck]
    if action=='capacity' and (key,'timing') in cache:
        r=cache[key,'timing']
        if r['status']=='PASS':return r
    target=OUT/(action+'-'+key);target.mkdir(parents=True);trials=[]
    for repeat in range(3):
        folder=target/str(repeat);folder.mkdir();command=[sys.executable,'-u',str(CONTROL/'worker.py')]
        for k,v in config.items():command+=['--'+k,str(v)]
        command+=['--action',action,'--output',str(folder)]
        start=time.time()
        with (folder/'process.log').open('w') as log:
            p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=p.wait(timeout=1800)
                r=json.loads((folder/'RESULT.json').read_text()) if (folder/'RESULT.json').exists() else {'status':'PROCESS_ERROR','exit_code':code}
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait();r={'status':'TIMEOUT','limit_s':1800}
        trials.append({'folder':str(folder),'status':r['status'],'elapsed_process_s':time.time()-start,'result':r})
        save('CURRENT.json',{'configuration':config,'action':action,'repeat':repeat,'status':r['status']})
        print(json.dumps({'configuration':config,'action':action,'repeat':repeat,'status':r['status'],'wall_s':r.get('elapsed_s'),'peak_bytes':r.get('peak_bytes')}),flush=True)
    statuses=[t['status'] for t in trials];status='PASS' if statuses==['PASS']*3 else 'OOM' if all(s in ('PASS','OOM') for s in statuses) and 'OOM' in statuses else 'ERROR'
    row={'configuration':config,'action':action,'status':status,'trials':trials,'repetitions':3}
    if status=='PASS':
        values=[v['result']['elapsed_s'] for v in trials];row.update(raw_times_s=values,median_s=statistics.median(values),
            median_throughput=statistics.median(v['result']['throughput'] for v in trials),
            median_allocated_bytes=statistics.median(v['result']['peak_bytes']['allocated'] for v in trials),
            median_reserved_bytes=statistics.median(v['result']['peak_bytes']['reserved'] for v in trials))
    records.append(row);cache[ck]=row;save('MEASUREMENTS.json',records);return row
def capacity_search(config):
    level=config['level'];start=256 if level=='operator' else 1;low=0;high=start;failure=None;result=None
    while True:
        r=triple({**config,'size':high},'capacity')
        if r['status']=='PASS':low=high;high*=2
        else:failure=r;break
    if failure['status']!='OOM':
        result={'configuration':config,'status':'UNBRACKETED_ERROR','largest_tested_success':low,'failed_size':high,'failed_status':failure['status']}
    else:
        step=1 if level=='network' else max(256,(high//32)//256*256)
        if not low and level=='operator':step=1
        while high-low>step:
            mid=((low+high)//2//step)*step
            if mid<=low:break
            r=triple({**config,'size':mid},'capacity')
            if r['status']=='PASS':low=mid
            elif r['status']=='OOM':high=mid
            else:
                result={'configuration':config,'status':'UNBRACKETED_ERROR','largest_tested_success':low,'failed_size':mid,'failed_status':r['status']};break
        if result is None:
            result={'configuration':config,'status':'STABLE_CAPACITY_3OF3','maximum_stable_size':low,'OOM_upper_size':high,'resolution':step,
                    'unit':'resident rows' if level=='operator' else '100-atom disconnected graphs','upper_is_actual_OOM':True}
            if low:
                boundary=triple({**config,'size':low},'timing')
                result['boundary_timing_status']=boundary['status']
                if boundary['status']=='PASS':result.update(boundary_median_s=boundary['median_s'],boundary_median_throughput=boundary['median_throughput'],boundary_allocated_bytes=boundary['median_allocated_bytes'],boundary_reserved_bytes=boundary['median_reserved_bytes'])
    capacity.append(result);save('CAPACITY.json',capacity);print('CAPACITY_RESULT '+json.dumps(result),flush=True)

families=[('uvw','canonical'),('uvw','full'),('bilinear','canonical')]
for family,parity in families:
    for order in range(1,9):
        for mode in ('infer','train'):
            base={'level':'operator','family':family,'parity':parity,'order':order,'mode':mode}
            small=points[f'op-{family}-{parity}-L{order}-N16-{mode}']['status']=='PASS'
            for size in (256,4096):
                key=f'op-{family}-{parity}-L{order}-N{size}-{mode}'
                if small and points[key]['status']=='PASS':
                    measured=triple({**base,'size':size})
                    if len(records)==1:assert measured['status']=='PASS','First timing configuration failed; inspect harness before continuing'
                else:records.append({'configuration':{**base,'size':size},'action':'timing','status':'GATE_HOLD','gate_status':points[key]['status'],'small_derivative_gate_passed':small})
save('MEASUREMENTS.json',records)
for order in [0,*range(1,9)]:
    for mode,key in (('infer','infer'),('train','train')):
        config={'level':'network','order':order,'mode':mode}
        if networks[order][key]=='PASS':
            measured=triple({**config,'size':1})
            if order==0:assert measured['status']=='PASS','Original snapshot anchor failed; inspect harness before continuing'
        else:records.append({'configuration':{**config,'size':1},'action':'timing','status':'GATE_HOLD','gate_status':networks[order][key]})
save('MEASUREMENTS.json',records);save('TIMING-COMPLETE.json',{'status':'TIMING_MATRIX_COMPLETE','records':len(records)})
print('TIMING_MATRIX_COMPLETE',flush=True)
for order in [0,*range(1,9)]:
    for mode in ('infer','train'):
        if networks[order][mode]=='PASS':capacity_search({'level':'network','order':order,'mode':mode})
        else:capacity.append({'configuration':{'level':'network','order':order,'mode':mode},'status':'GATE_HOLD','gate_status':networks[order][mode]})
for family,parity in families:
    for order in range(1,9):
        for mode in ('infer','train'):
            if points[f'op-{family}-{parity}-L{order}-N16-{mode}']['status']=='PASS':capacity_search({'level':'operator','family':family,'parity':parity,'order':order,'mode':mode})
            else:capacity.append({'configuration':{'level':'operator','family':family,'parity':parity,'order':order,'mode':mode},'status':'GATE_HOLD'})
save('CAPACITY.json',capacity);save('MANIFEST.json',{'status':'ALL_MEASUREMENT_ATTEMPTS_COMPLETE','job':os.environ['SLURM_JOB_ID'],'control':str(CONTROL),'gate_audit':str(gate_path),'records':len(records),'capacity_targets':len(capacity),'serial_single_GPU':True,'repetitions':3})
(ROOT/'MEASUREMENT-RESULT.txt').write_text(str(OUT)+'\n');print('ALL_MEASUREMENT_ATTEMPTS_COMPLETE',flush=True)
