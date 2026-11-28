"""Weight-GEMM-only cost; input flattening excluded and whole-network kept separate."""
import gc,json,os,statistics,time
from pathlib import Path
import torch
from context import ROOT,BASE,SOURCE,configure,cpu,save,sha,package_sha
from ictc_o2_hp.uvw_triton import channel_matmul
from candidate_sources import qualified_sources,load_module
def main():
    configure();gate=json.loads((Path((ROOT/'FULL-AUDIT.txt').read_text().strip())/'RESULT.json').read_text())
    assert gate['status']=='FULL_QUALIFICATION_PASS' and gate['operator_source_sha256']==sha(SOURCE/'ictc_o2_hp/uvw_triton.py')
    assert package_sha()==gate['package_sha256']
    sources=qualified_sources();c05=load_module('gemm_measure_c05',sources['c05']['path']);mask=load_module('gemm_measure_mask',sources['mask']['path'])
    old=BASE/'snapshots/qualification-119423/ictc-source/ictc_o2_hp/uvw_triton.py'
    assert sha(old)=='69bbd465591d77c7ce23697dad7fdecac5a4da24013168dc8fc224334ab5eba0'
    v2=load_module('gemm_measure_a80',old)
    capture=Path((BASE/'CAPTURE-RESULT.txt').read_text().strip());manifest=json.loads((capture/'CAPTURE.json').read_text())
    out=ROOT/'results'/('gemm-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir();rows=[]
    for row in manifest['backends']['native'][:4]:
        raw=torch.load(row['path'],map_location='cpu',weights_only=False);x=raw['x'].cuda().reshape(-1,raw['x'].shape[-1])
        for phase,v in raw['adjoints'].items():
            g=v['g'].cuda().reshape(-1,v['g'].shape[-1]);functions={'native':lambda:g.T@x,'a80':lambda:v2.channel_matmul(x.T,g).T,
                'c05':lambda:c05.channel_matmul(g.T,x),'mask':lambda:mask.channel_matmul(g.T,x),'candidate':lambda:channel_matmul(g.T,x)}
            arrays={};timings={};memory={}
            for backend,fn in functions.items():
                for _ in range(3):fn()
                values=[]
                for _ in range(30):
                    torch.cuda.synchronize();started=time.perf_counter();value=fn();torch.cuda.synchronize();values.append(time.perf_counter()-started);del value
                gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
                value=fn();torch.cuda.synchronize();memory[backend]={'resident_bytes':base,'peak_delta_allocated_bytes':torch.cuda.max_memory_allocated()-base}
                arrays[backend]=cpu(value);del value;timings[backend]={'median_s':statistics.median(values),'raw_times_s':values}
            path=out/f'{row["index"]:02d}-{phase}.pt';torch.save(arrays,path)
            rows.append({'shape':list(raw['K'].shape),'phase':phase,'path':str(path),'sha256':sha(path),'timings':timings,'memory':memory})
            del arrays,g;print(row['index'],phase,flush=True)
        del raw,x;gc.collect();torch.cuda.empty_cache()
    save(out/'RESULT.json',{'status':'GEMM_MEASURED_AFTER_FULL_AMD_PASS','operator_source_sha256':gate['operator_source_sha256'],'package_sha256':gate['package_sha256'],'rows':rows,
        'boundary':'synchronized wall time of dK GEMM on already flattened resident actual X/G; excludes flattening, complete loss/derivatives/optimizer and graph transfer',
        'v2':'actual frozen (X.T@G).T orientation; diagnostic original-trajectory FAIL retained','warmup':3,'calls_per_backend_case':30})
    (ROOT/'GEMM-RESULT.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
