import ast,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[1];assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
p=WS/'ICTC-O2-HP/src/ictc_o2_hp/uvw_triton.py';ast.parse(p.read_text())
record={'status':'INTEGRATED_CANDIDATE_PENDING_FULL_GATES','commit':(ROOT/'CANDIDATE-HEAD.txt').read_text().strip(),
    'operator_source_sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),
    'forward_segments':{'K1024_N256':[512,512],'K768_N192':[192,192,192,192]},
    'input_adjoint_segments':{'K512_N128':[86,86,86,86,86,82],'K384_N96':[128,128,128]},
    'reduction':'FP32 IEEE dot chains; explicit left-to-right add.rn of completed segments',
    'reduction_inference':(ROOT/'REDUCTION-INFERENCE.txt').read_text().strip(),
    'native_cotangents_fixed_in_training':False,'TF32':False,'AMP':False,'CUDA_Graph':False}
(ROOT/'SELECTED.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
