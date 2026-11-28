"""Observe actual FP32 operands without replacing any true-loss cotangent."""
import gc,json,os
from pathlib import Path
from common import *
configure(True)
out=ROOT/'l7-local-captures'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True)
model=network(7,'candidate',1)
previous=WS/'runs/o2-angular-repair-a100-20261127-r1/gate-captures/120780/network-L7-candidate'
set_rng(torch.load(previous/'rng0.pt',map_location='cpu',weights_only=False)['before'])
import ictc_o2_hp.uvw_triton as mm
import ictc_o2_hp.bilinear_triton as bi
import ictc_o2_hp.fp32_plans as fp
oldmm,oldbi=mm._Matmul.forward,bi.Bilinear.forward
phase='';sequence=0;records=[];captured=set()
def observe(kind,values,value,reference):
    global sequence
    sequence+=1
    different=int((value!=reference).sum())
    if not different:return
    key=fp.mm_key(*values) if kind=='mm' else fp.angular_key(*values,64 if len(values[1])>=64 else len(values[1]))
    den=reference.norm();d=value-reference
    node=torch._C._current_autograd_node()
    row={'phase':phase,'sequence':sequence,'kind':kind,'key':key,'different_elements':different,
         'relative_l2':float(d.norm()/den) if den else None,'max_abs':float(d.abs().max()),
         'shapes':[list(v.shape) for v in values],'strides':[list(v.stride()) for v in values],
         'pointer_mod_256':[v.data_ptr()%256 for v in values],
         'node':node.name() if node else None,'plan':fp.plans().get(key)}
    group=(phase.split('-')[0],key)
    if group not in captured:
        captured.add(group);file=out/f'operands-{len(captured):04d}.pt'
        torch.save({'values':cpu(values),'reference':cpu(reference),'candidate':cpu(value),'metadata':row},file)
        row['arrays']=str(file);row['sha256']=sha(file)
        print('LOCAL_DIFFERENCE',phase,kind,key,different,row['relative_l2'],flush=True)
    records.append(row)
    save(out/'RESULT.json',{'status':'TRACING','rows':records,'calls':sequence,'saved_operand_sets':len(captured)})
def matmul(ctx,left,right):
    value=oldmm(ctx,left,right)
    with torch.no_grad():reference=left@right
    observe('mm',(left,right),value,reference)
    return value
def angular(ctx,k,a,b):
    value=oldbi(ctx,k,a,b)
    with torch.no_grad():reference=bounded.product(k,a,b)
    observe('angular',(k,a,b),value,reference)
    return value
mm._Matmul.forward=staticmethod(matmul);bi.Bilinear.forward=staticmethod(angular)
try:
    for step in range(3):
        model.optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        phase=f'mixed-{step}';mixed=model.force_mixed();del mixed
        gc.collect();torch.cuda.empty_cache()
        phase=f'train-{step}';training=model.train_step(capture=True)
        torch.save(training,out/f'training-{step}.pt');del training
        phase=f'after-{step}';after=cpu(model.forward(model.batch('train'),False));del after
        gc.collect();torch.cuda.empty_cache();print('TRACE_STEP_FINISHED',step,flush=True)
finally:
    mm._Matmul.forward=staticmethod(oldmm);bi.Bilinear.forward=staticmethod(oldbi)
save(out/'RESULT.json',{'status':'CAPTURED','rows':records,'calls':sequence,'saved_operand_sets':len(captured),'hardware':hardware()})
(ROOT/'L7-LOCAL-CAPTURE.txt').write_text(str(out)+'\n')
