import argparse,gc,json,os
from pathlib import Path
from common import *
def main(a):
    configure(True);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);model=network(a.order,'candidate',1)
    import ictc_o2_hp.uvw_triton as mm
    import ictc_o2_hp.bilinear_triton as bi
    oldmm,oldbi=mm._Matmul.forward,bi.Bilinear.forward;rows={};phase=''
    def observe(kind,values):
        record={'kind':kind,'shapes':[list(v.shape) for v in values],'strides':[list(v.stride()) for v in values]}
        key=json.dumps(record,sort_keys=True)
        if key not in rows:rows[key]={**record,'order':a.order,'first_phase':phase}
    def matmul(ctx,left,right):observe('mm',(left,right));return oldmm(ctx,left,right)
    def bilinear(ctx,k,left,right):observe('angular',(k,left,right));return oldbi(ctx,k,left,right)
    mm._Matmul.forward=staticmethod(matmul);bi.Bilinear.forward=staticmethod(bilinear)
    for phase in ('joint_vjp','actual_force_mixed','actual_full_training'):
        value=model.vjp() if phase=='joint_vjp' else model.force_mixed() if phase=='actual_force_mixed' else model.train_step()
        del value;gc.collect();torch.cuda.empty_cache();save(out/'RESULT.json',{'status':'SHAPES_CAPTURED','hardware':hardware(),'rows':list(rows.values())})
        print('PHASE',a.order,phase,len(rows),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--order',type=int,required=True);p.add_argument('--output',required=True);main(p.parse_args())
