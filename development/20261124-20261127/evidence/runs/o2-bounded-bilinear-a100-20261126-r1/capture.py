"""Capture real bounded-bilinear operands/cotangents without changing the loss."""
import gc,json,os,types
import torch
from context import ROOT,configure,factory,common_state,restore,switches,cpu,save,sha,package_sha
def main():
    configure();out=ROOT/'results'/('capture-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir()
    model=factory('v3','candidate','checkpoint_ema','cuda');state,receipt=common_state('train');select=switches(model);select('baseline')
    import bounded
    original=bounded.Bilinear.backward;phase='';records=[];seen=set()
    names={m.canonical_kernel.data_ptr():name for name,m in model.model.named_modules() if hasattr(m,'canonical_kernel')}
    def observed(ctx,g):
        k,a,b=ctx.saved_tensors;site=names[k.data_ptr()];key=(phase,site,k.stride())
        if key not in seen:
            seen.add(key);path=out/f'{len(records):03d}.pt'
            torch.save(cpu({'k':k,'a':a,'b':b,'g':g}),path)
            row={'path':str(path),'sha256':sha(path),'site':site,'phase':phase,'shapes':{n:list(v.shape) for n,v in zip(('k','a','b','g'),(k,a,b,g))},'strides':{n:list(v.stride()) for n,v in zip(('k','a','b','g'),(k,a,b,g))}}
            records.append(row);print(phase,site,list(k.stride()),flush=True)
        # Nonreentrant checkpoint permits one unpack per backward invocation.
        # Give the unchanged original formula the values already unpacked above.
        return original(types.SimpleNamespace(saved_tensors=(k,a,b),needs_input_grad=ctx.needs_input_grad),g)
    bounded.Bilinear.backward=staticmethod(observed)
    for phase in ('actual_force_only_mixed','actual_full_training'):
        restore(model,state)
        value=model.force_mixed() if phase=='actual_force_only_mixed' else model.train_step()
        del value;gc.collect();torch.cuda.empty_cache()
    assert len({r['site'] for r in records})==14
    save(out/'CAPTURE.json',{'records':records,'reference':'8324b33 channel implementation with original bounded bilinear; actual own native force-only/full loss','package_sha256':package_sha(),'all_14_sites':True,'shared_loss_cotangents':False})
    (ROOT/'CAPTURE-RESULT.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
