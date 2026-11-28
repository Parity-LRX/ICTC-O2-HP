"""Independent O(2) group-action check for unequal U/V/W (not GPU qualification)."""
import json,math,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'ICTC-O2-HP/src'))
from ictc_o2_hp import ReferenceUvwTensorProduct
from ictc_o2_hp.uvu_o2 import _basis
from ictc_o2_hp.o2_math import o2_representation

def transform(x,entries,channels,g):
    values=[]
    for entry in entries:
        basis=_basis(entry).float()
        action=basis.T@o2_representation(g,entry.key)@basis
        block=x[:,entry.offset:entry.offset+channels*entry.dim].reshape(len(x),channels,entry.dim)
        values.append(torch.einsum('ab,nub->nua',action,block).reshape(len(x),-1))
    return torch.cat(values,-1)
def main():
    assert os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(2);torch.manual_seed(20261125);rows=[]
    for parity in ('canonical','full'):
        model=ReferenceUvwTensorProduct(3,5,4,2,parity=parity)
        x=torch.randn(11,model.input1_dimension,requires_grad=True)
        y=torch.randn(11,model.input2_dimension,requires_grad=True)
        base=model(x,y)
        for angle,reflect in ((.713,False),(.417,True),(0.,True)):
            g=torch.tensor([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
            if reflect:g=g@torch.diag(torch.tensor([1.,-1.]))
            gx=transform(x,model.entries_in1,3,g);gy=transform(y,model.entries_in2,5,g)
            actual=model(gx,gy);reference=transform(base,model.entries_out,4,g)
            delta=(actual-reference).detach()
            rows.append({'parity':parity,'U':3,'V':5,'W':4,'paths':len(model.paths),
                'angle':angle,'reflection':reflect,'max_abs':float(delta.abs().max()),
                'relative_l2':float(delta.norm()/reference.detach().norm()),'dtype':'float32'})
    receipt={'scope':'CPU mathematical layout/group-action diagnostic; not Triton or formal representation qualification',
        'job':os.environ['SLURM_JOB_ID'],'rows':rows,'formal_FP32_representation_gate':'HOLD_UNAPPROVED'}
    (ROOT/'CPU-SEMANTICS.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
