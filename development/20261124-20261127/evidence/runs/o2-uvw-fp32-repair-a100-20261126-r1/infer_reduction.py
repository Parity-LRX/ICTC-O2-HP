"""Infer native FP32 addition/FMA trees independently of weights and training loss."""
import json,os,time
from pathlib import Path
import torch
from common import ROOT,configure,save,sha

def main():
    configure();out=ROOT/'results'/('reduction-inference-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    source=Path((ROOT/'CAPTURE-RESULT.txt').read_text().strip());rows=json.loads((source/'CAPTURE.json').read_text())['backends']['native'][2:4]
    results=[]
    for row in rows:
        v=torch.load(row['path'],map_location='cpu',weights_only=False)
        m=v['x'].shape[0]*v['x'].shape[1];k,n=v['K'].shape
        a=torch.zeros((m,k),device='cuda',dtype=torch.float32);b=torch.ones((k,n),device='cuda',dtype=torch.float32)
        arange=torch.arange(m,device='cuda');records=[];calls=[0];nodes=[]
        def clusters(indices):
            s=len(indices);anchor=indices[0]
            js=torch.tensor(indices[1:],device='cuda');ks=torch.tensor(indices,device='cuda')
            count=(s-1)*s;bits=torch.empty(count,dtype=torch.bool,device='cpu')
            for start in range(0,count,m):
                size=min(m,count-start);r=arange[:size];position=torch.arange(start,start+size,device='cuda')
                j=js[position//s];z=ks[position%s]
                a.zero_();a[r,anchor]=float(2**26);a[r,j]=-float(2**26);a[r,z]+=1
                y=a@b;calls[0]+=1
                assert torch.equal(y,y[:,:1].expand_as(y)),'Column-dependent reduction tree'
                assert torch.all((y[:size,0]==0)|(y[:size,0]==1)),'Unexpected exact-product signature'
                bits[start:start+size]=(y[:size,0]==0).cpu()
            matrix=bits.reshape(s-1,s)
            clades={frozenset(indices[z] for z in mask.nonzero().flatten().tolist()) for mask in matrix}
            ordered=sorted(clades,key=len);current=frozenset((anchor,));chain=[]
            for clade in ordered:
                assert current<clade and clade<=frozenset(indices),('Non-laminar clades',k,n,anchor,len(current),len(clade))
                chain.append(sorted(clade-current));current=clade
            assert current==frozenset(indices)
            records.append({'anchor':anchor,'indices':indices,'siblings':chain})
            return chain
        def build(indices):
            if len(indices)==1:return int(indices[0])
            node=int(indices[0])
            for sibling in clusters(indices):node={'add':[node,build(sibling)]}
            return node
        def leaves(node):
            if isinstance(node,int):return [node]
            left,right=next(iter(node.values()));return leaves(left)+leaves(right)
        epsilon=2.0**-23;residual=2.0**-46
        def unrounded(leaf,other):
            a.zero_();b.fill_(1);a[:,leaf]=1+epsilon;a[:,other]=-(1+2*epsilon);b[leaf,:]=1+epsilon
            y=a@b;calls[0]+=1;lo=float(y.min());hi=float(y.max())
            assert lo==hi and lo in (0.0,residual),('Variable FMA signature',leaf,other,lo,hi)
            return lo==residual
        def label(node):
            if isinstance(node,int):return node
            left,right=node['add'];left=label(left);right=label(right)
            if isinstance(left,int) and unrounded(left,leaves(right)[0]):return {'fma':[left,right]}
            if isinstance(right,int) and unrounded(right,leaves(left)[0]):return {'fma':[right,left]}
            return {'add':[left,right]}
        started=time.time()
        tree=build(list(range(k)));b.fill_(1);tree=label(tree)
        result={'status':'REDUCTION_TREE_INFERRED_NOT_QUALIFICATION','shape_mnk':[m,n,k],'tree':tree,
            'clade_records':records,'reference_calls':calls[0],'elapsed_s':time.time()-started,
            'input_layout':'row-major both matrices; actual native input-adjoint MM orientation',
            'precision':'FP32 only; exact-product clade signatures and FP32 nonexact-product FMA signatures',
            'capture_sha256':sha(Path(row['path']))}
        save(out/f'K{k}-N{n}.json',result);results.append({'k':k,'n':n,'path':str(out/f'K{k}-N{n}.json'),'reference_calls':calls[0]})
        print(json.dumps(results[-1]),flush=True)
    save(out/'RESULT.json',{'records':results,'script_sha256':sha(Path(__file__)),'gpu':torch.cuda.get_device_name(),
        'TF32':torch.backends.cuda.matmul.allow_tf32,'AMP':torch.is_autocast_enabled('cuda')})
    (ROOT/'REDUCTION-INFERENCE.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
