"""FP32 cancellation tomography of the original F.linear 10504-row weight VJP."""
import gc,json,os,time
from pathlib import Path
import torch
from context import ROOT,BASE,configure,save,sha
def main():
    configure();out=ROOT/'results'/('weight-tree-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir()
    capture=Path((BASE/'CAPTURE-RESULT.txt').read_text().strip());rows=json.loads((capture/'CAPTURE.json').read_text())['backends']['native'][:4]
    records=[]
    for row in rows:
        raw=torch.load(row['path'],map_location='cpu',weights_only=False);x=torch.zeros_like(raw['x'],device='cuda',requires_grad=True);w=torch.zeros_like(raw['K'],device='cuda',requires_grad=True)
        sample=raw['adjoints']['energy_forward_and_force']['g'];g=torch.zeros_like(sample,device='cuda',requires_grad=True)
        m,n=w.shape;k=x.shape[0]*x.shape[1];assert k==10504
        a=torch.zeros((m,k),device='cuda');b=torch.ones((k,n),device='cuda');calls=0;clades=[];fma_tests=[]
        def native():
            nonlocal calls
            with torch.no_grad():x.copy_(b.reshape_as(x));g.copy_(a.T.reshape_as(g))
            y=torch.nn.functional.linear(x,w);dx,dw=torch.autograd.grad(y,(x,w),g,create_graph=True);calls+=1
            return dw.detach()
        def clusters(indices):
            s=len(indices);anchor=indices[0];js=indices[1:];matrix=torch.empty((s-1,s),dtype=torch.bool)
            allk=torch.tensor(indices,device='cuda');H=float(2**40)
            for start in range(0,len(js),m):
                chunk=js[start:start+m];j=torch.tensor(chunk,device='cuda');size=len(chunk);r=torch.arange(size,device='cuda')
                a.zero_();a[:size,allk]=1;a[:size,anchor]=H;a[r,j]=-H;b.fill_(1)
                baseline=native();assert torch.equal(baseline,baseline[:,:1].expand_as(baseline)), 'Column dependent native tree'
                for zstart in range(0,s,n):
                    z=allk[zstart:zstart+n];b.fill_(1);b[z,torch.arange(len(z),device='cuda')]=0
                    value=native();delta=baseline[:size,:1]-value[:size,:len(z)]
                    invalid=(z[None,:]==anchor)|(z[None,:]==j[:,None]);delta=torch.where(invalid,0.,delta)
                    assert torch.all((delta==0)|(delta==1)),('Unexpected cancellation signature',m,n,start,zstart)
                    matrix[start:start+size,zstart:zstart+len(z)]=(delta==0).cpu()
                if start%(m*8)==0:print('clade',m,n,s,start,calls,flush=True)
            unique={v.numpy().tobytes():v for v in matrix};ordered=sorted(unique.values(),key=lambda v:int(v.sum()))
            current={anchor};siblings=[]
            for mask in ordered:
                group={indices[i] for i in mask.nonzero().flatten().tolist()};assert current<group
                siblings.append(sorted(group-current));current=group
            assert current==set(indices);clades.append({'anchor':anchor,'size':s,'sibling_sizes':[len(v) for v in siblings]})
            return siblings
        def first(node):return node if isinstance(node,int) else first(node['start'])
        def unrounded(leaf,other):
            eps=2.0**-23;a.zero_();b.fill_(1);a[:,leaf]=1+eps;a[:,other]=-(1+2*eps);b[leaf,:]=1+eps
            value=native();lo=float(value.min());hi=float(value.max());assert lo==hi and lo in (0.,2.0**-46)
            return lo!=0
        def build(indices):
            if len(indices)==1:return int(indices[0])
            anchor=indices[0];siblings=clusters(indices);node={'start':anchor,'steps':[]};prefix_leaf=True
            for index,sibling in enumerate(siblings):
                child=build(sibling);kind='add'
                if isinstance(child,int) and unrounded(child,anchor):kind='fma_right'
                elif prefix_leaf and unrounded(anchor,first(child)):kind='fma_left'
                node['steps'].append([kind,child]);prefix_leaf=False
                if index%1000==0:print('fma',m,n,len(indices),index,calls,flush=True)
            return node
        started=time.time();tree=build(list(range(k)))
        def leaves(node):
            if isinstance(node,int):return [node]
            return leaves(node['start'])+[v for _,child in node['steps'] for v in leaves(child)]
        values=leaves(tree);assert len(values)==k and sorted(values)==list(range(k))
        result={'status':'WEIGHT_FP32_REDUCTION_TREE_INFERRED','shape_mnk':[m,n,k],'tree':tree,'clade_summary':clades,
            'calls':calls,'elapsed_s':time.time()-started,'reference':'F.linear native full VJP, create_graph=True, original x/g physical strides',
            'x_stride':list(x.stride()),'g_stride':list(g.stride()),'synthetic_data_independent_of_weights_and_loss':True,
            'large_cancellation_value':2**40,'precision':'FP32 only, TF32/AMP/half/CUDA Graph disabled','capture_sha256':sha(Path(row['path']))}
        path=out/f'W{m}-V{n}.json';save(path,result);records.append({'m':m,'n':n,'k':k,'path':str(path),'calls':calls})
        del x,w,g,a,b,raw;gc.collect();torch.cuda.empty_cache();print('DONE',m,n,calls,flush=True)
    save(out/'RESULT.json',{'records':records,'gpu':torch.cuda.get_device_name(),'script_sha256':sha(Path(__file__))});(ROOT/'WEIGHT-TREE.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
