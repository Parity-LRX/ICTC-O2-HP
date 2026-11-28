"""Infer the installed native FP32 angular BMM reduction using cancellation probes."""
import copy,gc,json,os
from pathlib import Path
import torch
from context import ROOT,configure,save,sha
def main():
    configure();out=ROOT/'results'/('bmm-tree-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir();records=[];seen=set()
    source=Path((ROOT/'BMM-LAYOUT.txt').read_text().strip());layouts=json.loads((source/'RESULT.json').read_text())['records']
    # The existing EF fixtures have 24-row tails for both 31512 edges and 600 nodes.
    full_layouts={}
    for row in layouts:
        if row['Nrows']==64:full_layouts[(row['L'],tuple(row['left_stride'][-2:]))]=row
    for row in full_layouts.values():
        v=copy.deepcopy(row);l=v['L'];h=v['H']
        v.update(Nrows=24,right_shape=[1,l*l,24*h],right_stride=[l*l*24*h,24*h,1],scope='EF 24-row tail derived from the original 64-row bounded lowering')
        layouts.append(v)
    for layout in layouts:
        key=tuple(layout['left_shape']+layout['left_stride']+layout['right_shape']+layout['right_stride'])
        if key in seen:continue
        seen.add(key);_,m,k=layout['left_shape'];n=layout['right_shape'][-1]
        aa=torch.empty_strided(layout['left_shape'],layout['left_stride'],device='cuda');bb=torch.empty_strided(layout['right_shape'],layout['right_stride'],device='cuda');a=aa[0];b=bb[0];calls=0;clades=[]
        def native():
            nonlocal calls
            calls+=1;return torch.bmm(aa,bb)[0]
        def clusters(indices):
            s=len(indices);anchor=indices[0];js=indices[1:];matrix=torch.empty((s-1,s),dtype=torch.bool);allk=torch.tensor(indices,device='cuda');H=float(2**40)
            for start in range(0,len(js),m):
                chunk=js[start:start+m];j=torch.tensor(chunk,device='cuda');size=len(chunk);r=torch.arange(size,device='cuda')
                a.zero_();a[:size,allk]=1;a[:size,anchor]=H;a[r,j]=-H;b.fill_(1)
                baseline=native();assert torch.equal(baseline,baseline[:,:1].expand_as(baseline))
                for zstart in range(0,s,n):
                    z=allk[zstart:zstart+n];b.fill_(1);b[z,torch.arange(len(z),device='cuda')]=0
                    value=native();delta=baseline[:size,:1]-value[:size,:len(z)]
                    invalid=(z[None,:]==anchor)|(z[None,:]==j[:,None]);delta=torch.where(invalid,0.,delta)
                    assert torch.all((delta==0)|(delta==1)),(m,n,k,start,zstart,'unexpected cancellation')
                    matrix[start:start+size,zstart:zstart+len(z)]=(delta==0).cpu()
            unique={v.numpy().tobytes():v for v in matrix};ordered=sorted(unique.values(),key=lambda v:int(v.sum()));current={anchor};siblings=[]
            for mask in ordered:
                group={indices[i] for i in mask.nonzero().flatten().tolist()};assert current<group,(m,n,k,anchor,'non-nested clade')
                siblings.append(sorted(group-current));current=group
            assert current==set(indices);clades.append({'anchor':anchor,'size':s,'sibling_sizes':[len(v) for v in siblings]})
            return siblings
        def first(node):return node if isinstance(node,int) else first(node['start'])
        def unrounded(leaf,other):
            eps=2.0**-23;a.zero_();b.fill_(1);a[:,leaf]=1+eps;a[:,other]=-(1+2*eps);b[leaf,:]=1+eps
            v=native();lo=float(v.min());hi=float(v.max());assert lo==hi and lo in (0.,2.0**-46)
            return lo!=0
        def build(indices):
            if len(indices)==1:return int(indices[0])
            anchor=indices[0];siblings=clusters(indices);node={'start':anchor,'steps':[]};prefix=True
            for sibling in siblings:
                child=build(sibling);kind='add'
                if isinstance(child,int) and unrounded(child,anchor):kind='fma_right'
                elif prefix and unrounded(anchor,first(child)):kind='fma_left'
                node['steps'].append([kind,child]);prefix=False
            return node
        print('START',m,n,k,layout['left_stride'],layout['right_stride'],flush=True);tree=build(list(range(k)))
        def leaves(t):return [t] if isinstance(t,int) else leaves(t['start'])+[v for _,c in t['steps'] for v in leaves(c)]
        assert sorted(leaves(tree))==list(range(k))
        path=out/f'{len(records):02d}-M{m}-N{n}-K{k}.json';save(path,{'status':'NATIVE_FP32_BMM_TREE_INFERRED','layout':layout,'tree':tree,'clades':clades,'calls':calls,'precision':'FP32 only, native direct BMM with the observed exact shapes/strides; synthetic coefficients are diagnostic only'})
        records.append({'path':str(path),'sha256':sha(path),'layout':layout,'calls':calls});print('DONE',m,n,k,calls,flush=True)
        del aa,bb,a,b;gc.collect();torch.cuda.empty_cache()
    save(out/'RESULT.json',{'status':'BMM_REDUCTION_TREES_INFERRED','records':records,'layout_receipt_sha256':sha(source/'RESULT.json')});(ROOT/'BMM-TREES.txt').write_text(str(out)+'\n')
if __name__=='__main__':main()
