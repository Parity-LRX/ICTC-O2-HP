import json,os
from pathlib import Path
from context import ROOT,save,sha
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
source=Path((ROOT/'BMM-TREES.txt').read_text().strip());records=[]
for row in json.loads((source/'RESULT.json').read_text())['records']:
    assert sha(row['path'])==row['sha256'];r=json.loads(Path(row['path']).read_text());chains=[]
    def visit(node):
        if isinstance(node,int):
            chains.append([node]);return len(chains)-1
        leaves=[node['start']];i=0
        for kind,child in node['steps']:
            if not isinstance(child,int) or kind=='add':break
            if kind=='fma_right':leaves.append(child)
            elif kind=='fma_left':assert len(leaves)==1;leaves.insert(0,child)
            else:raise ValueError(kind)
            i+=1
        assert all(isinstance(x,int) for x in leaves)
        chains.append(leaves);value=len(chains)-1
        for kind,child in node['steps'][i:]:
            assert kind=='add',(kind,child)
            value=[value,visit(child)]
        return value
    tree=visit(r['tree']);assert sorted(x for c in chains for x in c)==list(range(row['layout']['L']**2))
    record={**row,'chains':chains,'add_tree':tree};records.append(record)
    print({'L':row['layout']['L'],'Q':row['layout']['right_shape'][-1],'left_stride':row['layout']['left_stride'],'lengths':[len(c) for c in chains],'first':[c[:8] for c in chains],'last':[c[-4:] for c in chains],'add_tree':tree},flush=True)
save(ROOT/'BMM-PLANS.json',{'status':'EXTRACTED_NATIVE_FP32_PLANS','records':records,'source':str(source)})
