import json,os
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
def read(p):return json.loads(p.read_text())
def simplify(node):
    if isinstance(node,int):return {'chain':[node]}
    value=simplify(node['start'])
    for kind,child in node['steps']:
        rhs=simplify(child)
        if kind=='fma_right':
            assert 'chain' in rhs and len(rhs['chain'])==1
            if 'chain' in value:value['chain']+=rhs['chain']
            else:value={'fma':[rhs['chain'][0],value]}
        elif kind=='fma_left':
            assert 'chain' in value and len(value['chain'])==1
            if 'chain' in rhs:value={'chain':rhs['chain']+value['chain']}
            else:value={'fma':[value['chain'][0],rhs]}
        else:value={'add':[value,rhs]}
    return value
def describe(node):
    if 'chain' in node:
        c=node['chain'];delta=[b-a for a,b in zip(c,c[1:])]
        return {'length':len(c),'first':c[:9],'last':c[-9:],'strides':sorted(set(delta))}
    if 'fma' in node:return {'fma':[node['fma'][0],describe(node['fma'][1])]}
    return {'add':[describe(n) for n in node['add']]}
source=Path((ROOT/'WEIGHT-TREE.txt').read_text().strip());plans={};summaries={}
for r in read(source/'RESULT.json')['records']:
    full=read(Path(r['path']));node=simplify(full['tree']);key=f'{r["m"]}x{r["n"]}'
    plans[key]=node;summaries[key]=describe(node)
(ROOT/'WEIGHT-PLANS.json').write_text(json.dumps(plans,indent=2)+'\n')
(ROOT/'WEIGHT-PLAN-SUMMARY.json').write_text(json.dumps(summaries,indent=2)+'\n')
print(json.dumps(summaries),flush=True)
