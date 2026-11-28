import json,os
from pathlib import Path
from context import ROOT,save
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';r=json.loads((ROOT/'BMM-PLANS.json').read_text());rows=[]
for row in r['records']:
    l=row['layout']['L'];q=row['layout']['right_shape'][-1]
    column=row['layout']['left_stride'][-2]==1
    if l==19:p=({2048:4,128:1,768:5} if column else {2048:1,128:3,768:4})[q];four=True
    elif column:p=1;four=(q==32768)
    else:p={32768:1,18432:3,12288:2}[q];four=False
    span=(l*l+p-1)//p;chains=[];total=None
    for part in range(p):
        if four:
            ids=[]
            for lane in range(4):
                ids.append(len(chains));chains.append([part*span+i+j for i in range(lane*8,span,32) for j in range(8) if i+j<span and part*span+i+j<l*l])
            value=[[ids[0],ids[1]],[ids[2],ids[3]]]
        else:value=len(chains);chains.append(list(range(part*span,min((part+1)*span,l*l))))
        total=value if total is None else [total,value]
    assert chains==row['chains'] and total==row['add_tree'],(l,q,'plan mismatch')
    rows.append({'L':l,'Q':q,'coefficient_column_major':column,'parts':p,'four_chains_of_eight':four,'span':span,'probe_receipt':row['path'],'probe_sha256':row['sha256'],'complete_leaf_order_and_add_tree_match':True})
out={'status':'ALL_NATIVE_PLAN_LEAVES_AND_MERGES_MATCH','rows':rows,'precision':'FP32 outer mul.rn, FP32 FMA and explicit add.rn; no reassociation'}
save(ROOT/'PLAN-VALIDATION.json',out);save(ROOT.parents[1]/'ICTC-O2-HP/src/ictc_o2_hp/bilinear_plans.json',out);print(json.dumps(out),flush=True)
