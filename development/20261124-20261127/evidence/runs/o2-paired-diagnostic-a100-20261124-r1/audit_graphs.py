import ast,csv,json,os,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert os.environ.get('SLURM_JOB_ID')
for p in ROOT.glob('*.py'):ast.parse(p.read_text())
rows=[]
for p in (ROOT/'results').glob('a100-*/*-trace/trace.json'):
    if not (p.parent/'RESULT.json').exists():continue
    with p.open('rb') as f:payload=f.read()
    matches=re.findall(rb'"name"\s*:\s*"((?:cuda|cu)(?:GraphLaunch|GraphInstantiate|StreamBeginCapture|StreamEndCapture)[^"]*)"',payload)
    rows.append({'trace':str(p),'capture_or_graph_launch_calls':len(matches)})
    del payload
path=ROOT/'prepared'/('GRAPH-AUDIT-'+os.environ['SLURM_JOB_ID']+'.csv')
with path.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['trace','capture_or_graph_launch_calls']);writer.writeheader();writer.writerows(rows)
assert rows and all(r['capture_or_graph_launch_calls']==0 for r in rows)
print(json.dumps({'status':'ACTUAL_TRACE_GRAPH_AUDIT_PASS','job':os.environ['SLURM_JOB_ID'],'traces':len(rows),'graph_calls':0,'syntax':'PASS'}),flush=True)
