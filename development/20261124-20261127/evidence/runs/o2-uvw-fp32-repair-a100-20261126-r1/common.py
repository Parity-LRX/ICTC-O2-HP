"""Fixed old controls and same-state replay for UVW repair experiments."""
import copy,json,os,sys
from pathlib import Path
ROOT=Path(os.environ.get('O2_REPAIR_ROOT',Path(__file__).resolve().parent));WS=ROOT.parents[1]
OLD=WS/'runs/o2-uvw-triton-a100-20261125-r1';CONTROL=OLD/'snapshots/preflight-119174'
SOURCE=Path(os.environ.get('O2_REPAIR_SOURCE',WS/'ICTC-O2-HP/src'))
os.environ.update(O2_WHOLE_ROOT=str(OLD),O2_UVW_TILED='1',O2_UVW_SOURCE=str(SOURCE))
sys.path[:0]=[str(CONTROL),str(SOURCE)]
import torch
from whole import configure,factory,cpu,save,sha,seed,metric,compare_tree
from diagnose import flags,parameters,graph,state_identity
def common_state(mode):
    path=Path(json.loads((OLD/'PREFLIGHT.json').read_text())['common_states'][mode])
    receipt=json.loads((path/'CONDITION.json').read_text());assert sha(path/'START.pt')==receipt['start_sha256']
    return torch.load(path/'START.pt',map_location='cpu',weights_only=False),receipt
def restore(model,state,receipt):
    model.restore(dict(state,optimizer=copy.deepcopy(state['optimizer'])))
    assert state_identity(model.state())==state_identity(state)
    assert parameters(model)['sha256']==receipt['parameters']['sha256']
def startup(model,state,receipt,mode):
    restore(model,state,receipt);g,_=graph(model,mode);assert g==receipt['graph']
    return {'hardware':flags(model),'state_identity':state_identity(state),'parameters':receipt['parameters'],
        'graph':g,'common_state_sha256':receipt['start_sha256'],'operator_source_sha256':sha(SOURCE/'ictc_o2_hp/uvw_triton.py')}
