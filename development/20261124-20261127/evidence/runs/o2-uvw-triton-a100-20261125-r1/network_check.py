"""Capture original outputs/parameter gradients without changing the recipe."""
import copy,json,os,sys
from pathlib import Path
import torch
from whole import configure,factory,save,sha,cpu
from diagnose import parameters,graph,state_identity,flags
def main():
    mode,backend,common,folder=sys.argv[1:];common=Path(common);folder=Path(folder)
    configure();model=factory('v3','native' if backend=='native' else 'candidate','checkpoint_ema','cuda')
    condition=json.loads((common/'CONDITION.json').read_text())
    assert sha(common/'START.pt')==condition['start_sha256']
    state=torch.load(common/'START.pt',map_location='cpu',weights_only=False)
    def restore():model.restore(dict(state,optimizer=copy.deepcopy(state['optimizer'])))
    restore();identity=parameters(model);g,_=graph(model,mode)
    assert identity['sha256']==condition['parameters']['sha256'] and g==condition['graph']
    assert state_identity(model.state())==state_identity(state)
    if mode=='ef':
        result=cpu(model.infer());restore();repeat=cpu(model.infer())
    else:
        result=cpu(model.train_step(capture=True));restore();repeat=cpu(model.train_step(capture=True))
        restore();result['force_mixed']=cpu(model.force_mixed())
        restore();repeat['force_mixed']=cpu(model.force_mixed())
    torch.save(result,folder/'OBJECTS.pt');torch.save(repeat,folder/'REPEAT.pt')
    save(folder/'CONDITION.json',{'hardware':flags(model),'parameters':identity,'graph':g,
        'common_state_sha256':condition['start_sha256'],'complete_state_identity':state_identity(state),
        'backend':backend,'candidate_commit':os.environ.get('O2_CANDIDATE_COMMIT')})
    save(folder/'RESULT.json',{'status':'CAPTURED_ALL_ROLES','mode':mode,'backend':backend})
if __name__=='__main__':main()
