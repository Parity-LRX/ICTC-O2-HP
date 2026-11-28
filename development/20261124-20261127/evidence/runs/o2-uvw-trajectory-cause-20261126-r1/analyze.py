"""AMD counterfactual decomposition of every preserved three-step failure."""
import csv,gc,json,os,sys
from collections import Counter
from pathlib import Path
import torch
ROOT=Path(os.environ['O2_TRAJECTORY_ROOT']);WS=ROOT.parents[1];REPAIR=WS/'runs/o2-uvw-fp32-repair-a100-20261126-r1'
os.environ['O2_REPAIR_ROOT']=str(REPAIR);os.environ['O2_REPAIR_SOURCE']=str(REPAIR/'snapshots/qualification-119423/ictc-source')
sys.path.insert(0,str(REPAIR/'snapshots/qualification-119423/control'))
from common import common_state,metric,state_identity,sha,save
from metrics import full,ulp
def read(path):return json.loads(path.read_text())
def get(tree,path):
    for key in path.strip('/').split('/'):
        tree=tree[int(key)] if isinstance(tree,(list,tuple)) else tree[key]
    return tree
def brief(rows):
    failed={k:r for k,r in rows.items() if not r['passed']}
    return {'roles':len(rows),'failed':len(failed),'failed_roles':list(failed),'maximum_relative_l2':max((v.get('relative_l2',0) or 0) for v in rows.values()),
        'maximum_absolute':max(v.get('max_abs',0) for v in rows.values())}
def error(a,r):return (a-r).float()
def normratio(d,r):
    n=r.float().norm();v=d.norm()
    return float(v/n) if n else (0.0 if not v else None)
def difference(a,r):
    out={}
    for name in r:
        x,y=a[name],r[name];m=metric(x,y)
        if x is None or y is None:out[name]={**m,'numerically_equal':x is y};continue
        u=ulp(x,y);out[name]={**m,'numerically_equal':bool(torch.equal(x,y)), 'ulp':u,
            'different_elements':int((x!=y).sum()),'numel':x.numel()}
    return out
def delta_summary(a,r):
    rows=difference(a,r);different={k:v for k,v in rows.items() if not v['numerically_equal']}
    return {'roles':len(rows),'different_tensors':len(different),'different_elements':sum(v.get('different_elements',0) for v in different.values()),
        'failed':sum(not v['passed'] for v in rows.values()),'maximum_relative_l2':max((v.get('relative_l2',0) or 0) for v in rows.values()),
        'maximum_absolute':max(v.get('max_abs',0) for v in rows.values()),'different':different}
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
    out=ROOT/'results'/('analysis-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True)
    independent=Path((REPAIR/'VERIFY-RESULT.txt').read_text().strip());replay=Path((REPAIR/'REPLAY-RESULT.txt').read_text().strip())
    oldgate=read(REPAIR/'results/verify-audit-amd-119415/RESULT.json');gate=read(REPAIR/'results/replay-audit-amd-119437/RESULT.json')
    assert len(oldgate['failures'])==202 and not gate['failures'] and gate['status']=='WHOLE_TRAINING_GATE_PASS'
    manifest=read(independent/'MANIFEST.json');rmanifest=read(replay/'MANIFEST.json');initial,initial_receipt=common_state('train')
    failures=oldgate['failures'];breakdown=Counter((r['object'],r['role'].split('/')[1]) for r in failures)
    audit={'failure_count':len(failures),'breakdown':[{'object':k[0],'group':k[1],'count':v} for k,v in breakdown.items()],
        'native_repeat_failed':len([r for r in failures if r['case']=='native-repeat']),
        'candidate_repeat_failed':len([r for r in failures if r['case']=='candidate-repeat']),
        'all_component_failures':sum(r.get('component_fail_count',0) for r in failures),
        'worst':sorted(failures,key=lambda r:r.get('relative_l2',0) or 0,reverse=True)[:12]}
    save(out/'PRESERVED-FAILURES.json',audit)
    source_hashes={};proof={};decomp=[];trajectory=[];previous=initial['model']
    def load(path,digest):
        assert sha(path)==digest;source_hashes[str(path)]=digest
        return torch.load(path,map_location='cpu',weights_only=False)
    # The original loss-input capture supplies unchanged normalized labels.
    cause=WS/'runs/o2-uvw-training-cause-20261125-r1';probe=Path((cause/'GPU-RESULT.txt').read_text().strip())/'native.pt'
    inputs=torch.load(probe,map_location='cpu',weights_only=False);target=inputs['loss_inputs']['forces']['target'];probe_force=inputs['loss_inputs']['forces']['pred']
    save(out/'LABEL-PROVENANCE.json',{'path':str(probe),'sha256':sha(probe),'shape':list(target.shape),'module':inputs['loss_inputs']['forces']['module']})
    del inputs
    for step in range(3):
        original={};cross={};states={}
        for backend in ('native','candidate'):
            original[backend]=load(independent/backend/f'TRAINING-{step}.pt',manifest['cases'][backend][f'TRAINING-{step}.pt'])
            original[backend]['force_mixed']=load(independent/backend/f'MIXED-{step}.pt',manifest['cases'][backend][f'MIXED-{step}.pt'])
            point='initial' if step==0 else backend+f'-update-{step}'
            for engine in ('native','candidate'):
                cross[(engine,backend)]=load(replay/engine/(point+'.pt'),rmanifest['cases'][engine][point+'.pt'])
            prior=initial if step==0 else None
            if step:
                p=torch.load(independent/backend/f'TRAINING-{step-1}.pt',map_location='cpu',weights_only=False)['training']
                prior={'model':dict(initial['model'],**p['updated']),'optimizer':p['optimizer'],'ema':p['ema'],'scheduler':p['scheduler']};del p
            states[backend]=prior
            expected=read(independent/backend/f'STEP-{step}.json')['before_state_identity']
            assert state_identity(prior)==expected
            for engine in ('native','candidate'):
                assert read(replay/engine/(point+'-CONDITION.json'))['state_identity']==expected
            for label,old,new in (
                ('training', {k:original[backend][k] for k in ('training','after_update_output')}, {k:cross[(backend,backend)][k] for k in ('training','after_update_output')}),
                ('mixed',original[backend]['force_mixed'],cross[(backend,backend)]['force_mixed'])):
                rows=full(old,new);proof[f'{backend}-step{step}-{label}']=brief(rows)
        n,c=original['native'],original['candidate'];nn=cross[('native','native')];nc=cross[('native','candidate')];cn=cross[('candidate','native')];cc=cross[('candidate','candidate')]
        g=difference(c['training']['gradients'],n['training']['gradients']);updated=delta_summary(c['training']['updated'],n['training']['updated'])
        before=delta_summary({k:states['candidate']['model'][k] for k in c['training']['updated']},{k:states['native']['model'][k] for k in n['training']['updated']})
        outputs=full(c['training']['output'],n['training']['output']);moments=full(c['training']['optimizer'],n['training']['optimizer'])
        force_n=n['training']['output']['forces'];force_c=c['training']['output']['forces']
        if step==0:save(out/'LOSS-INPUT-CONTEXT.json',{'initial_force_vs_earlier_native_probe':metric(force_n,probe_force),'labels_only_reused':True,'RNG_context_not_matched':True})
        def cot(force):
            f=force.detach().clone().requires_grad_();residual=f-target;loss=20*torch.linalg.vector_norm(residual,dim=-1).mean()
            return torch.autograd.grad(loss,f)[0],residual.detach()
        cotn,resn=cot(force_n);cotc,resc=cot(force_c);atomdiff=(cotc-cotn).norm(dim=-1);top=torch.argsort(atomdiff,descending=True)[:8]
        trajectory.append({'step_zero_based':step,'before_state_hashes_equal':state_identity(states['native'])==state_identity(states['candidate']),
            'before_parameters':before,'raw_gradient':{'summary':brief(g),'different_tensors':sum(not v['numerically_equal'] for v in g.values()),'different_K':sum(not v['numerically_equal'] for k,v in g.items() if k.endswith('.K'))},
            'updated_parameters':updated,'optimizer':brief(moments),'outputs':outputs,'force_cotangent':{**metric(cotc,cotn),'ulp':ulp(cotc,cotn)},
            'force_residual_min_norm':float(resn.norm(dim=-1).min()),'cotangent_top_atoms':[{'atom':int(i),'native_residual_norm':float(resn[i].norm()),'candidate_residual_norm':float(resc[i].norm()),'cotangent_difference_norm':float(atomdiff[i])} for i in top],
            'optimizer_groups':[{k:v for k,v in group.items() if k!='params'} for group in states['native']['optimizer']['param_groups']],
            'optimizer_step_values':sorted({float(v['step']) for v in states['native']['optimizer']['state'].values()}),
            'ema_equal':states['native']['ema']['num_updates']==states['candidate']['ema']['num_updates']})
        save(out/f'RAW-GRADIENTS-step{step}.json',g)
        for failed in [v for v in failures if v['case']=='candidate' and v['object'].endswith(f'-{step}.pt')]:
            mixed=failed['object'].startswith('MIXED');path=failed['role']
            if mixed:
                path='/force_mixed'+path;oa=get(c,path);orr=get(n,path)
            else:oa=get(c,path);orr=get(n,path)
            a,b,d,r=(get(tree,path) for tree in (cc,nc,cn,nn))
            et=error(oa,orr);ei=error(a,b);es=error(b,r);ef=error(a,d);ein=error(d,r)
            qa=error(oa,a);qr=error(r,orr);den=et.square().sum()
            def projection(v):return float((v*et).sum()/den) if den else 0
            decomp.append({'step':step,'object':failed['object'],'role':failed['role'],
                'original_metric':failed,'same_candidate_state':metric(a,b),'native_state_difference':metric(b,r),
                'same_native_state':metric(d,r),'candidate_state_difference':metric(a,d),
                'implementation_R_on_original_scale':normratio(ei,orr),'state_R_on_original_scale':normratio(es,orr),
                'state_projection_on_total':projection(es),'implementation_projection_on_total':projection(ei),
                'candidate_replay_projection':projection(qa),'native_replay_projection':projection(qr),
                'closure_R_on_original_scale':normratio(et-(qa+ei+es+qr),orr),
                'other_path_closure_R':normratio(error(a,r)-(ef+ein),r)})
        del original,cross,states,n,c,nn,nc,cn,cc,g,updated,before,moments;gc.collect();print('step',step,'done',flush=True)
    assert len(decomp)==202
    result={'status':'SAVED_ARRAY_ANALYSIS_COMPLETE_RNG_CONTEXT_UNCONTROLLED','failure_inventory':audit,'same_backend_replay_checks':proof,
        'trajectory':trajectory,'decomposition_summary':{'roles':len(decomp),'same_state_failures':sum(not r['same_candidate_state']['passed'] for r in decomp),
            'native_state_only_failures':sum(not r['native_state_difference']['passed'] for r in decomp),
            'candidate_state_only_failures':sum(not r['candidate_state_difference']['passed'] for r in decomp),
            'state_projection_min':min(r['state_projection_on_total'] for r in decomp),'state_projection_max':max(r['state_projection_on_total'] for r in decomp)},
        'operator_source_sha256':manifest['operator_source_sha256'],'precision':'energy output FP64; all non-energy analysis FP32; integer ULP'}
    save(out/'DECOMPOSITION.json',decomp);save(out/'RESULT.json',result);save(out/'SOURCE-HASHES.json',source_hashes)
    (ROOT/'ANALYSIS-RESULT.txt').write_text(str(out)+'\n')
    print(json.dumps({'summary':result['decomposition_summary'],'inventory':audit['breakdown'],'trajectory':[{k:v for k,v in r.items() if k not in ('before_parameters','updated_parameters','optimizer_groups')} for r in trajectory]},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
