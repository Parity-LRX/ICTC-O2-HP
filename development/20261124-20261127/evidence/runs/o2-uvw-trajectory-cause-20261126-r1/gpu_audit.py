"""AMD acceptance of phase-matched causal controls, retaining all original gates."""
import gc,json,os
from pathlib import Path
import torch
from analyze import ROOT,REPAIR,read,save,sha,full,metric,ulp,brief,delta_summary,error,normratio,get
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD';torch.set_num_threads(4)
    source=Path((ROOT/'GPU-RESULT.txt').read_text().strip());manifest=read(source/'MANIFEST.json')
    old=Path((REPAIR/'VERIFY-RESULT.txt').read_text().strip());oldmanifest=read(old/'MANIFEST.json');oldgate=read(REPAIR/'results/verify-audit-amd-119415/RESULT.json')
    out=ROOT/'results'/('gpu-audit-amd-'+os.environ['SLURM_JOB_ID']);out.mkdir();verified={}
    def load(name):
        p=source/name;assert sha(p)==manifest['files'][name];verified[str(p)]=manifest['files'][name]
        return torch.load(p,map_location='cpu',weights_only=False)
    reproduction={};repeat={};independent={};crossgates={};decomposition=[];reproduced=[];state_matches={}
    for step in range(3):
        pair={}
        for engine in ('native','candidate'):
            a=load(f'{engine}-step{step}.pt');pair[engine]=a
            condition=read(source/f'{engine}-step{step}-CONDITION.json')
            state_matches[f'{engine}-step{step}']=condition['state']==read(old/engine/f'STEP-{step}.json')['before_state_identity']
            for label,name,arrays in (('training',f'TRAINING-{step}.pt',{k:a[k] for k in ('training','after_update_output')}),('mixed',f'MIXED-{step}.pt',a['mixed'])):
                assert sha(old/engine/name)==oldmanifest['cases'][engine][name]
                r=torch.load(old/engine/name,map_location='cpu',weights_only=False);metrics=full(arrays,r)
                save(out/f'{engine}-step{step}-{label}-REPRODUCTION.json',metrics);reproduction[f'{engine}-step{step}-{label}']=brief(metrics);del r
        a=load(f'native-repeat-step{step}.pt');repeat[str(step)]=brief(full(a,pair['native']));del a
        original=full(pair['candidate'],pair['native']);independent[str(step)]=brief(original)
        reproduced.extend({'step':step,'role':k,**v} for k,v in original.items() if not v['passed'])
        save(out/f'INDEPENDENT-step{step}.json',original)
        cross={}
        for state in ('native','candidate'):
            for engine in ('native','candidate'):cross[(engine,state)]=load(f'cross-step{step}-{engine}-at-{state}.pt')
            records=full(cross[('candidate',state)],cross[('native',state)])
            crossgates[f'step{step}-at-{state}']=brief(records);save(out/f'SAME-STATE-step{step}-at-{state}.json',records)
        # Decompose the original 202 roles, with any reproduction/context residual separate.
        for f in [r for r in oldgate['failures'] if r['object'].endswith(f'-{step}.pt')]:
            path=f['role'];path=path if f['object'].startswith('TRAINING') else '/mixed'+path
            a,b,d,r=[get(cross[k],path) for k in (('candidate','candidate'),('native','candidate'),('candidate','native'),('native','native'))]
            ca=get(pair['candidate'],path);nr=get(pair['native'],path)
            e=error(ca,nr);ei=error(a,b);es=error(b,r);qc=error(ca,a);qn=error(r,nr);dn=e.square().sum()
            def proj(v):return float((v*e).sum()/dn) if dn else 0
            decomposition.append({'step':step,'role':f['role'],'old_object':f['object'],'same_state':metric(a,b),'native_state_only':metric(b,r),
                'implementation_R_original_scale':normratio(ei,nr),'state_R_original_scale':normratio(es,nr),
                'implementation_projection':proj(ei),'state_projection':proj(es),'context_projection':proj(qc+qn),
                'context_R':normratio(qc+qn,nr),'closure_R':normratio(e-(qc+ei+es+qn),nr)})
        del pair,cross;gc.collect();print('step',step,'audited',flush=True)
    # Different random gauge with exactly the same model/optimizer state.
    normal=load('native-step0.pt')['mixed'];other=load('native-same-theta-replay-RNG.pt')['mixed']
    rng_only=full(other,normal);save(out/'NATIVE-SAME-THETA-DIFFERENT-RNG.json',rng_only)
    del normal,other
    # Original optimizer replay vs the actual full backward path, and same-gradient repetition.
    optimizer={};baseline=load('optimizer-native-grads.pt')
    again=load('optimizer-native-grads-repeat.pt');optimizer['same_gradient_repeat']=brief(full(again,baseline));del again
    for engine,label in (('native','native-grads'),('candidate','candidate-grads')):
        a=load('optimizer-'+label+'.pt');b=load(engine+'-step0.pt')['training'];state=a['state']
        expected={'model':{k:state['model'][k] for k in b['updated']},'optimizer':state['optimizer'],'ema':state['ema'],'scheduler':state['scheduler']}
        reference={'model':b['updated'],'optimizer':b['optimizer'],'ema':b['ema'],'scheduler':b['scheduler']}
        rows=full(expected,reference);optimizer[engine]={'check':brief(rows),'grad_norm':float(a['grad_norm']),
            'all_float_tensors_zero_difference':all(v.get('max_abs',0)==0 for v in rows.values())};save(out/('OPTIMIZER-'+engine+'.json'),rows)
        if engine=='candidate':optimizer['native_vs_candidate_update']=delta_summary({k:state['model'][k] for k in b['updated']},{k:baseline['state']['model'][k] for k in b['updated']})
        del a,b,expected,reference,rows
    del baseline
    cotangents={}
    for step in (1,2):
        n=load(f'cot-step{step}-at-native.pt');c=load(f'cot-step{step}-at-candidate.pt');entry={}
        for group in ('force','total'):
            own=full(c[group]['own'],n[group]['own']);shared=full(c[group]['shared'],n[group]['own'])
            cot=full(c[group]['cotangent'],n[group]['cotangent']) if group=='total' else {'force':{**metric(c[group]['cotangent'],n[group]['cotangent']),'ulp':ulp(c[group]['cotangent'],n[group]['cotangent'])}}
            entry[group]={'own':brief(own),'shared_native_cotangent':brief(shared),'cotangents':cot}
            save(out/f'COTANGENT-step{step}-{group}-OWN.json',own);save(out/f'COTANGENT-step{step}-{group}-SHARED.json',shared)
        # Validate diagnostic autograd traversal against the actual full helper.
        for branch,obj in (('native',n),('candidate',c)):
            actual=load(f'cross-step{step}-native-at-{branch}.pt')
            entry[branch+'_traversal']={'force':brief(full(obj['force']['own'],actual['mixed']['mixed'])),
                'total':brief(full(obj['total']['own'],actual['training']['gradients']))}
            del actual
        cotangents[str(step)]=entry;del n,c;gc.collect()
    frames=read(source/'FRAMES.json')
    def frame(phase):return next(r for r in frames if r['phase']==phase)
    gauge_checks={}
    for step in range(3):
        gauge_checks[str(step)]={part:frame(f'native-step{step}-{part}')['frame_sha256']==frame(f'candidate-step{step}-{part}')['frame_sha256'] for part in ('mixed','train','after')}
    result={'status':'PHASE_MATCHED_CAUSAL_AUDIT_COMPLETE','source':str(source),'operator_source_sha256':manifest['operator_source_sha256'],
        'reproduction':reproduction,'input_state_byte_hashes_match_old':state_matches,'repeat':repeat,'independent':independent,
        'independent_failures':reproduced,'same_state_actual_RNG':crossgates,'RNG_native_candidate_equal':manifest['state_and_rng_equalities'],
        'frames_native_candidate_equal':gauge_checks,'same_theta_different_RNG':brief(rng_only),
        'different_RNG_frame_changed':frame('native-step0-mixed')['frame_sha256']!=frame('native-same-theta-replay-RNG')['frame_sha256'],
        'optimizer_controls':optimizer,'cotangent_controls':cotangents,'decomposition':{'original_roles':len(decomposition),
            'same_state_failures':sum(not r['same_state']['passed'] for r in decomposition),
            'state_only_failures':sum(not r['native_state_only']['passed'] for r in decomposition),
            'state_projection_min':min(r['state_projection'] for r in decomposition),'state_projection_max':max(r['state_projection'] for r in decomposition),
            'max_context_R':max(r['context_R'] or 0 for r in decomposition),'max_closure_R':max(r['closure_R'] or 0 for r in decomposition)}}
    save(out/'RESULT.json',result);save(out/'DECOMPOSITION.json',decomposition);save(out/'VERIFIED-RAW-HASHES.json',verified)
    saved=read(Path((ROOT/'ANALYSIS-RESULT.txt').read_text().strip())/'RESULT.json')
    compact={'reproduction':reproduction,'state_hash_matches_old':state_matches,'repeat':repeat,'independent':independent,
        'same_state_actual_RNG':crossgates,'RNG_equal':result['RNG_native_candidate_equal'],'frames_equal':gauge_checks,
        'same_theta_different_RNG':result['same_theta_different_RNG'],'decomposition':result['decomposition'],
        'optimizer':{k:v for k,v in optimizer.items() if k!='native_vs_candidate_update'},
        'updated_parameters':{k:v for k,v in optimizer['native_vs_candidate_update'].items() if k!='different'},
        'cotangent_controls':cotangents,
        'old_trajectory_updates':[{key:(dict((k,v) for k,v in value.items() if k!='different') if key in ('before_parameters','updated_parameters') else value)
            for key,value in row.items() if key in ('step_zero_based','before_parameters','raw_gradient','updated_parameters','force_cotangent','optimizer_step_values')} for row in saved['trajectory']]}
    save(out/'COMPACT.json',compact)
    (ROOT/'GPU-AUDIT.txt').write_text(str(out)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('independent_failures','optimizer_controls','cotangent_controls')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
