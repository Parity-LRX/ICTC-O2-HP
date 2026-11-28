"""CPU closure: locate failed roles, retain frame boundary evidence, seal final documents."""
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT=Path(os.environ['O2_GENERAL_RUN_DIR'])
WS=ROOT.parents[1]
SOURCE=Path(__file__).resolve().parent


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    assert os.environ.get('SLURM_JOB_ID')
    spec=importlib.util.spec_from_file_location('canonical_final_prepare',ROOT/'prepare.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    seal=json.loads((SOURCE/'SEAL.json').read_text())
    assert seal['job_id']==os.environ['SLURM_JOB_ID']
    for name,digest in seal['files'].items():assert helper.sha(SOURCE/name)==digest,name
    result=json.loads((ROOT/'FINAL.json').read_text())
    failure_rows=[]
    for role,row in result['roles'].items():
        for label,comparison in row['consumers'].items():
            failed=comparison['failed_roles']
            if not failed:continue
            active=[{'role':name,**v} for name,v in failed.items() if 'max_abs' in v]
            failure_rows.append({'checkpoint_role':role,'site_id':label,'failed_roles':len(failed),
                'component_failed_roles':sum(v.get('component_fail_count',0)>0 for v in failed.values()),
                'relative_l2_failed_roles':sum(v.get('relative_l2') is None or v['relative_l2']>1e-5 for v in failed.values() if 'relative_l2' in v),
                'max_abs_failure':max(active,key=lambda x:x['max_abs']) if active else None,
                'max_relative_failure':max((v for v in active if v['relative_l2'] is not None),key=lambda x:x['relative_l2']) if active else None,
                'all_failed_metrics':failed})
    frame_folder=Path((ROOT/'FRAME-RESULT.txt').read_text().splitlines()[0])
    frame=json.loads((frame_folder/'RESULT.json').read_text())
    frame_rows={name:{k:row[k] for k in ('regular_forward_transport','all_rows_forward_transport',
                                         'regular_inverse_transport','all_rows_inverse_transport','same_mask_branch')}
                for name,row in frame['mirrors'].items()}
    for row in result['checkpoints'].values():assert helper.sha(Path(row['checkpoint']))==row['sha256']
    closure={'job_id':os.environ['SLURM_JOB_ID'],'failure_diagnostics':failure_rows,
             'frame_diagnostics':frame_rows,'baseline_files':seal['baseline_files'],'baseline_links':seal['baseline_links'],
             'baseline_manifest_sha256':seal['baseline_manifest_sha256'],'all_checkpoint_hashes_rechecked':True,
             'gpu_hours':0,'training_steps':0,'optimizer_steps':0}
    dump(ROOT/'CLOSURE.json',closure)
    result['closure']={k:v for k,v in closure.items() if k!='failure_diagnostics'}
    dump(ROOT/'FINAL.json',result)
    lines=['','## 失败角色与帧边界补充（CPU封存）','',
           '| 来源/位置 | 失败角色数 | 有分量超限的角色数 | relative-L2超限角色数 | 最大绝对误差 |',
           '| --- | --- | --- | --- | --- |']
    for row in failure_rows:
        maximum=row['max_abs_failure']
        lines.append(f"| {row['checkpoint_role']}/{row['site_id']} | {row['failed_roles']} | {row['component_failed_roles']} | {row['relative_l2_failed_roles']} | {maximum['max_abs']:.8g}（{maximum['role']}） |")
    lines+=['','以上是失败归类，不是已证明根因；不能仅因relative-L2很小而取消逐分量门，也不能对近零角色改分母。完整失败指标见CLOSURE.json及逐处JSON。','']
    for name,row in frame_rows.items():
        lines.append(f"- 实际帧{name}：regular forward max_abs={row['regular_forward_transport']['max_abs']:.8g}，含mask行forward max_abs={row['all_rows_forward_transport']['max_abs']:.8g}；mask不改，跨分支域保持HOLD。")
    lines+=['',f"最终CPU封存再次复核{seal['baseline_files']}个冻结文件及{seal['baseline_links']}条软链，原始/EMA新checkpoint哈希复核一致。所有基线、旧FAIL、原mask与图语义保持，新增GPUh=0，无训练。"]
    report=(ROOT/'FINAL.zh-CN.md').read_text()+'\n'.join(lines)+'\n'
    (ROOT/'FINAL.zh-CN.md').write_text(report)
    (WS/'architectures/equiformerv3/CANONICAL-CPU-20261124.zh-CN.md').write_text(report)
    snapshot=ROOT/'snapshots'/('final-'+os.environ['SLURM_JOB_ID'])
    snapshot.mkdir(parents=True,exist_ok=False)
    shutil.copytree(SOURCE,snapshot/'control',ignore=shutil.ignore_patterns('.git'))
    for pattern in ('*.py','*.sbatch','*.md','*.json','*.txt','*.psv'):
        for source in ROOT.glob(pattern):shutil.copy2(source,snapshot/source.name)
    for relative in ('AGENTS.md','HANDOFF.md','architectures','analysis/equiformerv3-optimization-plan-20261124-r1'):
        source=WS/relative;target=snapshot/'workspace'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if source.is_dir():shutil.copytree(source,target)
        else:shutil.copy2(source,target)
    index={}
    for stage in ('convert','local','consumers','frame','qa','summary'):
        folder=Path((ROOT/(stage.upper()+'-RESULT.txt')).read_text().splitlines()[0])
        pointer=(ROOT/(stage.upper()+'-SNAPSHOT.txt')).read_text().splitlines()
        index[stage]={'result':str(folder),'result_sha256':helper.sha(folder/'RESULT.json'),
            'snapshot':pointer[0],'snapshot_commit':pointer[1]}
    dump(snapshot/'STAGE-INDEX.json',index)
    dump(snapshot/'FINAL-SEAL.json',{str(p.relative_to(snapshot)):helper.sha(p) for p in snapshot.rglob('*') if p.is_file()})
    for args in (['init','--quiet','--template='],['config','user.name','O2 snapshot'],
                 ['config','user.email','o2-snapshot@localhost'],['add','.'],
                 ['commit','--quiet','-m','Seal canonical O2 checkpoints, all29 CPU derivative results and consumer failures']):helper.git(snapshot,args)
    commit=helper.git(snapshot,['rev-parse','HEAD'])
    (ROOT/'FINAL-SNAPSHOT.txt').write_text(str(snapshot)+'\n'+commit+'\n')
    print(json.dumps({'status':result['status'],'final_snapshot':str(snapshot),'commit':commit,
        'baseline_files':seal['baseline_files'],'failed_consumer_rows':len(failure_rows),'gpu_hours':0}),flush=True)


if __name__=='__main__':main()
