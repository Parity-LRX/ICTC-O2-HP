"""CPU-only aggregation of independent canonical checkpoints and all 29 site receipts."""
import csv
import json
import os
from pathlib import Path

from common import start,finish,save,ROOT,WS


def read_stage(stage):
    folder=Path((ROOT/(stage.upper()+'-RESULT.txt')).read_text().splitlines()[0])
    return json.loads((folder/'RESULT.json').read_text())


def metrics(comparison):
    for i,v in enumerate(comparison['output']):yield 'output_'+str(i),v
    for group in ('full_vjp','mixed'):
        for name,value in comparison[group].items():yield group+':'+name,value


def compact(comparison):
    failures={k:v for k,v in metrics(comparison) if not v.get('pass',True)}
    return {'output_pass':comparison['output_pass'],'full_vjp_pass':comparison['full_vjp_pass'],
        'mixed_pass':comparison['mixed_pass'],'all_pass':comparison['all_pass'],'failed_roles':failures}


def extrema(rows):
    values=[]
    for row in rows:
        for transformation,comparison in row['symmetry'].items():
            values.extend({'site_id':row['site_id'],'transform':transformation,'role':name,**value}
                          for name,value in metrics(comparison) if 'max_abs' in value)
    return {'max_abs':max(values,key=lambda x:x['max_abs']),
        'max_relative_l2':max((v for v in values if v['relative_l2'] is not None),key=lambda x:x['relative_l2']),
        'max_output_relative_l2':max((v for v in values if v['role'].startswith('output_') and v['relative_l2'] is not None),key=lambda x:x['relative_l2']),
        'threshold':'UNAPPROVED: numerical diagnostics only'}


def triplet(value):
    return '/'.join('PASS' if value[k] else 'FAIL' for k in ('output_pass','full_vjp_pass','mixed_pass'))


def main():
    output,_,_,_=start('summary')
    conversion,local,consumers,frame,qa=(read_stage(s) for s in ('convert','local','consumers','frame','qa'))
    result={'status':'CANONICAL_CPU_COMPLETE_WITH_CONSUMER_FAILS_AND_REPRESENTATION_HOLD',
        'model_id':conversion['model_id'],'jobs':{s:read_stage(s)['job_id'] for s in ('convert','local','consumers','frame','qa')},
        'checkpoints':{r:{k:v for k,v in row.items() if k in ('checkpoint','sha256','source_identity','unchanged_copied_parameters','projected_parameter_tensors','strict_reload_all_parameters_equal')}
                       for r,row in conversion['roles'].items()},'roles':{},
        'calibration':conversion['calibration'],'representation_gate':'HOLD_THRESHOLD_UNAPPROVED',
        'gpu_jobs':0,'gpu_hours':0,'training_steps':0,'optimizer_steps':0}
    text=['# canonical O2 CPU转换、29处及后续计算验证终态','',
        '2026-11-24。仅CPU验证，无A100/GPU、训练或优化器更新；原冻结基线和旧FAIL保持。',
        '', '**结论：独立canonical checkpoint已生成并严格重载；局部与后续计算分别验收，后续计算收缩顺序仍有FAIL，旋转/反射数值残差因新FP32门未审定保持HOLD，不授予完整接入资格。**',
        '', '## 固定基与checkpoint','',
        '实际Wigner/m布局在本次全部(l,m)上选择C=I；这次统一基下K=A，B被排除。该结论来自新校准，不能扩展为所有约定都可以直接删raw B。',
        '', '| 来源 | 原样复制Parameter张量 | 投影Parameter张量 | 独立checkpoint SHA256 |', '| --- | --- | --- | --- |']
    for role,row in conversion['roles'].items():
        text.append(f"| {role} | {row['unchanged_copied_parameters']} | {row['projected_parameter_tensors']} | `{row['sha256']}` |")
    text+=['','14个模块×2正频率=28个投影张量；其余264个Parameter原值一致，合计292个来源张量。新模型参数为packed K，无旧B自由度；新checkpoint不是原模型函数。球面系数采用Gaunt不变量结构及原网格标量拟合，完整系数/差异保存在转换回执。','',
        '## 分层结果','','序列均为前向/full VJP/混合二阶；局部按O2前向/VJP门、混合按既有高阶门，后续计算按完整block/整网门，角色不混合归一化。','']
    for role in ('checkpoint_original','checkpoint_ema'):
        lr=local['roles'][role];cr=consumers['roles'][role]
        assert len(lr)==29 and len(cr)==30
        assert local['checkpoint_identities'][role]['sha256']==consumers['checkpoint_identities'][role]['sha256']==qa['checkpoint_identities'][role]['sha256']==conversion['roles'][role]['sha256']
        counts={scope:{k:sum(int(row['implementation'][k]) for row in rows if row['site_id']!='ALL29')
                       for k in ('output_pass','full_vjp_pass','mixed_pass','all_pass')}
                for scope,rows in (('local',lr),('consumer',cr))}
        repeat_failures=[{'scope':scope,'site_id':row['site_id']} for scope,rows in (('local',lr),('consumer',cr)) for row in rows if not row['native_repeat']['all_pass']]
        qr=qa['roles'][role]
        qfail={name:{n:v for n,v in values.items() if not v['pass']} if isinstance(values,dict) and 'pass' not in values else values
               for name,values in qr['implementation'].items()}
        qpass=all(v['pass'] for name,values in qr['implementation'].items() for v in (values.values() if name.endswith(('vjp','mixed')) else (values,)))
        coefficient_max=max(row['frozen_grid_max_abs_difference'] for row in conversion['roles'][role]['coefficient_diagnostics'].values())
        rs={'counts':counts,'native_repeat_failures':repeat_failures,
            'local':{row['site_id']:compact(row['implementation']) for row in lr},
            'consumers':{row['site_id']:compact(row['implementation']) for row in cr},
            'local_symmetry_extrema':extrema(lr),'consumer_symmetry_extrema':extrema(cr),
            'actual_energy_force_mixed_all_pass':qpass,'actual_energy_force_mixed':qfail,
            'energy':qr['reference_energy'],'force_dtype':qr['force_dtype'],'energy_dtype':qr['energy_dtype'],
            'parameter_perturbation':qr['parameter_perturbation'],
            'coefficient_max_abs_change':coefficient_max}
        result['roles'][role]=rs
        text.append(f"- {role}：局部 {counts['local']['output_pass']}/{counts['local']['full_vjp_pass']}/{counts['local']['mixed_pass']}（分母29）；后续计算 {counts['consumer']['output_pass']}/{counts['consumer']['full_vjp_pass']}/{counts['consumer']['mixed_pass']}（分母29）；ALL29 {triplet(cr[-1]['implementation'])}；真实E/F及force-parameter混合VJP整体{'PASS' if qpass else 'FAIL'}。")
        text.append(f"  原生自重复失败位置：{repeat_failures or '无'}。新球面系数相对旧网格最大绝对变化 {coefficient_max:.8g}。")
    text+=['','## 29处逐项结果','','| 位置 | 原始局部 | EMA局部 | 原始权重：后续计算 | EMA：后续计算 |','| --- | --- | --- | --- | --- |']
    ids=[row['site_id'] for row in local['roles']['checkpoint_original']]
    for label in ids:
        values=[triplet(result['roles'][role][scope][label]) for scope in ('local','consumers') for role in ('checkpoint_original','checkpoint_ema')]
        text.append('| '+label+' | '+' | '.join(values)+' |')
    text+=['','## 旋转/反射、实际几何与导数边界','']
    for role,row in result['roles'].items():
        for scope in ('local','consumer'):
            ex=row[scope+'_symmetry_extrema'];worst=ex['max_output_relative_l2']
            text.append(f"- {role} {scope}：输出最大全对象relative-L2={worst['relative_l2']:.8g}（{worst['site_id']} / {worst['transform']}）；所有输出/导数角色最大绝对残差={ex['max_abs']['max_abs']:.8g}。完整分角色残差在JSON；未审定表示门，保持HOLD。")
    text+=['','局部覆盖每处旋转/反射及对应full VJP/输入-参数混合方向积；后续计算覆盖坐标/节点输入/全部活动参数及坐标-参数混合方向积。实际原帧和mask分支另由frame阶段重验，原mask未改。后续计算验证使用固定完整有向4节点图，含重复边及一对相反PBC位移，未取得动态构图/stress或公开数据任务资格。',
        '', '真实能量头输出/归约FP64，force与参数混合VJP为FP32。QA另对28个K作确定性内存扰动后验证结构，再精确恢复全部参数；不是训练、optimizer step或新checkpoint更新。',
        '', '## 未闭合项与下一步边界','',
        '后续计算FAIL保留，先定位具体窗口/归约传播及近零角色，不能更换权重/输入、放宽门或用ALL29通过覆盖逐处失败。新表示/系数容差未审定项HOLD；实际mask跨分支、动态图/PBC邻居重建、stress及公开数据任务精度另验。当前不提交A100/GPU或训练。',
        '', '## 快照与作业','']
    for stage in ('convert','local','consumers','frame','qa'):
        snapshot=(ROOT/(stage.upper()+'-SNAPSHOT.txt')).read_text().splitlines()
        text.append(f"- {stage}：CPU作业{result['jobs'][stage]}；Git `{snapshot[1]}`，路径 `{snapshot[0]}`。")
    report='\n'.join(text)+'\n'
    save(output/'SUMMARY.json',result);(output/'REPORT.zh-CN.md').write_text(report)
    save(ROOT/'FINAL.json',result);(ROOT/'FINAL.zh-CN.md').write_text(report)
    (WS/'architectures/equiformerv3/CANONICAL-CPU-20261124.zh-CN.md').write_text(report)
    tsv=WS/'architectures/equiformerv3/CANONICAL-O2-SITES.tsv'
    with tsv.open() as stream:reader=csv.DictReader(stream,delimiter='\t');fields=reader.fieldnames;rows=list(reader)
    assert len(rows)==29 and {r['site_id'] for r in rows}==set(ids)
    for row in rows:
        label=row['site_id']
        lp=all(result['roles'][r]['local'][label]['all_pass'] for r in result['roles'])
        cp=all(result['roles'][r]['consumers'][label]['all_pass'] for r in result['roles'])
        row['status']='CPU_LOCAL_'+('PASS' if lp else 'FAIL')+'_CONSUMER_'+('PASS' if cp else 'FAIL')+'_REPRESENTATION_HOLD'
    with tsv.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerows(rows)
    finish('summary',output,result)


if __name__=='__main__':main()
