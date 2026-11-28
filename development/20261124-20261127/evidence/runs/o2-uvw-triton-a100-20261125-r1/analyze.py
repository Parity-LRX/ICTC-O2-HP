"""CPU-only complete kernel inventories, same-state audits and numerical roles."""
import copy,csv,hashlib,importlib.util,json,os,shutil,subprocess,sys
from collections import Counter
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
WS=ROOT.parents[1]
from whole import compare_tree,save

def read(p):return json.loads(p.read_text())
def writecsv(p,rows):
    with p.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def normalize_receipt_fp32(original):
    """Recompute scalar normalization in FP32, preserving full-array max statistics.

    The original saved numerator is exactly FP32. Its denominator was either an
    exactly representable FP32 array maximum or the known Python 1e-7 floor.
    Recover that denominator and require bit-identical reproduction of the saved
    Python ratio before doing the actual metric division with FP32 tensors.
    No model output, derivative, reference, tolerance, or maximum is changed.
    """
    repaired=copy.deepcopy(original)
    for row in repaired['rows']:
        for group in ('roles','native_repeat'):
            for role in row[group].values():
                old=role['epsilon_max'];numerator=role['max_abs']
                if numerator:
                    inferred=numerator/old
                    denominator=1e-7 if abs(inferred-1e-7)<=1e-21 else float(torch.tensor(inferred,dtype=torch.float32))
                    assert numerator/denominator==old,'Could not exactly recover original normalization scalar'
                    assert denominator>=1e-7
                    provenance='exactly recovered FP32 maximum or known 1e-7 floor; original ratio reproduced bit for bit'
                else:
                    assert old==0
                    denominator=1e-7
                    provenance='zero residual; any valid denominator yields exactly zero'
                value=float(torch.tensor(numerator,dtype=torch.float32)/torch.tensor(denominator,dtype=torch.float32))
                assert (value<=1e-5)==role['passed'],'FP32 recomputation changed a gate; inspect separately'
                role.update(epsilon_max=value,original_python_scalar_ratio=old,
                    normalization_scale_used_fp32=float(torch.tensor(denominator,dtype=torch.float32)),
                    statistics_dtype='float32',normalization_provenance=provenance)
    repaired['statistics_correction']={'original_receipt_preserved':True,
        'scope':'only final scalar normalization; complete-array FP32 maxima from original GPU execution retained',
        'new_model_or_gpu_run':False,'gate_decisions_changed':False}
    return repaired
def main():
    assert os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(4)
    gpu=Path((ROOT/'LATEST-RESULT.txt').read_text().strip())
    out=ROOT/'results'/('analysis-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(out/'mpl-cache')
    spec=importlib.util.spec_from_file_location('old_analyzer',WS/'runs/o2-paired-diagnostic-a100-20261124-r1/analyze.py')
    parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    parser.ICTC_NAMES.add('_uvw_mm')
    save(out/'CLASSIFICATION.json',{'base_parser_sha256':sha(Path(spec.origin)),
        'adapter_sha256':sha(Path(__file__)),'added_ICTC_kernel_names':['_uvw_mm'],
        'denominator':'sum of actual cat=kernel X event durations; CPU and transfers separate'})
    workloads=[];conditions=[];gates={}
    for mode in ('ef','train'):
        for backend in ('native','ictc','tiled'):
            label=mode+'-'+backend
            row={'mode':mode,'backend':backend}
            for action in ('time','memory','trace'):
                folder=gpu/(label+'-'+action)
                if not (folder/'RESULT.json').exists():
                    row[action+'_status']='MISSING_OR_EXECUTION_ERROR';continue
                result=read(folder/'RESULT.json');condition=read(folder/'CONDITION.json')
                conditions.append({'label':label,'action':action,**condition})
                row[action+'_status']=result['status']
                if action=='time':row['wall_s']=result['median_s'];row['times_s']=result['times_s']
                elif action=='memory':
                    for kind in ('allocated','reserved'):row[kind+'_GiB']=result['peak_bytes'][kind]/2**30
                else:
                    cached=ROOT/'trace-cache'/label
                    destination=out/'traces'/label
                    if cached.exists() and not destination.exists():shutil.copytree(cached,destination)
                    parsed=parser.parse_trace(folder,out/'traces'/label)
                    row['kernel_ms']=parsed['kernel_ms'];row['kernel_calls']=parsed['kernel_calls']
                    row['HP_kernel_ms']=sum(v['total_ms'] for v in parsed['family'] if v['family']=='ICTC Triton')
                    row['HP_percent']=row['HP_kernel_ms']/row['kernel_ms']*100
            workloads.append(row)
        folder=gpu/(mode+'-native-gate')
        if not (folder/'OBJECTS.pt').exists():continue
        reference=torch.load(folder/'OBJECTS.pt',weights_only=False,map_location='cpu')
        for backend in ('native','ictc','tiled'):
            folder=gpu/(mode+'-'+backend+'-gate')
            if not (folder/'OBJECTS.pt').exists():continue
            objects=torch.load(folder/'OBJECTS.pt',weights_only=False,map_location='cpu')
            repeat=torch.load(folder/'REPEAT.pt',weights_only=False,map_location='cpu')
            metrics=compare_tree(objects,reference);native_repeat=compare_tree(repeat,objects)
            failed={k:v for k,v in metrics.items() if not v['passed']}
            repeat_failed={k:v for k,v in native_repeat.items() if not v['passed']}
            record={'metrics':metrics,'failed':failed,'self_repeat_failed':repeat_failed,
                'all_roles_pass':not failed,'historical_gates':'PRESERVED; independent candidate recheck'}
            save(out/(mode+'-'+backend+'-GATE.json'),record)
            gates[mode+'-'+backend]={'passed':not failed,'failed_roles':len(failed),'repeat_failed':len(repeat_failed),
                'failed_role_groups':dict(Counter(k.split('/')[1] for k in failed)),
                'examples':dict(list(failed.items())[:8])}
    audits={}
    for mode in ('ef','train'):
        selected=[r for r in conditions if r['mode']==mode]
        fields={}
        for key in ('parameter_sha256','common_state_sha256','graph','complete_state_identity'):
            values=[json.dumps(r.get(key),sort_keys=True) for r in selected]
            fields[key]=len(values)==9 and len(set(values))==1
        for key in ('gpu_uuid','node','torch','deterministic','TF32','AMP','CUDA_Graph','threads'):
            values=[json.dumps(r['hardware'].get(key),sort_keys=True) for r in selected]
            fields[key]=len(values)==9 and len(set(values))==1
        fields['complete_restored_state_verified']=len(selected)==9 and all(r.get('complete_restored_state_verified') for r in selected)
        audits[mode]={'fields':fields,'passed':all(fields.values())}
    save(out/'WORKLOADS.json',workloads);save(out/'PAIRED-AUDIT.json',audits);save(out/'GATES.json',gates)
    save(out/'ALL-CONDITIONS.json',conditions)
    comparisons=[]
    for mode in ('ef','train'):
        by={r['backend']:r for r in workloads if r['mode']==mode}
        if not all('wall_s' in by[b] for b in ('native','ictc','tiled')):continue
        r=by['tiled'];n=by['native'];old=by['ictc']
        comparisons.append({'mode':mode,'native_s':n['wall_s'],'frozen_hp_s':old['wall_s'],'candidate_s':r['wall_s'],
            'speedup_vs_frozen_hp':old['wall_s']/r['wall_s'],'speedup_vs_native':n['wall_s']/r['wall_s'],
            'native_allocated_GiB':n.get('allocated_GiB'),'old_allocated_GiB':old.get('allocated_GiB'),
            'candidate_allocated_GiB':r.get('allocated_GiB'),'candidate_HP_kernel_ms':r.get('HP_kernel_ms'),
            'candidate_HP_percent':r.get('HP_percent')})
    if comparisons:writecsv(out/'COMPARISON.csv',comparisons)
    save(out/'COMPARISON.json',comparisons)
    lines=['# UVW候选：V3 canonical配对热点诊断','',
        '本轮固定相同投影EMA、参数、输入、完整训练配方及7块重算；单A100串行测量。速度为同步墙钟10样本中位数，显存为独立峰值allocated。HP百分比分母是trace全部实际GPU kernel耗时和。','',
        '|模式|Torch s|冻结HP s|新候选 s|对冻结HP倍率|对Torch倍率|Torch/旧HP/新候选 GiB|新候选HP kernel占比|',
        '|---|---:|---:|---:|---:|---:|---|---:|']
    for r in comparisons:
        memory='/'.join(f'{r[k]:.6f}' if r[k] is not None else 'MISSING' for k in ('native_allocated_GiB','old_allocated_GiB','candidate_allocated_GiB'))
        lines.append(f"|{r['mode']}|{r['native_s']:.6f}|{r['frozen_hp_s']:.6f}|{r['candidate_s']:.6f}|{r['speedup_vs_frozen_hp']:.3f}x|{r['speedup_vs_native']:.3f}x|{memory}|{r['candidate_HP_percent']:.2f}%|")
    lines+=['','## 数值及条件','',f'配对完整字段：`{json.dumps(audits,ensure_ascii=False)}`。','']
    for label,gate in gates.items():lines.append(f"- {label}: all_roles_pass={gate['passed']}, failed_roles={gate['failed_roles']}, groups={gate['failed_role_groups']}, self_repeat_failed={gate['repeat_failed']}；[逐角色指标]({out/(label+'-GATE.json')})。")
    original_validation=read(gpu/'operator-validation/VALIDATION.json')
    save(out/'ORIGINAL-OPERATOR-VALIDATION.json',original_validation)
    validation=normalize_receipt_fp32(original_validation)
    save(out/'OPERATOR-VALIDATION.json',validation)
    maximum=max(role['epsilon_max'] for row in validation['rows'] for role in row['roles'].values())
    lines += ['',f"局部算子完整数组：first_order_pass={validation['first_order_pass']}，mixed_diagnostic_pass={validation['mixed_diagnostic_pass']}，全部已测角色最大epsilon_max={maximum:.9g}；[原生自重复及完整角色]({out/'OPERATOR-VALIDATION.json'})。新对象高阶正式门仍HOLD。"]
    lines += ['',f"统计精度补正：原局部回执的最终归一化比值曾用Python标量除法。已由保存的完整数组FP32最大差及其比值精确恢复原分母，并逐项要求重现原标量比值，再用FP32张量复算最终除法；全部判断不变。[原回执]({out/'ORIGINAL-OPERATOR-VALIDATION.json'})保留，已修正后续验证脚本。整网分量/relative-L2统计原本为FP32，训练FAIL未改。"]
    lines+=['','旧FAIL/HOLD不改判。通用UVW为共享权重三独立通道轴，角装配保留Torch；V3使用U=1固定标量特例，前向/输入与权重VJP/所需混合二阶矩阵乘均经同一IEEE FP32 Triton核。训练混合高阶从旧PyTorch重入转为Triton，kernel颜色分类覆盖随执行实现变化，比较整体墙钟及峰值才判断整网收益。新增高阶正式容差、任务精度和容量扫描不由本诊断授予资格。','',
        f'[完整kernel清单、逐事件、阶段及分类]({out/"traces"})；[原始10个样本及allocated/reserved]({gpu})。',
        '',f'候选源码提交：`{read(ROOT/"PREFLIGHT.json")["candidate_commit"]}`；[源码及独立解包封存回执]({ROOT/"FINAL-SNAPSHOT.json"})。',
        '',f'![同条件kernel分布、绝对耗时及显存]({ROOT/"figures/UVW-V3-COMPARISON-r2.png"})',
        '',f'[流程与软件来源]({ROOT/"SOFTWARE-REFERENCES.md"})中记录所用技能，并列出已核对的Scientific Agent Skills参考文献。']
    report='\n'.join(lines)+'\n'
    (out/'REPORT.zh-CN.md').write_text(report)
    (ROOT/'REPORT.zh-CN.md').write_text(report)
    save(out/'SOURCE-SEAL.json',{str(p.relative_to(ROOT)):sha(p) for p in ROOT.glob('*.py')})
    (ROOT/'LATEST-ANALYSIS.txt').write_text(str(out)+'\n')
    subprocess.run([sys.executable,str(ROOT/'figures/uvw_v3_hotspots.py')],check=True)
    print(json.dumps({'analysis':str(out),'comparison':comparisons,'gates':gates,'paired':audits}),flush=True)
if __name__=='__main__':main()
