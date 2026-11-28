"""Render measured, provenance-linked numerical and performance results on AMD."""
import json,os
from pathlib import Path
ROOT=Path(os.environ['O2_REPAIR_ROOT']);WS=ROOT.parents[1]
def read(path):return json.loads(path.read_text())
def main():
    assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
    numerical=read(ROOT/'NUMERICAL-SUMMARY.json');assert numerical['status']=='WHOLE_TRAINING_GATE_PASS'
    assert set(numerical['same_state_groups'])=={'EF','initial','native-update-1','native-update-2','candidate-update-1','candidate-update-2'}
    audit=Path((ROOT/'PERFORMANCE-AUDIT.txt').read_text().strip());perf=read(audit/'RESULT.json');assert perf['status']=='PERFORMANCE_AUDIT_PASS'
    replay=Path((ROOT/'REPLAY-AUDIT.txt').read_text().strip());gate=read(replay/'RESULT.json');selected=read(ROOT/'SELECTED.json')
    source=Path((ROOT/'REPLAY-RESULT.txt').read_text().strip());condition=read(source/'native/initial-CONDITION.json')
    count=condition['parameters']['count']
    groups={}
    for point,rows in numerical['same_state_groups'].items():
        for group,r in rows.items():
            g=groups.setdefault(group,{'roles':0,'failed':0,'maximum_relative_l2':0,'maximum_absolute':0})
            for key in ('roles','failed'):g[key]+=r[key]
            for key in ('maximum_relative_l2','maximum_absolute'):g[key]=max(g[key],r[key])
    lines=['# UVW FP32真实训练修复与配对性能（2026-11-26）','',
        f'候选`{selected["commit"]}`，分支`codex/uvw-fp32-repair-v2`，tag`snapshot/uvw-fp32-segments-v2-20261126`，未推送。算子源码SHA256：`{selected["operator_source_sha256"]}`。独立仓库`{WS/"ICTC-O2-HP"}`，只修改`src/ictc_o2_hp/uvw_triton.py`；旧`df12a3a`、冻结基线和历史FAIL/HOLD原样保留。','',
        '## 结论与适用边界','',
        '固定canonical投影EMA模型、公开真实输入/标签、原始Parameter和损失/优化器下，原先4个完整loss参数梯度与50个真实force-only混合角色的失败已在新候选的同状态对照中消除。初始点及两条实际训练轨迹分别更新1/2步后到达的状态，共5个状态全部通过原门，未固定损失余切。E/F、完整VJP、全部参数梯度、混合二阶、完整原生更新和更新后输出分别核验。这里只授予已测输入/状态/环境内的实现一致性，不声明任务预测精度、长训练轨迹一致性、MD或所有O2表示资格。','',
        '额外独立运行三步的旧审计119415保留202个超限角色及FAIL。从第二步开始，两条轨迹的模型及optimizer状态哈希已不同，因此这些梯度比较是不同参数点的轨迹差异。没有删除该现象或将其改写为通过。补充回放覆盖两条轨迹的全部5个实际参数点，每个点均严格恢复相同完整状态，保留原分量门和relative-L2门。详见[回放合同](REPLAY-CONTRACT.zh-CN.md)。','',
        '## 失败机制及修复','',
        '旧局部检查使用的形状、余切和归约调度没有覆盖真实10504行计算。整网E/F只约束函数值和坐标一阶导数；力L2MAE的余切为残差方向`(20/N)*(F-y)/||F-y||`，小残差会放大FP32力扰动对方向的影响，随后传播至参数梯度与坐标—参数混合二阶。标量loss相同不保证其余切相同。旧50项实际为49个坐标—参数角色加1个坐标二阶角色。共享原生余切只在原因定位中使用，不进入修复训练。详细因果控制见[上一轮报告](../o2-uvw-training-cause-20261125-r1/REPORT.zh-CN.md)。','',
        '本轮捕获native/v1各28处Cin后真实输入、原始K及求力/完整loss两阶段伴随，保留stride和Parameter身份；每处真实flatten行数10504。CPU119340检查捕获未引入门外扰动。通过独立于参数/标签的FP32消去与FMA探针，识别当前A100原生参考的分段累加顺序，再用Triton的独立FP32 dot累加链和显式`add.rn.f32`左结合合并复现：','',
        '|计算布局与K/N|分段长度|','|---|---|',
        '|前向1024/256|512,512|','|前向768/192|192,192,192,192|',
        '|输入伴随512/128|86,86,86,86,86,82|','|输入伴随384/96|128,128,128|','',
        '其余形状保留原IEEE FP32核。所有地址索引在乘法前转int64。递归自定义伴随`dA=gB^T,dB=A^Tg`保持完整可微二阶，原始K不拆Parameter、不detach、不缓存失效副本。未改图、标签、loss、clip、optimizer、EMA或scheduler，未启用TF32/AMP/半精度/CUDA Graph。能量输出及归约FP64，其余模型计算/累加/导数/误差统计FP32。这些分段规则针对当前固定A100/软件环境，其他形状和设备须独立验收。','',
        '## 数值验收','',
        '采用用户确认的原阈值：局部完整角色`epsilon_max<=1e-5`；整网每个完整对象同时满足分量`|a-r|<=2e-5+2e-5|r|`和relative-L2≤1e-5，拒绝非有限数，近零规则不变。ULP只报告，不设新增门。所有验收/哈希统计在AMD计算节点执行。','',
        '- 实际输入回归：2套捕获×28处×2阶段=112组，每组6角色，672个角色全部通过；最大epsilon_max=4.6677496357e-6，原生自重复通过。',
        '- 通用回归：9组/62角色，覆盖独立UVW通道、canonical/full parity、非连续布局与活权重更新，全部通过。',
        f'- 整网：600原子/31512边E/F；100原子/5252边完整训练。每个状态保留{count}个原始Parameter张量、全部活动/空导数角色及原生optimizer/EMA/scheduler。原生及候选自重复均按原门检查。','',
        '|对象（5状态合计，EF单列）|完整角色数|失败数|最大relative-L2|最大绝对差|','|---|---:|---:|---:|---:|']
    for group,g in groups.items():lines.append(f'|{group}|{g["roles"]}|{g["failed"]}|{g["maximum_relative_l2"]:.9g}|{g["maximum_absolute"]:.9g}|')
    lines+=['',f'[逐点/逐角色、原生自重复及ULP]({replay})；[归类及完整ULP统计](NUMERICAL-SUMMARY.json)。ULP在接近零或跨零分量可很大，需与绝对差及原归一化误差同时阅读，不代表全部元素≤1ULP。','',
        '|真实局部角色|检查数|非零最大差的角色数|最大ULP|','|---|---:|---:|---:|']
    for name,r in numerical['actual_local_groups'].items():lines.append(f'|{name}|{r["roles"]}|{r["nonzero_max_abs_roles"]}|{r["max_ulp"]}|')
    example=read(ROOT/'ULP-EXAMPLE.json')
    lines += ['',f'最大局部ULP的实际例子：`{example["site"]}`的`{example["role"]}`分量，candidate={example["candidate_fp32"]:.9g}、reference={example["reference_fp32"]:.9g}，跨零绝对差仅{example["absolute_difference_fp32"]:.9g}，ULP距离为{example["ulp_distance"]}；该完整角色epsilon_max={example["whole_role"]["epsilon_max"]:.9g}。这说明通过原全对象门不等于逐分量≤1ULP。[原始位置、尺度及完整统计](ULP-EXAMPLE.json)。']
    training_points=[v for k,v in numerical['same_state_groups'].items() if k!='EF']
    if all(v['training/output']['maximum_absolute']==0 and v['force_mixed/force']['maximum_absolute']==0 for v in training_points):
        lines += ['', '本次5个相同训练状态下，更新前的energy/force/stress输出与真实force-only损失所用force均无数值差异，因此实际训练采用各自loss计算时已经消除了原先的力余切扰动；全部原始参数梯度和二阶仍逐角色验收，并保留门内FP32差异。600原子E/F形状的输出有门内差异，未声称所有形状逐位一致。']
    lines+=['','## 同条件耗时与显存','',
        '同一个物理A100、单节点单GPU串行。每个后端/模式3个新进程，各3步预热、10个同步墙钟样本，表中为30样本中位数；轮次倒序平衡顺序。包含图构建/传输、求力和完整原生训练更新，公共状态恢复在计时外。显存和trace分别新进程采集；显存表为allocated峰值，reserved另列。配对完整state/parameter/graph/精度/物理GPU条件均经AMD核对。','',
        '|模式|Torch s|v1 s（训练FAIL，仅诊断）|修复版 s|Torch/修复版|Torch/v1/修复版 allocated GiB|修复版 reserved GiB|','|---|---:|---:|---:|---:|---|---:|']
    for r in perf['comparison']:
        memory='/'.join(f'{r[k]:.6f}' for k in ('native_allocated_GiB','v1_allocated_GiB','candidate_allocated_GiB'))
        lines.append(f'|{r["mode"]}|{r["native_s"]:.6f}|{r["v1_s"]:.6f}|{r["candidate_s"]:.6f}|{r["native_over_candidate"]:.4f}x|{memory}|{r["candidate_reserved_GiB"]:.6f}|')
    lines+=['',f'[全部30样本、分进程中位数/IQR/范围及原始峰值]({audit})。速度与显存分别判断，未达到目标倍率即不授予优化目标资格；v1完整训练仍FAIL，其耗时只作诊断。单次峰值的微小字节差不当作容量收益，未进行容量扫描。','',
        '性能预检最初保留了一组梯度张量引用，可能抬高训练驻留量。119444因此主动取消，已采样和实际283秒计入台账。119449重新封存的脚本在预热前释放预检输出、视图和梯度列表引用；本报告只采用随后119450的完整测量，不混用取消轮样本。','',
        '## 热点分布','',
        '每个trace为相同公共状态的一次完整计算。百分比分母为全部实际GPU kernel事件耗时之和，可跨流重叠，不能解释为墙钟占比。分类沿用冻结规则，新增`_uvw_mm`归入ICTC Triton；未分类、传输、CPU区间和全部kernel清单均保留。','',
        '|模式/后端|全部kernel ms|kernel调用数|ICTC Triton ms|ICTC占比|','|---|---:|---:|---:|---:|']
    for r in perf['traces']:lines.append(f'|{r["mode"]}/{r["backend"]}|{r["kernel_ms"]:.3f}|{r["kernel_calls"]}|{r["ICTC_ms"]:.3f}|{r["ICTC_percent"]:.2f}%|')
    lines += ['',f'[完整kernel/阶段/事件归因]({audit/"traces"})。热点仅代表当前模型/权重/输入/后端快照，不外推为架构或O2方法固有成本。',
        '',f'![同状态V3的全部计时样本、峰值allocated/reserved及kernel归一化分布]({ROOT/"figures/UVW-FP32-REPAIR-20261126-r1.png"})',
        '',f'[图注与证据边界]({ROOT/"figures/CAPTION.txt"})；[PDF]({ROOT/"figures/UVW-FP32-REPAIR-20261126-r1.pdf"})。此图用于内部结果审阅，未按投稿规格验证。','',
        '## 复现与封存','',
        '控制文件与各轮快照位于本run；实际接受源码为`snapshots/qualification-119423/ictc-source/`，性能源码见`PERFORMANCE-PREFLIGHT.json`。原始失败筛选、独立轨迹FAIL和所有原始数组保留。最终来源/依赖/内容哈希、独立解包校验和实际GPUh见`FINAL-SNAPSHOT.json`及`GPU-ACCOUNTING.json`。不因新候选通过而改写任何历史报告。接受源码的sm80离线PTX指令审计见`PTX-AUDIT.txt`，区分编译证据与GPU执行证据。',
        '', '应用技能：[/karpathy-guidelines](/public/home/sps-xia/bjxiao/.codex/skills/karpathy-guidelines/SKILL.md)，采用最小源码修改和明确的完整对象验收；[scientific-visualization](/public/home/sps-xia/bjxiao/.agents/skills/scientific-visualization/SKILL.md)，分开墙钟/显存/kernel统计并保留原始样本。踩坑复核见[PITFALL-REVIEW](PITFALL-REVIEW.zh-CN.md)。']
    (ROOT/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
    print(ROOT/'REPORT.zh-CN.md',flush=True)
if __name__=='__main__':main()
