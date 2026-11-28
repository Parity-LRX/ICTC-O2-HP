"""Report only sealed-source numerical and performance receipts."""
import json,os
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);WS=ROOT.parents[1];assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
def read(p):return json.loads(Path(p).read_text())
def receipt(name):return read(Path((ROOT/(name+'.txt')).read_text().strip())/'RESULT.json')
n=read(ROOT/'NUMERICAL-SUMMARY.json');p=receipt('PERFORMANCE-AUDIT');local=receipt('LOCAL-AUDIT');fusion=read(ROOT/'BILINEAR-TRACE-AUDIT.json')
assert p['status']=='PERFORMANCE_AUDIT_PASS' and p['operator_source_sha256']==n['operator_source_sha256']
assert fusion['status']=='BILINEAR_FUSION_TRACE_PASS'
commit=(ROOT/'CANDIDATE-HEAD.txt').read_text().strip();comp={r['mode']:r for r in p['comparison'] if r['backend']=='candidate'}
lines=['# bounded双线性、完整VJP与混合二阶融合结果','',
 f'最终候选 `{commit}`，嵌套仓库 `{WS/"ICTC-O2-HP"}`，分支 `codex/bounded-bilinear-v5`，未推送。operator SHA256 `{n["operator_source_sha256"]}`；完整源码/二进制包SHA256 `{n["package_sha256"]}`。',
 '', f'本轮E/F {comp["ef"]["median_s"]:.6f}s、完整训练{comp["train"]["median_s"]:.6f}s；对同轮8324耗时分别{comp["ef"]["time_change_vs_baseline_percent"]:+.2f}%/{comp["train"]["time_change_vs_baseline_percent"]:+.2f}%，对Torch分别{comp["ef"]["time_change_vs_native_percent"]:+.2f}%/{comp["train"]["time_change_vs_native_percent"]:+.2f}%。速度与显存分别验收，下表保留绝对值。',
 '', '## 保持的数学与训练边界','',
 '原canonical独立模型、292原始活动Parameter、28处8324通道实现、7块非重入checkpoint、所有后续计算、图/PBC/重复边和三项真实loss保持。clip/AdamW/EMA/scheduler未改；真实连续三步内不恢复模型/optimizer/RNG，双方各用自己的真实loss余切。旧8324、c05及全部历史FAIL/HOLD不改，不把三步通过外推为长期轨迹一致。',
 '', '能量输出/归约FP64，其他运算、乘积、累加、导数与误差统计FP32；禁TF32/AMP/half/CUDA Graph。完整每角色逐分量abs(a-r)<=2e-5+2e-5*abs(r)，且relative-L2<=1e-5；零参考仅零差。局部完整角色epsilon_max<=1e-5，尺度max(双方maxabs,1e-7)。先拒绝NaN/Inf，ULP描述不增加新门。',
 '', '## 融合实现与FP32修复','',
 '原bounded每64行执行一次三输入einsum，将块输出留在Python列表并cat。新核一次分配完整输出，完整区和尾区最多两次Triton网格发射；块内生成并消费a*b，不把完整角向外积写入全局内存。64只保留为复现原生FP32归约计划的逻辑边界，没有64行Python循环、块输出列表或最终拼接。其他网络cat/bmm仍有各自语义，未宣称全部消除。',
 '', '原固定canonical_kernel系数缓冲不训练的状态保持；所有292模型权重仍活动。前向保持FP32 mul.rn(a,b)的中间舍入，再按原生FMA链与add.rn合并。两输入VJP继续使用原角向轴置换的同一双线性算子，递归对输入与余切保持可微；没有detach活动权重、once_differentiable或原生高阶计算回退。',
 '', '首版b6ed4e6局部90组/540角色通过，但实际独立三步73/30/55角色FAIL；保留V5A-*指针。9b7ba77补齐行布局的完整区/尾区计划后，实际三步0/171/59 FAIL，V5D-*保留。后者局部前向/VJP已逐位一致，但部分二阶仍不同，继而在独立更新中放大。不能用局部PASS或共享余切覆盖这些FAIL。',
 '', '最终修复依据实际原生einsum/BMM布局和FP32抵消探针：完整递归二阶会触发全部6种系数轴置换，其中列主序coefficient matrix采用不同归约树。全部12种行/列布局×完整/训练尾/EF尾的叶子和合并顺序经AMD核对，再按实际系数stride选择计划。合成标签只用于数值诊断，真实模型系数/输入/损失未换。编译失败快照0e9e3ec、离线f86d810以及所有阶段代码/日志均保留。',
 '', '| 角向维 | 系数矩阵 | 区域Q | 分区P | 每分区FMA | 跨分区 |','|---|---|---:|---:|---|---|']
for r in read(ROOT/'PLAN-VALIDATION.json')['rows']:
    lines.append(f'| {r["L"]} | {"列" if r["coefficient_column_major"] else "行"} | {r["Q"]} | {r["parts"]} | {"四条8项交错链，平衡合并" if r["four_chains_of_eight"] else "连续单链"} | 按P升序左结合 |')
lines += ['', '## AMD原门与原独立三步','',f'完整AMD回执 `{n["AMD_audit"]}`；局部{local["cases"]}组/{local["roles"]}角色通过。真实捕获{local["groups"]["actual"]["cases"]}组/{local["groups"]["actual"]["roles"]}角色最大ULP={local["groups"]["actual"]["max_ulp"]}；通用/边界最大epsilon={local["groups"]["generic"]["epsilon_max"]:.9g}，其非零ULP保留。',
 '', '| 协议 | 步 | 全记录FAIL | 最大R | 最大ULP | 原始梯度角色 | force-only混合角色 |','|---|---:|---:|---:|---:|---:|---:|']
for r in n['rows']:lines.append(f'| {r["mode"]} | {r["step"]} | {r["all"]["failures"]} | {r["all"]["max_R"]:.9g} | {r["all"]["max_ulp"]} | {r["groups"]["raw_gradients"]["roles"]} | {r["groups"]["force_mixed"]["roles"]} |')
lines += ['', f'E/F完整输出最大R={n["E_F"]["max_R"]:.9g}、最大ULP={n["E_F"]["max_ulp"]}；联合完整VJP最大R={n["joint_VJP"]["max_R"]:.9g}、最大ULP={n["joint_VJP"]["max_ulp"]}。两项保留完整分量门与全对象门，不按近零ULP单独改判。',
 '', 'legacy原生数组重现119414原协议。isolated只隔离诊断RNG，train-only不插入诊断；各自真实训练状态与RNG连续推进。记录44份RNG文件、每次实际旋转矩阵和24个跨步状态/RNG转移。参数梯度、28K、实际混合、完整optimizer/EMA/scheduler及更新后输出均验；原source8324组件逐文件哈希保持。',
 '', '## 同一A100单GPU串行性能','',
 '最终完整AMD通过后才采集native Torch、8324 baseline、新候选。每模式/后端3个新进程，各3预热/5同步调用，全15样本保留。恢复公共起始状态位于计时外，仅定义同条件测量，不替代真实独立三步。图/传输、求力与完整原生更新包含在计时内；memory/trace分别独立进程。固定E/F600原子31512边、训练100原子5252边；输入、权重、RNG、原loss/optimizer、精度和设备UUID配对。',
 '', '| 模式 | 后端 | 中位s | allocated GiB | reserved GiB | ICTC GPU kernel占比 |','|---|---|---:|---:|---:|---:|']
for r in p['workloads']:lines.append(f'| {r["mode"]} | {r["backend"]} | {r["median_s"]:.6f} | {r["allocated_GiB"]:.6f} | {r["reserved_GiB"]:.6f} | {r["ICTC_percent"]:.2f}% |')
lines += ['', '| 模式 | 候选对8324耗时变化 | 对Torch耗时变化 | allocated对8324字节 | reserved对8324字节 |','|---|---:|---:|---:|---:|']
for r in comp.values():lines.append(f'| {r["mode"]} | {r["time_change_vs_baseline_percent"]:+.2f}% | {r["time_change_vs_native_percent"]:+.2f}% | {r["allocated_change_vs_baseline_bytes"]:+d} | {r["reserved_change_vs_baseline_bytes"]:+d} |')
lines += ['', '本轮两模式allocated峰值均未下降，训练allocated略增；reserved的变化只描述此次分配器保留量，不授予容量改善。',
 '', '| 模式 | 8324 kernel次数 | 候选kernel次数 | 8324 kernel合计ms | 候选kernel合计ms |',
 '|---|---:|---:|---:|---:|']
for mode in ('ef','train'):
    by={r['backend']:r for r in p['workloads'] if r['mode']==mode};b=by['baseline'];c=by['candidate']
    lines.append(f'| {mode} | {b["kernel_calls"]} | {c["kernel_calls"]} | {b["kernel_ms"]:.3f} | {c["kernel_ms"]:.3f} |')
lines += ['', 'kernel发射次数大幅减少，kernel合计耗时的降幅小于独立同步计时中的墙钟降幅。这与消除Python分块、细碎临时算子及发射开销的预期一致；此处是结合两类计量作出的归因，不将不同采集边界的差值直接当作CPU耗时。']
lines += ['', 'kernel份额按每份trace全部实际cat=kernel耗时和归一化，不是墙钟占比。此轮ICTC GPU类别包含原8324通道核及新增双线性核，覆盖范围比旧图扩大；不能只比较该类别百分比判断快慢。完整类别、阶段、kernel、算子、原始时序与条件均在PERFORMANCE-AUDIT目录。',
 '', '| 模式 | 后端 | 原角向BMM次数 | 新双线性核次数 | 新双线性kernel总ms |','|---|---|---:|---:|---:|']
for r in fusion['rows']:lines.append(f'| {r["mode"]} | {r["backend"]} | {r["old_angular_BMM_calls"]} | {r["fused_kernel_calls"]} | {r["fused_kernel_ms"]:.3f} |')
lines += ['', f'![整网耗时、显存与完整kernel分布]({ROOT/"figures/BOUNDED-BILINEAR-20261126-r1.png"})',
 '', '## 恢复与适用边界','',
 '当前结论限于固定canonical模型、已登记19/25角向布局及这两组整网输入/训练协议；非测试集任务MAE、长期训练轨迹或全部硬件/形状证明。通用输入仍是FP32同式执行，但没有给未验布局/批量继承真实三步资格。独立算子、整网速度与显存/容量目标分别判断，不因数值通过自动授予2×/5×/5×性能目标。',
 '', '8324b33原归档不改，最终封存另行核验其SHA。每阶段Git tag、源码、配置、环境与回执在PROGRESS.zh-CN.md及snapshots/。FINAL-SNAPSHOT.json指向可恢复Git/源码/计划/回执归档，经过独立解包、文件集合和逐文件哈希核对。大数组与原始trace仍在哈希引用的工作区路径，RAW-RESULT-MANIFEST明确列出，不声称小归档包含全部大数组。GPU-ACCOUNTING逐JobID记入实际分配，包含失败与取消，不重复记旧run。']
(ROOT/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n');print(ROOT/'REPORT.zh-CN.md',flush=True)
