"""Render only completed AMD receipts; no guessed or substituted measurements."""
import json,os
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);WS=ROOT.parents[1];assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
def read(p):return json.loads(p.read_text())
n=read(ROOT/'NUMERICAL-SUMMARY.json');audit=Path((ROOT/'PERFORMANCE-AUDIT.txt').read_text().strip());p=read(audit/'RESULT.json')
assert p['status']=='PERFORMANCE_AUDIT_PASS' and p['operator_source_sha256']==n['operator_source_sha256']
commit=(ROOT/'CANDIDATE-HEAD.txt').read_text().strip();rows=read(audit/'WORKLOADS.json')
lines=['# 28K权重伴随、RNG隔离与GEMM修复验收','',
    f'候选 `{commit}`，分支 `codex/uvw-kgrad-rng-v3`，tag `snapshot/kgrad-recursive-layout-v3c-20261126`，嵌套仓库 `{WS/"ICTC-O2-HP"}`，未推送。operator SHA256 `{n["operator_source_sha256"]}`。',
    '', '本轮使用各自真实损失与独立原生clip/AdamW/EMA/scheduler推进三步，原legacy和RNG隔离协议全部通过原门。没有共享loss余切、梯度或中途恢复参数/优化器状态。旧a80的202项FAIL与df12的4+50 FAIL仍保留；本轮通过不改写历史，也不外推为长期轨迹一致。',
    '', '## 实现和逐步快照','',
    'P：120044冻结，A100120046/AMD120049验证RNG隔离。CPU/CUDA/NumPy/Python RNG与每次实际随机旋转矩阵均登记。mixed/更新后输出诊断保存恢复RNG，真实训练保持自己的RNG连续推进；隔离后与train-only三步控制完全一致。legacy保持原来的mixed→train→after相位，精确复现旧原生数组。',
    '', 'K布局：029afec把权重伴随从`(X.T @ G).T`改为原权重方向`G.T @ X`，保留原始Parameter和递归autograd。A100120048/AMD120055表明仅改布局尚有门内误差，因此未当作修复。',
    '', 'K长归约/GEMM：A100120056识别实际10504项FP32归约树，AMD120058/120061验证全部叶子和规则计划一致。bfb3987把各K分区分配到并行CTA，四条FMA链在CTA内计算，临时partial经独立核按固定顺序相加，无atomic、无FP64累加；partials最多4MiB、不进入保存的反向图。32×32输出tile保持原算术依赖，split-K增加长归约的并行度。原来的前向/输入伴随分段不改。',
    '', '| dK形状[W,V] | K分区数 | 分区跨度ceil(10504/P) |', '|---|---:|---:|',
    '| 256×1024 | 4 | 2626 |','| 192×768 | 3 | 3502 |','| 512×128 | 5 | 2101 |','| 384×96 | 9 | 1168 |',
    '', '每个分区的局部索引q按`floor((q mod 32)/8)`分入四条链，各链按K升序FMA；分区内`(c0+c1)+(c2+c3)`，分区间从0开始左结合相加。尾部严格mask，索引在乘法前提升int64。仅在实际10504长K和四种已登记布局触发专核，其余形状保留原IEEE FP32通用核，不宣称新批量/硬件也与cuBLAS逐位相同。',
    '', '递归布局修复：bfb虽然局部一阶dK已一致，实际完整训练第三步仍超限（AMD120084最大参数梯度R=1.10263545e-5）。K与K.T在二阶中分别出现，统一翻转伴随方向仍会走错误的FP32布局。最终c05bc7b按两操作数的实际stride保留物理方向：转置G.T的伴随求`X @ dK.T`后转置，连续K的伴随仍求`G.T @ dX`，转置K.T则求原权重方向。A100120088/AMD120089的8组×6角色均为0误差。此失败候选与全部回执保留，未测其性能。',
    '', '完整二阶继续由同一个可微矩阵乘递归构造，前向不保存partial、不detach活动K，不使用原生cuBLAS作为候选高阶回退。AMD120092对最终源码的13份SM80编译结果确认FP32 FMA/显式add.rn，无TF32/半精度/Tensor Core指令。',
    '', '## 原门、ULP和独立三步','',
    '能量输出/求和FP64，其余运算、累加、导数和误差统计FP32；禁TF32/AMP/CUDA Graph。整网完整角色逐分量`|a-r|≤2e-5+2e-5|r|`且relative-L2≤1e-5，零参考仅零差；局部O2完整角色epsilon_max≤1e-5。NaN/Inf拒绝。ULP为描述统计，不增加逐位或1ULP门。',
    '', f'完整AMD回执：`{n["AMD_audit"]}`。实际112组×6角色和通用9组/62角色均通过，含原生自重复；局部最大epsilon分别 {n["local"]["REGRESSION"]["epsilon_max"]:.9g}、{n["local"]["GENERIC"]["epsilon_max"]:.9g}，112组实际K伴随最大ULP为{n["local"]["REGRESSION"]["weight_ulp_max"]}。',
    '', '| 协议 | 步 | 原始参数梯度 | 其中28K最大ULP | force-only混合 | 更新参数最大ULP | 全记录FAIL/最大R/最大ULP |', '|---|---:|---:|---:|---:|---:|---|']
by_mode={r['mode']:r for r in p['comparison']}
lines[5:5]=['',f'性能结论：相对本轮Torch，新候选E/F与训练中位耗时分别增加{by_mode["ef"]["candidate_time_change_vs_native_percent"]:.2f}%、{by_mode["train"]["candidate_time_change_vs_native_percent"]:.2f}%；相对冻结a80分别变化{by_mode["ef"]["candidate_time_change_vs_v2_percent"]:+.2f}%、{by_mode["train"]["candidate_time_change_vs_v2_percent"]:+.2f}%，allocated/reserved基本持平。本轮完成数值修复，未取得整网速度或显存收益；GEMM不同形状的得失需分开看。']
for r in n['rows']:
    g=r['groups'];all_=r['all']
    lines.append(f'| {r["mode"]} | {r["step"]} | {g["raw_gradients"]["roles"]}角色/0FAIL | {g["raw_K"]["max_ulp"]} | {g["force_mixed"]["roles"]}角色/0FAIL | {g["updated_parameters"]["max_ulp"]} | {all_["failures"]}/{all_["max_R"]:.9g}/{all_["max_ulp"]} |')
lines += ['',f'E/F完整输出最大R={n["E_F"]["max_R"]:.9g}、最大ULP={n["E_F"]["max_ulp"]}；联合完整VJP最大R={n["joint_VJP"]["max_R"]:.9g}、最大ULP={n["joint_VJP"]["max_ulp"]}。native/native-repeat/candidate/candidate-repeat全数组均验，所有活动与None角色、292参数更新、optimizer/EMA/scheduler和更新后输出均纳入。每相位RNG及旋转矩阵一致；中间只保存状态用于证据，不恢复轨迹。',
    '', '## 同条件耗时与显存','',
    '下列计量在新候选AMD完整通过后执行。固定同一A100-SXM4-40GB、同完整起始状态/参数/输入图/标签/RNG，单GPU串行；每模式/后端3个新进程，各3次预热、5次同步调用，列15次调用中位数。恢复状态在计时外，计时包括图/传输、求力及训练完整原生更新。memory/trace各为独立新进程。a80保留原独立三步FAIL，只作成本诊断对照。',
    '', '| 模式 | 后端 | 中位耗时s | allocated GiB | reserved GiB |', '|---|---|---:|---:|---:|']
for r in rows:lines.append(f'| {r["mode"]} | {r["backend"]} | {r["median_s"]:.6f} | {r["allocated_GiB"]:.6f} | {r["reserved_GiB"]:.6f} |')
lines+=['','| 模式 | 新候选对Torch耗时变化 | 对a80耗时变化 | 新候选HP kernel占比 |','|---|---:|---:|---:|']
for c in p['comparison']:
    r=next(r for r in rows if r['mode']==c['mode'] and r['backend']=='candidate')
    lines.append(f'| {c["mode"]} | {c["candidate_time_change_vs_native_percent"]:+.2f}% | {c["candidate_time_change_vs_v2_percent"]:+.2f}% | {r["ICTC_percent"]:.2f}% |')
gemm=read(audit/'GEMM.json')
lines += ['', 'kernel占比按每份trace全部实际GPU kernel耗时之和归一化，不是墙钟比例；完整kernel、phase、stage、operator和分类库存见AMD性能目录，原始15次调用全部保留，不筛异常值。速度、allocated与reserved分别判断，数值通过不授予2×/5×/5×性能或容量资格。',
    '', '## 权重GEMM单列诊断','',
    '使用四种真实形状及两种真实阶段的X/G，已展平并驻留，分别3次预热和30次同步墙钟调用。此表仅含dK矩阵乘，排除展平、完整求导/loss/optimizer和图传输，不能代替整网时间。冻结a80按真实生产方向`(X.T @ G).T`，新候选按`G.T @ X`及固定顺序split-K。原始调用、每种实现局部allocated增量和AMD输出复核见`GEMM.json`。',
    '', '| dK形状 | 阶段 | Torch ms | a80 ms | 新候选 ms | a80/新候选 | 新候选K最大ULP |', '|---|---|---:|---:|---:|---:|---:|']
for r in gemm['rows']:
    t=r['timings'];lines.append(f'| {r["shape"]} | {r["phase"]} | {1000*t["native"]["median_s"]:.4f} | {1000*t["v2"]["median_s"]:.4f} | {1000*t["candidate"]["median_s"]:.4f} | {r["v2_over_candidate"]:.3f}× | {r["candidate_metric"]["ulp"]["max_ulp"]} |')
shape_path=Path((ROOT/'TRACE-SHAPES-LATEST.txt').read_text().strip());shapes=read(shape_path);assert shapes['trace_source']==p['source']
lines += ['', '当前trace按实际输入形状识别的长K权重GEMM如下。每个native mm/bmm或候选_Matmul只计一次；这是调用次数，不是该角色的独立GPU计时。完整形状库存保留在`'+shape_path.name+'`。',
    '', '| 模式 | 后端 | 长K | 权重GEMM次数 |','|---|---|---:|---:|']
for r in shapes['weight_GEMM']:lines.append(f'| {r["mode"]} | {r["backend"]} | {r["long_K"]} | {r["weight_GEMM_calls"]} |')
by={(r['mode'],r['backend']):r['weight_GEMM_calls'] for r in shapes['weight_GEMM']}
if by['ef','native']==0 and by['ef','candidate']>0:
    lines += ['', f'E/F只请求几何导数，当前候选却仍生成{by["ef","candidate"]}次K=63024的权重伴随；原生同形状计数为0。源码中的ctx.needs_input_grad依据活动输入，当前Python自定义反向未按本次求导目标剪除这一路。此次10504专核未扩大到63024，因此不能期待它消除这部分推理成本。下一项GEMM方法应优先让执行层按求导目标只生成所需伴随，同时保留训练/完整VJP/混合二阶的参数路径；不能通过冻结权重或删除高阶取得训练资格。']
lines += ['', 'GEMM后续方案（未实施，不作已通过的速度结果）：',
    '', '1. 按本次求导目标剪枝未消费的K伴随输出。训练中也有84对56的差异；剪枝时必须保留输入伴随对K的依赖，继续通过真实力损失混合二阶及原三步，不通过detach或冻结原始权重取巧。',
    '2. 验证显式8项SIMT FMA外积微核，保留四链和最终合并树，移除当前每条8有效项链在K16 dot中的填零槽。当前安装Triton 3.3.1的NVIDIA后端将FP32 dot最小K限定为16，不能直接改成tl.dot K8。先比较PTX/实际指令与完整数值，再判断速度，不能仅凭源码认定全部退化的原因。',
    '3. 对256×1024等退化形状独立评估32×64、64×32、64×64输出tile与warp组合，保持K链/合并顺序并复验真实三步；不再对全部形状统一假定32×32 split-K更快。',
    '', '上述方法以本轮测量和调用链为依据。当前正式数值候选仍为c05bc7b，未用未验候选覆盖它。']
lines += [
    '', f'![完整耗时、显存与kernel分布]({ROOT/"figures/UVW-KGRAD-RNG-20261126-r1.png"})',
    '', '## 适用边界与恢复','',
    '当前是固定canonical投影EMA变体、E/F600原子31512边、训练100原子5252边的实现/三步更新验收，非公开测试集任务MAE、长轨迹或全部硬件/批量的证明。28处替换边界、原292参数、能量求导力、三项实际损失和7块重算保持。原生随机局部坐标系的RNG仍真实推进，没有选择新种子使结果过门。',
    '', '后续GEMM可在已固定的每条FMA链与合并树不变的条件下调M/N tile、warp和访存复用，再独立验收。更改K归约树、直接采用任意split-K atomic、TF32或半精度不能沿用本次数值资格。整网更大的收益需依据新热点逐项验收，不能把局部并行设计当成墙钟收益。',
    '', '逐步源码与协议在`PROGRESS.zh-CN.md`和`snapshots/`；最终源码/Git/回执归档与独立解包QA入口`FINAL-SNAPSHOT.json`，大数组以`RAW-RESULT-MANIFEST.json`引用原路径，不声称小归档包含所有大数组。GPU-ACCOUNTING.json包含失败启动和实际分配消耗，计入500GPUh总账，不重复记旧作业。']
(ROOT/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
print(str(ROOT/'REPORT.zh-CN.md'),flush=True)
