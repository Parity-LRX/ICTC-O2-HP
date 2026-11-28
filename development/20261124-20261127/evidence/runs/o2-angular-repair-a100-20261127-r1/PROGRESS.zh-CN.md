# 新阶数完整修复进度

当前候选5764f666d1278634f2a555f8cf621acc1c2c0801 / snapshot/angular-plans-v6d-20261127，未推送。已登记6576条计划，实际VJP/混合/训练捕获150种形状覆盖PASS。GPU120771原精确校准的317个未逐位匹配项全部保留；120775补充搜索按原门检查，AMD120776独立完整数组/ULP审核PASS，其中部分仍有门内非零误差，不声称全域0ULP。

AMD120777冻结候选与原三步控制；A100120780串行捕获9个整网配置的native/native-repeat/candidate、联合完整VJP、真实force-only混合及三步。AMD120787提前检查L2/L4各三步0FAIL，联合VJP通过、E/F 0ULP，三步R约2--3e-7，ULP完整保留，尚不代表全矩阵验收。

L8参考第二步mixed诊断新增OOM保留在120780：上一训练步.grad和Adam状态使活动量增加。原生_backward本来会在使用前zero_grad(set_to_none=True)，模型/诊断前向均不读取旧.grad；120790冻结将这次清零提前、并释放无引用缓存的控制。无模型/optimizer/EMA/scheduler/RNG恢复，A100120791在120780之后串行重验L8全部三步。旧L8原始OOM和本次OOM均保留。

后续GPU120789依赖120791，串行采144算子配置与240局部六轴/尾区/full-VJP/混合回归；AMD120792/120793对应审核。当前尚未提交性能作业；measurement_prepare.py要求9个整网配置两模式、144算子配置和局部回归全部通过，才能冻结计量控制。计量、容量和绘图准备不等于已执行。

以下为早期过程记录。

用户2026-11-27已授权修复并在AMD三步通过后补测1..8阶算子/整网性能及容量；原loss/精度/阈值和旧FAIL保持。详见CONTRACT.zh-CN.md。

嵌套仓库分支codex/angular-fp32-repair-v6，未推送。步骤快照：

- beaa640 / snapshot/angular-grid-v6a-20261127：仅交换双线性网格轴，保留算术。
- da7b502 / snapshot/angular-plan-scaffold-v6b-20261127：增加自写FP32计划执行核，尚无新资格。
- a8b3c77 / snapshot/angular-strides-v6c-20261127：计划键包含实际系数及两输入物理stride，防止将前64行/连续布局计划外推到其他递归布局。

诊断后gc.collect/empty_cache放在完整mixed诊断与真实train_step之间，双方同策略；不恢复模型/optimizer/RNG，不改变活动张量。原真实三步调用顺序保持，joint VJP另行执行并隔离其RNG。

AMD120770已冻结当前源码、控制、Git bundle和逐文件hash；单A100120771捕获原锚点及L1..8的实际完整VJP/force-mixed/full-train矩阵形状，随后同allocation串行校准FP32计划。校准覆盖角向六种系数轴排列、64行完整区、容量相关4倍数尾区、两输入连续/分片物理stride，以及通道前向/输入伴随/权重伴随。合成数据只用于计划识别，不改变真实模型/标签/loss。

计划校准和脚手架不是三步修复资格。下一步AMD核对计划覆盖与保存新源码快照，再进行各自真实损失独立三步、完整导数/原生更新审核；通过对应门后才测时间/显存/容量。当前未启动性能计量。

启动阶段AMD120767缺少默认PATH中的git，GPU120768因输出目录已存在退出；均已保留日志，120770/120771为新冻结/重提，不覆写失败回执。
