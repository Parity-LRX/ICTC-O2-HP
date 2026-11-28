# UVW候选：V3 canonical配对热点诊断

**诊断结论：相对冻结HP，E/F和完整训练耗时分别下降到1/6.16和1/3.60，峰值allocated基本回到原生水平。完整训练仍有4个参数梯度及50个力损失混合二阶角色FAIL，不授予训练数值或正式加速资格。**

本轮固定相同投影EMA、参数、输入、完整训练配方及7块重算；单A100串行测量。速度为同步墙钟10样本中位数，显存为独立峰值allocated。HP百分比分母是trace全部实际GPU kernel耗时和。

|模式|Torch s|冻结HP s|新候选 s|对冻结HP倍率|对Torch倍率|Torch/旧HP/新候选 GiB|新候选HP kernel占比|
|---|---:|---:|---:|---:|---:|---|---:|
|ef|3.611216|23.335713|3.787937|6.161x|0.953x|7.585464/7.904446/7.585517|14.70%|
|train|1.789743|6.637043|1.843804|3.600x|0.971x|11.343170/12.275710/11.343224|12.15%|

## 数值及条件

两模式三后端的参数、公共起始状态、完整model/optimizer/EMA/scheduler状态、图/标签、物理GPU UUID、4线程及精度条件一致。TF32、AMP、CUDA Graph均关闭。逐字段的true表示两侧相同，不表示启用了相应加速选项。详见`results/analysis-119186/PAIRED-AUDIT.json`。

- ef-native: all_roles_pass=True, failed_roles=0, groups={}, self_repeat_failed=0；[逐角色指标](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/results/analysis-119186/ef-native-GATE.json)。
- ef-ictc: all_roles_pass=True, failed_roles=0, groups={}, self_repeat_failed=0；[逐角色指标](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/results/analysis-119186/ef-ictc-GATE.json)。
- ef-tiled: all_roles_pass=True, failed_roles=0, groups={}, self_repeat_failed=0；[逐角色指标](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/results/analysis-119186/ef-tiled-GATE.json)。
- train-native: all_roles_pass=True, failed_roles=0, groups={}, self_repeat_failed=0；[逐角色指标](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/results/analysis-119186/train-native-GATE.json)。
- train-ictc: all_roles_pass=False, failed_roles=21, groups={'gradients': 2, 'force_mixed': 19}, self_repeat_failed=0；[逐角色指标](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/results/analysis-119186/train-ictc-GATE.json)。
- train-tiled: all_roles_pass=False, failed_roles=54, groups={'gradients': 4, 'force_mixed': 50}, self_repeat_failed=0；[逐角色指标](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/results/analysis-119186/train-tiled-GATE.json)。

局部算子完整数组：first_order_pass=True，mixed_diagnostic_pass=True，全部已测角色最大epsilon_max=1.37030509e-06；[原生自重复及完整角色](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/results/analysis-119186/OPERATOR-VALIDATION.json)。新对象高阶正式门仍HOLD。

统计精度补正：原局部回执的最终归一化比值曾用Python标量除法。已由保存的完整数组FP32最大差及其比值精确恢复非零残差的原分母，并逐项重现原标量比值；零残差对任意合法分母均为0。最终除法以FP32张量复算；全部判断不变。[原回执](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/results/analysis-119186/ORIGINAL-OPERATOR-VALIDATION.json)保留，已修正后续验证脚本。整网分量/relative-L2统计原本为FP32，训练FAIL未改。

旧FAIL/HOLD不改判。通用UVW为共享权重三独立通道轴，角装配保留Torch；V3使用U=1固定标量特例，前向/输入与权重VJP/所需混合二阶矩阵乘均经同一IEEE FP32 Triton核。训练混合高阶从旧PyTorch重入转为Triton，kernel颜色分类覆盖随执行实现变化，比较整体墙钟及峰值才判断整网收益。新增高阶正式容差、任务精度和容量扫描不由本诊断授予资格。

[完整kernel清单、逐事件、阶段及分类](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/results/analysis-119186/traces)；[原始10个样本及allocated/reserved](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/results/a100-119167)。

候选源码提交：`df12a3a85ec3b66c9e4909dca0c90a86a12b1516`；[源码及独立解包封存回执](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/FINAL-SNAPSHOT.json)。

![同条件kernel分布、绝对耗时及显存](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/figures/UVW-V3-COMPARISON-r2.png)

[流程与软件来源](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/SOFTWARE-REFERENCES.md)中记录所用技能，并列出已核对的Scientific Agent Skills参考文献。
