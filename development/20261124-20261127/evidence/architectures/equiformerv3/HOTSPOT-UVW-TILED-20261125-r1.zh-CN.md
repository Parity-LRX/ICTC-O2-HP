# V3 canonical：UVW分块候选的同条件诊断

2026-11-25。候选`df12a3a85ec3b66c9e4909dca0c90a86a12b1516`，独立本地fork位于[ICTC-O2-HP](/public/home/sps-xia/bjxiao/o2-benchmark/ICTC-O2-HP)，分支`codex/uvw-triton-v1`，未推送。旧[固定热点快照](HOTSPOT-ATTRIBUTION-20261125-r1.zh-CN.md)保持原样。

## 实际替换和条件

保持独立canonical投影EMA、30,333,409参数、7块C128/L4/M2、14个SO2Linear实例内的28个正频率块。原生分母为同一canonical变体，不能与未投影上游checkpoint互换。

新增[IEEE FP32分块核](/public/home/sps-xia/bjxiao/o2-benchmark/ICTC-O2-HP/src/ictc_o2_hp/uvw_triton.py:12)及[共享权重UVW接口](/public/home/sps-xia/bjxiao/o2-benchmark/ICTC-O2-HP/src/ictc_o2_hp/uvw_o2.py)。通用UVW保留独立U/V/W，角装配仍为Torch；V3使用U=1固定0e标量特例，原K直接执行矩阵乘，不创建虚拟1/0左输入。两侧反射换基、原Parameter、E00、m0、球激活/FFN、聚合、残差/norm和能量头保持既有实现。

前向、输入/权重伴随及需要的混合二阶矩阵乘全部经同一个可微Triton原语，无原生二阶重入。GPU FP32、IEEE dot，能量输出/归约FP64，TF32/AMP/半精度/CUDA Graph关闭。

XHPC-A100单节点单GPU119167，实际A100-SXM4-40GB，Torch2.7.1+cu128/Triton3.3.1、4线程。每模式三后端保持同参数、公共状态、完整model/optimizer/EMA/scheduler、图/标签/输入和GPU UUID，18份采集条件完整匹配。原生与候选使用相同7块非重入重算及bounded球乘法；3预热/10同状态同步墙钟，显存和trace分进程，恢复在计时外，调用内图/传输/布局计时。

## 实测结果

|模式|原生/冻结HP/候选墙钟 s|冻结HP/候选|原生/冻结HP/候选allocated GiB|HP kernel总耗时：冻结→候选 ms|占全部kernel和：冻结→候选|
|---|---|---:|---|---|---|
|E/F，600原子/31512边|3.611216/23.335713/3.787937|6.161×|7.585464/7.904446/7.585517|19913.989→320.386|91.55%→14.70%|
|完整训练，100原子/5252边|1.789743/6.637043/1.843804|3.600×|11.343170/12.275710/11.343224|5256.708→115.788|85.34%→12.15%|

峰值显存基本回到原生水平，候选墙钟仍略高于原生。百分比分母为每份trace所有实际GPU kernel时长之和，不是墙钟占比。训练高阶从Torch重入改为Triton，分类覆盖随实现变化，整体墙钟和独立显存共同判断收益。

候选E/F剩余类别为GEMM/GEMV40.89%、copy/layout17.90%、ICTC Triton14.70%；训练为GEMM/GEMV41.41%、elementwise16.88%、copy/layout13.43%、ICTC Triton12.15%。宽V通用归约已被分块矩阵计算替代，原生剩余矩阵乘与布局成本显现；分类名称不等于数学模块，不能把全部copy归给adapter或由此断言带宽/寄存器溢出瓶颈。

## 数值边界

局部9组/62角色的前向/full VJP/混合诊断通过，FP32最大epsilon_max=1.3703051e-6，实际K更新可见。通用UVW只测M≤2、U/V/W=3/5/4及两种归一化，V3另覆盖四类实际K与非连续布局；不扩为所有通道/路径资格。新表示及局部高阶正式门仍HOLD。

整网E/F输出和原生自重复通过。**完整训练候选仍FAIL：4个参数梯度、50个实际力损失混合二阶角色超限；本轮冻结HP也有2个参数梯度、19个混合角色失败。** 所有自重复通过。候选坐标混合对象有3个分量超限，max_abs=8.5264444e-5，不能以其relative-L2通过覆盖分量失败。该候选不取得合格训练加速或完整训练数值资格。

局部原回执最终归一化曾使用Python标量除法，已保留原件、精确恢复原分母并以FP32复算，判断均不变；整网统计原本为FP32。没有修改容差、输入、权重或旧FAIL/HOLD，不以FP64解决FP32门。

![同状态三后端热点、墙钟和显存](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/figures/UVW-V3-COMPARISON-r2.png)

证据：[完整报告](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/REPORT.zh-CN.md)、[原始结果](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/results/a100-119167)、[配对复核](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/results/analysis-119186/PAIRED-AUDIT.json)、[逐角色数值](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/results/analysis-119186/GATES.json)、[恢复封存](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/FINAL-SNAPSHOT.json)、[GPU记账](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/GPU-ACCOUNTING.json)。本轮结束于该候选诊断，没有启动下一候选、正式验收、重训或MD。
