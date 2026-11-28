# canonical O2 CPU转换、29处及后续计算验证终态

2026-11-24。仅CPU验证，无A100/GPU、训练或优化器更新；原冻结基线和旧FAIL保持。

**结论：独立canonical checkpoint已生成并严格重载；局部与后续计算分别验收，后续计算收缩顺序仍有FAIL，旋转/反射数值残差因新FP32门未审定保持HOLD，不授予完整接入资格。**

## 固定基与checkpoint

实际Wigner/m布局在本次全部(l,m)上选择C=I；这次统一基下K=A，B被排除。该结论来自新校准，不能扩展为所有约定都可以直接删raw B。

| 来源 | 原样复制Parameter张量 | 投影Parameter张量 | 独立checkpoint SHA256 |
| --- | --- | --- | --- |
| checkpoint_original | 264 | 28 | `323ab2d2f4e7d09d0c947506440e3899f70b485cbf43b86ef6f84ce529777880` |
| checkpoint_ema | 264 | 28 | `758c8b2a036e1e910f84840c590db6906d978f1d1734f443eb85c12e9d8f4113` |

14个模块×2正频率=28个投影张量；其余264个Parameter原值一致，合计292个来源张量。新模型参数为packed K，无旧B自由度；新checkpoint不是原模型函数。球面系数采用Gaunt不变量结构及原网格标量拟合，完整系数/差异保存在转换回执。

## 分层结果

序列均为前向/full VJP/混合二阶；局部按O2前向/VJP门、混合按既有高阶门，后续计算按完整block/整网门，角色不混合归一化。

- checkpoint_original：局部 29/29/29（分母29）；后续计算 27/27/27（分母29）；ALL29 PASS/PASS/PASS；真实E/F及force-parameter混合VJP整体PASS。
  原生自重复失败位置：无。新球面系数相对旧网格最大绝对变化 8.3446503e-07。
- checkpoint_ema：局部 29/29/29（分母29）；后续计算 27/27/26（分母29）；ALL29 PASS/PASS/PASS；真实E/F及force-parameter混合VJP整体PASS。
  原生自重复失败位置：无。新球面系数相对旧网格最大绝对变化 8.3446503e-07。

## 29处逐项结果

| 位置 | 原始局部 | EMA局部 | 原始权重：后续计算 | EMA：后续计算 |
| --- | --- | --- | --- | --- |
| E00 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| A01 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| S01 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| A02 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| F01 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| A03 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| S02 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| A04 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| F02 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| A05 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| S03 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| A06 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| F03 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| A07 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| S04 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| A08 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| F04 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| A09 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| S05 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| A10 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| F05 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| A11 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/FAIL |
| S06 | PASS/PASS/PASS | PASS/PASS/PASS | FAIL/FAIL/FAIL | FAIL/FAIL/FAIL |
| A12 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| F06 | PASS/PASS/PASS | PASS/PASS/PASS | FAIL/FAIL/FAIL | FAIL/FAIL/FAIL |
| A13 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| S07 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| A14 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |
| F07 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS |

## 旋转/反射、实际几何与导数边界

- checkpoint_original local：输出最大全对象relative-L2=4.6337036e-07（F04 / rotation_y_0.713）；所有输出/导数角色最大绝对残差=9.5367432e-07。完整分角色残差在JSON；未审定表示门，保持HOLD。
- checkpoint_original consumer：输出最大全对象relative-L2=1.9788272e-06（A09 / rotation_x_0.417）；所有输出/导数角色最大绝对残差=0.0046539307。完整分角色残差在JSON；未审定表示门，保持HOLD。
- checkpoint_ema local：输出最大全对象relative-L2=4.7572001e-07（F04 / rotation_y_0.713）；所有输出/导数角色最大绝对残差=1.1324883e-06。完整分角色残差在JSON；未审定表示门，保持HOLD。
- checkpoint_ema consumer：输出最大全对象relative-L2=2.2359238e-06（A07 / rotation_x_0.417）；所有输出/导数角色最大绝对残差=0.0048217773。完整分角色残差在JSON；未审定表示门，保持HOLD。

局部覆盖每处旋转/反射及对应full VJP/输入-参数混合方向积；后续计算覆盖坐标/节点输入/全部活动参数及坐标-参数混合方向积。实际原帧和mask分支另由frame阶段重验，原mask未改。后续计算验证使用固定完整有向4节点图，含重复边及一对相反PBC位移，未取得动态构图/stress或公开数据任务资格。

真实能量头输出/归约FP64，force与参数混合VJP为FP32。QA另对28个K作确定性内存扰动后验证结构，再精确恢复全部参数；不是训练、optimizer step或新checkpoint更新。

## 未闭合项与下一步边界

后续计算FAIL保留，先定位具体窗口/归约传播及近零角色，不能更换权重/输入、放宽门或用ALL29通过覆盖逐处失败。新表示/系数容差未审定项HOLD；实际mask跨分支、动态图/PBC邻居重建、stress及公开数据任务精度另验。当前不提交A100/GPU或训练。

## 快照与作业

- convert：CPU作业118059；Git `c303f2606193ce2f4de0ed34f4211ce31b82a597`，路径 `/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-v3-canonical-cpu-20261124-r1/snapshots/result-convert-118059`。
- local：CPU作业118078；Git `ebf8be4085c830367eb95be7ae8168fcbdb48d2e`，路径 `/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-v3-canonical-cpu-20261124-r1/snapshots/result-local-118078`。
- consumers：CPU作业118080；Git `da5eced34328057aa73dd5ed3f104d9661e4d0d1`，路径 `/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-v3-canonical-cpu-20261124-r1/snapshots/result-consumers-118080`。
- frame：CPU作业118089；Git `f70444d63bb9550b10b25a8666f5689f39452114`，路径 `/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-v3-canonical-cpu-20261124-r1/snapshots/result-frame-118089`。
- qa：CPU作业118089；Git `76230677189e9a37d8dce6653fe375d02e52bbd4`，路径 `/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-v3-canonical-cpu-20261124-r1/snapshots/result-qa-118089`。

## 失败角色与帧边界补充（CPU封存）

| 来源/位置 | 失败角色数 | 有分量超限的角色数 | relative-L2超限角色数 | 最大绝对误差 |
| --- | --- | --- | --- | --- |
| checkpoint_original/S06 | 31 | 31 | 0 | 0.0010986328（output_0） |
| checkpoint_original/F06 | 17 | 17 | 0 | 0.00048828125（output_0） |
| checkpoint_ema/A11 | 1 | 1 | 0 | 9.1552734e-05（mixed:blocks.5.ga.rad_func.net.3.weight） |
| checkpoint_ema/S06 | 27 | 27 | 0 | 0.0010986328（output_0） |
| checkpoint_ema/F06 | 15 | 15 | 0 | 0.00045776367（mixed:input_1） |

以上是失败归类，不是已证明根因；不能仅因relative-L2很小而取消逐分量门，也不能对近零角色改分母。完整失败指标见CLOSURE.json及逐处JSON。

- 实际帧mirror_x：regular forward max_abs=2.8312206e-06，含mask行forward max_abs=4.7087669e-06；mask不改，跨分支域保持HOLD。
- 实际帧mirror_z：regular forward max_abs=5.3048134e-06，含mask行forward max_abs=5.3048134e-06；mask不改，跨分支域保持HOLD。
- 实际帧householder_1_2_m1：regular forward max_abs=3.8458966e-06，含mask行forward max_abs=0.0010919567；mask不改，跨分支域保持HOLD。

最终CPU封存再次复核6497个冻结文件及11条软链，原始/EMA新checkpoint哈希复核一致。所有基线、旧FAIL、原mask与图语义保持，新增GPUh=0，无训练。
