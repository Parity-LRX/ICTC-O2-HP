# UVW完整训练4+50项FAIL：原因与修复方案

2026-11-26，诊断对象为`df12a3a85ec3b66c9e4909dca0c90a86a12b1516`，原始证据为已冻结的A100119167/CPU119186。**本例主因已定位到：Triton前向及生成力所需的一阶伴随改变FP32舍入结果，微小力值差异改变L2MAE的归一化损失余切，再传播到参数梯度及坐标混合二阶。** 同余切实验中原有4+50项失败全部消失；单独回退二阶矩阵乘不能修复。当前未改候选、loss、权重、输入、精度或阈值，原训练FAIL保留。

## 1. 条件及复现范围

- 模型：`equiformerv3-ictc-o2-canonical-min-v1`、固定投影EMA和原生训练配方，100原子、5252条有向周期边；28个正频率`CanonicalMLinear`保留原始Parameter。所有case从相同model/optimizer/EMA/scheduler状态恢复，原始Parameter身份、参数和图/标签哈希核对一致。
- 前向/伴随/二阶及误差统计均FP32，能量输出/求和FP64；TF32/AMP/CUDA Graph禁用。整网按每个完整角色同时检查分量`2e-5+2e-5*abs(reference)`和relative-L2 `<=1e-5`，不替换局部O2的epsilon_max口径。
- 两次GPU allocation均`GPU-A100/XHPC-A100`，1节点1卡、case串行：119324为8种后端切换，119328为真实三项loss的完整余切控制。实际GPU秒数109+32=141，即**0.039166667 GPUh**。CPU119322/119325/119329分别做旧回执核对、消融分析和综合。源码预检119323、119326/119327均保存，其中119327是补充实验实际使用版本。所有状态见[调度回执](SCHEDULER.txt)与[GPU记账](GPU-ACCOUNTING.json)。
- 本轮native的输出、完整梯度和force_mixed与原119167逐位复现；候选输出和force_mixed也逐位复现，完整参数梯度存在至多`5.9610e-7`的relative-L2差异，失败对象仍为相同4+50个。本轮8个case同一物理GPU，第二次allocation单独在自身原生参考下配对；未把不同GPU上的数值当作同一次配对。[复现逐对象结果](results/analysis-119325/REPRODUCTION.json)

## 2. 原54项具体失败在哪里

原119167训练批次的能量relative-L2为`5.5813e-9`，力为`5.2911e-6`（max_abs `1.6391e-6`），stress为`3.4902e-6`，均通过。总loss在FP32下逐位相等，为`1.514505744`。这里使用原生训练模型/损失的归一化单位，不将这些绝对值直接解释为eV/Å物理误差。因而失败不能解释为“只有600原子的E/F通过，100原子训练输入不同”。

四个完整loss参数梯度均只超relative-L2门，所有分量通过：

| 原始Parameter | 原119167 relative-L2 | max_abs |
|---|---:|---:|
| `blocks.3.ga.alpha_norm.bias` | 1.0650047e-5 | 2.1979213e-7 |
| `blocks.4.ga.alpha_dot` | 1.2497083e-5 | 4.3567270e-6 |
| `blocks.4.ga.alpha_norm.weight` | 1.1209688e-5 | 8.0186874e-7 |
| `blocks.4.ga.alpha_norm.bias` | 1.0619327e-5 | 3.3061951e-7 |

原标为50个`force_mixed`失败角色实际包括：

- **49个坐标—参数混合导数**，失败均来自relative-L2，最大`2.1266555e-5`。分布在block1～6的embedding、radial、norm、FFN和SO2 K等参数，并非50个独立的Triton权重梯度核错误。28个K的完整loss一阶参数梯度全部通过；其中10个K的力损失混合角色失败，均为block2～6两处SO2的m=2块。
- **1个坐标—坐标二阶角色**`coordinates`。relative-L2 `9.1829e-6`通过，但`[20,2]`、`[55,0]`、`[55,2]`三个分量超限。最大绝对差`8.5264e-5`，对应分量预算`2.7889e-5`，约3.06倍，因此该角色仍失败。

总共53个角色只超relative-L2，另1个角色只超逐分量门。失败角色参考范数最小`0.00614017`，没有零分母问题。力项与其他loss项的梯度抵消代理仅1.09～1.33，证据不支持把四项失败主要归于损失项之间的巨大抵消。[54项完整指标](results/cpu-119322/FAILED-ROLES.csv) · [三个失败分量](results/cpu-119322/FAILED-COMPONENTS.csv) · [CPU核对](results/cpu-119322/SUMMARY.json)

## 3. 因果对照

N表示原生FP32矩阵乘，T表示当前IEEE FP32 Triton矩阵乘；列顺序为通道混合前向/生成力所需一阶伴随/训练二阶。只在诊断进程内替换执行函数，不写回候选仓库。训练梯度统计292个原始Parameter登记角色，force-only统计293个登记角色（参数及坐标，保留None活动状态）。

| 后端 | 完整参数梯度FAIL | force-only二阶FAIL | 共享原生力余切后二阶FAIL |
|---|---:|---:|---:|
| N/N/N，原始module | 0 | 0 | 0 |
| N/N/N，adapter布局 | 0 | 0 | 0 |
| N/N/N，flatten matmul | 0 | 0 | 0 |
| **T/T/T，当前候选** | **4** | **50** | **0** |
| T/N/N | 2 | 192 | 0 |
| N/T/T | 2 | 93 | 0 |
| T/T/N | 4 | 51 | 0 |
| **N/N/T** | **0** | **0** | **0** |

T/T/N与T/T/T的能量、力和损失余切逐位相同；N/N/T与原生flatten路径的这些对象逐位相同，证明这两个实验确实仅切换所需二阶执行。布局/flatten控制未重现失败。仅回退一阶或前向也不能保证误差改善，说明不同阶段舍入差会相互增强或抵消，不能按单个核的局部误差大小推断整链单调收益。[完整消融](results/analysis-119325/ABLATIONS.json) · [阶段控制](results/analysis-119325/STAGED-CONTROLS.json) · [布局控制](results/analysis-119325/LAYOUT-CHANGES.json)

对力损失写成`g=J_F^T v_F`，其中`v_F=partial L_F/partial F`。令a/r表示候选/参考，共同使用原生`v_r`时比较`J_a^T v_r`与`J_r^T v_r`，再单独计算`J_a^T(v_a-v_r)`。本次直接loss梯度与用自身余切的functional VJP逐位相同，无需用“遍历顺序”解释这50项差异。

- 当前候选真实force-only二阶最大relative-L2为`2.1266555e-5`；共享原生余切后，**所有登记角色均通过**，最大relative-L2 `2.0314624e-6`，分量超限数从3降到0。
- 对原50个失败角色，余切变化项在总误差方向上的投影比例为`0.9511～1.0410`、中位`0.9890`，支持余切扰动主导。该比例是向量投影，可大于1，不是时间占比或非负误差预算。[逐角色分解](results/analysis-119325/ERROR-DECOMPOSITION.csv) · [分解统计](results/synthesis-119329/DECOMPOSITION-SUMMARY.json)
- 第二次GPU实验对**真实energy+forces+stress三项loss**进一步验证：能量和stress余切逐位一致，只有力余切不同。使用自身余切的线性化loss与候选原始loss给出的292个参数梯度逐位相同；换成原生共同余切后，4项FAIL全部消失，所有参数登记角色最大relative-L2 `1.5895088e-6`。原生线性化控制本身至多`4.7158e-7`，未引入门外误差。[完整三项loss控制](results/synthesis-119329/TOTAL-LOSS-CONTROL.json)

四个原始失败参数在该完整loss控制中的relative-L2变化：

| Parameter（省略`blocks.`） | 真实loss，119328 | 相同原生余切 |
|---|---:|---:|
| `3.ga.alpha_norm.bias` | 1.0744954e-5 | 5.1361707e-7 |
| `4.ga.alpha_dot` | 1.2455559e-5 | 2.3989816e-7 |
| `4.ga.alpha_norm.weight` | 1.1175311e-5 | 1.5309867e-7 |
| `4.ga.alpha_norm.bias` | 1.0584228e-5 | 2.1398569e-7 |

因此本例已经能区分主要误差入口与较小的固定余切Jacobian误差；尚未逐条确定哪个通道dot的哪次FMA贡献最大，也未证明某个新累加方案一定可修复。

## 4. 为什么局部/E/F通过仍不能通过训练

原生loss为energy `per_atom_mae×20`、forces `l2mae×20`、stress `mae×5`。force项对100个原子的残差`e_i=F_i-F_i^label`使用向量模长平均：

`L_F=(20/N) sum_i ||e_i||_2`，`v_i=(20/N) e_i/||e_i||_2`。

在非零残差处，余切对力的导数为`(20/N)(I-u_i u_i^T)/||e_i||`。本次没有零残差或NaN；原子55的残差范数最小，为`0.0024938274`，方向归一化仍足以放大小扰动。其余切相对差达到`1.0366e-4`，全力余切relative-L2为`1.3226392e-5`；能量和stress没有MAE符号翻转。该变化进入attention、norm、FFN等参数的力Jacobian转置后，原有49个参数混合角色及坐标二阶超限，完整loss中另有四个梯度超限。[余切与敏感原子](results/synthesis-119329/COTANGENT-SENSITIVITY.json) · [实际残差分布](results/analysis-119325/LOSS-RESIDUALS.json)

局部检查验证的是单个双线性通道混合，137×2行随机输入/权重/余切及其方向积，用完整角色epsilon_max。实际训练每处有5252×2行、28处组合，并经过球激活、attention/softmax、邻居聚合、残差、norm、FFN和能量头。局部双线性伴随公式成立，不保证这一非线性整链在真实loss余切下满足另一个完整数组relative-L2门。

E/F检查验证能量值与坐标一阶导数；训练验证`(partial F/partial theta)^T v_F`及坐标二阶。E/F允许的微小偏差可以改变`v_F`；标量loss是大量项的FP32归约，值相同也不要求其梯度相同。原生与候选都确定性自重复通过，故不是用随机重复波动可以消除的失败。

代码上，`_Matmul.backward`为`dA=g B^T`、`dB=A^T g`并递归保留高阶计算图，当前实验未发现缺项或断梯度作为主因；两边都是FP32，`input_precision="ieee"`也只约束乘法精度，不规定与原生GEMM相同的FMA/归约顺序。实际入口已缩小至生成力的前向及一阶伴随算术，而不是仅检查最终二阶权重归约。

## 5. 修复方案与验收顺序

1. **先建立实际训练算例回归。** 从固定状态的28处捕获真实输入、K、实际坐标伴随和原生/候选loss余切，覆盖四种K形状、m=1/2及10504行的实际归约。先用它们逐处比较前向及生成力所需输入伴随，再接回原来的后续计算定位误差放大位置。保留当前9组随机局部测试及全对象/分角色统计，加入原子55、坐标三个失败分量和四个attention参数作为定位索引，仍验完整数组。
2. **新候选优先修改前向及输入伴随的FP32累加。** 在`uvw_triton.py::_uvw_mm`为前向、输入伴随和长归约权重伴随分别选布局/tile/累加调度，评估分块多累加器及固定归约树，保持全部乘加FP32和IEEE输入，保留原Parameter及可微伴随。以真实力与真实loss梯度的改善选择方案，不以随机局部epsilon_max或单核速度单独定优。数值更准确也未必更接近固定FP32参考，必须实际配对验证。二阶本轮固定余切已过门，优先级低于生成力的数值一致性，但后续修改仍须重验完整二阶。
3. **保留N/N/T作为诊断桥接路径。** 它在本固定训练状态下所有参数梯度及force-only角色均通过，证明保留Triton二阶可行。若需临时安全训练，可回退完整原生module；这些控制不计作全Triton修复或新速度收益。T/N/N、N/T/T和T/T/N均有失败，不作为可直接启用的修复。
4. **按原门重验后再测性能。** 新候选独立保存源码快照，依次验局部完整VJP/所需二阶、完整层、同帧E/F、真实三项loss的292个原始参数梯度和force-only全部活动/空梯度角色、更新/EMA/optimizer及缓存失效。通过对应训练门后再采整步耗时、显存与完整热点，速度和显存分别报告。不得固定、detach、裁剪或替换真实训练的loss余切，也不改L2MAE、mask、输入、权重、FP32精度或批准阈值。

本轮停在原因定位与修复方案。`ICTC-O2-HP`仍为原`df12a3a`干净工作树，未实现下一候选，旧FAIL/HOLD不改。运行控制和原始对象保存在本run；恢复与校验见[封存回执](FINAL-SNAPSHOT.json)。
