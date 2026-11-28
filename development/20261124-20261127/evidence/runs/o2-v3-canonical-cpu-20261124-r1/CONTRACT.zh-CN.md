# canonical O2 CPU转换与29处验证合同

2026-11-24。用户授权提交CPU：校准实际反射基、投影14处正频率权重、生成独立checkpoint，验证29处及后续计算的旋转/反射、前向、full VJP和混合二阶；每步快照。禁止A100/GPU和训练/优化器更新作业。原两项冻结基线与旧FAIL只读保留。

## 来源与函数身份

新模型为`equiformerv3-ictc-o2-canonical-min-v1`，取自然球谐宇称`p_l=(-1)^l`，局域canonical为0e与正频率二维表示。固定C从实际冻结Wigner/m布局校准；14个SO2模块各m1/m2共28个Parameter转换为合法K，m0及其余合法参数原样初始化。原始/EMA两套来源分别明确加载，不能混用或假EMA。

S/F采用Gaunt不变量结构，并以原有限网格逐degree块的最小二乘标量校准其归一化；标量和旧K差完整记录，构成新模型的非训练系数定义，不宣称原有限网格checkpoint严格等价。此实现细化避免未说明地理想化旧系数；不是通过改旧模型的系数取得旧资格。

## 分层对照

- C1转换：固定基、原m0、投影K、各模块独立身份、所有未变化参数逐张量验证，独立checkpoint全state和strict reload；原基线逐文件/软链哈希复核。
- C2局部：29处分别检查实际表示的旋转/反射；各活动输入与新模型全部活动参数VJP、输入-参数方向混合二阶。A用packed与expanded交织矩阵两种实现；S/F用同一新K的两种收缩顺序，旧grid仅另报新旧函数偏差。
- C2后续计算：每处在原完整producer/block/consumer中验证坐标、节点输入、全部活动参数VJP及坐标-参数混合二阶；含alpha/完整softmax/envelope/逆旋转/聚合/proj/残差/norm，FFN至下一block与末端norm/readout；另验ALL29组合及真实能量头。固定有向小图含重复边、完整邻域及PBC位移例子，图边界与mask分支单列。

## 精度、统计与状态

能量输出及求和/归约FP64，其余运算、累加、导数、统计FP32；禁止TF32/AMP/半精度/CUDA Graph。合成feature probe不称作公开数据任务精度或真实力损失训练。

实现前向/full VJP按O2全对象每角色`epsilon_max=max_abs/max(max_abs(candidate),max_abs(reference),1e-7)<=1e-5`统计。混合二阶、完整block及E/F使用既有分量`2e-5+2e-5|ref|`与每角色全数组relative-L2<=1e-5，不叠加两套不同口径；精确零参考仅在差也为零时通过，活动性/NaN/Inf单列。原生自重复及所有近零失败保留。

新表示/系数等变残差记录绝对与相对全对象指标，FP32门未审定则`HOLD_THRESHOLD_UNAPPROVED`，不凭数值接近自行宣布资格。没有通过的对象保留FAIL/HOLD，不由Slurm COMPLETED推断canonical接入通过。参数更新结构约束用不写入checkpoint的确定性扰动检查，不执行训练或优化器一步。

## 快照

每个CPU作业先核验6497个冻结文件/11条软链（实际数量以回执为准），AST检查及冻结源码/文档，使用独立Git保存control/result及哈希清单。转换后checkpoint以SHA命名；后续作业读取转换回执并复核其哈希。所有计算、JSON聚合和哈希均在Slurm CPU节点。
