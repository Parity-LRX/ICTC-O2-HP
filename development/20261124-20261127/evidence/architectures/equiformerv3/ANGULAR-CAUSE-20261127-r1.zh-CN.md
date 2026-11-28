# 新阶数L8 OOM、网格与连续三步FAIL原因

完整证据、消融、原数组及恢复见[执行报告](../../runs/o2-angular-cause-20261127-r1/REPORT.zh-CN.md)。对象为固定seed的l=m=1..8 canonical结构模型及dd7a3db后端，原l4m2 checkpoint锚点另列；不是原checkpoint任务精度或所有full-parity整网资格。

L8原Torch参考侧在blocks.0.ga.rad_func的径向展开反向IndexSelectBackward0/index_add申请416MiB失败。异常前active35.795683GiB，宽边[5252,81,256]的49块存储占19.896431GiB；缓存闲置2.854707GiB，但最大闲置块406.559MiB。相同完整诊断/训练协议及训练前RNG，仅清空无引用缓存即可执行，allocated历史峰值仍36.720218GiB。高存活量和前置诊断缓存碎片共同导致本次OOM，不直接等于L8的不可运行边界。

双线性原grid.y=ceil(main_rows*H/32)，越过设备65535限制。交换网格维度的独立诊断副本消除该发射错误并保持对照；没有改dd7a或授予大N整网资格。

新阶数第一个非零通道差分别在blocks.0.ga.so2_linear_1.so2_m_linear.1/.3/.1/.2；第一处双线性差均在blocks.0.ga.act。默认一分区四链没有复现新Gaunt布局的原生FP32归约树，前64行同操作数匹配为L2四分区单链、L4六分区四链、L6/L7三分区四链，均不需交换角向轴。尾区、六种递归轴排列及高阶仍须后续修复独立验收。

完整原始梯度和混合二阶门内差，在少数小梯度分量经过原生Adam逐元素归一化后传到零初始化bias/EMA，造成20个原relative-L2 FAIL。L2/L4单双线性足以FAIL，L6本例为组合效应，L7通道/双线性单独都可FAIL。24次相同真实梯度的原生CUDA更新/EMA精确重放，未发现optimizer实现错误。

本轮仅诊断，原loss/精度/阈值、原三步状态连续性、dd7a3db及全部旧FAIL/HOLD/封存保持。单A100串行120614→120617→120636，1720秒/0.477777778GPUh。
