# 三路线诊断热点：执行条件先冻结

用户本轮授权补齐V3 canonical、TECE canonical、TECE full parity各自E/F推理及完整训练的诊断热点，并采集原生分布作对照。数值FAIL/HOLD保留，不再阻止诊断；本轮结束后停在分布审阅，不做第二步算子优化或第三步正式验收。GPU只用XHPC-A100一个节点、每次一GPU，所有路线/后端/采集进程串行。

## 配对矩阵与公平边界

主矩阵是三路线 × 两模式 × 原生Torch/当前ICTC，共12个后端工作负载。V3另测未投影的上游EMA模型，默认内存策略不可运行时再记录同方程重算版本，仅列绝对成本，不与canonical变体计算等价加速比。两种TECE另测上游可选Eqx加速路径，严格加载相同公共状态；若不支持所需导数或OOM，记录异常，不静默回退。

同一配对保持：同源码/架构参数、同参数张量字节、同完整optimizer/EMA/scheduler状态、同输入及标签/图哈希、同求力与stress路径、同内存重算策略、同precision、相同warmup/测量边界。唯一设计变量是执行后端。原生和ICTC独立子进程。

不同架构保留各自原生cutoff/邻居规则、层数/宽度、归一化、loss及optimizer：V3和TECE-C保留原生stress；TECE-F按历史E/F配方，无stress。三者不强行改成同数学模型，也不跨架构报速度倍率。TECE-F固定随机初始化，只支持结构/实现性能；V3为独立canonical投影EMA模型，非原始上游模型同checkpoint改写。

|路线|固定架构身份|模型条件|
|---|---|---|
|V3 canonical|独立`equiformerv3-ictc-o2-canonical-min-v1`，投影EMA权重|7块，30,333,409参数；原始上游V3另列绝对成本|
|TECE canonical|`8593380e6cbe`，官方`TECE-OAM-RRA-1.0` checkpoint|8层，C64，L/M=4，cutoff=6 Å|
|TECE full parity|`5b428ee36ffd`，历史O2-ECE递归架构，`parity=True`|2层，C64，L/M=3，cutoff=5 Å，seed=20261124固定随机初始化|

因此，两个TECE模式的分布分别属于上述固定版本和配置；公平配对发生在每条路线的原生/ICTC之间。完整原生config、loss权重与实际图见结果中的MODEL-CONFIGS.json。

## 状态、输入和计时

- E/F为公开Al帧的原定600原子超胞、batch=1；训练为100原子原始帧、batch=1。公开E/F/stress标签、未修正总能量选择、单位换算与原生自由原子mask保持。不同模式的原子数明确列出，不作推理/训练速度倍数。
- 推理使用初始checkpoint/已声明随机初始化；训练先用原生执行3个真实完整步形成公共暖机状态，保存全部权重/optimizer/EMA/scheduler。原生和ICTC每次测量前都严格恢复这一公共状态，恢复和H2D拷贝在计时外。这保证每个采样的输入状态逐字节一致，不让两端沿不同更新轨迹漂移。
- 每个后端的time、memory、trace分别在新进程采集；各3次代码warmup。time记录10个同状态完整步的同步墙钟，含原生图生成、传输、求力，训练还含loss、完整反向、原生clip、optimizer、EMA及对应scheduler间隔；不计状态恢复。
- memory记录公共状态驻留后的base及一个完整步的allocated/reserved峰值，包含常驻EMA等统一harness对象，不称纯部署最小显存或最大容量。
- trace记录一个同公共暖机状态完整步，optimizer状态已预热，避免把初始化开销当稳定训练热点。仅trace加语义阶段标签，time/memory不加标签。

## 精度与语义

训练状态恢复时为optimizer状态建立独立副本，阻止PyTorch共享CPU Adam step张量导致公共模板漂移；采集前逐组件核对恢复后的model/optimizer/EMA/scheduler完整状态哈希，采集后再次检查CPU模板未变。第一allocation的E/F不含optimizer更新，可保留；旧控制下的任何部分训练不纳入新结果，日志仍保存。

能量输出及所有能量求和/归约FP64，其他模型计算/导数/误差统计FP32；禁止TF32、AMP/半精度、CUDA Graph，确定性归约，4CPU线程。保持原Parameter对象、图/PBC/重复边、有向贡献、原径向MLP、attention、激活、残差/norm、能量求导力及真实力损失混合二阶。没有新算子优化、mask修改、精度放宽或重训。

V3双方用此前已验证的bounded canonical球乘法和7块原生非重入重算；TECE-C双方用同一embedding/edge-update/interaction/product非重入重算；TECE-F双方按原源码执行。原始默认执行OOM另保留身份，不将修改内存策略后的数据称默认生产路径。

## 分布统计与验收

118630实测边界：TECE canonical的native-eqx-module和ictc-eqx在align步骤均报`batch1 must be a 3D tensor`，发生在ECE之前。原生加速几何使用packed Wigner，而当前fused=False module route要求dense bmm输入；这两组E/F与训练标为不可运行，不能计算同加速切口的速度对比。原生融合Eqx与主矩阵原生Torch/ICTC的分布分别完整保留。当前不扩大到修复该加速切口，待用户先查看分布再决定下一阶段。

补充原生加速对照：TECE canonical的原生Eqx融合RRA与ICTC当前module route分别登记；另采保留Eqx但同样选择module route（fused=False）、ECE仍为原生Eqx的配对参考。这样原始融合执行、拆开融合切口、替换ECE的成本均可分辨。执行118630只补这些诊断条件，不开发新kernel。候选预检必须记录ICTC实际pair调用，防止融合路径绕过接入；数值FAIL/HOLD仍不阻断诊断。

TECE full的原生Eqx融合卷积也单列；同切口参考和ICTC双方均将两处EceO2Interaction选择原源码已有module分支，保留其他模块的Eqx设置，以实际调用RecursiveContraction。该选择仅改变执行路径，参数和原函数保留；相同的实际use_eqx模块标志与ICTC调用计数一起登记，避免把融合分支绕过接入的数据冒充ICTC。

所有样本按公共状态回放。恢复本身在计时之外，但Parameter版本变化导致派生权重布局缓存失效时，其call内重建成本计入本轮诊断；代码warmup不等同于生产部署持续复用权重缓存。这一边界在报告中公开，下一步是否改为持续缓存推理由用户审阅后决定。

完整kernel表只统计Chrome trace中cat=kernel的事件，CPU算子、GPU注释不重复加算；给出调用次数、总ms、占该trace所有kernel时间的百分比及完整名称。CUDA memcpy/memset单列。阶段归因依据CPU launch correlation与trace注释，无法映射的显式列unattributed，公开映射覆盖率；不再将所有不认识的核统称geometry。

报告必须附配对条件逐项检查、参数/状态/图/标签SHA、硬件UUID和实际后端flags、loss/optimizer/stress差异、原始10个耗时、trace及CSV。诊断的可运行检查只拒绝执行异常/非有限值/缺失完整训练链；数值误差阈值不作为profile开关。旧FAIL不删除，不授予新的正式数值资格或投稿速度资格。
