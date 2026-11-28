# UVW混合通道Triton候选及V3热点诊断

用户2026-11-25授权：将ICTC-O2-HP fork到当前工作区，重写混合通道uvw Triton核，写完保存快照，再测热点与耗时。只执行本候选、局部验证和V3 canonical同条件诊断，不扩大为数据集重训或MD。旧FAIL/HOLD不改。

## 源码与方程

独立本地Git fork位于工作区根目录`ICTC-O2-HP/`，分支`codex/uvw-triton-v1`，不推送。保留上游Git历史，活动包恢复到固定基线e4763c76的official源码，CPU逐文件核对。后续实现及每次修复独立提交并保存bundle/源码哈希。

UVW定义：每个合法角路径p，`z[e,w,c] += sum(u,v,a,b) W[p,u,v,w] CG[p,c,a,b] x[e,u,a] y[e,v,b]`。u/v/w独立；保留canonical/full parity表示和原合法系数。第一版限定共享W、FP32 CUDA；未实现的per-edge权重明确拒绝，不声称已覆盖。一般UVW采用两阶段通道收缩，不存储`E*U*V*W`外积。

V3实际热路径是在两侧反射换基之间的`z[e,a,w]=sum_v K[w,v] x[e,a,v]`，为左输入固定0e标量1、U=1的UVW特例；专用入口直接消费`[E,2,V]`，不构造虚拟1/0左输入，不改变K/Parameter/通道/表示。使用FP32 IEEE `tl.dot`，禁止TF32/半精度/AMP/CUDA Graph。前向、输入/权重VJP和需要的混合二阶由同一个可微Triton矩阵乘原语闭合，不以冻结K或原生高阶替代训练覆盖。

## 数值与诊断

算子输出、全部活动输入/原始权重VJP使用完整数组、每角色独立的`max_abs/max(max_abs(candidate),max_abs(reference),1e-7) <= 1e-5`；高阶同时报同指标，但新对象正式高阶容差保持HOLD。整网按既有分量`2e-5+2e-5*abs(reference)`及全对象relative-L2<=1e-5，近零两边为0才通过；全部检查拒绝NaN/Inf，保留原生自重复与失败。能量输出/归约FP64，其余FP32。先完成局部实现检查，再采本轮诊断；整网训练旧失败不阻止已授权热点采集、不授予数值资格。

XHPC-A100单节点单GPU、4CPU、64GiB、所有后端/模式/采集子进程串行。native Torch、冻结HP、候选三方，E/F为600原子31512边，完整训练100原子5252边；原配置/投影EMA、bounded球乘法、7块非重入重算保持。严格回放同一公共model/optimizer/EMA/scheduler状态，分别新进程测3次预热、10次同步墙钟、一次独立allocated/reserved峰值和完整kernel trace。复用旧公共状态但重新核对全部组件哈希；恢复在计时外，图构建/传输/调用内布局计时。旧记录只作历史参考，本轮配对分母来自同allocation测量。

运行前固定源码快照、CPU预检和哈希，GPU启动核对A100实卡、环境和禁止项。所有allocation失败/取消也按实际Elapsed记GPUh，延续500GPUh总账。结束保存原始时间、完整kernel事件/分类、参数/状态/图身份和报告，速度与显存独立判断，不把热点占比下降单独当作收益。
