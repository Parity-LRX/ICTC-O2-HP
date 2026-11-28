# dd7a3db热点与原生框架对照

用户授权：对新候选进行热点分析，对比原生框架延迟及显存。复用已完整验收和封存的A100120352及AMD120353结果，候选dd7a3db；不修改算子、原loss/optimizer/精度/阈值、权重、训练状态或任何旧FAIL。

native指同一个V3 canonical独立模型、同checkpoint/输入/图/RNG/原生训练配方的Torch执行，不是另一方程的上游未投影模型。E/F600原子31512边，训练100原子5252边。沿原3个新进程各5次同步调用的15样本中位数，显存单独新进程采集，保留allocated/reserved及原始字节。

只在CPU-AMD调度节点解析已有trace和CSV、作图与哈希，新增GPUh=0，不提交GPU作业，不创建并行代理。所有cat=kernel && ph=X事件保留，逐类及阶段归约回总量；kernel和、活跃区间并集、profile区间和独立稳态墙钟分别记录，不能相互分摊。

模块注释未覆盖的反向事件保留未归因。矩阵形状由实际CPU external id关联到GPU kernel，不能由kernel名称猜测层。profile_memory中的峰值仅定位该profile的存活期，不替代独立memory回执。报告图为内部分析图，非投稿规格认证；保留全部源表、脚本、哈希和输出。
