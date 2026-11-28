# 按求导目标剪枝与256×1024 GEMM优化

2026-11-26用户明确授权：先按本次求导目标去掉多余28次K伴随，再以8项FMA和分形状tile优化退化256×1024 GEMM。原c05bc7b源码、tag、封存及所有旧FAIL/HOLD保留。新分支codex/uvw-target-gemm-v4。遵循karpathy-guidelines与踩坑台账；不创建并行代理，不询问已授权的常规执行。

只优化执行方式：不冻结/替换原Parameter，不detach权重，不共享loss余切，不更换原loss、clip/AdamW、EMA或scheduler，不在真实独立三步中重置模型/优化器/RNG。坐标输入伴随必须继续保留对K的可微依赖；需要K梯度或混合二阶时须按真实autograd目标计算。

固定同一canonical投影EMA模型、292原始Parameter、7块重算，E/F600原子31512边、训练100原子5252边；保留图/PBC/重复边、全部后续计算和三项原生loss。能量输出与归约FP64，其余模型运算/累加/导数/误差统计FP32；禁TF32/AMP/半精度/CUDA Graph。整网完整每角色分量门abs(a-r)<=2e-5+2e-5*abs(r)，并且relative-L2<=1e-5；零参考只接受零差。局部O2完整每角色epsilon_max<=1e-5、尺度max(maxabs双方,1e-7)。NaN/Inf拒绝，ULP描述不加门。

阶段A先在AMD验证实际graph-task查询的语义，再保存mask-only候选。以原legacy真实独立三步、isolated/train-only、自重复、EF、完整VJP、force-only混合、全部原始梯度与完整原生更新验收；记录全部RNG/实际旋转矩阵及相邻状态连续性。不得用仅needs_input_grad、eval/train标签或只读infer模式取代真实求导目标。

阶段B在已识别的四条8项FP32 FMA链与合并树不变的条件下构造显式FMA内核，去掉K16填零槽。优先只替换256×1024、K=10504及已登记布局，其余形状保持c05。分形状tile/warp候选各有独立源码快照和配置身份。每个要测耗时/显存的候选，均先通过AMD原独立三步完整验收与受影响范围完整导数回归；不先测速度再找能过门的输入或精度。

最终同一个A100单GPU串行测native/c05/合格新候选。计时每后端/模式3个新进程，各3次预热、5次同步调用，全部15样本保留。benchmark恢复公共起始状态在计时外，仅定义相同工作负载，不能替代数值资格中的真实连续三步。memory/trace独立进程；allocated/reserved与速度分别验收；kernel总和占比不是墙钟比例。单列GEMM驻留/展平X/G计时不含完整loss/optimizer/图传输，不代替整网收益。

具体采集为native、c05、mask-only、最终FMA四后端，每模式各20份条件（3次time进程+1次memory+1次trace，四后端）。先在完整AMD通过后用同一实际X/G单列GEMM作候选选择，每形状/阶段/后端3次预热和30次同步调用；保留native/c05/mask/新候选及a80诊断列，a80原三步FAIL不改、不作合格整网分母。两个实际余切阶段各覆盖四种形状，未优化形状也保留原始结果。候选选择只用已通过原独立三步的源码；如更改tile或实现，重新保存源码并通过AMD三步后才测。

所有Python、哈希、构建、数值和汇总经Slurm CPU-AMD/GPU-A100；登录节点只控制/轻量读/Git/文件组织。GPU限XHPC-A100、--nodes=1 --gres=gpu:1，所有候选/模式/后端串行；不使用TTV/4090D/H100。500GPUh累计，失败与取消实际分配也计账。源码/输入/配置/环境/回执共同定义结果；每个优化保存Git tag和可恢复快照，原c05及历史失败不改写，不扩展DFT/MD/数据集重训。
