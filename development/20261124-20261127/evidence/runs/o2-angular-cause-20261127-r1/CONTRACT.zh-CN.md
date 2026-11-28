# 新阶数三类异常因果诊断

用户2026-11-26/27明确授权接手：定位L8 OOM峰值张量、双线性Triton网格发射错误、L2/L4/L6/L7实际连续三步的最早计算偏差。仅诊断与隔离复现，原dd7a3db及全部历史FAIL/快照不改。

来源 runs/o2-dd7a-angular-capacity-a100-20261126-r1；后端dd7a3db136c1d2946aa0e41b87a167d8f940b122。新GPU作业afterany:120441，单XHPC-A100、单GPU、各子进程串行；AMD读取/哈希/重算。无并行代理，不干预其他项目作业。

保持原模型、初始状态、真实energy/force/stress损失、clip/AdamW/EMA/scheduler。能量FP64，其余FP32；禁TF32/AMP/half/CUDA Graph。原每角色分量门2e-5+2e-5*abs(ref)且relative-L2<=1e-5、局部epsilon<=1e-5不变，ULP仅描述。不共享真实损失余切，不在连续三步中恢复状态。

复现分三类：1) 原L8协议/仅训练/候选，记录allocator历史及大分配对应算子、形状、调用栈，区分存活分配与reserved；2) 原双线性网格在65535上下边界及原84032边配置，诊断副本仅交换网格维度，不改算术，核对结果；3) 原全native、全candidate、仅channel候选、仅bilinear候选四组同初态各自连续三步，保留完整数组、RNG和阶段/局部误差；用实际梯度和原生optimizer重放补因果链。消融仅诊断，不授予修复资格。

独立执行包、输入/源码/结果哈希与GPUh回执分别记录。旧容量作业继续完成。原扫描训练FAIL及非OOM错误原判保留。
