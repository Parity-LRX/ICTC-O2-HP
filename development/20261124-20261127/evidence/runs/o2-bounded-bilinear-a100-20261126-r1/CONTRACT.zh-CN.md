# bounded双线性融合执行合同

用户2026-11-26明确授权：提速优先，融合bounded双线性及完整VJP/混合二阶，去掉64行Python循环和拼接；每步快照，各自真实损失的原独立三步AMD通过后才测整网耗时/显存。8324b33及所有历史FAIL/HOLD不改。本run独立，嵌套仓库分支codex/bounded-bilinear-v5，遵循karpathy-guidelines与踩坑台账，不创建并行代理。

固定原canonical模型、292原始Parameter、7块非重入checkpoint、实际几何/重复边/PBC、E/F600原子31512边、训练100原子5252边；loss三项、clip/AdamW/EMA/scheduler、RNG和实际旋转相位保持。原8324通道核保持。只替换Bilinear(k,a,b)的执行，k为原本不训练的canonical_kernel缓冲，不冻结任何活动原权重；a/b及余切的完整可微链必须保留，不共享真实loss余切、不在独立三步中重置模型/optimizer/RNG。

能量输出/求和FP64，其余运算、乘积、累加、导数、误差统计FP32；禁TF32/AMP/half/CUDA Graph。完整每角色分量abs(a-r)<=2e-5+2e-5*abs(r)，且relative-L2<=1e-5；零参考仅零差。局部完整O2角色epsilon_max<=1e-5，分母max(双方maxabs,1e-7)。NaN/Inf拒绝，ULP仅附加统计。不能通过改变系数、函数、权重、输入或更换损失过门。

先验证受影响14处的真实输入/余切与19/25角向系数、置换/非连续/尾部输入的前向、双输入VJP及所需混合二阶，再完整整网EF/VJP、原legacy三步和isolated/train-only三步及自重复；原生对照精确复现119414原协议。隔离诊断只恢复诊断RNG，不恢复实际训练状态。记录28通道K伴随剪枝和双线性执行身份。仅通过原门的最终完整包可进入性能采集。

性能固定同A100单GPU串行native/8324/新候选，每模式每后端3新进程，每进程3预热、5同步调用，保留15样本；公共起始状态恢复仅在计时外，不能替代数值验收的真实连续三步。计时包括图/传输、求力和完整原生更新；memory/trace独立进程，速度和allocated/reserved分别判断，kernel和不是墙钟分摊。

登录节点仅调度/Git/文件组织/轻量文本读写，全部Python/哈希/编译/数值/聚合在CPU-AMD或GPU-A100计算节点。GPU只XHPC-A100 --nodes=1 --gres=gpu:1，全部工作串行，不用其他GPU硬件；实际GPUh（包括失败/取消）继续500GPUh总账。每次候选保存源码/Git、协议、系数/输入/结果哈希和可恢复快照，不原地修改冻结证据，不扩展训练集重训或MD。
