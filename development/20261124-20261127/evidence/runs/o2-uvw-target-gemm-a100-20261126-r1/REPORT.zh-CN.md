# 求导目标剪枝与8项FMA分形状优化

最终候选 `8324b335244e2b382123a5e89183afa7194ed83b`，独立仓库 `/public/home/sps-xia/bjxiao/o2-benchmark/ICTC-O2-HP`，分支 `codex/uvw-target-gemm-v4`，未推送。operator SHA256 `8b2af4ac04906f2ed8ef8e6f040417b4aa93df84f6b897d02b7adea4b28ba585`；完整源码/二进制包SHA256 `7cf759a5bc62cec6bdac1ba152e287032375dc3898081b035df4ff35083c9c25`。CUDA版本以完整包身份区分，不只比较Python算子文件。

固定canonical投影EMA模型、原292参数、28处替换、7块重算、图/PBC/重复边、三项各自真实loss、clip/AdamW/EMA/scheduler均保持。能量输出/归约FP64，其他运算/累加/导数/误差统计FP32；禁TF32/AMP/半精度/CUDA Graph。没有冻结权重、detach活动参数、共享余切或在真实独立三步中恢复状态。

## 实现与逐步快照

A：73e8367（tag `snapshot/uvw-target-mask-v4a-20261126`）按当前autograd graph task查询所需输入边，保守保留叶子；不再仅凭静态needs_input_grad。只剪去不会被当前求导目标消费的K伴随输出，输入伴随继续保留对K的可微依赖。真实求力28次K伴随被剪除；训练保留实际所需56次。API不可用时保守回退原反向。AMD120159通过完整原三步，112组×6角色及通用9组/62角色通过。

B：d6ae976（tag `snapshot/uvw-fma8-wide-v4b-20261126`）显式Triton外积FMA、16×64 tile/8 warps，AMD120169完整通过；随后获准的单列测量120188/AMD120189显示256×1024约1.87–2.01ms，比c05更慢。保留该退化候选和回执，不作为最终实现。

C：tag `snapshot/uvw-fma8-shared-v4c-20261126`。256×1024、K=10504、原方向G.T@X使用32×64输出tile/4 warps、共享内存复用的CUDA SIMT 8项FMA微核；其余三种真实权重形状保持原32×32 Triton核。新专核由NVRTC12.8离线生成SM80 CUBIN，源码/PTX/CUBIN及manifest全部封存，运行时校验哈希并在当前Torch stream启动；无运行时编译、cuBLAS计算回退或参数值缓存。其他设备/布局保守走既有Triton核。

每个原分区保留四条独立FP32 FMA链，每条8个有效项，32项为一组；分区内(c0+c1)+(c2+c3)，分区间按原顺序左结合相加。分区P分别为4/3/5/9，跨度ceil(10504/P)；尾部mask，地址乘法前int64。新核128寄存器、3072字节共享内存、0 spill，不含Tensor Core/TF32/half/double指令。完整高阶继续通过同一可微矩阵乘递归构建，不保存partial到反向图。

C版本AMD120249及112/672角色0FAIL/0ULP，随后120253/AMD120254发现256×1024仍退化为1.34–1.61ms，全部证据保留。D最终tag `snapshot/uvw-fma8-shared32-v4d-20261126`一次装入32项共享tile，四条8项链依次复用，每32项从8次屏障降到2次；32×64/4warps、127寄存器、12288共享字节、0 spill。数学链和最终树不变。D重新执行完整GPU/AMD原三步后才获准测量。上段128寄存器/3072共享字节为保留的C版本，不是D资源。

采用CUDA专核是本轮实际实现边界；其余矩阵乘/归约仍为Triton，通用角装配保持Torch。不能将最终后端称为纯Triton。

## 原门与原独立三步

完整每角色逐分量abs(a-r)<=2e-5+2e-5*abs(r)，同时relative-L2<=1e-5；零参考仅零差。局部O2完整每角色epsilon_max<=1e-5，分母max(双方maxabs,1e-7)。拒绝非有限值。ULP为附加描述统计，不加门。

最终AMD回执 `/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-target-gemm-a100-20261126-r1/results/full-audit-amd-120266`。112实际组×6角色及通用9组/62角色通过，实际K伴随最大ULP=0。

| 协议 | 步 | 原292梯度FAIL | 28K最大ULP | 实际force-only混合FAIL | 更新参数ULP | 全记录FAIL / R / ULP |
|---|---:|---:|---:|---:|---:|---|
| legacy | 1 | 0 | 0 | 0 | 0 | 0 / 0 / 0 |
| legacy | 2 | 0 | 0 | 0 | 0 | 0 / 0 / 0 |
| legacy | 3 | 0 | 0 | 0 | 0 | 0 / 0 / 0 |
| isolated | 1 | 0 | 0 | 0 | 0 | 0 / 0 / 0 |
| isolated | 2 | 0 | 0 | 0 | 0 | 0 / 0 / 0 |
| isolated | 3 | 0 | 0 | 0 | 0 | 0 / 0 / 0 |
| train-only | 1 | 0 | 0 | 0 | 0 | 0 / 0 / 0 |
| train-only | 2 | 0 | 0 | 0 | 0 | 0 / 0 / 0 |
| train-only | 3 | 0 | 0 | 0 | 0 | 0 / 0 / 0 |

E/F完整输出最大R=5.17874196e-06、最大ULP=10033191；联合完整VJP最大R=4.6408033e-07、最大ULP=1530920960。近零分量ULP可很大，按原全对象门判断。原生自重复、候选自重复、全部活动与None角色、optimizer/EMA/scheduler、更新后输出均覆盖。

legacy精确复现原119414原生轨迹；isolated控制仅隔离诊断RNG，train-only无诊断。各自真实loss、状态、RNG和实际旋转矩阵均登记，跨步连续性验收；不把共享余切控制当作修复。原a80的202项FAIL、df12的4参数+50混合FAIL、bfb第三步FAIL及所有旧HOLD保持不改。当前三步通过不外推长期轨迹一致。

## 同A100单卡串行性能

以下均在最终AMD完整验收之后采集。各模式/后端3个新进程，各3次预热与5次同步调用；列全15次中位数，原始样本保留。公共起始状态恢复位于计时外，只定义同条件基准，不替代真实连续三步。计时含图/传输、求力、完整原生训练更新；memory/trace为独立进程。allocated、reserved与速度分别判断。

| 模式 | 后端 | 中位s | allocated GiB | reserved GiB | ICTC kernel占比 |
|---|---|---:|---:|---:|---:|
| ef | native | 3.628391 | 7.586998 | 10.566406 | 0.00% |
| ef | c05 | 3.824343 | 7.587051 | 10.566406 | 14.47% |
| ef | mask | 3.795842 | 7.587051 | 10.566406 | 11.95% |
| ef | candidate | 3.760130 | 7.587051 | 10.566406 | 12.00% |
| train | native | 1.821443 | 11.331550 | 11.980469 | 0.00% |
| train | c05 | 1.862068 | 11.331603 | 11.980469 | 12.33% |
| train | mask | 1.863538 | 11.331603 | 11.980469 | 11.40% |
| train | candidate | 1.862938 | 11.331603 | 11.980469 | 11.25% |

| 模式 | 最终对c05耗时变化 | 对mask耗时变化 | 对Torch耗时变化 | allocated对c05字节 | reserved对c05字节 |
|---|---:|---:|---:|---:|---:|
| ef | -1.68% | -0.94% | +3.63% | +0 | +0 |
| train | +0.05% | -0.03% | +2.28% | +0 | +0 |

性能判定：最终E/F对c05耗时减少1.679%，完整训练+0.0467%视为基本持平；最终仍比原生Torch慢3.631%/2.278%。allocated/reserved相对c05均无收益，未达整网速度/显存目标。E/F剪枝后不执行新权重FMA专核，mask与final的算术工作相同；两者0.941%的墙钟差不能归因为FMA收益。训练单列核心虽快约14%，完整步并无可辨认的改善；不把计时样本当作统计独立科学重复，也不从单列核倍率推断整网收益。

kernel百分比按每份trace全部实际GPU kernel耗时和归一化，不是墙钟比例；CUDA专核归入ICTC GPU。完整分类/算子/阶段库存与配对条件在性能AMD回执。2×/5×/5×目标及容量门不因数值通过而自动授予。

## GEMM单列与实际调用次数

实际捕获X/G已展平驻留，每种形状/阶段/后端3次预热、30次同步调用，仅包括dK矩阵乘及partial归约，排除展平、loss、高阶、optimizer和图传输。a80保留原三步FAIL，仅为诊断成本列；c05/mask/final均有完整AMD资格。

| 形状 | 阶段 | Torch ms | a80 ms | c05 ms | mask ms | final ms | c05/final | final ULP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| [256, 1024] | energy_forward_and_force | 0.4796 | 0.6451 | 1.0423 | 1.0425 | 0.8968 | 1.162× | 0 |
| [256, 1024] | loss_backward | 0.4798 | 0.6447 | 1.0417 | 1.0408 | 0.8945 | 1.165× | 0 |
| [192, 768] | energy_forward_and_force | 0.2611 | 0.4938 | 0.5068 | 0.5079 | 0.5076 | 0.998× | 0 |
| [192, 768] | loss_backward | 0.2608 | 0.4943 | 0.5074 | 0.5081 | 0.5076 | 1.000× | 0 |
| [512, 128] | energy_forward_and_force | 0.1640 | 0.2802 | 0.2625 | 0.2622 | 0.2626 | 1.000× | 0 |
| [512, 128] | loss_backward | 0.1622 | 0.2790 | 0.2611 | 0.2616 | 0.2614 | 0.999× | 0 |
| [384, 96] | energy_forward_and_force | 0.1209 | 0.4602 | 0.1726 | 0.1725 | 0.1729 | 0.998× | 0 |
| [384, 96] | loss_backward | 0.1208 | 0.4596 | 0.1722 | 0.1724 | 0.1722 | 1.000× | 0 |

| 模式 | 后端 | 长K | 权重GEMM次数 |
|---|---|---:|---:|
| ef | native | 63024 | 0 |
| ef | c05 | 63024 | 28 |
| ef | mask | 63024 | 0 |
| ef | candidate | 63024 | 0 |
| train | native | 10504 | 56 |
| train | c05 | 10504 | 84 |
| train | mask | 10504 | 56 |
| train | candidate | 10504 | 56 |

![耗时、显存与全部kernel分布](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-target-gemm-a100-20261126-r1/figures/UVW-TARGET-FMA8-20261126-r1.png)

## 适用边界与恢复

固定E/F600原子31512边、训练100原子5252边的实现和原三步验收；非任务测试集MAE、长MD或全部形状/硬件证明。剪枝与专核分别留有中间候选，方便分辨收益来自少算工作还是更快的矩阵乘。其他形状保守保留原实现。

c05bc7b原源码/tag/归档保留不改；本轮每步快照见PROGRESS.zh-CN.md与snapshots/。最终SOURCE-AND-RECEIPTS归档经过独立解包、文件集合与逐文件哈希校验，入口FINAL-SNAPSHOT.json。大数组保持工作区原路径并列RAW-RESULT-MANIFEST.json，不声称小归档包含所有原始数组。GPU-ACCOUNTING.json包含失败/取消的实际分配，按显式单GPU请求和运行时设备记录记账。
