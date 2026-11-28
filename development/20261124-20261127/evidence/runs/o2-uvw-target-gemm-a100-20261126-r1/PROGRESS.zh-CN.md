# 当前进度

- c05bc7b的tag、源码封存和独立解包QA不改，基线入口复制为C05-BASELINE-SNAPSHOT.json；新工作独立分支codex/uvw-target-gemm-v4。
- AMD120151发现Torch2.7.1的_will_engine_execute_node不支持autograd.grad中的叶子节点，保留探测失败和源码。AMD120152验证保守保留叶子、对非叶子使用本次graph-task查询的方案，覆盖视图捕获、别名、双输入/单输入、完整backward及真实形式force-mixed-K；全部与原生CPU对照一致。
- 正式mask候选73e8367，tag snapshot/uvw-target-mask-v4a-20261126。不改变requires_grad或detach权重，只省略本次graph task未消费的输入梯度输出，保留输入伴随对K的依赖。CPU120157冻结，A100120158执行112组/672局部角色、通用9组/62角色及原legacy/isolated/train-only四后端自重复完整三步；GPU已完成428秒，AMD120159验收中。实际结构记录为E/F K伴随28→0，训练84→56，每次真实求力剪28个输出；数值资格以AMD最终回执为准，未测速度。
- 8项FMA原型暂存本run，未覆盖production mask-only源码。CPU120161证明全展开版本寄存器溢出较多，保留现场；120163改为交错四链和禁止内层展开，候选均0 spill。120164进一步编译分形状tile：16×64/8 warps为64 registers、0共享内存、0 spill；16×128/8 warps为128 registers。只做离线FP32指令/资源审计，不代表GPU数值或速度通过。后续优先验证较小寄存器压力的分形状配置，仍保持四条8项FMA链及最终固定合并树。

当前仍按原loss/optimizer/FP32与FP64边界/原阈值、完整二阶、真实损失和三步连续状态执行；不共享余切、不冻结权重。AMD原独立三步通过前不作耗时/显存测量。所有GPU单A100单卡串行，CPU汇总/编译仅AMD。

阶段A终态：AMD120159完整PASS，真实112组/672角色全部0ULP，通用9组/62角色原门通过。原legacy、isolated、train-only每一步的全部原始梯度/force-mixed/optimizer/EMA/scheduler/更新后输出0FAIL/0ULP；保存MASK-*指针，AMD120166 RNG与连续性通过，120174生成MASK-NUMERICAL-SUMMARY。剪枝次数为每次求力28，EF长K伴随0、训练56，完整VJP/真实force-only混合仍保留需要的参数路径。

阶段B候选d6ae976（tag snapshot/uvw-fma8-wide-v4b-20261126）只对256×1024、K10504的实际原权重布局使用显式四链8项FMA和16×64/8-warps；其他三形状保持c05的32×32/4-warps。内层loop_unroll_factor=1避免离线发现的溢出，固定分区及合并树不变。CPU120167冻结，A100120168原协议完整回归/三步，AMD120169随后；AMD120175对正式源13份SM80产物验证全FP32、无TF32/半精度/FP64/Tensor Core，实际FMA配置0 spill。尚未测候选耗时/显存。

阶段B终态：A100120168/AMD120169完整原门PASS（真实112/672角色与三套真实三步均0FAIL/0ULP，通用62角色原门通过）；RNG120190、数值汇总120191通过。通过后才采GEMM120188/AMD120189；256×1024显式全局读取外积为1.87–2.01ms，比c05的0.90–1.04ms更慢。该候选和回执完整保留，不选为最终性能实现。

阶段C共享内存CUDA SIMT候选de713b0，tag snapshot/uvw-fma8-shared-v4c-20261126。按踩坑台账Triton/CUDA后端边界，只把实际256×1024、K10504改为32×64/4warps专核，其余形状保持c05。共享内存加载每8个K的A/B并复用，保留四条FP32链与全部加法树，不更改数学/精度。NVRTC12.8离线SM80构建120218：128 registers、3072共享内存、0 spill；CUDA源码/PTX/CUBIN打包并在运行时核验哈希，unsupported设备或布局走原Triton。不使用cuBLAS计算回退。120216使用旧Triton ptxas不能识别新PTX8.7，失败现场保留，改由同版本NVRTC内置assembler输出CUBIN，未篡改PTX版本。120224安装，120245/120246冻结；正式完整GPU120247，随后AMD120249/120250。完整包SHA纳入局部、通用、三步清单并交叉核对；当前验收中，尚未测该候选性能。

阶段C终态：A100120247/AMD120249及extras120250完整PASS，112/672角色和原三套独立三步0FAIL/0ULP。通过后GEMM120253/AMD120254全数组0ULP，256×1024仍1.34–1.61ms，比c05慢，保留SHAREDC-*指针，尚未作整网性能采集。没有因局部数值通过而把该退化版本当作最终结果。

阶段D 8324b33（snapshot/uvw-fma8-shared32-v4d-20261126）：每32项一次装入共享内存，四条8项链复用，各分区32项的barrier次数从8降至2；不改FMA/加法依赖。CPU120260构建、120261安装、120262冻结，32×64/4warps为127 registers、12288 bytes shared、0 spill。正式GPU120263，AMD120266/120267，随后门控冻结120268、单列GEMM120270/AMD120271；这些依赖必须在完整AMD PASS之后才能开始测量。完整包hash同时覆盖CUDA源码/二进制，不能因Python算子SHA未变而沿用C资格。

阶段D终态：A100120263完成441秒；AMD120266完整PASS，原legacy/isolated/train-only每步全训练记录0FAIL/0ULP，112组/672角色0ULP、通用62角色通过。120267的RNG连续性/完整ULP/正式源PTX审计全部通过。之后120268冻结，GEMM120270/AMD120271：256×1024两个实际阶段分别0.896769/0.894460ms，对同轮c05的1.042312/1.041651ms为1.1623×/1.1646×（耗时约减14%），输出全数组0ULP；其他三形状保留原核。120272门控选择与冻结，最终整网120273四后端单卡串行启动，AMD120274依赖其完成。120276再次将单列GEMM原记录与正式执行capsule的完整源码/CUBIN包绑定并复核输出；不向原GPU记录伪造此前未写的字段。

CPU120275另有64×64/8warp离线资源探索（127 registers、16KiB shared、0 spill），未覆盖正式32×64生产源，未运行GPU资格或性能，不授予数值/速度资格。本轮已取得合格32×64形状收益，因此保留其原型但不扩展测试范围。

最终已完成：

2026-11-26求导目标与8项FMA候选`8324b33`完成原独立三步AMD完整验收：legacy/isolated/train-only各自真实损失，原292梯度、实际force-only混合、原生optimizer/EMA/scheduler及更新后输出全部0FAIL/0ULP；112组×6实际角色与通用9组/62角色通过，EF/VJP按原门通过（后两者仍有门内非零误差）。每次求力剪去28个未消费K伴随，EF长K权重GEMM28→0、训练84→56。最终256×1024长归约采用共享内存CUDA SIMT 8项FMA的32×64 tile，其他形状保留Triton；四链/合并树、完整二阶、原loss/optimizer/精度/阈值不变，不冻结权重/共享余切。c05bc7b及旧202等全部FAIL/HOLD保留，三步通过不等于长期轨迹一致。

最终同A100单卡串行四后端配对：E/F 3.760130s、训练1.862938s，对c05耗时分别-1.68%/+0.05%，对Torch+3.63%/+2.28%；allocated 7.587051/11.331603GiB，reserved 10.566406/11.980469GiB。训练基本持平，显存无收益；单列256×1024约1.042→0.895ms（减14%）未变成明显训练墙钟收益。EF不运行新FMA专核，mask/final差值不归因于该核。速度/显存分开判断，不授予2×/5×/5×目标。完整数字、ULP、失败性能候选和每步快照见[runs/o2-uvw-target-gemm-a100-20261126-r1/REPORT.zh-CN.md](runs/o2-uvw-target-gemm-a100-20261126-r1/REPORT.zh-CN.md)、[恢复入口](runs/o2-uvw-target-gemm-a100-20261126-r1/FINAL-SNAPSHOT.json)。嵌套仓库ICTC-O2-HP分支codex/uvw-target-gemm-v4，未推送。

