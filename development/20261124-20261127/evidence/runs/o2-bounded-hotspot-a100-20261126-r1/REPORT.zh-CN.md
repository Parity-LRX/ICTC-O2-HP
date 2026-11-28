# dd7a3db热点与原生Torch延迟/显存对照

候选dd7a3db；复用A100120352、AMD120353已验收计量和纯CPU分析120390。本轮不修改实现或旧FAIL，新增GPUh=0。

原生对照为同一`equiformerv3-ictc-o2-canonical-min-v1`的Torch执行，采用原bounded64角向收缩及七块非重入checkpoint策略。checkpoint、原292活动权重、图/PBC/重复边、RNG、loss、clip/AdamW/EMA/scheduler和精度按原回执配对；不把不同模型/未投影上游加速实现当作同一分母。

## 延迟和显存

| 模式 | 原生Torch s | dd7a3db s | 加速比 | 耗时变化 |
|---|---:|---:|---:|---:|
| ef | 3.620289 | 1.823517 | 1.9853× | -49.63% |
| train | 1.839475 | 0.926420 | 1.9856× | -49.64% |

每格为3个新进程各3预热/5同步调用，共15样本的中位数；包含建图/传输、求力和完整原生更新。公共状态恢复在计时外；它仅定义计量条件，完整各自真实损失连续三步已在原run独立验收。E/F600原子31512边，训练100原子5252边，二者工作量不同。原15个样本、进程中位数和范围保留在原WORKLOADS/RAW-TIMINGS与本轮TRACE-METADATA。

| 模式 | 原生allocated GiB | 候选allocated GiB | 差值MiB | 原生reserved GiB | 候选reserved GiB | 差值MiB |
|---|---:|---:|---:|---:|---:|---:|
| ef | 7.586998 | 7.587051 | +0.05469 | 10.566406 | 10.501953 | -66 |
| train | 11.331550 | 11.343708 | +12.45020 | 11.980469 | 11.937500 | -44 |

allocated峰值没有改善：E/F仅增加56KiB，训练增加12.45020MiB（+0.10730%）。reserved分别少66/44MiB，反映此次分配器保留量；本轮未测最大容量。memory使用独立新进程，不把profile内存统计替代此表。

## 当前热点：绝对GPU时间

| 类别 | EF原生ms | EF候选ms | 候选占比 | 训练原生ms | 训练候选ms | 候选占比 |
|---|---:|---:|---:|---:|---:|---:|
| GEMM/GEMV | 966.577 | 693.612 | 39.42% | 461.982 | 304.711 | 38.49% |
| ICTC channel GEMM | 0.000 | 252.264 | 14.34% | 0.000 | 86.861 | 10.97% |
| ICTC bilinear | 0.000 | 190.328 | 10.82% | 0.000 | 95.908 | 12.11% |
| ICTC weight gradient | 0.000 | 0.000 | 0.00% | 0.000 | 19.522 | 2.47% |
| copy/layout | 391.479 | 233.441 | 13.27% | 135.242 | 55.821 | 7.05% |
| elementwise | 274.039 | 157.357 | 8.94% | 167.225 | 108.096 | 13.65% |
| index/scatter/gather | 114.289 | 114.484 | 6.51% | 48.248 | 46.203 | 5.84% |
| fill/zero | 162.759 | 92.914 | 5.28% | 83.741 | 47.857 | 6.05% |
| reduction/norm | 19.349 | 19.338 | 1.10% | 16.486 | 15.688 | 1.98% |
| unclassified kernel | 5.590 | 5.599 | 0.32% | 11.388 | 11.005 | 1.39% |

分母为每份trace全部实际GPU kernel时间和，不是同步墙钟。原生Torch的部分GEMM、copy和逐元素计算被融合后转入ICTC类别，所以ICTC占比升到约25%不能解释成整体变慢或“绝大多数时间仍在Triton”。

候选E/F kernel合计1759.338ms，训练791.673ms。最大单类仍是GEMM/GEMV（39.42%/38.49%）；通道核、双线性核、权重伴随分别如表。训练权重伴随归约合计约19.52ms，其中weight_fma8约9.30ms，仅占候选训练kernel合计约1.17%，应按整网份额确定优化优先级。

![以绝对毫秒比较原生Torch与新候选的类别及阶段耗时](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-bounded-hotspot-a100-20261126-r1/figures/NATIVE-VS-DD7A-HOTSPOTS.png)

## 阶段与矩阵形状

| 模式 | 阶段 | 原生ms | 候选ms | 候选占比 |
|---|---|---:|---:|---:|
| ef | model_forward | 503.795 | 497.925 | 28.30% |
| ef | force_from_energy | 1430.125 | 1261.252 | 71.69% |
| ef | graph_build_transfer | 0.013 | 0.014 | 0.00% |
| ef | unattributed | 0.148 | 0.147 | 0.01% |
| train | model_forward | 92.373 | 91.612 | 11.57% |
| train | force_from_energy | 259.606 | 220.628 | 27.87% |
| train | loss_backward | 567.488 | 474.594 | 59.95% |
| train | loss | 0.277 | 0.255 | 0.03% |
| train | clip | 0.694 | 0.695 | 0.09% |
| train | optimizer | 2.419 | 2.436 | 0.31% |
| train | EMA | 1.438 | 1.435 | 0.18% |
| train | graph_build_transfer | 0.017 | 0.016 | 0.00% |

E/F主要在从能量求力（71.69%）；训练loss反向59.95%、求力27.87%。optimizer/EMA/clip合计不足1%，当前主要计算瓶颈在完整求导链。

| 模式 | BMM形状族（含伴随） | 候选ms | 调用kernel数 |
|---|---|---:|---:|
| ef | canonical 2x2 basis shape family | 308.484 | 168 |
| ef | edge rotation shape family | 225.894 | 59 |
| ef | other BMM shapes | 15.188 | 175 |
| train | canonical 2x2 basis shape family | 106.285 | 336 |
| train | edge rotation shape family | 112.610 | 163 |
| train | other BMM shapes | 12.001 | 443 |

上表按实际输入形状分组，不将所有未映射模块的反向耗时猜测为geometry。边旋转形状族与原执行源码`so3.py:359/366`的Wigner正/逆旋转、`input_block.py:96`的初始嵌入旋转相符；2×2形状族与实际adapter/install.py:68/85的C_in/C_out换基相符。它们及各自伴随仍是Torch路径，属于进一步融合的候选。

| 模式 | 主要矩阵调用 | 输入形状 | GPU合计ms | kernel数 |
|---|---|---|---:|---:|
| ef | Bilinear | `[[19,19,19],[31512,19,32],[31512,19,32]]` | 150.076 | 56 |
| ef | _Matmul | `[[63024,1024],[1024,256]]` | 112.045 | 14 |
| ef | aten::bmm | `[[31512,19,256],[31512,256,25]]` | 98.555 | 7 |
| ef | aten::bmm | `[[1024,2,2],[1024,2,31512]]` | 73.187 | 21 |
| ef | _Matmul | `[[63024,768],[768,192]]` | 63.452 | 14 |
| ef | aten::bmm | `[[512,2,2],[512,2,31512]]` | 62.481 | 21 |
| ef | aten::addmm | `[[928],[31512,1280],[1280,928],[],[]]` | 58.736 | 14 |
| train | Bilinear | `[[19,19,19],[5252,19,32],[5252,19,32]]` | 73.229 | 154 |
| train | _Matmul | `[[10504,1024],[1024,256]]` | 38.465 | 28 |
| train | aten::bmm | `[[5252,19,256],[5252,256,25]]` | 33.164 | 14 |
| train | aten::bmm | `[[1024,2,2],[1024,2,5252]]` | 25.519 | 42 |
| train | Bilinear | `[[25,25,25],[100,25,512],[100,25,512]]` | 22.679 | 154 |
| train | _Matmul | `[[10504,768],[768,192]]` | 22.108 | 28 |
| train | aten::bmm | `[[512,2,2],[512,2,5252]]` | 21.542 | 42 |

形状表合并同形状的各相位/模块，计数为关联的GPU kernel数。逐相位/模块版本保留在MATRIX-SHAPES.csv；不是把Bilinear的两次网格发射当作两次数学调用。模块stage前向映射未覆盖所有反向，覆盖率保留在TRACE-METADATA，phase与kernel类别则独立完整统计。

## 显存峰值为何没有下降

profile_memory事件显示：原生与候选的E/F峰值都出现在求力中的index_add_/index_put_边聚合，输入边消息形状[31512,25,128]。训练峰值都出现在loss_backward的BmmBackward0，原始输入形状[5252,5,128]×[5252,128,25]，与输入嵌入旋转反向边界相符。

这些是触发profile峰值的算子和边界：E/F触发分配约0.5MB、训练约2.6MB，不能把全部7.59/11.34GiB归到该分配。共同存活的消息、伴随及高阶计算图仍在；双线性临时量的消除没有降低整网全局峰值。完整张量存活期尚未逐项分解。profile进程的峰值数值保留在PROFILE-MEMORY-PEAKS.json，独立memory进程的正式峰值以上表为准。

## 墙钟与trace边界

| 模式 | 后端 | kernel和ms | kernel并集ms | 首末kernel间隔ms | profile包含ms | 独立计时中位ms |
|---|---|---:|---:|---:|---:|---:|
| ef | native | 1934.1 | 1934.1 | 4838.0 | 4843.7 | 3620.3 |
| ef | candidate | 1759.3 | 1759.3 | 1863.9 | 1868.3 | 1823.5 |
| train | native | 924.3 | 924.3 | 2953.3 | 2961.9 | 1839.5 |
| train | candidate | 791.7 | 791.7 | 1083.3 | 1090.7 | 926.4 |

原生/候选E/F kernel次数88,968→11,478，训练65,072→29,267；kernel总时间降幅约9.04%/14.35%，小于独立墙钟约49.63%的降幅。结合kernel次数和profile中发射间隙大幅减少，支持本次收益包含显著Python调度/细碎发射开销消除。profile受记录开销影响，不能把上表间隔相减直接解释成稳态CPU耗时，更不能用kernel份额分摊独立墙钟。

后续提速应优先检查边旋转及其伴随、C_in/C_out换基、布局copy与通道计算之间的融合边界；显存优化应沿已定位的边聚合/输入嵌入旋转反向峰值，检查边消息及其伴随的存活期和分块消费。保留完整混合二阶、原FP32归约语义及每次独立三步门控。本轮止于分析，未启动下一实现。

## 数据与制图方法

原始回执、trace SHA、CPU external-id关联、完整CSV与本轮脚本已保留；每个类别/相位的calls和ms均回算至原始kernel总量。原候选封存及8324/历史FAIL不改。图为内部报告，零基线、颜色加纹理区分后端、不隐藏0值、无过滤/平滑；一份trace不产生统计置信区间。

制图遵循本地主机scientific-visualization技能，参考Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065；2026-11-26核对当前v2，无期刊版本信息。[当前arXiv记录](https://arxiv.org/abs/2609.00065)。这项引用说明流程来源，不支撑本工作区的性能数字。
