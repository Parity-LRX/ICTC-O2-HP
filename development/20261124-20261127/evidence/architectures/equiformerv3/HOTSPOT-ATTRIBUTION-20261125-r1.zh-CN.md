# EquiformerV3 canonical：当前快照的耗时与显存归因

快照日期：2026-11-25；证据版本 `o2-four-route-bottlenecks-20261125-r1`。本记录对应已测的 canonical 独立变体及当前 HP adapter，不外推为原始 EquiformerV3 或 O(2) 方法的固有性能。原图、公共测量合同及封存入口见[架构根目录快照](/public/home/sps-xia/bjxiao/o2-benchmark/architectures/HOTSPOT-SNAPSHOT-20261125-r1.zh-CN.md)。

## 1. 实际模型及替换边界

| 对象 | 本快照身份 |
| --- | --- |
| 官方架构源码 | `a7300c58df683dc99cb48027d5bfd4c887486c48`，见[源码入口](/public/home/sps-xia/bjxiao/o2-benchmark/architectures/equiformerv3/SOURCE.zh-CN.md) |
| canonical 实现 | 控制快照 `c14109a12653bcdd224bf0ad6f20e378623de35c` 的[canonical.py](/public/home/sps-xia/bjxiao/o2-benchmark/architectures/equiformerv3/source/canonical-cpu-control-c14109a12653/canonical.py:91) |
| 权重及配置 | 独立 `equiformerv3-ictc-o2-canonical-min-v1` 投影 EMA；30,333,409 参数、7 blocks、C128、L4/M2、cutoff6 Å；训练保留原生 E/F/stress 与 AdamW |
| 实际 HP 接口 | [CanonicalLinear.forward](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-paired-diagnostic-a100-20261124-r1/adapter/install.py:59)，作用于14个 SO2Linear 类实例内的28个正频率块 |
| 共享内存策略 | 双方使用相同7块非重入重算和 bounded canonical 球乘法，属于公开的诊断执行条件 |
| 输入 | 推理600原子/31512有向边；训练100原子/5252边；batch1 |

此处的28是实际正频率实现块数，不是29个逻辑登记点全部由HP执行。E00、球激活、FFN、聚合、残差/norm和能量头等按各自既有实现保留。canonical 投影改变了原模型的权重约束，故本表 native 是同一独立变体的数学参考，不能换成未投影上游 checkpoint 作等价分母。

归因源码 SHA256：`canonical.py` 为 `7f7ef1a8aeef0bb3fe089693ee52de580aedabed55e5d31c7dccce027c38e3a7`；`adapter/install.py` 为 `c0423e972f96e16f6dbbaacff77532cb7477aebb84709110393e93fa8cab1511`；冻结 `uvu_triton.py` 为 `17d35ecf415c32dfe0b031fef323771636aec22e3b95efbe200226950b2704d9`。可恢复副本见[源码封存](/public/home/sps-xia/bjxiao/o2-benchmark/analysis/o2-four-route-bottlenecks-20261125-r1/SOURCE-CODE-SEAL.json)。

## 2. 实测结果

| 模式 | 同步墙钟原生 → ICTC s | 耗时比 | 峰值 allocated 原生 → ICTC GiB | 全部 kernel 和原生 → ICTC ms | HP 占候选 kernel 和 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E/F | 3.557517 → 23.334539 | 6.559× | 7.585464 → 7.904446 | 1934.149 → 21753.284 | 91.53% |
| 完整训练 | 1.797980 → 6.619900 | 3.682× | 11.343170 → 12.275710 | 891.730 → 6160.535 | 85.30% |

原生 GEMM/GEMV 占推理50.01%、训练49.64%；候选主要转为 HP kernel。推理 HP kernel 和为19.912s，训练为5.255s。wall 与 kernel 来自独立采集，不能相减后把差值当纯 CPU 时间。

## 3. 耗时为何增加

### 3.1 宽矩阵乘进入通用 UVU 归约：主要归因

**源码机制。** 原生 factored canonical 层先按 `C_in` 换基，再 `F.linear(x,K)`，最后按 `C_out` 换回。adapter 保留原 K 和两侧换基，但把该线性映射写成 `0e × m → m` 的 UVU：构造1/0左输入，转置并连续化右输入，使用 shared K 调用 HP。

实际 K 的 `(C_out,C_in)` 为256×1024、192×768、512×128、384×96。全部输入轴 `V>8`，经[派发函数](/public/home/sps-xia/bjxiao/o2-benchmark/baselines/icml-o2-stage1-20261124-r1/payload/operator/upstream/official/src/ictc_o2_hp/uvu_triton.py:235)进入[通用 _forward](/public/home/sps-xia/bjxiao/o2-benchmark/baselines/icml-o2-stage1-20261124-r1/payload/operator/upstream/official/src/ictc_o2_hp/uvu_triton.py:91)。该核以 `(edge,output)` 展开，加载整个输入轴并 `tl.sum`，输入轴补到2的幂；没有实现原生 GEMM 式的边/输出/输入分块复用。数学阶数仍为 `O(E*C_out*C_in)`，没有凭空新增耦合路径，但执行方式不匹配当前宽矩阵。

**实测。** 推理 `_forward` 56次共17.821s，输入伴随28次共2.091s；两类最宽前向配置合计17.301s。训练 HP 前向约4.467s、输入伴随0.702s、shared weight梯度0.086s。因此本快照首要成本明确在前向，不能笼统归为权重梯度归约。

**归因判断。** 当前 adapter 用通用宽 V 归约承担原本的正频率矩阵乘，是这次显著变慢的主要原因。E/F 本身要对能量求坐标导数，7块原生 checkpoint 重算又重复执行相关前向；完整训练还包含混合二阶。重算策略两侧一致，不是只向候选增加了重算。

### 3.2 训练二阶重新进入 PyTorch：存在，但不是本路线首要热点

冻结HP的[_double_backward](/public/home/sps-xia/bjxiao/o2-benchmark/baselines/icml-o2-stage1-20261124-r1/payload/operator/upstream/official/src/ictc_o2_hp/uvu_triton.py:442)对保存的 left/right/weight/grad_output 重新建立可微 operands，调用 `torch_uvu_forward` 并两次 `autograd.grad`。其中的 einsum、乘法和拼接进入真实训练反向，不能把训练链称为纯 Triton。现有前向占比仍表明：仅用该高阶路径解释 V3 的全部变慢会忽略更大的前向成本。

## 4. 显存为何增加

**源码机制。** adapter 显式创建大小随边数和通道增长的1/0左输入，右输入需要 `transpose/reshape/contiguous`，HP autograd 又保存 left/right/K 以及反向需要的张量。Parameter 对象及参数数目没有因此增加；额外成本来自布局与中间量的存活。

**实测。** 独立推理峰值增加0.318982GiB；trace自身峰值增加约0.317GiB，其中峰值时 `aten::cat`（包括stack内部拼接）的存活分配增加约0.315GiB。完整训练独立峰值增加0.932540GiB；trace峰值增加约0.919GiB，`empty_like` 与 `cat` 分组分别增加约0.555和0.370GiB。

**归因判断。** 分配存活与源码新增输入/复制/保存吻合。分组可能包含其他嵌套操作，不能逐字节声称一个代码行独自解释全部峰值。也不能把不同阶段的静态张量尺寸相加冒充峰值。

## 5. 条件、控制与结论范围

- 同路线两侧已记录的参数、公共状态文件、图/标签、GPU、精度和内存策略一致。只有 V3 E/F 缺少恢复后完整 model/optimizer/EMA/scheduler 的逐组件状态哈希；训练该项齐全。该缺口保留，不把缺字段当 MATCH。
- 每次公共状态回放在计时外；调用内的缓存失效/重建计时。该结果不是持续缓存部署的吞吐结论。
- 原始未投影上游 V3 的默认 E/F OOM 留在旧报告；重算 E/F 1.4185s/8.762GiB、默认训练0.6415s/17.243GiB仅列绝对成本。其权重函数不同，不作本 canonical 的加速比分母。
- 现有 trace 没有提供 DRAM/L2 流量、寄存器溢出或完整硬件关键路径测量。因此不将 memory-bound、spill 或某个 occupancy 推测写成已测事实。
- 旧训练梯度/混合二阶 FAIL、表示与局部对象 HOLD 保持；本归因不授予正式数值或任务精度资格，也未开始下一步优化。

证据：[四路线报告](/public/home/sps-xia/bjxiao/o2-benchmark/analysis/o2-four-route-bottlenecks-20261125-r1/REPORT.zh-CN.md)、[V3专项trace解释](/public/home/sps-xia/bjxiao/o2-benchmark/analysis/v3-triton-hotspot-explanation-20261125-r1/README.zh-CN.md)、[峰值存活差分](/public/home/sps-xia/bjxiao/o2-benchmark/analysis/o2-four-route-bottlenecks-20261125-r1/data/PEAK-LIVE-DELTAS.csv)。
