# EquiformerV3 canonical O2 最小结构改动设计

日期：2026-11-24。状态：`CPU_PROTOTYPE_CONVERTED / LOCAL29_PASS / CONSUMER_PARTIAL_FAIL / REPRESENTATION_HOLD`。

最新CPU回执：[29处与后续计算结果](CANONICAL-CPU-20261124.zh-CN.md)。转换118059、局部118078、后续计算118080、帧/真实能量力及混合VJP118089、汇总118094完成。原始/EMA各生成独立checkpoint，264个Parameter张量原样复制，28个正频率张量投影为K；局部29处前向/full VJP/混合均通过，S06/F06接入后续计算后失败，EMA另有A11混合失败；ALL29固定图及真实能量头E/F/force-parameter混合VJP通过。表示FP32门未审定保持HOLD，不授予完整后端/任务资格，无A100/GPU或训练。

用户确认采用 ICTC 现有 `parity="canonical"` O2 表示，允许必要的权重约束与重新训练。因此本设计取代未实施的“新增 full-parity V3 变体”方向；历史 full-parity 映射审计与失败记录保留。后续最新授权已恢复CPU转换/验证，每步保存源码、参数与结果快照；A100/GPU、训练和性能作业仍不提交。

## 1. 目标、身份与“最小”的含义

拟议独立模型名：`equiformerv3-ictc-o2-canonical-min-v1`。它不是冻结的 `equiformerv3-ictc-o2`，也不是原 checkpoint 的等价后端替换。

“最小”指保留原 7 个 Transformer block、L=4/M=2、C=128、attention hidden=32（乘法前64）、8 heads、每 head alpha/value=64/16、FFN hidden=512，以及图、RBF、完整径向 MLP、残差、norm、attention 和能量求导流程。**不增加通道，不新增自由宇称变量，不将网络宽度翻倍。** 这不是已经证明全局最小的代码补丁或任务精度损失最小的模型。

主要可训练结构变化集中在14个 SO2Linear 的正频率块：用固定反射基下合法的 O2 权重替代原任意 A/B；其余参数优先原样初始化。球面乘法保持原两因子结构，非训练系数须建立与 canonical 表示相容的明确参考。

两项冻结基线均不修改：

| 对象 | 冻结身份 |
| --- | --- |
| 算子 | `ictc-o2-hp` HEAD `e4763c76bf183499357e40cb39cc0934bb98779b` |
| 原整网 | `equiformerv3-ictc-o2`，原源码/输入 SHA `44cc668a579c2041a21ab9f276e2ca3ab25af59abc1f85cc687d5986adbbdfc2` |
| 官方源码 | `a7300c58df683dc99cb48027d5bfd4c887486c48` |
| 原 checkpoint | `omat24-mptrj-salex_gradient.pt`，SHA `429ccded98163122e7ba588d78e2441653f37f3e091e106c432807fe373c8f98` |

恢复入口：[冻结基线](../../baselines/icml-o2-stage1-20261124-r1/README.zh-CN.md)。独立算子的 full-parity 能力和优化主线继续保留；本网络只使用其 canonical 子集，不把网络覆盖冒称 full-parity 覆盖。

## 2. canonical O2 仍然包含反射

冻结 `native_o2.py::canonical_blocks` 的表示是 `0e` 加每个 m>0 的一份实二维 O2 irrep；不包含独立 `0o`，也不额外增加正频率的第二份副本。二维 irrep 可有很多 multiplicity copies，“一份类型”不等于每频率只能有一个通道。

在统一实 cos/sin 基中，取

`R_m(phi)=[[cos(m phi),-sin(m phi)],[sin(m phi),cos(m phi)]]`，`S=diag(1,-1)`。

canonical 特征同时满足旋转 R_m 和反射 S 的变换规则。原 SO2 块 `T=A tensor I+B tensor J`，`J=[[0,-1],[1,0]]`，只有旋转约束时 A/B 可独立；在相同 canonical 反射基中再要求 `T S_in=S_out T`，就只剩 `T=K tensor I`。因此 `canonical` 不能作为保留原任意 A/B 的开关。

节点采用自然球谐族：全局 degree l 的反演宇称为 `p_l=(-1)^l`，即 `0e,1o,2e,3o,4e`，每个仍有128通道。**全局1o/3o与局域0o不是同一概念**。在将边轴旋到局域+y后，自然球谐族的 m=0 副本都是局域0e；每个正频率在固定换基后都是上述 canonical 二维表示。

这是一套新模型的表示约束，不是为原特征事后补一个±1标签；也不引入由外部输入的“帧宇称”。实际帧、mask、PBC及边反转仍须独立核实。

## 3. 先校准固定基，再投影权重，不能直接删除源码中的B

设实际局域镜面 `S_x=diag(-1,1,1)` 固定+y，原自然 degree-l 表示为

`D_l^natural(S_x)=(-1)^l D_l(-S_x)`，其中 `-S_x` 是 proper rotation。

从冻结 Wigner、m-major 排列和各(l,m)实际两维基提取 `S_lm` 与 `R_lm(phi)`，构造一次性的实正交 `C_lm`，同时满足

`C_lm S_lm C_lm^-1=S`，`C_lm R_lm(phi) C_lm^-1=R_m(phi)`。

m=0核实作用为+1。C只依赖基约定、l/m和代码版本，不依赖坐标、训练权重或SVD；不能只对齐反射而漏掉旋转方向。CPU118059在实际冻结Wigner/m布局下对全部(l,m)选出C=I，并生成两套转换checkpoint；校准和残差原始数组见新run的CALIBRATION.pt/转换回执，不将此特例推广到任意基。

对一个正频率块，将原执行矩阵T转换到统一基：`T_c=C_out T C_in^-1`。定义新模型初始化

`T_new,c=(T_c+S_out T_c S_in)/2 = K tensor I`。

回到原基时 `T_new=C_out^-1 T_new,c C_in`。这是固定表示下线性空间的反射投影；在正交基及普通Frobenius度量下，它是到合法交织子空间的最近投影。**只保证这个局部矩阵距离性质，不保证整网函数误差或任务精度最小。**

若原T_c写作 `A_c tensor I+B_c tensor J`，则K=A_c，B_c被排除。由于不同(l,m)原基的相位/符号可能不同，K可能是原A/B的带符号线性组合；禁止未经校准就把所有原始 `fc.weight` 的B半行清零。m0及extra-m0均为0e，原m0的W和bias无需宇称投影。

新模型用packed K作为独立Parameter，梯度和优化器只覆盖合法自由度；不把禁用B作为待训练参数保留。原checkpoint和292个原Parameter来源记录不改写。初始化映射需另存逐key/切片/基转换/投影清单，不能`strict=False`静默跳过。

| 位置族 | 原m0参数，原样初始化 | 新canonical正频率K（不含二维I） |
| --- | --- | --- |
| A奇数：PRE | W `[928,1280]`，b `[928]`；含608 extra | m1 `[256,1024]`，m2 `[192,768]` |
| A偶数：VALUE | W `[640,160]`，b `[640]` | m1 `[512,128]`，m2 `[384,96]` |

原PRE的正频率 `fc.weight` 是 `[512,1024]`、`[384,768]`；原VALUE是 `[1024,128]`、`[768,96]`。统一基后的正频率独立实参数数目减半，不据此推断整层/整网速度翻倍。

## 4. 各族的宇称传递、激活和融合边界

### E：E00 embedding

原子类型、RBF、径向/type embedding都是0e。保留完整 `320→128→128→640` 径向MLP、LayerNorm/SiLU及source/target embedding；输出640重排成5×128个局域m0/0e。逆Wigner得到全局自然 `l,(-1)^l` 特征，再按原envelope、接收聚合与avg_degree处理。初始node `sphere_embedding` 仍是128个全局0e。

全部E00可训练参数可原样复制用于初始化；不能把末层640改成320。候选融合边界为完整径向生产者→m0列逆旋转→envelope/聚合，保留两方向和PBC重复边语义。

### P：7处PRE，A01/A03/…/A13

节点自然族→原norm→source/target gather/concat与完整径向调制→原旋转/截断→canonical C换基→受约束K→按需返回原球谐基。径向系数为0e，保持每l/通道的原m共享。

`[E,19,256]→[E,19,64]+extra608`。64个main通道仍为自然球面场的两组32，均不拆宇称。extra为512个alpha、64个scalar-SwiGLU输入、32个sphere gate，**全部局域0e**。m0 W/b原样初始化；m1/m2改为投影K并须微调。

融合可含原gather/径向乘法、固定C、K及输出布局；alpha分支、球激活、完整softmax依赖不能丢失。

### S：S01～S07 attention球激活

64 main通道分成两路自然球面场U/V，每路32；二者没有额外pseudoscalar族。原球面乘法 `U(omega)*V(omega)` 在反射时只改变球面自变量，输出仍为自然场，投回后为32个canonical通道。

scalar支路仍 `SiLU(s_1)*s_2`，两组32个s都是0e；另外32个0e经sigmoid门控球乘法输出。保持源码实际顺序：ToGrid→两因子乘法→FromGrid→0e gate→l0 scalar merge。这里无需新增奇激活、奇gate或新增Parameter。输出 `[E,19,32]` 接VALUE；不得把m0以外的所有特征叫作“偶标量”。

若在canonical坐标直接收缩，球面系数必须同步作 `K_c=C_out K(C_1^-1 tensor C_2^-1)`；也可在S边界decode到原球谐基，先保留原模块实现，再encode。不能只改PRE/VALUE而让S消费错误坐标。

### V：7处VALUE，A02/A04/…/A14

自然族32→受约束canonical K→128个自然族value通道，保持8 heads×16。m0 W/b原样初始化；正频率K由本层原checkpoint独立投影，不能共用PRE的反射SVD。输出经原0e attention权重/envelope、逆Wigner、receiver聚合、SO3投影和同类型残差。

融合可含固定换基、K和head布局，扩大到聚合须保留全邻居softmax分母。`ga.proj` 对每l只混通道，所有通道同一自然宇称，参数原样初始化。

### F：F01～F07完整FFN球面子图

保留 `SO3Linear128→512→ToGrid14×14→grid_linear_1:512→1024→两路512乘法/0e gate→grid_linear_2:512→512→FromGrid→l0 scalar merge→SO3Linear512→128`。

两因子都是自然场，所以乘积仍是自然场。`scalar_mlp.0.linear` 的1024个输出分成两个512个0e，SwiGLU合法；`grid_mlp.gating_linear` 从128个l0/0e生成512个0e，sigmoid合法；两个SO3Linear、grid linear、scalar MLP、gating linear的全部可训练参数均可原样初始化。这里无需把0o塞进原SiLU，也无需新增四路宇称乘法。

FFN在全局节点上操作，保留完整l≤4、|m|≤l，O2诊断须覆盖m≤4；不能把attention的M=2截断套到FFN。F01～F06后续计算包含残差→下一block norm1→径向/gather/旋转/PRE；F07后续计算为末端norm和能量头。跨边界融合不能跳过这些运算。

### 球面非训练系数：明确新参考与旧有限网格的区别

新架构数学参考采用自然球谐场乘法的Gaunt不变量结构：attention使用L=4/M=2保留子空间，FFN使用L=4/M=4。CPU原型以冻结网格每个(lout,l1,l2)块对对应Gaunt张量作最小二乘标量归一化校准，明确保存这些非训练标量，构成新模型的系数定义；不冒称已独立解析证明原求积规范完全一致。Gaunt非零路径满足角动量规则及 `l1+l2+lout` 为偶数，故自然全局宇称闭合；局域O2输出只保留canonical目标，无0o输出。

CPU已分别记录原有限网格 `K_frozen=P(T tensor T)`、新参考 `K_natural` 的归一化及完整差异；本轮最大系数绝对变化8.3446503e-7。原14×8/14×14网格仍仅可作为后续执行候选，未由此取得新参考实现资格。不得直接把原浮点禁路径泄漏清零后声称旧checkpoint等价，也不得把旧直接K高阶FAIL改判为通过。本轮数值对照为同一新K的direct/factorized收缩，以及K的packed/expanded线性执行，不把旧grid当作同一新函数的严格参考。

若系数/归一化合同尚未闭合，该位置HOLD。Gaunt选择规则不是冻结UVU/Radial全部路径已自动覆盖；须按实际乘法路径建映射，不能换成任意TP后继续声称同一新模型。

## 5. 29处逐项登记

下表P/S/V/F规则均包含上节完整定义；不是将多个位置合并计为一个kernel。机器清单：[CANONICAL-O2-SITES.tsv](CANONICAL-O2-SITES.tsv)。

| ID | 原模块 | 输入→输出宇称/类型 | 权重、球激活与checkpoint初始化 | 直接后续计算/融合终点 |
| --- | --- | --- | --- | --- |
| E00 | edge_degree_embedding | 0e→5×128局域0e→节点自然族128 | E：全部原样初始化；保留全MLP | node加和→block0.norm1 |
| A01 | blocks.0.ga.so2_linear_1 | canonical256→自然64+608个0e | P：m0复制，m1/m2投影K | S01及block0.alpha |
| S01 | blocks.0.ga.act | 自然64+96个0e→自然32 | S：两自然因子相乘，0e SwiGLU/gate；无Parameter | A02 |
| A02 | blocks.0.ga.so2_linear_2 | canonical32→自然128 | V：m0复制，m1/m2投影K | block0加权/逆旋转/聚合/proj/残差 |
| F01 | blocks.0.ffn | 节点自然128→512→128 | F：全部权重复制，球面系数另核；需整体微调 | block0残差→block1.norm1/A03 |
| A03 | blocks.1.ga.so2_linear_1 | canonical256→自然64+608个0e | P：m0复制，m1/m2投影K | S02及block1.alpha |
| S02 | blocks.1.ga.act | 自然64+96个0e→自然32 | S：两自然因子相乘，0e SwiGLU/gate；无Parameter | A04 |
| A04 | blocks.1.ga.so2_linear_2 | canonical32→自然128 | V：m0复制，m1/m2投影K | block1加权/逆旋转/聚合/proj/残差 |
| F02 | blocks.1.ffn | 节点自然128→512→128 | F：全部权重复制，球面系数另核；需整体微调 | block1残差→block2.norm1/A05 |
| A05 | blocks.2.ga.so2_linear_1 | canonical256→自然64+608个0e | P：m0复制，m1/m2投影K | S03及block2.alpha |
| S03 | blocks.2.ga.act | 自然64+96个0e→自然32 | S：两自然因子相乘，0e SwiGLU/gate；无Parameter | A06 |
| A06 | blocks.2.ga.so2_linear_2 | canonical32→自然128 | V：m0复制，m1/m2投影K | block2加权/逆旋转/聚合/proj/残差 |
| F03 | blocks.2.ffn | 节点自然128→512→128 | F：全部权重复制，球面系数另核；需整体微调 | block2残差→block3.norm1/A07 |
| A07 | blocks.3.ga.so2_linear_1 | canonical256→自然64+608个0e | P：m0复制，m1/m2投影K | S04及block3.alpha |
| S04 | blocks.3.ga.act | 自然64+96个0e→自然32 | S：两自然因子相乘，0e SwiGLU/gate；无Parameter | A08 |
| A08 | blocks.3.ga.so2_linear_2 | canonical32→自然128 | V：m0复制，m1/m2投影K | block3加权/逆旋转/聚合/proj/残差 |
| F04 | blocks.3.ffn | 节点自然128→512→128 | F：全部权重复制，球面系数另核；需整体微调 | block3残差→block4.norm1/A09 |
| A09 | blocks.4.ga.so2_linear_1 | canonical256→自然64+608个0e | P：m0复制，m1/m2投影K | S05及block4.alpha |
| S05 | blocks.4.ga.act | 自然64+96个0e→自然32 | S：两自然因子相乘，0e SwiGLU/gate；无Parameter | A10 |
| A10 | blocks.4.ga.so2_linear_2 | canonical32→自然128 | V：m0复制，m1/m2投影K | block4加权/逆旋转/聚合/proj/残差 |
| F05 | blocks.4.ffn | 节点自然128→512→128 | F：全部权重复制，球面系数另核；需整体微调 | block4残差→block5.norm1/A11 |
| A11 | blocks.5.ga.so2_linear_1 | canonical256→自然64+608个0e | P：m0复制，m1/m2投影K | S06及block5.alpha |
| S06 | blocks.5.ga.act | 自然64+96个0e→自然32 | S：两自然因子相乘，0e SwiGLU/gate；无Parameter | A12 |
| A12 | blocks.5.ga.so2_linear_2 | canonical32→自然128 | V：m0复制，m1/m2投影K | block5加权/逆旋转/聚合/proj/残差 |
| F06 | blocks.5.ffn | 节点自然128→512→128 | F：全部权重复制，球面系数另核；需整体微调 | block5残差→block6.norm1/A13 |
| A13 | blocks.6.ga.so2_linear_1 | canonical256→自然64+608个0e | P：m0复制，m1/m2投影K | S07及block6.alpha |
| S07 | blocks.6.ga.act | 自然64+96个0e→自然32 | S：两自然因子相乘，0e SwiGLU/gate；无Parameter | A14 |
| A14 | blocks.6.ga.so2_linear_2 | canonical32→自然128 | V：m0复制，m1/m2投影K | block6加权/逆旋转/聚合/proj/残差 |
| F07 | blocks.6.ffn | 节点自然128→512→128 | F：全部权重复制，球面系数另核；需整体微调 | block6残差→末端norm→energy_block |

## 6. checkpoint哪些能复用，哪些需要重新训练

“原样复制”仅指同一模块相同输入时的参数初始化；投影改变上游函数后，不保证整个网络的输出或任务精度保留。“重新训练”不等于所有权重从随机值开始。

| 原参数/状态 | 新模型初始化 | 后续训练处理 |
| --- | --- | --- |
| `sphere_embedding.weight` | 全部原样复制 | 可训练；输入仍128个0e |
| E00 source/target embedding、rad_func全部W/b、LN参数 | 全部原样复制 | 可训练，保留原网络深度/激活 |
| 各block径向MLP及atom-edge embedding | 全部原样复制 | 可训练；径向函数仍为0e |
| `blocks.i.ga.so2_linear_{1,2}.fc_m0.weight/bias` | 全部原样复制 | 可训练；含alpha/scalar/gate全部extra |
| `blocks.i.ga.so2_linear_{1,2}.so2_m_linear.{0,1}.fc.weight` | 基校准后投影、pack成K；不直接strict加载原形状 | K须训练/微调；被排除方向没有新自由参数，不重新训练被禁用B |
| `blocks.i.ga.alpha_norm`、`alpha_dot` | 全部原样复制 | 可训练，alpha/logit/softmax始终0e |
| `blocks.i.ga.proj.weight/bias` | 全部原样复制 | 可训练；每l同一自然宇称，通道混合合法 |
| `blocks.i.norm1/norm2`、末端`norm`的affine_weight/bias | 全部原样复制 | 可训练；原l0中心化、跨l平方统计和epsilon可保留 |
| FFN `so3_linear_1/2.weight/bias` | 全部原样复制 | 可训练 |
| FFN `scalar_mlp.0.linear.weight/bias` | 全部原样复制 | 可训练；两支均0e |
| FFN `grid_mlp.gating_linear.weight/bias`、`grid_linear_1/2.weight` | 全部原样复制 | 可训练；gate是0e，网格场为自然族 |
| `energy_block.linear_1/2.weight/bias` | 全部原样复制，输入仍全部l0/0e | 可训练，需适应改变后的特征分布 |
| S01～S07 | 没有原始Parameter | 不存在单独“重训激活权重”；训练两侧K及上游模块 |
| RBF常数、envelope、排列/旋转规范、degree平衡常数 | 原定义保留，逐项登记 | 非训练；实际mask/帧分支与导数仍待验证 |
| 球面T/P和K、canonical C及路径表 | 保存旧数组，按新数学参考生成/核验派生系数 | 非训练；不能用训练吸收非法反射路径 |
| optimizer/scheduler、EMA运行状态 | 不原样续接旧优化器；新packed参数需新状态 | 默认全部活动参数微调；新EMA从转换后的完整模型初始化 |

选择原始state_dict或完整EMA.shadow_params其中一套作为初始化来源，不能按模块混用；两套转换已分别生成身份并在新CPU进程严格重载。每套264个Parameter张量原样复制、28个正频率Parameter张量投影，不能将张量数量当作复用元素比例。没有必须新增随机初始化的可训练层；任务精度恢复仍需训练证据，不能承诺仅微调K就足够。

## 7. 后续计算与读出是什么意思

**后续计算**指接收当前层输出并继续处理的步骤。例如PRE输出一支进入球激活S，一支进入alpha归一化/logit/softmax；VALUE输出接着进行attention加权、逆旋转、聚合、proj和残差相加；F01输出经过残差和下一层norm后，进入A03前的gather/旋转。若改变当前层输出的坐标基，这些后续步骤也必须使用一致的表示，残差两边才能正确相加。

**读出（readout）**是把末端节点特征变为任务预测的最后一段。当前V3为：末端norm→取l0/128个0e→`energy_block.linear_1(128→512)`→SiLU→`linear_2(512→1)`→逐原子能量转FP64→按graph求和并保留原avg_num_nodes归一化/单位约定。再用 `F=-grad_R E` 求力。它不是另一个SO2Linear，不在29个逻辑位置里，但决定整网E/F是否正确。

这里的128个l0都是0e，所以原能量MLP的全部W/b可用于初始化。力损失训练还必须把坐标-所有活动参数的混合二阶传回K、径向、norm、FFN和能量头，不能只验证局部输出。

## 8. 接入ICTC时还需实现的接口，不把canonical等同现成可替换

attention各m的multiplicity分别与剩余l数相乘，节点128通道对应m0/m1/m2为640/512/384；不是 `canonical_blocks(M=2,C=128)` 那样各m固定128。l必须作为multiplicity身份保留。FFN m≤4的multiplicity也不均匀。

冻结 `FullParityPathWeightedTensorProductO2` 的cyclic/depthwise multiplicity及UVU的sharing规则不是本设计的任意dense K。未来应先写显式canonical O2 dense-intertwiner数学参考，以及S/F受约束球乘法的路径参考；再核对哪些现有ICTC核心可直接复用，哪些需要保持相同方程的布局/参数化扩展。不能通过改成cyclic/depthwise或任意UVU删去dense耦合来宣称接入。

K作用仍是受反射约束的O2线性子图；S/F是特定球乘法路径，三者不能仅按名字统一成同一个TP。Triton可融合固定基、布局、合法路径及伴随，是否更快/更省显存均待实测。

## 9. 后续验收边界

1. 先在新架构的PyTorch数学参考中校准固定C、投影所有14处K、构造S/F系数，完整登记29处及后续计算。验证旋转和反射、任意角度、m0与正频率、权重更新后仍满足结构约束；不能只测投影初始化。
2. 核验实际R(v)、图/PBC/重复边/半键两个方向、mask分支以及全局自然O(3)变换。代数表示设计不代替实际代码证据；跨分支失败保留HOLD，不改冻结baseline来过门。
3. 候选后端与**同一个新canonical模型**的参考使用同输入/转换权重/余切，验前向、全部活动参数及输入full VJP、坐标-参数混合二阶、整网E/F与真实力损失梯度。固定C不产生额外参数/坐标导数，原Wigner/径向几何链仍必须可微。
4. 能量输出/求和归约FP64，其余计算/累加/导数/统计FP32；禁TF32、AMP、内部半精度、CUDA Graph。O2前向/VJP的全对象归一化最大误差口径与整网/高阶口径分开；新表示残差或系数合同未审定处HOLD，不借用论文FP64残差作为FP32门限。
5. K的derived布局缓存随参数更新/load/EMA失效；几何缓存随坐标/cell/图/帧/mask和autograd generation失效。固定C/K系数缓存记录规范、L/M、dtype/device与代码版本。
6. 任务精度在后续阶段用公开既有E/F标签，先审split、checkpoint训练重叠、单位/能量修正/力mask。投影初始化只支持结构性能或实现一致性，不代表任务模型已恢复精度。不新增DFT，不因允许重训而立即启动训练。
7. 加速比只在新canonical参考与新canonical后端之间计算；与旧V3的任务精度/绝对成本可另表比较，不能称同checkpoint等价加速。速度和显存/容量分别验收，每项保存快照。

## 10. 来源与当前证据范围

- 原V3论文：[§3.1及§4.3](https://arxiv.org/html/2604.09130v1)，原文以SE(3)为范围，球激活为自然球面场乘法。
- 表示约定：[e3nn irreps](https://docs.e3nn.org/en/stable/api/o3/o3_irreps.html)、[scalar/gate parity](https://docs.e3nn.org/en/stable/api/nn/nn_gate.html)。这些支持表示与非线性的约束，不是本设计已实现的证据。
- 冻结源码：[canonical/full blocks](../../baselines/icml-o2-stage1-20261124-r1/payload/operator/upstream/official/src/ictc_o2_hp/native_o2.py)、[原SO2参数](../../baselines/icml-o2-stage1-20261124-r1/payload/network/upstream/experimental/models/equiformer_v3/so2_ops.py)、[原activation](../../baselines/icml-o2-stage1-20261124-r1/payload/network/upstream/experimental/models/equiformer_v3/activation.py)、[原attention/FFN](../../baselines/icml-o2-stage1-20261124-r1/payload/network/upstream/experimental/models/equiformer_v3/transformer_block.py)、[原readout](../../baselines/icml-o2-stage1-20261124-r1/payload/network/upstream/experimental/models/equiformer_v3/output_block.py)。
- 历史同checkpoint联合映射仍见[几何相关CPU终态](GEOMETRY-DEPENDENT-MAPPING-20261124.zh-CN.md)。本设计允许新权重约束/重训，不利用这种范围变化改判任何旧FAIL。

当前已有独立CPU模型原型、两套转换checkpoint及29处/后续计算导数回执，见[CPU终态](CANONICAL-CPU-20261124.zh-CN.md)与[执行包](../../runs/o2-v3-canonical-cpu-20261124-r1/README.zh-CN.md)。后续计算部分FAIL及表示/几何HOLD保留；没有Triton/A100后端、训练、任务精度或性能资格。后续仍按[新canonical工作计划](../../analysis/equiformerv3-optimization-plan-20261124-r1/CANONICAL-MIN-WORKPLAN.zh-CN.md)分层推进。
