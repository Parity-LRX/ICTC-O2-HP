# 三路线原生/ICTC诊断热点：第一步完成

已补齐V3 canonical、TECE canonical、TECE full parity的E/F推理与完整训练，以及每组Torch原生对照。主矩阵六组配对条件检查通过；数值FAIL/HOLD保留，未启动第二、第三步。

本轮4次allocation均为XHPC-A100单节点单GPU串行，共1.160833 GPUh（包含停止及所有异常）。

|路线|模式|同步墙钟中位数 Torch原生 / ICTC (s)|峰值allocated Torch原生 / ICTC (GiB)|kernel时间和 Torch原生 / ICTC (ms)|
|---|---|---:|---:|---:|
|v3|ef|3.558 / 23.335|7.585 / 7.904|1934.149 / 21753.284|
|v3|train|1.798 / 6.620|11.343 / 12.276|891.730 / 6160.535|
|tece-canonical|ef|3.648 / 2.867|24.012 / 24.012|3578.189 / 2715.022|
|tece-canonical|train|2.089 / 2.031|17.979 / 17.846|1622.081 / 1357.488|
|tece-full|ef|0.368 / 0.431|14.081 / 16.780|321.789 / 380.010|
|tece-full|train|0.231 / 0.451|4.330 / 4.757|195.159 / 237.379|

推理600原子，训练100原子，均batch=1。训练逐样本恢复同一个原生3步预热后的完整状态，恢复不计时。time/memory/trace为独立进程，3次代码预热、10次耗时采样、1次trace。kernel时间和按实际GPU事件统计，不能当同步墙钟；memcpy/memset单列。显存包含统一harness常驻对象。

![kernel分布](results/analysis-118587/kernel-distribution.png)

![耗时与显存](results/analysis-118587/time-memory.png)

[详细报告、原始V3与可选Eqx对照、完整分布入口](results/analysis-118587/REPORT.zh-CN.md) · [可搜索全kernel表](results/analysis-118587/inventory.html) · [原始10次耗时](results/analysis-118587/RAW-TIMES.csv) · [配对条件CSV](results/analysis-118587/PAIRED-CHECKS.csv) · [完整原生配置](results/analysis-118587/MODEL-CONFIGS.json)

![原生加速及相同ECE切口](results/analysis-118587/native-fast-comparison.png)

另列上游原生融合Eqx路径、保留Eqx的原生module route与相同module route的ICTC。主表的纯Torch数学参考不代替上游融合后端的整网性能参考；见详细报告及FAST-PAIRED-CHECKS。

额外同加速切口：full parity推理/训练两组条件一致并确认实际ICTC调用；canonical两组在ECE之前遇到packed Wigner与dense bmm接口不兼容，标为不可运行。未完成项的比较字段留空，不表示已测得不一致，也不填零耗时/吞吐。

本轮按公共状态回放。恢复本身不计时；派生权重布局缓存失效后若在call内重建，重建成本计入。本轮不声称持续缓存部署时的正式速度。全部kernel按实际事件完整统计；前向模块stage无法覆盖全部反向线程，未归因比例在CSV单列，不将其猜测为geometry。

[执行条件](CONDITIONS.zh-CN.md) · [资格边界](QUALIFICATION-BOUNDARY.zh-CN.md) · [实际GPU账目](GPU-ACCOUNTING.txt) · [实际trace禁Graph核查](TRACE-CUDA-GRAPH-AUDIT.csv) · [最终回执](FINAL-RECEIPT.json)

同路线内严格配对；不同路线保留原生架构/图/loss/optimizer/stress差异，跨架构只列绝对成本。V3 canonical是独立投影模型，原始上游V3另列绝对成本；TECE full为固定随机初始化结构性能。共享节点未锁时钟，配对核对物理GPU UUID。

完成分布后停止，等待用户审阅条件和分布，再决定后续工作。
