# L1–L8 推理与训练修复验收

数值源码 `13c9328e2c5e1a177e6b301078904be595d42e97`，个人 fork 分支 `codex/angular-stability-v7`。

144项算子配置、240组局部完整导数回归、额外72组L7尾区回归，以及原L4/M2 checkpoint锚点与L1–L8的原完整三步均通过原门。

每阶额外20步各自真实损失连续训练和三次新进程、每进程10次E/F推理均由独立CPU-AMD审核通过。model/optimizer/EMA/scheduler/RNG在训练轨迹中连续，无中途恢复和共享真实损失余切。

|角向阶数|原E/F与VJP|原三步及真实混合导数|连续20步|三新进程×10次推理|训练最大relative-L2|推理最大relative-L2|
|---|---|---|---|---|---|---|
|L1|PASS|PASS|PASS|PASS|0|0|
|L2|PASS|PASS|PASS|PASS|3.0566693e-07|0|
|L3|PASS|PASS|PASS|PASS|2.84178583e-07|0|
|L4|PASS|PASS|PASS|PASS|2.1182602e-07|0|
|L5|PASS|PASS|PASS|PASS|2.04743046e-07|0|
|L6|PASS|PASS|PASS|PASS|9.03452565e-06|2.91362448e-06|
|L7|PASS|PASS|PASS|PASS|1.77256965e-07|0|
|L8|PASS|PASS|PASS|PASS|4.00223689e-06|8.81714584e-07|

L7修复仅将一项四行角向尾区计划改为54分区单链FP32，原真实操作数校准与完整导数/训练均另行验收。L8修复移除coverage.py导入冲突，并统一使用expandable_segments分配器及提前释放原反向本来会清理的旧梯度；默认分配器OOM回执保留。wheel已验证包含完整6576项计划。

原分量门2e-5+2e-5*abs(ref)、完整对象relative-L2<=1e-5和局部epsilon_max<=1e-5不变。ULP只作描述，逐角色完整误差/ULP文件及原始数组均保存在本run，轻量回执随fork快照发布。部分角色仍有门内非零误差。

此资格覆盖冻结输入与所列稳定性窗口。L1–L8使用固定seed独立结构模型；原L4/M2 checkpoint锚点分列。它不证明数据集预测精度、任意时长训练轨迹或长期MD稳定。新源码的速度、显存和容量须重新计量，不能继承旧dd7a3db的倍率。

本run全部8次单A100串行allocation（含失败）合计9414秒 / 2.615000000 GPUh，纳入原500 GPUh总账。

恢复入口：FINAL-SNAPSHOT.json、SOURCE.json、snapshots/continuity-120888/、GPU-ACCOUNTING.json；原v6及所有历史FAIL/OOM/HOLD均保持。

[个人fork](https://github.com/LeutOrca/ICTC-O2-HP) · [上游PR #2](https://github.com/Parity-LRX/ICTC-O2-HP/pull/2)
