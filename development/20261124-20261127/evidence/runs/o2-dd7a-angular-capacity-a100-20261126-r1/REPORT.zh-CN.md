# dd7a3db角向阶数、延迟/吞吐与A100 40G容量

状态：COMPLETE_WITH_RECORDED_HOLDS。源快照dd7a3db未修改；每个成功配置均有三个独立进程并重新核对中位数。

算子：UVW U=V=W=32、canonical/full；Gaunt双线性H=32。推理为forward，训练为forward+完整输入/权重VJP，所需混合二阶另验。固定N256测延迟、N4096测吞吐。它们是不同算子方程，只比较绝对成本。

整网：7层/128通道、l=m=1..8，固定seed独立结构模型和统一20×20系数生成网格；不是原checkpoint任务准确率。原l4/m2 checkpoint单列星标。固定100原子真实帧、原图/PBC/重复边/标签，推理含E/F/stress，训练含原loss、clip/AdamW/EMA/scheduler及真实力损失混合二阶。

精度和原门不变：能量输出/求和FP64，其他数学/累加/导数/误差FP32，禁TF32/AMP/half/CUDA Graph。数值失败、OOM、构建错误、超时及未执行分别保留，未给缺失项填零。详见COVERAGE.csv与AMD GATE-AUDIT回执。

容量指数搜索+二分，每个探测规模三个新进程，最大3/3成功规模及下一实际OOM上界在CAPACITY.csv。整网按100原子一图精确搜索；算子保留实际行数分辨率，通常约初始OOM上界1/32。容量是完整执行边界，不是所有N的新数值资格证明；不同batch容量边界吞吐另列，不与固定N混淆。

计时两次预热，容量一次预热，正式调用前在计量外恢复公共状态并清除无引用缓存；真实三步验收内不恢复状态/共享loss余切。图上点为三个进程值，中位数折线不代表独立科学样本置信区间。

![算子阶数扫描](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-dd7a-angular-capacity-a100-20261126-r1/figures/operator-angular-throughput-latency-capacity.png)

![整网阶数扫描](/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-dd7a-angular-capacity-a100-20261126-r1/figures/network-angular-throughput-latency-capacity.png)

|层级|族/宇称|阶数|模式|N或图数|状态|中位延迟ms|吞吐/s|allocated GiB|reserved GiB|
|---|---|---:|---|---:|---|---:|---:|---:|---:|
|operator|uvw/canonical|1|infer|256|PASS|2.74873|93134.1|0.0359502|0.0546875|
|operator|uvw/canonical|1|infer|4096|PASS|4.31171|949971|0.0991216|0.134766|
|operator|uvw/canonical|1|train|256|PASS|5.78448|44256.3|0.0715032|0.0878906|
|operator|uvw/canonical|1|train|4096|PASS|8.97922|456164|0.186402|0.201172|
|operator|uvw/canonical|2|infer|256|PASS|4.4808|57132.7|0.0369573|0.0527344|
|operator|uvw/canonical|2|infer|4096|PASS|7.18339|570204|0.104737|0.136719|
|operator|uvw/canonical|2|train|256|PASS|12.322|20775.8|0.0867314|0.0898438|
|operator|uvw/canonical|2|train|4096|PASS|22.539|181729|0.373658|0.396484|
|operator|uvw/canonical|3|infer|256|PASS|8.36465|30605|0.0383005|0.0546875|
|operator|uvw/canonical|3|infer|4096|PASS|11.6231|352400|0.109254|0.138672|
|operator|uvw/canonical|3|train|256|PASS|22.4054|11425.8|0.10608|0.130859|
|operator|uvw/canonical|3|train|4096|PASS|40.8661|100230|0.657228|0.6875|
|operator|uvw/canonical|4|infer|256|PASS|13.156|19458.9|0.0399485|0.0546875|
|operator|uvw/canonical|4|infer|4096|PASS|17.9839|227760|0.114259|0.138672|
|operator|uvw/canonical|4|train|256|PASS|36.6006|6994.43|0.134156|0.154297|
|operator|uvw/canonical|4|train|4096|PASS|68.0396|60200.2|1.03931|1.08203|
|operator|uvw/canonical|5|infer|256|PASS|18.9072|13539.8|0.0419631|0.0546875|
|operator|uvw/canonical|5|infer|4096|PASS|26.1953|156364|0.11841|0.160156|
|operator|uvw/canonical|5|train|256|PASS|53.9006|4749.48|0.171449|0.177734|
|operator|uvw/canonical|5|train|4096|PASS|91.5183|44756.1|1.5177|1.55469|
|operator|uvw/canonical|6|infer|256|PASS|25.6509|9980.16|0.0443439|0.0546875|
|operator|uvw/canonical|6|infer|4096|PASS|35.9359|113981|0.123537|0.164062|
|operator|uvw/canonical|6|train|256|PASS|72.9519|3509.16|0.215151|0.222656|
|operator|uvw/canonical|6|train|4096|PASS|127.834|32041.5|2.09509|2.13867|
|operator|uvw/canonical|7|infer|256|PASS|34.3784|7446.53|0.0470915|0.0664062|
|operator|uvw/canonical|7|infer|4096|PASS|47.4163|86383.8|0.129031|0.175781|
|operator|uvw/canonical|7|train|256|PASS|95.9954|2666.79|0.267093|0.28125|
|operator|uvw/canonical|7|train|4096|PASS|161.37|25382.6|2.77027|2.82031|
|operator|uvw/canonical|8|infer|256|PASS|44.4958|5753.35|0.050571|0.0683594|
|operator|uvw/canonical|8|infer|4096|PASS|61.0567|67085.2|0.136234|0.179688|
|operator|uvw/canonical|8|train|256|PASS|126.612|2021.93|0.327274|0.337891|
|operator|uvw/canonical|8|train|4096|PASS|208.407|19653.9|3.54309|3.58008|
|operator|uvw/full|1|infer|256|PASS|11.1616|22935.8|0.0392466|0.0527344|
|operator|uvw/full|1|infer|4096|PASS|13.9082|294502|0.107911|0.136719|
|operator|uvw/full|1|train|256|PASS|32.128|7968.13|0.12021|0.140625|
|operator|uvw/full|1|train|4096|PASS|53.053|77205.8|0.809571|0.820312|
|operator|uvw/full|2|infer|256|PASS|29.8977|8562.52|0.0455346|0.0546875|
|operator|uvw/full|2|infer|4096|PASS|33.4655|122395|0.121096|0.160156|
|operator|uvw/full|2|train|256|PASS|86.1449|2971.74|0.233248|0.255859|
|operator|uvw/full|2|train|4096|PASS|136.494|30008.6|2.25623|2.27148|
|operator|uvw/full|3|infer|256|PASS|59.4851|4303.6|0.0546923|0.0742188|
|operator|uvw/full|3|infer|4096|PASS|65.5174|62517.7|0.135259|0.183594|
|operator|uvw/full|3|train|256|PASS|168.146|1522.49|0.405858|0.433594|
|operator|uvw/full|3|train|4096|PASS|274.149|14940.8|4.48768|4.55273|
|operator|uvw/full|4|infer|256|PASS|95.2333|2688.13|0.0667801|0.0859375|
|operator|uvw/full|4|infer|4096|PASS|102.399|40000.3|0.15284|0.199219|
|operator|uvw/full|4|train|256|PASS|271.169|944.062|0.639015|0.652344|
|operator|uvw/full|4|train|4096|PASS|477.793|8572.76|7.5033|7.57617|
|operator|uvw/full|5|infer|256|PASS|143.569|1783.11|0.0822864|0.0996094|
|operator|uvw/full|5|infer|4096|PASS|153.847|26623.8|0.175792|0.201172|
|operator|uvw/full|5|train|256|PASS|412.394|620.765|0.933696|0.957031|
|operator|uvw/full|5|train|4096|PASS|744.457|5502|11.3202|11.3809|
|operator|uvw/full|6|infer|256|PASS|199.926|1280.47|0.100235|0.119141|
|operator|uvw/full|6|infer|4096|PASS|218.963|18706.4|0.199234|0.226562|
|operator|uvw/full|6|train|256|PASS|564.527|453.477|1.28697|1.29883|
|operator|uvw/full|6|train|4096|PASS|1090.62|3755.65|15.9179|15.9805|
|operator|uvw/full|7|infer|256|PASS|263.183|972.707|0.120626|0.140625|
|operator|uvw/full|7|infer|4096|PASS|295.366|13867.6|0.225118|0.255859|
|operator|uvw/full|7|train|256|PASS|785.245|326.013|1.69884|1.71484|
|operator|uvw/full|7|train|4096|PASS|1528.5|2679.75|21.3002|21.3691|
|operator|uvw/full|8|infer|256|PASS|341.185|750.326|0.144435|0.162109|
|operator|uvw/full|8|infer|4096|PASS|384.187|10661.5|0.254421|0.287109|
|operator|uvw/full|8|train|256|PASS|996.874|256.803|2.17437|2.19531|
|operator|uvw/full|8|train|4096|PASS|2085.37|1964.16|27.4682|27.5742|
|operator|bilinear/canonical|1|infer|256|PASS|0.197063|1.29908e+06|0.000366688|0.00195312|
|operator|bilinear/canonical|1|infer|4096|PASS|0.21333|1.92003e+07|0.00585985|0.0214844|
|operator|bilinear/canonical|1|train|256|PASS|0.53628|477363|0.000732899|0.00195312|
|operator|bilinear/canonical|1|train|4096|PASS|0.540675|7.57572e+06|0.0117192|0.0214844|
|operator|bilinear/canonical|2|infer|256|PASS|0.200725|1.27538e+06|0.000826836|0.00195312|
|operator|bilinear/canonical|2|infer|4096|PASS|0.420418|9.74269e+06|0.0131865|0.0214844|
|operator|bilinear/canonical|2|train|256|PASS|0.563218|454531|0.00165081|0.00195312|
|operator|bilinear/canonical|2|train|4096|PASS|1.31113|3.12403e+06|0.02637|0.0410156|
|operator|bilinear/canonical|3|infer|256|PASS|0.211252|1.21182e+06|0.0014801|0.00195312|
|operator|bilinear/canonical|3|infer|4096|PASS|0.81669|5.01537e+06|0.0234528|0.0410156|
|operator|bilinear/canonical|3|train|256|PASS|0.738008|346880|0.00294495|0.00390625|
|operator|bilinear/canonical|3|train|4096|PASS|1.7774|2.30449e+06|0.0468903|0.0605469|
|operator|bilinear/canonical|4|infer|256|PASS|0.307635|832154|0.00234747|0.00390625|
|operator|bilinear/canonical|4|infer|4096|PASS|2.46574|1.66116e+06|0.0366797|0.0429688|
|operator|bilinear/canonical|4|train|256|PASS|0.893226|286602|0.00463629|0.00585938|
|operator|bilinear/canonical|4|train|4096|PASS|5.82271|703452|0.0733008|0.0839844|
|operator|bilinear/canonical|5|infer|256|PASS|0.635949|402548|0.00346994|0.0214844|
|operator|bilinear/canonical|5|infer|4096|PASS|7.66573|534326|0.0529084|0.0546875|
|operator|bilinear/canonical|5|train|256|PASS|1.31175|195159|0.00676584|0.0214844|
|operator|bilinear/canonical|5|train|4096|PASS|17.541|233510|0.105643|0.107422|
|operator|bilinear/canonical|6|infer|256|PASS|1.08857|235170|0.00492477|0.0214844|
|operator|bilinear/canonical|6|infer|4096|PASS|14.3269|285895|0.072216|0.078125|
|operator|bilinear/canonical|6|train|256|PASS|2.63606|97114.5|0.00941086|0.0214844|
|operator|bilinear/canonical|6|train|4096|PASS|36.6413|111786|0.143993|0.154297|
|operator|bilinear/canonical|7|infer|256|PASS|2.55656|100134|0.00683594|0.0214844|
|operator|bilinear/canonical|7|infer|4096|PASS|37.6973|108655|0.0947266|0.0957031|
|operator|bilinear/canonical|7|train|256|PASS|4.46617|57319.8|0.0126953|0.0214844|
|operator|bilinear/canonical|7|train|4096|PASS|65.4171|62613.6|0.188477|0.189453|
|operator|bilinear/canonical|8|infer|256|PASS|4.0229|63635.7|0.0093956|0.0195312|
|operator|bilinear/canonical|8|infer|4096|PASS|57.6467|71053.5|0.120632|0.142578|
|operator|bilinear/canonical|8|train|256|PASS|10.2978|24859.6|0.0168114|0.0195312|
|operator|bilinear/canonical|8|train|4096|PASS|115.993|35312.6|0.239285|0.265625|
|network|network/canonical|4,m2 anchor|infer|1|PASS|383.372|260.843|1.49324|2.00781|
|network|network/canonical|4,m2 anchor|train|1|PASS|953.834|104.84|11.3395|11.9805|
|network|network/canonical|1|infer|1|PASS|162.232|616.4|0.465943|0.542969|
|network|network/canonical|1|train|1|PASS|428.763|233.229|3.06228|3.43164|
|network|network/canonical|2|infer|1|PASS|203.262|491.976|0.740583|0.90625|
|network|network/canonical|2|train|1|GATE_HOLD|—|—|—|—|
|network|network/canonical|3|infer|1|PASS|291.2|343.406|1.12168|1.36523|
|network|network/canonical|3|train|1|PASS|758.282|131.877|8.37089|8.8125|
|network|network/canonical|4|infer|1|PASS|461.092|216.876|1.61908|1.96289|
|network|network/canonical|4|train|1|GATE_HOLD|—|—|—|—|
|network|network/canonical|5|infer|1|PASS|775.522|128.945|2.25024|2.71289|
|network|network/canonical|5|train|1|PASS|1949.26|51.3015|17.1181|17.6738|
|network|network/canonical|6|infer|1|PASS|1207.67|82.8041|3.03106|3.60938|
|network|network/canonical|6|train|1|GATE_HOLD|—|—|—|—|
|network|network/canonical|7|infer|1|PASS|1924.29|51.9671|3.9755|4.68945|
|network|network/canonical|7|train|1|GATE_HOLD|—|—|—|—|
|network|network/canonical|8|infer|1|PASS|3188.14|31.3663|5.10819|5.94141|
|network|network/canonical|8|train|1|GATE_HOLD|—|—|—|—|

全部原始三次值见MEASUREMENTS.csv；每个试次RESULT/日志路径和原始结果均在measurements/。每个容量点上下界、失败状态及边界吞吐见CAPACITY.csv。

## 流程参考

制图遵循scientific-visualization技能：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. [当前arXiv记录](https://arxiv.org/abs/2609.00065)。内部分析图，不声明投稿规范或自动可访问性认证。
