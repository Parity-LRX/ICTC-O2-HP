import json,os
from pathlib import Path
from support import *
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
audit=Path((ROOT/'AUDIT-RESULT.txt').read_text().strip());supp=Path((ROOT/'SUPPLEMENT-RESULT.txt').read_text().strip())
ab=json.loads((audit/'ABLATIONS.json').read_text());mem=json.loads((audit/'MEMORY.json').read_text());extra=json.loads((supp/'RESULT.json').read_text())
first=json.loads((audit/'FIRST-COMPUTE.json').read_text());live=next(x for x in mem if x.get('snapshot',{}).get('at_failure'))
lines=['# 新阶数OOM、网格发射与训练数值失败的因果诊断',
       '', '最终证据来自单A100串行120614→120617→120636，AMD120612/120618/120648审核。原dd7a3db、旧扫描回执、旧FAIL及loss/精度/阈值保持；本轮没有提交后端修复或授予新训练资格。',
       '', '## L8：高存活量与缓存碎片共同触发原参考侧OOM',
       '', f"精确触发点为 `{live['failed_op']['autograd_metadata'].get('cause_module')}` 的 `IndexSelectBackward0 → {live['failed_op']['op']}`。径向按l的 `[5252,9,256]` 展开到81角向槽，求力(create_graph=True)反向的确定性index_add申请约416MiB。",
       '', f"异常抛出前active allocated={live['snapshot']['active_allocated_bytes']/2**30:.6f}GiB，inactive={live['snapshot']['inactive_bytes']/2**30:.6f}GiB，最大单块inactive={live['snapshot']['largest_inactive_block']/2**20:.3f}MiB，小于申请。申请失败算子不是全部峰值的归属。",
       '', '| 异常前存储块对应的最后记录视图形状 | 块数 | allocated GiB |', '|---|---:|---:|']
for r in live['snapshot'].get('shapes',[])[:10]:lines.append(f"| `{r['shape']}` | {r['blocks']} | {r['allocated_bytes']/2**30:.6f} |")
lines += ['', '按allocator唯一存储地址去重；同一存储的视图不重复累计。形状说明宽边张量的存活规模，不能仅凭形状把所有存储都归为径向权重。完整分配栈、未映射块及原始allocator快照保留。']
operations=json.loads((ROOT/'results/PEAK-LIVE-OPERATIONS.json').read_text())
lines += ['', '| 对应视图/生成操作 | 当时autograd节点 | 块数 | allocated GiB |','|---|---|---:|---:|']
for v in operations[:8]:lines.append(f"| `{v['shape']}` / `{v['op']}` | `{v['node']}` | {v['blocks']} | {v['bytes']/2**30:.6f} |")
lines += ['', '例如21个MulBackward生成的宽边乘法结果合计约8.53GiB，另有按l展开的径向权重、旋转BMM伴随、Slice反向累加以及拼接的宽边存储；不是一个双线性外积占据整个峰值。',
          '', '| L8对照 | 状态 | allocated峰值 GiB |', '|---|---|---:|']
for r in mem:
    if r.get('final_memory'):lines.append(f"| `{Path(r['folder']).name}` ({Path(r['folder']).parent.name}) | {r['status']} | {r['final_memory']['peak_allocated']/2**30:.6f} |")
lines += ['', '完整保留前置混合导数诊断，仅在train_step前gc.collect/empty_cache即可完成；训练前CPU/CUDA/NumPy/Python RNG逐项一致。清理没有改变活动张量、权重、loss或导数，只释放无引用缓存。旧协议的OOM保持，因此原因是接近40G上限的大量存活边张量，加上前置诊断形成的缓存分块；不是仅由参数量或O2独立算子决定的容量。',
          '', f"AMD完整数组对照：`{json.dumps(extra['L8'],ensure_ascii=False)}`。这是单步诊断补充，不把一次训练步外推为L8独立三步资格。",
          '', '## 双线性Triton：CUDA网格第二维越界',
          '', '设备实际maxGridSize=(2147483647,65535,65535)。原main grid=(ceil(OA/32),ceil(main_rows*H/32))；H=32时，65535行仍通过，65536行失败。原1600原子/84032边配置的grid.y=84032，也超过65535。H=512时限制更早，不能混称为显存OOM。',
          '', '仅在独立诊断副本中交换grid.x/y及对应program_id索引，其他算术完全保持；原核可运行区间与副本输出逐元素一致，越界点副本可运行并与原bounded参考完整数组对照。修正只证明发射问题，未证明对应整网规模的显存或完整导数容量。',
          '', '## L2/L4/L6/L7：最早非零计算差与实际越门原因',
          '', '| L | 组合 | 第1/2/3步FAIL角色数 |', '|---:|---|---|']
for r in ab:lines.append(f"| {r['order']} | {r['variant']} | {' / '.join(str(len(s['vs_original_native']['failures'])) for s in r['steps'])} |")
lines += ['', '所有组合各用真实loss连续推进三步，无共享余切、无中途状态恢复。完整native和candidate分别重现旧数组；两输入/参数梯度及实际force-only混合对象按原门判断。首个非零局部误差不等于导致最终FAIL的充分原因：L2/L4仅双线性即可失败，L6需要本次两者共同作用，L7通道单独也会失败。',
          '', '| L | 类型 | 最早定位的模块 | 操作数形状 | same-input relative-L2 |', '|---:|---|---|---|---:|']
for r in first:lines.append(f"| {r['order']} | {r['kind']} | `{r['site']}` | `{r['shapes']}` | {r['metric']['relative_l2']:.8g} |")
lines += ['', '上述定位使用原step0训练前RNG和原初态；每个候选算子与Torch在完全相同操作数上比较，额外参考不进入真实loss或更新。原角向系数布局并非统一连续布局，新阶数会进入默认归约路径。',
          '', '| L | 系数stride | 相同操作数上与native前64行完全相同的诊断归约计划 |', '|---:|---|---|']
axis_meta=json.loads((Path(extra['axis_source'])/'RESULT.json').read_text())['rows']
for r in extra['axis_rows']:
    stride=next(v['original_k_stride'] for v in axis_meta if v['order']==r['order'])
    plans=[{k:p[k] for k in ('swapped_reduction_axes','parts','four_chains')} for p in r['exact_plans']]
    lines.append(f"| {r['order']} | `{stride}` | `{plans}` |")
lines += ['', '本次精确匹配计划均无需交换角向轴；关键差别是分区数/FMA合并，而非索引错误。默认一分区四链分别不同于这里测得的四分区单链、六分区四链、三分区四链。归约探针的输入、系数、乘法FP32和权重均不变；AMD从完整计划输出独立复算误差与ULP。相同操作数的反例和计划对照把局部差异定位到FP32求和执行顺序，而非借用固定loss余切掩盖训练失败。前64行计划探针不替代尾区、全部6种递归系数排列或完整高阶训练验收。',
          '', '原失败均落在更新后零初始化bias及EMA的relative-L2，逐分量门通过。首步lr=0；后续lr约5.546927e-9、1.109385e-8。全梯度张量的门内差掩盖了少量小分量相对差；Adam对每分量的m/(sqrt(v)+eps)归一化把它们传到接近零的新bias，EMA随之传播。',
          '', '具体例：L7第2步blocks.2.ffn.so3_linear_2.bias[96]，原始梯度native=-3.0212104320526123e-6、candidate=-2.8815120458602905e-6；Adam方向=-0.1698719412/-0.1662175804；更新相差-2.027056300590857e-11，而更新后全bias norm约6.2511489e-8，R=3.242694e-4。完整梯度角色仍通过原门。',
          '', '将各自保存的真实梯度交给原生CUDA clip/AdamW/EMA，4阶×2后端×3步共24次均逐元素精确重放实际更新和EMA，因此没有optimizer实现错误证据。原20个FAIL不改判，不把绝对差很小当作豁免。',
          '', '## 证据与恢复边界',
          '', f'- 原始数组复核：`{ROOT}/results/cpu-120612/RESULT.json`。',f'- 完整消融/内存审核：`{audit}`。',f'- 归约计划/缓存控制AMD复算：`{supp}`。',
          f'- 标准执行合同：`{ROOT}/CONTRACT.zh-CN.md`。',
          '- GPU用量见GPU-ACCOUNTING.json；不重复计入旧120441。控制归档经独立解包校验，原始大数组与allocator快照由RAW-RESULT-MANIFEST.json逐文件引用。',
          '- 后续若修复，应分别处理网格索引、峰值宽边张量/缓存策略和新形状全部递归归约计划；仍须保留完整导数并重验各自真实三步。本轮只诊断，不自动授予修复资格。','']
(ROOT/'REPORT.zh-CN.md').write_text('\n'.join(lines))
print('REPORT_WRITTEN',ROOT/'REPORT.zh-CN.md',flush=True)
