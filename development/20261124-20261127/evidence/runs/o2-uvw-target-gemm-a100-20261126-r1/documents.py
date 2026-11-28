"""Refresh current workspace status from finalized numerical/performance receipts."""
import json,os
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);WS=ROOT.parents[1]
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
def read(p):return json.loads(Path(p).read_text())
p=read(Path((ROOT/'PERFORMANCE-AUDIT.txt').read_text().strip())/'RESULT.json')
n=read(ROOT/'NUMERICAL-SUMMARY.json');assert p['status']=='PERFORMANCE_AUDIT_PASS'
by={(v['mode'],v['backend']):v for v in p['workloads']}
change={v['mode']:v for v in p['comparison'] if v['backend']=='candidate'}
text=f'''2026-11-26求导目标与8项FMA候选`{(ROOT/'CANDIDATE-HEAD.txt').read_text().strip()[:7]}`完成原独立三步AMD完整验收：legacy/isolated/train-only各自真实损失，原292梯度、实际force-only混合、原生optimizer/EMA/scheduler及更新后输出全部0FAIL/0ULP；112组×6实际角色与通用9组/62角色通过，EF/VJP按原门通过（后两者仍有门内非零误差）。每次求力剪去28个未消费K伴随，EF长K权重GEMM28→0、训练84→56。最终256×1024长归约采用共享内存CUDA SIMT 8项FMA的32×64 tile，其他形状保留Triton；四链/合并树、完整二阶、原loss/optimizer/精度/阈值不变，不冻结权重/共享余切。c05bc7b及旧202等全部FAIL/HOLD保留，三步通过不等于长期轨迹一致。

最终同A100单卡串行四后端配对：E/F {by['ef','candidate']['median_s']:.6f}s、训练{by['train','candidate']['median_s']:.6f}s，对c05耗时分别{change['ef']['time_change_vs_c05_percent']:+.2f}%/{change['train']['time_change_vs_c05_percent']:+.2f}%，对Torch{change['ef']['time_change_vs_native_percent']:+.2f}%/{change['train']['time_change_vs_native_percent']:+.2f}%；allocated {by['ef','candidate']['allocated_GiB']:.6f}/{by['train','candidate']['allocated_GiB']:.6f}GiB，reserved {by['ef','candidate']['reserved_GiB']:.6f}/{by['train','candidate']['reserved_GiB']:.6f}GiB。训练基本持平，显存无收益；单列256×1024约1.042→0.895ms（减14%）未变成明显训练墙钟收益。EF不运行新FMA专核，mask/final差值不归因于该核。速度/显存分开判断，不授予2×/5×/5×目标。完整数字、ULP、失败性能候选和每步快照见[runs/o2-uvw-target-gemm-a100-20261126-r1/REPORT.zh-CN.md](runs/o2-uvw-target-gemm-a100-20261126-r1/REPORT.zh-CN.md)、[恢复入口](runs/o2-uvw-target-gemm-a100-20261126-r1/FINAL-SNAPSHOT.json)。嵌套仓库ICTC-O2-HP分支codex/uvw-target-gemm-v4，未推送。
'''
for name in ('AGENTS.md','HANDOFF.md','OPERATOR_STATUS.md'):
    path=WS/name;original=path.read_text();first,sep,rest=original.partition('\n')
    path.write_text(first+'\n\n'+text+'\n'+rest)
arch=WS/'architectures/equiformerv3/TARGET-FMA8-20261126-r1.zh-CN.md'
arch.write_text('# 求导目标剪枝与分形状FMA最终记录\n\n'+text.replace('(runs/','(../../runs/')+'\n详细实现/原门/三步/耗时显存与调用库存见完整报告；最终SOURCE-AND-RECEIPTS封存经过独立解包核对，原c05归档也重新验证未变。\n')
path=WS/'architectures/equiformerv3/README.zh-CN.md';original=path.read_text();first,sep,rest=original.partition('\n')
path.write_text(first+'\n\n最新执行结果：[求导目标剪枝与分形状8项FMA](TARGET-FMA8-20261126-r1.zh-CN.md)。原门与原三步通过；最终共享内存CUDA特例与其他Triton路径分开登记，速度和显存见报告；c05及历史FAIL保持。\n'+rest)
with (ROOT/'PROGRESS.zh-CN.md').open('a') as f:f.write('\n最终已完成：\n\n'+text+'\n')
print(text,flush=True)
