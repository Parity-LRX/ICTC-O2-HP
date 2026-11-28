import json,os
from pathlib import Path
ROOT=Path(os.environ['O2_KROOT']);WS=ROOT.parents[1];assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
def read(p):return json.loads(Path(p).read_text())
p=read(Path((ROOT/'PERFORMANCE-AUDIT.txt').read_text().strip())/'RESULT.json');n=read(ROOT/'NUMERICAL-SUMMARY.json');assert p['status']=='PERFORMANCE_AUDIT_PASS'
by={(r['mode'],r['backend']):r for r in p['workloads']};c={r['mode']:r for r in p['comparison'] if r['backend']=='candidate'}
text=f'''2026-11-26 bounded双线性融合完成，候选`{(ROOT/'CANDIDATE-HEAD.txt').read_text().strip()[:7]}`，嵌套仓库ICTC-O2-HP分支codex/bounded-bilinear-v5（未推送）。AMD通过原legacy/isolated/train-only各自真实损失的三步、完整VJP/混合二阶和原生optimizer/EMA/scheduler更新；实际84组/504局部角色0ULP，边界/通用6组通过。每次Bilinear最多完整区/尾区两次Triton发射，直接写完整输出，去掉64行Python循环、块列表与拼接。按全部6种递归系数轴置换保留12种原生FP32归约计划；k为原本不训练的角向缓冲，原292活动权重仍训练。原8324通道核、loss/optimizer/精度/原门不变，不共享余切、不中途恢复三步状态。

AMD通过后同A100单卡三后端串行测量：E/F {by['ef','candidate']['median_s']:.6f}s、训练{by['train','candidate']['median_s']:.6f}s，对8324耗时{c['ef']['time_change_vs_baseline_percent']:+.2f}%/{c['train']['time_change_vs_baseline_percent']:+.2f}%，对Torch{c['ef']['time_change_vs_native_percent']:+.2f}%/{c['train']['time_change_vs_native_percent']:+.2f}%；allocated {by['ef','candidate']['allocated_GiB']:.6f}/{by['train','candidate']['allocated_GiB']:.6f}GiB，reserved {by['ef','candidate']['reserved_GiB']:.6f}/{by['train','candidate']['reserved_GiB']:.6f}GiB。速度与显存分开判断；ICTC GPU类别现包含新双线性子图，不与旧图份额直接当倍率。完整ULP/边界/形状/性能见[报告](runs/o2-bounded-bilinear-a100-20261126-r1/REPORT.zh-CN.md)，恢复见[封存](runs/o2-bounded-bilinear-a100-20261126-r1/FINAL-SNAPSHOT.json)。8324原快照及历史FAIL/HOLD保持；本轮首版b6ed三步73/30/55 FAIL、9b7三步0/171/59 FAIL也完整保留。三步通过不外推长期轨迹一致。
'''
for name in ('AGENTS.md','HANDOFF.md','OPERATOR_STATUS.md'):
    path=WS/name;first,_,rest=path.read_text().partition('\n');path.write_text(first+'\n\n'+text+'\n'+rest)
arch=WS/'architectures/equiformerv3/BOUNDED-BILINEAR-20261126-r1.zh-CN.md'
arch.write_text('# bounded双线性融合结果\n\n'+text.replace('(runs/','(../../runs/')+'\n实现、归约计划与实际调用证据以独立run完整报告为准。\n')
path=WS/'architectures/equiformerv3/README.zh-CN.md';first,_,rest=path.read_text().partition('\n')
path.write_text(first+'\n\n最新：[bounded双线性与完整高阶融合](BOUNDED-BILINEAR-20261126-r1.zh-CN.md)。各自真实损失的原三步经AMD通过，耗时/显存独立列示，8324及全部历史FAIL保留。\n'+rest)
with (ROOT/'PROGRESS.zh-CN.md').open('a') as f:f.write('\n最终终态：\n\n'+text)
print(text,flush=True)
