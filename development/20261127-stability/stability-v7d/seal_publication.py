"""Publish lightweight successful receipts without copying raw tensors to Git."""
import json,os,subprocess,sys
from pathlib import Path
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
root=Path(__file__).resolve().parent;ws=root.parents[1]
subprocess.run([sys.executable,'-u',str(root/'report_final.py')],check=True)
result=json.loads((root/'FINAL-SNAPSHOT.json').read_text());assert result['status']=='PASS'
readme=ws/'ICTC-O2-HP/development/20261127-stability/README.md'
text=readme.read_text()
pending="GPU job 120889 has started the additional 20 continuous training steps per order, followed by three fresh inference processes with ten calls each, using each backend's own updated weights. Independent per-order AMD audits are required before a bounded stability claim. These are bounded validation windows, not an indefinite trajectory guarantee. Historical failures and all earlier source tags remain unchanged. Performance/capacity must be remeasured for the new source; older `dd7a3db` results do not transfer automatically."
passed="All L1–L8 structural networks also passed 20 continuous own-loss training steps and three fresh inference processes of ten calls each, using each backend's own updated raw weights. Independent per-order CPU-AMD audits passed all 162 comparison groups per order, including state/RNG continuity, complete gradients and native optimizer/EMA/scheduler updates. [The final report](stability-v7d/REPORT.zh-CN.md) and [recoverable receipt](stability-v7d/FINAL-SNAPSHOT.json) contain the per-order errors and audit IDs. This is a bounded validation window, not an indefinite trajectory or task-accuracy guarantee. Historical failures and all earlier source tags remain unchanged. Performance/capacity is a separate measurement run for this source; older `dd7a3db` results do not transfer automatically."
assert text.count(pending)==1 or text.count(passed)==1
progress=root/'PROGRESS.zh-CN.md';progress_text=progress.read_text()
marker='## 完整稳定性窗口终态'
if marker not in progress_text:
    summary='\n\n'+marker+'\n\n数值源码13c9328：L1–L8均通过各自20步真实损失连续训练和三新进程各10次E/F推理，独立AMD每阶162组全部PASS。当前结果以REPORT.zh-CN.md和FINAL-SNAPSHOT.json为准；以下保留各推进时点的过程记录。\n\n'
    title,body=progress_text.split('\n',1);progress.write_text(title+summary+body)
files=[root/name for name in ('REPORT.zh-CN.md','FINAL-SNAPSHOT.json','GPU-ACCOUNTING.json','GPU-SACCT.txt','SCONTROL-120889.txt',
                             'gate-audits/120902/RESULT.json','local-audits/120902/RESULT.json','training-window-audits/120928/RESULT.json',
                             'gate-captures/120842/network-L8-native/RESULT.json','gate-captures/120842/network-L8-native-repeat/RESULT.json')]
files.extend(root.glob('*.py'))
files.extend(Path(row['receipt']) for row in result['bounded_stability'])
subprocess.run([sys.executable,'-u',str(root/'snapshot_stage.py'),'stability-v7d',*map(str,files)],check=True)
readme.write_text(text.replace(pending,passed))
print('QUALIFIED_PUBLICATION_SNAPSHOT_READY',result['source_commit'],flush=True)
