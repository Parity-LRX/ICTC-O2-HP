#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=o2-uvw-figure-final
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-uvw-triton-a100-20261125-r1/results/final-figure-mpl-cache
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python figures/uvw_v3_hotspots.py
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python - <<'PY'
import json,os
from pathlib import Path
root=Path.cwd();fig=root/'figures';a=json.loads((fig/'UVW-V3-COMPARISON-r1.json').read_text());b=json.loads((fig/'UVW-V3-COMPARISON-r2.json').read_text())
assert a['workloads_sha256']==b['workloads_sha256']
report=(root/'REPORT.zh-CN.md').read_text().replace('UVW-V3-COMPARISON-r1.png','UVW-V3-COMPARISON-r2.png')
lines=report.splitlines()
for i,line in enumerate(lines):
    if line.startswith('配对完整字段：'):
        lines[i]='两模式三后端的参数、公共起始状态、完整model/optimizer/EMA/scheduler状态、图/标签、物理GPU UUID、4线程及精度条件一致。TF32、AMP、CUDA Graph均关闭。逐字段的true表示两侧相同，不表示启用了相应加速选项。详见`results/analysis-119186/PAIRED-AUDIT.json`。'
lines.insert(2,'**诊断结论：相对冻结HP，E/F和完整训练耗时分别下降到1/6.16和1/3.60，峰值allocated基本回到原生水平。完整训练仍有4个参数梯度及50个力损失混合二阶角色FAIL，不授予训练数值或正式加速资格。**\n')
(root/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
(root/'FIGURE-FINALIZATION.json').write_text(json.dumps({'job':os.environ['SLURM_JOB_ID'],'same_source_data_sha256':b['workloads_sha256'],'r1_retained':True,'r2_changes':'shorter titles; median labels placed beyond all raw samples; no numerical changes'},indent=2)+'\n')
print('FIGURE_R2_SAME_DATA_VERIFIED',flush=True)
PY
