import json,os
from pathlib import Path
import analyze
ROOT=Path(__file__).resolve().parent
folder=ROOT/'results/a100-118576/v3-ef-native-trace'
target=ROOT/('prepared/parser-check-'+os.environ['SLURM_JOB_ID'])
result=analyze.parse_trace(folder,target)
print(json.dumps({k:result[k] for k in ('kernel_calls','kernel_ms','phase_coverage_percent','stage_coverage_percent','phase','family')},ensure_ascii=False),flush=True)
