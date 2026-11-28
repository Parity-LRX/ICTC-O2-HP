"""All measurements serial on the allocated single A100."""
import os,subprocess,sys
from context import ROOT
out=ROOT/'results'/('performance-a100-'+os.environ['SLURM_JOB_ID']);out.mkdir()
for mode in ('ef','train'):
    for repetition in range(3):
        order=('native','v2','candidate') if repetition%2==0 else ('candidate','v2','native')
        for backend in order:
            path=out/f'{mode}-{backend}-time-{repetition}'
            subprocess.run([sys.executable,'-u',str(__import__('pathlib').Path(__file__).parent/'performance.py'),mode,backend,'time',str(path)],check=True);print(path.name,flush=True)
    for backend in ('native','v2','candidate'):
        path=out/f'{mode}-{backend}-memory'
        subprocess.run([sys.executable,'-u',str(__import__('pathlib').Path(__file__).parent/'performance.py'),mode,backend,'memory',str(path)],check=True);print(path.name,flush=True)
    for backend in ('native','candidate'):
        path=out/f'{mode}-{backend}-trace'
        subprocess.run([sys.executable,'-u',str(__import__('pathlib').Path(__file__).parent/'performance.py'),mode,backend,'trace',str(path)],check=True);print(path.name,flush=True)
subprocess.run([sys.executable,'-u',str(__import__('pathlib').Path(__file__).parent/'gemm_measure.py')],check=True)
(ROOT/'PERFORMANCE-RESULT.txt').write_text(str(out)+'\n')
