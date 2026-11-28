import os,subprocess,sys
from pathlib import Path
from candidate_sources import qualified_sources
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
qualified_sources()
subprocess.run([sys.executable,str(Path(__file__).parent/'preflight.py'),sys.argv[1]],check=True)
