"""Prepare a separate, qualification-gated measurement run on CPU-AMD."""
import ast,json,os,shutil
from pathlib import Path
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
root=Path(__file__).resolve().parent;ws=root.parents[1]
out=ws/'runs/o2-angular-qualified-measure-a100-20261127-r1';out.mkdir()
(out/'logs').mkdir()
old=ws/'runs/o2-dd7a-angular-capacity-a100-20261126-r1'
for name in ('worker.py','measure_driver.py','post.py'):shutil.copy2(old/name,out/name)
for name in ('common.py','launch.py','control_freeze.py'):shutil.copy2(root/name,out/name)
shutil.copy2(root/'SOURCE.json',out/'SOURCE.json')
shutil.copy2(root/'CANDIDATE-HEAD.txt',out/'CANDIDATE-HEAD.txt')
(out/'models').symlink_to(root/'models',target_is_directory=True)
(out/'gates').symlink_to(root/'gates',target_is_directory=True)
(out/'GATE-AUDIT.txt').write_text(str(root/'gate-audits/120902')+'\n')
driver=(out/'measure_driver.py').read_text()
needle="assert gates['status']=='GATES_REVIEWED'"
assert driver.count(needle)==1
guard="""
qualification=json.loads((ROOT.parent/'o2-angular-stability-a100-20261127-r1/FINAL-SNAPSHOT.json').read_text())
assert qualification['status']=='PASS'
assert qualification['source_package']['package_sha256']==json.loads((ROOT/'SOURCE.json').read_text())['package_sha256']
assert len(qualification['bounded_stability'])==8
assert all(r['continuous_20_steps']==r['inference_3_fresh_processes_10_calls']=='PASS' for r in qualification['bounded_stability'])
assert len(gates['rows'])==153 and all((r['status']=='PASS') if 'key' in r else r['infer']==r['train']=='PASS' for r in gates['rows'])
"""
(out/'measure_driver.py').write_text(driver.replace(needle,needle+'\n'+guard))
post=(out/'post.py').read_text()
post=post.replace('dd7a3db136c1d2946aa0e41b87a167d8f940b122','13c9328e2c5e1a177e6b301078904be595d42e97').replace('dd7a3db','13c9328')
post=post.replace('import csv,hashlib,json,math,os,shutil,statistics,subprocess,tarfile','import csv,hashlib,json,math,os,shutil,statistics,subprocess,tarfile,types')
post=post.replace("('smoke_gpu','gates_gpu','measurement')","('measurement',)")
start="account=subprocess.run(['sacct'"
i=post.index(start);j=post.index('\n',i)
post=post[:i]+"account=types.SimpleNamespace(returncode=0,stdout=(ROOT/'SACCT-GPU.tsv').read_text(),stderr='')"+post[j:]
post=post.replace("str(WS/'runs/o2-bounded-bilinear-a100-20261126-r1/GPU-ACCOUNTING.json')", "str(WS/'runs/o2-angular-stability-a100-20261127-r1/GPU-ACCOUNTING.json')")
post=post.replace("status='COMPLETE_WITH_RECORDED_HOLDS' if manifest.get('status')=='ALL_MEASUREMENT_ATTEMPTS_COMPLETE' else 'INCOMPLETE_EXECUTION'", "status=('COMPLETE' if len(coverage)==114 and all(r['status']=='PASS' for r in coverage) and len(capacity_rows)==66 and all(r['status']=='STABLE_CAPACITY_3OF3' for r in capacity_rows) else 'COMPLETE_WITH_RECORDED_HOLDS') if manifest.get('status')=='ALL_MEASUREMENT_ATTEMPTS_COMPLETE' else 'INCOMPLETE_EXECUTION'")
post=post.replace("save(ROOT/'GPU-ACCOUNTING.json'", "assert len(alloc)==1, 'Missing measurement allocation accounting'\nsave(ROOT/'GPU-ACCOUNTING.json'")
(out/'post.py').write_text(post)
for name in ('cpu.sbatch','gpu.sbatch'):
    text=(root/name).read_text().replace(str(root),str(out)).replace('--job-name=o2-angular-stability','--job-name=o2-angular-qualified-measure')
    if name=='gpu.sbatch':text=text.replace('--time=04:00:00','--time=12:00:00')
    (out/name).write_text(text)
(out/'CONTRACT.zh-CN.md').write_text('''# L1–L8 修复通过后的同版本计量

依据用户既有门后补测授权，准备个人fork数值源码13c9328的算子/整网延迟、吞吐和A100 40G容量。执行前必须存在同源码完整资格FINAL-SNAPSHOT.json：144算子、240局部、原锚点及L1–L8原三步，以及每阶20步连续训练和三新进程各10次推理全部PASS。准备文件不等于提交/通过GPU计量。

保持原模型、输入、图/PBC/重复边、标签、全部活动参数、loss、精度、clip/AdamW/EMA/scheduler和计时边界。算子UVW canonical/full及Gaunt双线性L1–L8，N256延迟、N4096吞吐；整网L=M=1..8固定seed结构模型，原L4/M2 checkpoint锚点分列，固定100原子/5252有向边单图。不同方程只报绝对成本。

每配置三个新进程，计时两次预热、容量一次预热，正式同步调用前在计时外恢复公共状态并清无引用缓存；保留每次值及中位数。容量指数搜索/二分，每点3/3完整执行并保存下一实际OOM上界；整网按100原子不连边图精确搜索，算子按原分辨率。容量只证明完整执行，不作为所有新N/B的数值资格证明。

仅XHPC-A100单节点单卡串行、CPU-AMD冻结/复算/报告，无并行代理。统一expandable_segments:True，能量输出/求和FP64，其他运算/累加/导数FP32；禁TF32/AMP/half/CUDA Graph。所有失败/OOM/超时保留、分项报告，实际GPUh含失败计入500GPUh总账，不继承旧dd7a性能或改写旧FAIL/HOLD。调度回执在登录控制面只读捕获，CPU-AMD解析，避免计算节点无法访问记账端点却误报0GPUh。
''')
for path in out.glob('*.py'):ast.parse(path.read_text())
(out/'PREPARATION.json').write_text(json.dumps({'status':'PREPARED_NOT_SUBMITTED','source_commit':(out/'CANDIDATE-HEAD.txt').read_text().strip(),'source_package_sha256':json.loads((out/'SOURCE.json').read_text())['package_sha256'],'preparation_job':os.environ['SLURM_JOB_ID'],'gate_root':str(root)},indent=2)+'\n')
print('MEASUREMENT_PREPARED_NOT_SUBMITTED',out,flush=True)
