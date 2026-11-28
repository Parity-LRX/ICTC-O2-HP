"""Seal successful qualification only after every independent bounded audit passes."""
import argparse,hashlib,json,os
from pathlib import Path
assert os.environ['SLURM_JOB_PARTITION']=='CPU-AMD'
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--stability-job',default='120889');args=parser.parse_args()
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
source=read(root/'SOURCE.json');commit=(root/'CANDIDATE-HEAD.txt').read_text().strip()
for name,digest in source['files'].items():assert sha(Path(source['path'])/name)==digest,('Frozen numerical source changed',name)
gate_path=root/'gate-audits/120876/RESULT.json';gate=read(gate_path)
operators=[r for r in gate['rows'] if r.get('identity',{}).get('level')=='operator']
networks=[r for r in gate['rows'] if r.get('level')=='network']
assert len(operators)==144 and all(r['status']=='PASS' for r in operators)
assert len(networks)==9 and all(r['infer']==r['train']=='PASS' for r in networks)
local_path=root/'local-audits/120877/RESULT.json';local=read(local_path)
assert local['status']=='PASS' and len(local['rows'])==240
detail_path=root/'gate-audits/120902/RESULT.json';details=read(detail_path)
assert len(details['rows'])==len(gate['rows'])
assert all((r['status']=='PASS') if 'key' in r else r['infer']==r['train']=='PASS' for r in details['rows'])
local_detail_path=root/'local-audits/120902/RESULT.json';assert read(local_detail_path)['status']=='PASS'
package_path=root/'package-checks/120865/RESULT.json';assert read(package_path)['status']=='PASS'
capture=root/'stability-captures'/args.stability_job;campaign=read(capture/'RESULT.json')
assert not (capture/'STOP.json').exists()
assert len(campaign['groups'])==len(campaign['audits'])==8
assert sorted(r['order'] for r in campaign['audits'])==list(range(1,9))
rows=[];receipts=[gate_path,local_path,detail_path,local_detail_path,package_path,root/'SOURCE.json',capture/'RESULT.json']
for item in sorted(campaign['audits'],key=lambda x:x['order']):
    order=item['order'];path=root/'stability-audits'/str(item['job'])/'RESULT.json';audit=read(path)
    assert audit['status']=='PASS' and audit['order']==order
    assert audit['steps']==20 and audit['fresh_inference_processes']==3 and audit['inference_calls_per_process']==10
    assert audit['original_thresholds'] and not audit['restored_inside_trajectory'] and not audit['shared_true_loss_cotangents']
    assert len(audit['summaries'])==162 and all(not v['failures'] for v in audit['summaries'])
    manifest=read(audit['manifest']);assert manifest['source_package_sha256']==source['package_sha256']
    net,=[r for r in networks if r['order']==order]
    def maximum(prefix,field):
        return max(v[field] for v in audit['summaries'] if v['label'].startswith(prefix) and v['label'].endswith('-candidate'))
    rows.append({'order':order,'original_inference':net['infer'],'original_three_steps':net['train'],
                 'continuous_20_steps':audit['status'],'inference_3_fresh_processes_10_calls':audit['status'],
                 'training_max_relative_l2':maximum('train-step','max_relative_l2'),
                 'inference_max_relative_l2':maximum('infer-','max_relative_l2'),
                 'training_max_ulp':maximum('train-step','max_ULP'),'inference_max_ulp':maximum('infer-','max_ULP'),
                 'audit_job':str(item['job']),'receipt':str(path),'receipt_sha256':sha(path)})
    receipts.extend([path,Path(audit['manifest'])])
jobs=['120842','120845','120847','120853','120857','120866','120870',args.stability_job]
# Capture this read-only scheduler receipt on the login control plane, where the
# accounting endpoint is available, then parse and seal it on CPU-AMD.
raw=(root/'GPU-SACCT.txt').read_text()
accounting=[]
for line in raw.splitlines():
    if not line.strip():continue
    values=line.split('|');job,state,seconds,start,end,tres,node=values[:7]
    assert job in jobs and node=='XHPC-A100'
    if 'gres/gpu=' in tres:assert 'gres/gpu=1' in tres.split(',')
    assert state in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY'),(job,state)
    accounting.append({'job':job,'state':state,'seconds':int(seconds),'start':start,'end':end,'allocated_resources':tres,'node':node})
assert len(accounting)==len(jobs)
ordered=sorted(accounting,key=lambda r:r['start'])
assert all(a['end']<=b['start'] for a,b in zip(ordered,ordered[1:])),'Own GPU allocations overlapped'
batch=root/'gpu.sbatch';controller=root/('SCONTROL-'+args.stability_job+'.txt')
assert '#SBATCH --gres=gpu:1' in batch.read_text() and 'TresPerNode=gres:gpu:1' in controller.read_text()
gpu={'allocations':accounting,'seconds_including_failures':sum(r['seconds'] for r in accounting),
     'GPUh_including_failures':sum(r['seconds'] for r in accounting)/3600,'single_A100_serial':True,'parent_budget_GPUh':500,
     'GPU_count_provenance':'One-GPU batch request and runtime configure/device-count assertions; this cluster omits GPUs in accounting TRES',
     'batch_script_sha256':sha(batch),'live_controller_receipt':str(controller),'live_controller_sha256':sha(controller)}
save(root/'GPU-ACCOUNTING.json',gpu)
result={'status':'PASS','source_commit':commit,'source_package':source,'operator_configurations':144,'local_regressions':240,
        'extra_tail_regressions':72,'original_network_configurations':9,'bounded_stability':rows,
        'original_loss_precision_thresholds_and_active_parameters_preserved':True,
        'source_scope':'O2 full-parity/canonical operators; canonical L=M=1..8 structural networks; original L4/M2 checkpoint anchor separate',
        'stability_scope':'20 own-loss continuous updates and 3 fresh inference processes of 10 calls per backend and order',
        'performance_capacity_status':'NOT_MEASURED_FOR_THIS_SOURCE','long_term_training_or_task_accuracy_claim':False,
        'receipts':[{'path':str(p),'sha256':sha(p)} for p in receipts],'gpu_accounting':gpu,
        'fork':'https://github.com/LeutOrca/ICTC-O2-HP','pull_request':'https://github.com/Parity-LRX/ICTC-O2-HP/pull/2'}
save(root/'FINAL-SNAPSHOT.json',result)
save(root/'MEASUREMENT-ELIGIBILITY.json',{'status':'ALL_REQUESTED_MODES_ELIGIBLE','source_package_sha256':source['package_sha256'],
                                       'gate_receipt':str(gate_path),'stability_receipt':str(root/'FINAL-SNAPSHOT.json')})
text=['# L1–L8 推理与训练修复验收','',f'数值源码 `{commit}`，个人 fork 分支 `codex/angular-stability-v7`。',
      '', '144项算子配置、240组局部完整导数回归、额外72组L7尾区回归，以及原L4/M2 checkpoint锚点与L1–L8的原完整三步均通过原门。',
      '', '每阶额外20步各自真实损失连续训练和三次新进程、每进程10次E/F推理均由独立CPU-AMD审核通过。model/optimizer/EMA/scheduler/RNG在训练轨迹中连续，无中途恢复和共享真实损失余切。',
      '', '|角向阶数|原E/F与VJP|原三步及真实混合导数|连续20步|三新进程×10次推理|训练最大relative-L2|推理最大relative-L2|',
      '|---|---|---|---|---|---|---|']
for r in rows:text.append(f"|L{r['order']}|PASS|PASS|PASS|PASS|{r['training_max_relative_l2']:.9g}|{r['inference_max_relative_l2']:.9g}|")
text += ['', 'L7修复仅将一项四行角向尾区计划改为54分区单链FP32，原真实操作数校准与完整导数/训练均另行验收。L8修复移除coverage.py导入冲突，并统一使用expandable_segments分配器及提前释放原反向本来会清理的旧梯度；默认分配器OOM回执保留。wheel已验证包含完整6576项计划。',
         '', '原分量门2e-5+2e-5*abs(ref)、完整对象relative-L2<=1e-5和局部epsilon_max<=1e-5不变。ULP只作描述，逐角色完整误差/ULP文件及原始数组均保存在本run，轻量回执随fork快照发布。部分角色仍有门内非零误差。',
         '', '此资格覆盖冻结输入与所列稳定性窗口。L1–L8使用固定seed独立结构模型；原L4/M2 checkpoint锚点分列。它不证明数据集预测精度、任意时长训练轨迹或长期MD稳定。新源码的速度、显存和容量须重新计量，不能继承旧dd7a3db的倍率。',
         '', f"本run全部{len(accounting)}次单A100串行allocation（含失败）合计{gpu['seconds_including_failures']}秒 / {gpu['GPUh_including_failures']:.9f} GPUh，纳入原500 GPUh总账。",
         '', '恢复入口：FINAL-SNAPSHOT.json、SOURCE.json、snapshots/continuity-120888/、GPU-ACCOUNTING.json；原v6及所有历史FAIL/OOM/HOLD均保持。',
         '', '[个人fork](https://github.com/LeutOrca/ICTC-O2-HP) · [上游PR #2](https://github.com/Parity-LRX/ICTC-O2-HP/pull/2)']
(root/'REPORT.zh-CN.md').write_text('\n'.join(text)+'\n')
print('FINAL_QUALIFICATION_PASS',commit,len(rows),gpu['GPUh_including_failures'],flush=True)
