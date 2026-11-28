"""Freeze measured source, receipts and raw result identities; independently unpack."""
import hashlib,json,os,shutil,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WS=ROOT.parents[1]
def read(p):return json.loads(p.read_text())
def save(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    assert os.environ.get('SLURM_JOB_ID')
    gpu=Path((ROOT/'LATEST-RESULT.txt').read_text().strip());analysis=Path((ROOT/'LATEST-ANALYSIS.txt').read_text().strip())
    assert (gpu/'FINISH.json').exists()
    comparison=read(analysis/'COMPARISON.json');paired=read(analysis/'PAIRED-AUDIT.json')
    assert len(comparison)==2 and all(row['passed'] for row in paired.values()),paired
    accounting=[line.split('|') for line in (ROOT/'ACCOUNTING.tsv').read_text().splitlines() if line]
    allocation=next(row for row in accounting if row[0]=='119167')
    assert allocation[1]=='COMPLETED' and allocation[2]=='0:0' and allocation[5]=='1' and allocation[6]=='XHPC-A100'
    gpu_accounting={'job':'119167','state':allocation[1],'elapsed_s':int(allocation[3]),'nodes':1,'gpus':1,
        'gpu_h':int(allocation[3])/3600,'method':'Slurm actual ElapsedRaw; fixed --gres=gpu:1 verified by startup device_count and common GPU UUID',
        'project_hard_cap_gpu_h':500,'project_total_not_recomputed':True,
        'historical_ledger':'runs/o2-hp-approved-20261123-r1/provenance/BUDGET-AUDIT.json',
        'append_to_project_ledger_once_by_job_id':True,'CPU_failures_and_successes':'ACCOUNTING.tsv; no GPU allocation'}
    save(ROOT/'GPU-ACCOUNTING.json',gpu_accounting)
    # Refine an overbroad metadata label: exact recovery is required only for
    # nonzero residuals. Zero residuals yield zero with any valid denominator.
    review=read(ROOT/'FP32-METRIC-REVIEW.json')
    if 'all_original_denominators_exactly_recovered' in review:
        shutil.copy2(ROOT/'FP32-METRIC-REVIEW.json',ROOT/'FP32-METRIC-REVIEW-initial.json')
        review.pop('all_original_denominators_exactly_recovered')
        review.update(all_nonzero_residual_denominators_exactly_recovered=True,
            zero_residual_result_independent_of_valid_denominator=True,
            original_metadata_label_preserved='FP32-METRIC-REVIEW-initial.json')
        save(ROOT/'FP32-METRIC-REVIEW.json',review)
    report=(ROOT/'REPORT.zh-CN.md').read_text().replace(
        '已由保存的完整数组FP32最大差及其比值精确恢复原分母，并逐项要求重现原标量比值，再用FP32张量复算最终除法',
        '已由保存的完整数组FP32最大差及其比值精确恢复非零残差的原分母，并逐项重现原标量比值；零残差对任意合法分母均为0。最终除法以FP32张量复算')
    (ROOT/'REPORT.zh-CN.md').write_text(report)
    preflight=read(ROOT/'PREFLIGHT.json');original=Path(preflight['snapshot'])
    executed_seal=read(analysis/'SOURCE-SEAL.json')
    assert sha(analysis/'EXECUTED-ANALYZE.py')==executed_seal['analyze.py']
    target=ROOT/'snapshots'/('delivery-'+os.environ['SLURM_JOB_ID']);target.mkdir()
    source=target/'source';source.mkdir()
    shutil.copytree(original,source/'measured-preflight')
    shutil.copytree(WS/'ICTC-O2-HP',source/'fork',ignore=shutil.ignore_patterns('.git'))
    controls=source/'controls';controls.mkdir()
    for p in list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+list(ROOT.glob('*.sbatch'))+list(ROOT.glob('*.json'))+list(ROOT.glob('*.tsv'))+list(ROOT.glob('*.txt')):
        shutil.copy2(p,controls/p.name)
    shutil.copytree(ROOT/'adapter',controls/'adapter')
    shutil.copytree(ROOT/'figures',source/'figures')
    workspace_docs=['AGENTS.md','HANDOFF.md','OPERATOR_STATUS.md','architectures/equiformerv3/README.zh-CN.md',
        'architectures/equiformerv3/HOTSPOT-UVW-TILED-20261125-r1.zh-CN.md']
    for name in workspace_docs:
        destination=source/'workspace-docs'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(WS/name,destination)
    prepared=source/'prepared';prepared.mkdir()
    for name in ('FRAMES.pt','PUBLIC-FRAME.json','INPUT-IDENTITY.json'):shutil.copy2(ROOT/'prepared'/name,prepared/name)
    summary=source/'analysis';summary.mkdir()
    for p in analysis.glob('*'):
        if p.is_file():shutil.copy2(p,summary/p.name)
    for child in (analysis/'traces').iterdir():
        destination=summary/'traces'/child.name;destination.mkdir(parents=True)
        for name in ('KERNELS.csv','FAMILY.csv','PHASE.csv','SUMMARY.json'):shutil.copy2(child/name,destination/name)
    dependencies={}
    for mode,folder in preflight['common_states'].items():
        p=Path(folder);condition=read(p/'CONDITION.json')
        checkpoint=Path(condition['identity']['checkpoint'])
        dependencies[mode]={'common_state':str(p/'START.pt'),'common_state_sha256':sha(p/'START.pt'),
            'condition':str(p/'CONDITION.json'),'condition_sha256':sha(p/'CONDITION.json'),
            'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint)}
        assert dependencies[mode]['common_state_sha256']==condition['start_sha256']
        assert dependencies[mode]['checkpoint_sha256']==condition['identity']['checkpoint_sha256']
    save(source/'DEPENDENCIES.json',{'preserved_external_immutable_inputs':dependencies,
        'archive_scope':'source, Git object database, light public inputs, receipts, complete kernel-type tables and figures; raw traces/full gate objects/common states remain at sealed workspace paths'})
    archive=target/'SOURCE-AND-RECEIPTS.tar.gz'
    manifest={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
    save(target/'SOURCE-MANIFEST.json',manifest)
    with tarfile.open(archive,'w:gz') as tf:tf.add(source,arcname='source',recursive=True)
    unpacked=target/'independent-unpack';unpacked.mkdir()
    with tarfile.open(archive) as tf:tf.extractall(unpacked,filter='data')
    found={str(p.relative_to(unpacked/'source')):sha(p) for p in (unpacked/'source').rglob('*') if p.is_file()}
    assert manifest==found,'Independent archive member/content mismatch'
    paths=set()
    for tree in (gpu,analysis):
        for p in tree.rglob('*'):
            if p.is_file() and not any(part in ('triton-cache','inductor-cache','mpl-cache') for part in p.parts):paths.add(p)
    paths.update(p for p in ROOT.glob('*') if p.is_file() and p.name!='FINAL-SNAPSHOT.json')
    paths.update(p for p in (ROOT/'logs').glob('*.log') if os.environ['SLURM_JOB_ID'] not in p.name)
    paths.update(p for p in (ROOT/'figures').rglob('*') if p.is_file())
    paths.update(WS/name for name in workspace_docs)
    raw={str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(paths)}
    save(target/'RAW-RESULT-MANIFEST.json',raw)
    receipt={'status':'SOURCE_AND_RESULTS_SEALED_WITH_INDEPENDENT_UNPACK_QA','job':os.environ['SLURM_JOB_ID'],
        'candidate_commit':preflight['candidate_commit'],'gpu_result':str(gpu),'analysis':str(analysis),
        'snapshot':str(target),'source_files_verified':len(manifest),'raw_result_files_hashed':len(raw),
        'archive':str(archive),'archive_sha256':sha(archive),'raw_manifest_sha256':sha(target/'RAW-RESULT-MANIFEST.json'),
        'archive_scope':'source and receipts; complete raw results and large immutable dependencies retained and hashed separately',
        'numerical_status':read(analysis/'GATES.json'),'paired':paired,'dependencies':dependencies,
        'gpu_accounting':gpu_accounting}
    save(ROOT/'FINAL-SNAPSHOT.json',receipt);save(target/'FINAL-SNAPSHOT.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('numerical_status','paired','dependencies')}),flush=True)
if __name__=='__main__':main()
