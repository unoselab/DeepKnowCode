#!/usr/bin/env python3
"""G-03d v1, independent read-only verification of G-03c v4 M3 lineage.

No training, model loading, modification of frozen inputs, or GPU use.
All experiment settings are passed explicitly by the standalone Shell wrapper.
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def ensure(ok, why):
    if not ok:
        raise RuntimeError(why)


def timestamp():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def hash_json(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    with Path(path).open(encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            ensure(bool(line.strip()), f'BLANK_JSONL_LINE:{path}:{n}')
            yield json.loads(line)


def bound(path, digest, msg):
    ensure(Path(path).is_file(), f'{msg}_FILE_MISSING:{path}')
    actual = sha(path)
    ensure(actual == digest, f'{msg}_SHA256_DRIFT expected={digest} actual={actual} file={path}')


def contained(root, relative):
    name = Path(relative)
    ensure(not name.is_absolute() and '..' not in name.parts and name.parts,
           'UNSAFE_MANIFEST_RELATIVE_PATH')
    output = (root / name).resolve()
    ensure(output.is_relative_to(root.resolve()), 'MANIFEST_PATH_ESCAPE')
    return output


def adapter_check(folder, manifest, expected, kind):
    ensure(manifest['adapter_files_sha256']['adapter_model.safetensors'] == expected,
           kind + '_MANIFEST_WEIGHT_SHA_DRIFT')
    hashes = manifest['adapter_files_sha256']
    ensure(len(hashes) >= 2 and 'adapter_config.json' in hashes,
           kind + '_INCOMPLETE_ADAPTER_MANIFEST')
    for rel, digest in hashes.items():
        bound(contained(folder, rel), digest, kind + '_ADAPTER')
    config = read(folder / 'adapter_config.json')
    ensure(config.get('peft_type') == 'LORA' and config.get('r') > 0,
           kind + '_INVALID_PEFT_CONFIG')
    return {"weight_sha256": expected, "files_verified": len(hashes),
            "adapter_dir": str(folder.resolve())}


def verify(a):
    ensure(not a.output_dir.exists(), 'REFUSE_OVERWRITE_G03D_RUN')
    for key in ('m1_sha256', 'm2_sha256', 'm3_sha256', 'g03b_order_sha256',
                'g03b_manifest_sha256', 'g02_train_sha256', 'g02_validation_sha256',
                'g02_fingerprint','g02_prepared_manifest_sha256'):
        ensure(re.fullmatch(r'[0-9a-f]{64}', getattr(a, key)), 'INVALID_SHELL_HASH:' + key)
    ensure(a.steps * a.batch == a.pairs and a.steps > 0 and a.batch > 0,
           'TRAINING_BUDGET_INCONSISTENT')
    ensure(len(a.phase_names)==len(a.phase_steps)==len(a.phase_source)==len(a.phase_target)
           and sum(a.phase_steps)==a.steps and sum(a.phase_source)+sum(a.phase_target)==a.pairs
           and all(x>0 for x in a.phase_steps), 'SHELL_PHASE_BUDGET_INVALID')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    g03c = a.g03c_pipeline
    train = g03c / '03-train'
    g02train = a.g02_pipeline / '03-train'
    m1train = a.m1_pipeline / '03-train'
    paths = (train, g02train, m1train, a.g03b_run)
    ensure(all(p.is_dir() for p in paths), 'FROZEN_SERVER_RUN_MISSING')
    gp = read(g03c/'pipeline_result.json')
    ensure(gp['status'] == 'PASS' and gp['exit_code'] == 0 and gp['last_stage'] == 'verify',
           'G03C_PIPELINE_NOT_PASS')
    ver = read(g03c/'04-verify/verification_result.json')
    ensure(ver['status'] == a.expected_g03c_verify_status and ver['steps'] == a.steps and
           ver['pairs'] == a.pairs and ver['curriculum_order_sha256'] == a.g03b_order_sha256 and
           ver['adapter_model_sha256'] == a.m3_sha256,
           'G03C_INDEPENDENT_VERIFICATION_PROOF_DRIFT')
    bm = read(a.g03b_run/'curriculum_manifest.json')
    bound(a.g03b_run/'curriculum_manifest.json', a.g03b_manifest_sha256, 'G03B_MANIFEST')
    bound(a.g03b_run/'curriculum_order.jsonl', a.g03b_order_sha256, 'G03B_ORDER')
    ensure(bm['expected_curriculum_order_sha256'] == a.g03b_order_sha256 and
           bm['common_frozen_m1_adapter_sha256'] == a.m1_sha256 and
           bm['source_g02_train_sha256'] == a.g02_train_sha256,
           'G03B_FROZEN_LINEAGE_DRIFT')
    prepared = a.g02_pipeline/'00-prepare/prepared'
    bound(prepared/'manifest.json',a.g02_prepared_manifest_sha256,
          'G02_PREPARED_MANIFEST')
    ensure(read(prepared/'manifest.json')['config_fingerprint']==a.g02_fingerprint,
           'G02_PREPARED_CONFIGURATION_DRIFT')
    bound(prepared/'train.jsonl', a.g02_train_sha256, 'G02_FROZEN_TRAIN')
    bound(prepared/'validation.jsonl', a.g02_validation_sha256, 'G02_FROZEN_VALIDATION')
    s = read(train/'training_summary.json')
    trm = read(train/'g03c_train_manifest.json')
    am = read(train/'adapter_manifest.json')
    ensure(s['status'] == a.expected_m3_status == trm['status'] and
           am['status'] == a.expected_m3_adapter_status and
           s['model_id'] == a.model_id and s['model_revision'] == a.revision and
           s['optimizer_steps'] == a.steps and s['samples_per_step'] == a.batch and
           s['total_unique_consumed_pairs'] == a.pairs and
           s['g02_frozen_train_sha256'] == a.g02_train_sha256 and
           s['g02_frozen_validation_sha256'] == a.g02_validation_sha256 and
           s['g02_compatible_core_fingerprint'] == a.g02_fingerprint and
           s['initial_m1_adapter_sha256'] == a.m1_sha256 and
           s['m2_adapter_used_as_initial_m3_weights'] is False and
           s['g03b_order_sha256'] == a.g03b_order_sha256 and
           s['g03b_manifest_sha256'] == a.g03b_manifest_sha256 and
           s['heldout_test_evaluated'] is False,
           'G03C_TRAINING_SUMMARY_PROVENANCE_DRIFT')
    ensure(s['training_order'] == a.expected_m3_order and
           am['initial_m1_adapter_sha256'] == a.m1_sha256 and
           am['g03b_order_sha256'] == a.g03b_order_sha256 and
           am['optimizer_steps'] == a.steps and
           am['consumption_ledger_sha256'] == s['training_consumption_ledger_sha256'],
           'G03C_ADAPTER_LINEAGE_DRIFT')
    for key, rel in [('training_summary_sha256','training_summary.json'),
                     ('training_history_sha256','training_history.jsonl'),
                     ('consumption_history_sha256','consumption_history.jsonl'),
                     ('adapter_manifest_sha256','adapter_manifest.json')]:
        bound(train/rel,trm[key],'G03C_MANIFEST_'+key)
    ensure(trm['g03b_order_sha256']==a.g03b_order_sha256 and
           trm['g03b_manifest_sha256']==a.g03b_manifest_sha256 and
           s['adapter_manifest_sha256']==sha(train/'adapter_manifest.json') and
           s['training_consumption_ledger_sha256']==sha(train/'consumption_history.jsonl') and
           s['training_history_sha256']==sha(train/'training_history.jsonl'),
           'G03C_TRAIN_ARTIFACT_HASH_DRIFT')
    bound(train/'token_audit.json',s['token_audit_sha256'],'G03C_TOKEN_AUDIT')
    m2 = read(g02train/'training_summary.json')
    m2manifest = read(g02train/'adapter_manifest.json')
    m1 = read(m1train/'training_summary.json')
    m1manifest=read(m1train/'adapter_manifest.json')
    ensure(m2['status']==a.expected_m2_status and m2['optimizer_steps']==a.steps and
           m2['initial_m1_adapter_sha256']==a.m1_sha256 and
           m2['config_fingerprint']==a.g02_fingerprint and
           m2['training_order']==a.expected_m2_order and
           m1['status']==a.expected_m1_status and
           m1manifest['status']==a.expected_m1_adapter_status,
           'M1_M2_BRANCH_LINEAGE_DRIFT')
    ensure(m2['adapter_manifest_sha256']==sha(g02train/'adapter_manifest.json'),
           'M2_ADAPTER_MANIFEST_DRIFT')
    adapters={
        'M1':adapter_check(m1train/'adapter',m1manifest,a.m1_sha256,'M1'),
        'M2':adapter_check(g02train/'adapter',m2manifest,a.m2_sha256,'M2'),
        'M3':adapter_check(train/'adapter',am,a.m3_sha256,'M3')}
    expected = list(lines(a.g03b_run/'curriculum_order.jsonl'))
    ensure(len(expected)==a.pairs,'G03B_ORDER_PAIR_COUNT')
    phase_counts=collections.Counter(); labels=collections.Counter(); indices=[]
    phase_corpus=collections.Counter(); phase_labels=collections.Counter()
    step_to_phase={}
    step_num=1
    for name,steps in zip(a.phase_names,a.phase_steps):
        for step in range(step_num,step_num+steps): step_to_phase[step]=name
        step_num+=steps
    plan=bm['phase_plan']
    ensure(len(plan)==len(a.phase_names),'G03B_PHASE_MANIFEST_COUNT_DRIFT')
    for i,(name,steps,source,target) in enumerate(zip(a.phase_names,a.phase_steps,
             a.phase_source,a.phase_target)):
        entry=plan[i]
        ensure(entry['name']==name and entry['steps']==steps and
               entry['java_source']==source and
               entry['java_mixed']+entry['python_mixed']==target and
               entry['java_mixed']==entry['python_mixed'],
               'G03B_PHASE_PLAN_MANIFEST_DRIFT:'+name)
    for index,item in enumerate(expected):
        ensure(item['position']==index and item['optimizer_step']==index//a.batch+1 and
               item['microbatch_index']==index%a.batch and
               item['phase_name']==step_to_phase[index//a.batch+1],
               'G03B_ORDER_STRUCTURE_OR_PHASE_DRIFT')
        indices.append(item['row_index']); phase_counts[item['phase_name']]+=1
        phase_corpus[(item['phase_name'],item['corpus'])]+=1
        phase_labels[(item['phase_name'],item['label'])]+=1
        labels[item['label']]+=1
    for name,steps,source,target in zip(a.phase_names,a.phase_steps,
            a.phase_source,a.phase_target):
        ensure(phase_counts[name]==steps*a.batch and
               phase_corpus[(name,'java_source')]==source and
               phase_corpus[(name,'java_mixed')]==target//2 and
               phase_corpus[(name,'python_mixed')]==target//2 and
               phase_labels[(name,0)]==steps*a.batch//2 and
               phase_labels[(name,1)]==steps*a.batch//2,
               'G03B_PHASE_CORPUS_OR_LABEL_QUOTA_DRIFT:'+name)
    ensure(set(indices)==set(range(a.pairs)) and len(indices)==a.pairs and
           labels[0]==labels[1]==a.pairs//2,'G03B_PAIRS_NOT_UNIQUE_BALANCED')
    events=list(lines(train/'consumption_history.jsonl'))
    ensure(len(events)==a.steps,'G03C_LEDGER_STEPS_DRIFT')
    all_consumed=[]; step_hashes=[]; per_step=[]
    for step,e in enumerate(events,1):
        block=expected[(step-1)*a.batch:step*a.batch]
        chunk=[x['row_index'] for x in block]
        ensure(e['step']==step and e['row_indices']==chunk and
               e['phase']==block[0]['phase_name'] and
               e['g03b_order_sha256']==a.g03b_order_sha256 and
               e['row_indices_sha256']==hash_json(chunk),
               'G03C_LEDGER_STEP_ORDER_DRIFT:'+str(step))
        ensure(sum(x['label']==0 for x in block)==a.batch//2 and
               sum(x['label']==1 for x in block)==a.batch//2,
               'G03C_LEDGER_LABEL_DRIFT:'+str(step))
        all_consumed.extend(chunk);step_hashes.append(hash_json(chunk))
        ensure(e['consumed_prefix_row_index_sha256']==hash_json(all_consumed),
               'G03C_LEDGER_PREFIX_DRIFT:'+str(step))
        if step in a.checkpoint_steps:
            per_step.append({'step':step,'prefix_sha256':hash_json(all_consumed)})
    ensure(all_consumed==indices and s['consumed_order_sha256']==hash_json(indices) and
           s['expected_order_row_index_sha256']==hash_json(indices) and
           s['per_step_ledger_sha256']==hash_json(step_hashes),
           'G03C_FINAL_FULL_CONSUMPTION_DRIFT')
    ckpt_root=train/'checkpoints'
    folders=sorted(x for x in ckpt_root.glob('step-*') if x.is_dir())
    discovered=[]
    for folder in folders:
        cm=read(folder/'checkpoint_manifest.json')
        step=cm['optimizer_step']; discovered.append(step)
        ensure(cm['status']==a.expected_checkpoint_status and
               cm['g03b_curriculum_order_sha256']==a.g03b_order_sha256 and
               cm['g03c_experiment_fingerprint']==s['g03c_experiment_fingerprint'] and
               cm['initial_m1_adapter_sha256']==a.m1_sha256 and
               cm['g02_fingerprint']==a.g02_fingerprint and
               cm['consumed_prefix_row_index_sha256']==hash_json(indices[:step*a.batch]) and
               cm['optimizer_state_saved'] is True and
               'optimizer_state.pt' in cm['files_sha256'] and
               'adapter/adapter_model.safetensors' in cm['files_sha256'],
               'G03C_CHECKPOINT_LINEAGE_DRIFT:'+str(step))
        for rel,digest in cm['files_sha256'].items():
            bound(contained(folder,rel),digest,'M3_CHECKPOINT')
    ensure(discovered==a.checkpoint_steps and
           len(s['checkpoints'])==len(folders) and
           all(Path(p).resolve()==f.resolve() for p,f in zip(s['checkpoints'],folders)),
           'M3_CHECKPOINT_SEQUENCE_MISMATCH')
    ensure(read(g03c/'04-verify/verification_result.json')['checkpoints_verified']==discovered,
           'G03C_FINAL_VERIFICATION_CHECKPOINT_DRIFT')
    report={
        'status':'G03D_V1_INDEPENDENT_M3_VERIFY_PASS','version':'v1',
        'at_chicago':timestamp(),'m3_source_run':str(train.resolve()),
        'm1_source_run':str(m1train.resolve()),'m2_source_run':str(g02train.resolve()),
        'model_id':a.model_id,'revision':a.revision,
        'm2_m3_initialization':'independent-branches-from-frozen-M1',
        'adapter_weights':adapters,'g03b_order_sha256':a.g03b_order_sha256,
        'g03b_manifest_sha256':a.g03b_manifest_sha256,
        'g02_train_sha256':a.g02_train_sha256,
        'g02_validation_sha256':a.g02_validation_sha256,
        'g02_prepared_manifest_sha256':a.g02_prepared_manifest_sha256,
        'g02_core_fingerprint':a.g02_fingerprint,
        'g03c_experiment_fingerprint':s['g03c_experiment_fingerprint'],
        'consumption_history_sha256':sha(train/'consumption_history.jsonl'),
        'training_history_sha256':sha(train/'training_history.jsonl'),
        'optimizer_steps_verified':len(events), 'consumed_pairs_verified':len(all_consumed),
        'positive_pairs':labels[1], 'negative_pairs':labels[0],
        'checkpoint_steps_verified':discovered, 'checkpoint_count':len(discovered),
        'phase_counts':dict(phase_counts),'m3_last_mixed_validation':{
            k:s['last_validation'][k] for k in ('step','validation_nll','validation_accuracy_diagnostic','examples')},
        'heldout_test_evaluated':False,
        'approved_for_training':s['approved_for_training'],
        'manual_label_review_status':s['manual_label_review_status'],
        'independent_label_review_verified_by_g03d':False,
        'actual_adapter_weight_bytes_verified':True,
        'notes':['Read-only independent M3 provenance/consumption/adapter/checkpoint SHA256 audit',
                 'Does not evaluate held-out test or independently adjudicate training labels']}
    dest=a.output_dir/'verification_report.json'
    with dest.open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2,ensure_ascii=False,sort_keys=True); f.write('\n')
    manifest={'status':'G03D_V1_REPORT_MANIFEST_PASS','version':'v1',
              'verification_report_sha256':sha(dest),'at_chicago':timestamp()}
    with (a.output_dir/'verification_manifest.json').open('x',encoding='utf-8') as f:
        json.dump(manifest,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'status':report['status'],'pairs':len(indices),'steps':len(events),
           'checkpoints':len(discovered),'m3_adapter_sha256':a.m3_sha256,
           'run':str(a.output_dir),'at_chicago':timestamp()}),flush=True)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('g03c-pipeline','g03b-run','g02-pipeline','m1-pipeline','output-dir','log-file'):
        p.add_argument('--'+key,required=True,type=Path)
    for key in ('model-id','revision','m1-sha256','m2-sha256','m3-sha256',
                'g03b-order-sha256','g03b-manifest-sha256','g02-train-sha256',
                'g02-validation-sha256','g02-fingerprint','g02-prepared-manifest-sha256',
                'expected-g03c-verify-status','expected-m3-status','expected-m3-adapter-status',
                'expected-m3-order','expected-m2-status','expected-m2-order',
                'expected-m1-status','expected-m1-adapter-status','expected-checkpoint-status'):
        p.add_argument('--'+key,required=True)
    for key in ('pairs','steps','batch'):
        p.add_argument('--'+key,required=True,type=int)
    p.add_argument('--checkpoint-steps',required=True)
    for flag in ('phase-names','phase-steps','phase-source','phase-target'):
        p.add_argument('--'+flag,required=True)
    a=p.parse_args()
    a.phase_names=a.phase_names.split(',')
    for key in ('phase_steps','phase_source','phase_target'):
        setattr(a,key,[int(x) for x in getattr(a,key).split(',')])
    a.checkpoint_steps=[int(x) for x in a.checkpoint_steps.split(',')]
    ensure(a.checkpoint_steps==sorted(set(a.checkpoint_steps)) and a.checkpoint_steps[-1]==a.steps,
           'INVALID_CHECKPOINT_STEP_POLICY')
    return a

if __name__=='__main__':
    try:
        verify(cli())
    except Exception as exc:
        print(json.dumps({'event':'G03D_VERIFY_FAIL','error_type':type(exc).__name__,
                          'error':str(exc),'at_chicago':timestamp()}),file=sys.stderr,flush=True)
        raise
