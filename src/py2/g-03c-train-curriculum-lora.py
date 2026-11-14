#!/usr/bin/env python3
"""G-03c v1: audited GPT-OSS-120B M3 curriculum; M1 -> M3 independent branch.

This controller vendors the exact completed G-02 v2 MXFP4 training kernel as a
separate, hash-pinned Python module. No G-02 wrapper or M2 adapter is invoked.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

VERSION = 'v1'
DOMAINS = ('java_source', 'java_mixed', 'python_mixed')
G03B_STATUS = 'G03B_V1_CURRICULUM_PREPARED_AUDIT_PASS_NOT_TRAINED'
AUDIT_STATUS = 'G03C_V1_CPU_CURRICULUM_AUDIT_PASS_NOT_TRAINED'
TRAIN_STATUS = 'G03C_V1_M3_CURRICULUM_TRAINING_COMPLETE_RESEARCH_PENDING_REVIEW'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def chicago():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def hash_json(x):
    return hashlib.sha256(canonical(x).encode('utf-8')).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def records(path):
    with Path(path).open('r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f, 1):
            require(bool(line.strip()), 'BLANK_JSONL_ROW '+str(path)+':'+str(i))
            yield json.loads(line)


def write_new(path, x):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(x, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')


def parse_control():
    raw = sys.argv[1:]
    require('--kernel-args' in raw, 'MISSING_EXPLICIT_KERNEL_ARGS')
    i = raw.index('--kernel-args')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('audit', 'stress', 'stability', 'train', 'verify'))
    for name in ('version','g03b-manifest-sha256','g03b-order-sha256',
                 'g03b-phase-audit-sha256','g03b-comparison-sha256',
                 'g02-prepared-manifest-sha256','g02-train-sha256','g02-validation-sha256',
                 'g02-standard-order-sha256','g02-fingerprint','g03a-score-sha256',
                 'm1-adapter-sha256','research-acknowledged','training-approved',
                 'label-review-status','engine-sha256','phase-policy','model-id','model-revision'):
        p.add_argument('--'+name, required=True)
    for name in ('pipeline','stage-run','g03b-run','g02-pipeline','g03a-run','audit-run',
                 'stress-run','stability-run','train-run','engine-path'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('train-pairs','validation-pairs','optimizer-steps','global-batch',
                 'expected-imputed','expected-variants','seed'):
        p.add_argument('--'+name, type=int, required=True)
    for name in ('phase-names',):
        p.add_argument('--'+name, nargs=3, required=True)
    for name in ('phase-steps','phase-source','phase-target'):
        p.add_argument('--'+name, nargs=3, type=int, required=True)
    c = p.parse_args(raw[:i])
    require(c.version == VERSION, 'G03C_VERSION_MISMATCH')
    for name in ('g03b_manifest_sha256','g03b_order_sha256','g03b_phase_audit_sha256',
                 'g03b_comparison_sha256','g02_prepared_manifest_sha256','g02_train_sha256',
                 'g02_validation_sha256','g02_standard_order_sha256','g02_fingerprint',
                 'g03a_score_sha256','m1_adapter_sha256','engine_sha256'):
        require(re.fullmatch('[0-9a-f]{64}', getattr(c,name)) is not None,
                'INVALID_SHA256 '+name)
    require(c.training_approved == 'false' and bool(c.label_review_status.strip()),
            'RESEARCH_LIMITATION_DISCLOSURE_DRIFT')
    require(c.research_acknowledged in ('true','false'), 'INVALID_RESEARCH_ACKNOWLEDGEMENT')
    require(c.optimizer_steps * c.global_batch == c.train_pairs and
            c.global_batch > 0 and c.global_batch % 2 == 0, 'BAD_TRAIN_BUDGET')
    require(sum(c.phase_steps)==c.optimizer_steps and
            sum(c.phase_source)+sum(c.phase_target)==c.train_pairs,
            'BAD_PHASE_BUDGET')
    require(len(set(c.phase_names))==3 and all(s > 0 for s in c.phase_steps),
            'BAD_PHASE_NAMES_OR_STEPS')
    for j in range(3):
        require(c.phase_source[j] + c.phase_target[j] == c.phase_steps[j]*c.global_batch,
                'PHASE_TOTAL_MISMATCH '+str(j))
    require(c.expected_imputed >= 0 and c.expected_variants >= 0,
            'NEGATIVE_EXPECTED_COUNTS')
    for name in ('pipeline','stage_run','g03b_run','g02_pipeline','g03a_run',
                 'audit_run','stress_run','stability_run','train_run','engine_path'):
        path = getattr(c,name)
        setattr(c,name, path.expanduser().resolve() if str(path) != '-' else None)
    require(c.engine_path.is_file() and sha(c.engine_path)==c.engine_sha256,
            'VENDORED_G02_MXFP4_KERNEL_SHA256_MISMATCH')
    require(c.stage_run.is_dir(), 'STAGE_DIRECTORY_MISSING')
    require(c.g02_pipeline.is_dir() and c.g03b_run.is_dir() and c.g03a_run.is_dir(),
            'SOURCE_DIRECTORY_MISSING')
    if c.stage in ('stress','stability','train'):
        require(c.research_acknowledged=='true',
                'BLOCKED_RESEARCH_GPU_TRAINING_REQUIRES_EXPLICIT_ACKNOWLEDGEMENT')
    elif c.stage == 'audit':
        require(c.research_acknowledged=='false' or c.research_acknowledged=='true',
                'INVALID_AUDIT_ACK')
    return c, raw[i+1:]


def load_engine(c, argv):
    spec = importlib.util.spec_from_file_location('g03c_vendored_m2_kernel', c.engine_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel_mode={'audit':'prepare','stress':'stress','stability':'stability',
                 'train':'train','verify':'prepare'}[c.stage]
    expected_mode = argv[0] if argv else None
    require(expected_mode==kernel_mode, 'KERNEL_MODE_MISMATCH')
    old = sys.argv
    try:
        sys.argv=[str(c.engine_path),*argv]
        a=module.parse()  # every original G-02 parameter is CLI-required
    finally:
        sys.argv=old
    require(a.run_dir==c.stage_run, 'SHELL_STAGE_RUN_DRIFT')
    require(a.model_id==c.model_id and a.revision==c.model_revision and
            a.m1_adapter_sha256==c.m1_adapter_sha256,
            'M1_MODEL_OR_ADAPTER_DRIFT')
    require(a.dataset_train_count==c.train_pairs and
            a.dataset_val_count==c.validation_pairs and
            a.max_steps==c.optimizer_steps and
            a.gradient_accumulation_steps*a.per_device_batch_size==c.global_batch and
            a.seed==c.seed, 'G03C_CONFIG_DRIFT')
    require(module.fingerprint(a)==c.g02_fingerprint,
            'M3_CORE_CONFIGURATION_NOT_IDENTICAL_TO_COMPLETED_G02')
    return module,a



def experiment_fingerprint(c, a, core):
    """All effective static M3 controls; run IDs/log paths intentionally excluded."""
    return hash_json({
        'version': VERSION, 'model_id':c.model_id, 'revision':c.model_revision,
        'g02_core_fingerprint':core.fingerprint(a),
        'g03b_manifest_sha256':c.g03b_manifest_sha256,
        'g03b_curriculum_order_sha256':c.g03b_order_sha256,
        'g02_frozen_train_sha256':c.g02_train_sha256,
        'g02_frozen_validation_sha256':c.g02_validation_sha256,
        'm1_adapter_sha256':c.m1_adapter_sha256,
        'phase_names':c.phase_names,'phase_steps':c.phase_steps,
        'phase_source':c.phase_source,'phase_target':c.phase_target,
        'optimizer_steps':c.optimizer_steps,'global_batch':c.global_batch,
        'train_pairs':c.train_pairs,'validation_pairs':c.validation_pairs,
        'seed':c.seed,'score_policy':c.phase_policy,
        'label_review_status':c.label_review_status,
        'independently_approved_for_training':False,
    })


def check_bound(path, expected, label):
    require(Path(path).is_file(), 'MISSING '+label+': '+str(path))
    require(sha(path)==expected, 'SHA256_DRIFT '+label)


def verify_input_order(c, a, core):
    """CPU-only cryptographic and record-level audit before GPU/shard loading."""
    # Revalidate original frozen M1 training provenance and F01c heldout overlap proof.
    core.verify_m1(a)
    core.verify_frozen(a)
    g03b=c.g03b_run
    gprep=c.g02_pipeline/'00-prepare'/'prepared'
    check_bound(g03b/'curriculum_manifest.json', c.g03b_manifest_sha256, 'G03B_MANIFEST')
    check_bound(g03b/'curriculum_order.jsonl', c.g03b_order_sha256, 'G03B_ORDER')
    check_bound(g03b/'curriculum_phase_audit.json', c.g03b_phase_audit_sha256, 'G03B_AUDIT')
    check_bound(g03b/'standard_order_comparison.json', c.g03b_comparison_sha256,
                'G03B_STANDARD_COMPARISON')
    check_bound(gprep/'manifest.json', c.g02_prepared_manifest_sha256, 'G02_PREPARED_MANIFEST')
    check_bound(gprep/'train.jsonl', c.g02_train_sha256, 'G02_TRAIN')
    check_bound(gprep/'validation.jsonl', c.g02_validation_sha256, 'G02_VALIDATION')
    check_bound(gprep/'order.jsonl', c.g02_standard_order_sha256, 'G02_STANDARD_ORDER')
    check_bound(c.g03a_run/'verified_scores.jsonl', c.g03a_score_sha256, 'G03A_SCORES')
    check_bound(a.m1_run/'adapter'/'adapter_model.safetensors', c.m1_adapter_sha256,
                'FROZEN_COMMON_M1_ADAPTER')
    cm=load(g03b/'curriculum_manifest.json')
    gm=load(gprep/'manifest.json')
    gp=load(c.g02_pipeline/'pipeline_result.json')
    m2=load(c.g02_pipeline/'03-train'/'training_summary.json')
    require(cm.get('version')=='v1' and cm.get('status')==G03B_STATUS and
            cm.get('approved_for_training') is False and
            cm.get('m2_adapter_used_to_initialize_m3') is False,
            'G03B_UNAPPROVED_STATE_OR_LINEAGE_CHANGED')
    require(cm.get('manual_label_review_status')==c.label_review_status and
            cm.get('curriculum_policy')==c.phase_policy and
            cm.get('imputed_pairs')==c.expected_imputed and
            cm.get('expected_structural_variants')==c.expected_variants,
            'G03B_SCORE_OR_REVIEW_POLICY_DRIFT')
    require(cm['source_g02_train_sha256']==c.g02_train_sha256 and
            cm['source_g02_validation_sha256']==c.g02_validation_sha256 and
            cm['source_g02_standard_order_sha256']==c.g02_standard_order_sha256 and
            cm['source_g03a_score_sha256']==c.g03a_score_sha256 and
            cm['common_frozen_m1_adapter_sha256']==c.m1_adapter_sha256 and
            cm['output_sha256']['curriculum_order.jsonl']==c.g03b_order_sha256 and
            cm['output_sha256']['curriculum_phase_audit.json']==c.g03b_phase_audit_sha256 and
            cm['output_sha256']['standard_order_comparison.json']==c.g03b_comparison_sha256,
            'G03B_INPUT_OUTPUT_PROVENANCE_DRIFT')
    require(cm['g02_pipeline']==str(c.g02_pipeline) and
            cm['source_g03a_manifest_sha256']==cm['input_pinned_hashes']['g03a_score_manifest_sha256'] and
            sha(c.g03a_run/'score_manifest.json')==cm['source_g03a_manifest_sha256'],
            'G03B_G03A_LINEAGE_DRIFT')
    require(cm['model_id']==a.model_id and cm['model_revision']==a.revision and
            cm['optimizer_steps']==c.optimizer_steps and cm['global_batch']==c.global_batch and
            cm['train_pairs']==c.train_pairs and
            cm['validation_pairs_unchanged']==c.validation_pairs,
            'G03B_MODEL_OR_EXPOSURE_BUDGET_DRIFT')
    require(gm.get('config_fingerprint')==c.g02_fingerprint and
            gm.get('status')=='G02_V2_PREPARE_PASS_NOT_TRAINED' and
            gm['files']['train']['sha256']==c.g02_train_sha256 and
            gm['files']['validation']['sha256']==c.g02_validation_sha256 and
            gm['files']['order']['sha256']==c.g02_standard_order_sha256,
            'G02_PREPARED_INTEGRITY_DRIFT')
    require(gp.get('status')=='G02_V2_FULL_M2_TRAINING_COMPLETE' and
            gp.get('exit_code')==0 and
            m2.get('optimizer_steps')==c.optimizer_steps and
            m2.get('initial_m1_adapter_sha256')==c.m1_adapter_sha256,
            'G02_REFERENCE_M2_NOT_COMPLETED_OR_WRONG_M1')
    require(all(cm['phase_plan'][i]['name']==c.phase_names[i] and
                cm['phase_plan'][i]['steps']==c.phase_steps[i] and
                cm['phase_plan'][i]['java_source']==c.phase_source[i] and
                cm['phase_plan'][i]['java_mixed']+cm['phase_plan'][i]['python_mixed']==c.phase_target[i]
                for i in range(3)), 'G03B_SHELL_PHASE_POLICY_MISMATCH')
    rows=list(records(gprep/'train.jsonl'))
    order=list(records(g03b/'curriculum_order.jsonl'))
    require(len(rows)==len(order)==c.train_pairs, 'INPUT_PAIR_COUNT_MISMATCH')
    scores=list(records(c.g03a_run/'verified_scores.jsonl'))
    require(len(scores)==c.train_pairs, 'SCORED_PAIR_COUNT_MISMATCH')
    # G03a scores are indexed by their original G02 prepared row order.
    score_by_idx={r['row_index']:r for r in scores}
    require(len(score_by_idx)==c.train_pairs and
            set(score_by_idx)==set(range(c.train_pairs)), 'G03A_SCORE_ROW_INDEX_DRIFT')
    seen=set(); label_hist=collections.Counter();phase_hist=collections.Counter()
    imputed=0;step_hashes=[]
    for position,item in enumerate(order):
        idx=item.get('row_index')
        require(type(idx) is int and 0<=idx<c.train_pairs and idx not in seen,
                'CURRICULUM_DUPLICATE_OR_BAD_INDEX position='+str(position))
        seen.add(idx)
        row=rows[idx];sr=score_by_idx[idx]
        require(item['position']==position and
                item['optimizer_step']==position//c.global_batch+1 and
                item['microbatch_index']==position%c.global_batch,
                'CURRICULUM_STEP_POSITION_DRIFT')
        require(item['g02_row_sha256']==hash_json(row) and
                item['source_row_sha256']==row['original_row_sha256'] and
                item['corpus']==row['corpus']==sr['corpus'] and
                item['label']==row['label']==sr['label'],
                'CURRICULUM_ROW_DATA_IDENTITY_DRIFT')
        require(item['imputed_proxy']==sr['imputed_proxy'] and
                math.isclose(item['reduced_4d_score'],sr['reduced_4d_score'],rel_tol=0,abs_tol=1e-12),
                'CURRICULUM_DIFFICULTY_SCORE_DRIFT')
        imputed+=int(bool(item['imputed_proxy']))
        ph=next((i for i,p in enumerate(cm['phase_plan']) if
                 p['first_optimizer_step']<=item['optimizer_step']<=p['last_optimizer_step']),None)
        require(ph is not None and item['phase_index']==ph and
                item['phase_name']==c.phase_names[ph], 'CURRICULUM_PHASE_IDENTITY_DRIFT')
        label_hist[item['optimizer_step'],item['label']]+=1
        phase_hist[ph,item['corpus'],item['label']]+=1
        if (position+1)%c.global_batch==0:
            block=order[position+1-c.global_batch:position+1]
            step_hashes.append(hash_json([x['row_index'] for x in block]))
    require(len(seen)==c.train_pairs and imputed==c.expected_imputed,
            'CURRICULUM_MISSING_ROWS_OR_IMPUTATIONS')
    for step in range(1,c.optimizer_steps+1):
        require(label_hist[step,0]==c.global_batch//2 and
                label_hist[step,1]==c.global_batch//2,
                'CURRICULUM_LABEL_IMBALANCE '+str(step))
    for ph,p in enumerate(cm['phase_plan']):
        for domain in DOMAINS:
            for label in (0,1):
                require(phase_hist[ph,domain,label]==p['corpus_label_counts'][domain][str(label)],
                        'CURRICULUM_PHASE_DOMAIN_LABEL_DRIFT '+str(ph)+':'+domain)
    expected_indices=[x['row_index'] for x in order]
    result={'version':VERSION,'status':AUDIT_STATUS,'at_chicago':chicago(),
            'g03b_manifest_sha256':c.g03b_manifest_sha256,
            'curriculum_order_sha256':c.g03b_order_sha256,
            'g02_prepared_manifest_sha256':c.g02_prepared_manifest_sha256,
            'g02_prepared_train_sha256':c.g02_train_sha256,
            'g02_prepared_validation_sha256':c.g02_validation_sha256,
            'm1_initial_adapter_sha256':c.m1_adapter_sha256,
            'g02_compatible_core_fingerprint':core.fingerprint(a),
            'g03c_experiment_fingerprint':experiment_fingerprint(c,a,core),
            'expected_row_index_ledger_sha256':hash_json(expected_indices),
            'expected_step_hashes_sha256':hash_json(step_hashes),
            'steps':len(step_hashes),'pairs':len(expected_indices),
            'imputed_pairs':imputed,'phase_names':c.phase_names,
            'approved_for_training':False,'manual_label_review_status':c.label_review_status,
            'heldout_evaluated':False,'gpu_weights_loaded':False,
            'limitations':['CPU order verification is not actual GPU dataloader consumption',
                           'Five heuristic score imputations and pending independent label review',
                           'M2 and M3 evaluation must share identical heldout Python test']}
    return result,rows,order,step_hashes


def verify_prior_audit(c, audit_dir, a, core):
    require(audit_dir is not None, 'MISSING_CPU_AUDIT_STAGE')
    result=load(audit_dir/'order_audit.json')
    require(result.get('status')==AUDIT_STATUS and
            result.get('curriculum_order_sha256')==c.g03b_order_sha256 and
            result.get('m1_initial_adapter_sha256')==c.m1_adapter_sha256 and
            result.get('g02_compatible_core_fingerprint')==core.fingerprint(a) and
            result.get('g03c_experiment_fingerprint')==experiment_fingerprint(c,a,core),
            'G03C_PRIOR_CPU_AUDIT_NOT_VALID')
    m=load(audit_dir/'audit_manifest.json')
    check_bound(audit_dir/'order_audit.json',m['order_audit_sha256'],'G03C_AUDIT_PROOF')
    require(m['status']==AUDIT_STATUS,'G03C_AUDIT_MANIFEST_DRIFT')
    return result


def load_gpu_context(c,a,core):
    tok,evidence=core.model_preflight(a)
    m1=core.verify_m1(a)
    require(core.sha_text(tok.chat_template)==m1['chat_template_sha256'],
            'GPTOSS_CHAT_TEMPLATE_MISMATCH_M1')
    _,rows,_=core.bind_prepared(a)
    need={key:set() for key in DOMAINS}
    for subset in rows.values():
        for row in subset:
            need[row['corpus']].update((str(row['id1']),str(row['id2'])))
    codes=core.load_codes(a,need)
    core.precache_prepared_tokens(a,tok,rows,codes)
    return tok,rows,codes,evidence


def run_smoke(c,a,core):
    verify_prior_audit(c,c.audit_run,a,core)
    verify_input_order(c,a,core)
    tok,rows,codes,_=load_gpu_context(c,a,core)
    prepared_hash=sha(c.g02_pipeline/'00-prepare'/'prepared'/'manifest.json')
    if c.stage=='stability':
        core.verify_stages(a,prepared_hash)
    core.smoke(a,tok,rows,codes,prepared_hash)
    proof={'version':VERSION,'status':'G03C_V1_'+c.stage.upper()+'_PASS_REUSING_PINNED_G02_KERNEL',
           'at_chicago':chicago(),'g03b_order_sha256':c.g03b_order_sha256,
           'm1_initial_adapter_sha256':c.m1_adapter_sha256,
           'core_runtime_status':'G02_V2_'+c.stage.upper()+'_PASS_NOT_M2_TRAINING',
           'g02_kernel_run_manifest_sha256':sha(a.run_dir/'run_manifest.json'),
           'not_full_m3_training':True,'training_approved':False,
           'research_pending_label_review_acknowledged':True}
    write_new(a.run_dir/'g03c_stage_proof.json',proof)
    print(json.dumps({'status':proof['status'],'at_chicago':chicago()}),flush=True)


def require_smoke_proofs(c,a,core):
    prepared_hash=c.g02_prepared_manifest_sha256
    core.verify_stages(a,prepared_hash)
    for stage,d in (('stress',c.stress_run),('stability',c.stability_run)):
        require(d is not None,'MISSING_'+stage.upper()+'_PROOF')
        proof=load(d/'g03c_stage_proof.json')
        require(proof['status']=='G03C_V1_'+stage.upper()+'_PASS_REUSING_PINNED_G02_KERNEL' and
                proof['g03b_order_sha256']==c.g03b_order_sha256 and
                proof['m1_initial_adapter_sha256']==c.m1_adapter_sha256 and
                proof['g02_kernel_run_manifest_sha256']==sha(d/'run_manifest.json'),
                'UNVERIFIED_M3_'+stage.upper()+'_PROOF')


def m3_checkpoint(c,a,core,model,opt,tok,step,ledger_digest):
    import torch,random
    root=a.run_dir/'checkpoints'
    root.mkdir(exist_ok=True)
    dest=root/('step-%06d'%step)
    dest.mkdir(exist_ok=False)
    model.save_pretrained(str(dest/'adapter'),safe_serialization=True)
    tok.save_pretrained(str(dest/'adapter'))
    torch.save({'optimizer':opt.state_dict(), 'torch_cpu_rng':torch.get_rng_state(),
        'torch_cuda_rng':torch.cuda.get_rng_state_all(), 'python_rng':random.getstate(),
        'optimizer_step':step,'g02_fingerprint':core.fingerprint(a),
        'prepared_manifest_sha256':c.g02_prepared_manifest_sha256,
        'g03b_curriculum_order_sha256':c.g03b_order_sha256,
        'g03c_experiment_fingerprint':experiment_fingerprint(c,a,core),
        'initial_m1_adapter_sha256':c.m1_adapter_sha256,
        'consumed_prefix_row_index_sha256':ledger_digest},dest/'optimizer_state.pt')
    files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}
    require('optimizer_state.pt' in files,'EMPTY_M3_CHECKPOINT')
    meta={'version':VERSION,'status':'G03C_V1_CURRICULUM_CHECKPOINT_COMPLETE',
          'at_chicago':chicago(),'optimizer_step':step,
          'g02_fingerprint':core.fingerprint(a),
          'prepared_manifest_sha256':c.g02_prepared_manifest_sha256,
          'g03b_curriculum_order_sha256':c.g03b_order_sha256,
          'g03c_experiment_fingerprint':experiment_fingerprint(c,a,core),
          'initial_m1_adapter_sha256':c.m1_adapter_sha256,
          'consumed_prefix_row_index_sha256':ledger_digest,
          'files_sha256':files,'optimizer_state_saved':True}
    write_new(dest/'checkpoint_manifest.json',meta)
    print(json.dumps({'event':'G03C_CHECKPOINT_PASS','step':step,
                      'path':str(dest),'at_chicago':chicago()}),flush=True)
    return dest


def run_train(c,a,core):
    import torch
    verify_prior_audit(c,c.audit_run,a,core)
    _,_,order,expected_hashes=verify_input_order(c,a,core)
    require_smoke_proofs(c,a,core)
    tok,rows,codes,_=load_gpu_context(c,a,core)
    # Explicitly load from FROZEN M1. Stress/stability adapters must never initialize full M3.
    model,trainable,placement=core.attach_model(a,resume_adapter=None)
    require(sha(a.m1_run/'adapter'/'adapter_model.safetensors')==c.m1_adapter_sha256,
            'M3_WRONG_INITIAL_ADAPTER')
    model.train()
    opt=torch.optim.AdamW([p for _,p in trainable],lr=a.learning_rate,
                          weight_decay=a.weight_decay)
    id_to_idx={id(row):idx for idx,row in enumerate(rows['train'])}
    require(len(id_to_idx)==len(rows['train']),'DUPLICATED_PYTHON_ROW_OBJECTS')
    fed=[]; expected_all=[x['row_index'] for x in order]
    actual_step_hashes=[];last=None;vals=[];ckpts=[]
    original=core.sample_tensors
    feed_enabled=[False]
    def tracked_sample(args,tokens,row,code_map,device):
        if feed_enabled[0]:
            require(id(row) in id_to_idx,'UNTRACKED_TRAINING_ROW')
            idx=id_to_idx[id(row)]
            require(len(fed)<len(expected_all) and idx==expected_all[len(fed)],
                    'ACTUAL_MICROBATCH_ORDER_MISMATCH pos='+str(len(fed)))
            require(hash_json(row_without_cache(row))==order[len(fed)]['g02_row_sha256'],
                    'ACTUAL_MICROBATCH_ROW_SHA_MISMATCH')
            fed.append(idx)
        return original(args,tokens,row,code_map,device)
    core.sample_tensors=tracked_sample
    history=a.run_dir/'training_history.jsonl'
    ledger=a.run_dir/'consumption_history.jsonl'
    try:
        with history.open('x',encoding='utf-8') as hist, ledger.open('x',encoding='utf-8') as cons:
            for step in range(1,c.optimizer_steps+1):
                begin=(step-1)*c.global_batch
                block=order[begin:begin+c.global_batch]
                require(len(block)==c.global_batch and
                        all(x['optimizer_step']==step for x in block),
                        'STEP_ORDER_NOT_CONTIGUOUS')
                micro=[rows['train'][item['row_index']] for item in block]
                rate=core.schedule_lr(a,step)
                feed_enabled[0]=True
                try:
                    log=core.optimizer_step(a,model,trainable,opt,tok,micro,codes,
                         step,rate,grad_accum=c.global_batch)
                finally:
                    feed_enabled[0]=False
                consumed=fed[begin:begin+c.global_batch]
                digest=hash_json(consumed)
                require(consumed==[r['row_index'] for r in block] and
                        digest==expected_hashes[step-1],
                        'ACTUAL_TRAINING_CONSUMPTION_STEP_MISMATCH '+str(step))
                actual_step_hashes.append(digest)
                phase=block[0]['phase_name']
                log.update({'curriculum_phase':phase,'curriculum_step_sha256':digest,
                            'g03b_order_sha256':c.g03b_order_sha256})
                last=log
                core.append_history(hist,{'event':'OPTIMIZER_STEP',**log})
                core.append_history(cons,{'step':step,'phase':phase,
                     'row_indices':consumed,'row_indices_sha256':digest,
                     'consumed_prefix_row_index_sha256':hash_json(fed),
                     'g03b_order_sha256':c.g03b_order_sha256,
                     'at_chicago':chicago()})
                if step==1 or step%a.log_steps==0 or step==c.optimizer_steps:
                    print(json.dumps({'event':'G03C_CURRICULUM_TRAIN_PROGRESS',**log}),flush=True)
                if step%a.eval_steps==0 or step==c.optimizer_steps:
                    val=core.evaluate(a,model,tok,rows['validation'],codes,step)
                    vals.append(val)
                    core.append_history(hist,{'event':'VALIDATION',**val})
                    print(json.dumps({'event':'G03C_FROZEN_MIXED_VALIDATION',**val}),flush=True)
                if step%a.save_steps==0 or step==c.optimizer_steps:
                    ck=m3_checkpoint(c,a,core,model,opt,tok,step,hash_json(fed))
                    ckpts.append(str(ck))
                    core.append_history(hist,{'event':'CHECKPOINT','optimizer_step':step,
                        'path':str(ck),'g03b_order_sha256':c.g03b_order_sha256,
                        'consumed_prefix_row_index_sha256':hash_json(fed),
                        'at_chicago':chicago()})
    finally:
        core.sample_tensors=original
    require(len(fed)==c.train_pairs and fed==expected_all and
            hash_json(actual_step_hashes)==hash_json(expected_hashes),
            'FINAL_FULL_CURRICULUM_CONSUMPTION_LEDGER_MISMATCH')
    final=a.run_dir/'adapter'
    require(not final.exists(),'REFUSE_OVERWRITE_M3_ADAPTER')
    model.save_pretrained(str(final),safe_serialization=True)
    tok.save_pretrained(str(final))
    files={str(p.relative_to(final)):sha(p) for p in final.rglob('*') if p.is_file()}
    require('adapter_model.safetensors' in files,'M3_FINAL_ADAPTER_MISSING')
    write_new(a.run_dir/'adapter_manifest.json',{
        'version':VERSION,'status':'G03C_V1_M3_ADAPTER_COMPLETE',
        'adapter_files_sha256':files,'initial_m1_adapter_sha256':c.m1_adapter_sha256,
        'g03b_order_sha256':c.g03b_order_sha256,'optimizer_steps':c.optimizer_steps,
        'consumption_ledger_sha256':sha(ledger),'at_chicago':chicago()})
    summary={'version':VERSION,'status':TRAIN_STATUS,'at_chicago':chicago(),
        'model_id':a.model_id,'model_revision':a.revision,
        'g02_compatible_core_fingerprint':core.fingerprint(a),
        'g03c_experiment_fingerprint':experiment_fingerprint(c,a,core),
        'g03b_manifest_sha256':c.g03b_manifest_sha256,
        'g03b_order_sha256':c.g03b_order_sha256,
        'g02_frozen_train_sha256':c.g02_train_sha256,
        'g02_frozen_validation_sha256':c.g02_validation_sha256,
        'initial_m1_adapter_sha256':c.m1_adapter_sha256,
        'm2_adapter_used_as_initial_m3_weights':False,
        'training_order':'G03B_V1_VERIFIED_THREE_PHASE_CURRICULUM',
        'optimizer_steps':c.optimizer_steps,'samples_per_step':c.global_batch,
        'total_unique_consumed_pairs':len(fed),'training_consumption_ledger_sha256':sha(ledger),
        'training_history_sha256':sha(history),
        'consumed_order_sha256':hash_json(fed),
        'expected_order_row_index_sha256':hash_json(expected_all),
        'per_step_ledger_sha256':hash_json(actual_step_hashes),
        'frozen_mixed_validation_events':vals,'last_validation':vals[-1],
        'checkpoints':ckpts,'gpu_placement':placement,
        'adapter_manifest_sha256':sha(a.run_dir/'adapter_manifest.json'),
        'token_audit_sha256':sha(a.run_dir/'token_audit.json'),
        'final_adapter_path':str(final),
        'm3_effective_parameters':{k:(str(v) if isinstance(v,Path) else v)
            for k,v in vars(c).items() if k not in
            ('stage','pipeline','stage_run','audit_run','stress_run','stability_run','train_run')},
        'approved_for_training':False,'research_pending_review_acknowledged':True,
        'manual_label_review_status':c.label_review_status,
        'difficulty_heuristic_not_independently_calibrated':True,
        'heldout_test_evaluated':False,
        'effective_parameters':{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()},
        'limitations':['Exploratory M3 run; not independently label-approved',
                       'Not an independent held-out test result',
                       'Single-process 3-GPU balanced sharding, not DDP/FSDP',
                       'M1 common initialization; G02 M2 adapter never loaded']}
    write_new(a.run_dir/'training_summary.json',summary)
    write_new(a.run_dir/'g03c_train_manifest.json',{
       'version':VERSION,'status':TRAIN_STATUS,'g03b_manifest_sha256':c.g03b_manifest_sha256,
       'g03b_order_sha256':c.g03b_order_sha256,
       'training_summary_sha256':sha(a.run_dir/'training_summary.json'),
       'training_history_sha256':sha(history),
       'consumption_history_sha256':sha(ledger),
       'adapter_manifest_sha256':sha(a.run_dir/'adapter_manifest.json'),
       'at_chicago':chicago()})
    print(json.dumps({'status':TRAIN_STATUS,'steps':c.optimizer_steps,
        'train_pairs':len(fed),'g03b_order_sha256':c.g03b_order_sha256,
        'consumed_order_sha256':hash_json(fed),'final_adapter':str(final),
        'at_chicago':chicago()}),flush=True)


def row_without_cache(row):
    return {key:val for key,val in row.items() if not key.startswith('_')}


def verify_training(c,a,core):
    require(c.train_run is not None and c.train_run.is_dir(), 'M3_TRAINING_RUN_MISSING')
    s=load(c.train_run/'training_summary.json')
    m=load(c.train_run/'g03c_train_manifest.json')
    am=load(c.train_run/'adapter_manifest.json')
    require(s.get('status')==TRAIN_STATUS and m.get('status')==TRAIN_STATUS and
            am.get('status')=='G03C_V1_M3_ADAPTER_COMPLETE',
            'M3_TRAINING_STATUS_INCOMPLETE')
    require(s['g03b_order_sha256']==c.g03b_order_sha256 and
            s['initial_m1_adapter_sha256']==c.m1_adapter_sha256 and
            s['m2_adapter_used_as_initial_m3_weights'] is False and
            s['optimizer_steps']==c.optimizer_steps and
            s['total_unique_consumed_pairs']==c.train_pairs and
            s['g02_compatible_core_fingerprint']==c.g02_fingerprint and
            s['g03c_experiment_fingerprint']==experiment_fingerprint(c,a,core),
            'M3_TRAINING_IDENTITY_DRIFT')
    for key,file in (('training_summary_sha256','training_summary.json'),
                     ('training_history_sha256','training_history.jsonl'),
                     ('consumption_history_sha256','consumption_history.jsonl'),
                     ('adapter_manifest_sha256','adapter_manifest.json')):
        check_bound(c.train_run/file, m[key], key)
    for relative,digest in am['adapter_files_sha256'].items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                'UNSAFE_ADAPTER_RELATIVE_PATH')
        check_bound(c.train_run/'adapter'/relative,digest,'M3_ADAPTER_FILE')
    expected=[x['row_index'] for x in records(c.g03b_run/'curriculum_order.jsonl')]
    cons=list(records(c.train_run/'consumption_history.jsonl'))
    require(len(cons)==c.optimizer_steps,'M3_LEDGER_STEPS_MISMATCH')
    flat=[]
    for ix,event in enumerate(cons,1):
        indices=event['row_indices']; flat.extend(indices)
        require(event['step']==ix and len(indices)==c.global_batch and
                event['row_indices_sha256']==hash_json(indices) and
                event['consumed_prefix_row_index_sha256']==hash_json(flat) and
                event['g03b_order_sha256']==c.g03b_order_sha256,
                'M3_LEDGER_CHECKSUM_OR_STEP_DRIFT')
    require(flat==expected and len(set(flat))==c.train_pairs and
            s['consumed_order_sha256']==hash_json(flat) and
            s['per_step_ledger_sha256']==hash_json([x['row_indices_sha256'] for x in cons]),
            'M3_LEDGER_ORDER_MISMATCH')
    ckpts=[]
    for folder in sorted((c.train_run/'checkpoints').glob('step-*')):
        meta=load(folder/'checkpoint_manifest.json')
        step=meta['optimizer_step']
        require(meta['status']=='G03C_V1_CURRICULUM_CHECKPOINT_COMPLETE' and
                meta['g03b_curriculum_order_sha256']==c.g03b_order_sha256 and
                meta['g03c_experiment_fingerprint']==experiment_fingerprint(c,a,core) and
                meta['initial_m1_adapter_sha256']==c.m1_adapter_sha256 and
                meta['consumed_prefix_row_index_sha256']==hash_json(expected[:step*c.global_batch]),
                'M3_CHECKPOINT_LINEAGE_DRIFT')
        for relative,digest in meta['files_sha256'].items():
            require('..' not in Path(relative).parts and not Path(relative).is_absolute(),
                    'UNSAFE_CHECKPOINT_PATH')
            check_bound(folder/relative,digest,'CHECKPOINT_CONTENT')
        ckpts.append(step)
    require(ckpts and ckpts[-1]==c.optimizer_steps,
            'FINAL_M3_CHECKPOINT_NOT_VERIFIED')
    out={'version':VERSION,'status':'G03C_V1_INDEPENDENT_FULL_M3_VERIFY_PASS',
         'run':str(c.train_run),'steps':c.optimizer_steps,'pairs':c.train_pairs,
         'curriculum_order_sha256':c.g03b_order_sha256,
         'adapter_model_sha256':am['adapter_files_sha256']['adapter_model.safetensors'],
         'checkpoints_verified':ckpts,'heldout_evaluated':False,
         'approved_for_training':False,'at_chicago':chicago()}
    write_new(c.stage_run/'verification_result.json',out)
    print(json.dumps(out),flush=True)


def main():
    c,argv=parse_control()
    core,a=load_engine(c,argv)
    if c.stage=='audit':
        report,_,_,_=verify_input_order(c,a,core)
        write_new(c.stage_run/'order_audit.json',report)
        write_new(c.stage_run/'audit_manifest.json',{
            'version':VERSION,'status':AUDIT_STATUS,
            'order_audit_sha256':sha(c.stage_run/'order_audit.json'),
            'g03b_order_sha256':c.g03b_order_sha256,'at_chicago':chicago()})
        print(json.dumps({'status':AUDIT_STATUS,'pairs':report['pairs'],
                          'steps':report['steps'],'gpu_training_started':False,
                          'at_chicago':chicago()}),flush=True)
    elif c.stage in ('stress','stability'):
        run_smoke(c,a,core)
    elif c.stage=='train':
        run_train(c,a,core)
    elif c.stage=='verify':
        verify_input_order(c,a,core)
        verify_training(c,a,core)


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print(json.dumps({'event':'G03C_FAIL','error_type':type(e).__name__,
                          'error':str(e),'at_chicago':chicago()}),file=sys.stderr,flush=True)
        raise
