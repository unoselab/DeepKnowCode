#!/usr/bin/env python3
"""I-01b v1. Gemma 4 31B-IT source-only frozen Java pair preparation.

Reuses exact F-01c/G-01c pair IDs, labels and order. Full per-pair Gemma text
chat-template + contextual YES/NO token audit; no training/model weight loading.
All experiment-specific parameters are required CLI arguments from the Shell.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


class PreflightError(RuntimeError):
    pass


def need(ok, reason):
    if not ok:
        raise PreflightError(reason)


def chicago():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def canonical_json(data):
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')


def write_rows_new(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(canonical_json(row) + '\n')


def read_jsonl(path):
    with Path(path).open('r', encoding='utf-8-sig') as f:
        for n, line in enumerate(f, 1):
            need(bool(line.strip()), f'EMPTY_JSONL_ROW:{path}:{n}')
            row = json.loads(line)
            need(isinstance(row, dict), f'NON_OBJECT_JSONL_ROW:{path}:{n}')
            yield row


def read_file_pin(path, expected, name):
    path = Path(path)
    need(path.is_file(), f'REQUIRED_FILE_MISSING:{name}:{path}')
    need(re.fullmatch('[0-9a-f]{64}', expected) is not None, f'INVALID_PIN:{name}')
    observed = sha256(path)
    need(observed == expected, f'SHA256_MISMATCH:{name}:{observed}')
    return {'path': str(path.resolve()), 'sha256': observed, 'bytes': path.stat().st_size}


def flag(text):
    need(text in ('true', 'false'), f'EXPECTED_BOOLEAN_STRING:{text}')
    return text == 'true'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    str_names = '''version experiment mode conda-env model-id model-mode model-revision hf-home
      chat-template-sha256 expected-config-sha256 expected-tokenizer-config-sha256
      tokenizer-policy tokenizer-use-fast tokenizer-local-only trust-remote-code enable-thinking
      base-precision finetuning-method quantization lora-dropout lora-bias lora-task-type
      lora-target-regex device-ids device-map learning-rate answer-yes answer-no
      system-text user-prefix user-separator user-suffix probe-code-a probe-code-b
      i01a-preflight-sha256 i01a-tokenizer-sha256 i01a-lora-sha256 i01a-weights-sha256
      frozen-train-sha256 frozen-validation-sha256 frozen-manifest-sha256
      g01c-manifest-sha256 g01c-train-sha256 g01c-validation-sha256 g01c-order-sha256
      log-prefix log-timestamp-format log-tee log-stderr stage'''.split()
    path_names = '''root run-dir log-file verify-source snapshot i01a-run frozen-prepared
      source-codes g01c-prepared log-dir'''.split()
    int_names = '''lora-rank lora-alpha expected-lora-target-modules expected-trainable-parameters
      expected-gpus seed per-device-batch-size gradient-accumulation-steps max-steps
      expected-epochs max-seq-length audit-progress-every expected-yes-token-id
      expected-no-token-id frozen-train-count frozen-validation-count source-train-count
      source-validation-count source-train-positives source-train-negatives
      source-validation-positives source-validation-negatives'''.split()
    for name in str_names:
        p.add_argument('--' + name, type=str, required=True)
    for name in path_names:
        p.add_argument('--' + name, type=Path, required=True)
    for name in int_names:
        p.add_argument('--' + name, type=int, required=True)
    p.add_argument('--run-status', choices=('PASS', 'FAIL'), required=False)
    p.add_argument('--exit-code', type=int, required=False)
    a = p.parse_args(argv)
    need(a.stage in ('provenance', 'frozen', 'tokenizer', 'prepare', 'verify', 'finalize'),
         'UNKNOWN_STAGE')
    need(a.mode in ('--prepare', '--verify'), 'UNKNOWN_MODE')
    if a.stage == 'finalize':
        need(a.run_status in ('PASS', 'FAIL') and a.exit_code is not None,
             'FINALIZE_REQUIRES_STATUS_AND_EXIT_CODE')
    else:
        need(a.run_status is None and a.exit_code is None, 'UNEXPECTED_FINALIZE_FIELDS')
    for name in ('tokenizer_use_fast', 'tokenizer_local_only', 'trust_remote_code',
                 'enable_thinking', 'log_tee', 'log_stderr'):
        setattr(a, name, flag(getattr(a, name)))
    return a


def parameters(a):
    # Every effective experiment value, including shell path, flows through required CLI.
    exclude = {'stage', 'run_status', 'exit_code'}
    return {k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(a).items() if k not in exclude}


def validate_config(a):
    need(a.version == 'v1' and a.experiment == 'i-01b', 'WRONG_EXPERIMENT_OR_VERSION')
    need(a.model_id == 'google/gemma-4-31B-it' and a.model_mode == 'gemma-4-31b-it',
         'MODEL_IDENTITY_MISMATCH')
    need(re.fullmatch('[0-9a-f]{40}', a.model_revision) is not None,
         'UNPINNED_MODEL_REVISION')
    need(a.snapshot.name == a.model_revision and a.snapshot.is_dir(),
         'PINNED_MODEL_SNAPSHOT_MISSING')
    need(a.finetuning_method == 'lora' and a.base_precision == 'bf16' and
         a.quantization == 'none', 'NO_QLORA_BF16_LORA_POLICY_MISMATCH')
    need(a.lora_rank > 0 and a.lora_alpha > 0 and float(a.lora_dropout) >= 0 and
         float(a.lora_dropout) < 1 and a.lora_bias == 'none' and
         a.lora_task_type == 'CAUSAL_LM', 'LORA_SETTINGS_INVALID')
    need(a.expected_lora_target_modules > 0 and a.expected_trainable_parameters > 0,
         'LORA_PROOF_COUNTS_INVALID')
    need(a.tokenizer_policy == 'pinned-jinja-autotokenizer-text-only' and
         a.tokenizer_local_only and not a.trust_remote_code and not a.enable_thinking,
         'GEMMA_TEXT_ONLY_TOKENIZER_POLICY_DRIFT')
    need(a.expected_yes_token_id > 0 and a.expected_no_token_id > 0 and
         a.expected_yes_token_id != a.expected_no_token_id,
         'YES_NO_IDS_INVALID')
    need(a.answer_yes != a.answer_no and a.answer_yes.strip() and a.answer_no.strip(),
         'YES_NO_LABELS_INVALID')
    need(a.source_train_count == a.source_train_positives + a.source_train_negatives and
         a.source_validation_count == a.source_validation_positives + a.source_validation_negatives,
         'EXPECTED_LABEL_COUNTS_INCONSISTENT')
    need(a.source_train_count == a.max_steps * a.gradient_accumulation_steps *
         a.per_device_batch_size * a.expected_epochs,
         'TRAINING_BUDGET_CANNOT_CONSUME_SOURCE_EXACTLY_ONCE')
    need(a.expected_epochs == 1 and a.max_steps > 0 and a.source_train_count > 0,
         'TRAINING_BUDGET_INVALID')
    need(a.frozen_train_count >= a.source_train_count and
         a.frozen_validation_count >= a.source_validation_count,
         'FROZEN_COUNTS_TOO_SMALL')
    need(a.max_seq_length > 0 and a.audit_progress_every > 0,
         'TOKEN_AUDIT_LIMITS_INVALID')
    need(a.seed >= 0 and a.expected_gpus > 0 and len(a.device_ids.split(',')) == a.expected_gpus,
         'GPU_OR_SEED_CONFIGURATION_INVALID')
    need(a.learning_rate and float(a.learning_rate) > 0, 'LEARNING_RATE_INVALID')
    need((a.mode == '--prepare' and str(a.verify_source) == 'NONE') or
         (a.mode == '--verify' and str(a.verify_source) != 'NONE'),
         'VERIFY_SOURCE_MODE_MISMATCH')
    need(a.run_dir.is_dir(), 'RUN_DIR_MISSING')
    need(a.log_file.is_file(), 'LOG_FILE_MISSING')
    need(a.root.is_dir(), 'ROOT_MISSING')
    need(os.environ.get('CONDA_DEFAULT_ENV') == a.conda_env,
         'CONDA_ENV_MISMATCH')


def provenance(a):
    evidence = {
        'preflight': read_file_pin(a.i01a_run/'preflight.json', a.i01a_preflight_sha256, 'I01A_PREFLIGHT'),
        'tokenizer': read_file_pin(a.i01a_run/'tokenizer.json', a.i01a_tokenizer_sha256, 'I01A_TOKENIZER'),
        'lora_probe': read_file_pin(a.i01a_run/'lora_probe.json', a.i01a_lora_sha256, 'I01A_LORA_PROBE'),
        'weights': read_file_pin(a.i01a_run/'weights.json', a.i01a_weights_sha256, 'I01A_WEIGHTS'),
        'config': read_file_pin(a.snapshot/'config.json', a.expected_config_sha256, 'GEMMA_CONFIG'),
        'tokenizer_config': read_file_pin(a.snapshot/'tokenizer_config.json', a.expected_tokenizer_config_sha256, 'GEMMA_TOKENIZER_CONFIG'),
        'chat_template': read_file_pin(a.snapshot/'chat_template.jinja', a.chat_template_sha256, 'GEMMA_CHAT_TEMPLATE'),
    }
    proof = load_json(a.i01a_run/'preflight.json')
    need(proof.get('status') == 'LORA_BACKWARD_PROBE_PASS_NOT_TRAINING' and
         proof.get('version') == 'v3' and proof.get('shell_status') == 'PASS' and
         all(proof.get('stages', {}).get(stage) == 'PASS' for stage in
             ('cache','gpu','environment','tokenizer','weights','lora_probe')),
         'I01A_VERIFIED_PROOF_NOT_PASS')
    prev = proof.get('effective_parameters', {})
    for key, expected in [('model_id',a.model_id),('model_revision',a.model_revision),
                          ('chat_template_sha256',a.chat_template_sha256),
                          ('base_precision',a.base_precision),('quantization',a.quantization),
                          ('seed',a.seed),('max_seq_length',a.max_seq_length),
                          ('answer_yes',a.answer_yes),('answer_no',a.answer_no),
                          ('system_text',a.system_text),('user_prefix',a.user_prefix),
                          ('user_separator',a.user_separator),('user_suffix',a.user_suffix),
                          ('probe_code_a',a.probe_code_a),('probe_code_b',a.probe_code_b),
                          ('lora_rank',a.lora_rank),('lora_alpha',a.lora_alpha),
                          ('lora_dropout',float(a.lora_dropout)),('lora_bias',a.lora_bias),
                          ('lora_target_regex',a.lora_target_regex),
                          ('device_ids',a.device_ids),('expected_gpus',a.expected_gpus),
                          ('device_map',a.device_map),('learning_rate',float(a.learning_rate))]:
        need(prev.get(key) == expected, f'I01A_PARAMETER_DRIFT:{key}')
    tok = load_json(a.i01a_run/'tokenizer.json')['details']
    need(tok['pinned_chat_template_sha256'] == a.chat_template_sha256,
         'I01A_TEMPLATE_PIN_DRIFT')
    need(tok['label_token_ids']['YES']['token_id'] == a.expected_yes_token_id and
         tok['label_token_ids']['NO']['token_id'] == a.expected_no_token_id,
         'I01A_LABEL_TOKEN_ID_DRIFT')
    probe = load_json(a.i01a_run/'lora_probe.json')['details']
    need(probe['bf16_base_loaded'] is True and probe['quantization'] == 'none' and
         probe['optimizer_steps'] == 0 and probe['checkpoint_saved'] is False and
         probe['actual_target_modules'] == a.expected_lora_target_modules and
         probe['trainable_lora_parameters'] == a.expected_trainable_parameters,
         'I01A_LORA_FEASIBILITY_PROOF_DRIFT')
    need(load_json(a.i01a_run/'weights.json')['details']['full_weight_sha256_verified'] is True,
         'I01A_WEIGHT_AUDIT_NOT_COMPLETE')
    return {'i01a_version': 'v3', 'proof_status': proof['status'],
            'model_identity': {'model_id':a.model_id,'revision':a.model_revision,
                               'chat_template_sha256':a.chat_template_sha256},
            'provenance_files': evidence, 'heldout_accessed':False}


def frozen(a):
    pins = {
        'f01c_train': read_file_pin(a.frozen_prepared/'train.jsonl', a.frozen_train_sha256, 'F01C_TRAIN'),
        'f01c_validation': read_file_pin(a.frozen_prepared/'validation.jsonl', a.frozen_validation_sha256, 'F01C_VALIDATION'),
        'f01c_manifest': read_file_pin(a.frozen_prepared/'manifest.json', a.frozen_manifest_sha256, 'F01C_MANIFEST'),
        'g01c_train': read_file_pin(a.g01c_prepared/'train.jsonl', a.g01c_train_sha256, 'G01C_TRAIN'),
        'g01c_validation': read_file_pin(a.g01c_prepared/'validation.jsonl', a.g01c_validation_sha256, 'G01C_VALIDATION'),
        'g01c_order': read_file_pin(a.g01c_prepared/'order.jsonl', a.g01c_order_sha256, 'G01C_ORDER'),
        'g01c_manifest': read_file_pin(a.g01c_prepared/'manifest.json', a.g01c_manifest_sha256, 'G01C_MANIFEST'),
    }
    frozen_manifest = load_json(a.frozen_prepared/'manifest.json')
    source_digest = frozen_manifest['dataset']['sha256']['java_source']['data.jsonl']
    pins['source_codes'] = read_file_pin(a.source_codes, source_digest, 'JAVA_SOURCE_CODE')
    selections = {}
    counts = {}
    for split in ('train','validation'):
        path = a.frozen_prepared / (split + '.jsonl')
        rows = []
        seen = set()
        labels = collections.Counter()
        total = 0
        for record in read_jsonl(path):
            total += 1
            need(record.get('corpus') in ('java_source','java_mixed','python_mixed'),
                 f'UNKNOWN_FROZEN_CORPUS:{split}')
            need(type(record.get('label')) is int and record['label'] in (0,1),
                 f'INVALID_FROZEN_LABEL:{split}')
            if record['corpus'] != 'java_source':
                continue
            need(not record.get('code_a_override') and not record.get('code_b_override'),
                 'FROZEN_SOURCE_CODE_OVERRIDE_NOT_ALLOWED')
            row = {'id1':str(record['id1']), 'id2':str(record['id2']), 'label':record['label']}
            key = (row['id1'],row['id2'],row['label'])
            need(key not in seen, f'FROZEN_SOURCE_PAIR_DUPLICATE:{split}')
            seen.add(key)
            rows.append(row)
            labels[row['label']] += 1
        need(total == (a.frozen_train_count if split=='train' else a.frozen_validation_count),
             f'FROZEN_TOTAL_ROW_COUNT_DRIFT:{split}:{total}')
        expected_n = a.source_train_count if split=='train' else a.source_validation_count
        expected_p = a.source_train_positives if split=='train' else a.source_validation_positives
        expected_neg = a.source_train_negatives if split=='train' else a.source_validation_negatives
        need(len(rows) == expected_n and labels[1]==expected_p and labels[0]==expected_neg,
             f'FROZEN_SOURCE_LABEL_OR_COUNT_DRIFT:{split}')
        reference = list(read_jsonl(a.g01c_prepared / (split+'.jsonl')))
        need(len(reference) == len(rows), f'G01C_SOURCE_COUNT_DRIFT:{split}')
        for index, (item, ref) in enumerate(zip(rows,reference)):
            need(all(ref.get(k) == item[k] for k in ('id1','id2','label')),
                 f'G01C_PAIR_IDENTITY_DRIFT:{split}:{index}')
        selections[split] = rows
        counts[split] = {'rows':len(rows), 'labels':{'0':labels[0], '1':labels[1]}}
    train_keys = {(r['id1'],r['id2']) for r in selections['train']}
    valid_keys = {(r['id1'],r['id2']) for r in selections['validation']}
    need(not (train_keys & valid_keys), 'SOURCE_TRAIN_VALIDATION_PAIR_LEAKAGE')
    indices = []
    for i, row in enumerate(read_jsonl(a.g01c_prepared / 'order.jsonl')):
        index = row.get('index')
        need(type(index) is int and 0 <= index < a.source_train_count,
             f'G01C_ORDER_INVALID_INDEX:{i}')
        indices.append(index)
    expected_indices = list(range(a.source_train_count))
    random.Random(a.seed).shuffle(expected_indices)
    need(indices == expected_indices, 'G01C_TRAINING_ORDER_SEED_DRIFT')
    need(len(set(indices)) == len(indices) == a.source_train_count,
         'G01C_TRAINING_ORDER_NOT_PERMUTATION')
    needed_ids = {row[k] for split in selections.values() for row in split for k in ('id1','id2')}
    codes = {}
    for record in read_jsonl(a.source_codes):
        rid = str(record.get('idx'))
        if rid in needed_ids:
            need(rid not in codes and isinstance(record.get('func'),str) and record['func'].strip(),
                 f'JAVA_SOURCE_FUNCTION_DUPLICATE_OR_EMPTY:{rid}')
            codes[rid] = record['func']
    need(set(codes) == needed_ids,
         f'JAVA_SOURCE_FUNCTION_IDS_MISSING:{len(needed_ids-set(codes))}')
    # Frozen Gemma data must retain all G01C source code identities across models.
    for split in ('train','validation'):
        for i,(row,ref) in enumerate(zip(selections[split],read_jsonl(a.g01c_prepared/(split+'.jsonl')))):
            need(ref.get('code1_sha256') == text_sha(codes[row['id1']]) and
                 ref.get('code2_sha256') == text_sha(codes[row['id2']]),
                 f'G01C_CODE_CONTENT_DRIFT:{split}:{i}')
    return {'selections':selections, 'codes':codes, 'order':indices,
            'evidence': {'frozen_files':pins,'source_counts':counts,
                         'selected_code_count':len(codes),
                         'pair_overlap_train_validation':0,
                         'pair_identity_equals_g01c':True,
                         'training_order_equals_g01c':True,
                         'source_identity_policy':'F01C_PINNED_JAVA_SOURCE_SUBSET',
                         'heldout_accessed':False}}


def load_tokenizer(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(a.snapshot), use_fast=a.tokenizer_use_fast,
                                        local_files_only=a.tokenizer_local_only,
                                        trust_remote_code=a.trust_remote_code)
    template = (a.snapshot/'chat_template.jinja').read_text(encoding='utf-8')
    need(text_sha(template) == a.chat_template_sha256,
         'GEMMA_CHAT_TEMPLATE_SHA_DRIFT')
    return tok, template


def render_prompt(a, tok, template, first, second):
    messages = [
        {'role':'system', 'content':a.system_text},
        {'role':'user','content':a.user_prefix+first+a.user_separator+second+a.user_suffix},
    ]
    try:
        text = tok.apply_chat_template(messages, chat_template=template, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=a.enable_thinking)
    except Exception as exc:
        raise PreflightError(f'GEMMA_PROMPT_TEMPLATE_RENDER_FAILED:{exc}') from exc
    need(isinstance(text,str) and bool(text.strip()), 'EMPTY_RENDERED_GEMMA_PROMPT')
    return text


def contextual_ids(a, tok, prompt):
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    need(isinstance(prompt_ids,list) and bool(prompt_ids), 'EMPTY_PROMPT_TOKEN_IDS')
    token_ids = {}
    for name,answer in [('YES',a.answer_yes),('NO',a.answer_no)]:
        full_ids = tok.encode(prompt+answer, add_special_tokens=False)
        need(full_ids[:len(prompt_ids)] == prompt_ids and
             len(full_ids) == len(prompt_ids) + 1,
             f'CONTEXTUAL_{name}_NOT_ONE_TOKEN')
        token_ids[name] = full_ids[-1]
    need(token_ids['YES'] == a.expected_yes_token_id and
         token_ids['NO'] == a.expected_no_token_id,
         f'GEMMA_CONTEXTUAL_YES_NO_TOKEN_ID_DRIFT:{token_ids}')
    need(len(prompt_ids)+1 <= a.max_seq_length,
         f'NO_TRUNCATION_MAX_TOKENS_EXCEEDED:{len(prompt_ids)+1}>{a.max_seq_length}')
    return prompt_ids,token_ids


def tokenizer(a):
    tok,template=load_tokenizer(a)
    p=render_prompt(a,tok,template,a.probe_code_a,a.probe_code_b)
    ids,labels=contextual_ids(a,tok,p)
    need(labels == {'YES':a.expected_yes_token_id,'NO':a.expected_no_token_id},
         'GEMMA_TOKENIZER_PROBE_DRIFT')
    return {'tokenizer_class':type(tok).__name__,
            'pinned_chat_template_sha256':text_sha(template),
            'probe_prompt_tokens':len(ids),
            'label_token_ids':labels,
            'text_only':True,'processor_instantiated':False,
            'heldout_accessed':False}


def audit_pair(a,tok,template,row,codes):
    prompt=render_prompt(a,tok,template,codes[row['id1']],codes[row['id2']])
    ids,labels=contextual_ids(a,tok,prompt)
    return {'id1':row['id1'],'id2':row['id2'],'label':row['label'],
            'code1_sha256':text_sha(codes[row['id1']]),
            'code2_sha256':text_sha(codes[row['id2']]),
            'prompt_sha256':text_sha(prompt),
            'n_tokens':len(ids)+1,
            'answer_id':labels['YES'] if row['label']==1 else labels['NO']}


def token_stats(rows):
    vals=sorted(r['n_tokens'] for r in rows)
    need(bool(vals), 'TOKEN_STATS_EMPTY')
    return {'min':vals[0], 'median':vals[len(vals)//2],
            'p95':vals[math.ceil(0.95*len(vals))-1],
            'max':vals[-1], 'mean':sum(vals)/len(vals)}


def prepared_folder(a):
    return (a.run_dir if a.mode=='--prepare' else a.verify_source) / 'prepared'


def prepare(a):
    frozen_data=frozen(a)
    tok,template=load_tokenizer(a)
    pdir=prepared_folder(a)
    need(not pdir.exists(), f'OUTPUT_ALREADY_EXISTS:{pdir}')
    pdir.mkdir(parents=False,exist_ok=False)
    outputs={}
    stats={}
    for split in ('train','validation'):
        records=[]
        for i,row in enumerate(frozen_data['selections'][split], 1):
            records.append(audit_pair(a,tok,template,row,frozen_data['codes']))
            if i % a.audit_progress_every == 0 or i == len(frozen_data['selections'][split]):
                print(json.dumps({'event':'I01B_TOKEN_AUDIT_PROGRESS','split':split,
                                  'processed':i,'chicago':chicago()}),flush=True)
        f=pdir/(split+'.jsonl')
        write_rows_new(f,records)
        outputs[split]={'path':str(f),'sha256':sha256(f),'rows':len(records)}
        stats[split]=token_stats(records)
        print(json.dumps({'event':'I01B_SPLIT_PASS','split':split,
                          'stats':stats[split],'count':len(records),'chicago':chicago()}),flush=True)
    rows_by_index=list(read_jsonl(pdir/'train.jsonl'))
    order=[{'index':idx,'row_sha256':text_sha(canonical_json(rows_by_index[idx]))}
           for idx in frozen_data['order']]
    order_path=pdir/'order.jsonl'
    write_rows_new(order_path,order)
    outputs['order']={'path':str(order_path),'sha256':sha256(order_path),'rows':len(order)}
    config=parameters(a)
    fp=text_sha(canonical_json({k:v for k,v in config.items()
                               if k not in ('run_dir','log_file','log_dir','mode','verify_source')}))
    manifest={'version':a.version,'experiment':a.experiment,
              'status':'I01B_V1_SOURCE_PREPARE_PASS_NOT_TRAINED',
              'at_chicago':chicago(),'config_fingerprint':fp,
              'effective_parameters':config,
              'frozen_source':frozen_data['evidence'],
              'i01a_provenance_sha256':a.i01a_preflight_sha256,
              'model':{'model_id':a.model_id,'revision':a.model_revision,
                       'base_precision':a.base_precision,'quantization':a.quantization,
                       'fine_tuning_method':a.finetuning_method},
              'tokenizer_identity':{'chat_template_sha256':a.chat_template_sha256,
                                    'yes_token_id':a.expected_yes_token_id,
                                    'no_token_id':a.expected_no_token_id,
                                    'answer_policy':'ASSISTANT_FINAL_ONE_TARGET_TOKEN_ONLY',
                                    'no_input_truncation':True},
              'training_budget':{'examples':a.source_train_count,
                                 'validation_examples':a.source_validation_count,
                                 'micro_batch':a.per_device_batch_size,
                                 'gradient_accumulation':a.gradient_accumulation_steps,
                                 'optimizer_steps':a.max_steps,'epochs':a.expected_epochs,
                                 'world_size':1,'model_shards_gpu_count':a.expected_gpus,
                                 'learning_rate':float(a.learning_rate),
                                 'seed':a.seed},
              'files':outputs,'token_stats':stats,
              'limitations':['Read-only source selection with new Gemma-specific token audit; no model weights loaded',
                             'F-01c leakage audit inherited, not independently recomputed',
                             'No train/validation function-ID disjointness established',
                             'Single-process multi-GPU-sharded LoRA training still requires 2048-token stress/stability',
                             'No held-out data accessed; no model training or optimizer step']}
    write_new(pdir/'manifest.json',manifest)
    result={'status':manifest['status'],'prepared_path':str(pdir),
            'prepared_manifest_sha256':sha256(pdir/'manifest.json'),
            'config_fingerprint':fp,'heldout_accessed':False}
    write_new(a.run_dir/'prepare_result.json',result)
    return result


def verify(a):
    data=frozen(a)
    pdir=prepared_folder(a)
    need(pdir.is_dir(), 'PREPARED_DIR_MISSING')
    manifest=load_json(pdir/'manifest.json')
    need(manifest['status']=='I01B_V1_SOURCE_PREPARE_PASS_NOT_TRAINED',
         'PREPARED_MANIFEST_STATUS_NOT_PASS')
    need(manifest['version']==a.version and manifest['experiment']==a.experiment,
         'PREPARED_MANIFEST_IDENTITY_DRIFT')
    original=manifest['effective_parameters']
    for key,value in parameters(a).items():
        if key not in ('run_dir','log_file','log_dir','mode','verify_source'):
            need(original.get(key)==value, f'PREPARED_EFFECTIVE_PARAMETER_DRIFT:{key}')
    need(manifest['model']['revision']==a.model_revision and
         manifest['model']['base_precision']==a.base_precision and
         manifest['model']['quantization']==a.quantization and
         manifest['i01a_provenance_sha256']==a.i01a_preflight_sha256,
         'PREPARED_MODEL_PROOF_DRIFT')
    need(manifest['training_budget']['optimizer_steps']==a.max_steps and
         manifest['training_budget']['examples']==a.source_train_count,
         'PREPARED_TRAIN_BUDGET_DRIFT')
    frozen_report=manifest['frozen_source']
    need(frozen_report['pair_identity_equals_g01c'] is True and
         frozen_report['training_order_equals_g01c'] is True,
         'PREPARED_SOURCE_IDENTITY_NOT_CONFIRMED')
    tok,template=load_tokenizer(a)
    result={}
    for split in ('train','validation'):
        f=pdir/(split+'.jsonl')
        observed=read_file_pin(f,manifest['files'][split]['sha256'],f'PREPARED_{split.upper()}')
        prepared_rows=list(read_jsonl(f))
        need(len(prepared_rows)==len(data['selections'][split]),
             f'PREPARED_RECORD_COUNT_DRIFT:{split}')
        total=0
        for row,ref in zip(prepared_rows, data['selections'][split]):
            fresh=audit_pair(a,tok,template,ref,data['codes'])
            need(row==fresh, f'PREPARED_PER_PAIR_TOKEN_AUDIT_DRIFT:{split}:{total}')
            total+=1
            if total%a.audit_progress_every==0 or total==len(data['selections'][split]):
                print(json.dumps({'event':'I01B_INDEPENDENT_VERIFY_PROGRESS',
                                  'split':split,'processed':total,'chicago':chicago()}),flush=True)
        need(total==len(data['selections'][split])==manifest['files'][split]['rows'],
             f'PREPARED_RECORD_COUNT_DRIFT:{split}')
        result[split]=observed
    order_path=pdir/'order.jsonl'
    result['order']=read_file_pin(order_path,manifest['files']['order']['sha256'],
                                  'PREPARED_ORDER')
    order=list(read_jsonl(order_path)); train=list(read_jsonl(pdir/'train.jsonl'))
    need(len(order)==len(data['order'])==len(train), 'PREPARED_ORDER_LENGTH_DRIFT')
    for position,(entry,index) in enumerate(zip(order,data['order'])):
        need(entry.get('index')==index and
             entry.get('row_sha256')==text_sha(canonical_json(train[index])),
             f'PREPARED_ORDER_ROW_HASH_DRIFT:{position}')
    result['manifest']={'path':str(pdir/'manifest.json'),
                        'sha256':sha256(pdir/'manifest.json')}
    verify_result={'status':'I01B_V1_INDEPENDENT_VERIFY_PASS_NOT_TRAINED',
                   'at_chicago':chicago(),'prepared_dir':str(pdir),
                   'verified_files':result,'source_pair_identity_g01c':True,
                   'no_truncation':True,'heldout_accessed':False,
                   'full_token_audit_performed':True,
                   'training_started':False}
    write_new(a.run_dir/'verification_report.json',verify_result)
    return verify_result


def finalize(a):
    required=['provenance','frozen','tokenizer','verify']
    if a.mode=='--prepare':
        required.insert(-1,'prepare')
    statuses={}
    for name in required:
        f=a.run_dir/(name+'.json')
        statuses[name]=load_json(f)['status'] if f.is_file() else 'NOT_RUN'
    achieved=all(v=='PASS' for v in statuses.values())
    need((a.run_status=='PASS') == achieved,
         'FINAL_STATUS_INCONSISTENT_WITH_STAGES')
    status=('I01B_V1_PREPARE_VERIFIED_PASS_NOT_TRAINED' if a.mode=='--prepare' else
            'I01B_V1_INDEPENDENT_VERIFY_PASS_NOT_TRAINED') if achieved else 'FAIL'
    out={'status':status,'experiment':a.experiment,'version':a.version,
         'mode':a.mode,'at_chicago':chicago(),'exit_code':a.exit_code,
         'effective_parameters':parameters(a),'stages':statuses,
         'prepared_dir':str(prepared_folder(a)),
         'model_weights_loaded':False,'training_started':False,
         'optimizer_steps_executed':0,'heldout_accessed':False,
         'next_action':'Review prepared/manifest.json and verification_report.json before i-01c 2048-token stress/stability',
         'log_file':str(a.log_file)}
    write_new(a.run_dir/'run_manifest.json',out)
    return out


STAGES={'provenance':provenance,'frozen':frozen,'tokenizer':tokenizer,
        'prepare':prepare,'verify':verify,'finalize':finalize}


def main(argv=None):
    a=parse_args(argv)
    if a.stage!='finalize':
        validate_config(a)
    start=datetime.now(ZoneInfo('America/Chicago'))
    try:
        details=STAGES[a.stage](a)
        elapsed=(datetime.now(ZoneInfo('America/Chicago'))-start).total_seconds()
        if a.stage!='finalize':
            output={'status':'PASS','stage':a.stage,'at_chicago':chicago(),
                    'duration_seconds':round(elapsed,3),
                    'details':details if a.stage in ('provenance','frozen','tokenizer') else
                    {'status':details['status'],'prepared':str(prepared_folder(a))}}
            # Avoid duplicating huge source code objects in frozen.json stage reports.
            if a.stage=='frozen':
                output['details']=details['evidence']
            write_new(a.run_dir/(a.stage+'.json'),output)
            print(json.dumps({'status':'I01B_V1_STAGE_PASS','stage':a.stage,
                              'at_chicago':chicago(),'report':str(a.run_dir/(a.stage+'.json'))}),flush=True)
        else:
            print(json.dumps({'final_status':details['status'],
                              'run_manifest':str(a.run_dir/'run_manifest.json')}),flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({'status':'I01B_V1_STAGE_FAIL','stage':a.stage,
                          'error':type(exc).__name__+':'+str(exc),
                          'at_chicago':chicago()}),flush=True)
        traceback.print_exc()
        return 2


if __name__=='__main__':
    sys.exit(main())
