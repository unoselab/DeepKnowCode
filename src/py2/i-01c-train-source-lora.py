#!/usr/bin/env python3
"""I-01c v2: Gemma 4 BF16 source-only LoRA with immutable I-01b input and staged proof.

No experiment-specific value is defaulted here. Shell passes ALL configuration by CLI.
Preflight is read-only on source/model; modes create only unique run directories.
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
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

CHICAGO_ZONE = ZoneInfo('America/Chicago')  # Timestamp policy, not an experiment parameter.
class StageError(RuntimeError):
    pass


def need(condition, error):
    if not condition:
        raise StageError(error)


def now():
    return datetime.now(CHICAGO_ZONE).isoformat()


def sha_file(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')


def new_text(path, text):
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(text)


def load_json(path):
    with Path(path).open(encoding='utf-8') as f:
        data = json.load(f)
    need(isinstance(data, dict), f'JSON_NOT_OBJECT:{path}')
    return data


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for n, line in enumerate(stream, 1):
            need(bool(line.strip()), f'BLANK_RECORD:{path}:{n}')
            data = json.loads(line)
            need(isinstance(data, dict), f'NON_OBJECT_RECORD:{path}:{n}')
            yield data


def event(event_name, **data):
    """Emit a named event while preserving optional payload fields like kind."""
    print(json.dumps({'event': event_name, 'chicago': now(), **data}, sort_keys=True), flush=True)


def boolean(text):
    need(text in ('true', 'false'), f'BOOLEAN_EXPECTED_TRUE_OR_FALSE:{text}')
    return text == 'true'


# All items are REQUIRED CLI params. Numeric values and strings are deliberately not
# embedded as Python defaults; they are specified in the shell configuration.
STR_FIELDS = '''version experiment mode root run_dir log_file stress_run stability_run resume_checkpoint
conda_env model_id model_mode model_revision hf_home snapshot expected_config_sha256
expected_tokenizer_config_sha256 chat_template_sha256 tokenizer_policy tokenizer_use_fast
tokenizer_local_only trust_remote_code enable_thinking base_precision finetuning_method
quantization model_auto_class attention_backend low_cpu_mem_usage device_ids device_map
lora_task_type lora_target_regex lora_dropout lora_bias gradient_checkpointing
gradient_checkpoint_use_reentrant prepared_run prepared_manifest_sha256 prepared_train_sha256
prepared_validation_sha256 prepared_order_sha256 expected_prepared_fingerprint
prepared_verify_report_status prepared_verification_sha256 prepared_provenance_sha256
i01a_run i01a_preflight_sha256 i01a_tokenizer_sha256 i01a_lora_sha256
i01a_weights_sha256 expected_engine_sha256 source_codes frozen_train_sha256 frozen_validation_sha256
frozen_manifest_sha256 answer_yes answer_no system_text user_prefix user_separator
user_suffix optimizer learning_rate weight_decay max_grad_norm lr_scheduler
stability_balanced_labels save_optimizer_state log_dir log_prefix log_timestamp_format
log_tee log_stderr expected_torch expected_torch_cuda expected_transformers
expected_peft expected_accelerate'''.split()
INT_FIELDS = '''expected_gpus min_compute_major max_gpu_placement_gib max_cpu_placement_gib
min_gpu_headroom_gib max_gpu_peak_reserved_gib expected_lora_target_modules
expected_trainable_parameters lora_rank lora_alpha seed source_train_count
source_validation_count source_train_positives source_train_negatives source_validation_positives
source_validation_negatives max_seq_length expected_yes_token_id expected_no_token_id
per_device_batch_size gradient_accumulation_steps max_steps expected_epochs warmup_steps
stress_steps stress_min_tokens stability_steps log_steps progress_microbatch_every
eval_steps eval_batch_size save_steps'''.split()
DYNAMIC_FIELDS = {'mode', 'run_dir', 'log_file', 'stress_run', 'stability_run', 'resume_checkpoint'}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in STR_FIELDS:
        p.add_argument('--' + key.replace('_', '-'), required=True)
    for key in INT_FIELDS:
        p.add_argument('--' + key.replace('_', '-'), required=True, type=int)
    p.add_argument('--stage', required=True, choices=('preflight', 'stress', 'stability', 'train', 'resume', 'finalize'))
    p.add_argument('--run-status', choices=('PASS', 'FAIL'))
    p.add_argument('--exit-code', type=int)
    a = p.parse_args(argv)
    for key in ('root', 'run_dir', 'log_file', 'hf_home', 'snapshot', 'prepared_run',
                'i01a_run',
                'source_codes', 'log_dir', 'stress_run', 'stability_run', 'resume_checkpoint'):
        value = getattr(a, key)
        setattr(a, key, None if value == 'NONE' else Path(value))
    for key in ('tokenizer_use_fast', 'tokenizer_local_only', 'trust_remote_code',
                'enable_thinking', 'low_cpu_mem_usage', 'gradient_checkpointing',
                'gradient_checkpoint_use_reentrant', 'stability_balanced_labels',
                'save_optimizer_state', 'log_tee', 'log_stderr'):
        setattr(a, key, boolean(getattr(a, key)))
    for key in ('learning_rate', 'weight_decay', 'max_grad_norm', 'lora_dropout'):
        setattr(a, key, float(getattr(a, key)))
    validate_config(a)
    return a


def config(a):
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(a).items()
            if key not in ('stage', 'run_status', 'exit_code')}


def stable_config(a):
    return {key: value for key, value in config(a).items() if key not in DYNAMIC_FIELDS}


def fingerprint(a):
    return sha_text(canonical(stable_config(a)))


def validate_config(a):
    need(a.mode in ('--stress', '--stability', '--train', '--resume'), 'MODE_INVALID')
    need(a.version == 'v2' and a.experiment == 'i-01c', 'EXPERIMENT_OR_VERSION_DRIFT')
    need(a.base_precision == 'bf16' and a.finetuning_method == 'lora' and
         a.quantization == 'none', 'QLORA_OR_UNAPPROVED_PRECISION_FORBIDDEN')
    need(a.model_auto_class == 'AutoModelForMultimodalLM', 'GEMMA_MODEL_CLASS_DRIFT')
    need(a.lora_task_type == 'CAUSAL_LM' and a.lora_bias == 'none', 'LORA_TASK_DRIFT')
    need(a.tokenizer_policy == 'pinned-jinja-autotokenizer-text-only' and
         a.tokenizer_local_only and not a.trust_remote_code and not a.enable_thinking,
         'TOKENIZER_POLICY_DRIFT')
    need(a.per_device_batch_size > 0 and a.expected_epochs > 0 and
         a.source_train_count == a.max_steps * a.gradient_accumulation_steps * a.per_device_batch_size,
         'SOURCE_BUDGET_INCONSISTENT')
    need(a.source_validation_count > 0 and
         a.source_train_positives + a.source_train_negatives == a.source_train_count and
         a.source_validation_positives + a.source_validation_negatives == a.source_validation_count,
         'DATASET_LABEL_COUNTS_INCONSISTENT')
    need(a.expected_gpus == len(a.device_ids.split(',')) and
         all(x.isdigit() for x in a.device_ids.split(',')), 'GPU_IDS_INVALID')
    need(a.max_gpu_placement_gib > 0 and a.max_cpu_placement_gib > 0 and
         a.min_gpu_headroom_gib > 0 and a.max_gpu_peak_reserved_gib > 0,
         'GPU_MEMORY_CONFIGURATION_INVALID')
    need(a.lora_rank > 0 and a.lora_alpha > 0 and 0 <= a.lora_dropout < 1 and
         a.expected_lora_target_modules > 0 and a.expected_trainable_parameters > 0,
         'LORA_CONFIGURATION_INVALID')
    need(a.optimizer == 'adamw_torch' and a.lr_scheduler == 'cosine' and
         0 < a.learning_rate < 1 and a.weight_decay >= 0 and a.max_grad_norm > 0,
         'OPTIMIZER_CONFIGURATION_INVALID')
    need(0 < a.warmup_steps < a.max_steps and a.stress_steps > 0 and
         a.stability_steps > 0 and a.stress_min_tokens <= a.max_seq_length,
         'STAGE_BUDGET_INVALID')
    need(all(x > 0 for x in (a.log_steps, a.progress_microbatch_every,
                             a.eval_steps, a.eval_batch_size, a.save_steps)),
         'LOG_EVAL_SAVE_INTERVAL_INVALID')
    need(a.max_steps % a.eval_steps == 0 and a.max_steps % a.save_steps == 0 and
         a.max_steps % a.log_steps == 0 and a.save_optimizer_state,
         'FINAL_EVAL_SAVE_OR_RESUME_NOT_GUARANTEED')
    need(a.expected_yes_token_id != a.expected_no_token_id and
         a.answer_yes != a.answer_no and len(a.answer_yes.strip()) > 0 and
         len(a.answer_no.strip()) > 0, 'ANSWER_POLICY_INVALID')
    for key in ('expected_config_sha256', 'expected_tokenizer_config_sha256',
                'chat_template_sha256', 'prepared_manifest_sha256', 'prepared_train_sha256',
                'prepared_validation_sha256', 'prepared_order_sha256',
                'expected_prepared_fingerprint', 'prepared_verification_sha256',
                'prepared_provenance_sha256', 'i01a_preflight_sha256',
                'i01a_tokenizer_sha256', 'i01a_lora_sha256', 'i01a_weights_sha256',
                'expected_engine_sha256', 'frozen_train_sha256',
                'frozen_validation_sha256', 'frozen_manifest_sha256'):
        need(re.fullmatch('[0-9a-f]{64}', getattr(a, key)) is not None,
             f'INVALID_SHA256_PIN:{key}')
    need(a.root.is_dir() and a.run_dir.is_dir() and a.log_dir.is_dir(),
         'REQUIRED_RUNTIME_DIRECTORY_MISSING')
    need(a.prepared_run != a.run_dir, 'INPUT_RUN_WOULD_BE_OVERWRITTEN')
    need(a.mode == '--stress' or a.stress_run is not None, 'STRESS_PROOF_REQUIRED')
    need(a.mode not in ('--train', '--resume') or a.stability_run is not None,
         'STABILITY_PROOF_REQUIRED')
    need(a.mode != '--resume' or a.resume_checkpoint is not None,
         'RESUME_CHECKPOINT_REQUIRED')
    if a.mode == '--stress':
        need(a.stress_run is None and a.stability_run is None and
             a.resume_checkpoint is None, 'STRESS_UNEXPECTED_PROOFS')
    if a.mode == '--stability':
        need(a.stability_run is None and a.resume_checkpoint is None,
             'STABILITY_UNEXPECTED_PROOFS')
    if a.mode == '--train':
        need(a.resume_checkpoint is None, 'TRAIN_MUST_START_FRESH_BASE')


def file_pin(path, digest, label):
    need(path.is_file() and not path.is_symlink(), f'PINNED_FILE_MISSING_OR_LINK:{label}:{path}')
    found = sha_file(path)
    need(found == digest, f'SHA256_MISMATCH:{label}:expected={digest}:actual={found}')
    return {'path': str(path), 'sha256': found, 'bytes': path.stat().st_size}


def snapshot_pin(path, digest, label):
    need(path.is_file(), f'MODEL_SNAPSHOT_FILE_MISSING:{label}:{path}')
    found = sha_file(path)
    need(found == digest, f'SNAPSHOT_SHA256_DRIFT:{label}:{found}')
    return {'sha256': found, 'bytes': path.stat().st_size}


def verify_prepared(a):
    p = a.prepared_run / 'prepared'
    checks = {
        'manifest': file_pin(p / 'manifest.json', a.prepared_manifest_sha256, 'MANIFEST'),
        'train': file_pin(p / 'train.jsonl', a.prepared_train_sha256, 'TRAIN'),
        'validation': file_pin(p / 'validation.jsonl', a.prepared_validation_sha256, 'VALIDATION'),
        'order': file_pin(p / 'order.jsonl', a.prepared_order_sha256, 'ORDER')}
    manifest = load_json(p / 'manifest.json')
    need(manifest.get('status') == 'I01B_V1_SOURCE_PREPARE_PASS_NOT_TRAINED' and
         manifest.get('config_fingerprint') == a.expected_prepared_fingerprint,
         'I01B_VERIFIED_MANIFEST_STATUS_OR_FINGERPRINT_DRIFT')
    need(manifest['model'] == {'model_id': a.model_id, 'revision': a.model_revision,
                              'base_precision':a.base_precision,
                              'fine_tuning_method':a.finetuning_method,
                              'quantization':a.quantization}, 'PREPARED_MODEL_IDENTITY_DRIFT')
    src = manifest['effective_parameters']
    same = {'seed':a.seed,'answer_yes':a.answer_yes,'answer_no':a.answer_no,
            'system_text':a.system_text,'user_prefix':a.user_prefix,
            'user_separator':a.user_separator,'user_suffix':a.user_suffix,
            'max_seq_length':a.max_seq_length, 'expected_yes_token_id':a.expected_yes_token_id,
            'expected_no_token_id':a.expected_no_token_id,'chat_template_sha256':a.chat_template_sha256,
            'source_train_count':a.source_train_count,
            'source_validation_count':a.source_validation_count,
            'gradient_accumulation_steps':a.gradient_accumulation_steps,
            'max_steps':a.max_steps,'per_device_batch_size':a.per_device_batch_size,
            'expected_epochs':a.expected_epochs,'lora_rank':a.lora_rank,
            'lora_alpha':a.lora_alpha,'lora_dropout':str(a.lora_dropout),
            'lora_bias':a.lora_bias,'lora_target_regex':a.lora_target_regex,
            'expected_lora_target_modules':a.expected_lora_target_modules,
            'expected_trainable_parameters':a.expected_trainable_parameters,
            'learning_rate':str(a.learning_rate),'device_map':a.device_map,
            'device_ids':a.device_ids,'expected_gpus':a.expected_gpus}
    for key, value in same.items():
        need(src.get(key) == value, f'I01B_I01C_PARAMETER_DRIFT:{key}:{src.get(key)!r}!={value!r}')
    for key, value in (('train', a.prepared_train_sha256),
                       ('validation', a.prepared_validation_sha256),
                       ('order', a.prepared_order_sha256)):
        need(manifest['files'][key]['sha256'] == value and
             manifest['files'][key]['rows'] == (a.source_validation_count if key=='validation' else a.source_train_count),
             f'PREPARED_FILE_MANIFEST_DRIFT:{key}')
    need(manifest['training_budget']['examples'] == a.source_train_count and
         manifest['training_budget']['optimizer_steps'] == a.max_steps and
         manifest['training_budget']['gradient_accumulation'] == a.gradient_accumulation_steps and
         manifest['training_budget']['epochs'] == a.expected_epochs,
         'PREPARED_TRAINING_BUDGET_DRIFT')
    froz = manifest['frozen_source']['frozen_files']
    for key, expected in [('f01c_train', a.frozen_train_sha256),
                          ('f01c_validation',a.frozen_validation_sha256),
                          ('f01c_manifest',a.frozen_manifest_sha256)]:
        need(froz[key]['sha256'] == expected, f'FROZEN_PARENT_SHA_DRIFT:{key}')
    file_pin(a.prepared_run/'verification_report.json', a.prepared_verification_sha256,
             'I01B_INDEPENDENT_VERIFICATION_REPORT')
    file_pin(a.prepared_run/'provenance.json', a.prepared_provenance_sha256,
             'I01B_PROVENANCE_REPORT')
    provenance=load_json(a.prepared_run/'provenance.json')['details']
    need(provenance['proof_status']=='LORA_BACKWARD_PROBE_PASS_NOT_TRAINING' and
         provenance['model_identity']['model_id']==a.model_id and
         provenance['model_identity']['revision']==a.model_revision,
         'I01A_PROVENANCE_OR_PROBE_NOT_VERIFIED')
    need(a.i01a_run.is_dir(),'I01A_PROBE_RUN_MISSING')
    for name,digest in [('preflight',a.i01a_preflight_sha256),
                        ('tokenizer',a.i01a_tokenizer_sha256),
                        ('lora_probe',a.i01a_lora_sha256),
                        ('weights',a.i01a_weights_sha256)]:
        file_pin(a.i01a_run/(name+'.json'),digest,'I01A_'+name)
        need(provenance['provenance_files'][name]['sha256']==digest,
             f'I01B_I01A_EVIDENCE_SHA_DRIFT:{name}')
    need(load_json(a.i01a_run/'lora_probe.json')['details']['optimizer_steps']==0,
         'I01A_PROBE_MUST_NOT_TRAIN')
    verification = load_json(a.prepared_run / 'verification_report.json')
    need(verification.get('status') == a.prepared_verify_report_status and
         verification.get('full_token_audit_performed') is True and
         verification.get('source_pair_identity_g01c') is True and
         verification.get('no_truncation') is True and
         verification.get('heldout_accessed') is False,
         'I01B_PREPARED_INDEPENDENT_VERIFICATION_NOT_PASS')
    for key, digest in [('manifest',a.prepared_manifest_sha256),
                        ('train',a.prepared_train_sha256),
                        ('validation',a.prepared_validation_sha256),
                        ('order',a.prepared_order_sha256)]:
        need(verification['verified_files'][key]['sha256'] == digest,
             f'I01B_INDEPENDENT_VERIFIED_SHA_DRIFT:{key}')
    source_pin = manifest['frozen_source']['frozen_files']['source_codes']['sha256']
    checks['source_codes'] = file_pin(a.source_codes, source_pin, 'JAVA_SOURCE_CODES')
    need(sha_file(a.prepared_run / 'prepare_result.json') and
         load_json(a.prepared_run / 'prepare_result.json')['prepared_manifest_sha256'] == a.prepared_manifest_sha256,
         'I01B_PREPARE_RESULT_DRIFT')
    return {'pinned_files':checks,'i01b_fingerprint':a.expected_prepared_fingerprint,
            'i01b_verification_status':a.prepared_verify_report_status,
            'heldout_accessed':False}


def verify_hardware(a):
    import torch
    import importlib.metadata as md
    need(os.environ.get('CONDA_DEFAULT_ENV') == a.conda_env,
         'WRONG_CONDA_ENVIRONMENT')
    expected = {'torch':a.expected_torch,'transformers':a.expected_transformers,
                'peft':a.expected_peft,'accelerate':a.expected_accelerate}
    found = {name:md.version(name).split('+')[0] for name in expected}
    for name, pin in expected.items():
        need(found[name] == pin, f'ENVIRONMENT_VERSION_DRIFT:{name}:{found[name]}!={pin}')
    need(torch.version.cuda == a.expected_torch_cuda,
         'CUDA_RUNTIME_VERSION_DRIFT')
    need(torch.cuda.is_available() and torch.cuda.device_count() == a.expected_gpus,
         f'GPU_COUNT_OR_CUDA_MISMATCH:{torch.cuda.device_count()}')
    need(torch.cuda.is_bf16_supported(), 'BF16_NOT_SUPPORTED')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == a.device_ids,
         'VISIBLE_CUDA_IDS_DRIFT')
    gpus=[]
    for index in range(a.expected_gpus):
        cap=torch.cuda.get_device_capability(index)
        total=torch.cuda.get_device_properties(index).total_memory / 2**30
        free=torch.cuda.mem_get_info(index)[0] / 2**30
        need(cap[0] >= a.min_compute_major and free > a.min_gpu_headroom_gib,
             f'GPU_CAPABILITY_OR_FREE_VRAM_TOO_LOW:{index}')
        need(total > a.max_gpu_placement_gib and total > a.max_gpu_peak_reserved_gib,
             f'GPU_VRAM_SIZE_TOO_LOW:{index}')
        gpus.append({'index':index,'name':torch.cuda.get_device_name(index),
                     'compute_capability':f'{cap[0]}.{cap[1]}',
                     'total_gib':round(total,3),'free_gib':round(free,3)})
    return {'package_versions':found,'torch_cuda':torch.version.cuda,'gpus':gpus}


def preflight(a):
    file_pin(Path(__file__),a.expected_engine_sha256,'I01C_ENGINE')
    need(a.prepared_run.exists(), 'PREPARED_RUN_MISSING')
    prepared=verify_prepared(a)
    snapshot = {name:snapshot_pin(a.snapshot/name,digest,name) for name,digest in (
        ('config.json',a.expected_config_sha256),
        ('tokenizer_config.json',a.expected_tokenizer_config_sha256),
        ('chat_template.jinja',a.chat_template_sha256))}
    hardware=verify_hardware(a)
    need((a.snapshot/'model.safetensors.index.json').is_file(),
         'MODEL_SAFETENSORS_INDEX_MISSING')
    token=load_json(a.prepared_run / 'tokenizer.json')['details']
    need(token['label_token_ids']=={'YES':a.expected_yes_token_id,'NO':a.expected_no_token_id}
         and token['pinned_chat_template_sha256']==a.chat_template_sha256,
         'PREPARED_TOKENIZER_IDENTITY_DRIFT')
    if a.mode in ('--stability', '--train', '--resume'):
        verify_stage_run(a, a.stress_run, 'stress', 'I01C_V2_STRESS_PASS_NOT_M1_TRAINING')
    if a.mode in ('--train', '--resume'):
        verify_stage_run(a, a.stability_run, 'stability', 'I01C_V2_STABILITY_PASS_NOT_M1_TRAINING')
    if a.mode == '--resume':
        verify_checkpoint(a, a.resume_checkpoint, require_training=True)
    evidence={'version':a.version,'status':'I01C_V2_PREFLIGHT_PASS_NOT_TRAINED',
              'at_chicago':now(),'config_fingerprint':fingerprint(a),
              'effective_parameters':config(a),'source':prepared,
              'snapshot':snapshot,'environment':hardware,
              'stage_prerequisites':{'stress_run':str(a.stress_run) if a.stress_run else None,
                                     'stability_run':str(a.stability_run) if a.stability_run else None},
              'training_started':False,'heldout_accessed':False}
    new_json(a.run_dir/'preflight.json',evidence)
    event('I01C_PREFLIGHT_PASS',prepared_manifest_sha256=a.prepared_manifest_sha256,
          fingerprint=fingerprint(a),mode=a.mode)


def load_tokenizer(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(a.snapshot), use_fast=a.tokenizer_use_fast,
           trust_remote_code=a.trust_remote_code,local_files_only=a.tokenizer_local_only)
    template=(a.snapshot/'chat_template.jinja').read_text(encoding='utf-8')
    need(sha_text(template)==a.chat_template_sha256, 'MODEL_CHAT_TEMPLATE_CHANGED')
    return tok,template


def prompt_for(a, tok, template, code_a, code_b):
    messages=[{'role':'system','content':a.system_text},
              {'role':'user','content':a.user_prefix+code_a+a.user_separator+code_b+a.user_suffix}]
    prompt=tok.apply_chat_template(messages,chat_template=template,tokenize=False,
                                   add_generation_prompt=True,enable_thinking=a.enable_thinking)
    need(isinstance(prompt,str) and bool(prompt), 'RENDERED_PROMPT_EMPTY')
    return prompt


def contextual_tokens(a,tok,prompt):
    prefix=tok.encode(prompt,add_special_tokens=False)
    need(bool(prefix), 'PROMPT_TOKENS_EMPTY')
    labels={}
    for answer,expected in [(a.answer_yes,a.expected_yes_token_id),
                            (a.answer_no,a.expected_no_token_id)]:
        appended=tok.encode(prompt+answer,add_special_tokens=False)
        need(len(appended)==len(prefix)+1 and appended[:-1]==prefix and appended[-1]==expected,
             f'UNSTABLE_SINGLE_TOKEN_YES_NO:{answer!r}')
        labels[answer]=appended[-1]
    need(len(prefix)+1<=a.max_seq_length,'PROMPT_TOO_LONG')
    return prefix,labels


def source_codes(a, needed):
    codes={}
    for record in read_rows(a.source_codes):
        ident=str(record['idx'])
        if ident in needed:
            need(ident not in codes and isinstance(record['func'],str) and record['func'].strip(),
                 f'MISSING_OR_DUPLICATED_SOURCE_FUNCTION:{ident}')
            codes[ident]=record['func']
    need(set(codes)==needed,'SOURCE_FUNCTION_IDS_NOT_FOUND')
    return codes


def rows_and_order(a):
    prepared=a.prepared_run/'prepared'
    rows={part:list(read_rows(prepared/(part+'.jsonl'))) for part in ('train','validation')}
    need(len(rows['train'])==a.source_train_count and
         len(rows['validation'])==a.source_validation_count,'PREPARED_ROW_COUNT_DRIFT')
    for part in ('train','validation'):
        observed=collections.Counter()
        keys=set()
        for record in rows[part]:
            need(type(record.get('label')) is int and record['label'] in (0,1),
                 'ROW_LABEL_CORRUPT')
            observed[record['label']]+=1
            ident=(record['id1'],record['id2'])
            need(ident not in keys,'DUPLICATE_SOURCE_PAIR')
            keys.add(ident)
            need(record['answer_id']==(a.expected_yes_token_id if record['label'] else a.expected_no_token_id),
                 'PREPARED_ANSWER_ID_DRIFT')
            need(1<record['n_tokens']<=a.max_seq_length, 'PREPARED_TOKEN_LENGTH_INVALID')
        need(observed[1]==(a.source_train_positives if part=='train' else a.source_validation_positives)
             and observed[0]==(a.source_train_negatives if part=='train' else a.source_validation_negatives),
             f'PREPARED_LABEL_COUNTS_DRIFT:{part}')
    order_rows=list(read_rows(prepared/'order.jsonl'))
    order=[row['index'] for row in order_rows]
    need(len(order)==a.source_train_count and
         sorted(order)==list(range(a.source_train_count)), 'ORDER_NOT_FULL_PERMUTATION')
    seed_order=list(range(a.source_train_count)); random.Random(a.seed).shuffle(seed_order)
    need(order==seed_order,'TRAINING_ORDER_SEED_CHANGED')
    for item,idx in zip(order_rows,order):
        need(item['row_sha256']==sha_text(canonical(rows['train'][idx])),
             'ORDER_ROW_SHA256_DRIFT')
    return rows,order


def selected_stress(a,rows):
    indices=sorted(range(len(rows)),key=lambda i:(-rows[i]['n_tokens'],i))
    selected=[]
    for idx in indices:
        if rows[idx]['n_tokens']>=a.stress_min_tokens:
            selected.append(idx)
            if len(selected)==a.stress_steps:break
    need(len(selected)==a.stress_steps,'NO_ENOUGH_LONG_STRESS_SAMPLES')
    return selected


def selected_stability(a,rows):
    selected={}
    for label in (0,1):
        pool=sorted((i for i,row in enumerate(rows) if row['label']==label),
                    key=lambda i: (-rows[i]['n_tokens'],i))
        selected[label]=pool
    if not a.stability_balanced_labels:
        all_indices=sorted(range(len(rows)),key=lambda i:(-rows[i]['n_tokens'],i))
        need(len(all_indices)>=a.stability_steps,'STABILITY_SAMPLES_UNAVAILABLE')
        return all_indices[:a.stability_steps]
    out=[]; ptr={0:0,1:0}
    for ix in range(a.stability_steps):
        label=ix%2
        need(ptr[label]<len(selected[label]),'STABILITY_SAMPLES_UNAVAILABLE')
        out.append(selected[label][ptr[label]])
        ptr[label]+=1
    need(len(set(out))==a.stability_steps,'STABILITY_SELECTION_NOT_UNIQUE')
    return out


def tokenize_samples(a,tok,template,rows,codes,indices,part):
    result={}
    for n,idx in enumerate(indices,1):
        rec=rows[idx]
        code_a=codes[rec['id1']]; code_b=codes[rec['id2']]
        need(sha_text(code_a)==rec['code1_sha256'] and
             sha_text(code_b)==rec['code2_sha256'], 'CODE_SHA_DRIFT')
        prompt=prompt_for(a,tok,template,code_a,code_b)
        need(sha_text(prompt)==rec['prompt_sha256'],
             f'GEMMA_PROMPT_DRIFT:{part}:{idx}')
        prefix,labels=contextual_tokens(a,tok,prompt)
        token_id=labels[a.answer_yes if rec['label'] else a.answer_no]
        ids=prefix+[token_id]
        need(len(ids)==rec['n_tokens'] and token_id==rec['answer_id'],
             f'PREPARED_TOKEN_AUDIT_DRIFT:{part}:{idx}')
        result[idx]=ids
        if n%1000==0 or n==len(indices):
            event('TOKEN_AUDIT',split=part,count=n,total=len(indices))
    return result


def get_training_inputs(a, phase):
    # Mandatory second-stage pin verification before expensive GPU/model work.
    verify_prepared(a)
    rows,order=rows_and_order(a)
    if phase=='stress':
        indices={'train':selected_stress(a,rows['train']),'validation':[]}
    elif phase=='stability':
        indices={'train':selected_stability(a,rows['train']),'validation':[]}
    else:
        indices={'train':list(range(len(rows['train']))),
                 'validation':list(range(len(rows['validation'])))}
    needed={r[k] for part in ('train','validation') for i in indices[part]
            for r in [rows[part][i]] for k in ('id1','id2')}
    codes=source_codes(a,needed)
    tok,template=load_tokenizer(a)
    tokens={part:tokenize_samples(a,tok,template,rows[part],codes,indices[part],part)
            for part in ('train','validation')}
    report={'version':a.version,'status':'I01C_V2_TOKEN_AUDIT_PASS',
            'at_chicago':now(),'prepared_manifest_sha256':a.prepared_manifest_sha256,
            'config_fingerprint':fingerprint(a),'phase':phase,
            'tokenized_sample_counts':{k:len(v) for k,v in tokens.items()},
            'token_length_min_max':{k:[min(map(len,v.values())),max(map(len,v.values()))]
                if v else [] for k,v in tokens.items()},
            'no_truncation':True,'heldout_accessed':False,
            'full_token_reaudit': phase not in ('stress','stability')}
    new_json(a.run_dir/'token_audit.json',report)
    return rows,order,tokens


def attach_model(a, adapter=None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForMultimodalLM
    need(a.model_auto_class==AutoModelForMultimodalLM.__name__,
         'MODEL_AUTO_CLASS_UNEXPECTED')
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.cuda.empty_cache()
    event('BASE_MODEL_LOADING_START', revision=a.model_revision,precision=a.base_precision,
          device_map=a.device_map,quantization=a.quantization)
    memory_limits={i:f'{a.max_gpu_placement_gib}GiB' for i in range(a.expected_gpus)}
    memory_limits['cpu']=f'{a.max_cpu_placement_gib}GiB'
    model=AutoModelForMultimodalLM.from_pretrained(
         str(a.snapshot),dtype=torch.bfloat16,device_map=a.device_map,
         max_memory=memory_limits,attn_implementation=a.attention_backend,
         local_files_only=True,trust_remote_code=False,
         low_cpu_mem_usage=a.low_cpu_mem_usage)
    need(not getattr(model,'is_loaded_in_4bit',False) and
         not getattr(model,'is_loaded_in_8bit',False) and
         getattr(model,'hf_quantizer',None) is None, 'QUANTIZATION_NOT_ALLOWED')
    device_map=getattr(model,'hf_device_map',None)
    need(isinstance(device_map,dict) and device_map and
         all(str(place).lower() not in ('cpu','disk','meta') for place in device_map.values()),
         'MODEL_CPU_OFFLOAD_OR_MISSING_DEVICE_MAP')
    devices=set(str(x).replace('cuda:','') for x in device_map.values())
    need(all(x.isdigit() and int(x)<a.expected_gpus for x in devices),
         f'UNEXPECTED_DEVICE_PLACEMENT:{sorted(devices)}')
    target=re.compile(a.lora_target_regex)
    targets=[n for n,_ in model.named_modules() if target.fullmatch(n)]
    need(len(targets)==a.expected_lora_target_modules,
         f'LORA_TARGET_MODULE_COUNT_DRIFT:{len(targets)}')
    need(all('language_model.' in n and 'vision' not in n for n in targets),
         'VISION_OR_NON_LANGUAGE_LORA_TARGET_FORBIDDEN')
    if adapter is None:
        cfg=LoraConfig(task_type=a.lora_task_type, r=a.lora_rank,
                       lora_alpha=a.lora_alpha,lora_dropout=a.lora_dropout,
                       bias=a.lora_bias,target_modules=a.lora_target_regex,
                       inference_mode=False)
        model=get_peft_model(model,cfg)
    else:
        need(Path(adapter).is_dir(),f'RESUME_ADAPTER_MISSING:{adapter}')
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=True,
                                       local_files_only=True)
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    need(bool(trainable) and all('lora_' in n for n,_ in trainable),
         'BASE_WEIGHTS_NOT_FULLY_FROZEN')
    params=sum(p.numel() for _,p in trainable)
    need(params==a.expected_trainable_parameters,
         f'LORA_TRAINABLE_COUNT_DRIFT:{params}')
    model.config.use_cache=False
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant':a.gradient_checkpoint_use_reentrant})
        if callable(getattr(model,'enable_input_require_grads',None)):
            model.enable_input_require_grads()
    model.train()
    event('BF16_LORA_MODEL_READY', model_placement=device_map,
          target_modules=len(targets),trainable_parameters=params,
          checkpointing=a.gradient_checkpointing,adapter_reloaded=adapter is not None)
    return model,trainable,{str(k):str(v) for k,v in device_map.items()}


def gpu_snapshot(a):
    import torch
    values=[]
    for i in range(a.expected_gpus):
        free,total=torch.cuda.mem_get_info(i)
        values.append({'gpu':i,'free_gib':round(free/2**30,4),
                       'total_gib':round(total/2**30,4),
                       'allocated_gib':round(torch.cuda.memory_allocated(i)/2**30,4),
                       'reserved_gib':round(torch.cuda.memory_reserved(i)/2**30,4),
                       'peak_reserved_gib':round(torch.cuda.max_memory_reserved(i)/2**30,4),
                       'peak_allocated_gib':round(torch.cuda.max_memory_allocated(i)/2**30,4)})
    return values


def check_memory(a, phase, step=None, microbatch=None):
    vals=gpu_snapshot(a)
    for v in vals:
        need(v['free_gib']>=a.min_gpu_headroom_gib,
             f'GPU_HEADROOM_EXHAUSTED:{phase}:gpu={v["gpu"]}:free={v["free_gib"]}')
        need(v['peak_reserved_gib']<=a.max_gpu_peak_reserved_gib,
             f'GPU_PEAK_RESERVED_BUDGET_EXCEEDED:{phase}:gpu={v["gpu"]}:peak={v["peak_reserved_gib"]}')
    if step is not None:
        event('GPU_MEMORY',phase=phase,step=step,microbatch=microbatch,gpus=vals)
    return vals


def supervised_inputs(model, ids):
    import torch
    embedding=model.get_input_embeddings()
    need(embedding is not None and embedding.weight.device.type=='cuda',
         'TOKEN_EMBEDDINGS_NOT_ON_CUDA')
    device=embedding.weight.device
    x=torch.tensor([ids],dtype=torch.long,device=device)
    mask=torch.ones_like(x)
    labels=torch.full_like(x,-100)
    labels[0,-1]=x[0,-1]
    need(int(torch.count_nonzero(labels!=-100))==1,'LABEL_MASK_MUST_HAVE_ONE_TARGET')
    return {'input_ids':x,'attention_mask':mask,'labels':labels}


def optimizer_for(a, trainable):
    import torch
    return torch.optim.AdamW([p for _,p in trainable],lr=a.learning_rate,
                             weight_decay=a.weight_decay)


def lr_for(a,step):
    need(1<=step<=a.max_steps, 'LR_STEP_OUT_OF_RANGE')
    if step<=a.warmup_steps:
        return a.learning_rate*step/a.warmup_steps
    fraction=(step-a.warmup_steps)/(a.max_steps-a.warmup_steps)
    return a.learning_rate*0.5*(1+math.cos(math.pi*fraction))


def real_optimizer_step(a,model,trainable,opt,examples,step,rate):
    import torch
    need(bool(examples),'OPTIMIZER_STEP_MISSING_MICROBATCHES')
    opt.zero_grad(set_to_none=True)
    losses=[];sizes=[]
    for j,ids in enumerate(examples,1):
        tensors=supervised_inputs(model,ids)
        outputs=model(**tensors,use_cache=False)
        need(outputs.loss is not None and torch.isfinite(outputs.loss).item(),
             f'NONFINITE_FORWARD_LOSS:step={step}:micro={j}')
        loss=outputs.loss
        (loss/len(examples)).backward()
        losses.append(float(loss.detach().cpu()))
        sizes.append(len(ids))
        if j%a.progress_microbatch_every==0 or j==len(examples):
            event('MICROBATCH_PROGRESS',optimizer_step=step,microbatch=j,
                  microbatches=len(examples),input_tokens=len(ids),
                  loss=losses[-1],learning_rate=rate)
    counts=0
    for name,parameter in trainable:
        need(parameter.grad is not None,f'MISSING_LORA_GRADIENT:{name}')
        need(torch.isfinite(parameter.grad).all().item(),
             f'NONFINITE_LORA_GRADIENT:{name}')
        counts+=1
    norm=torch.nn.utils.clip_grad_norm_([p for _,p in trainable],a.max_grad_norm,
                                        error_if_nonfinite=True)
    need(torch.isfinite(norm).item(),'NONFINITE_GRADIENT_NORM')
    for group in opt.param_groups: group['lr']=rate
    opt.step()
    opt.zero_grad(set_to_none=True)
    mean=sum(losses)/len(losses)
    need(math.isfinite(mean),'NONFINITE_MEAN_TRAIN_LOSS')
    return {'optimizer_step':step,'microbatches':len(examples),
            'mean_loss':mean,'last_loss':losses[-1], 'learning_rate':rate,
            'min_tokens':min(sizes),'max_tokens':max(sizes),
            'grad_norm':float(norm.detach().cpu()),
            'finite_gradient_tensors':counts,'optimizer_updated':True}


def checkpoint(a,model,opt,step,phase):
    import torch
    base=a.run_dir/'checkpoints'
    base.mkdir(exist_ok=True)
    dest=base/f'step-{step:06d}'
    tmp=base/f'.step-{step:06d}-incomplete'
    need(not dest.exists() and not tmp.exists(),f'REFUSING_CHECKPOINT_OVERWRITE:{dest}')
    tmp.mkdir(exist_ok=False)
    adapter=tmp/'adapter'; adapter.mkdir(exist_ok=False)
    model.save_pretrained(str(adapter),safe_serialization=True)
    need(a.save_optimizer_state,'OPTIMIZER_STATE_SAVING_IS_REQUIRED')
    state={'optimizer':opt.state_dict(),
           'torch_cpu_rng':torch.get_rng_state(),
           'torch_cuda_rng':torch.cuda.get_rng_state_all(),
           'optimizer_step':step,
           'config_fingerprint':fingerprint(a),
           'prepared_manifest_sha256':a.prepared_manifest_sha256,
           'model_revision':a.model_revision}
    torch.save(state,tmp/'optimizer_state.pt')
    files={str(p.relative_to(tmp)):sha_file(p) for p in sorted(tmp.rglob('*')) if p.is_file()}
    need('optimizer_state.pt' in files and any(k.endswith('.safetensors') for k in files),
         'CHECKPOINT_ADAPTER_OR_OPTIMIZER_MISSING')
    manifest={'version':a.version,'status':'I01C_V2_CHECKPOINT_COMPLETE',
              'at_chicago':now(),'phase':phase,'optimizer_step':step,
              'config_fingerprint':fingerprint(a),
              'prepared_manifest_sha256':a.prepared_manifest_sha256,
              'model_revision':a.model_revision,'files_sha256':files,
              'save_optimizer_state':True}
    new_json(tmp/'checkpoint_manifest.json',manifest)
    tmp.rename(dest)
    result={'checkpoint':str(dest),'optimizer_step':step,
            'checkpoint_manifest_sha256':sha_file(dest/'checkpoint_manifest.json')}
    event('CHECKPOINT_SAVED',**result)
    return result


def verify_checkpoint(a,folder,require_training=False):
    import torch
    folder=Path(folder)
    meta_path=folder/'checkpoint_manifest.json'
    need(meta_path.is_file(),'CHECKPOINT_MANIFEST_MISSING')
    meta=load_json(meta_path)
    need(meta.get('status')=='I01C_V2_CHECKPOINT_COMPLETE' and
         meta.get('config_fingerprint')==fingerprint(a) and
         meta.get('prepared_manifest_sha256')==a.prepared_manifest_sha256 and
         meta.get('model_revision')==a.model_revision,
         'CHECKPOINT_LINEAGE_OR_STATUS_MISMATCH')
    need(meta.get('save_optimizer_state') is True and meta.get('files_sha256'),
         'CHECKPOINT_OPTIMIZER_NOT_SAVED')
    for name,digest in meta['files_sha256'].items():
        need('..' not in Path(name).parts and not Path(name).is_absolute(),
             'CHECKPOINT_UNSAFE_RELATIVE_FILE_PATH')
        file_pin(folder/name,digest,'CHECKPOINT_'+name)
    state=torch.load(folder/'optimizer_state.pt',map_location='cpu',weights_only=True)
    need(state.get('optimizer_step')==meta['optimizer_step'] and
         state.get('config_fingerprint')==fingerprint(a) and
         state.get('prepared_manifest_sha256')==a.prepared_manifest_sha256,
         'CHECKPOINT_OPTIMIZER_STATE_DRIFT')
    if require_training:
        need(meta['phase']=='train' and 0<meta['optimizer_step']<a.max_steps and
             meta['optimizer_step']%a.save_steps==0,
             'RESUME_REQUIRES_INTERMEDIATE_M1_TRAIN_CHECKPOINT')
        need(folder.name==f'step-{meta["optimizer_step"]:06d}' and
             folder.parent.name=='checkpoints', 'INVALID_RESUME_CHECKPOINT_STRUCTURE')
        prior=folder.parent.parent/'preflight.json'
        need(prior.is_file() and load_json(prior)['config_fingerprint']==fingerprint(a),
             'RESUME_PARENT_RUN_CONFIG_DRIFT')
    return meta,state


def restore_state(a, opt, state):
    import torch
    opt.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_cpu_rng'])
    torch.cuda.set_rng_state_all(state['torch_cuda_rng'])
    event('OPTIMIZER_AND_RNG_RESTORED',optimizer_step=state['optimizer_step'])


def stage_history(a,records):
    target=a.run_dir/'training_history.jsonl'
    with target.open('x',encoding='utf-8') as stream:
        for record in records:
            stream.write(canonical(record)+'\n')
    return str(target)


def write_stage_report(a,kind,status,history,checkpoints,extra):
    path=a.run_dir/'stage_report.json'
    report={'version':a.version,'at_chicago':now(),'status':status,'kind':kind,
            'config_fingerprint':fingerprint(a),
            'prepared_manifest_sha256':a.prepared_manifest_sha256,
            'model_id':a.model_id,'model_revision':a.model_revision,
            'bf16_base':True,'quantization':'none','finetuning_method':'lora',
            'optimizer_steps_completed':len(history),
            'optimizer_steps': [x['optimizer_step'] for x in history],
            'min_tokens':min(x['min_tokens'] for x in history),
            'max_tokens':max(x['max_tokens'] for x in history),
            'optimizer_updates_validated':all(x['optimizer_updated'] for x in history),
            'history_file_sha256':sha_file(a.run_dir/'training_history.jsonl'),
            'checkpoints':checkpoints,'heldout_accessed':False,
            **extra}
    new_json(path,report)
    event('STAGE_PASS',kind=kind,status=status,report=str(path))
    return report


def verify_stage_run(a,folder,kind,status):
    folder=Path(folder)
    need(folder.is_dir() and folder!=a.run_dir,f'STAGE_RUN_MISSING:{kind}')
    manifest=load_json(folder/'run_manifest.json')
    report=load_json(folder/'stage_report.json')
    need(manifest.get('final_status')=='PASS' and manifest.get('status')==status and
         report.get('status')==status and report.get('kind')==kind,
         f'STAGE_PROOF_STATUS_MISMATCH:{kind}')
    need(report['config_fingerprint']==fingerprint(a) and
         report['prepared_manifest_sha256']==a.prepared_manifest_sha256 and
         report['model_revision']==a.model_revision,
         f'STAGE_CONFIG_OR_SOURCE_DRIFT:{kind}')
    need(report.get('optimizer_updates_validated') is True and
         report.get('heldout_accessed') is False,
         f'STAGE_DID_NOT_PROVE_SAFE_UPDATES:{kind}')
    need(sha_file(folder/'training_history.jsonl')==report['history_file_sha256'],
         f'STAGE_HISTORY_SHA_DRIFT:{kind}')
    expected_steps=a.stress_steps if kind=='stress' else a.stability_steps
    need(report['optimizer_steps_completed']==expected_steps and
         len(report['optimizer_steps'])==expected_steps,
         f'STAGE_STEP_COUNT_DRIFT:{kind}')
    if kind=='stress':
        need(report['min_tokens']>=a.stress_min_tokens and
             report['fresh_frozen_base'] is True,
             'STRESS_NOT_LONG_OR_NOT_FRESH_MODEL')
    else:
        need(report.get('reloaded_previous_adapter_and_optimizer') is True and
             report.get('previous_stage_run')==str(a.stress_run),
             'STABILITY_DID_NOT_RESTORE_STRESS_CHECKPOINT')
    need(bool(report['checkpoints']),'MISSING_STAGE_CHECKPOINT_PROOF')
    for ck in report['checkpoints']:
        cp=Path(ck['checkpoint'])
        need(cp.is_relative_to(folder/'checkpoints'), 'STAGE_CHECKPOINT_OUTSIDE_STAGE_RUN')
        need(sha_file(cp/'checkpoint_manifest.json')==ck['checkpoint_manifest_sha256'],
             'STAGE_CHECKPOINT_MANIFEST_SHA_CHANGED')
        verify_checkpoint(a,cp)
    return {'run':str(folder),'kind':kind,'status':status,
            'stage_report_sha256':sha_file(folder/'stage_report.json'),
            'history_sha256':report['history_file_sha256'],
            'checkpoints':report['checkpoints']}


def run_smoke(a,kind):
    import torch
    rows,order,tokens=get_training_inputs(a,kind)
    indices=(selected_stress(a,rows['train']) if kind=='stress' else
             selected_stability(a,rows['train']))
    parent=None; previous_step=0
    if kind=='stability':
        proof=load_json(a.stress_run/'stage_report.json')
        parent=Path(proof['checkpoints'][-1]['checkpoint'])
        meta,state=verify_checkpoint(a,parent)
        previous_step=meta['optimizer_step']
    model,trainable,placement=attach_model(a,adapter=parent/'adapter' if parent else None)
    opt=optimizer_for(a,trainable)
    if parent is not None:
        restore_state(a,opt,state)
    mem_start=check_memory(a,kind+'_after_model_loading')
    history=[]
    for offset,idx in enumerate(indices,1):
        step=previous_step+offset
        training=real_optimizer_step(a,model,trainable,opt,[tokens['train'][idx]],
                                     step,a.learning_rate)
        training['chicago']=now()
        history.append(training)
        check_memory(a,kind+'_post_step',step)
        event('SMOKE_OPTIMIZER_PASS',kind=kind,**training)
    proof=checkpoint(a,model,opt,previous_step+len(history),kind)
    meta,verified=verify_checkpoint(a,Path(proof['checkpoint']))
    need(meta['optimizer_step']==previous_step+len(history),'SMOKE_CHECKPOINT_STEP_DRIFT')
    stage_history(a,history)
    report=write_stage_report(a,kind,
        'I01C_V2_STRESS_PASS_NOT_M1_TRAINING' if kind=='stress'
        else 'I01C_V2_STABILITY_PASS_NOT_M1_TRAINING',history,[proof],{
            'previous_stage_run':str(a.stress_run) if a.stress_run else None,
            'reloaded_previous_adapter_and_optimizer':parent is not None,
            'base_optimizer_step':previous_step,'final_optimizer_step':meta['optimizer_step'],
            'fresh_frozen_base':parent is None,
            'optimizer_state_reloaded':parent is not None,
            'adapter_state_reloaded':parent is not None,
            'memory_after_base':mem_start,
            'memory_final':gpu_snapshot(a), 'token_audit_sha256':sha_file(a.run_dir/'token_audit.json'),
            'vision_tower_lora':False,'training_is_smoke_only':True,
            'model_device_map':placement})


def evaluate(a,model,tokenized,step):
    import torch
    model.eval();losses=[]
    with torch.no_grad():
        for i,ids in enumerate(tokenized.values(),1):
            out=model(**supervised_inputs(model,ids),use_cache=False)
            need(out.loss is not None and torch.isfinite(out.loss).item(),
                 f'VALIDATION_LOSS_NOT_FINITE:{step}:{i}')
            losses.append(float(out.loss.detach().cpu()))
            if i % a.log_steps==0 or i==len(tokenized):
                event('VALIDATION_PROGRESS',step=step,done=i,total=len(tokenized))
    model.train()
    need(len(losses)==a.source_validation_count,'VALIDATION_NOT_FULLY_CONSUMED')
    return {'optimizer_step':step,'examples':len(losses),
            'mean_validation_loss':sum(losses)/len(losses),
            'min_validation_loss':min(losses),'max_validation_loss':max(losses),
            'chicago':now()}


def append_line(handle,record):
    handle.write(canonical(record)+'\n')
    handle.flush()


def run_training(a,resume=False):
    import torch
    rows,order,tokens=get_training_inputs(a,'train')
    need(len(tokens['train'])==a.source_train_count and
         len(tokens['validation'])==a.source_validation_count,
         'TRAINING_TOKEN_CACHE_INCOMPLETE')
    resume_state=None;start_step=0
    if resume:
        meta,resume_state=verify_checkpoint(a,a.resume_checkpoint,require_training=True)
        start_step=meta['optimizer_step']
        event('RESUME_PROOF_PASS',checkpoint=str(a.resume_checkpoint),step=start_step)
    model,trainable,placement=attach_model(a,
         adapter=(a.resume_checkpoint/'adapter') if resume else None)
    optimizer=optimizer_for(a,trainable)
    if resume:
        restore_state(a,optimizer,resume_state)
    mem_start=check_memory(a,'train_after_model_loading')
    checkpoints=[];validations=[];last=None
    history_file=a.run_dir/'training_history.jsonl'
    with history_file.open('x',encoding='utf-8') as stream:
        for step in range(start_step+1,a.max_steps+1):
            subset=order[(step-1)*a.gradient_accumulation_steps*a.per_device_batch_size:
                         step*a.gradient_accumulation_steps*a.per_device_batch_size]
            need(len(subset)==a.gradient_accumulation_steps*a.per_device_batch_size,
                 'TRAINING_ORDER_EXHAUSTED_EARLY')
            ids=[tokens['train'][ix] for ix in subset]
            rate=lr_for(a,step)
            last=real_optimizer_step(a,model,trainable,optimizer,ids,step,rate)
            last['chicago']=now()
            append_line(stream,{'event':'OPTIMIZER_STEP',**last})
            mem=check_memory(a,'post_optimizer')
            if step%a.log_steps==0 or step==a.max_steps:
                event('TRAINING_PROGRESS',**last,gpu_memory=mem)
            if step%a.eval_steps==0 or step==a.max_steps:
                validation=evaluate(a,model,tokens['validation'],step)
                validations.append(validation)
                append_line(stream,{'event':'VALIDATION',**validation})
                event('VALIDATION_PASS',**validation)
            if step%a.save_steps==0 or step==a.max_steps:
                ck=checkpoint(a,model,optimizer,step,'train')
                checkpoints.append(ck)
                append_line(stream,{'event':'CHECKPOINT',**ck,'chicago':now()})
                verify_checkpoint(a,Path(ck['checkpoint']))
    need(last is not None and last['optimizer_step']==a.max_steps,
         'TRAINING_DID_NOT_REACH_FINAL_STEP')
    need(validations and validations[-1]['optimizer_step']==a.max_steps and
         len(checkpoints)>0 and checkpoints[-1]['optimizer_step']==a.max_steps,
         'FINAL_EVAL_OR_CHECKPOINT_MISSING')
    final_adapter=a.run_dir/'final_adapter'
    final_adapter.mkdir(exist_ok=False)
    model.save_pretrained(str(final_adapter),safe_serialization=True)
    adapter_files={str(p.relative_to(final_adapter)):sha_file(p)
                   for p in sorted(final_adapter.rglob('*')) if p.is_file()}
    need(any(x.endswith('.safetensors') for x in adapter_files),
         'FINAL_ADAPTER_SAFETENSORS_MISSING')
    final_manifest={'status':'I01C_V2_FINAL_LORA_ADAPTER_COMPLETE',
                    'version':a.version,'at_chicago':now(),
                    'config_fingerprint':fingerprint(a),
                    'prepared_manifest_sha256':a.prepared_manifest_sha256,
                    'model_id':a.model_id,'model_revision':a.model_revision,
                    'source_optimizer_steps':a.max_steps,
                    'files_sha256':adapter_files,
                    'trainable_parameters':a.expected_trainable_parameters,
                    'finetuning_method':'lora','base_precision':'bf16',
                    'quantization':'none','vision_tower_lora':False}
    new_json(a.run_dir/'adapter_manifest.json',final_manifest)
    summary={'status':'I01C_V2_M1_SOURCE_ONLY_TRAINING_COMPLETE',
             'version':a.version,'at_chicago':now(),
             'config_fingerprint':fingerprint(a),
             'effective_parameters':config(a),
             'base_revision':a.model_revision,
             'prepared_manifest_sha256':a.prepared_manifest_sha256,
             'prepared_train_sha256':a.prepared_train_sha256,
             'prepared_validation_sha256':a.prepared_validation_sha256,
             'prepared_order_sha256':a.prepared_order_sha256,
             'total_source_records':a.source_train_count,
             'optimizer_steps':a.max_steps,
             'optimizer_steps_in_this_run':a.max_steps-start_step,
             'per_device_batch_size':a.per_device_batch_size,
             'gradient_accumulation_steps':a.gradient_accumulation_steps,
             'total_effective_batch':a.per_device_batch_size*a.gradient_accumulation_steps,
             'completed_full_epoch':True,
             'starting_checkpoint':str(a.resume_checkpoint) if resume else None,
             'initial_adapter_policy':'FRESH_FROZEN_BASE' if not resume else 'VERIFIED_RESUME',
             'stress_run':str(a.stress_run), 'stability_run':str(a.stability_run),
             'validation_events':validations,
             'final_optimizer_log':last,
             'model_device_map':placement,
             'memory_after_loading':mem_start, 'memory_final':gpu_snapshot(a),
             'adapter_manifest_sha256':sha_file(a.run_dir/'adapter_manifest.json'),
             'training_history_sha256':sha_file(history_file),
             'token_audit_sha256':sha_file(a.run_dir/'token_audit.json'),
             'checkpoints':checkpoints,
             'heldout_accessed':False,
             'limitations':['Source selection inherited from F-01c and G-01c; no independent cross-function train/validation independence proof',
                            'Single-process three-GPU model sharding, not DDP/FSDP',
                            'Validation is M1 source-only loss; held-out evaluation reserved for I-04']}
    new_json(a.run_dir/'training_summary.json',summary)
    event('I01C_M1_TRAINING_COMPLETE',steps=a.max_steps,
          adapter_manifest_sha256=summary['adapter_manifest_sha256'])


def finalize(a):
    need(a.run_status in ('PASS','FAIL') and a.exit_code is not None,
         'FINALIZER_STATUS_AND_EXIT_REQUIRED')
    target=a.run_dir/'run_manifest.json'
    step= a.mode[2:]
    status_name= {
      'stress':'I01C_V2_STRESS_PASS_NOT_M1_TRAINING',
      'stability':'I01C_V2_STABILITY_PASS_NOT_M1_TRAINING',
      'train':'I01C_V2_M1_SOURCE_ONLY_TRAINING_COMPLETE',
      'resume':'I01C_V2_M1_SOURCE_ONLY_TRAINING_COMPLETE'}[step]
    present=[p for p in sorted(a.run_dir.rglob('*')) if p.is_file() and not p.is_symlink()
             and p!=target]
    # Checkpoint optimizer state can be large: preserve bytes and integrity hashes.
    artifact_hashes={str(p.relative_to(a.run_dir)):{'sha256':sha_file(p),'bytes':p.stat().st_size}
                     for p in present}
    if a.run_status=='PASS':
        need(a.exit_code==0 and (a.run_dir/'preflight.json').is_file(),
             'PASS_CANNOT_OMIT_PREFLIGHT')
        target_report=a.run_dir/('stage_report.json' if step in ('stress','stability')
                                 else 'training_summary.json')
        need(target_report.is_file() and load_json(target_report)['status']==status_name,
             'PASS_CANNOT_OMIT_COMPLETE_STAGE_RESULT')
    manifest={'version':a.version,'experiment':a.experiment,'mode':a.mode,
              'status':status_name if a.run_status=='PASS' else 'I01C_V2_STAGE_FAILED',
              'final_status':a.run_status,'exit_code':a.exit_code,
              'at_chicago':now(),'config_fingerprint':fingerprint(a),
              'effective_parameters':config(a),
              'prepared_manifest_sha256':a.prepared_manifest_sha256,
              'artifacts':artifact_hashes,'run':str(a.run_dir),
              'log_file':str(a.log_file),'heldout_accessed':False}
    new_json(target,manifest)
    event('I01C_FINAL_MANIFEST',status=manifest['status'],file=str(target))


def main(argv=None):
    a=None
    try:
        a=parse(argv)
        stage=a.stage
        if stage=='preflight':preflight(a)
        elif stage=='stress':run_smoke(a,'stress')
        elif stage=='stability':run_smoke(a,'stability')
        elif stage=='train':run_training(a,resume=False)
        elif stage=='resume':run_training(a,resume=True)
        elif stage=='finalize':finalize(a)
        else:raise StageError('UNKNOWN_STAGE')
        return 0
    except Exception as error:
        print(json.dumps({'status':'I01C_V2_STAGE_FAIL',
                          'stage':a.stage if a else 'argparse',
                          'error':type(error).__name__+':'+str(error),
                          'chicago':now()}),file=sys.stderr,flush=True)
        traceback.print_exc()
        if a is not None and a.stage!='finalize':
            target=a.run_dir/'stage_failure.json'
            if not target.exists():
                try:new_json(target,{'stage':a.stage,'error':str(error),
                                     'type':type(error).__name__,
                                     'at_chicago':now(),'mode':a.mode,
                                     'config_fingerprint':fingerprint(a)})
                except Exception:pass
        return 2


if __name__=='__main__':
    sys.exit(main())
