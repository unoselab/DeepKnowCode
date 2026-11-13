#!/usr/bin/env python3
"""G-02 v2. GPT-OSS-120B M2 Standard Mixed LoRA from G-01d v4 M1 adapter.

Single process, 3-GPU sharded MXFP4 LoRA. Not DDP, FSDP, full finetuning or heldout evaluation.
All experiment configuration is CLI-required and supplied by the independent shell entrypoint.
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
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v2'
DYNAMIC_FIELDS = {'mode', 'run_dir', 'log_file', 'prepared_run', 'stress_run',
                  'stability_run', 'resume_checkpoint'}


def chicago():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def write_new(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as out:
        json.dump(data, out, ensure_ascii=False, sort_keys=True, indent=2)
        out.write('\n')


def write_jsonl_new(path, items):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as out:
        for item in items:
            out.write(canonical_json(item) + '\n')


def records(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for n, line in enumerate(stream, 1):
            require(bool(line.strip()), f'blank JSONL row {path}:{n}')
            obj = json.loads(line)
            require(isinstance(obj, dict), f'non-object JSONL row {path}:{n}')
            yield obj


def configured(a):
    return {key: str(val) if isinstance(val, Path) else val for key, val in vars(a).items()}


def fingerprint(a):
    # Static experiment identity shared across prepare/stress/stability/train.
    payload = {k:v for k,v in configured(a).items() if k not in DYNAMIC_FIELDS}
    return sha_text(canonical_json(payload))


def validate_cli(a):
    require(a.version == VERSION, 'shell/python version mismatch')
    require(re.fullmatch(r'\d{4}-\d{2}-\d{2}', a.template_date) is not None,
            'template-date must be YYYY-MM-DD')
    require(datetime.fromisoformat(a.template_date).date().isoformat() == a.template_date,
            'invalid pinned template date')
    require(re.fullmatch('[0-9a-f]{40}', a.revision) is not None, 'invalid pinned model revision')
    require('/' in a.model_id and a.model_type == 'gpt_oss', 'unsupported model family/identity')
    require(a.quant_method == 'mxfp4' and a.precision == 'bf16', 'no MXFP4/BF16 drift allowed')
    require(a.snapshot.name == a.revision, 'snapshot path must identify pinned revision')
    require(a.device_map == 'balanced', 'only validated balanced device_map supported')
    require(a.attention_backend in ('eager', 'sdpa'), 'unapproved attention backend')
    device_ids = a.device_ids.split(',')
    require(len(device_ids) == a.expected_gpus and len(set(device_ids)) == len(device_ids)
            and all(x.isdigit() for x in device_ids), 'bad GPU mapping')
    require(a.expected_gpus > 1, 'sharded execution needs multiple GPUs')
    require(a.max_gpu_gib > 0 and a.max_cpu_gib > 0 and a.min_headroom_gib > 0,
            'invalid GPU memory safety policy')
    require(0 < a.cache_reclaim_trigger_gib < a.max_gpu_gib - a.min_headroom_gib,
            'invalid inactive CUDA cache reclaim threshold')
    require(a.max_seq_length >= a.stress_min_tokens >= 128, 'stress length outside training bound')
    require(a.stress_steps >= 2 and a.stability_steps >= 4, 'smoke step policy invalid')
    require(a.dataset_train_count > 0 and a.dataset_val_count > 0,
            'mixed dataset counts must be positive')
    require(a.frozen_train_count == a.dataset_train_count and
            a.frozen_val_count == a.dataset_val_count, 'M2 must consume EXACT complete F01c frozen dataset')
    require(a.source_train_count + a.java_train_count + a.python_train_count == a.dataset_train_count,
            'M2 domain train budget mismatch')
    require(a.source_val_count + a.java_val_count + a.python_val_count == a.dataset_val_count,
            'M2 domain val budget mismatch')
    require(a.source_prepared.is_dir() and a.m1_run.is_dir(), 'M1 source/proof paths missing')
    require(a.frozen_val_quota_policy == 'SHA256_PINNED_F01C_MANIFEST',
            'M2 validation quotas must be derived in Shell from a pinned F01c manifest')
    require(a.pending_label_review_policy == 'EXISTING_F01C_FROZEN_PROTOCOL_WITH_DISCLOSED_RISK',
            'pending label risk policy must be explicit')
    for val in (a.m1_adapter_sha256,a.m1_summary_sha256,a.m1_adapter_manifest_sha256,
                a.source_prepared_train_sha256,a.source_prepared_val_sha256,a.m1_source_manifest_sha256):
        require(re.fullmatch('[0-9a-f]{64}',val) is not None,'invalid M1 Source/Adapter pin')
    require(a.per_device_batch_size == 1 and a.eval_batch_size == 1,
            'single-process sharded microbatch 1 is required by implementation')
    require(a.expected_epochs == 1 and a.max_steps > 0 and a.gradient_accumulation_steps > 0,
            'invalid epoch/step control')
    require(a.max_steps * a.per_device_batch_size * a.gradient_accumulation_steps ==
            a.dataset_train_count * a.expected_epochs,
            'global optimizer budget must consume frozen M2 mixed training once and only once')
    require(a.optimizer == 'adamw_torch' and a.lr_scheduler == 'cosine', 'unsupported optimizer/scheduler')
    require(a.lora_rank > 0 and a.lora_alpha > 0 and 0 <= a.lora_dropout < 1,
            'bad LoRA hyperparameters')
    require(a.expected_trainable_parameters > 0 and a.min_target_modules >= 4,
            'LoRA identity incomplete')
    require(a.gradient_checkpointing and not a.gradient_checkpoint_use_reentrant,
            'only audited gradient checkpointing scheme allowed')
    require(a.learning_rate > 0 and a.max_grad_norm > 0 and a.weight_decay >= 0,
            'invalid optimizer values')
    require(0 < a.warmup_steps < a.max_steps, 'warmup schedule invalid')
    require(all(x > 0 and x <= a.max_steps for x in
                (a.log_steps, a.eval_steps, a.save_steps)), 'logging/eval/save interval invalid')
    require(a.eval_steps > 0 and a.save_steps > 0, 'invalid validation/checkpoint cadence')
    require(a.allow_network in ('true', 'false') and a.save_optimizer_state,
            'network/optimizer-state controls invalid')
    require(len(set(a.target_modules.split(','))) == len(a.target_modules.split(',')),
            'duplicate LoRA target module')
    require(a.frozen_prepared.is_dir(), 'frozen M2 dataset path missing')
    for val in (a.frozen_train_sha256, a.frozen_val_sha256, a.frozen_manifest_sha256):
        require(re.fullmatch('[0-9a-f]{64}', val) is not None, 'invalid source SHA256 pin')
    if a.mode == 'prepare':
        require(not a.prepared_run and not a.stress_run and not a.stability_run and
                not a.resume_checkpoint, 'prepare takes no previous run')
    if a.mode == 'stress':
        require(bool(a.prepared_run) and not a.stress_run and not a.stability_run and
                not a.resume_checkpoint, 'stress requires only PREPARED_RUN')
    if a.mode == 'stability':
        require(bool(a.prepared_run) and bool(a.stress_run) and not a.stability_run and
                not a.resume_checkpoint, 'stability requires PREPARED_RUN STRESS_RUN')
    if a.mode in ('train', 'resume'):
        require(bool(a.prepared_run) and bool(a.stress_run) and bool(a.stability_run),
                'train requires three proof paths')
    if a.mode == 'train':
        require(not a.resume_checkpoint, 'train cannot implicitly resume')
    if a.mode == 'resume':
        require(bool(a.resume_checkpoint), 'resume requires explicit checkpoint')


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'stress', 'stability', 'train', 'resume'))
    string_names = ('version','model-id','revision','model-type','quant-method','device-ids',
                    'attention-backend','device-map','precision','allow-network','expected-torch',
                    'expected-transformers','expected-peft','expected-accelerate','expected-kernels',
                    'expected-triton','expected-huggingface-hub','frozen-train-sha256','m1-adapter-sha256','m1-summary-sha256',
                    'm1-adapter-manifest-sha256','source-prepared-train-sha256','source-prepared-val-sha256',
                    'pending-label-review-policy','frozen-val-quota-policy','m1-source-manifest-sha256',
                    'frozen-val-sha256','frozen-manifest-sha256','answer-yes','answer-no',
                    'system-text','user-prefix','user-separator','user-suffix','template-date','target-modules',
                    'lora-bias','optimizer','lr-scheduler')
    for name in string_names:
        p.add_argument('--' + name, required=True)
    for name in ('root','run-dir','log-file','snapshot','frozen-prepared','source-codes','source-prepared','m1-run'):
        p.add_argument('--' + name, required=True, type=Path)
    for name in ('prepared-run','stress-run','stability-run','resume-checkpoint'):
        p.add_argument('--' + name, required=True)  # empty string explicitly represents NOT USED
    for name in ('expected-gpus','min-compute-major','expected-shards','min-shard-bytes',
                 'frozen-train-count','frozen-val-count','dataset-train-count','dataset-val-count',
                 'source-train-count','java-train-count','python-train-count',
                 'm1-optimizer-steps','audit-failure-examples',
                 'source-val-count','java-val-count','python-val-count',
                 'max-seq-length','audit-progress-every','seed','min-target-modules',
                 'expected-trainable-parameters','lora-rank','lora-alpha','stress-steps',
                 'stress-min-tokens','stability-steps','per-device-batch-size',
                 'gradient-accumulation-steps','max-steps','expected-epochs','warmup-steps',
                 'log-steps','eval-steps','save-steps','eval-batch-size'):
        p.add_argument('--' + name, required=True, type=int)
    for name in ('max-gpu-gib','max-cpu-gib','min-headroom-gib','cache-reclaim-trigger-gib','lora-dropout',
                 'learning-rate','weight-decay','max-grad-norm'):
        p.add_argument('--' + name, required=True, type=float)
    for name in ('gradient-checkpointing','gradient-checkpoint-use-reentrant',
                 'save-optimizer-state'):
        p.add_argument('--' + name, required=True, choices=('true','false'))
    a = p.parse_args()
    for name in ('root','run_dir','log_file','snapshot','frozen_prepared','source_codes','source_prepared','m1_run'):
        setattr(a, name, getattr(a, name).expanduser().resolve())
    for name in ('prepared_run','stress_run','stability_run','resume_checkpoint'):
        setattr(a, name, Path(getattr(a,name)).expanduser().resolve()
                if getattr(a,name) else None)
    for name in ('gradient_checkpointing','gradient_checkpoint_use_reentrant',
                 'save_optimizer_state'):
        setattr(a,name, getattr(a,name) == 'true')
    validate_cli(a)
    return a


def model_preflight(a):
    from importlib.metadata import version
    import torch
    import transformers
    import peft
    import accelerate
    from transformers import AutoConfig, AutoTokenizer
    require(torch.cuda.is_available() and torch.cuda.device_count() == a.expected_gpus,
            'CUDA GPU count drift')
    require(torch.version.cuda and torch.cuda.is_bf16_supported(), 'BF16 CUDA unavailable')
    for name, actual in (('torch',torch.__version__), ('transformers',transformers.__version__),
                         ('peft',peft.__version__), ('accelerate',accelerate.__version__),
                         ('kernels',version('kernels')), ('triton',version('triton')),
                         ('huggingface-hub',version('huggingface-hub'))):
        expected = getattr(a, 'expected_' + name.replace('-', '_'))
        require(actual.startswith(expected), f'dependency mismatch {name}: {actual} != {expected}')
    for gpu in range(a.expected_gpus):
        prop = torch.cuda.get_device_properties(gpu)
        require(torch.cuda.get_device_capability(gpu)[0] >= a.min_compute_major,
                f'GPU{gpu} capability insufficient')
        require(prop.total_memory / 2**30 >= a.max_gpu_gib + a.min_headroom_gib,
                f'GPU{gpu} VRAM insufficient')
    config_path = a.snapshot / 'config.json'
    index_path = a.snapshot / 'model.safetensors.index.json'
    require(config_path.is_file() and index_path.is_file(), 'pinned snapshot missing')
    shards = set(json.loads(index_path.read_text())['weight_map'].values())
    require(len(shards) == a.expected_shards, 'weight shard count drift')
    for name in shards:
        f = a.snapshot / name
        require(f.is_file() and f.stat().st_size >= a.min_shard_bytes, 'weight shard missing/too small ' + name)
    config = AutoConfig.from_pretrained(str(a.snapshot), local_files_only=True, trust_remote_code=False)
    require(config.model_type == a.model_type and config.quantization_config.get('quant_method') == a.quant_method,
            'wrong model type or MXFP4 quantization')
    tok = AutoTokenizer.from_pretrained(str(a.snapshot), local_files_only=True, trust_remote_code=False)
    require(tok.is_fast, 'GPT-OSS tokenizer must be fast')
    return tok, {'model_id':a.model_id,'revision':a.revision,'model_type':a.model_type,
                 'quantization':a.quant_method,'config_sha256':sha256(config_path),
                 'index_sha256':sha256(index_path),'shards':len(shards)}


def verify_m1(a):
    """Pin actual completed G-01d v4 M1 weights, not a merely named directory."""
    run = a.m1_run
    summary_path, manifest_path = run/'training_summary.json', run/'adapter_manifest.json'
    require(sha256(summary_path)==a.m1_summary_sha256 and
            sha256(manifest_path)==a.m1_adapter_manifest_sha256,
            'M1 completed training evidence changed')
    summary=json.loads(summary_path.read_text()); manifest=json.loads(manifest_path.read_text())
    require(summary['status']=='G01D_V4_M1_SOURCE_ONLY_TRAINING_COMPLETE' and
            manifest['status']=='G01D_V4_M1_ADAPTER_COMPLETE', 'M1 source model is not complete')
    require(summary['optimizer_steps']==a.m1_optimizer_steps and
            manifest['source_only_training_pairs']==a.source_train_count,
            'M1 source training budget mismatch')
    require(summary['adapter_manifest_sha256']==a.m1_adapter_manifest_sha256,
            'M1 adapter summary does not bind manifest')
    f=run/'adapter'/'adapter_model.safetensors'
    require(sha256(f)==a.m1_adapter_sha256 and
            manifest['adapter_files_sha256']['adapter_model.safetensors']==a.m1_adapter_sha256,
            'M1 adapter binary checksum differs')
    for relative,digest in manifest['adapter_files_sha256'].items():
        require('..' not in Path(relative).parts and not Path(relative).is_absolute(),
                'unsafe M1 adapter path')
        require(sha256(run/'adapter'/relative)==digest, 'M1 adapter file changed '+relative)
    require(manifest['prepared_manifest_sha256']==a.m1_source_manifest_sha256,
            'M1 frozen prepared identity drift')
    require(summary['effective_parameters']['model_id']==a.model_id and
            summary['effective_parameters']['revision']==a.revision,
            'M1 model or revision differs from M2')
    return {'m1_run':str(run),'adapter_sha256':a.m1_adapter_sha256,
            'adapter_manifest_sha256':a.m1_adapter_manifest_sha256,
            'summary_sha256':a.m1_summary_sha256,
            'status':summary['status'],'m1_source_prepared_sha256':a.m1_source_manifest_sha256,
            'chat_template_sha256':manifest['adapter_files_sha256']['chat_template.jinja']}


def verify_frozen(a):
    """Verify frozen F01c Mixed pairs, all three source corpora and M1 source identity."""
    paths={'train':a.frozen_prepared/'train.jsonl',
           'validation':a.frozen_prepared/'validation.jsonl',
           'manifest':a.frozen_prepared/'manifest.json'}
    expected={'train':a.frozen_train_sha256,'validation':a.frozen_val_sha256,
              'manifest':a.frozen_manifest_sha256}
    for key,path in paths.items():
        require(path.is_file() and sha256(path)==expected[key], 'Frozen F01c M2 hash drift: '+key)
    manifest=json.loads(paths['manifest'].read_text())
    require(manifest['status']=='PREPARED_PENDING_LABEL_REVIEW',
            'unexpected F01c dataset status: review frozen approval explicitly')
    require(manifest['sample']['train']['sha256']==expected['train'] and
            manifest['sample']['validation']['sha256']==expected['validation'],
            'F01c manifest sample hash mismatch')
    require(manifest['sample']['train']['total']==a.dataset_train_count and
            manifest['sample']['validation']['total']==a.dataset_val_count,
            'F01c frozen M2 counts differ')
    require(manifest['training_plan']['order']=='standard-mixed-shuffled' and
            manifest['training_plan']['global_batch']==a.per_device_batch_size*a.gradient_accumulation_steps and
            manifest['training_plan']['steps']==a.max_steps,
            'F01c frozen M2 protocol/budget incompatible')
    planned=manifest['training_plan']
    desired_train={'java_source':a.source_train_count,'java_mixed':a.java_train_count,
                   'python_mixed':a.python_train_count}
    desired_val={'java_source':a.source_val_count,'java_mixed':a.java_val_count,
                 'python_mixed':a.python_val_count}
    require(planned['train']==desired_train and planned['validation']==desired_val,
            'M2 per-domain frozen quotas do not match Shell: '
            +repr({'manifest_train':planned['train'],'shell_train':desired_train,
                   'manifest_validation':planned['validation'],'shell_validation':desired_val}))
    corpora=('java_source','java_mixed','python_mixed')
    codes={}; hashes={}; code_paths={}
    for corpus in corpora:
        entry=manifest['dataset']['paths'][corpus]
        code_file=Path(entry['data.jsonl']).expanduser().resolve()
        require(code_file.is_file(), 'missing F01c corpus code file '+str(code_file))
        digest=manifest['dataset']['sha256'][corpus]['data.jsonl']
        require(sha256(code_file)==digest, 'F01c data.jsonl content drift '+corpus)
        code_paths[corpus]=str(code_file)
        codes[corpus]={}; hashes[corpus]=digest
    selected={}; stats={}; raw_count={}
    for split in ('train','validation'):
        seen=set(); arr=[]; tally=collections.Counter()
        for i,record in enumerate(records(paths[split]),1):
            corpus=record.get('corpus')
            require(corpus in corpora and type(record.get('label')) is int and record['label'] in (0,1),
                    f'invalid F01c {split} corpus/label row={i}')
            ids=(str(record['id1']),str(record['id2']))
            require(ids[0] and ids[1],f'empty F01c IDs row={i}')
            identity=(corpus,ids[0],ids[1],record['label'])
            require(identity not in seen,'duplicate frozen F01c pair '+str(identity))
            seen.add(identity)
            for field in ('code_a_override','code_b_override'):
                if field in record:
                    require(corpus=='java_mixed' and isinstance(record[field],str) and record[field].strip(),
                            'non-Java or empty transformed override: '+field)
            arr.append(record);tally[corpus]+=1
        require(len(arr)==(a.dataset_train_count if split=='train' else a.dataset_val_count),
                'F01c frozen row count drift')
        require(dict(tally)==(desired_train if split=='train' else desired_val),
                'F01c domain quota drift '+split)
        selected[split]=arr;stats[split]=dict(tally)
    # Load only code IDs referenced by the immutable selected pairs; no reselection.
    needed={c:set() for c in corpora}
    for arr in selected.values():
        for row in arr:
            needed[row['corpus']].update((str(row['id1']),str(row['id2'])))
    for c in corpora:
        for obj in records(code_paths[c]):
            idx=str(obj['idx']).strip()
            if idx in needed[c]:
                require(idx not in codes[c] and isinstance(obj['func'],str) and obj['func'].strip(),
                        'duplicate/missing code body: '+c+'/'+idx)
                codes[c][idx]=obj['func']
        require(set(codes[c])==needed[c], 'missing code IDs in frozen F01c: '+c)
    # M2 must reuse the EXACT 12,000 M1 source TRAIN pairs. The M1 source
    # validation set (412) is a different sampling policy from frozen F01c M2
    # validation. Do not conflate the two or modify M2 validation records.
    source_train=a.source_prepared/'train.jsonl'
    source_val=a.source_prepared/'validation.jsonl'
    require(source_train.is_file() and sha256(source_train)==a.source_prepared_train_sha256,
            'G01c M1 source training SHA256 changed')
    require(source_val.is_file() and sha256(source_val)==a.source_prepared_val_sha256,
            'G01c M1 source validation SHA256 changed')
    m1_source={(str(r['id1']),str(r['id2']),r['label']) for r in records(source_train)}
    mixed_source={(str(r['id1']),str(r['id2']),r['label'])
                  for r in selected['train'] if r['corpus']=='java_source'}
    require(len(m1_source)==a.source_train_count and
            len(mixed_source)==a.source_train_count and m1_source==mixed_source,
            'BLOCKED: M2 Java Source Training Pair IDs differ from G01d trained M1')
    require(isinstance(manifest.get('heldout_overlap'),dict) and
            manifest['heldout_overlap'].get('status')=='PASS' and
            manifest['heldout_overlap'].get('overlap_count')==0,
            'F01c heldout overlap audit is missing or not PASS')
    return selected,codes,{'f01c_frozen_sha256':expected,'code_paths':code_paths,
             'code_sha256':hashes,'domain_counts':stats,
             'm1_source_train_sha256':a.source_prepared_train_sha256,
             'm1_source_validation_sha256':a.source_prepared_val_sha256,
             'm1_source_pair_identity':'TRAIN_EXACT_MATCH',
             'm1_source_validation_identity':'DISTINCT_VALIDATION_PROTOCOL_SHA256_PINNED',
             'heldout_audit_inherited_from_f01c':manifest['heldout_overlap'],
             'manual_label_review_status':'PENDING_AND_NOT_INDEPENDENTLY_VERIFIED',
             'pending_label_review_policy':a.pending_label_review_policy,
             'frozen_data_modified':False}


def effective_codes(row,codes):
    corpus=row['corpus']; base=codes[corpus]
    return (row.get('code_a_override',base[str(row['id1'])]),
            row.get('code_b_override',base[str(row['id2'])]))


def messages_for(a,c1,c2,language):
    require(language in ('java','python'), 'unsupported mixed language')
    require('Language: java\n' in a.user_prefix, 'Shell prompt prefix language anchor absent')
    prefix=a.user_prefix.replace('Language: java\n','Language: '+language+'\n')
    return [{'role':'system','content':a.system_text},
            {'role':'user','content':prefix+c1+a.user_separator+c2+a.user_suffix}]


def boundary(tok, messages, yes, no, template_date):
    """G-01b-proven assistant final-turn contextual one-token label with full template roundtrip."""
    # GPT-OSS uses strftime_now("%Y-%m-%d") to add a dynamic Current date.
    # When the G-01c v1 dataset was prepared (2026-11-12 Chicago), the resulting
    # prompt hash incorporated THAT date. Pin it without editing the official
    # chat template, tokenizer, source code or frozen prepared hashes.
    stamp = datetime.fromisoformat(template_date)
    def pinned_strftime(fmt):
        return stamp.strftime(fmt)
    result = {}
    for label, answer in ((1,yes),(0,no)):
        full = tok.apply_chat_template(messages + [{'role':'assistant','channel':'final','content':answer}],
                                       tokenize=False, add_generation_prompt=False,
                                       strftime_now=pinned_strftime)
        require(isinstance(full,str), 'Harmony chat template failed to render')
        pos = full.rfind(answer)
        require(pos > 0, 'answer absent in final assistant turn')
        prefix, suffix = full[:pos],full[pos+len(answer):]
        require(bool(suffix), 'Harmony terminator absent')
        pids = tok.encode(prefix, add_special_tokens=False)
        joined = tok.encode(prefix+answer, add_special_tokens=False)
        require(joined[:len(pids)] == pids and len(joined) == len(pids)+1,
                'answer token is not contextually one stable token')
        require(tok.apply_chat_template(messages + [{'role':'assistant','channel':'final','content':answer}],
                                        tokenize=True, add_generation_prompt=False,
                                        strftime_now=pinned_strftime) ==
                tok.encode(full, add_special_tokens=False), 'template ID/text mismatch')
        result[label] = {'prefix':prefix,'suffix':suffix,
                         'prefix_ids':pids,'answer_id':joined[-1]}
    require(result[0]['prefix'] == result[1]['prefix'] and
            result[0]['suffix'] == result[1]['suffix'], 'YES/NO completion prefixes differ')
    require(result[0]['answer_id'] != result[1]['answer_id'], 'YES/NO share same token')
    return result


def audit_pair(a,tok,row,codes):
    c1,c2=effective_codes(row,codes)
    corpus=row['corpus']
    language='python' if corpus=='python_mixed' else 'java'
    b=boundary(tok,messages_for(a,c1,c2,language),a.answer_yes,a.answer_no,a.template_date)
    label=row['label']
    data={'corpus':corpus,'id1':str(row['id1']),'id2':str(row['id2']),
          'label':label,'original_row_sha256':sha_text(canonical_json(row)),
          'code1_sha256':sha_text(c1),'code2_sha256':sha_text(c2),
          'prompt_sha256':sha_text(b[label]['prefix']),
          'n_tokens':len(b[label]['prefix_ids'])+1,'answer_id':b[label]['answer_id']}
    for field in ('code_a_override','code_b_override','origin'):
        if field in row:data[field]=row[field]
    return data


def lengths_stats(values):
    s = sorted(values)
    require(bool(s), 'missing length observations')
    return {'min':s[0], 'median':s[len(s)//2], 'p95':s[math.ceil(0.95*len(s))-1],
            'max':s[-1], 'mean':sum(s)/len(s)}


def prepare(a,tok,model_evidence,pairs,codes,frozen):
    pdir=a.run_dir/'prepared';pdir.mkdir(exist_ok=False)
    files={}; stats={}; over=[]; counts={}
    # Audit EVERY frozen pair before approving GPU loading. No silent truncation,
    # resampling, label rewrite or threshold change based on the outcome.
    for split in ('train','validation'):
        rows=[];seen=set();domain_counts=collections.Counter()
        for i,original in enumerate(pairs[split],1):
            audit=audit_pair(a,tok,original,codes)
            ident=(audit['corpus'],audit['id1'],audit['id2'],audit['label'])
            require(ident not in seen,'duplicate frozen mixed pair')
            seen.add(ident); rows.append(audit);domain_counts[audit['corpus']]+=1
            if audit['n_tokens']>a.max_seq_length:
                over.append({'split':split,'index_1based':i,'corpus':audit['corpus'],
                             'ids':[audit['id1'],audit['id2']],
                             'tokens':audit['n_tokens'],'limit':a.max_seq_length})
            if i%a.audit_progress_every==0:
                print(json.dumps({'event':'M2_TOKEN_AUDIT_PROGRESS','split':split,
                                  'processed':i,'at_chicago':chicago()}),flush=True)
        counts[split]=dict(domain_counts)
        stats[split]=lengths_stats([r['n_tokens'] for r in rows])
        print(json.dumps({'event':'M2_TOKEN_AUDIT_SPLIT','split':split,'count':len(rows),
                          'token_stats':stats[split],'overlength':sum(x['split']==split for x in over),
                          'at_chicago':chicago()}),flush=True)
        files[split]=rows
    if over:
        diagnostic={'version':VERSION,'status':'G02_V2_BLOCKED_OVERLENGTH_FROZEN_MIXED',
                    'at_chicago':chicago(),'max_seq_length':a.max_seq_length,
                    'total_overlength':len(over),'examples':over[:a.audit_failure_examples],
                    'domain_counts':counts,'token_stats':stats,
                    'no_dataset_modified':True,'no_training_started':True}
        write_new(a.run_dir/'token_length_blocker.json',diagnostic)
        raise RuntimeError(f'Frozen F01c M2 has {len(over)} GPTOSS pairs longer than {a.max_seq_length}; '
                           'no truncation or resampling; inspect token_length_blocker.json')
    for split in ('train','validation'):
        path=pdir/(split+'.jsonl');write_jsonl_new(path,files[split])
        files[split]={'path':str(path),'sha256':sha256(path),'rows':len(files[split])}
    train=list(records(pdir/'train.jsonl'))
    order=list(range(len(train)));random.Random(a.seed).shuffle(order)
    require(len(set(order))==len(train) and
            len(order)==a.max_steps*a.per_device_batch_size*a.gradient_accumulation_steps,
            'incomplete frozen M2 epoch exposure')
    order_path=pdir/'order.jsonl'
    write_jsonl_new(order_path,({'index':i,'row_sha256':sha_text(canonical_json(train[i]))} for i in order))
    files['order']={'path':str(order_path),'sha256':sha256(order_path),'rows':len(order)}
    manifest={'version':VERSION,'status':'G02_V2_PREPARE_PASS_NOT_TRAINED',
              'at_chicago':chicago(),'config_fingerprint':fingerprint(a),
              'effective_parameters':configured(a),'model':model_evidence,
              'm1_initial_adapter':verify_m1(a), 'frozen_mixed':frozen,
              'training_budget':{'examples':len(train),'micro_batch':a.per_device_batch_size,
                 'gradient_accumulation':a.gradient_accumulation_steps,
                 'optimizer_steps':a.max_steps,'epochs':a.expected_epochs,
                 'world_size':1,'model_shards_gpu_count':a.expected_gpus},
              'files':files,'token_stats':stats,'domain_counts':counts,
              'tokenizer_identity':{'chat_template_sha256':sha_text(tok.chat_template),
                'answer_policy':'ASSISTANT_FINAL_ONE_CONTEXTUAL_TOKEN_ONLY',
                'unmodified_prompt_policy':'G01B_VERIFIED_HARMONY_PLUS_LANGUAGE',
                'template_date':a.template_date,'no_input_truncation':True},
              'limitations':['Frozen F01c Java-mixed transformed negatives were marked PENDING_LABEL_REVIEW',
                'Inherited heldout overlap audit, not independently repeated',
                'Prepared row identities are byte-pinned and never resampled',
                'Single-process sharded 3 GPU, not DDP/FSDP',
                'This is preparation only; no GPU weights or optimizer steps']}
    write_new(pdir/'manifest.json',manifest)
    result={'status':manifest['status'],'prepared_manifest_sha256':sha256(pdir/'manifest.json'),
            'prepared_path':str(pdir),'config_fingerprint':fingerprint(a),
            'manual_label_review_status':'PENDING'}
    write_new(a.run_dir/'prepare_result.json',result)
    finish_manifest(a,result['status'],['prepare_result.json','prepared/manifest.json',
           'prepared/train.jsonl','prepared/validation.jsonl','prepared/order.jsonl'])
    print(json.dumps(result),flush=True)


def validate_artifact_manifest(run, filename, status, key):
    run = Path(run)
    f = run/filename
    require(f.is_file(), f'missing {key} proof: {f}')
    envelope_file = run/'run_manifest.json'
    require(envelope_file.is_file(), f'missing {key} run manifest')
    envelope = json.loads(envelope_file.read_text(encoding='utf-8'))
    require(envelope.get('status') == status, f'{key} run manifest is not PASS')
    require(envelope.get('artifacts',{}).get(filename) == sha256(f),
            f'{key} proof checksum differs')
    data = json.loads(f.read_text(encoding='utf-8'))
    require(data.get('status') == status, f'{key} proof is not PASS: {data.get("status")}')
    return data


def bind_prepared(a):
    envelope=json.loads((a.prepared_run/'run_manifest.json').read_text())
    require(envelope['status']=='G02_V2_PREPARE_PASS_NOT_TRAINED', 'unapproved G02 prepared status')
    manifest=validate_artifact_manifest(a.prepared_run,'prepared/manifest.json',envelope['status'],'prepare')
    require(manifest['config_fingerprint']==fingerprint(a) and
            manifest['tokenizer_identity']['template_date']==a.template_date,
            'M2 prepared experiment/date fingerprint changed')
    require(manifest['model']['revision']==a.revision and
            manifest['training_budget']['optimizer_steps']==a.max_steps,
            'M2 model revision or optimizer budget drift')
    require(manifest['m1_initial_adapter']['adapter_sha256']==a.m1_adapter_sha256,
            'M1 initial adapter changed after preparation')
    verify_m1(a)
    pdir=a.prepared_run/'prepared'
    for name in ('train','validation','order'):
        entry=manifest['files'][name]
        fp=pdir/(name+'.jsonl')
        require(fp.is_file() and sha256(fp)==entry['sha256'] and
                envelope['artifacts']['prepared/'+name+'.jsonl']==sha256(fp),
                'G02 prepared file drift '+name)
    rows={split:list(records(pdir/(split+'.jsonl'))) for split in ('train','validation')}
    require(len(rows['train'])==a.dataset_train_count and
            len(rows['validation'])==a.dataset_val_count,'G02 prepared count drift')
    order_items=list(records(pdir/'order.jsonl'))
    order=[x['index'] for x in order_items]
    require(len(order)==len(rows['train']) and set(order)==set(range(len(rows['train']))),
            'G02 standard mixed order incomplete/duplicated')
    require(all(x['row_sha256']==sha_text(canonical_json(rows['train'][x['index']]))
                for x in order_items),'M2 prepared order SHA256/row identity mismatch')
    return manifest,rows,order


def verify_stage_proof(a, run, filename, status, prepared_manifest_sha256, label):
    data = validate_artifact_manifest(run,filename,status,label)
    env = json.loads((Path(run)/'run_manifest.json').read_text())
    token_file = Path(run)/'token_audit.json'
    require(token_file.is_file() and env['artifacts'].get('token_audit.json') == sha256(token_file),
            label+' token audit missing or altered')
    audit = json.loads(token_file.read_text())
    require(audit['status'] == 'G02_V2_PREPARED_TOKEN_CACHE_VERIFIED' and
            audit['config_fingerprint'] == fingerprint(a) and
            audit.get('template_date') == a.template_date,
            label+' token audit identity drift')
    require(data['config_fingerprint'] == fingerprint(a), label+' configuration drift')
    require(data['prepared_manifest_sha256'] == prepared_manifest_sha256,
            label+' prepared identity drift')
    require(data['steps_completed'] >= 1 and data['adapter_parameters_changed'],
            label+' did not validate adapter optimizer updates')
    ckpt = Path(run)/'checkpoints'/f'step-{data["steps_completed"]:06d}'
    ckpt_meta = ckpt/'checkpoint_manifest.json'
    require(ckpt_meta.is_file() and sha256(ckpt_meta)==data['checkpoint_manifest_sha256'],
            label+' checkpoint manifest hash drift')
    ckpt_data = json.loads(ckpt_meta.read_text())
    require(ckpt_data['prepared_manifest_sha256']==prepared_manifest_sha256 and
            ckpt_data['config_fingerprint']==fingerprint(a),
            label+' checkpoint identity drift')
    for relative,digest in ckpt_data['files_sha256'].items():
        require(sha256(ckpt/relative)==digest, label+' checkpoint content drift: '+relative)
    for sample in data['history']:
        require(math.isfinite(sample['loss']) and sample['loss'] >= 0 and
                math.isfinite(sample['gradient_norm_before_clip']),
                label+' contains nonfinite training evidence')
        require(max(item['peak_allocated_gib'] for item in sample['gpu_memory']) < a.max_gpu_gib,
                label+' peak memory violates budget')
    return data


def verify_stages(a, prepared_manifest_sha256):
    stress = verify_stage_proof(a,a.stress_run,'stress.json',
                 'G02_V2_STRESS_PASS_NOT_M2_TRAINING',prepared_manifest_sha256,'stress')
    require(stress['steps_completed'] == a.stress_steps and
            min(x['tokens'] for x in stress['history']) >= a.stress_min_tokens,
            'long sequence stress did not meet configured minimum')
    if a.mode in ('train','resume'):
        stability = verify_stage_proof(a,a.stability_run,'stability.json',
                    'G02_V2_STABILITY_PASS_NOT_M2_TRAINING',prepared_manifest_sha256,'stability')
        require(stability['steps_completed'] == a.stability_steps and
                stability['reloaded_previous_adapter_and_optimizer'] is True,
                'stability did not validate adapter + optimizer checkpoint reload')
    return stress


def load_codes(a,needed):
    # Every corpus SHA256 revalidated before loading. The F01c manifest is immutable.
    frozen=json.loads((a.frozen_prepared/'manifest.json').read_text())
    mapping={}
    for corpus, ids in needed.items():
        fp=Path(frozen['dataset']['paths'][corpus]['data.jsonl']).expanduser().resolve()
        digest=frozen['dataset']['sha256'][corpus]['data.jsonl']
        require(sha256(fp)==digest,'F01c code drift: '+corpus)
        result={}
        for obj in records(fp):
            idx=str(obj['idx']).strip()
            if idx in ids:
                require(idx not in result and isinstance(obj['func'],str) and obj['func'].strip(),
                        'invalid selected code ID '+corpus+'/'+idx)
                result[idx]=obj['func']
        require(set(result)==ids, 'selected mixed code ID missing '+corpus)
        mapping[corpus]=result
    return mapping


def precache_prepared_tokens(a,tok,rows,codes):
    from array import array
    streamhash=hashlib.sha256();counts={};lengths={}
    for split in ('train','validation'):
        counts[split]=0;lengths[split]=[]
        for i,row in enumerate(rows[split],1):
            c1,c2=effective_codes(row,codes)
            require(sha_text(c1)==row['code1_sha256'] and
                    sha_text(c2)==row['code2_sha256'],
                    f'M2 effective Code SHA mismatch split={split} row={i}')
            language='python' if row['corpus']=='python_mixed' else 'java'
            b=boundary(tok,messages_for(a,c1,c2,language),a.answer_yes,a.answer_no,a.template_date)
            chosen=b[row['label']]
            require('Current date: '+a.template_date in chosen['prefix'],
                    'GPTOSS pinned template date was not rendered')
            ids=chosen['prefix_ids']+[chosen['answer_id']]
            differences={}
            for key,expected,actual in [('prompt_sha256',row['prompt_sha256'],sha_text(chosen['prefix'])),
                                        ('answer_id',row['answer_id'],chosen['answer_id']),
                                        ('n_tokens',row['n_tokens'],len(ids))]:
                if expected != actual: differences[key]={'expected':expected,'actual':actual}
            if differences:
                proof={'status':'G02_V2_TOKEN_CACHE_FAIL_CLOSED','split':split,
                       'row_index_1based':i,'corpus':row['corpus'],
                       'ids':[row['id1'],row['id2']], 'differences':differences,
                       'at_chicago':chicago()}
                write_new(a.run_dir/'token_audit_failure.json',proof)
                raise RuntimeError('G02 Harmony Token Identity mismatch before GPU loading')
            require(len(ids)<=a.max_seq_length,'G02 GPTOSS max sequence exceeded after preparation')
            row['_verified_input_ids']=array('I',ids)
            row['_yes_answer_id']=b[1]['answer_id'];row['_no_answer_id']=b[0]['answer_id']
            streamhash.update(canonical_json({'split':split,'corpus':row['corpus'],
                'id1':row['id1'],'id2':row['id2'],'label':row['label']}).encode())
            streamhash.update(row['_verified_input_ids'].tobytes())
            counts[split]+=1;lengths[split].append(len(ids))
            if i%a.audit_progress_every==0:
                print(json.dumps({'event':'G02_TOKEN_CACHE_PROGRESS','split':split,
                                  'processed':i,'at_chicago':chicago()}),flush=True)
    obj={'version':VERSION,'status':'G02_V2_PREPARED_TOKEN_CACHE_VERIFIED',
         'at_chicago':chicago(),'prepared_run':str(a.prepared_run),
         'config_fingerprint':fingerprint(a),'chat_template_sha256':sha_text(tok.chat_template),
         'template_date':a.template_date,'split_counts':counts,
         'token_sequence_sha256':streamhash.hexdigest(),
         'lengths':{k:lengths_stats(v) for k,v in lengths.items()},
         'verified_before_model_loading':True,'unmodified_frozen_dataset':True,
         'training_tokenization_policy':'USE_ONLY_VERIFIED_CACHED_TOKEN_IDS'}
    write_new(a.run_dir/'token_audit.json',obj)
    print(json.dumps({'event':'G02_TOKEN_CACHE_PASS','counts':counts,
                      'sha256':streamhash.hexdigest(),'at_chicago':chicago()}),flush=True)
    return obj


def sample_tensors(a,tok,row,codes,device):
    import torch
    # No runtime Harmony re-render here. IDs are cached after strict validation
    # against immutable prepared SHA256, answer ID and length BEFORE model loading.
    require('_verified_input_ids' in row and '_yes_answer_id' in row and
            '_no_answer_id' in row, 'unverified training sample; token audit missing')
    ids = row['_verified_input_ids']
    require(len(ids) == row['n_tokens'] and len(ids) <= a.max_seq_length and
            ids[-1] == row['answer_id'], 'cached token identity drift')
    labels = [-100]*(len(ids)-1) + [row['answer_id']]
    return {'input_ids':torch.tensor([ids],dtype=torch.long,device=device),
            'labels':torch.tensor([labels],dtype=torch.long,device=device)},len(ids)


def check_quantization(a,model):
    placement = getattr(model,'hf_device_map',None)
    require(isinstance(placement,dict) and bool(placement), 'no sharded placement map')
    values = {str(v) for v in placement.values()}
    require(not any(v in ('cpu','disk','meta') for v in values),
            'CPU/disk/meta offload prohibited')
    require(all(str(g) in values or f'cuda:{g}' in values
                for g in range(a.expected_gpus)), 'not all configured GPUs assigned')
    quant = getattr(model,'hf_quantizer',None)
    require(quant is not None, 'native MXFP4 quantizer absent')
    qc = quant.quantization_config
    method = qc.get('quant_method') if isinstance(qc,dict) else getattr(qc,'quant_method',None)
    dequant = qc.get('dequantize') if isinstance(qc,dict) else getattr(qc,'dequantize',None)
    require(method == a.quant_method and dequant is not True,
            'dequantized model or unexpected quantization method')
    return placement


def attach_model(a,resume_adapter=None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);random.seed(a.seed)
    verify_m1(a)  # CHECK binary SHA before expensive GPU model loading.
    model=AutoModelForCausalLM.from_pretrained(
        str(a.snapshot),local_files_only=True,trust_remote_code=False,
        device_map=a.device_map,dtype=torch.bfloat16,attn_implementation=a.attention_backend,
        max_memory={**{i:f'{a.max_gpu_gib}GiB' for i in range(a.expected_gpus)},
                    'cpu':f'{a.max_cpu_gib}GiB'})
    placement=check_quantization(a,model)
    targets=a.target_modules.split(','); hits=collections.Counter()
    for name,module in model.named_modules():
        fam=name.rsplit('.',1)[-1]
        if fam in targets:
            require(isinstance(module,torch.nn.Linear),'LoRA target is not nn.Linear: '+name)
            hits[fam]+=1
    require(all(hits[x] for x in targets) and sum(hits.values())>=a.min_target_modules,
            'G02 LoRA target modules absent')
    # CRITICAL: M2 begins from G01d M1 trained PEFT adapter, never random init.
    # Stability resumes its stress adapter instead; full train independently starts M1.
    initial=Path(resume_adapter) if resume_adapter else (a.m1_run/'adapter')
    model=PeftModel.from_pretrained(model,str(initial),is_trainable=True,local_files_only=True)
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    require(trainable and all('lora_' in name for name,_ in trainable),
            'M1 LoRA adapter not trainable or frozen base drift')
    nparams=sum(p.numel() for _,p in trainable)
    require(nparams==a.expected_trainable_parameters,'M1 LoRA parameter count drift')
    config=model.peft_config[model.active_adapter]
    require(config.r==a.lora_rank and config.lora_alpha==a.lora_alpha and
            abs(config.lora_dropout-a.lora_dropout)<1e-9 and
            set(config.target_modules)==set(targets), 'M1 adapter LoRA hyperparameter drift')
    model.config.use_cache=False
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            'use_reentrant':a.gradient_checkpoint_use_reentrant})
    mem,_=maintain_memory(a,phase='m2_lora_attach')
    print(json.dumps({'event':'M2_FROM_M1_ADAPTER_PASS','initial_adapter':str(initial),
                      'initial_M1_SHA256':a.m1_adapter_sha256,
                      'trainable_parameters':nparams,'target_family_counts':dict(hits),
                      'memory':mem,'at_chicago':chicago()}),flush=True)
    return model,trainable,placement


def memory_snapshot(a):
    """Tensor memory, CUDA allocator cache and *physical* free memory are different.

    max_memory in from_pretrained limits model placement, not peak training cache.
    mem_get_info is driver reported free memory (includes other GPU processes).
    """
    import torch
    rows = []
    for gpu in range(a.expected_gpus):
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu)
        allocated = torch.cuda.memory_allocated(gpu) / 2**30
        reserved = torch.cuda.memory_reserved(gpu) / 2**30
        rows.append({
            'gpu': gpu,
            'allocated_gib': allocated,
            'reserved_gib': reserved,
            'inactive_cache_gib': max(0.0, reserved - allocated),
            'peak_allocated_gib': torch.cuda.max_memory_allocated(gpu) / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved(gpu) / 2**30,
            'driver_free_gib': free_bytes / 2**30,
            'driver_total_gib': total_bytes / 2**30,
        })
    return rows


def enforce_memory(a, rows, phase='unknown', optimizer_step=None, microbatch=None):
    """Fail on real live/peak tensor pressure or post-reclaim driver headroom.

    PyTorch reserved includes reclaimable inactive cache. A high reserved value
    is a reclaim trigger, *not* evidence that tensors exceeded the hard cap.
    """
    for m in rows:
        problems = []
        if m['allocated_gib'] >= a.max_gpu_gib:
            problems.append('active allocated memory above hard cap')
        if m['peak_allocated_gib'] >= a.max_gpu_gib:
            problems.append('historical peak tensor allocation above hard cap')
        if m['driver_free_gib'] < a.min_headroom_gib:
            problems.append('driver free memory below minimum headroom after reclaim')
        if problems:
            proof = {'status': 'G02_V2_MEMORY_GUARD_FAIL',
                     'phase': phase, 'optimizer_step': optimizer_step,
                     'microbatch': microbatch, 'gpu': m['gpu'],
                     'reasons': problems, 'limits': {
                         'max_allocated_gib': a.max_gpu_gib,
                         'min_driver_free_gib': a.min_headroom_gib},
                     'gpu_memory': rows, 'at_chicago': chicago()}
            proof_file = a.run_dir / 'memory_guard_failure.json'
            if not proof_file.exists():
                write_new(proof_file, proof)
            print(json.dumps({'event': 'MEMORY_GUARD_FAIL', **proof}),
                  file=sys.stderr, flush=True)
            raise RuntimeError(f'GPU {m["gpu"]} memory safety failure: '
                               + ', '.join(problems) + f'; phase={phase}')


def maintain_memory(a, phase, optimizer_step=None, microbatch=None):
    """Reclaim only at safe graph-free boundaries, never during forward/backward.

    This does not relax the model-placement limit or the peak-tensor budget.
    Threshold derives only from shell-defined memory cap and headroom.
    """
    import torch
    before = memory_snapshot(a)
    trigger_gpus = [m['gpu'] for m in before
                    if m['reserved_gib'] >= a.cache_reclaim_trigger_gib
                    or m['driver_free_gib'] < a.min_headroom_gib]
    if trigger_gpus:
        # synchronize() and empty_cache() operate on the selected CUDA device.
        # Only inactive allocations may be released; active tensors are preserved.
        for gpu in trigger_gpus:
            with torch.cuda.device(gpu):
                torch.cuda.synchronize(gpu)
                torch.cuda.empty_cache()
        after = memory_snapshot(a)
    else:
        after = before
    # Never permit a high *actual* peak or depleted driver headroom to pass.
    enforce_memory(a, after, phase=phase,
                   optimizer_step=optimizer_step, microbatch=microbatch)
    event = None
    if trigger_gpus:
        event = {'event': 'GPU_INACTIVE_CACHE_RECLAIMED',
                 'phase': phase, 'optimizer_step': optimizer_step,
                 'microbatch': microbatch, 'gpus': trigger_gpus,
                 'trigger_reserved_gib': a.cache_reclaim_trigger_gib,
                 'before': before, 'after': after,
                 'at_chicago': chicago()}
        print(json.dumps(event), flush=True)
    return after, event


def optimizer_step(a,model,trainable,opt,tok,row,codes,step,rate,grad_accum=1):
    """One real optimizer step with explicit gradient accumulation and supervised final token."""
    import torch
    require(len(row) == grad_accum, 'optimizer microbatch count drift')
    device = model.get_input_embeddings().weight.device
    require(device.type == 'cuda', 'model embeddings unexpectedly not on CUDA')
    opt.zero_grad(set_to_none=True)
    losses,token_counts = [],[]
    cache_reclaims = []
    for i,example in enumerate(row,1):
        tensors,n = sample_tensors(a,tok,example,codes,device)
        out = model(**tensors,use_cache=False)
        loss = out.loss
        require(loss is not None and torch.isfinite(loss.detach()).item(),
                f'nonfinite loss on microbatch {i}')
        (loss/grad_accum).backward()
        losses.append(float(loss.detach().float().item()))
        token_counts.append(n)
        del out,loss,tensors
        # The current microbatch autograd graph is now gone; gradient buffers
        # remain valid for accumulation. Never empty cache inside backward.
        _, event = maintain_memory(a, phase='post_microbatch',
                                   optimizer_step=step, microbatch=i)
        if event is not None:
            cache_reclaims.append(event)
    require(any(p.grad is not None and torch.any(p.grad.detach()!=0).item()
                for _,p in trainable), 'no nonzero LoRA gradients')
    for _,p in trainable:
        if p.grad is not None:
            require(bool(torch.isfinite(p.grad.detach()).all().item()), 'nonfinite LoRA gradient')
    norm = torch.nn.utils.clip_grad_norm_([p for _,p in trainable],a.max_grad_norm)
    require(bool(torch.isfinite(norm).item()), 'nonfinite gradient norm')
    for group in opt.param_groups:
        group['lr'] = rate
    opt.step()
    # AdamW has consumed gradients; release them prior to allocator audit.
    # This does not change optimizer states, adapter parameters or RNG order.
    opt.zero_grad(set_to_none=True)
    mem, event = maintain_memory(a, phase='post_optimizer', optimizer_step=step,
                                 microbatch=grad_accum)
    if event is not None:
        cache_reclaims.append(event)
    log = {'optimizer_step':step,'samples':len(row),'min_tokens':min(token_counts),
           'max_tokens':max(token_counts),'mean_tokens':sum(token_counts)/len(token_counts),
           'labels':dict(collections.Counter(str(r['label']) for r in row)),
           'loss':sum(losses)/len(losses),'loss_min':min(losses),'loss_max':max(losses),
           'gradient_norm_before_clip':float(norm.detach().float().item()),
           'learning_rate':rate,'gpu_memory':mem,
           'inactive_cache_reclaim_events':len(cache_reclaims),
           'inactive_cache_reclaim_gpus':dict(collections.Counter(
               gpu for event in cache_reclaims for gpu in event['gpus'])),
           'at_chicago':chicago()}
    return log


def finish_manifest(a,status,artifact_names,extra=None):
    mapping = {}
    for name in artifact_names:
        fp = a.run_dir/name
        require(fp.is_file(), 'cannot register missing artifact '+str(fp))
        mapping[name] = sha256(fp)
    obj = {'version':VERSION,'status':status,'mode':a.mode,'at_chicago':chicago(),
           'config_fingerprint':fingerprint(a),'effective_parameters':configured(a),
           'artifacts':mapping,'extra':extra or {}}
    write_new(a.run_dir/'run_manifest.json',obj)
    return obj


def select_stress_samples(a,rows):
    # Longest examples of alternating labels, no duplicate Pair ID.
    grouped = {label:sorted((r for r in rows if r['label']==label),
                            key=lambda r:(-r['n_tokens'],r['id1'],r['id2']))
               for label in (0,1)}
    chosen=[]
    for i in range(a.stress_steps):
        label=1 if i%2==0 else 0
        rank=i//2
        require(len(grouped[label])>rank,'insufficient long-sequence examples')
        row=grouped[label][rank]
        require(row['n_tokens'] >= a.stress_min_tokens,
                'not enough long examples for both labels')
        chosen.append(row)
    return chosen


def select_stability_samples(a,rows):
    # Deterministic spread of lengths within each label, alternating YES/NO.
    per_label = {}
    count_each = math.ceil(a.stability_steps/2)
    for label in (0,1):
        ordered = sorted((r for r in rows if r['label']==label),
                         key=lambda x:(x['n_tokens'],x['id1'],x['id2']))
        require(len(ordered) >= count_each, 'insufficient stability pairs')
        idx = [round((i+1)*len(ordered)/(count_each+1)) for i in range(count_each)]
        require(len(set(idx)) == len(idx), 'stability selection duplicates')
        per_label[label] = [ordered[j] for j in idx]
    return [per_label[i%2][i//2] for i in range(a.stability_steps)]


def smoke(a,tok,rows,codes,prepared_hash):
    import torch
    selected = select_stress_samples(a,rows['train']) if a.mode=='stress' else \
               select_stability_samples(a,rows['train'])
    # Stability additionally verifies exact PEFT adapter + AdamW/RNG checkpoint restoration.
    resume_state = None
    from_checkpoint = None
    if a.mode == 'stability':
        from_checkpoint = a.stress_run/'checkpoints'/f'step-{a.stress_steps:06d}'
        resume_state = validate_checkpoint(a,from_checkpoint,prepared_hash,train_checkpoint=False)
    model,trainable,placement = attach_model(
        a,resume_adapter=from_checkpoint/'adapter' if from_checkpoint else None)
    model.train()
    opt = torch.optim.AdamW([p for _,p in trainable],lr=a.learning_rate,
                            weight_decay=a.weight_decay)
    if resume_state:
        opt.load_state_dict(resume_state['optimizer'])
        torch.set_rng_state(resume_state['torch_cpu_rng'])
        torch.cuda.set_rng_state_all(resume_state['torch_cuda_rng'])
        random.setstate(resume_state['python_rng'])
    _,target_param = next((n,p) for n,p in trainable if 'lora_B' in n)
    initial = target_param.detach().float().cpu().clone()
    history = []
    for ix,row in enumerate(selected,1):
        log = optimizer_step(a,model,trainable,opt,tok,[row],codes,ix,a.learning_rate)
        log.update({'label':row['label'],'tokens':row['n_tokens'],
                    'pair':[row['id1'],row['id2']]})
        history.append(log)
        print(json.dumps({'event':a.mode.upper()+'_OPTIMIZER_STEP_PASS',**log}),flush=True)
    changed = bool(torch.any(initial != target_param.detach().float().cpu()).item())
    require(changed,'LoRA B weights did not change')
    # Smoke must prove checkpoint save/read, not just backward feasibility.
    ckpt = checkpoint(a,model,opt,tok,len(history),prepared_hash)
    check = validate_checkpoint(a,ckpt,prepared_hash,train_checkpoint=False)
    require(check['optimizer_step']==len(history),'smoke optimizer checkpoint incorrect')
    status = ('G02_V2_STRESS_PASS_NOT_M2_TRAINING' if a.mode=='stress' else
              'G02_V2_STABILITY_PASS_NOT_M2_TRAINING')
    output = {'version':VERSION,'status':status,'at_chicago':chicago(),
              'config_fingerprint':fingerprint(a),'prepared_manifest_sha256':prepared_hash,
              'steps_completed':len(history),'adapter_parameters_changed':changed,
              'target_family':'MXFP4_SHARDED_LORA','label_policy':'ONE_HARMONY_FINAL_TOKEN',
              'history':history,'gpu_placement':placement,
              'checkpoint_manifest_sha256':sha256(ckpt/'checkpoint_manifest.json'),
              'reloaded_previous_adapter_and_optimizer':bool(resume_state),
              'mixed_stress_checkpoint':str(from_checkpoint) if from_checkpoint else None,
              'full_m2_training_performed':False}
    fn = a.mode+'.json'
    write_new(a.run_dir/fn,output)
    finish_manifest(a,status,[fn,'token_audit.json',str((ckpt/'checkpoint_manifest.json').relative_to(a.run_dir))],
                    {'prepared_run':str(a.prepared_run),'m1_run':str(a.m1_run),'smoke_checkpoint':str(ckpt)})
    print(json.dumps({'status':status,'steps':len(history),'at_chicago':chicago()}),flush=True)


def schedule_lr(a,step):
    """1-based optimizer step; same explicit warmup+cosine convention as M1 baseline."""
    require(1 <= step <= a.max_steps,'learning rate step outside training')
    if step <= a.warmup_steps:
        return a.learning_rate * step/a.warmup_steps
    ratio = (step-a.warmup_steps)/(a.max_steps-a.warmup_steps)
    return a.learning_rate * 0.5 * (1+math.cos(math.pi*ratio))


def evaluate(a,model,tok,rows,codes,step):
    import torch
    model.eval()
    device = model.get_input_embeddings().weight.device
    nlls=[];correct=0
    try:
        with torch.no_grad():
            for ix,row in enumerate(rows,1):
                tensors,_ = sample_tensors(a,tok,row,codes,device)
                out = model(**tensors,use_cache=False)
                loss = out.loss
                require(loss is not None and torch.isfinite(loss.detach()).item(),
                        f'evaluation nonfinite loss row={ix}')
                nlls.append(float(loss.detach().float().item()))
                # Contextual one-token YES/NO margin at last prompt position.
                logits = out.logits[0,-2,:]
                yes,no = row['_yes_answer_id'],row['_no_answer_id']
                pred = int((logits[yes]-logits[no]).float().item()>0)
                correct += int(pred == row['label'])
                del tensors,out,logits,loss
        mem = memory_snapshot(a)
        enforce_memory(a,mem)
        result = {'step':step,'validation_nll':sum(nlls)/len(nlls),
                  'validation_accuracy_diagnostic':correct/len(rows),
                  'examples':len(rows),'gpu_memory':mem,'at_chicago':chicago(),
                  'note':'Validation is frozen mixed Java/Python target, NOT independent heldout Python test.'}
        return result
    finally:
        model.train()


def checkpoint(a,model,opt,tok,step,prepared_sha256):
    import torch
    ckpt_root = a.run_dir/'checkpoints'
    ckpt_root.mkdir(exist_ok=True)
    dest = ckpt_root/f'step-{step:06d}'
    require(not dest.exists(),'refusing to replace checkpoint '+str(dest))
    dest.mkdir()
    adapter = dest/'adapter'
    model.save_pretrained(str(adapter),safe_serialization=True)
    tok.save_pretrained(str(adapter))
    # Torch payload consists exclusively of trusted tensors/dicts, loaded weights_only=True.
    state_path = dest/'optimizer_state.pt'
    require(a.save_optimizer_state,'checkpoint optimizer state must be preserved')
    torch.save({'optimizer':opt.state_dict(),
                'torch_cpu_rng':torch.get_rng_state(),
                'torch_cuda_rng':torch.cuda.get_rng_state_all(),
                'python_rng':random.getstate(),
                'optimizer_step':step,'config_fingerprint':fingerprint(a),
                'prepared_manifest_sha256':prepared_sha256},state_path)
    files = {str(f.relative_to(dest)):sha256(f) for f in dest.rglob('*') if f.is_file()}
    require(bool(files) and 'optimizer_state.pt' in files,'empty checkpoint')
    meta={'version':VERSION,'status':'G02_V2_CHECKPOINT_COMPLETE',
          'at_chicago':chicago(),'optimizer_step':step,
          'config_fingerprint':fingerprint(a),
          'prepared_manifest_sha256':prepared_sha256,
          'files_sha256':files,'save_optimizer_state':True}
    write_new(dest/'checkpoint_manifest.json',meta)
    print(json.dumps({'event':'CHECKPOINT_PASS','optimizer_step':step,
                      'path':str(dest),'at_chicago':chicago()}),flush=True)
    return dest


def validate_checkpoint(a,ckpt,prepared_hash,train_checkpoint):
    import torch
    ckpt = Path(ckpt)
    meta_path = ckpt/'checkpoint_manifest.json'
    require(meta_path.is_file(), 'checkpoint is not complete')
    meta = json.loads(meta_path.read_text())
    require(meta['status']=='G02_V2_CHECKPOINT_COMPLETE', 'checkpoint completion marker invalid')
    require(meta['config_fingerprint']==fingerprint(a) and
            meta['prepared_manifest_sha256']==prepared_hash,
            'checkpoint model/dataset/config mismatch')
    for relative,digest in meta['files_sha256'].items():
        file = ckpt/relative
        require(file.is_file() and sha256(file)==digest, 'checkpoint file hash mismatch '+relative)
    state = torch.load(ckpt/'optimizer_state.pt',map_location='cpu',weights_only=True)
    require(state['config_fingerprint']==fingerprint(a) and
            state['prepared_manifest_sha256']==prepared_hash,
            'optimizer checkpoint identity mismatch')
    require(state['optimizer_step']==meta['optimizer_step'] and
            0 < state['optimizer_step'] <= a.max_steps,
            'invalid optimizer checkpoint step')
    if train_checkpoint:
        require(state['optimizer_step'] < a.max_steps and
                state['optimizer_step'] % a.save_steps == 0,
                'resumption requires an intermediate full-training checkpoint')
        require(ckpt.parent.name == 'checkpoints' and
                ckpt.parent.parent.name.endswith(('-train','-resume')),
                'resumption requires training-run checkpoint')
    return state


def append_history(handle,data):
    handle.write(canonical_json(data)+'\n')
    handle.flush()
    os.fsync(handle.fileno())


def train(a,tok,rows,codes,order,prepared_hash):
    import torch
    state = None
    if a.mode=='resume':
        state = validate_checkpoint(a,a.resume_checkpoint,prepared_hash,train_checkpoint=True)
    model,trainable,placement = attach_model(
        a,resume_adapter=(a.resume_checkpoint/'adapter') if state else None)
    model.train()
    opt = torch.optim.AdamW([p for _,p in trainable],lr=a.learning_rate,
                            weight_decay=a.weight_decay)
    begin = 0
    if state:
        begin = state['optimizer_step']
        opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_cpu_rng'])
        torch.cuda.set_rng_state_all(state['torch_cuda_rng'])
        random.setstate(state['python_rng'])
    out_history = a.run_dir/'training_history.jsonl'
    final_validations=[];last_log=None;checkpoints=[]
    with out_history.open('x',encoding='utf-8') as stream:
        for step in range(begin+1,a.max_steps+1):
            offset=(step-1)*a.gradient_accumulation_steps*a.per_device_batch_size
            upto=offset+a.gradient_accumulation_steps*a.per_device_batch_size
            micro = [rows['train'][index] for index in order[offset:upto]]
            require(len(micro)==a.gradient_accumulation_steps, 'train schedule did not cover full microbatches')
            rate = schedule_lr(a,step)
            last_log = optimizer_step(a,model,trainable,opt,tok,micro,codes,step,rate,
                                      grad_accum=a.gradient_accumulation_steps)
            append_history(stream,{'event':'OPTIMIZER_STEP',**last_log})
            if step==begin+1 or step % a.log_steps==0 or step==a.max_steps:
                print(json.dumps({'event':'M2_TRAIN_PROGRESS',**last_log}),flush=True)
            if step % a.eval_steps==0 or step==a.max_steps:
                val = evaluate(a,model,tok,rows['validation'],codes,step)
                final_validations.append(val)
                append_history(stream,{'event':'VALIDATION',**val})
                print(json.dumps({'event':'M2_MIXED_VALIDATION',**val}),flush=True)
            if step % a.save_steps==0 or step==a.max_steps:
                check = checkpoint(a,model,opt,tok,step,prepared_hash)
                checkpoints.append(str(check))
                append_history(stream,{'event':'CHECKPOINT','optimizer_step':step,
                                       'path':str(check),'at_chicago':chicago()})
    require(last_log is not None and last_log['optimizer_step']==a.max_steps,
            'full M2 did not reach configured final optimizer step')
    adapter_path=a.run_dir/'adapter'
    require(not adapter_path.exists(),'refusing to replace final adapter')
    model.save_pretrained(str(adapter_path),safe_serialization=True)
    tok.save_pretrained(str(adapter_path))
    files={str(x.relative_to(adapter_path)):sha256(x) for x in adapter_path.rglob('*') if x.is_file()}
    require(files,'final adapter files missing')
    adapter_identity = {'status':'G02_V2_M2_ADAPTER_COMPLETE',
                        'at_chicago':chicago(),
                        'config_fingerprint':fingerprint(a),
                        'prepared_manifest_sha256':prepared_hash,
                        'adapter_files_sha256':files,
                        'mixed_training_pairs':a.dataset_train_count,
                        'optimizer_steps':a.max_steps,
                        'adapter_path':str(adapter_path)}
    write_new(a.run_dir/'adapter_manifest.json',adapter_identity)
    summary={'version':VERSION,'status':'G02_V2_M2_STANDARD_MIXED_TRAINING_COMPLETE',
             'at_chicago':chicago(),'config_fingerprint':fingerprint(a),
             'effective_parameters':configured(a),
             'memory_policy':{'max_active_gib':a.max_gpu_gib,
                              'min_driver_free_gib':a.min_headroom_gib,
                              'cache_reclaim_trigger_gib':a.cache_reclaim_trigger_gib,
                              'reserved_gib_is_not_hard_cap':True},
             'prepared_manifest_sha256':prepared_hash,
             'prepared_run':str(a.prepared_run),'m1_run':str(a.m1_run),'stress_run':str(a.stress_run),
             'stability_run':str(a.stability_run),
             'is_resumed':bool(state),'resumed_from_step':begin,
             'optimizer_steps':a.max_steps,
             'global_effective_batch':a.gradient_accumulation_steps*a.per_device_batch_size,
             'train_sample_exposures':a.dataset_train_count*a.expected_epochs,
             'initial_m1_adapter_sha256':a.m1_adapter_sha256,
             'training_order':'STANDARD_MIXED_FROZEN_SEEDED_SHUFFLE_ONCE',
             'newly_executed_steps':a.max_steps-begin,
             'final_optimizer_log':last_log,
             'last_validation':final_validations[-1] if final_validations else None,
             'validation_events':final_validations,
             'checkpoints_written_this_run':checkpoints,
             'gpu_placement':placement,
             'adapter_manifest_sha256':sha256(a.run_dir/'adapter_manifest.json'),
             'training_history_sha256':sha256(out_history),
             'token_audit_sha256':sha256(a.run_dir/'token_audit.json'),
             'heldout_test_evaluated':False,
             'limitations':['This training uses a single-process sharded 3-GPU model, not DDP/FSDP',
                            'Frozen mixed validation is not independent heldout Python test performance',
                            'M2 source sample identity is checked against G01d M1; historic F01b differs',
                            'F01c frozen transformed Java negatives are flagged PREPARED_PENDING_LABEL_REVIEW; independent label adjudication not claimed']}
    write_new(a.run_dir/'training_summary.json',summary)
    finish_manifest(a,summary['status'],['training_summary.json','training_history.jsonl',
                'adapter_manifest.json','token_audit.json'],{'adapter_path':str(adapter_path),
                'prepared_run':str(a.prepared_run),'m1_run':str(a.m1_run),'is_resumed':bool(state)})
    print(json.dumps({'status':summary['status'],'steps':a.max_steps,
                      'adapter':str(adapter_path),'at_chicago':chicago()}),flush=True)


def main():
    a = parse()
    os.environ['HF_HUB_OFFLINE']='1' if a.allow_network=='false' else '0'
    require(a.run_dir.is_dir() and not any(a.run_dir.iterdir()),
            'Shell must create a new empty run directory')
    start = {'version':VERSION,'status':'G02_V2_STARTED_NOT_COMPLETED',
             'at_chicago':chicago(),'mode':a.mode,
             'template_date':a.template_date,
             'config_fingerprint':fingerprint(a),
             'effective_parameters':configured(a),
             'memory_policy':{'max_active_gib':a.max_gpu_gib,
                              'min_driver_free_gib':a.min_headroom_gib,
                              'cache_reclaim_trigger_gib':a.cache_reclaim_trigger_gib,
                              'reserved_gib_is_not_hard_cap':True}}
    write_new(a.run_dir/'run_start.json',start)
    tok,model_evidence = model_preflight(a)
    m1=verify_m1(a)
    require(sha_text(tok.chat_template)==m1['chat_template_sha256'],
            'BLOCKED: GPTOSS Harmony template differs from the completed G01d M1 adapter')
    if a.mode=='prepare':
        pairs,codes,frozen = verify_frozen(a)
        prepare(a,tok,model_evidence,pairs,codes,frozen)
        return
    manifest,rows,order = bind_prepared(a)
    require(manifest['tokenizer_identity']['chat_template_sha256']==sha_text(tok.chat_template),
            'Harmony chat template changed after preparation')
    require(manifest['tokenizer_identity']['template_date'] == a.template_date,
            'frozen M2 template date differs from Shell-pinned Harmony date')
    need = {c:set() for c in ('java_source','java_mixed','python_mixed')}
    for arr in rows.values():
        for r in arr:need[r['corpus']].update((str(r['id1']),str(r['id2'])))
    codes=load_codes(a,need)
    precache_prepared_tokens(a,tok,rows,codes)
    prepared_hash=sha256(a.prepared_run/'prepared'/'manifest.json')
    if a.mode=='stress':
        smoke(a,tok,rows,codes,prepared_hash)
    elif a.mode=='stability':
        verify_stages(a,prepared_hash)
        smoke(a,tok,rows,codes,prepared_hash)
    else:
        verify_stages(a,prepared_hash)
        train(a,tok,rows,codes,order,prepared_hash)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'event':'G02_FAIL','error_type':type(exc).__name__,
                          'error':str(exc),'at_chicago':chicago()}),file=sys.stderr,flush=True)
        raise
