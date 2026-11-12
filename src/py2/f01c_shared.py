#!/usr/bin/env python3
"""F-01c shared infrastructure extracted from audited F-01b v1; standalone copy.

All model and experiment parameters are required CLI args from the shell.
"""
import hashlib
import inspect
import json
import os
import platform
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from f01c_variants import TokenPolicy, checked_training_tokens, resolve_answer_ids
VERSION = 'v3'

def chicago_now():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def canonical_hash(value):
    return sha_text(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')))


def exclusive_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write('\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'Missing or empty required file: {path}')
    return path.resolve()


def prompt(args, code_a, code_b):
    return args.prompt_prefix + code_a + args.prompt_b_separator + code_b + args.prompt_suffix


def code_hash(code):
    return sha_text(re.sub(r'\s+', '', code))


def load_codes(path):
    result = {}
    with Path(path).open(encoding='utf-8-sig') as file:
        for n, line in enumerate(file, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            key, body = str(obj['idx']).strip(), obj['func']
            if not key or key in result or not isinstance(body, str) or not body.strip():
                raise ValueError(f'Bad code record at {path}:{n}')
            result[key] = body
    if not result:
        raise ValueError('Empty source corpus')
    return result


def pair_rows(path, codes):
    with Path(path).open(encoding='utf-8-sig') as file:
        for n, line in enumerate(file, 1):
            if not line.strip():
                continue
            items = line.split()
            if len(items) != 3 or items[2] not in ('0', '1'):
                raise ValueError(f'Invalid pair at {path}:{n}')
            a, b, label = items
            if a not in codes or b not in codes:
                raise ValueError(f'Unresolved code ID at {path}:{n}')
            yield a, b, int(label)


def cached_model(args):
    from transformers import AutoConfig, AutoTokenizer
    import torch
    if not re.fullmatch(r'[0-9a-f]{40}', args.model_revision):
        raise ValueError('Pinned --model-revision must be a full immutable 40-hex SHA')
    model_dir = args.hf_home / 'hub' / ('models--' + args.model_id.replace('/', '--'))
    snapshot = model_dir / 'snapshots' / args.model_revision
    if not snapshot.is_dir():
        raise FileNotFoundError('Pinned model snapshot missing: ' + str(snapshot))
    index = require_file(snapshot / 'model.safetensors.index.json')
    mapping = read_json(index).get('weight_map')
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('Empty/invalid safetensors index')
    shards = sorted(set(mapping.values()))
    if not all(re.fullmatch(r'[^/]+\.safetensors', x) for x in shards):
        raise ValueError('Invalid weight shard name')
    shard_sizes = {}
    for x in shards:
        p = require_file(snapshot / x)
        if p.stat().st_size < 4096:
            raise ValueError(f'Incomplete weight shard: {x}')
        shard_sizes[x] = p.stat().st_size
    config_file = require_file(snapshot / 'config.json')
    config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True,
                                        trust_remote_code=False)
    if config.model_type != args.expected_model_type:
        raise RuntimeError(f'Model architecture mismatch: {config.model_type} != {args.expected_model_type}')
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError('Fast tokenizer is required')
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError('No EOS/pad token available')
        tokenizer.pad_token = tokenizer.eos_token
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if torch.cuda.device_count() != args.expected_world_size:
        raise RuntimeError(f'Expected {args.expected_world_size} visible GPUs, got {torch.cuda.device_count()}')
    if args.precision != 'bf16' or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 requested but unavailable or incorrectly configured')
    gpu = []
    for rank in range(args.expected_world_size):
        prop = torch.cuda.get_device_properties(rank)
        gib = prop.total_memory / (1024 ** 3)
        if gib < args.min_vram_gib:
            raise RuntimeError(f'GPU {rank} has only {gib:.1f} GiB')
        gpu.append({'rank': rank, 'name': prop.name, 'vram_gib': round(gib, 3)})
    policy = TokenPolicy(args.token_policy, args.answer_yes, args.answer_no,
                         args.expected_yes_token, args.expected_no_token)
    answer_ids, answer_audit = resolve_answer_ids(tokenizer,
        prompt(args, args.probe_code_a, args.probe_code_b), policy)
    return {'snapshot': str(snapshot.resolve()), 'revision': args.model_revision,
            'index_sha256': sha256(index), 'config_sha256': sha256(config_file),
            'shards': shard_sizes,
            'tokenizer_sha256': sha256(require_file(snapshot / 'tokenizer.json')),
            'tokenizer_class': type(tokenizer).__name__, 'answer_ids': answer_ids,
            'answer_audit': answer_audit, 'model_type': config.model_type,
            'gpu': gpu, 'torch': torch.__version__}, tokenizer


def configuration(args):
    """Record every effective Shell -> Python argument; no omitted controls."""
    effective = {key: (str(value) if isinstance(value, Path) else value)
                 for key, value in vars(args).items() if key != 'command'}
    effective['version'] = VERSION
    effective['global_batch'] = (
        args.per_device_batch_size * args.gradient_accumulation_steps * args.expected_world_size)
    effective['target_modules'] = args.target_modules.split(',')
    return effective


def read_eval_fingerprints(args):
    heldout = set()
    info = {}
    for path in (args.heldout_java_codes, args.heldout_python_codes):
        path = require_file(path)
        rows = 0
        with path.open(encoding='utf-8-sig') as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                body = obj.get('code')
                if not isinstance(body, str):
                    raise ValueError(f'Invalid held-out code {path}:{lineno}')
                heldout.add(code_hash(body))
                rows += 1
        info[str(path)] = {'sha256': sha256(path), 'rows': rows}
    return heldout, info


def collate(batch, pad_id):
    import torch
    longest = max(len(x['input_ids']) for x in batch)
    fields = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for item in batch:
        n = longest - len(item['input_ids'])
        fields['input_ids'].append(item['input_ids'] + [pad_id] * n)
        fields['attention_mask'].append(item['attention_mask'] + [0] * n)
        fields['labels'].append(item['labels'] + [-100] * n)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in fields.items()}


def make_training_arguments(args, checkpoints):
    from transformers import TrainingArguments
    config = dict(
        output_dir=str(checkpoints), seed=args.seed, data_seed=args.seed,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps, weight_decay=args.weight_decay,
        bf16=(args.precision == 'bf16'), fp16=(args.precision == 'fp16'), optim=args.optimizer,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': args.gradient_checkpoint_use_reentrant},
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_drop_last=args.dataloader_drop_last,
        full_determinism=args.full_determinism, bf16_full_eval=args.bf16_full_eval,
        max_grad_norm=args.max_grad_norm,
        label_smoothing_factor=args.label_smoothing_factor,
        logging_steps=args.log_steps,
        logging_first_step=args.logging_first_step, report_to=args.report_to,
        save_strategy=args.save_strategy, save_steps=args.eval_steps,
        save_total_limit=(None if args.save_total_limit == 0 else args.save_total_limit),
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        remove_unused_columns=args.remove_unused_columns,
        save_safetensors=args.save_safetensors,
        disable_tqdm=args.disable_tqdm,
        eval_accumulation_steps=args.eval_accumulation_steps,
    )
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if 'eval_strategy' in signature:
        config['eval_strategy'] = args.eval_strategy
    elif 'evaluation_strategy' in signature:
        config['evaluation_strategy'] = args.eval_strategy
    else:
        raise RuntimeError('Installed transformers lacks evaluation strategy option')
    missing = set(config) - set(signature)
    if missing:
        raise RuntimeError(f'Transformers TrainingArguments incompatible; no silent dropping: {missing}')
    return TrainingArguments(**config)

