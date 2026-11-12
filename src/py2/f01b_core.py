#!/usr/bin/env python3
"""F-01b v1 independent common M1 source-only LoRA training engine.

Model choices / experiment parameters are ONLY in the Shell entrypoint and
passed as required CLI arguments. This module contains implementation logic.
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

from f01b_variants import TokenPolicy, checked_training_tokens, resolve_answer_ids

VERSION = 'v1'  # implementation version, not a training parameter
A10_OVERLAP_FIELDS = (
    'train_test_function_overlap_exact', 'train_test_function_overlap_normalized',
    'validation_test_function_overlap_exact', 'validation_test_function_overlap_normalized',
    'train_test_pair_overlap_exact', 'train_test_pair_overlap_normalized',
    'validation_test_pair_overlap_normalized',
)


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


def find_a10(args):
    if args.a10_audit_file:
        return require_file(Path(args.a10_audit_file))
    if args.a10_discovery != 'newest-valid':
        raise ValueError('Explicit --a10-audit-file required for selected A10 policy')
    paths = list((args.root / 'results/a-10').glob('*/audit.json'))
    # 'newest-valid' is an explicit Shell policy; resolved absolute path + hash
    # are frozen in the manifest. Never silently accept an invalid audit.
    paths.sort(key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)
    for path in paths:
        try:
            obj = read_json(path)
            if obj.get('status') == 'EXACT_SPLIT_CHECK_PASSED_REVIEW_PENDING' and not obj.get('blocking_findings'):
                return path.resolve()
        except (OSError, ValueError, TypeError):
            continue
    raise FileNotFoundError('No passing A-10 audit found; specify F01B_A10_AUDIT_FILE in Shell')


def validate_a10(args, file, count_by_split):
    a10 = read_json(file)
    if a10.get('status') != 'EXACT_SPLIT_CHECK_PASSED_REVIEW_PENDING' or a10.get('blocking_findings'):
        raise RuntimeError('A-10 audit not approved for this preflight')
    if a10.get('test', {}).get('pair_rows') != args.expected_a10_test_pairs:
        raise RuntimeError('A-10 test panel count differs from Shell pin')
    source = a10.get('corpora', {}).get(args.source_corpus, {})
    for name in A10_OVERLAP_FIELDS:
        if source.get(name) != 0:
            raise RuntimeError(f'A-10 leakage finding: {name}={source.get(name)}')
    for split in ('train', 'validation'):
        field = 'train' if split == 'train' else 'validation'
        if sum(count_by_split[split].values()) != source.get(field + '_pairs'):
            raise RuntimeError(f'Raw {split} pair count differs from A-10')
        expected = source.get(field + '_labels', {})
        if {str(k): count_by_split[split][k] for k in (0, 1)} != {
                str(k): v for k, v in expected.items()}:
            raise RuntimeError(f'Raw {split} class counts differ from A-10')
    return {'path': str(file), 'sha256': sha256(file),
            'status': a10['status'], 'a10_test_pairs': args.expected_a10_test_pairs,
            'zero_overlap_fields': list(A10_OVERLAP_FIELDS)}


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


def validate_source(args):
    source = {name: require_file(args.source_dir / name) for name in
              ('data.jsonl', 'train.txt', 'valid.txt')}
    staging = require_file(args.staging_manifest)
    if read_json(staging).get('status') != 'STAGED_CLEANED_NOT_APPROVED_FOR_TRAINING':
        raise RuntimeError('Unexpected source staging manifest status')
    ground_truth = require_file(args.ground_truth)
    ground_truth_hash = sha256(ground_truth)
    if ground_truth_hash != args.expected_ground_truth_sha256:
        raise RuntimeError('Held-out Ground Truth SHA256 mismatch')
    codes = load_codes(source['data.jsonl'])
    counts = {}
    for name, file in [('train', source['train.txt']), ('validation', source['valid.txt'])]:
        counts[name] = Counter(label for _, _, label in pair_rows(file, codes))
    a10_file = find_a10(args)
    a10_info = validate_a10(args, a10_file, counts)
    return {
        'source_paths': {k: str(v) for k, v in source.items()},
        'source_sha256': {k: sha256(v) for k, v in source.items()},
        'staging_manifest': str(staging), 'staging_sha256': sha256(staging),
        'ground_truth_sha256': ground_truth_hash, 'ground_truth_path': str(ground_truth),
        'a10': a10_info, 'raw_pairs': {k: dict(v) for k, v in counts.items()},
    }, codes, counts


def configuration(args):
    """Record every effective Shell -> Python argument; no omitted controls."""
    effective = {key: (str(value) if isinstance(value, Path) else value)
                 for key, value in vars(args).items() if key != 'command'}
    effective['version'] = VERSION
    effective['global_batch'] = (
        args.per_device_batch_size * args.gradient_accumulation_steps * args.expected_world_size)
    effective['target_modules'] = args.target_modules.split(',')
    return effective


def source_signature(args, source, model):
    """Stable smoke->train gate, excluding only intentional smoke sample/step budgets."""
    c = configuration(args)
    for key in ('max_train_examples', 'max_val_examples', 'max_steps', 'eval_steps',
                'log_steps', 'warmup_steps', 'purpose', 'run_dir', 'prepared_dir',
                'adapter_dir', 'checkpoint_dir', 'log_file', 'log_dir',
                'log_timestamp', 'log_prefix', 'log_timestamp_format'):
        del c[key]
    return canonical_hash({
        'shared_training_controls': c, 'source': source['source_sha256'],
        'staging': source['staging_sha256'], 'a10': source['a10']['sha256'],
        'ground_truth': source['ground_truth_sha256'],
        'answers': {str(k): v for k, v in model['answer_ids'].items()},
        'implementation_sha256': implementation_hashes(args),
    })


def implementation_hashes(args):
    return {p.name: sha256(p) for p in (
        args.root / 'src/py2/f-01b-train-source-lora.py',
        args.root / 'src/py2/f01b_core.py', args.root / 'src/py2/f01b_variants.py',
        args.root / 'src/sh2/f-01b-train-source-lora.sh')}


def validate_args(args):
    if not args.model_mode or not args.model_id or not args.model_family:
        raise ValueError('Model identity must be explicitly defined')
    if not re.fullmatch(r'[a-z0-9-]+', args.model_mode):
        raise ValueError('Invalid model mode')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_ids:
        raise ValueError('CUDA_VISIBLE_DEVICES differs from GPU list explicitly supplied in Shell')
    if args.train_order != 'shuffled':
        raise ValueError('M1 requires Trainer shuffled training order')
    if args.adapter_dir != args.run_dir/'adapter' or args.prepared_dir != args.run_dir/'prepared' or args.checkpoint_dir != args.run_dir/'checkpoints':
        raise ValueError('Explicit prepared/adapter/checkpoint paths must reside in run directory')
    if args.sample_rng_label_stride < 1 or args.sample_oversample_min < 1 or args.sample_oversample_divisor < 1:
        raise ValueError('Invalid deterministic sampling controls')
    if args.max_overlap_examples < 1 or args.save_total_limit < 0 or args.eval_accumulation_steps < 1:
        raise ValueError('Invalid overlap/save/evaluation control')
    if args.model_family not in ('llama', 'starcoder2') or args.expected_model_type != args.model_family:
        raise ValueError('Unsupported or inconsistent model family; extend variant layer explicitly')
    if args.sample_policy != 'balanced':
        raise ValueError('F-01b M1 requires balanced class sampling')
    if args.max_train_examples < 2 or args.max_val_examples < 2:
        raise ValueError('Training and validation counts must be >= 2')
    if args.max_train_examples % 2 or args.max_val_examples % 2:
        raise ValueError('Training and validation counts must be even')
    if args.max_seq_length < 256:
        raise ValueError('max_seq_length must be >= 256')
    if args.per_device_batch_size < 1 or args.per_device_eval_batch_size < 1:
        raise ValueError('Per-device batch sizes must be positive')
    if args.gradient_accumulation_steps < 1 or args.expected_world_size < 1:
        raise ValueError('DDP parameters must be positive')
    batch = args.per_device_batch_size * args.gradient_accumulation_steps * args.expected_world_size
    if args.max_steps < 1 or args.eval_steps < 1 or args.max_steps % args.eval_steps:
        raise ValueError('max_steps must be a positive multiple of eval_steps')
    if not 0 <= args.warmup_steps < args.max_steps:
        raise ValueError('Warmup must be lower than max_steps')
    if args.purpose == 'train' and args.max_train_examples != args.max_steps * batch:
        raise ValueError('Full M1 training pairs must equal global_batch * max_steps')
    if args.lora_rank < 1 or args.lora_alpha < 1 or not 0 <= args.lora_dropout < 1:
        raise ValueError('Invalid LoRA settings')
    if args.learning_rate <= 0 or args.min_vram_gib <= 0 or args.max_grad_norm < 0 or args.label_smoothing_factor < 0:
        raise ValueError('Invalid LR or VRAM requirement')
    if not args.target_modules or len(set(args.target_modules.split(','))) != len(args.target_modules.split(',')):
        raise ValueError('LoRA target modules missing or repeated')
    if args.expected_yes_token != 'derive' and not args.expected_yes_token.isdecimal():
        raise ValueError('YES expected token must be derive or decimal')
    if args.expected_no_token != 'derive' and not args.expected_no_token.isdecimal():
        raise ValueError('NO expected token must be derive or decimal')
    if args.token_policy == 'contextual-one-token' and (
        args.expected_yes_token == 'derive' or args.expected_no_token == 'derive'):
        raise ValueError('Contextual policy requires pinned YES/NO Token IDs in Shell')
    if args.verify_holdout_overlap not in ('strict', 'off'):
        raise ValueError('Unknown held-out overlap policy')
    if args.purpose not in ('preflight', 'prepare', 'smoke', 'train'):
        raise ValueError('Unknown execution purpose')
    if args.log_steps < 1:
        raise ValueError('Log steps must be positive')


def quota(n, available):
    if min(available[0], available[1]) < n // 2:
        raise RuntimeError(f'Insufficient class-balanced examples for {n}: {available}')
    return {0: n // 2, 1: n // 2}


def sampled_candidates(path, codes, available, desired, seed, args):
    rng = {y: random.Random(seed + args.sample_rng_label_stride * y) for y in (0, 1)}
    candidates = {0: [], 1: []}
    seen = Counter()
    capacities = {y: min(available[y], desired[y] + max(args.sample_oversample_min, desired[y] // args.sample_oversample_divisor)) for y in (0, 1)}
    for example in pair_rows(path, codes):
        y = example[2]
        seen[y] += 1
        chosen = candidates[y]
        if len(chosen) < capacities[y]:
            chosen.append(example)
        else:
            idx = rng[y].randrange(seen[y])
            if idx < capacities[y]:
                chosen[idx] = example
    if seen != available:
        raise RuntimeError('Input source pair file drift during sampling')
    for label in (0, 1):
        rng[label].shuffle(candidates[label])
    return candidates


def select_split(args, split, file, codes, available, tokenizer, answer_ids):
    train = split == 'train'
    n = args.max_train_examples if train else args.max_val_examples
    seed = args.seed if train else args.seed + args.sample_val_seed_offset
    needed = quota(n, available)
    candidates = sampled_candidates(file, codes, available, needed, seed, args)
    selected, lengths, overlength = [], [], Counter()
    accepted = Counter()
    for y in (0, 1):
        for a, b, label in candidates[y]:
            text = prompt(args, codes[a], codes[b])
            token = checked_training_tokens(tokenizer, text,
                args.answer_yes if label == 1 else args.answer_no,
                answer_ids[label], args.max_seq_length)
            if token is None:
                overlength[label] += 1
                continue
            selected.append({'id1': a, 'id2': b, 'label': label, 'tokens': len(token['input_ids'])})
            lengths.append(len(token['input_ids']))
            accepted[y] += 1
            if accepted[y] == needed[y]:
                break
        else:
            raise RuntimeError(f'Insufficient token-eligible {split} label={y}, '
                               f'need={needed[y]}; no silent replacement/truncation')
    random.Random(seed + args.sample_order_seed_offset).shuffle(selected)
    meta = {'count': len(selected), 'labels': dict(Counter(x['label'] for x in selected)),
            'overlength_skipped': dict(overlength), 'mean_tokens': round(sum(lengths)/len(lengths), 3),
            'min_tokens': min(lengths), 'max_tokens': max(lengths),
            'pair_ids_sha256': canonical_hash([(v['id1'], v['id2'], v['label']) for v in selected])}
    return selected, meta


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


def verify_no_heldout_overlap(args, selected, codes):
    if args.verify_holdout_overlap == 'off':
        return {'status': 'SKIPPED_EXPLICITLY', 'reason': 'Shell policy=off'}
    heldout, files = read_eval_fingerprints(args)
    overlap = []
    for split, rows in selected.items():
        for item in rows:
            for key in ('id1', 'id2'):
                if code_hash(codes[item[key]]) in heldout:
                    overlap.append({'split': split, 'id': item[key]})
                    if len(overlap) >= args.max_overlap_examples:
                        break
            if len(overlap) >= args.max_overlap_examples:
                break
    if overlap:
        raise RuntimeError(f'BLOCKED: M1 selected samples overlap held-out function codes: {overlap}')
    return {'status': 'PASS', 'files': files, 'overlap_count': 0,
            'normalization': 'SHA256 after removing all whitespace'}


def validate_model_and_data(args):
    validate_args(args)
    source, codes, counts = validate_source(args)
    model, tokenizer = cached_model(args)
    sig = source_signature(args, source, model)
    return source, codes, counts, model, tokenizer, sig


def run_preflight(args):
    source, _, _, model, _, sig = validate_model_and_data(args)
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    report = {
        'version': VERSION, 'status': 'F01B_V1_PREFLIGHT_PASS_NOT_TRAINED',
        'created_at_chicago': chicago_now(), 'configuration': configuration(args),
        'source': source, 'model': model, 'compatibility_signature': sig,
        'implementation_sha256': implementation_hashes(args),
        'limitations': ['Does not train or prove BF16 DDP memory feasibility',
                        'Only an example Java prompt is tokenized in preflight',
                        'Full Java pair tokenization and heldout leakage checked in prepare',
                        'Cached shard existence is not whole-weights checksum verification']}
    exclusive_json(run / 'preflight.json', report)
    print(json.dumps({'status': report['status'], 'run': str(run),
                      'compatibility_signature': sig}), flush=True)


def require_matching_smoke(args, sig):
    home = args.root / 'results/f-01b' / args.model_mode
    candidates = sorted(home.glob('*-smoke/smoke_success.json'))
    for file in reversed(candidates):
        try:
            data = read_json(file)
            if (data.get('status') == 'F01B_V1_SMOKE_PASS' and
                    data.get('compatibility_signature') == sig and
                    data.get('model_id') == args.model_id and
                    data.get('model_revision') == args.model_revision):
                return str(file.resolve())
        except (OSError, ValueError):
            continue
    raise RuntimeError('No compatible F-01b v1 DDP smoke proof found; run --smoke first')


def run_prepare(args):
    source, codes, counts, model, tokenizer, sig = validate_model_and_data(args)
    smoke = require_matching_smoke(args, sig) if args.purpose == 'train' else None
    selected, stats = {}, {}
    for split, filename in (('train', 'train.txt'), ('validation', 'valid.txt')):
        rows, meta = select_split(args, split, source['source_paths'][filename], codes,
                                  counts[split], tokenizer, model['answer_ids'])
        selected[split], stats[split] = rows, meta
        print(f'[F-01b v1] SELECTED {split} count={meta["count"]} '
              f'labels={meta["labels"]} max_tokens={meta["max_tokens"]}', flush=True)
    if stats['train']['count'] != args.max_train_examples or stats['validation']['count'] != args.max_val_examples:
        raise RuntimeError('Prepared pair budget mismatch')
    overlap = verify_no_heldout_overlap(args, selected, codes)
    train_ids = {x[k] for x in selected['train'] for k in ('id1', 'id2')}
    val_ids = {x[k] for x in selected['validation'] for k in ('id1', 'id2')}
    split_overlap = len(train_ids & val_ids)
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    prepared = args.prepared_dir
    prepared.mkdir(exist_ok=False)
    for split, rows in selected.items():
        dest = prepared / f'{split}.jsonl'
        with dest.open('x', encoding='utf-8') as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        stats[split]['sha256'] = sha256(dest)
    manifest = {
        'version': VERSION, 'status': 'PREPARED_NOT_TRAINED',
        'created_at_chicago': chicago_now(), 'task': 'M1 source-only LoRA',
        'model': model, 'source': source, 'configuration': configuration(args),
        'sample': stats, 'answer_ids': {str(k): v for k, v in model['answer_ids'].items()},
        'compatibility_signature': sig, 'implementation_sha256': implementation_hashes(args),
        'heldout_overlap': overlap,
        'train_validation_function_id_overlap': split_overlap,
        'validated_smoke_manifest': smoke,
        'limitations': [
            'Follows A-11 source-only sampling, including possible train/validation function ID overlap',
            'Same pair budget does not guarantee identical selected Pair IDs between model tokenizers',
            'CodeLlama contextual policy is validated for each selected Java training/validation prompt',
            'Does not adjudicate near-duplicate code or project-level independence',
            'No held-out labels accessed by optimizer or data selection',
        ],
    }
    exclusive_json(prepared / 'manifest.json', manifest)
    print(json.dumps({'status': 'F01B_V1_PREPARED_NOT_TRAINED',
                      'run': str(run), 'train': stats['train']['count'],
                      'validation': stats['validation']['count'],
                      'pair_ids_sha256': stats['train']['pair_ids_sha256'],
                      'manifest_sha256': sha256(prepared / 'manifest.json')}), flush=True)


class PairDataset:
    def __init__(self, prepared_file, codes, tokenizer, answers, args):
        self.items = [json.loads(s) for s in Path(prepared_file).read_text(encoding='utf-8').splitlines() if s.strip()]
        self.codes, self.tokenizer, self.answers, self.args = codes, tokenizer, answers, args
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        if index not in self.cache:
            row = self.items[index]
            y = row['label']
            text = prompt(self.args, self.codes[row['id1']], self.codes[row['id2']])
            data = checked_training_tokens(self.tokenizer, text,
                       self.args.answer_yes if y else self.args.answer_no,
                       self.answers[y], self.args.max_seq_length)
            if data is None or len(data['input_ids']) != row['tokens']:
                raise RuntimeError(f'Tokenization drift for prepared example {index}')
            self.cache[index] = data
        return self.cache[index]


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


def verify_prepared(args, need_gpu=False):
    run = args.run_dir.resolve()
    if not run.is_dir():
        raise FileNotFoundError(f'Run dir missing: {run}')
    prepared = args.prepared_dir
    manifest_path = prepared / 'manifest.json'
    manifest = read_json(require_file(manifest_path))
    if manifest.get('status') != 'PREPARED_NOT_TRAINED':
        raise RuntimeError('Incorrect prepared manifest status')
    config = configuration(args)
    baseline = manifest['configuration'].copy()
    if args.command == 'verify':
        # Each verification invocation has a NEW log file / Chicago timestamp;
        # these are audit metadata, not training configuration changes.
        for field in ('log_file', 'log_timestamp'):
            baseline.pop(field, None)
            config.pop(field, None)
    if baseline != config:
        changed = [k for k in sorted(set(baseline) | set(config)) if baseline.get(k) != config.get(k)]
        raise RuntimeError('Training configuration differs from prepared manifest: ' + repr(changed))
    for name, digest in manifest['source']['source_sha256'].items():
        if sha256(require_file(Path(manifest['source']['source_paths'][name]))) != digest:
            raise RuntimeError('Source data changed: ' + name)
    for name, value in [('staging_manifest', 'staging_sha256'), ('ground_truth_path', 'ground_truth_sha256')]:
        if sha256(require_file(manifest['source'][name])) != manifest['source'][value]:
            raise RuntimeError(f'Input drift: {name}')
    a10 = manifest['source']['a10']
    if sha256(require_file(a10['path'])) != a10['sha256']:
        raise RuntimeError('A-10 audit drift after preparation')
    for split in ('train', 'validation'):
        path = require_file(prepared / f'{split}.jsonl')
        if sha256(path) != manifest['sample'][split]['sha256']:
            raise RuntimeError(f'Prepared {split} drift')
    if manifest['implementation_sha256'] != implementation_hashes(args):
        raise RuntimeError('F-01b implementation drift from prepared manifest')
    if manifest['model']['revision'] != args.model_revision:
        raise RuntimeError('Pinned model revision drift')
    return manifest, manifest_path


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


def run_worker(args):
    validate_args(args)
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
    from peft import LoraConfig, TaskType, get_peft_model
    rank = int(os.environ.get('RANK', '-1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '-1'))
    world = int(os.environ.get('WORLD_SIZE', '-1'))
    if min(rank, local_rank) < 0 or world != args.expected_world_size:
        raise RuntimeError('Expected torchrun DDP environment and exact visible WORLD_SIZE')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA/BF16 unavailable')
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    manifest, manifest_path = verify_prepared(args)
    if args.run_dir.joinpath('training_summary.json').exists() or args.run_dir.joinpath('adapter').exists():
        raise FileExistsError('Refusing to overwrite completed run adapter/summary')
    cache = Path(manifest['model']['snapshot'])
    if not cache.is_dir():
        raise RuntimeError('Pinned model snapshot missing during worker launch')
    tokenizer = AutoTokenizer.from_pretrained(str(cache), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = TokenPolicy(args.token_policy, args.answer_yes, args.answer_no,
                         args.expected_yes_token, args.expected_no_token)
    answer_ids, _ = resolve_answer_ids(tokenizer,
        prompt(args, args.probe_code_a, args.probe_code_b), policy)
    if {str(k): v for k, v in answer_ids.items()} != manifest['answer_ids']:
        raise RuntimeError('Tokenizer Answer IDs drift since preparation')
    codes = load_codes(Path(manifest['source']['source_paths']['data.jsonl']))
    train_ds = PairDataset(args.prepared_dir/'train.jsonl', codes, tokenizer, answer_ids, args)
    validation_ds = PairDataset(args.prepared_dir/'validation.jsonl', codes, tokenizer, answer_ids, args)
    if len(train_ds) != args.max_train_examples or len(validation_ds) != args.max_val_examples:
        raise RuntimeError('Prepared sample count drift')
    # Explicitly verify ALL rows again before GPU allocation, not just lazy batches.
    for dataset in (train_ds, validation_ds):
        for idx in range(len(dataset)):
            dataset[idx]
    model = AutoModelForCausalLM.from_pretrained(str(cache), local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if model.config.model_type != args.expected_model_type:
        raise RuntimeError('Unexpected model family in loaded training model')
    model.config.use_cache = args.use_cache
    targets = args.target_modules.split(',')
    actual = {name.rsplit('.', 1)[-1] for name, _ in model.named_modules()}
    if not set(targets).issubset(actual):
        raise RuntimeError('LoRA targets not found: ' + repr(set(targets)-actual))
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias=args.lora_bias, target_modules=targets)
    model = get_peft_model(model, lora)
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    if not any(x.requires_grad for x in model.parameters()):
        raise RuntimeError('LoRA parameters are not trainable')
    targs = make_training_arguments(args, args.checkpoint_dir)
    tkwargs = dict(model=model, args=targs, train_dataset=train_ds,
                   eval_dataset=validation_ds,
                   data_collator=lambda batch: collate(batch, tokenizer.pad_token_id))
    tsig = inspect.signature(Trainer.__init__).parameters
    if 'processing_class' in tsig:
        tkwargs['processing_class'] = tokenizer
    elif 'tokenizer' in tsig:
        tkwargs['tokenizer'] = tokenizer
    else:
        raise RuntimeError('Trainer has neither processing_class nor tokenizer argument')
    trainer = Trainer(**tkwargs)
    if rank == 0:
        print(f'[F-01b v1] DDP world={world} GPUs per-process=1 mode={args.model_mode} '
              f'train={len(train_ds)} val={len(validation_ds)} steps={args.max_steps}', flush=True)
        model.print_trainable_parameters()
    output = trainer.train()
    if trainer.state.global_step != args.max_steps:
        raise RuntimeError(f'Trainer steps mismatch: {trainer.state.global_step} vs {args.max_steps}')
    trainer.save_model(str(args.adapter_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.adapter_dir)
        exclusive_json(args.run_dir/'training_history.json', trainer.state.log_history)
        adapter = args.adapter_dir
        required = [require_file(adapter/'adapter_model.safetensors'),
                    require_file(adapter/'adapter_config.json')]
        summary = {
            'version': VERSION, 'status': 'TRAINING_COMPLETE',
            'finished_at_chicago': chicago_now(),
            'model_mode': args.model_mode, 'model_id': args.model_id,
            'revision': args.model_revision,
            'global_step': trainer.state.global_step, 'train_loss': output.training_loss,
            'best_checkpoint': trainer.state.best_model_checkpoint,
            'best_metric': trainer.state.best_metric,
            'prepared_manifest_sha256': sha256(manifest_path),
            'compatibility_signature': manifest['compatibility_signature'],
            'effective_configuration': configuration(args),
            'trainer_arguments': targs.to_dict(),
            'adapter_sha256': {p.name: sha256(p) for p in adapter.iterdir() if p.is_file()},
            'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                         'transformers': transformers.__version__, 'peft': peft.__version__},
            'hardware': {'ddp_world': world, 'rank0_gpu': torch.cuda.get_device_name(0)},
            'limits': ['No proof of cross-model score calibration or multi-seed significance',
                       'DDP sampler actual ordered consumption not independently reconstructed'],
        }
        if len(required) != 2:
            raise RuntimeError('Missing adapter files')
        exclusive_json(args.run_dir/'training_summary.json', summary)
        print(f'[F-01b v1] TRAINING_COMPLETE adapter={adapter}', flush=True)


def run_finalize(args):
    manifest, manifest_path = verify_prepared(args)
    summary_path = require_file(args.run_dir/'training_summary.json')
    data = read_json(summary_path)
    if data.get('status') != 'TRAINING_COMPLETE' or data.get('global_step') != args.max_steps:
        raise RuntimeError('Training not complete at expected optimizer steps')
    if data.get('prepared_manifest_sha256') != sha256(manifest_path):
        raise RuntimeError('Trainer did not use this immutable prepared manifest')
    if data.get('compatibility_signature') != manifest['compatibility_signature']:
        raise RuntimeError('Compatibility signature drift')
    adapter = args.adapter_dir
    for name, expected in data.get('adapter_sha256', {}).items():
        if sha256(require_file(adapter/name)) != expected:
            raise RuntimeError('Adapter file changed since training: ' + name)
    required = {'adapter_config.json', 'adapter_model.safetensors'}
    if not required.issubset(data.get('adapter_sha256', {})):
        raise RuntimeError('Missing required adapter checksums')
    status = 'F01B_V1_SMOKE_PASS' if args.purpose == 'smoke' else 'F01B_V1_M1_TRAINING_COMPLETE'
    result = {
        'version': VERSION, 'status': status, 'created_at_chicago': chicago_now(),
        'model_mode': args.model_mode, 'model_id': args.model_id,
        'model_revision': args.model_revision,
        'compatibility_signature': manifest['compatibility_signature'],
        'run': str(args.run_dir.resolve()), 'adapter': str(adapter.resolve()),
        'prepared_manifest_sha256': sha256(manifest_path),
        'training_summary_sha256': sha256(summary_path),
        'adapter_sha256': data['adapter_sha256'],
        'train_pair_ids_sha256': manifest['sample']['train']['pair_ids_sha256'],
        'config': configuration(args),
    }
    dest = args.run_dir / ('smoke_success.json' if args.purpose == 'smoke' else 'f01b_m1_manifest.json')
    exclusive_json(dest, result)
    print(json.dumps({'status': status, 'manifest': str(dest),
                      'compatibility_signature': result['compatibility_signature']}), flush=True)


def run_verify(args):
    result_file = args.run_dir / ('smoke_success.json' if args.purpose == 'smoke' else 'f01b_m1_manifest.json')
    result = read_json(require_file(result_file))
    manifest, manifest_path = verify_prepared(args)
    if result['prepared_manifest_sha256'] != sha256(manifest_path):
        raise RuntimeError('Prepared manifest changed after finalization')
    summary = require_file(args.run_dir/'training_summary.json')
    if result['training_summary_sha256'] != sha256(summary):
        raise RuntimeError('Training summary changed')
    for name, digest in result['adapter_sha256'].items():
        if sha256(require_file(args.adapter_dir/name)) != digest:
            raise RuntimeError(f'Adapter hash mismatch: {name}')
    if result['compatibility_signature'] != manifest['compatibility_signature']:
        raise RuntimeError('Model-specific training signature mismatch')
    print(json.dumps({'status': 'F01B_V1_VERIFY_PASS', 'run': str(args.run_dir.resolve()),
                      'adapter': result['adapter']}), flush=True)
