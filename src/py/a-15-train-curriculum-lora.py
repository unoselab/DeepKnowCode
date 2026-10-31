#!/usr/bin/env python3
"""Prepare and train an auditable clone-classification adapter."""

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MODEL_ID = 'bigcode/starcoder2-7b'
REVISION = 'bb9afde76d7945da5745592525db122d4d729eb1'
CORPUS = 'java_source'
CORPORA = ('java_source', 'java_mixed', 'python_mixed')
TEST_FIELDS = (
    'train_test_function_overlap_exact', 'train_test_function_overlap_normalized',
    'validation_test_function_overlap_exact', 'validation_test_function_overlap_normalized',
    'train_test_pair_overlap_exact', 'train_test_pair_overlap_normalized',
    'validation_test_pair_overlap_normalized',
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)


def prompt(code_a, code_b, language="java"):
    # EXACTLY the A-06 prompt (with language='java').
    return ('Determine whether the following two code fragments are code clones. '
            'Answer YES if they are clones and NO otherwise.\n'
            f'Language: {language}\nCode A:\n{code_a}\n'
            f'Code B:\n{code_b}\nAnswer:')


def read_codes(path):
    codes = {}
    with path.open(encoding='utf-8-sig') as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = str(item['idx']).strip()
            body = item['func']
            if not key or key in codes or not isinstance(body, str) or not body.strip():
                raise ValueError(f'Invalid code record: {path}:{lineno}')
            codes[key] = body
    if not codes:
        raise ValueError(f'Empty code corpus: {path}')
    return codes


def pair_rows(path, known_ids=None):
    with path.open(encoding='utf-8-sig') as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 3 or parts[2] not in ('0', '1'):
                raise ValueError(f'Invalid pair record: {path}:{lineno}')
            a, b, label = parts
            if known_ids is not None and (a not in known_ids or b not in known_ids):
                raise ValueError(f'Unresolved function in {path}:{lineno}')
            yield (a, b, int(label))


def a10_audit(root, explicit):
    if explicit:
        file = explicit
    else:
        candidates = sorted((root / 'results' / 'a-10').glob('*/audit.json'),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError('No A-10 audit.json found. Run A-10 first.')
        file = candidates[0]
    audit = json.loads(file.read_text(encoding='utf-8'))
    if audit.get('status') != 'EXACT_SPLIT_CHECK_PASSED_REVIEW_PENDING':
        raise RuntimeError(f'A-10 did not pass: {file}: {audit.get("status")}')
    if audit.get('blocking_findings'):
        raise RuntimeError(f'A-10 reports blockers: {audit["blocking_findings"]}')
    if audit.get('test', {}).get('pair_rows') != 39002:
        raise RuntimeError('A-10 test-panel size differs from expected 39,002')
    source = audit.get('corpora', {}).get(CORPUS, {})
    for field in TEST_FIELDS:
        if source.get(field) != 0:
            raise RuntimeError(f'A-10 leakage check is nonzero/missing: {field}={source.get(field)}')
    return file, audit


def quotas(max_items, counts, policy):
    if max_items < 2:
        raise ValueError('At least two examples required')
    if any(counts[x] <= 0 for x in (0, 1)):
        raise ValueError(f'Both classes must be present: {counts}')
    if policy == 'balanced':
        desired = {0: max_items // 2, 1: max_items - max_items // 2}
    else:
        total = counts[0] + counts[1]
        zero = min(max_items - 1, max(1, round(max_items * counts[0] / total)))
        desired = {0: zero, 1: max_items - zero}
    return desired


def reservoir_by_label(path, count, desired, seed, codes):
    rng = {y: random.Random(seed + 991 * y) for y in (0, 1)}
    picked = {0: [], 1: []}
    seen = Counter()
    # Slight oversampling permits replacing over-length prompts without truncation.
    capacity = {y: min(count[y], desired[y] + max(64, desired[y] // 4)) for y in (0, 1)}
    for item in pair_rows(path, codes):
        y = item[2]
        seen[y] += 1
        box = picked[y]
        if len(box) < capacity[y]:
            box.append(item)
        else:
            index = rng[y].randrange(seen[y])
            if index < capacity[y]:
                box[index] = item
    if seen != count:
        raise RuntimeError(f'Pair file changed during sampling: {path}: {seen} vs {count}')
    rng[0].shuffle(picked[0])
    rng[1].shuffle(picked[1])
    return picked


def encode_answer_ids(tokenizer):
    result = {}
    for y, text in ((1, ' YES'), (0, ' NO')):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f'A-06 answer token must be single-token: {text!r} -> {ids}')
        result[y] = ids[0]
    if result[0] == result[1]:
        raise RuntimeError('YES and NO token IDs unexpectedly coincide')
    return result


def build_tokens(code_a, code_b, label, tokenizer, answer_ids, limit, language="java"):
    inputs = tokenizer.encode(prompt(code_a, code_b, language), add_special_tokens=False)
    if not inputs:
        raise RuntimeError('Tokenizer returned empty prompt')
    if len(inputs) + 1 > limit:
        return None
    return {'input_ids': inputs + [answer_ids[label]],
            'attention_mask': [1] * (len(inputs) + 1),
            'labels': [-100] * len(inputs) + [answer_ids[label]]}


def select_with_length_filter(items, desired, codes, tokenizer, answer_ids, limit):
    chosen = []
    lengths = []
    discarded = Counter()
    for label in (0, 1):
        accepted = 0
        for a, b, y in items[label]:
            data = build_tokens(codes[a], codes[b], y, tokenizer, answer_ids, limit)
            if data is None:
                discarded[label] += 1
                continue
            chosen.append({'id1': a, 'id2': b, 'label': y, 'tokens': len(data['input_ids'])})
            lengths.append(len(data['input_ids']))
            accepted += 1
            if accepted >= desired[label]:
                break
        if accepted < desired[label]:
            raise RuntimeError(f'Only {accepted}/{desired[label]} label={label} examples fit '
                               f'within {limit} tokens. Increase A15_MAX_SEQ_LENGTH or reduce sample size.')
    return chosen, lengths, dict(discarded)


def load_tokenizer(model_id, revision):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True,
                                               trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError('Tokenizer has no padding or EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def code_fingerprint(code):
    import re
    return hashlib.sha256(re.sub(r'\s+', '', code).encode('utf-8')).hexdigest()


def target_pool(pair_file, codes, source_hashes):
    # A-08 mixed corpora share source BigCloneBench records. Only pairs with
    # at least one non-source endpoint count as target-domain adaptation.
    result = {0: [], 1: []}
    fingerprints = {key: code_fingerprint(code) for key, code in codes.items()}
    for a, b, y in pair_rows(pair_file, codes):
        if fingerprints[a] not in source_hashes or fingerprints[b] not in source_hashes:
            result[y].append((a, b, y))
    return result


def selected_split(args, split, codes_by_corpus, source_hashes, tokenizer, answer_ids):
    rng = random.Random(args.seed + (0 if split == 'train' else 1234567))
    if split == 'train':
        amounts = {'java_source': args.source_train, 'java_mixed': args.java_target_train,
                   'python_mixed': args.python_target_train}
    else:
        amounts = {'java_source': args.source_val, 'java_mixed': args.java_target_val,
                   'python_mixed': args.python_target_val}
    result, stats = [], {}
    for corpus in CORPORA:
        quantity = amounts[corpus]
        if quantity < 0 or (quantity and quantity < 2):
            raise RuntimeError(f'Invalid sample size {corpus}/{split}: {quantity}')
        if not quantity:
            stats[corpus] = {'selected': 0}
            continue
        codes = codes_by_corpus[corpus]
        path = args.root / 'data/training/clean' / corpus / ('train.txt' if split == 'train' else 'valid.txt')
        if corpus == 'java_source':
            counts = Counter(y for _, _, y in pair_rows(path, codes))
            pool = reservoir_by_label(path, counts, quotas(quantity, counts, 'balanced'),
                                      args.seed + (1 if split == 'train' else 31337), codes)
        else:
            pool = target_pool(path, codes, source_hashes)
            counts = Counter({y: len(pool[y]) for y in (0, 1)})
            for y in (0, 1):
                rng.shuffle(pool[y])
        desired = quotas(quantity, counts, 'balanced')
        for y in (0, 1):
            if counts[y] < desired[y]:
                raise RuntimeError(f'Insufficient TARGET {corpus}/{split} class={y}: '
                                   f'available={counts[y]}, requested={desired[y]}. '
                                   'Set A15_*_TARGET_* to available counts; never oversample silently.')
        accepted, dropped = Counter(), Counter()
        for y in (0, 1):
            for a, b, label in pool[y]:
                tok = build_tokens(codes[a], codes[b], label, tokenizer, answer_ids,
                                   args.max_seq_length, 'python' if corpus == 'python_mixed' else 'java')
                if tok is None:
                    dropped[label] += 1
                    continue
                result.append({'corpus': corpus, 'id1': a, 'id2': b, 'label': label,
                               'tokens': len(tok['input_ids'])})
                accepted[label] += 1
                if accepted[label] == desired[label]:
                    break
            if accepted[label] < desired[label]:
                raise RuntimeError(f'Not enough <= {args.max_seq_length} tokens examples for '
                                   f'{corpus}/{split} label={label}: {accepted[label]}/{desired[label]}')
        stats[corpus] = {'selected': sum(accepted.values()), 'labels': dict(accepted),
                         'available_by_label': dict(counts), 'overlength_skipped': dict(dropped)}
        print(f'[A-15] {split}/{corpus}: available={dict(counts)} selected={dict(accepted)}', flush=True)
    rng.shuffle(result)
    return result, stats


def proxy_complexity(code_a, code_b):
    """Deterministic lexical/structural proxy, NOT paper's AST-based 5D metric."""
    def tokens(s):
        return re.findall(r'[A-Za-z_][A-Za-z_0-9]*|\S', s)
    def flow(s):
        return 1 + len(re.findall(r'\b(?:if|else|for|while|case|catch|except|elif|switch|try)\b|&&|\|\|', s))
    def depth(s):
        level = maximum = 0
        for c in s:
            if c == '{':
                level += 1; maximum = max(maximum, level)
            elif c == '}':
                level = max(level - 1, 0)
        indentation = max((len(line) - len(line.lstrip(' '))) // 4 for line in s.splitlines()) if s.splitlines() else 0
        return max(maximum, indentation)
    a, b = tokens(code_a), tokens(code_b)
    sa, sb = set(a), set(b)
    lex_distance = 1 - len(sa & sb) / max(1, len(sa | sb))
    ca, cb = flow(code_a), flow(code_b)
    flow_gap = max(ca, cb) + abs(ca-cb)
    nesting = max(depth(code_a), depth(code_b))
    length_gap = abs(len(a)-len(b)) / max(1, len(a), len(b))
    # Stable bounded score; length is a proxy, not AST/data-flow/architecture.
    return round(0.35 * min(flow_gap / 30, 1) + 0.30 * lex_distance +
                 0.20 * min(nesting / 12, 1) + 0.15 * length_gap, 9)


def curriculum_order(rows, codes_by_corpus, steps, global_batch):
    if len(rows) != steps * global_batch:
        raise RuntimeError('Expected exactly one pass at fixed global batch for ordered curriculum')
    if steps % 5 or global_batch % 2:
        raise RuntimeError('Curriculum requires max_steps multiple of 5 and even global batch')
    by_corpus = {name: {0: [], 1: []} for name in CORPORA}
    for row in rows:
        code = codes_by_corpus[row['corpus']]
        row = dict(row)
        row['complexity_proxy'] = proxy_complexity(code[row['id1']], code[row['id2']])
        by_corpus[row['corpus']][row['label']].append(row)
    for corpus in CORPORA:
        for y in (0,1):
            by_corpus[corpus][y].sort(key=lambda x:(x['complexity_proxy'],x['id1'],x['id2']))
    # Five equal phases; each has target counts [0,100,200,300,400]
    # for default 12,000 samples. Preserve exact M2 corpus totals and 1:1 labels.
    counts={k:sum(len(v) for v in by_corpus[k].values()) for k in CORPORA}
    if counts != {'java_source':11000,'java_mixed':800,'python_mixed':200} or steps != 500 or global_batch != 24:
        raise RuntimeError('A-15 v1 fixed comparison budget requires 11000/800/200, 500 steps, global_batch 24')
    targets = [0,100,200,300,400]
    schedule=[]
    ordered=[]
    phase_size=len(rows)//5
    for phase,total_target in enumerate(targets):
        quotas={'java_source':phase_size-total_target,
                'java_mixed':total_target*4//5,'python_mixed':total_target//5}
        buckets=[]
        for corpus in CORPORA:
            need=quotas[corpus]
            assert need%2 == 0
            for label in (0,1):
                take=need//2
                picked=by_corpus[corpus][label][:take]
                if len(picked)!=take: raise RuntimeError(f'Insufficient {corpus}/{label} for phase {phase}')
                del by_corpus[corpus][label][:take]
                buckets.extend(picked)
        # Within each phase distribute targets smoothly across batches,
        # while retaining easy-to-hard progression inside domain/class.
        buckets.sort(key=lambda x:(x['complexity_proxy'],x['corpus'],x['label'],x['id1']))
        # Select evenly ranked examples within each of 100 steps, preserving per-phase quotas.
        # Ordered difficulty is the primary schedule; domain mix rises between phases.
        ordered.extend(buckets)
        schedule.append({'phase':phase+1,'steps':[phase*100+1,(phase+1)*100],
                         'source':quotas['java_source'],'java_target':quotas['java_mixed'],
                         'python_target':quotas['python_mixed'],
                         'target_fraction':total_target/phase_size,
                         'proxy_min':min(r['complexity_proxy'] for r in buckets),
                         'proxy_max':max(r['complexity_proxy'] for r in buckets)})
    if any(by_corpus[c][y] for c in CORPORA for y in (0,1)):
        raise RuntimeError('Curriculum left unconsumed examples')
    assert len(ordered)==len(rows)
    return ordered,schedule


def prepare(args):
    if args.max_seq_length < 256:
        raise ValueError('max_seq_length must be >= 256')
    a10_file, a10 = a10_audit(args.root, args.a10_audit)
    a08_manifest = args.root / 'data/training/manifest.json'
    a08 = json.loads(a08_manifest.read_text(encoding='utf-8'))
    if a08.get('status') != 'STAGED_CLEANED_NOT_APPROVED_FOR_TRAINING':
        raise RuntimeError('A-08 staging manifest status mismatch')
    input_paths, codes_by_corpus, hashes = {}, {}, {}
    for corpus in CORPORA:
        paths = {name: args.root / 'data/training/clean' / corpus / name
                 for name in ('data.jsonl', 'train.txt', 'valid.txt')}
        if any(not x.is_file() for x in paths.values()):
            raise FileNotFoundError(f'Missing A-08 staged data: {paths}')
        for split, filename in (('train', 'train.txt'), ('validation', 'valid.txt')):
            expected = a10['corpora'][corpus]['train_pairs' if split == 'train' else 'validation_pairs']
            actual = sum(1 for _ in pair_rows(paths[filename]))
            if actual != expected:
                raise RuntimeError(f'A-10 count mismatch {corpus}/{split}: {actual} vs {expected}')
            for field in TEST_FIELDS:
                val = a10['corpora'][corpus].get(field)
                if val != 0:
                    raise RuntimeError(f'A-10 leakage guard {corpus}/{field}: {val}')
        input_paths[corpus] = {k: str(v.resolve()) for k, v in paths.items()}
        hashes[corpus] = {k: sha256(v) for k, v in paths.items()}
        codes_by_corpus[corpus] = read_codes(paths['data.jsonl'])
    source_hashes = {code_fingerprint(x) for x in codes_by_corpus['java_source'].values()}
    tokenizer = load_tokenizer(args.model_id, args.revision)
    answer_ids = encode_answer_ids(tokenizer)
    args.prepared_dir.mkdir(parents=True, exist_ok=True)
    reference = args.reference_prepared.resolve()
    reference_manifest_path = reference / 'manifest.json'
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding='utf-8'))
    if reference_manifest.get('task') != 'M2 static mixed-domain continuation from M1':
        raise RuntimeError('Reference is not an A-13 M2 preparation manifest')
    if reference_manifest.get('model') != {'id':args.model_id,'revision':args.revision}:
        raise RuntimeError('Reference model revision mismatch')
    sample_meta = {}
    schedule = None
    for split in ('train', 'validation'):
        refpath = reference / f'{split}.jsonl'
        if sha256(refpath) != reference_manifest['sample'][split]['sha256']:
            raise RuntimeError(f'A-13 reference sample changed: {refpath}')
        rows = [json.loads(x) for x in refpath.read_text(encoding='utf-8').splitlines() if x.strip()]
        if split == 'train':
            rows,schedule = curriculum_order(rows,codes_by_corpus,args.max_steps,
                 args.per_device_batch_size*args.gradient_accumulation_steps*args.expected_world_size)
        path = args.prepared_dir / f'{split}.jsonl'
        with path.open('w', encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row,sort_keys=True)+'\n')
        sample_meta[split]={'selected':len(rows),'by_corpus':reference_manifest['sample'][split]['by_corpus'],
                            'sha256':sha256(path),'labels':dict(Counter(r['label'] for r in rows))}
    # The ordering routine consumes all A-13 sampled rows exactly once.
    if sample_meta['train']['selected'] != 12000 or sample_meta['validation']['selected'] != 512:
        raise RuntimeError('A-13 reference sample count mismatch')
    if reference_manifest['initial_adapter_sha256'] != {x.name:sha256(x) for x in
            (args.initial_adapter.resolve()/'adapter_config.json',args.initial_adapter.resolve()/'adapter_model.safetensors')}:
        raise RuntimeError('Initial M1 adapter differs from A-13')
    adapter = args.initial_adapter.resolve()
    cfg = adapter / 'adapter_config.json'
    weights = adapter / 'adapter_model.safetensors'
    if not cfg.is_file() or not weights.is_file():
        raise FileNotFoundError(f'M1 LoRA adapter files not found at {adapter}')
    adapter_config = json.loads(cfg.read_text(encoding='utf-8'))
    if adapter_config.get('base_model_name_or_path') not in (args.model_id, None):
        raise RuntimeError('M1 Adapter base model differs from A-15 base')
    metadata = {
        'status': 'PREPARED_NOT_TRAINED', 'task': 'M3 scheduled complexity-proxy curriculum continuation from M1',
        'model': {'id': args.model_id, 'revision': args.revision},
        'a10_audit': str(a10_file.resolve()), 'a10_audit_sha256': sha256(a10_file),
        'a08_manifest': str(a08_manifest.resolve()), 'a08_manifest_sha256': sha256(a08_manifest),
        'initial_adapter': str(adapter), 'initial_adapter_sha256':
            {x.name: sha256(x) for x in (cfg, weights)},
        'data_paths': input_paths, 'data_sha256': hashes,
        'sample': sample_meta, 'curriculum_schedule': schedule,
        'reference_a13_prepared':str(reference), 'reference_a13_manifest_sha256':sha256(reference_manifest_path),
        'reference_a13_train_sha256':reference_manifest['sample']['train']['sha256'],
        'reference_a13_validation_sha256':reference_manifest['sample']['validation']['sha256'],
        'complexity_metric':'lexical/structural proxy; not paper AST 5D',
        'max_seq_length': args.max_seq_length,
        'answer_token_ids': answer_ids, 'random_seed': args.seed,
        'sampling': 'five-phase easy-to-hard proxy with increasing target proportion; same sampled pairs as A-13',
        'target_definition': 'pair has >=1 function code hash absent from A-08 java_source code corpus',
        'limitations': ['A-09 near-duplicates require independent adjudication.',
                        'Target pairs are from cleaned mixed corpora; project overlap is permitted.',
                        'A-08 validation shares some functions with training.',
                        'Targets may include heterogeneous languages; lexical corpus tag is a proxy.',
                        'Only a proxy of 5D complexity; missing AST def-use and repo paths.']}
    write_json(args.prepared_dir / 'manifest.json', metadata)
    print(f'[A-15] PREPARED {args.prepared_dir}', flush=True)


class PairDataset:
    def __init__(self, path, codes_by_corpus, tokenizer, answer_ids, limit):
        self.items = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        self.codes = codes_by_corpus
        self.tokenizer = tokenizer
        self.answer_ids = answer_ids
        self.limit = limit
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        if index not in self.cache:
            example = self.items[index]
            data = build_tokens(self.codes[example['corpus']][example['id1']],
                                self.codes[example['corpus']][example['id2']], example['label'],
                                self.tokenizer, self.answer_ids, self.limit,
                                'python' if example['corpus'] == 'python_mixed' else 'java')
            if data is None or len(data['input_ids']) != example['tokens']:
                raise RuntimeError(f'Tokenization drift or overlength in prepared example index={index}')
            self.cache[index] = data
        return self.cache[index]


def collate(batch, pad_id):
    import torch
    length = max(len(row['input_ids']) for row in batch)
    padded_ids, masks, labels = [], [], []
    for item in batch:
        n = length - len(item['input_ids'])
        padded_ids.append(item['input_ids'] + [pad_id] * n)
        masks.append(item['attention_mask'] + [0] * n)
        labels.append(item['labels'] + [-100] * n)
    return {'input_ids': torch.tensor(padded_ids, dtype=torch.long),
            'attention_mask': torch.tensor(masks, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)}


def train(args):
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    from peft import PeftModel

    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world != args.expected_world_size:
        raise RuntimeError(f'Expected DDP world_size={args.expected_world_size}, got {world}. '
                           'Use the supplied A-15 shell runner with torchrun.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('A-15 expects bf16-capable GPUs; no silent precision fallback.')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    manifest_path = args.prepared_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['status'] != 'PREPARED_NOT_TRAINED':
        raise RuntimeError('Invalid preparation manifest')
    if manifest['model'] != {'id': args.model_id, 'revision': args.revision}:
        raise RuntimeError('Model/revision changed between preparation and training')
    for pathkey, hashkey in (('a08_manifest', 'a08_manifest_sha256'),
                             ('a10_audit', 'a10_audit_sha256')):
        if sha256(Path(manifest[pathkey])) != manifest[hashkey]:
            raise RuntimeError(f'{pathkey} changed after preparation')
    for name, expected in manifest['initial_adapter_sha256'].items():
        if sha256(Path(manifest['initial_adapter']) / name) != expected:
            raise RuntimeError(f'Initial M1 adapter changed: {name}')
    if args.max_seq_length != manifest['max_seq_length']:
        raise RuntimeError('Sequence limit changed between preparation and training')
    for corpus, entries in manifest['data_sha256'].items():
        for filename, expected_hash in entries.items():
            if sha256(Path(manifest['data_paths'][corpus][filename])) != expected_hash:
                raise RuntimeError(f'Input changed after preparation: {corpus}/{filename}')
    for split in ('train', 'validation'):
        f = args.prepared_dir / f'{split}.jsonl'
        if sha256(f) != manifest['sample'][split]['sha256']:
            raise RuntimeError(f'Prepared selection changed: {f}')
    tokenizer = load_tokenizer(args.model_id, args.revision)
    answers = encode_answer_ids(tokenizer)
    if {str(k): v for k, v in answers.items()} != {str(k): v for k, v in manifest['answer_token_ids'].items()}:
        raise RuntimeError('Answer token IDs changed since preparation')
    codes = {corpus: read_codes(Path(paths['data.jsonl']))
             for corpus, paths in manifest['data_paths'].items()}
    train_dataset = PairDataset(args.prepared_dir / 'train.jsonl', codes, tokenizer, answers, args.max_seq_length)
    val_dataset = PairDataset(args.prepared_dir / 'validation.jsonl', codes, tokenizer, answers, args.max_seq_length)
    if rank == 0:
        print(f'[A-15] DDP ranks={world}, GPU={torch.cuda.get_device_name(local_rank)}, '
              f'train={len(train_dataset)}, val={len(val_dataset)}', flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.revision, local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, manifest['initial_adapter'],
                                      is_trainable=True, local_files_only=True)
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('M1 adapter did not load as trainable')
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    if rank == 0:
        model.print_trainable_parameters()
    checkpoints = args.run_dir / 'checkpoints'
    kwargs = dict(
        output_dir=str(checkpoints), seed=args.seed, data_seed=args.seed,
        max_steps=args.max_steps, per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, lr_scheduler_type='cosine', warmup_steps=args.warmup_steps,
        weight_decay=0.0, bf16=True, fp16=False, optim='adamw_torch',
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=False, dataloader_num_workers=0,
        logging_steps=args.log_steps, logging_first_step=True, report_to='none',
        save_strategy='steps', save_steps=args.eval_steps, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        remove_unused_columns=False, save_safetensors=True, disable_tqdm=False,
        eval_accumulation_steps=1,
    )
    ta_sig = inspect.signature(TrainingArguments.__init__).parameters
    if 'eval_strategy' in ta_sig:
        kwargs['eval_strategy'] = 'steps'
    elif 'evaluation_strategy' in ta_sig:
        kwargs['evaluation_strategy'] = 'steps'
    else:
        raise RuntimeError('Transformers TrainingArguments has no evaluation strategy argument')
    kwargs['eval_steps'] = args.eval_steps
    kwargs = {key: value for key, value in kwargs.items() if key in ta_sig}
    required = {'max_steps', 'gradient_accumulation_steps', 'bf16', 'gradient_checkpointing',
                'load_best_model_at_end', 'metric_for_best_model', 'save_steps', 'eval_steps'}
    if not required.issubset(kwargs):
        raise RuntimeError(f'Installed Transformers lacks required TrainingArguments: {required - set(kwargs)}')
    training_args = TrainingArguments(**kwargs)
    trainer_kwargs = dict(model=model, args=training_args, train_dataset=train_dataset,
                          eval_dataset=val_dataset,
                          data_collator=lambda batch: collate(batch, tokenizer.pad_token_id))
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if 'processing_class' in trainer_signature:
        trainer_kwargs['processing_class'] = tokenizer
    elif 'tokenizer' in trainer_signature:
        trainer_kwargs['tokenizer'] = tokenizer
    from torch.utils.data import SequentialSampler
    class CurriculumTrainer(Trainer):
        def _get_train_sampler(self, *unused_args, **unused_kwargs):
            return SequentialSampler(self.train_dataset)
    trainer = CurriculumTrainer(**trainer_kwargs)
    if rank == 0:
        print('[A-15] SequentialSampler enabled (no randomized Training sampler)', flush=True)
    if rank == 0:
        print('[A-15] Starting training: max_steps=', args.max_steps,
              'global_batch=', args.per_device_batch_size * world * args.gradient_accumulation_steps,
              'lr=', args.learning_rate, flush=True)
    outcome = trainer.train()
    trainer.save_model(str(args.run_dir / 'adapter'))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.run_dir / 'adapter')
        log_history = trainer.state.log_history
        write_json(args.run_dir / 'training_history.json', log_history)
        info = {
            'status': 'TRAINING_COMPLETE', 'finished_utc': datetime.now(timezone.utc).isoformat(),
            'step': trainer.state.global_step, 'best_checkpoint': trainer.state.best_model_checkpoint,
            'best_eval_metric': trainer.state.best_metric, 'train_loss': outcome.training_loss,
            'model_id': args.model_id, 'revision': args.revision,
            'curriculum_schedule': manifest['curriculum_schedule'],
            'reference_a13_train_sha256': manifest['reference_a13_train_sha256'],
            'initial_adapter': manifest['initial_adapter'], 'initial_adapter_sha256': manifest['initial_adapter_sha256'],
            'adapter_dir': str((args.run_dir / 'adapter').resolve()),
            'prepared_manifest_sha256': sha256(manifest_path),
            'training_config': {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))},
            'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                         'transformers': transformers.__version__, 'peft': peft.__version__},
            'hardware': {'gpus': world, 'gpu_name': torch.cuda.get_device_name(local_rank)},
            'adapter_files_sha256': {p.name: sha256(p) for p in (args.run_dir / 'adapter').iterdir() if p.is_file()},
        }
        write_json(args.run_dir / 'training_summary.json', info)
        print(f'[A-15] TRAINING_COMPLETE adapter={args.run_dir / "adapter"}', flush=True)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'train'])
    parser.add_argument('--root', type=Path, default=root)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prepared-dir', type=Path)
    parser.add_argument('--a10-audit', type=Path)
    parser.add_argument('--reference-prepared', type=Path, required=True)
    parser.add_argument('--model-id', default=MODEL_ID)
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--sample-policy', choices=['balanced', 'natural'], default='balanced')
    parser.add_argument('--source-train', type=int, default=11000)
    parser.add_argument('--java-target-train', type=int, default=800)
    parser.add_argument('--python-target-train', type=int, default=200)
    parser.add_argument('--source-val', type=int, default=412)
    parser.add_argument('--java-target-val', type=int, default=60)
    parser.add_argument('--python-target-val', type=int, default=40)
    parser.add_argument('--initial-adapter', type=Path, required=True)
    parser.add_argument('--max-seq-length', type=int, default=2048)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--log-steps', type=int, default=10)
    parser.add_argument('--warmup-steps', type=int, default=20)
    parser.add_argument('--per-device-batch-size', type=int, default=1)
    parser.add_argument('--per-device-eval-batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8)
    parser.add_argument('--expected-world-size', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--allow-custom-budget', action='store_true')
    args = parser.parse_args()
    if args.prepared_dir is None:
        args.prepared_dir = args.run_dir / 'prepared'
    if args.source_train + args.java_target_train + args.python_target_train != 12000:
        parser.error('M3 requires exactly 12,000 examples')
    if args.max_steps != 500 or args.expected_world_size != 3 or args.gradient_accumulation_steps != 8 or args.per_device_batch_size != 1:
        parser.error('A-15 comparison requires 500 steps, 3 GPUs, grad_accum=8, per_device_batch=1')
    if args.max_steps < 1 or args.eval_steps < 1 or args.max_steps % args.eval_steps != 0:
        parser.error('max_steps must be a positive multiple of eval_steps')
    if args.per_device_batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error('Batch sizes and gradient accumulation must be positive')
    if args.command == 'prepare':
        prepare(args)
    else:
        train(args)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[A-15] ERROR: {exc}', file=sys.stderr, flush=True)
        raise
