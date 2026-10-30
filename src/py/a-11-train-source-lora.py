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
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MODEL_ID = 'bigcode/starcoder2-7b'
REVISION = 'bb9afde76d7945da5745592525db122d4d729eb1'
CORPUS = 'java_source'
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


def prompt(code_a, code_b):
    # EXACTLY the A-06 prompt (with language='java').
    return ('Determine whether the following two code fragments are code clones. '
            'Answer YES if they are clones and NO otherwise.\n'
            f'Language: java\nCode A:\n{code_a}\n'
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


def build_tokens(code_a, code_b, label, tokenizer, answer_ids, limit):
    inputs = tokenizer.encode(prompt(code_a, code_b), add_special_tokens=False)
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
                               f'within {limit} tokens. Increase A11_MAX_SEQ_LENGTH or reduce sample size.')
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


def prepare(args):
    if args.max_seq_length < 256:
        raise ValueError('Maximum sequence length must be >= 256')
    if args.max_train_examples < 2 or args.max_val_examples < 2:
        raise ValueError('Sampling caps must be >= 2')
    root = args.root
    code_dir = root / 'data' / 'training' / 'clean' / CORPUS
    paths = {name: code_dir / name for name in ('data.jsonl', 'train.txt', 'valid.txt')}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Missing staged A-08 data: {path}')
    a10_file, a10 = a10_audit(root, args.a10_audit)
    a08_manifest = root / 'data' / 'training' / 'manifest.json'
    if not a08_manifest.is_file():
        raise FileNotFoundError(f'Missing A-08 staging manifest: {a08_manifest}')
    a08 = json.loads(a08_manifest.read_text(encoding='utf-8'))
    if a08.get('status') != 'STAGED_CLEANED_NOT_APPROVED_FOR_TRAINING':
        raise RuntimeError(f'Unexpected A-08 staging status: {a08.get("status")}')
    hashes = {key: sha256(path) for key, path in paths.items()}
    codes = read_codes(paths['data.jsonl'])
    counts = {}
    for split, filename in (('train', 'train.txt'), ('validation', 'valid.txt')):
        classes = Counter(y for _, _, y in pair_rows(paths[filename], codes))
        expected = a10['corpora'][CORPUS]['train_pairs' if split == 'train' else 'validation_pairs']
        if sum(classes.values()) != expected:
            raise RuntimeError(f'Staged {split} count differs from A-10: {sum(classes.values())} vs {expected}')
        expected_classes = a10['corpora'][CORPUS]['train_labels' if split == 'train' else 'validation_labels']
        if {str(y): classes[y] for y in (0, 1)} != {str(k): v for k, v in expected_classes.items()}:
            raise RuntimeError(f'{split} class distribution does not match A-10')
        counts[split] = classes
        print(f'[A-11] {split}: {sum(classes.values())} pairs, labels={dict(classes)}', flush=True)
    tokenizer = load_tokenizer(args.model_id, args.revision)
    answer_ids = encode_answer_ids(tokenizer)
    sample_meta = {}
    args.prepared_dir.mkdir(parents=True, exist_ok=True)
    for split, filename, cap, seed in (
            ('train', 'train.txt', args.max_train_examples, args.seed),
            ('validation', 'valid.txt', args.max_val_examples, args.seed + 1234567)):
        goal = quotas(cap, counts[split], args.sample_policy)
        items = reservoir_by_label(paths[filename], counts[split], goal, seed, codes)
        selected, lengths, too_long = select_with_length_filter(
            items, goal, codes, tokenizer, answer_ids, args.max_seq_length)
        random.Random(seed + 97813).shuffle(selected)
        path = args.prepared_dir / f'{split}.jsonl'
        with path.open('w', encoding='utf-8') as output:
            for example in selected:
                output.write(json.dumps(example, ensure_ascii=False) + '\n')
        sample_meta[split] = {
            'selected': len(selected), 'labels': dict(Counter(item['label'] for item in selected)),
            'skipped_overlength_in_candidates': too_long, 'sample_sha256': sha256(path),
            'min_tokens': min(lengths), 'max_tokens': max(lengths),
            'mean_tokens': round(sum(lengths) / len(lengths), 2),
            'parent_pair_count': sum(counts[split].values()),
        }
        print(f'[A-11] selected {split}={len(selected)} labels={sample_meta[split]["labels"]} '
              f'max_tokens={max(lengths)}', flush=True)
    metadata = {
        'status': 'PREPARED_NOT_TRAINED',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'task': 'source-only supervised LoRA next-token YES/NO classification',
        'model': {'id': args.model_id, 'revision': args.revision},
        'dataset': CORPUS, 'data_paths': {k: str(v.resolve()) for k, v in paths.items()},
        'data_sha256': hashes, 'a08_manifest': str(a08_manifest.resolve()),
        'a08_manifest_sha256': sha256(a08_manifest),
        'a10_audit': str(a10_file.resolve()), 'a10_audit_sha256': sha256(a10_file),
        'a10_status': a10['status'], 'heldout_test_pairs': a10['test']['pair_rows'],
        'sample_policy': args.sample_policy, 'random_seed': args.seed,
        'max_seq_length': args.max_seq_length, 'answer_token_ids': answer_ids,
        'sample': sample_meta,
        'limitations': [
            'Source-only baseline; does not use target-domain training or test labels.',
            'Random class-stratified subset, not a full-epoch pass over the 627,112 pairs.',
            'Overlong pairs are excluded from selected subsets, never silently truncated.',
            'A-10 permits overlapping projects; exact/normalized test-function overlap is zero.',
            'A-09 near-duplicate candidates have not been adjudicated.',
            'A-08 train/validation share some functions; checkpoint selection uses held-out pairs.',
            'YES/NO answer-token objective is aligned with A-06, whose scoring uses logits.',
        ],
    }
    write_json(args.prepared_dir / 'manifest.json', metadata)
    print(f'[A-11] PREPARED: {args.prepared_dir}', flush=True)


class PairDataset:
    def __init__(self, path, codes, tokenizer, answer_ids, limit):
        self.items = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        self.codes = codes
        self.tokenizer = tokenizer
        self.answer_ids = answer_ids
        self.limit = limit
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        if index not in self.cache:
            example = self.items[index]
            data = build_tokens(self.codes[example['id1']], self.codes[example['id2']],
                                example['label'], self.tokenizer, self.answer_ids, self.limit)
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
    from peft import LoraConfig, TaskType, get_peft_model

    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world != args.expected_world_size:
        raise RuntimeError(f'Expected DDP world_size={args.expected_world_size}, got {world}. '
                           'Use the supplied A-11 shell runner with torchrun.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('A-11 expects bf16-capable GPUs; no silent precision fallback.')
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
    if args.max_seq_length != manifest['max_seq_length']:
        raise RuntimeError('Sequence limit changed between preparation and training')
    for filename, expected_hash in manifest['data_sha256'].items():
        if sha256(Path(manifest['data_paths'][filename])) != expected_hash:
            raise RuntimeError(f'Input changed after preparation: {filename}')
    for split in ('train', 'validation'):
        f = args.prepared_dir / f'{split}.jsonl'
        if sha256(f) != manifest['sample'][split]['sample_sha256']:
            raise RuntimeError(f'Prepared selection changed: {f}')
    tokenizer = load_tokenizer(args.model_id, args.revision)
    answers = encode_answer_ids(tokenizer)
    if {str(k): v for k, v in answers.items()} != {str(k): v for k, v in manifest['answer_token_ids'].items()}:
        raise RuntimeError('Answer token IDs changed since preparation')
    codes = read_codes(Path(manifest['data_paths']['data.jsonl']))
    train_dataset = PairDataset(args.prepared_dir / 'train.jsonl', codes, tokenizer, answers, args.max_seq_length)
    val_dataset = PairDataset(args.prepared_dir / 'validation.jsonl', codes, tokenizer, answers, args.max_seq_length)
    if rank == 0:
        print(f'[A-11] DDP ranks={world}, GPU={torch.cuda.get_device_name(local_rank)}, '
              f'train={len(train_dataset)}, val={len(val_dataset)}', flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.revision, local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.config.use_cache = False
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_rank,
                      lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      bias='none', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    model = get_peft_model(model, lora)
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
    trainer = Trainer(**trainer_kwargs)
    if rank == 0:
        print('[A-11] Starting training: max_steps=', args.max_steps,
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
            'adapter_dir': str((args.run_dir / 'adapter').resolve()),
            'prepared_manifest_sha256': sha256(manifest_path),
            'training_config': {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))},
            'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                         'transformers': transformers.__version__, 'peft': peft.__version__},
            'hardware': {'gpus': world, 'gpu_name': torch.cuda.get_device_name(local_rank)},
            'adapter_files_sha256': {p.name: sha256(p) for p in (args.run_dir / 'adapter').iterdir() if p.is_file()},
        }
        write_json(args.run_dir / 'training_summary.json', info)
        print(f'[A-11] TRAINING_COMPLETE adapter={args.run_dir / "adapter"}', flush=True)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'train'])
    parser.add_argument('--root', type=Path, default=root)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prepared-dir', type=Path)
    parser.add_argument('--a10-audit', type=Path)
    parser.add_argument('--model-id', default=MODEL_ID)
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--sample-policy', choices=['balanced', 'natural'], default='balanced')
    parser.add_argument('--max-train-examples', type=int, default=12000)
    parser.add_argument('--max-val-examples', type=int, default=512)
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
    args = parser.parse_args()
    if args.prepared_dir is None:
        args.prepared_dir = args.run_dir / 'prepared'
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
        print(f'[A-11] ERROR: {exc}', file=sys.stderr, flush=True)
        raise
