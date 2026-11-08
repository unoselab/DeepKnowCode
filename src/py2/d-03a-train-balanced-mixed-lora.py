#!/usr/bin/env python3
"""D-03 v5 standard mixed-domain BF16 LoRA M2, with target availability scenarios."""

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
# Exact C-10 promoted dataset hashes; reject stale/different Python corpus.
C10_HASHES = {
 'data.jsonl': 'da8fdd6362df692e225a5bf5b72278fd6211cef75e48e6a539dda637b76a1843',
 'train.txt': 'e8c8998f2e73a3587190904fb258ac1ccda306f0be62dec499622f4258603693',
 'valid.txt': '33c5a0506fdbbd8ff5f9604a9be1d4fba1b6b6d52943ff11ff53928861ac48a7',
}

# A copied SHA256 must always be 64 lowercase hexadecimal characters.
for _filename, _digest in C10_HASHES.items():
    if re.fullmatch(r'[0-9a-f]{64}', _digest) is None:
        raise RuntimeError(f'Invalid pinned SHA256 literal for {_filename}: {_digest!r}')

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
        print(f'[D-03] {split}/{corpus}: available={dict(counts)} selected={dict(accepted)}', flush=True)
    rng.shuffle(result)
    return result, stats


def candidate_statistics(root, codes_by_corpus, source_hashes):
    """Count class-labelled target candidates after source-only fingerprint exclusion."""
    report = {}
    for corpus in CORPORA:
        code = codes_by_corpus[corpus]
        path = root / 'data/training/clean' / corpus / 'train.txt'
        if corpus == 'java_source':
            pool = {0: [], 1: []}
            for row in pair_rows(path, code):
                pool[row[2]].append(row)
        else:
            pool = target_pool(path, code, source_hashes)
        report[corpus] = {'class_0': len(pool[0]), 'class_1': len(pool[1]), 'total':len(pool[0])+len(pool[1])}
    return report


def budget_scenarios(counts, fixed_source=12000, fixed_total=12000):
    scenarios=[]
    for target in (1000,2000,2500,3000,3500,3600,3700,3756,3760):
        options={
          'fixed_total_12000': {'java_source':fixed_total-2*target,'java_target':target,'python_target':target},
          'fixed_bcb_12000': {'java_source':fixed_source,'java_target':target,'python_target':target},
        }
        for policy, quota in options.items():
            need={'java_source':quota['java_source'],'java_mixed':target,'python_mixed':target}
            feasible=all(counts[k]['class_0']>=n//2 and counts[k]['class_1']>=n//2 for k,n in need.items())
            scenarios.append({'target_per_language':target,'policy':policy,'requested':quota,
                              'candidate_count_feasible_before_token_filter':feasible,
                              'divisible_by_default_global_batch_24':sum(quota.values())%24==0,
                              'default_schedule_feasible_before_token_filter':feasible and sum(quota.values())%24==0})
    return scenarios



def max_token_eligibility(args, codes, source_hashes, counts, hashes):
    """Exact class-balanced maxima after applying the real model tokenizer.

    Does not sample, train, alter dataset content, or certify project provenance.
    """
    tokenizer = load_tokenizer(args.model_id, args.revision)
    answer_ids = encode_answer_ids(tokenizer)
    report = {
        'task': 'D-03 M2 exact token-filtered target capacity audit',
        'source_train_pairs_fixed': args.fixed_source,
        'max_seq_length': args.max_seq_length,
        'label_policy': 'one positive clone per one negative candidate',
        'global_batch': args.per_device_batch_size * args.gradient_accumulation_steps * args.expected_world_size,
        'data_sha256': hashes,
        'domains': {},
        'warning': 'Project-level provenance, near-duplicate independence and unadjudicated negative labels are NOT resolved.'
    }
    for corpus in ('java_mixed', 'python_mixed'):
        codes_by_id = codes[corpus]
        pairs = target_pool(args.root/'data/training/clean'/corpus/'train.txt', codes_by_id, source_hashes)
        accepted = {0:0, 1:0}
        overlength = {0:0, 1:0}
        for label in (0,1):
            for a,b,_ in pairs[label]:
                item = build_tokens(codes_by_id[a],codes_by_id[b],label,tokenizer,answer_ids,
                                    args.max_seq_length,'python' if corpus=='python_mixed' else 'java')
                if item is None:
                    overlength[label] += 1
                else:
                    accepted[label] += 1
            print(f'[D-03 v5] MAX_AUDIT corpus={corpus} class={label} eligible={accepted[label]} overlength={overlength[label]}',flush=True)
        max_balanced = 2*min(accepted.values())
        report['domains'][corpus] = {
            'candidate_before_token_filter': counts[corpus],
            'after_token_filter': {'class_0':accepted[0], 'class_1':accepted[1], 'total':sum(accepted.values())},
            'overlength_rejected': {'class_0':overlength[0], 'class_1':overlength[1]},
            'max_balanced_pairs': max_balanced,
        }
    common_max = min(report['domains'][c]['max_balanced_pairs'] for c in report['domains'])
    batch = report['global_batch']
    # With 12,000 BCB examples (divisible by batch 24), a 2*T addition
    # needs 2*T divisible by batch. Other global batches are handled too.
    import math
    stride = batch // math.gcd(2,batch)
    recommended = (common_max//stride)*stride
    report['max_equal_target_per_language_before_training_leakage_audit'] = common_max
    report['max_equal_target_per_language_batch_aligned'] = recommended
    report['training_plan_if_selected'] = {
        'bcb_pairs': args.fixed_source,
        'java_target_pairs': recommended,
        'python_target_pairs': recommended,
        'total_pairs': args.fixed_source + 2*recommended,
        'global_batch': batch,
        'steps_if_exactly_one_pass': (args.fixed_source + 2*recommended)//batch
    }
    report['scenarios'] = [
        {'target_per_language': n, 'token_capacity_feasible': n<=common_max,
         'batch_aligned_with_fixed_source': (args.fixed_source + 2*n)%batch == 0}
        for n in (1000,2000,2500,3000,3500,3600,3700,3756,3760)
    ]
    write_json(args.run_dir/'max_availability.json',report)
    print('[D-03 v5] EXACT_TOKEN_MAX_AUDIT '+json.dumps({
        'java_max':report['domains']['java_mixed']['max_balanced_pairs'],
        'python_max':report['domains']['python_mixed']['max_balanced_pairs'],
        'common_max':common_max,'recommended_batch_aligned':recommended,
        'total_pairs':report['training_plan_if_selected']['total_pairs'],
        'steps':report['training_plan_if_selected']['steps_if_exactly_one_pass'],
        'report':str(args.run_dir/'max_availability.json')},sort_keys=True),flush=True)
    return report


def prepare(args):
    if args.max_seq_length < 256:
        raise ValueError('max_seq_length must be >=256')
    paths_by_corpus={}
    codes={}
    hashes={}
    for c in CORPORA:
        d=args.root/'data/training/clean'/c
        paths={k:d/k for k in ('data.jsonl','train.txt','valid.txt')}
        for k,p in paths.items():
            if not p.is_file():
                raise FileNotFoundError(f'Missing {c}/{k}: {p}')
        paths_by_corpus[c]={k:str(v.resolve()) for k,v in paths.items()}
        hashes[c]={k:sha256(v) for k,v in paths.items()}
        if c == 'python_mixed':
            # Compare by filename, not by an opaque whole-dictionary equality.
            # Print actual and expected hashes so stale scripts vs. changed data
            # can be distinguished without disabling the approved-dataset gate.
            mismatches = {}
            for filename, expected in C10_HASHES.items():
                actual = hashes[c].get(filename)
                if actual != expected:
                    mismatches[filename] = {'expected': expected, 'actual': actual}
            if mismatches:
                raise RuntimeError(
                    'C-10 approved Python corpus SHA256 mismatch: '
                    + json.dumps(mismatches, sort_keys=True)
                    + '; D-03a v6 refuses to substitute an unapproved dataset.'
                )
            print('[D-03a v6] C-10 approved Python dataset SHA256 VERIFIED '
                  '(data.jsonl, train.txt, valid.txt)', flush=True)
        codes[c]=read_codes(paths['data.jsonl'])
    source_hashes={code_fingerprint(x) for x in codes['java_source'].values()}
    counts=candidate_statistics(args.root,codes,source_hashes)
    scenario=budget_scenarios(counts, fixed_source=args.fixed_source, fixed_total=args.fixed_total)
    audit={'task':'D-03 new M2 target pair availability','counts_before_token_filter':counts,
           'scenarios':scenario,'warning':'Java/Python per-project provenance NOT verified; counts are not token-filtered.',
           'python_dataset':'data/training/clean/python_mixed (C-10 only)',
           'data_sha256':hashes}
    if args.command=='audit-max':
        write_json(args.run_dir/'availability.json',audit)
        max_token_eligibility(args,codes,source_hashes,counts,hashes)
        return
    if args.command=='audit':
        write_json(args.run_dir/'availability.json',audit)
        print(json.dumps(audit,indent=2),flush=True)
        return
    budget=args.fixed_total if args.budget_policy=='fixed-total' else args.fixed_source + 2*args.target_per_language
    source_train=budget-2*args.target_per_language
    if source_train<=0 or source_train%2 or args.target_per_language%2:
        raise ValueError('Budget must yield even positive samples in all domains')
    if args.budget_policy=='fixed-total' and source_train<0:
        raise ValueError('Fixed total too small for targets')
    args.source_train=source_train
    args.java_target_train=args.python_target_train=args.target_per_language
    wanted={'java_source':source_train,'java_mixed':args.target_per_language,'python_mixed':args.target_per_language}
    for c,n in wanted.items():
        if min(counts[c]['class_0'],counts[c]['class_1']) < n//2:
            write_json(args.run_dir/'availability.json',audit)
            raise RuntimeError(f'NOT ENOUGH labelled candidates in {c}: need {n//2} each class; actual={counts[c]}. See availability.json')
    adapter=args.initial_adapter.resolve()
    files=[adapter/'adapter_config.json',adapter/'adapter_model.safetensors']
    if any(not x.is_file() for x in files):
        raise FileNotFoundError('M1 adapter missing: '+str(adapter))
    config=json.loads(files[0].read_text(encoding='utf-8'))
    if config.get('base_model_name_or_path') not in (None,args.model_id):
        raise RuntimeError('Initial M1 adapter base-model mismatch')
    tokenizer=load_tokenizer(args.model_id,args.revision)
    answer_ids=encode_answer_ids(tokenizer)
    selected={}
    meta={}
    # The TRAIN split is sampled exactly as in D-03 v5. For validation,
    # reject any pair whose endpoint function ID appeared in TRAIN (within
    # the same corpus). Never remove or replace training examples.
    training_endpoint_ids = {}
    for split in ('train','validation'):
        # A-13 validation convention: 412/60/40, independent of the training budget.
        if split=='train':
            nums=wanted
        else:
            nums={'java_source':args.source_val,'java_mixed':args.java_target_val,'python_mixed':args.python_target_val}
        rng=random.Random(args.seed+(0 if split=='train' else 1234567))
        allrows=[]
        bydomain={}
        for c in CORPORA:
            n=nums[c]
            if n==0:
                bydomain[c]={'selected':0};continue
            if n%2:
                raise ValueError(f'Balanced quota for {c}/{split} must be even')
            src=Path(paths_by_corpus[c]['train.txt' if split=='train' else 'valid.txt'])
            if c=='java_source':
                available=Counter(y for _,_,y in pair_rows(src,codes[c]))
                pool=reservoir_by_label(src,available,{0:n//2,1:n//2},args.seed+(1 if split=='train' else 31337),codes[c])
            else:
                pool=target_pool(src,codes[c],source_hashes)
                available=Counter({y:len(pool[y]) for y in (0,1)})
                for y in (0,1):rng.shuffle(pool[y])
            for y in (0,1):
                if available[y]<n//2:raise RuntimeError(f'{c}/{split} class {y} has {available[y]} < {n//2}')
            acc=Counter();drop=Counter();disjoint_rejections=Counter()
            disallowed = training_endpoint_ids.get(c, set()) if split == 'validation' else set()
            for y in (0,1):
                for left,right,_ in pool[y]:
                    if left in disallowed or right in disallowed:
                        disjoint_rejections[y] += 1
                        continue
                    token=build_tokens(codes[c][left],codes[c][right],y,tokenizer,answer_ids,
                        args.max_seq_length,'python' if c=='python_mixed' else 'java')
                    if token is None:
                        drop[y]+=1;continue
                    allrows.append({'corpus':c,'id1':left,'id2':right,'label':y,'tokens':len(token['input_ids'])})
                    acc[y]+=1
                    if acc[y]==n//2:break
                if acc[y]<n//2:raise RuntimeError(f'{c}/{split} class {y} TOKEN-FILTERED accepted {acc[y]} < {n//2}')
            bydomain[c]={'selected':sum(acc.values()),'class_counts':dict(acc),
                         'eligible_before_token_filter':dict(available),
                         'overlength_skipped':dict(drop),
                         'validation_train_function_rejections':dict(disjoint_rejections)}
        if split=='train':
            rng.shuffle(allrows)  # static standard mixing; M3 will reuse same rows
            training_endpoint_ids = {corpus: set() for corpus in CORPORA}
            for row in allrows:
                training_endpoint_ids[row['corpus']].update((row['id1'], row['id2']))
        else:
            overlap = {}
            for corpus in CORPORA:
                matched = training_endpoint_ids[corpus] & {
                    endpoint for row in allrows if row['corpus'] == corpus
                    for endpoint in (row['id1'], row['id2'])
                }
                overlap[corpus] = len(matched)
            if any(overlap.values()):
                raise RuntimeError('BLOCKED: TRAIN/VALIDATION function-ID overlap: ' + str(overlap))
            print('[D-03a v6] VALIDATION_FUNCTION_DISJOINT ' + json.dumps(overlap,sort_keys=True),flush=True)
        selected[split]=allrows
        meta[split]={'count':len(allrows),'by_corpus':bydomain}
    # Hard stop on exact/whitespace-normalized held-out code overlap.
    evaluation_dir=args.root/'data/evaluation/code'
    heldout=set()
    for lang in ('java','python'):
        ep=evaluation_dir/f'{lang}.jsonl'
        if not ep.is_file():
            raise FileNotFoundError(f'Required held-out code corpus missing: {ep}')
        with ep.open(encoding='utf-8-sig') as f:
            for ln,line in enumerate(f,1):
                if not line.strip():continue
                obj=json.loads(line)
                body=obj.get('code')
                if not isinstance(body,str):
                    raise ValueError(f'Invalid evaluation code {ep}:{ln}')
                heldout.add(code_fingerprint(body))
    leaking=[]
    for split,rows in selected.items():
        for row in rows:
            c=codes[row['corpus']]
            for key in ('id1','id2'):
                if code_fingerprint(c[row[key]]) in heldout:
                    leaking.append((split,row['corpus'],row[key]))
                    if len(leaking)>=8:break
            if len(leaking)>=8:break
    if leaking:
        raise RuntimeError(f'BLOCKED: selected training/validation functions overlap held-out evaluation codes: {leaking}')
    # Global-batch exact-once is required for clean M2/M3 comparison.
    global_batch=args.per_device_batch_size*args.gradient_accumulation_steps*args.expected_world_size
    if len(selected['train'])%global_batch:
        raise RuntimeError(f'Train examples {len(selected["train"])} must be divisible by global batch {global_batch}; choose a target quota that makes (12000 + 2*target) divisible by global batch; see --audit-max')
    args.max_steps=len(selected['train'])//global_batch
    if args.max_steps%args.eval_steps:
        # Last-step evaluation; never change training budget or silently drop samples.
        args.eval_steps=args.max_steps
    if args.warmup_steps>=args.max_steps:raise RuntimeError('Warmup steps must be less than max_steps')
    if args.prepared_dir.exists() and any(args.prepared_dir.iterdir()):
        raise FileExistsError('Prepared directory is non-empty; do not overwrite: '+str(args.prepared_dir))
    args.prepared_dir.mkdir(parents=True,exist_ok=True)
    for split,rows in selected.items():
        f=args.prepared_dir/f'{split}.jsonl'
        with f.open('x',encoding='utf-8') as w:
            for item in rows:w.write(json.dumps(item,sort_keys=True)+'\n')
        meta[split]['sha256']=sha256(f)
    manifest={'status':'PREPARED_NOT_TRAINED','task':'D-03 M2 standard mixed-domain BF16 LoRA continuation from M1',
        'model':{'id':args.model_id,'revision':args.revision},'initial_adapter':str(adapter),
        'initial_adapter_sha256':{p.name:sha256(p) for p in files},
        'data_paths':paths_by_corpus,'data_sha256':hashes,'sample':meta,
        'answer_token_ids':answer_ids,'max_seq_length':args.max_seq_length,
        'training_plan':{'budget_policy':args.budget_policy,'train_counts':wanted,'global_batch':global_batch,
                         'max_steps':args.max_steps,'eval_steps':args.eval_steps,'seed':args.seed,'heldout_normalized_code_overlap':0,'train_validation_function_id_overlap':{c:0 for c in CORPORA}},
        'limitations':['Project provenance and project-level independence NOT proven.',
                       'C-10 negative candidates include unadjudicated examples.',
                       'Exact pair/code overlap with held-out panel must be independently audited.',
                       'Train/validation function-ID disjointness is enforced within each corpus, not across differently keyed corpora.',
                       'Target definition follows A-13: at least one non-BCB endpoint by normalized hash.']}
    write_json(args.prepared_dir/'manifest.json',manifest)
    write_json(args.run_dir/'availability.json',audit)
    print('[D-03] PREPARED',args.prepared_dir,'train',len(selected['train']), 'valid',len(selected['validation']), 'max_steps',args.max_steps,flush=True)


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
    if (args.run_dir/'adapter').exists() or (args.run_dir/'training_summary.json').exists():
        raise FileExistsError('Refusing to overwrite an existing adapter or completed training result')
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
    if manifest['status'] != 'PREPARED_NOT_TRAINED' or not manifest['task'].startswith('D-03'):
        raise RuntimeError('Invalid preparation manifest')
    if manifest['model'] != {'id': args.model_id, 'revision': args.revision}:
        raise RuntimeError('Model/revision changed between preparation and training')
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
    actual_batch=args.per_device_batch_size*args.gradient_accumulation_steps*args.expected_world_size
    if actual_batch!=manifest['training_plan']['global_batch']:
        raise RuntimeError('Training batch schedule differs from prepared plan')
    args.max_steps = manifest['training_plan']['max_steps']
    args.eval_steps = manifest['training_plan']['eval_steps']
    tokenizer = load_tokenizer(args.model_id, args.revision)
    answers = encode_answer_ids(tokenizer)
    if {str(k): v for k, v in answers.items()} != {str(k): v for k, v in manifest['answer_token_ids'].items()}:
        raise RuntimeError('Answer token IDs changed since preparation')
    codes = {corpus: read_codes(Path(paths['data.jsonl']))
             for corpus, paths in manifest['data_paths'].items()}
    train_dataset = PairDataset(args.prepared_dir / 'train.jsonl', codes, tokenizer, answers, args.max_seq_length)
    val_dataset = PairDataset(args.prepared_dir / 'validation.jsonl', codes, tokenizer, answers, args.max_seq_length)
    if rank == 0:
        print(f'[D-03] DDP ranks={world}, GPU={torch.cuda.get_device_name(local_rank)}, '
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
    trainer = Trainer(**trainer_kwargs)  # standard shuffled mixing, NOT curriculum
    if rank == 0:
        print('[D-03] Starting training: max_steps=', args.max_steps,
              'global_batch=', args.per_device_batch_size * world * args.gradient_accumulation_steps,
              'lr=', args.learning_rate, flush=True)
    outcome = trainer.train()
    trainer.save_model(str(args.run_dir / 'adapter'))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.run_dir / 'adapter')
        log_history = trainer.state.log_history
        write_json(args.run_dir / 'training_history.json', log_history)
        info = {
            'status': 'TRAINING_COMPLETE', 'finished_chicago': datetime.now(__import__('zoneinfo').ZoneInfo('America/Chicago')).isoformat(),
            'step': trainer.state.global_step, 'best_checkpoint': trainer.state.best_model_checkpoint,
            'best_eval_metric': trainer.state.best_metric, 'train_loss': outcome.training_loss,
            'model_id': args.model_id, 'revision': args.revision,
            'training_plan': manifest['training_plan'],
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
        print(f'[D-03] TRAINING_COMPLETE adapter={args.run_dir / "adapter"}', flush=True)


def main():
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description='D-03 New M2: availability -> BF16 LoRA standard mixed training')
    parser.add_argument('command', choices=['audit','audit-max','prepare','train'])
    parser.add_argument('--root',type=Path,default=root)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--prepared-dir',type=Path)
    parser.add_argument('--model-id',default=MODEL_ID)
    parser.add_argument('--revision',default=REVISION)
    parser.add_argument('--initial-adapter',type=Path)
    parser.add_argument('--budget-policy',choices=['fixed-total','fixed-source'],default='fixed-source')
    parser.add_argument('--target-per-language',type=int,default=3000)
    parser.add_argument('--fixed-total',type=int,default=12000)
    parser.add_argument('--fixed-source',type=int,default=12000)
    parser.add_argument('--source-val',type=int,default=412)
    parser.add_argument('--java-target-val',type=int,default=60)
    parser.add_argument('--python-target-val',type=int,default=40)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--max-seq-length',type=int,default=2048)
    parser.add_argument('--eval-steps',type=int,default=100)
    parser.add_argument('--log-steps',type=int,default=10)
    parser.add_argument('--warmup-steps',type=int,default=20)
    parser.add_argument('--per-device-batch-size',type=int,default=1)
    parser.add_argument('--per-device-eval-batch-size',type=int,default=1)
    parser.add_argument('--gradient-accumulation-steps',type=int,default=8)
    parser.add_argument('--expected-world-size',type=int,default=3)
    parser.add_argument('--learning-rate',type=float,default=2e-4)
    parser.add_argument('--lora-rank',type=int,default=16)
    parser.add_argument('--lora-alpha',type=int,default=32)
    parser.add_argument('--lora-dropout',type=float,default=0.05)
    args=parser.parse_args()
    args.root=args.root.resolve()
    if args.target_per_language <= 0 or args.target_per_language % 2:
        parser.error('--target-per-language must be a positive even integer')
    if args.fixed_source != 12000 and args.budget_policy == 'fixed-source':
        parser.error('D-03 v5 requires exactly 12,000 BCB source examples')
    if args.prepared_dir is None:args.prepared_dir=args.run_dir/'prepared'
    if args.command=='train':
        if args.initial_adapter is None:
            args.initial_adapter=Path(json.loads((args.prepared_dir/'manifest.json').read_text())['initial_adapter'])
        train(args)
    else:
        if args.command=='prepare' and args.initial_adapter is None:
            parser.error('--initial-adapter required for prepare')
        prepare(args)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'[D-03] ERROR: {exc}',file=sys.stderr,flush=True)
        raise
