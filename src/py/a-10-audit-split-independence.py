#!/usr/bin/env python3
"""Audit clone-pair split independence without changing input data."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

CORPORA = ('java_source', 'java_mixed', 'python_mixed')


def digest(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def normalized(s):
    return digest(re.sub(r'\s+', '', s))


def pair_key(a, b):
    return (a, b) if a <= b else (b, a)


def codes(path, id_key, body_key):
    data = {}
    with path.open(encoding='utf-8-sig') as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            idx = str(r[id_key]).strip()
            if idx in data:
                raise ValueError(f'Duplicate function ID: {path}:{n}: {idx}')
            data[idx] = str(r[body_key])
    return data


def pairs(path, ids):
    exact, norm, labels, function_ids = set(), set(), Counter(), set()
    count = 0
    with path.open(encoding='utf-8-sig') as f:
        for n, line in enumerate(f, 1):
            if not line.strip(): continue
            parts = line.split()
            if len(parts) != 3 or parts[2] not in ('0', '1'):
                raise ValueError(f'Invalid pair: {path}:{n}')
            a, b, y = parts
            if a not in ids or b not in ids:
                raise ValueError(f'Unresolved ID: {path}:{n}: {a}/{b}')
            count += 1
            labels[y] += 1
            function_ids.update((a, b))
            exact.add(pair_key(ids[a][0], ids[b][0]))
            norm.add(pair_key(ids[a][1], ids[b][1]))
    return {'count': count, 'labels': dict(labels), 'functions': function_ids, 'exact_pairs': exact, 'normalized_pairs': norm}


def output_csv(path, fields, records):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def main():
    p = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    p.add_argument('--training', type=Path, default=root/'data/training')
    p.add_argument('--evaluation', type=Path, default=root/'data/evaluation')
    p.add_argument('--near-csv', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-examples', type=int, default=100)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evaluation_codes = {}
    test_exact_functions, test_norm_functions = set(), set()
    test_exact_pairs, test_norm_pairs = set(), set()
    test_pairs_count = 0
    test_pair_files = 0
    for lang in ('java', 'python'):
        originals = codes(args.evaluation/'code'/f'{lang}.jsonl', 'id', 'code')
        code_hashes = {idx:(digest(body), normalized(body)) for idx, body in originals.items()}
        evaluation_codes[lang] = code_hashes
        test_exact_functions.update(h[0] for h in code_hashes.values())
        test_norm_functions.update(h[1] for h in code_hashes.values())
        paths = sorted((args.evaluation/'ground_truth'/lang).glob('*.csv'))
        if not paths: raise ValueError(f'No test pairs found for {lang}')
        for path in paths:
            test_pair_files += 1
            with path.open(newline='', encoding='utf-8-sig') as f:
                rdr = csv.DictReader(f)
                if not {'pair_left_func', 'pair_right_func'}.issubset(rdr.fieldnames or []):
                    raise ValueError(f'Missing pair columns: {path}')
                for rowno, row in enumerate(rdr, 2):
                    a, b = row['pair_left_func'].strip(), row['pair_right_func'].strip()
                    if a not in code_hashes or b not in code_hashes:
                        raise ValueError(f'Missing test code: {path}:{rowno}')
                    test_pairs_count += 1
                    x, y = code_hashes[a], code_hashes[b]
                    test_exact_pairs.add(pair_key(x[0], y[0]))
                    test_norm_pairs.add(pair_key(x[1], y[1]))
    stats = {}
    train_hashes = set()
    rows = []
    critical = []
    for corpus in CORPORA:
        base = args.training/'clean'/corpus
        original = codes(base/'data.jsonl', 'idx', 'func')
        hashes = {idx:(digest(body), normalized(body)) for idx, body in original.items()}
        train = pairs(base/'train.txt', hashes)
        valid = pairs(base/'valid.txt', hashes)
        train_exact = {hashes[i][0] for i in train['functions']}
        valid_exact = {hashes[i][0] for i in valid['functions']}
        train_norm = {hashes[i][1] for i in train['functions']}
        valid_norm = {hashes[i][1] for i in valid['functions']}
        train_hashes.update(train_norm)
        d = {
            'train_pairs':train['count'], 'validation_pairs':valid['count'],
            'train_labels':train['labels'], 'validation_labels':valid['labels'],
            'train_referenced_function_ids':len(train['functions']),
            'validation_referenced_function_ids':len(valid['functions']),
            'train_test_function_overlap_exact':len(train_exact & test_exact_functions),
            'train_test_function_overlap_normalized':len(train_norm & test_norm_functions),
            'validation_test_function_overlap_exact':len(valid_exact & test_exact_functions),
            'validation_test_function_overlap_normalized':len(valid_norm & test_norm_functions),
            'train_test_pair_overlap_exact':len(train['exact_pairs'] & test_exact_pairs),
            'train_test_pair_overlap_normalized':len(train['normalized_pairs'] & test_norm_pairs),
            'validation_test_pair_overlap_normalized':len(valid['normalized_pairs'] & test_norm_pairs),
            'train_validation_function_overlap_exact':len(train_exact & valid_exact),
            'train_validation_function_overlap_normalized':len(train_norm & valid_norm),
            'train_validation_pair_overlap_exact':len(train['exact_pairs'] & valid['exact_pairs']),
            'train_validation_pair_overlap_normalized':len(train['normalized_pairs'] & valid['normalized_pairs']),
        }
        stats[corpus] = d
        for name, value in d.items():
            if isinstance(value, int): rows.append({'corpus':corpus, 'metric':name, 'value':value})
            if name.startswith(('train_test_', 'validation_test_')) and 'overlap' in name and value:
                critical.append(f'{corpus}/{name}={value}')
        print(f'[A-10] {corpus}: train={train["count"]} validation={valid["count"]} train/test function={d["train_test_function_overlap_normalized"]} pair={d["train_test_pair_overlap_normalized"]} train/validation shared functions={d["train_validation_function_overlap_normalized"]}', flush=True)
    near_path = args.near_csv
    if near_path is None:
        candidates = sorted((root/'results/a-09').glob('*/near_duplicate_candidates.csv'), key=lambda x:x.stat().st_mtime, reverse=True)
        near_path = candidates[0] if candidates else None
    near_stats = {'available': bool(near_path and near_path.is_file()),'source':str(near_path) if near_path else None}
    review = []
    if near_stats['available']:
        with near_path.open(newline='',encoding='utf-8-sig') as f:
            r = csv.DictReader(f)
            for row in r:
                review.append(row)
        near_stats['candidate_rows'] = len(review)
        near_stats['jaccard_at_least_0_95'] = sum(float(x.get('token_set_jaccard') or 0)>=0.95 for x in review)
        near_stats['jaccard_at_least_0_98'] = sum(float(x.get('token_set_jaccard') or 0)>=0.98 for x in review)
        print(f'[A-10] Near-duplicate REVIEW candidates={len(review)}; not automatically classified as leakage',flush=True)
    else:
        near_stats['candidate_rows'] = None
        print('[A-10] No near-duplicate CSV supplied/found; near-duplicate review incomplete',flush=True)
    output_csv(args.out/'split_metrics.csv',['corpus','metric','value'],rows)
    output_csv(args.out/'near_duplicate_review.csv',list(review[0]) if review else ['test_language','test_id','training_code_hash','token_set_jaccard'], review[:max(0,args.max_examples)])
    report = {
        'status':'CRITICAL_OVERLAP' if critical else 'EXACT_SPLIT_CHECK_PASSED_REVIEW_PENDING',
        'training_approval':'NOT_GRANTED_BY_A10',
        'evaluation_protocol':'Target-domain adaptation with held-out test examples; project overlap allowed by design',
        'test':{'pair_rows':test_pairs_count,'pair_files':test_pair_files,'unique_exact_functions':len(test_exact_functions),'unique_normalized_functions':len(test_norm_functions),'unique_exact_pairs':len(test_exact_pairs),'unique_normalized_pairs':len(test_norm_pairs)},
        'corpora':stats,'blocking_findings':critical,'near_duplicate_review':near_stats,
        'limitations':['No project-disjoint requirement: adaptation may intentionally use samples from evaluated projects.', 'Function overlap is measured by SHA-256 of exact and whitespace-stripped code, not semantic equivalence.', 'Train/validation shared functions can inflate validation accuracy even if code pairs differ.', 'Near-duplicate search is approximate; non-candidates do not prove absence of semantic leakage.', 'Input A-08 cleaned data and A-06 evaluation data are not modified.', 'Pair overlap is based on staged evaluation ground truth, not model predictions.']}
    (args.out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'[A-10] {report["status"]}; no files modified; training approval pending',flush=True)
    if critical: raise SystemExit(2)

if __name__ == '__main__': main()
