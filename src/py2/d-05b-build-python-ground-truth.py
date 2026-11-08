#!/usr/bin/env python3
"""D-05b v1: construct UNLABELED Python candidate and provenance audit; no pseudo-gold labels."""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

M = 'D05B_CANDIDATES_ONLY_GOLD_LABELS_NOT_VERIFIED'
BASE = ('python_mixed', 'python_mixed_legacy', 'python_mixed_legacy_v2')

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def normalized_code(code):
    # Whitespace-only canonicalization is NOT a semantic or near-duplicate test.
    return hashlib.sha256(''.join(code.split()).encode('utf-8')).hexdigest()

def read_jsonl(path):
    with path.open('r', encoding='utf-8-sig') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'Invalid JSON at {path}:{line_no}: {e}') from e
            if not isinstance(obj, dict):
                raise ValueError(f'Expected JSON object at {path}:{line_no}')
            yield line_no, obj

def load_corpus(path):
    mapping = {}
    duplicate_ids = set()
    for line_no, obj in read_jsonl(path):
        ident, code = obj.get('id'), obj.get('code')
        if not isinstance(ident, str) or not ident or not isinstance(code, str) or not code.strip():
            raise ValueError(f'Invalid id/code at {path}:{line_no}')
        if ident in mapping:
            duplicate_ids.add(ident)
        else:
            mapping[ident] = {'sha256': hashlib.sha256(code.encode()).hexdigest(), 'normalized_sha256': normalized_code(code)}
    if duplicate_ids:
        raise ValueError(f'Duplicate IDs in code corpus ({len(duplicate_ids)}), examples: {sorted(duplicate_ids)[:3]}')
    return mapping

def load_training_splits(root):
    split_dir = root / 'data/training/prepared/balanced_multilingual'
    files = {}
    ids = set()
    pairs = set()
    bad = 0
    counts = Counter()
    for split in ('train', 'validation'):
        path = split_dir / f'{split}.jsonl'
        if not path.is_file():
            raise FileNotFoundError(f'Missing active M2/M3 training split: {path}')
        n = 0
        for lineno, obj in read_jsonl(path):
            n += 1
            corpus = str(obj.get('corpus', ''))
            if corpus != 'python_mixed':
                continue
            a, b = obj.get('id1'), obj.get('id2')
            if not isinstance(a, (str, int)) or not isinstance(b, (str, int)):
                bad += 1
                continue
            a, b = str(a), str(b)
            ids.update((a,b))
            pairs.add(tuple(sorted((a,b))))
            counts[split] += 1
        files[split] = {'path': str(path.relative_to(root)), 'sha256': digest(path), 'rows_all_corpora': n, 'python_rows': counts[split]}
    if bad:
        raise ValueError(f'{bad} active Python training rows missing id1/id2')
    return ids, pairs, files

def inspect_other_training(root):
    report = {}
    # File hashes and structural field samples only; do not assume lineage semantics.
    for name in BASE:
        dr = root / 'data/training/clean' / name
        files = {}
        for basename in ('data.jsonl','train.txt','valid.txt','negative_candidates.csv','positive_provenance.csv','audit.json','manifest.json'):
            p = dr / basename
            if not p.is_file():
                continue
            item = {'sha256': digest(p), 'size_bytes': p.stat().st_size}
            if basename == 'data.jsonl':
                n = 0; fields = Counter(); first = None
                for _, row in read_jsonl(p):
                    n += 1; fields.update(row.keys())
                    if first is None: first = sorted(row)
                item.update({'rows': n,'field_frequency': dict(fields),'sample_keys': first})
            elif basename.endswith('.csv'):
                with p.open(newline='', encoding='utf-8-sig') as f:
                    rdr = csv.reader(f); item['columns'] = next(rdr, [])
            files[basename] = item
        report[name] = {'exists': dr.is_dir(), 'files': files}
    return report

def load_candidates(legacy_dir, corpus):
    rows = {}
    summary = []
    collisions = []
    unmatched = []
    invalid = Counter()
    for path in sorted(legacy_dir.glob('*.csv')):
        stats = Counter()
        with path.open('r', newline='', encoding='utf-8-sig') as f:
            rdr = csv.DictReader(f)
            if not {'pair_left_func','pair_right_func'}.issubset(rdr.fieldnames or []):
                raise ValueError(f'CSV missing pair IDs: {path}')
            for csv_lineno, item in enumerate(rdr, 2):
                a, b = (item.get('pair_left_func') or '').strip(), (item.get('pair_right_func') or '').strip()
                if not a or not b or a == b:
                    invalid['blank_or_self_pair'] += 1
                    continue
                key = tuple(sorted((a,b)))
                previous = rows.get(key)
                prediction = (item.get('clone_predict') or '').strip()
                if prediction not in ('0','1'):
                    invalid['legacy_prediction_not_binary'] += 1
                provenance = {'source_file': str(path), 'source_line': csv_lineno, 'legacy_prediction': prediction}
                if previous:
                    previous['provenance'].append(provenance)
                    stats['duplicate_unordered_pair'] += 1
                    if any(p['legacy_prediction'] != prediction for p in previous['provenance']):
                        collisions.append({'left_id': key[0], 'right_id': key[1]})
                    continue
                rows[key] = {'left_id': key[0], 'right_id': key[1], 'provenance': [provenance]}
                stats['unique_pairs'] += 1
                stats['missing_code_id_pairs'] += int(a not in corpus or b not in corpus)
        summary.append({'name': path.name, 'sha256': digest(path), **dict(stats)})
    if not summary:
        raise ValueError(f'No legacy CSV candidates found in {legacy_dir}')
    return rows, summary, collisions, invalid

def dump_json(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path.cwd())
    ap.add_argument('--out', type=Path, required=True, help='Unique new results directory')
    ap.add_argument('--candidate-output', type=Path, help='New ground truth/python folder; must already exist and be empty')
    args = ap.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    candidate_output = (args.candidate_output or (root / 'data/evaluation/ground_truth/python')).resolve()
    legacy = root / 'data/evaluation/ground_truth/python_legacy'
    codepath = root / 'data/evaluation/code/python.jsonl'
    if not legacy.is_dir():
        raise SystemExit(f'BLOCKED: missing legacy Python ground truth: {legacy}')
    if not codepath.is_file():
        raise SystemExit(f'BLOCKED: missing code corpus: {codepath}')
    if not candidate_output.is_dir() or any(candidate_output.iterdir()):
        raise SystemExit(f'BLOCKED: candidate target must exist and be empty: {candidate_output}')
    if out.exists():
        raise SystemExit(f'BLOCKED: results output already exists: {out}')
    if candidate_output == out or candidate_output in out.parents or out in candidate_output.parents:
        raise SystemExit('BLOCKED: candidate output and results output must be independent')
    corpus = load_corpus(codepath)
    train_ids, train_pairs, training_files = load_training_splits(root)
    other = inspect_other_training(root)
    candidates, csv_summary, conflicts, invalid = load_candidates(legacy, corpus)
    now = datetime.now(ZoneInfo('America/Chicago')).isoformat()
    counts = Counter()
    annotatable = []
    candidate_rows = []
    for (a,b), v in sorted(candidates.items()):
        known_a, known_b = corpus.get(a), corpus.get(b)
        has_missing = known_a is None or known_b is None
        same_normalized = bool(known_a and known_b and known_a['normalized_sha256'] == known_b['normalized_sha256'])
        train_id_overlap = a in train_ids or b in train_ids
        train_pair_overlap = (a,b) in train_pairs
        # Never infer 'no leakage' from ID absence; IDs can change across corpora.
        blocking = []
        if has_missing: blocking.append('CODE_ID_UNRESOLVED')
        if train_pair_overlap: blocking.append('TRAIN_VALIDATION_PAIR_ID_MATCH')
        elif train_id_overlap: blocking.append('TRAIN_VALIDATION_FUNCTION_ID_MATCH')
        if len(set(p['legacy_prediction'] for p in v['provenance'])) > 1:
            blocking.append('CONFLICTING_LEGACY_PREDICTIONS')
        if same_normalized: blocking.append('WITHIN_PAIR_IDENTICAL_NORMALIZED_CODE_REVIEW')
        counts['total_unique_candidates'] += 1
        if blocking: counts['blocked_or_needs_review'] += 1
        else: counts['eligible_for_annotation_not_for_evaluation'] += 1
        if has_missing: counts['unresolved_code_id_pairs'] += 1
        if train_pair_overlap: counts['active_train_validation_pair_id_overlap'] += 1
        if train_id_overlap: counts['active_train_validation_function_id_overlap'] += 1
        if same_normalized: counts['within_pair_identical_normalized_code'] += 1
        if not blocking:
            annotatable.append((a,b))
        candidate_rows.append({
            'pair_id': hashlib.sha256(('python\0'+a+'\0'+b).encode()).hexdigest()[:24],
            'language':'python', 'left_id':a, 'right_id':b,
            'left_code_sha256': known_a['sha256'] if known_a else None,
            'right_code_sha256': known_b['sha256'] if known_b else None,
            'legacy_predictions': sorted(set(p['legacy_prediction'] for p in v['provenance'])),
            'legacy_provenance': [{'source_file':str(Path(p['source_file']).relative_to(root)),'source_line':p['source_line']} for p in v['provenance']],
            'gold_label':None,'label_source':None,'annotation_status':'pending',
            'exact_id_leakage_flags': blocking,
            'code_hash_train_overlap_status':'NOT_ESTABLISHED',
            'near_duplicate_leakage_status':'NOT_CHECKED',
            'evaluation_eligible':False,
        })
    report = {
        'status':M,'created_at_chicago':now,
        'source_code':{'path':str(codepath.relative_to(root)), 'sha256':digest(codepath), 'unique_functions':len(corpus)},
        'legacy_source_files':csv_summary,
        'active_training_splits':training_files,
        'other_training_provenance_inventory':other,
        'counts':dict(counts),
        'invalid_legacy_rows':dict(invalid),
        'conflicting_legacy_predictions_examples':conflicts[:20],
        'annotation_queue_count':len(annotatable),
        'methodology': {
            'candidate_origin':'Unordered pair IDs from existing legacy Python CSVs, NOT independently annotated gold labels',
            'legacy_prediction_policy':'legacy clone_predict retained as provenance only; NEVER assigned to gold_label',
            'active_train_id_check':'Exact Python id1/id2 match in M2/M3 active balanced_multilingual train and validation splits',
            'limitations':[
                'Active training split ID non-overlap does not establish semantic or code-hash independence.',
                'Training code bodies are not guaranteed recoverable from split records; train-to-code SHA256 cross-check is NOT established.',
                'Legacy training variants are inventoried but their code-level overlap is NOT established.',
                'Near-duplicate leakage screening is NOT performed.',
                'Candidate pairs originate from legacy predicted pairs and may have selection bias.',
                'M0/M1 may have distinct training provenance; M1 training inputs are not fully audited here.',
                'No human labels, verified gold pairs, accuracy, F1, or model predictions are generated.',
            ],
        },
        'next_gate':'Independent gold annotation, M1-M3 full code-based leakage screen, label adjudication, freeze before M0-M3 full evaluation',
    }
    # Write to a new result directory first. Do not touch any legacy/training source.
    out.mkdir(parents=True, exist_ok=False)
    candpath = out/'candidates.jsonl'
    with candpath.open('x',encoding='utf-8') as f:
        for obj in candidate_rows: f.write(json.dumps(obj,ensure_ascii=False,sort_keys=True)+'\n')
    reportpath = out/'provenance_leakage_audit.json'
    dump_json(reportpath, report)
    reviewpath = out/'annotation_queue.csv'
    with reviewpath.open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f)
        w.writerow(('pair_id','left_id','right_id','gold_label','annotator','annotation_evidence','adjudication_status'))
        for obj in candidate_rows:
            if not obj['exact_id_leakage_flags']:
                w.writerow((obj['pair_id'],obj['left_id'],obj['right_id'],'','','','pending'))
    manifest = {
        'created_at_chicago':now, 'status':M,
        'source_files':{'code/python.jsonl':digest(codepath), **{f"legacy/{s['name']}":s['sha256'] for s in csv_summary}},
        'artifacts': {p.name:digest(p) for p in (candpath,reportpath,reviewpath)},
    }
    manifestpath = out/'manifest.json'
    dump_json(manifestpath, manifest)
    # Only create fresh target outputs with exclusive creation; never overwrite existing artifacts.
    for src in (candpath, reportpath, reviewpath, manifestpath):
        destination = candidate_output/src.name
        with destination.open('xb') as target, src.open('rb') as source:
            for chunk in iter(lambda:source.read(1024*1024), b''):
                target.write(chunk)
    print(json.dumps({'status':M,'python_code_functions':len(corpus),'candidate_pairs':len(candidate_rows),'annotation_queue':len(annotatable),'counts':dict(counts),'out':str(out),'ground_truth_target':str(candidate_output)},ensure_ascii=False))

if __name__=='__main__':
    try:
        main()
    except (ValueError,FileNotFoundError,OSError) as e:
        print(f'D05B_BLOCKED: {e}',file=sys.stderr)
        sys.exit(2)
