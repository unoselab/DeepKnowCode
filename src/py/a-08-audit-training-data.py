#!/usr/bin/env python3
"""Audit and stage clone-pair datasets without modifying source files."""
import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path


def exact_hash(code):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def code_hash(code):
    return hashlib.sha256(re.sub(r'\s+', '', code).encode('utf-8')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def load_codes(path, field_id='idx', field_code='func'):
    codes, hashes = {}, {}
    with path.open(encoding='utf-8-sig') as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = str(record[field_id]).strip()
                body = record[field_code]
                if not key or not isinstance(body, str) or not body.strip() or key in codes:
                    raise ValueError('empty/duplicate ID or empty code')
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{path}:{n}: {exc}') from exc
            codes[key] = body
            hashes[key] = code_hash(body)
    return codes, hashes


def load_test(eval_root):
    test_hashes = set()
    stats = {}
    for language in ('java', 'python'):
        code_file = eval_root / 'code' / f'{language}.jsonl'
        _, hashes = load_codes(code_file, field_id='id', field_code='code')
        gt = list((eval_root / 'ground_truth' / language).glob('*.csv'))
        if not gt:
            raise FileNotFoundError(f'Missing held-out ground truth: {language}')
        test_hashes.update(hashes.values())
        stats[language] = {'functions': len(hashes), 'unique_code_hashes': len(set(hashes.values())), 'ground_truth_files': len(gt)}
    return test_hashes, stats


def ingest_pairs(path, hashes):
    rows = []
    issues = Counter()
    samples = []
    seen_ids = {}
    with path.open(encoding='utf-8-sig') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 3 or parts[2] not in ('0', '1'):
                issues['invalid_format'] += 1
                if len(samples) < 20: samples.append(['invalid_format', line_no, line.strip()[:120]])
                continue
            a, b, label = parts[0], parts[1], int(parts[2])
            if a not in hashes or b not in hashes:
                issues['unresolved_id'] += 1
                if len(samples) < 20: samples.append(['unresolved_id', line_no, f'{a} {b}'])
                continue
            id_key = tuple(sorted((a,b)))
            if id_key in seen_ids:
                if seen_ids[id_key] != label:
                    issues['conflicting_id_pair_label'] += 1
                    if len(samples) < 20: samples.append(['conflicting_id_pair_label', line_no, f'{a} {b}: previous={seen_ids[id_key]} current={label}'])
                else:
                    issues['duplicate_id_pair'] += 1
            else:
                seen_ids[id_key] = label
            ha, hb = hashes[a], hashes[b]
            hash_key = tuple(sorted((ha, hb)))
            rows.append((a,b,label,hash_key,line_no))
    return rows, issues, samples


def emit_clean(dest, corpus, records):
    dest.mkdir(parents=True, exist_ok=False)
    ids = set()
    for split, filename in (('train','train.txt'), ('validation','valid.txt')):
        with (dest/filename).open('w', encoding='utf-8') as f:
            for a,b,label,_,_ in records[split]:
                f.write(f'{a}\t{b}\t{label}\n')
                ids.update((a,b))
    with (dest/'data.jsonl').open('w', encoding='utf-8') as f:
        for key in sorted(ids):
            f.write(json.dumps({'idx':key,'func':corpus[key]}, ensure_ascii=False)+'\n')
    return len(ids)


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--source-root', type=Path, default=Path.home()/'research_dream/2026JournalData/RQ3')
    parser.add_argument('--eval-root', type=Path, default=root/'data/evaluation')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--destination', type=Path, default=root/'data/training')
    parser.add_argument('--stage', action='store_true')
    parser.add_argument('--strict-function-split', action='store_true', help='also disallow a function hash in both train and validation')
    parser.add_argument('--max-removal-fraction', type=float, default=0.95)
    args = parser.parse_args()
    if not 0 <= args.max_removal_fraction <= 1: parser.error('max-removal-fraction must be between 0 and 1')
    args.output.mkdir(parents=True, exist_ok=True)
    test_hashes, test_stats = load_test(args.eval_root)
    specs = {
        'java_source': (args.source_root/'java/data/org', 'train.txt', 'valid.txt'),
        'java_mixed': (args.source_root/'java/data/combined_org10_plus_repos', 'train_mix.txt', 'valid_mix.txt'),
        'python_mixed': (args.source_root/'python/data/combined_py_plus_org10', 'train_mix.txt', 'valid_mix.txt'),
    }
    datasets, overview = {}, {}
    diagnostics = []
    blockers = []
    for name,(folder,train_file,valid_file) in specs.items():
        paths = [folder/'data.jsonl',folder/train_file,folder/valid_file]
        for p in paths:
            if not p.is_file(): raise FileNotFoundError(p)
        corpus, hashes = load_codes(paths[0])
        splits = {}
        count = {}
        for split,p in (('train',paths[1]),('validation',paths[2])):
            rows, issues, samples = ingest_pairs(p,hashes)
            count[split] = {'input_lines':sum(1 for line in p.open(encoding='utf-8-sig') if line.strip()),
                            'parsed_unique_id_pairs':len(rows),'input_issues':dict(issues)}
            for issue,line_no,detail in samples:
                diagnostics.append([name,split,issue,line_no,detail])
            if issues['invalid_format'] or issues['unresolved_id']:
                blockers.append(f'{name}/{split}: invalid formats or unresolved function IDs')
            splits[split] = rows
        datasets[name] = {'corpus':corpus,'hashes':hashes,'splits':splits}
        overview[name] = {'source_files':{p.name: {'path':str(p),'sha256':file_hash(p)} for p in paths},
                          'original_functions':len(corpus), 'input':count, 'removed':{}, 'clean':{}}

    # Test code exclusion is function-level: remove ANY pair with an endpoint matching a held-out code hash.
    for name,dataset in datasets.items():
        for split,records in dataset['splits'].items():
            keep=[]; removed=Counter()
            for rec in records:
                if rec[3][0] in test_hashes or rec[3][1] in test_hashes:
                    removed['test_code_endpoint'] += 1
                else:
                    keep.append(rec)
            dataset['splits'][split]=keep
            overview[name]['removed'][split] = dict(removed)

    # Globally quarantine every normalized-code pair with incompatible labels.
    # Do not infer labels by majority voting, and do not let file-order select a label.
    label_sets = {}
    for name, dataset in datasets.items():
        for split, records in dataset['splits'].items():
            for rec in records:
                label_sets.setdefault(rec[3], set()).add(rec[2])
    conflicting_keys = {key for key, labels in label_sets.items() if len(labels) > 1}
    conflict_records = []
    exact_pair_signatures = {}
    per_corpus_conflict_counts = Counter()
    for name, dataset in datasets.items():
        for split, records in dataset['splits'].items():
            keep = []
            removed = Counter(overview[name]['removed'][split])
            for a,b,label,key,line_no in records:
                if key not in conflicting_keys:
                    keep.append((a,b,label,key,line_no))
                    continue
                removed['conflicting_code_pair_label'] += 1
                per_corpus_conflict_counts[(name,split)] += 1
                # Track whether normalized collisions are also exact-source collisions.
                codes = dataset['corpus']
                exact_key = tuple(sorted((exact_hash(codes[a]), exact_hash(codes[b]))))
                exact_pair_signatures.setdefault(key, set()).add(exact_key)
                conflict_records.append([name,split,line_no,a,b,label,key[0],key[1],exact_key[0],exact_key[1]])
            dataset['splits'][split] = keep
            overview[name]['removed'][split] = dict(removed)
    with (args.output/'conflicting_pairs.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['dataset','split','line','id1','id2','label','normalized_hash1','normalized_hash2','exact_hash1','exact_hash2'])
        writer.writerows(conflict_records)
    conflicting_id_count = sum(overview[name]['input'][split]['input_issues'].get('conflicting_id_pair_label',0)
        for name in datasets for split in ('train','validation'))
    # Conflicting ID pairs are quarantined via the code-pair conflict union.
    if conflicting_id_count:
        print(f'[A-08] Conflicting ID-pair label rows: {conflicting_id_count}', flush=True)
    print(f'[A-08] Quarantined {len(conflicting_keys)} conflicting normalized code pairs ({len(conflict_records)} rows)',flush=True)
    validation_keys = set()
    # Cross-corpus validation union: same code pair cannot be train in another corpus.
    for dataset in datasets.values():
        validation_keys.update(rec[3] for rec in dataset['splits']['validation'])
    # Dedupe inside each split, then exclude training pair keys present in any validation corpus.
    for name,dataset in datasets.items():
        for split, records in dataset['splits'].items():
            keys=set(); keep=[]; removed=Counter(overview[name]['removed'][split])
            for rec in records:
                key=rec[3]
                if key in keys:
                    removed['duplicate_code_pair_within_split']+=1
                elif split=='train' and key in validation_keys:
                    removed['global_validation_pair_conflict']+=1
                else:
                    keep.append(rec); keys.add(key)
            dataset['splits'][split]=keep
            overview[name]['removed'][split]=dict(removed)

    # Report function-level train/validation reuse but do not silently destroy a pair-level benchmark.
    global_train_hashes=set()
    global_valid_hashes=set()
    for dataset in datasets.values():
        for a,b,_,key,_ in dataset['splits']['train']: global_train_hashes.update(key)
        for a,b,_,key,_ in dataset['splits']['validation']: global_valid_hashes.update(key)
    shared_functions=global_train_hashes & global_valid_hashes
    if args.strict_function_split and shared_functions:
        blockers.append(f'{len(shared_functions)} function hashes reused between training and validation across corpora')
    # Validate no held-out matches remain, and check output size and class balance.
    for name, dataset in datasets.items():
        for split, records in dataset['splits'].items():
            labels=Counter(rec[2] for rec in records)
            n_input=overview[name]['input'][split]['input_lines']
            n_rem=n_input-len(records)
            overlap=sum(1 for rec in records if rec[3][0] in test_hashes or rec[3][1] in test_hashes)
            overview[name]['clean'][split]={'pairs':len(records),'labels':dict(sorted(labels.items())),
                'removed_total':n_rem,'removed_fraction':round(n_rem/max(n_input,1),6), 'test_overlap_pairs':overlap}
            if not records or set(labels)!={0,1}:
                blockers.append(f'{name}/{split}: empty or single-label dataset after filtering')
            if n_rem/max(n_input,1)>args.max_removal_fraction:
                blockers.append(f'{name}/{split}: removal fraction too high ({n_rem}/{n_input})')
            if overlap: blockers.append(f'{name}/{split}: test endpoint remained after filter')
        print(f"[A-08] {name}: train {overview[name]['input']['train']['input_lines']} -> {overview[name]['clean']['train']['pairs']}, validation {overview[name]['input']['validation']['input_lines']} -> {overview[name]['clean']['validation']['pairs']}",flush=True)

    # Pair overlap check across all datasets.
    train_keys = {rec[3] for dataset in datasets.values() for rec in dataset['splits']['train']}
    valid_keys = {rec[3] for dataset in datasets.values() for rec in dataset['splits']['validation']}
    pair_overlap=len(train_keys & valid_keys)
    if pair_overlap: blockers.append(f'{pair_overlap} code pairs are shared between global training and validation')
    report = {'status':'BLOCKED' if blockers else 'CLEANED_AUDIT_PASS_PROVENANCE_REVIEW_REQUIRED',
              'source_test_hash_algorithm':'SHA256(whitespace_removed_source_code)',
              'heldout_test':test_stats, 'datasets':overview,
              'conflicts':{'unique_normalized_code_pairs':len(conflicting_keys),
                  'quarantined_rows':len(conflict_records),
                  'unique_exact_pair_signatures_across_conflicts':sum(len(v) for v in exact_pair_signatures.values()),
                  'normalized_keys_with_multiple_exact_pair_signatures':sum(len(v)>1 for v in exact_pair_signatures.values()),
                  'conflicting_id_pair_label_rows':conflicting_id_count,
                  'rows_by_dataset_split':{f'{n}/{sp}':cnt for (n,sp),cnt in sorted(per_corpus_conflict_counts.items())}},
              'cross_corpus':{'global_train_validation_shared_code_pairs':pair_overlap,
                              'global_train_validation_shared_function_hashes':len(shared_functions),
                              'train_unique_code_pairs':len(train_keys),'validation_unique_code_pairs':len(valid_keys)},
              'blocking_issues':blockers,
              'limitations':['Whitespace-eliding hashes may conflate different source strings; see conflicting_pairs.csv exact hashes.',
                 'This is normalized-code leakage filtering, not a semantic/near-duplicate check.',
                 'Function overlap between training and validation is reported, not removed unless strict mode is requested.',
                 'No project provenance is established by ID/code alone; project-level isolation still requires mapping.',
                 'Original curriculum difficulty scores and dynamic source/target mixing are not recovered here.',
                 'A-06 test labels are never read for selecting training or validation records.'],
              'training_approval':'PENDING_PROJECT_AND_NEAR_DUPLICATE_REVIEW'}
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with (args.output/'audit_findings.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['dataset','split','issue','line','detail']);writer.writerows(diagnostics)
    if blockers:
        print('[A-08] BLOCKED; no staging. See audit.json',file=sys.stderr)
        return 2
    if not args.stage:
        print('[A-08] Audit passed; run with A08_STAGE=1 to copy CLEANED data. No data copied.',flush=True)
        return 0
    if args.destination.exists():
        print(f'[A-08] BLOCKED destination exists, refusing overwrite: {args.destination}',file=sys.stderr)
        return 2
    # Write to temporary sibling, then publish only after full success.
    temp=args.destination.with_name(args.destination.name+'.a08v3-partial')
    if temp.exists():
        print(f'[A-08] BLOCKED temporary destination exists: {temp}',file=sys.stderr)
        return 2
    try:
        temp.mkdir(parents=True)
        for name,dataset in datasets.items():
            n_functions=emit_clean(temp/'clean'/name,dataset['corpus'],dataset['splits'])
            overview[name]['clean']['referenced_functions']=n_functions
        report['status']='STAGED_CLEANED_NOT_APPROVED_FOR_TRAINING'
        manifest={'status':report['status'],'audit_output':str(args.output), 'audit':report,
                  'README':'Cleaned clone-pair datasets. Training approval pending project and near-duplicate review.'}
        (temp/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        # update audit with function counts and actual status
        (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        temp.rename(args.destination)
    except Exception:
        shutil.rmtree(temp,ignore_errors=True)
        raise
    print(f'[A-08] CLEANED data staged at {args.destination}; training approval pending',flush=True)
    return 0


if __name__=='__main__':
    try: sys.exit(main())
    except Exception as exc:
        print(f'[A-08] ERROR: {exc}',file=sys.stderr)
        sys.exit(2)
