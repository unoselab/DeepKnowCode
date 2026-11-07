#!/usr/bin/env python3
"""C-10 v1: non-destructive, adjudication-informed Python dataset candidate.

Excludes *only* explicitly adjudicated clones from C-04 negative labels.
Unreviewed negatives remain in the candidate with clear unverified provenance.
Never approves training or mutates C-04/C-09/legacy datasets.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v1'
PAIR_KEY = ('project', 'split', 'id_a', 'id_b')
VALID_STATUS = {'EXCLUDE_CONFIRMED_CLONE', 'REVIEWED_NONCLONE', 'UNREVIEWED', 'UNCERTAIN_REVIEW'}
NEG_COLS = ['project', 'split', 'id_a', 'id_b', 'label', 'verification', 'c09_status', 'review_id', 'label_origin', 'training_approved']


class AuditError(ValueError):
    pass


def hfile(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def csv_rows(path, required):
    if not path.is_file():
        raise AuditError(f'Missing input: {path}')
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(required) <= set(reader.fieldnames):
            raise AuditError(f'CSV schema error {path}: missing {sorted(set(required) - set(reader.fieldnames or []))}')
        return list(reader)


def pair(a, b):
    if not a or not b or a == b:
        raise AuditError(f'Invalid pair IDs: {a!r}, {b!r}')
    return tuple(sorted((a, b)))


def key(project, split, a, b):
    if not project or split not in ('train', 'valid'):
        raise AuditError(f'Bad project/split: {project!r}/{split!r}')
    return project, split, *pair(a, b)


def read_pairs(path, ids):
    if not path.is_file():
        raise AuditError(f'Missing pair file: {path}')
    records = {}
    with path.open(encoding='utf-8-sig') as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            cols = line.split()
            if len(cols) != 3 or cols[2] not in ('0', '1'):
                raise AuditError(f'Invalid 3-column pair: {path}:{lineno}')
            a, b, lab = cols
            if a not in ids or b not in ids:
                raise AuditError(f'Undefined function ID: {path}:{lineno}')
            p = pair(a, b)
            if p in records:
                raise AuditError(f'Duplicate pair within split: {path}:{lineno}: {p}')
            records[p] = int(lab)
    return records


def code_records(path):
    if not path.is_file():
        raise AuditError(f'Missing corpus: {path}')
    out = {}
    with path.open(encoding='utf-8-sig') as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                i, code = obj['idx'], obj['func']
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise AuditError(f'Invalid corpus JSONL at line {lineno}') from exc
            if not isinstance(i, str) or not i or not isinstance(code, str) or not code.strip() or i in out:
                raise AuditError(f'Duplicate/invalid corpus record at line {lineno}')
            out[i] = code
    return out


def read_c04_negatives(path):
    rows = csv_rows(path, ['project', 'split', 'id_a', 'id_b', 'label'])
    out = {}
    for r in rows:
        k = key(*(r[x].strip() for x in PAIR_KEY))
        if k in out:
            raise AuditError(f'Duplicate C04 negative record: {k}')
        if r['label'].strip() != '0':
            raise AuditError(f'C04 negative record label != 0: {k}')
        out[k] = r
    return out


def read_ledger(path):
    rows = csv_rows(path, list(PAIR_KEY) + ['c09_status', 'training_approved', 'adjudication'])
    out = {}
    for r in rows:
        k = key(*(r[x].strip() for x in PAIR_KEY))
        if k in out:
            raise AuditError(f'Duplicate C09 ledger record: {k}')
        status = r['c09_status'].strip()
        if status not in VALID_STATUS or r['training_approved'].strip().lower() != 'false':
            raise AuditError(f'Invalid C09 status/approval: {k} {status}')
        expected = {'EXCLUDE_CONFIRMED_CLONE':'CONFIRMED_CLONE', 'REVIEWED_NONCLONE':'CONFIRMED_NONCLONE', 'UNCERTAIN_REVIEW':'UNCERTAIN', 'UNREVIEWED':''}[status]
        if r['adjudication'].strip() != expected:
            raise AuditError(f'C09 adjudication/status conflict: {k}')
        out[k] = r
    return out


def ensure_disjoint_code(splits, corpus):
    split_ids = {}
    for split in ('train', 'valid'):
        these = set()
        for a,b in splits[split]:
            these.update((a,b))
        split_ids[split] = these
    shared = split_ids['train'] & split_ids['valid']
    if shared:
        raise AuditError(f'Cross-split shared function IDs: {len(shared)}')
    # Verify exact / normalized text as in C04's existing split audit, without AST heuristics.
    sha_by_split = {}
    norm_by_split = {}
    for split in ('train', 'valid'):
        sha_by_split[split] = {hashlib.sha256(corpus[i].encode()).digest() for i in split_ids[split]}
        norm_by_split[split] = {hashlib.sha256(''.join(corpus[i].split()).encode()).digest() for i in split_ids[split]}
    exact = len(sha_by_split['train'] & sha_by_split['valid'])
    normalized = len(norm_by_split['train'] & norm_by_split['valid'])
    if exact or normalized:
        raise AuditError(f'Cross-split code hash overlap: exact={exact} normalized={normalized}')
    return {'function_id_overlap':0, 'exact_code_hash_overlap':0, 'whitespace_normalized_code_hash_overlap':0}, split_ids


def compute(c04, c09):
    corpus = code_records(c04/'data.jsonl')
    original = {s:read_pairs(c04/f'{s}.txt', corpus) for s in ('train','valid')}
    candidates = read_c04_negatives(c04/'negative_candidates.csv')
    ledger = read_ledger(c09/'negative_candidate_correction_ledger.csv')
    if set(candidates) != set(ledger):
        raise AuditError(f'C04/C09 candidate identity mismatch: C04-only={len(set(candidates)-set(ledger))}, C09-only={len(set(ledger)-set(candidates))}')
    original_negative = set()
    positives = set()
    for split, pairs in original.items():
        for p, lab in pairs.items():
            (positives if lab == 1 else original_negative).add((split, *p))
    if len(original_negative) != len(candidates):
        raise AuditError(f'C04 pair files / negative_candidates count mismatch: {len(original_negative)} vs {len(candidates)}')
    for k in candidates:
        _, split, a, b = k
        if (split,a,b) not in original_negative:
            raise AuditError(f'C04 negative CSV pair missing in train/valid: {k}')
    status_count = Counter(r['c09_status'].strip() for r in ledger.values())
    corrected = {s:dict(original[s]) for s in ('train','valid')}
    exclusions = []
    retained_rows = []
    for k in sorted(ledger):
        project, split, a, b = k
        r = ledger[k]
        status = r['c09_status'].strip()
        if status in ('EXCLUDE_CONFIRMED_CLONE','UNCERTAIN_REVIEW'):
            del corrected[split][(a,b)]
            if status == 'EXCLUDE_CONFIRMED_CLONE':
                exclusions.append({**dict(zip(PAIR_KEY,k)), 'reason':'MANUAL_CONFIRMED_CLONE_NEGATIVE_LABEL_CONFLICT'})
        else:
            retained_rows.append({'project':project,'split':split,'id_a':a,'id_b':b,'label':'0',
                'verification':'HUMAN_REVIEWED_NONCLONE' if status == 'REVIEWED_NONCLONE' else 'SCREENING_ONLY_UNVERIFIED',
                'c09_status':status, 'review_id':r.get('review_id',''), 'label_origin':r.get('label_origin',''), 'training_approved':'false'})
    orig_pos = {(s,*p) for s,ps in original.items() for p,y in ps.items() if y == 1}
    new_pos = {(s,*p) for s,ps in corrected.items() for p,y in ps.items() if y == 1}
    if orig_pos != new_pos:
        raise AuditError('Positive pair preservation failure')
    for split, pairs in corrected.items():
        if len(pairs) != len(set(pairs)):
            raise AuditError('Duplicate pair in corrected split')
    leakage, split_ids = ensure_disjoint_code(corrected, corpus)
    use_ids = split_ids['train'] | split_ids['valid']
    before = {s:Counter(original[s].values()) for s in ('train','valid')}
    after = {s:Counter(corrected[s].values()) for s in ('train','valid')}
    if sum(after[s][0] for s in after) != len(retained_rows):
        raise AuditError('Retained negative count mismatch')
    audit = {
        'version':VERSION, 'status':'SCREENING_BASED_DATASET_CANDIDATE_NOT_APPROVED',
        'policy':'C09-confirmed-clone negatives removed; uncertain removed; all other C04 negatives retained as candidates with label provenance, including unreviewed.',
        'original_counts':{s:{'positive':before[s][1],'negative':before[s][0]} for s in ('train','valid')},
        'corrected_counts':{s:{'positive':after[s][1],'negative':after[s][0]} for s in ('train','valid')},
        'positive_pairs_preserved':len(orig_pos), 'original_negative_candidates':len(candidates),
        'confirmed_clone_exclusions':len(exclusions),'uncertain_exclusions':status_count['UNCERTAIN_REVIEW'],
        'negative_candidates_retained':len(retained_rows),
        'reviewed_nonclone_retained':status_count['REVIEWED_NONCLONE'],
        'unreviewed_retained':status_count['UNREVIEWED'],
        'retained_unverified_negative_fraction': round(status_count['UNREVIEWED']/len(retained_rows),8) if retained_rows else 0,
        'c09_status_counts':dict(sorted(status_count.items())),
        'pair_identity_reconciliation':'PASS','positive_preservation':'PASS','cross_split_leakage':leakage,
        'training_approval':False,'negative_labels_verified':False,'eligible_for_automatic_promotion':False,
        'heldout_leakage':'NOT_RECHECKED_IN_C10; C05 recorded heldout_overlap=0 on its source snapshot',
        'near_duplicate_and_project_level_leakage':'NOT_ESTABLISHED',
        'warnings':['The 13,322 unreviewed negatives are candidate labels, not verified non-clone ground truth.',
                    '12/274 is not a population-wide negative label error rate.',
                    'C07 blank-as-nonclone is a user-declared convention.',
                    'Confirmed clone exclusions are pair-level, not propagated through clone components.',
                    'No training approval and no changes to the original C04 or existing model data.'],
        'function_records_retained':len(use_ids),
    }
    return corpus, corrected, retained_rows, exclusions, audit, use_ids


def write_csv(path, rows, columns):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=columns, extrasaction='ignore');w.writeheader();w.writerows(rows)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--c04',type=Path,required=True)
    p.add_argument('--c09',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    corpus,corrected,retained,exclusions,audit,use_ids = compute(args.c04,args.c09)
    print('C-10 '+json.dumps({k:audit[k] for k in ('status','positive_pairs_preserved','confirmed_clone_exclusions','negative_candidates_retained','reviewed_nonclone_retained','unreviewed_retained','training_approval')},sort_keys=True))
    if args.dry_run:
        print('DRY RUN: no output created')
        return 0
    output=args.output or Path('results/c-10')/datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
    if output.exists():
        raise AuditError(f'Refusing to overwrite existing output: {output}')
    output.mkdir(parents=True)
    d=output/'python_mixed_screened_candidate'
    d.mkdir()
    for split in ('train','valid'):
        with (d/f'{split}.txt').open('w',encoding='utf-8') as f:
            for (a,b),lab in sorted(corrected[split].items()):
                f.write(f'{a}\t{b}\t{lab}\n')
    with (d/'data.jsonl').open('w',encoding='utf-8') as f:
        for idx in sorted(use_ids):
            f.write(json.dumps({'idx':idx,'func':corpus[idx]},ensure_ascii=False)+'\n')
    write_csv(d/'negative_candidates.csv',retained,NEG_COLS)
    write_csv(output/'confirmed_clone_exclusions.csv',exclusions,[*PAIR_KEY,'reason'])
    write_csv(output/'negative_label_provenance.csv',retained,NEG_COLS)
    (output/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    inputs=[args.c04/n for n in ('train.txt','valid.txt','data.jsonl','negative_candidates.csv')]+[args.c09/'negative_candidate_correction_ledger.csv']
    manifest={'version':VERSION,'created_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
      'inputs_sha256':{str(x):hfile(x) for x in inputs},
      'output_sha256':{str(x.relative_to(output)):hfile(x) for x in sorted(output.rglob('*')) if x.is_file()},
      'training_approval':False,'immutable_inputs':True,'outputs_only_in_new_timestamp_dir':True}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('OUTPUT:',output.resolve())
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (AuditError,OSError,ValueError) as ex:
        print(f'C-10 AUDIT ERROR: {ex}',file=sys.stderr);sys.exit(2)
