#!/usr/bin/env python3
"""C-09: auditable, non-destructive reconciliation of C-05 and C-08."""
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

KEY = ('project', 'split', 'id_a', 'id_b')
REVIEW_LABELS = {'CONFIRMED_CLONE', 'CONFIRMED_NONCLONE', 'UNCERTAIN'}
LEDGER_COLUMNS = ['project', 'split', 'id_a', 'id_b', 'screening', 'flags', 'review_group', 'review_id', 'adjudication', 'label_origin', 'c09_status', 'training_approved']

class AuditError(ValueError):
    pass


def rows(path):
    if not path.is_file():
        raise AuditError(f'Missing input: {path}')
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise AuditError(f'CSV has no header: {path}')
        return list(reader), set(reader.fieldnames)


def key(r):
    return tuple(r[x].strip() for x in KEY)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, records, columns):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(records)


def reconcile(c05, c08, c08_summary=None):
    c05_rows, fields05 = rows(c05)
    c08_rows, fields08 = rows(c08)
    need05 = set(KEY) | {'screening', 'flags'}
    need08 = set(KEY) | {'review_id', 'review_group', 'adjudication', 'label_origin'}
    if not need05 <= fields05 or not need08 <= fields08:
        raise AuditError(f'Missing columns: C05={sorted(need05-fields05)}, C08={sorted(need08-fields08)}')
    bykey = {}
    for row in c05_rows:
        k = key(row)
        if not all(k):
            raise AuditError('C05 has empty identity')
        if k in bykey:
            raise AuditError(f'Duplicate C05 identity: {k}')
        bykey[k] = row
    reviewed = {}
    ids = set()
    for row in c08_rows:
        k = key(row)
        rid = row['review_id'].strip()
        label = row['adjudication'].strip()
        if not rid or rid in ids or k in reviewed:
            raise AuditError(f'Duplicate/empty review_id or Pair: {rid} {k}')
        if k not in bykey:
            raise AuditError(f'C08 identity absent from C05: {k}')
        if label not in REVIEW_LABELS:
            raise AuditError(f'Invalid C08 adjudication: {label!r} for {rid}')
        if row['label_origin'].strip() not in {'EXPLICIT_CSV', 'BLANK_AS_NONCLONE_USER_POLICY', 'UNREVIEWED_BLANK', 'EXPLICIT_NONCLONE', 'EXPLICIT_UNCERTAIN'}:
            raise AuditError(f'Unrecognized label_origin for {rid}: {row["label_origin"]!r}')
        # C08 user-specific blank interpretation is expected and must be traceable.
        if row['label_origin'].strip() == 'BLANK_AS_NONCLONE_USER_POLICY' and label != 'CONFIRMED_NONCLONE':
            raise AuditError(f'Conflicting blank label policy: {rid}')
        reviewed[k] = row
        ids.add(rid)
    if c08_summary:
        with c08_summary.open(encoding='utf-8') as f: s = json.load(f)
        if s.get('status') != 'AUDIT_PASS_REVIEWED_SUBSET_ONLY':
            raise AuditError(f'C08 summary status is not audited: {s.get("status")}')
        if s.get('review_rows') != len(c08_rows) or s.get('matched_rows') != len(c08_rows):
            raise AuditError('C08 summary count differs from reviewed CSV')
        labels = Counter(x['adjudication'].strip() for x in c08_rows)
        if dict(labels) != s.get('labels'):
            raise AuditError('C08 summary label counts differ from reviewed CSV')
    ledger = []
    for k, candidate in bykey.items():
        review = reviewed.get(k, {})
        label = review.get('adjudication', '').strip()
        status = {'CONFIRMED_CLONE': 'EXCLUDE_CONFIRMED_CLONE', 'CONFIRMED_NONCLONE': 'REVIEWED_NONCLONE', 'UNCERTAIN': 'UNCERTAIN_REVIEW'}.get(label, 'UNREVIEWED')
        ledger.append({**{name: candidate.get(name, '') for name in KEY}, 'screening': candidate['screening'], 'flags': candidate['flags'], 'review_group': review.get('review_group',''), 'review_id': review.get('review_id',''), 'adjudication': label, 'label_origin': review.get('label_origin',''), 'c09_status': status, 'training_approved': 'false'})
    ledger.sort(key=lambda r: (r['project'], r['split'], r['id_a'], r['id_b']))
    return ledger, reviewed


def report(ledger, reviewed, c05, c08, c08_summary):
    status = Counter(x['c09_status'] for x in ledger)
    bygroup = defaultdict(lambda: Counter())
    byproject = defaultdict(lambda: Counter())
    for k, r in reviewed.items():
        bygroup[r['review_group']][r['adjudication']] += 1
        byproject[r['project']][r['adjudication']] += 1
    # Census in high priority stratum; other two strata were project-stratified purposive samples.
    result = {
        'version':'v1', 'status':'CORRECTION_LEDGER_PREPARED_TRAINING_NOT_APPROVED',
        'total_negative_candidates':len(ledger), 'reviewed_pairs':len(reviewed),
        'confirmed_clone_exclusions':status['EXCLUDE_CONFIRMED_CLONE'],
        'reviewed_nonclones':status['REVIEWED_NONCLONE'],
        'uncertain_reviewed':status['UNCERTAIN_REVIEW'],
        'unreviewed_candidates':status['UNREVIEWED'],
        'status_counts':dict(sorted(status.items())),
        'review_group_labels':{k: dict(sorted(v.items())) for k,v in sorted(bygroup.items())},
        'project_labels':{k: dict(sorted(v.items())) for k,v in sorted(byproject.items())},
        'training_approval':False,
        'negative_population_verified':False,
        'c08_blank_label_policy':'BLANK_AS_NONCLONE_USER_POLICY',
        'sampling_policy':'C06 selected all high priority plus project-stratified length-only and screened-baseline samples; not a probability sample with documented inclusion probabilities',
        'inference_limitations':[
            'Observed 12/274 clone fraction is NOT a population-wide error rate.',
            'Zero clones in two reviewed samples does NOT establish zero population clone prevalence.',
            'Unreviewed pairs retain unverified labels and cannot be automatically promoted.',
            'C08 blank-as-nonclone is a user-declared adjudication convention; reviewer metadata is incomplete.',
            'Confirmed clone exclusion applies to exact project/split/function pair only; no automatic clone-component propagation.',
            'C09 does not edit C04 data.jsonl/train.txt/valid.txt or any model adapter.',
        ],
        'input_files':{'c05':str(c05),'c08':str(c08),'c08_summary':str(c08_summary) if c08_summary else None},
        'next_step':'Define explicit population-label policy and review representativeness; revalidate original C04 pair mapping before constructing a new training dataset.',
    }
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--c05',type=Path,required=True,help='C05 negative_verification.csv or parent directory')
    p.add_argument('--c08',type=Path,required=True,help='C08 adjudicated_review_subset.csv or parent directory')
    p.add_argument('--output',type=Path)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    c05=args.c05/'negative_verification.csv' if args.c05.is_dir() else args.c05
    c08=args.c08/'adjudicated_review_subset.csv' if args.c08.is_dir() else args.c08
    c08_summary=c08.parent/'summary.json'
    if not c08_summary.exists():c08_summary=None
    ledger, reviewed=reconcile(c05,c08,c08_summary)
    summary=report(ledger,reviewed,c05,c08,c08_summary)
    print('C-09 '+json.dumps({k:summary[k] for k in ('status','total_negative_candidates','reviewed_pairs','confirmed_clone_exclusions','reviewed_nonclones','unreviewed_candidates','training_approval')},sort_keys=True))
    if args.dry_run:
        print('DRY RUN: no outputs created')
        return 0
    output=args.output or Path('results/c-09') / datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
    if output.exists():raise AuditError(f'Output already exists, will not overwrite: {output}')
    output.mkdir(parents=True)
    groups={
        'negative_candidate_correction_ledger.csv':ledger,
        'confirmed_clone_exclusions.csv':[r for r in ledger if r['c09_status']=='EXCLUDE_CONFIRMED_CLONE'],
        'reviewed_nonclone_pairs.csv':[r for r in ledger if r['c09_status']=='REVIEWED_NONCLONE'],
        'unreviewed_negative_candidates.csv':[r for r in ledger if r['c09_status']=='UNREVIEWED'],
        'uncertain_reviewed_pairs.csv':[r for r in ledger if r['c09_status']=='UNCERTAIN_REVIEW'],
    }
    for name, rs in groups.items():write_csv(output/name,rs,LEDGER_COLUMNS)
    summary_path=output/'summary.json'
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest={
        'version':'v1','created_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
        'inputs_sha256':{str(p):sha256(p) for p in (c05,c08) if p is not None},
        'output_sha256':{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
        'training_approval':False,'preservation':'Inputs and existing dataset not modified',
    }
    if c08_summary:manifest['inputs_sha256'][str(c08_summary)]=sha256(c08_summary)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('OUTPUT:',output.resolve())
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (AuditError, ValueError, OSError, json.JSONDecodeError) as e:
        print(f'C-09 AUDIT ERROR: {e}',file=sys.stderr);sys.exit(2)
