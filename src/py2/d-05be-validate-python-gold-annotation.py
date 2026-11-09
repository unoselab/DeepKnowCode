#!/usr/bin/env python3
"""D-05be v1: validate independent Python clone annotations, reconcile, never fabricate labels."""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

EXPECTED = 3971
REVIEW_COLUMNS = ('pair_id', 'project', 'clone_label', 'confidence', 'notes')
ADJ_COLUMNS = ('pair_id', 'project', 'reviewer_a_label', 'reviewer_b_label', 'resolved_gold_label', 'resolution_notes', 'adjudicator')
LABELS = {'0', '1', 'uncertain', ''}
CONFIDENCE = {'', 'high', 'medium', 'low'}


def fail(msg):
    raise ValueError(msg)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_csv(path, required, packet_index, kind):
    if not path.is_file():
        fail(f'Missing {kind} CSV: {path}')
    out = {}
    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or any(x not in reader.fieldnames for x in required):
            fail(f'{kind} unexpected header: {reader.fieldnames}')
        for lineno, row in enumerate(reader, 2):
            pid = (row.get('pair_id') or '').strip()
            if pid not in packet_index:
                fail(f'{kind}:{lineno} unknown pair_id {pid!r}')
            if pid in out:
                fail(f'{kind}:{lineno} duplicate pair_id {pid}')
            if (row.get('project') or '').strip() != packet_index[pid]['project']:
                fail(f'{kind}:{lineno} project mismatch for {pid}')
            out[pid] = {k: (v or '').strip() for k, v in row.items() if k}
    if set(out) != set(packet_index):
        fail(f'{kind} pair-id coverage mismatch: missing={len(set(packet_index)-set(out))}')
    return out


def manifest_validate(src):
    path = src / 'manifest.json'
    if not path.is_file():
        fail('Missing D-05bd manifest.json')
    m = json.loads(path.read_text(encoding='utf-8'))
    files = m.get('artifacts_sha256') or m.get('files')
    if not isinstance(files, dict):
        fail('Unknown D-05bd manifest checksum schema')
    for filename in ('annotation_packets.jsonl', 'reviewer_a_template.csv', 'reviewer_b_template.csv', 'adjudication_template.csv', 'annotation_preparation_audit.json'):
        entry = files.get(filename)
        expected = entry if isinstance(entry, str) else entry.get('sha256') if isinstance(entry, dict) else None
        if not expected or digest(src / filename) != expected:
            fail('D-05bd input checksum mismatch: ' + filename)
    return digest(path)


def load_packets(path):
    packets = {}
    with path.open(encoding='utf-8') as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                fail(f'Empty annotation packet line {lineno}')
            obj = json.loads(line)
            pid = obj.get('pair_id')
            project = obj.get('project')
            if not isinstance(pid, str) or not pid or pid in packets:
                fail(f'Duplicate/invalid annotation pair_id at {lineno}')
            if not isinstance(project, str) or not project or project == 'azure-sdk-for-python':
                fail(f'Invalid project at packet {lineno}')
            for side in ('left', 'right'):
                v = obj.get(side)
                if not isinstance(v, dict) or not isinstance(v.get('code'), str):
                    fail(f'Missing {side} source code at packet {lineno}')
                if hashlib.sha256(v['code'].encode('utf-8')).hexdigest() != v.get('code_sha256'):
                    fail(f'{side} code digest mismatch at packet {lineno}')
            packets[pid] = obj
    if len(packets) != EXPECTED:
        fail(f'Expected {EXPECTED} packets, got {len(packets)}')
    projects = {x['project'] for x in packets.values()}
    if len(projects) != 24 or 'scikit-learn' not in projects:
        fail('Active project coverage not 24 or scikit-learn missing')
    return packets


def kappa(a, b):
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in ('0', '1'))
    return (po - pe) / (1 - pe) if pe < 1 else (1.0 if po == 1 else None)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')


def write_jsonl(path, items):
    with path.open('w', encoding='utf-8') as fh:
        for obj in items:
            fh.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + '\n')


def write_csv(path, columns, rows):
    with path.open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def run(args):
    root = args.root.resolve()
    src = (root / args.source).resolve() if args.source else None
    if src is None:
        matches = sorted((root / 'results/d-05b').glob('python-gold-annotation-preparation-v1-*'))
        if len(matches) != 1:
            fail(f'Expected one D-05bd source, found {len(matches)}; pass --source')
        src = matches[0]
    source_manifest_sha = manifest_validate(src)
    packets = load_packets(src / 'annotation_packets.jsonl')
    orig_a = load_csv(src / 'reviewer_a_template.csv', REVIEW_COLUMNS, packets, 'D05BD_TEMPLATE_A')
    orig_b = load_csv(src / 'reviewer_b_template.csv', REVIEW_COLUMNS, packets, 'D05BD_TEMPLATE_B')
    orig_adj = load_csv(src / 'adjudication_template.csv', ADJ_COLUMNS, packets, 'D05BD_ADJUDICATION_TEMPLATE')
    if any(x['clone_label'] for x in list(orig_a.values()) + list(orig_b.values())) or any(x['resolved_gold_label'] for x in orig_adj.values()):
        fail('D-05bd templates must remain blank and immutable; supply separate submitted CSV files')

    a_path = Path(args.reviewer_a).resolve() if args.reviewer_a else src / 'reviewer_a_template.csv'
    b_path = Path(args.reviewer_b).resolve() if args.reviewer_b else src / 'reviewer_b_template.csv'
    adj_path = Path(args.adjudication).resolve() if args.adjudication else src / 'adjudication_template.csv'
    a = load_csv(a_path, REVIEW_COLUMNS, packets, 'REVIEWER_A')
    b = load_csv(b_path, REVIEW_COLUMNS, packets, 'REVIEWER_B')
    ad = load_csv(adj_path, ADJ_COLUMNS, packets, 'ADJUDICATION')
    if (args.reviewer_a is None) != (args.reviewer_b is None):
        fail('Supply both --reviewer-a and --reviewer-b, or neither')
    if args.reviewer_a:
        if not args.reviewer_a_id or not args.reviewer_b_id or args.reviewer_a_id == args.reviewer_b_id:
            fail('Independent reviewer CSVs require distinct --reviewer-a-id and --reviewer-b-id')
        if a_path == b_path or digest(a_path) == digest(b_path):
            fail('Reviewer A/B files must be separate and non-identical; cannot establish independent submissions')
    if args.adjudication and adj_path in (a_path, b_path):
        fail('Adjudication submission must be separate from reviewer submissions')
    verified, pending, stats = [], [], Counter()
    paired_a, paired_b = [], []
    for pid, packet in packets.items():
        av, bv, row = a[pid], b[pid], ad[pid]
        la, lb = av['clone_label'], bv['clone_label']
        if la not in LABELS or lb not in LABELS:
            fail(f'Invalid reviewer label for {pid} (only 0, 1, uncertain, blank)')
        for reviewer in (av, bv):
            if reviewer['confidence'].lower() not in CONFIDENCE:
                fail(f'Invalid confidence for {pid}')
            if reviewer['clone_label'] and not reviewer['notes']:
                fail(f'Annotation notes required for labeled pair {pid}')
        rv = row['resolved_gold_label']
        if rv not in ('', '0', '1'):
            fail(f'Invalid adjudicated label at {pid}')
        if row['reviewer_a_label'] and row['reviewer_a_label'] != la:
            fail(f'Adjudication reviewer A label mismatch for {pid}')
        if row['reviewer_b_label'] and row['reviewer_b_label'] != lb:
            fail(f'Adjudication reviewer B label mismatch for {pid}')
        both_valid = la in ('0', '1') and lb in ('0', '1')
        agreed = both_valid and la == lb
        if both_valid:
            paired_a.append(la)
            paired_b.append(lb)
        if not la or not lb:
            status = 'awaiting_independent_review'
            if rv:
                fail(f'Adjudication cannot precede two completed reviews: {pid}')
        elif agreed:
            status = 'verified_reviewer_agreement'
            if rv and rv != la:
                fail(f'Adjudicator cannot silently override reviewer consensus: {pid}')
        elif rv:
            if not row['resolution_notes'] or not row['adjudicator']:
                fail(f'Adjudication requires resolution_notes and adjudicator at {pid}')
            status = 'verified_adjudicated'
        else:
            status = 'awaiting_adjudication'
        if status.startswith('verified_'):
            label = int(rv if status == 'verified_adjudicated' else la)
            verified.append({'pair_id':pid, 'project':packet['project'],
                             'left_code_sha256':packet['left']['code_sha256'],
                             'right_code_sha256':packet['right']['code_sha256'],
                             'gold_label':label, 'verification_method':status,
                             'reviewer_a_id':args.reviewer_a_id, 'reviewer_b_id':args.reviewer_b_id,
                             'reviewer_a_label':int(la) if la in ('0','1') else la,
                             'reviewer_b_label':int(lb) if lb in ('0','1') else lb,
                             'adjudicator':row['adjudicator'] if status == 'verified_adjudicated' else None})
        else:
            pending.append({'pair_id':pid, 'project':packet['project'],
                            'reviewer_a_label':la, 'reviewer_b_label':lb,
                            'pending_reason':status})
        stats[status] += 1
    if args.strict_complete and pending:
        fail(f'--strict-complete requested but {len(pending)} unresolved records remain')
    labels = Counter(str(x['gold_label']) for x in verified)
    now = datetime.now(ZoneInfo('America/Chicago')).isoformat()
    summary = {'status':'D05BE_V1_VALIDATION_COMPLETE_NO_GOLD' if not verified else ('D05BE_V1_ALL_PAIRS_VERIFIED_C01_ONLY' if not pending else 'D05BE_V1_PARTIAL_GOLD_VERIFICATION'),
               'created_at_chicago':now, 'version':'v1', 'source_policy':'C01_ONLY',
               'input_annotation_packets':len(packets),'active_projects':24,
               'verified_pairs':len(verified),'unresolved_pairs':len(pending),
               'gold_label_distribution':dict(labels), 'review_status_counts':dict(stats),
               'two_rater_binary_cases':len(paired_a),'cohen_kappa_binary_complete_cases':kappa(paired_a, paired_b),
               'source_manifest_sha256':source_manifest_sha,'reviewer_a_file_sha256':digest(a_path),
               'reviewer_b_file_sha256':digest(b_path), 'adjudication_file_sha256':digest(adj_path),
               'reviewer_a_id':args.reviewer_a_id, 'reviewer_b_id':args.reviewer_b_id,
               'scope_warning':'C-01 detected pairs only, not a representative negative/non-detected-pair evaluation sample. Verified labels here are not a complete population-level benchmark.',
               'independence_warning':'Distinct reviewer IDs/files do not independently prove human review independence.',
               'test_eligibility':'NOT_ESTABLISHED: separate representative negative sampling and evaluation protocol required',
               'label_provenance':'human reviewer submissions only; no detector-derived labels'}
    short = {k:summary[k] for k in ('status','input_annotation_packets','verified_pairs','unresolved_pairs','gold_label_distribution','review_status_counts','cohen_kappa_binary_complete_cases','test_eligibility')}
    if args.dry_run:
        short['status']='DRY_RUN'; print(json.dumps(short,ensure_ascii=False,sort_keys=True));return
    output_root = root / 'results/d-05b'
    output_root.mkdir(parents=True,exist_ok=True)
    stamp = datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
    dest=output_root / ('python-gold-annotation-validation-v1-' + stamp)
    dest.mkdir(exist_ok=False)
    write_jsonl(dest/'verified_pairs.jsonl',verified)
    write_csv(dest/'unresolved_review.csv',('pair_id','project','reviewer_a_label','reviewer_b_label','pending_reason'),pending)
    write_json(dest/'annotation_validation_audit.json',summary)
    names=('verified_pairs.jsonl','unresolved_review.csv','annotation_validation_audit.json')
    write_json(dest/'manifest.json',{'version':'v1','created_at_chicago':now,
                  'source_d05bd_manifest_sha256':source_manifest_sha,
                  'artifacts_sha256':{name:digest(dest/name) for name in names}})
    short['out']=str(dest)
    print(json.dumps(short,ensure_ascii=False,sort_keys=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--source',help='D-05bd v1 annotation-preparation output folder')
    p.add_argument('--reviewer-a',help='Submitted reviewer A CSV (never edit D-05bd template in place)')
    p.add_argument('--reviewer-b',help='Submitted reviewer B CSV')
    p.add_argument('--adjudication',help='Completed adjudication CSV; optional unless resolving conflicts')
    p.add_argument('--reviewer-a-id',help='Pseudonymous independent annotator ID')
    p.add_argument('--reviewer-b-id',help='Pseudonymous independent annotator ID')
    p.add_argument('--strict-complete',action='store_true',help='Abort unless all 3971 pairs are resolved')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    try:run(args)
    except (ValueError, OSError, KeyError, json.JSONDecodeError, csv.Error) as exc:
        print('D05BE ERROR:',str(exc),file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':
    sys.exit(main())
