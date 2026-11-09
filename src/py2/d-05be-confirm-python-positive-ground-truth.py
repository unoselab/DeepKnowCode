#!/usr/bin/env python3
"""D-05be v2: validate detector-confirmation CSV, produce reviewer-confirmed positive labels only."""
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
REQUIRED = ('pair_id', 'project', 'reviewer_a_label', 'reviewer_b_label', 'pending_reason')
POLICY = 'BLANK_MEANS_REVIEWER_CONFIRMS_C01_CLONE_POSITIVE'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')


def fail(msg):
    raise ValueError(msg)


def source_dir(root, selected):
    if selected:
        p = Path(selected)
        p = p if p.is_absolute() else root / p
        return p.resolve()
    candidates = sorted((root / 'results/d-05b').glob('python-gold-annotation-preparation-v1-*'))
    if len(candidates) != 1:
        fail(f'Expected precisely one D-05bd source, found {len(candidates)}; use --source')
    return candidates[0].resolve()


def verify_d05bd(path):
    mf = path / 'manifest.json'
    if not mf.is_file():
        fail('D-05bd manifest missing')
    manifest = json.loads(mf.read_text(encoding='utf-8'))
    refs = manifest.get('artifacts_sha256', {})
    for name in ['annotation_packets.jsonl', 'annotation_preparation_audit.json',
                 'reviewer_a_template.csv', 'reviewer_b_template.csv', 'adjudication_template.csv']:
        exp = refs.get(name)
        if not exp or not (path / name).is_file() or sha256(path / name) != exp:
            fail('D-05bd SHA256 mismatch: ' + name)
    policy_audit = json.loads((path / 'annotation_preparation_audit.json').read_text())
    if policy_audit.get('counts', {}).get('annotation_packets') != EXPECTED:
        fail('D-05bd annotation packet count mismatch')
    if policy_audit.get('counts', {}).get('active_projects') != 24:
        fail('D-05bd active project count mismatch')
    if 'C01_ONLY' not in policy_audit.get('policy', ''):
        fail('D-05bd source policy not C01_ONLY')
    return sha256(mf)


def load_packets(path):
    packets = {}
    with path.open(encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                fail(f'Blank packet line {n}')
            record = json.loads(line)
            pid = record.get('pair_id')
            project = record.get('project')
            if not isinstance(pid, str) or not pid or pid in packets:
                fail(f'Invalid or duplicated packet ID on line {n}')
            if not isinstance(project, str) or not project or project == 'azure-sdk-for-python':
                fail(f'Invalid packet project line {n}')
            for s in ('left', 'right'):
                side = record.get(s)
                if not isinstance(side, dict) or not isinstance(side.get('code'), str):
                    fail(f'Incomplete {s} source in packet {pid}')
                if hashlib.sha256(side['code'].encode('utf-8')).hexdigest() != side.get('code_sha256'):
                    fail(f'{s} SHA mismatch for {pid}')
            packets[pid] = record
    if len(packets) != EXPECTED:
        fail(f'Expected {EXPECTED} packets; got {len(packets)}')
    if len({p['project'] for p in packets.values()}) != 24 or not any(p['project']=='scikit-learn' for p in packets.values()):
        fail('24 active projects including scikit-learn required')
    return packets


def read_confirmation_csv(path, packets):
    if not path.is_file():
        fail('Missing submitted confirmation CSV: ' + str(path))
    result = {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames or not all(k in rd.fieldnames for k in REQUIRED):
            fail('Unexpected confirmation CSV columns')
        for lineno, r in enumerate(rd, 2):
            pid = (r.get('pair_id') or '').strip()
            if pid not in packets or pid in result:
                fail(f'Unknown or duplicated CSV pair_id at line {lineno}: {pid}')
            if (r.get('project') or '').strip() != packets[pid]['project']:
                fail(f'Project mismatch: {pid}')
            # Explicit convention declared by user: both blank cells = explicit approval of C-01 result.
            # Reject all other values rather than silently reinterpret nonblank cells.
            if (r.get('reviewer_a_label') or '').strip() != '' or (r.get('reviewer_b_label') or '').strip() != '':
                fail(f'Nonblank reviewer cells require separate adjudication protocol: {pid}')
            if (r.get('pending_reason') or '').strip() not in ('', 'N/A'):
                fail(f'Unexpected pending_reason for confirmed positive: {pid}')
            result[pid] = r
    if set(result) != set(packets):
        fail(f'CSV coverage incomplete: {len(result)} vs {len(packets)}')
    return result


def run(args):
    root = args.root.resolve()
    source = source_dir(root, args.source)
    input_csv = Path(args.confirmations)
    input_csv = (input_csv if input_csv.is_absolute() else root / input_csv).resolve()
    source_manifest_sha = verify_d05bd(source)
    packets = load_packets(source / 'annotation_packets.jsonl')
    read_confirmation_csv(input_csv, packets)
    counts = Counter(p['project'] for p in packets.values())
    summary = {
        'status': 'DRY_RUN' if args.dry_run else 'D05BE_V2_REVIEWER_CONFIRMED_C01_POSITIVES_NO_NEGATIVES',
        'source_policy': 'C01_ONLY', 'confirmation_policy': POLICY,
        'candidate_packets': len(packets), 'reviewer_confirmed_positive_pairs': len(packets),
        'verified_negative_pairs': 0, 'active_projects': len(counts),
        'evaluation_ready_for_binary_metrics': False,
        'independent_blind_annotation_claim': False,
        'reviewer_identities_verified': False,
        'annotation_method': 'reviewer_agreement_with_existing_detector_output',
    }
    if args.dry_run:
        print(json.dumps(summary, sort_keys=True))
        return
    now = datetime.now(ZoneInfo('America/Chicago'))
    out = root / 'results/d-05b' / ('python-confirmed-positive-v2-' + now.strftime('%Y%m%d-%H%M%S-%f'))
    out.mkdir(parents=True, exist_ok=False)
    verified_name = 'reviewer_confirmed_positive_pairs.jsonl'
    with (out / verified_name).open('w', encoding='utf-8') as f:
        for pid, p in packets.items():
            item = {
                'pair_id': pid, 'project': p['project'], 'gold_label': 1,
                'label_scope': 'reviewer_confirmed_positive_candidate',
                'label_source': 'C01_DETECTED_REVIEWER_A_B_CONFIRMED',
                'annotation_convention': POLICY,
                'verification_method': 'reviewer_confirmation_of_detector_prediction',
                'reviewer_a_confirmed': True, 'reviewer_b_confirmed': True,
                'independent_blind_annotation': False,
                'left_code_sha256': p['left']['code_sha256'],
                'right_code_sha256': p['right']['code_sha256'],
                'left_locator': {k:p['left'].get(k) for k in ('path','start_line','end_line','function')},
                'right_locator': {k:p['right'].get(k) for k in ('path','start_line','end_line','function')},
                'binary_evaluation_ready': False,
            }
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n')
    audit = dict(summary)
    audit.update({
        'created_at_chicago':now.isoformat(),
        'input_reviewers_csv':str(input_csv), 'input_reviewers_csv_sha256':sha256(input_csv),
        'annotation_packet_manifest_sha256':source_manifest_sha,
        'source_annotation_package':str(source),
        'project_counts':dict(sorted(counts.items())),
        'excluded_source':'data/evaluation/ground_truth/python_legacy',
        'legacy_training_used':False,
        'limitations':['Confirmation is agreement with C-01 detections, not an independently blinded clone discovery study.',
                       'No independently verified negative sample; F1/Recall/MCC against a representative binary population not authorized.',
                       'Reviewers identities and actual review process not externally authenticated by this CSV.']
    })
    write_json(out / 'confirmation_audit.json', audit)
    mf = {
        'version':'v2', 'created_at_chicago':now.isoformat(), 'source_policy':'C01_ONLY',
        'confirmation_policy':POLICY,
        'input_reviewers_csv_sha256':sha256(input_csv),
        'source_annotation_manifest_sha256':source_manifest_sha,
        'files':{name: {'sha256':sha256(out / name), 'bytes':(out / name).stat().st_size}
                 for name in (verified_name, 'confirmation_audit.json')}
    }
    write_json(out / 'manifest.json', mf)
    summary['out'] = str(out)
    print(json.dumps(summary, sort_keys=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--source', help='Exact D-05bd annotation preparation directory')
    p.add_argument('--confirmations', required=True, help='Submitted reviewer confirmation CSV')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print('D-05be v2 ERROR:', e, file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
