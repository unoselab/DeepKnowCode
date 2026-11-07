#!/usr/bin/env python3
"""C-11b v2: scope-limited controlled-training decision; never mutate datasets."""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v2'
EXPECTED = {'positive': 13596, 'negative': 13584, 'total': 27180, 'reviewed': 262, 'unverified': 13322, 'excluded': 12}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with path.open(encoding='utf-8-sig') as f:
        return json.load(f)


def find_run(root, section, name):
    parent = root / 'results' / section
    if not parent.is_dir():
        raise ValueError(f'Missing {section} run directory: {parent}')
    matches = sorted(p for p in parent.iterdir() if p.is_dir() and (p / name).is_file())
    if not matches:
        raise ValueError(f'No {section} results containing {name}')
    return matches[-1]


def resolve(root, path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else root / p


def verify_hash_listing(directory, entries, label, errors):
    for rel, expected_hash in entries.items():
        if not isinstance(expected_hash, str) or not re.fullmatch('[a-f0-9]{64}', expected_hash):
            errors.append(f'{label}: invalid hash for {rel}')
            continue
        path = directory / rel
        if not path.is_file():
            errors.append(f'{label}: missing output {path}')
        elif sha256(path) != expected_hash:
            errors.append(f'{label}: SHA256 mismatch {path}')


def verify_upstream_hashes(c11_manifest, c10, errors):
    hashes = c11_manifest.get('inputs_sha256', {})
    files = ['audit.json', 'manifest.json', 'negative_label_provenance.csv',
             'python_mixed_screened_candidate/data.jsonl',
             'python_mixed_screened_candidate/train.txt',
             'python_mixed_screened_candidate/valid.txt',
             'python_mixed_screened_candidate/negative_candidates.csv']
    for rel in files:
        p = c10 / rel
        expected = [v for k, v in hashes.items() if k.replace('\\', '/').endswith('/' + rel)]
        if len(expected) != 1:
            errors.append(f'C-11a input manifest cannot identify unique C-10 hash: {rel}')
        elif not p.is_file() or sha256(p) != expected[0]:
            errors.append(f'C-10 source missing/changed since C-11a: {rel}')


def count_pairs(c10, errors):
    split_counts = {}
    seen = set()
    for split in ('train', 'valid'):
        p = c10 / 'python_mixed_screened_candidate' / f'{split}.txt'
        c = Counter()
        with p.open(encoding='utf-8-sig') as f:
            for no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 3 or fields[2] not in ('0','1') or fields[0] == fields[1]:
                    errors.append(f'Invalid pair {p}:{no}')
                    continue
                pair = (split, *sorted(fields[:2]))
                if pair in seen:
                    errors.append(f'Duplicate pair {p}:{no}')
                seen.add(pair)
                c['negative' if fields[2] == '0' else 'positive'] += 1
        split_counts[split] = dict(c)
    positive = sum(c.get('positive', 0) for c in split_counts.values())
    negative = sum(c.get('negative', 0) for c in split_counts.values())
    if (positive,negative) != (EXPECTED['positive'],EXPECTED['negative']):
        errors.append(f'Unexpected pair counts: positive={positive}, negative={negative}')
    return {'by_split':split_counts,'positive':positive,'negative':negative,'total':positive+negative}


def validate_evidence(root, evidence_path, errors):
    if evidence_path is None:
        return {'near_duplicate_test':{'status':'NOT_ESTABLISHED'},
                'project_independence':{'status':'NOT_ESTABLISHED'},
                'source':'NO_ADDITIONAL_EVIDENCE'}
    record = read_json(evidence_path)
    result = {'source':str(evidence_path), 'sha256':sha256(evidence_path)}
    for key in ('near_duplicate_test', 'project_independence'):
        item = record.get(key)
        if not isinstance(item, dict):
            errors.append(f'Evidence {key}: missing structured record')
            result[key] = {'status':'NOT_ESTABLISHED'}
            continue
        status = item.get('status','NOT_ESTABLISHED')
        report = item.get('report_path')
        report_hash = item.get('report_sha256')
        method = item.get('method','')
        result[key] = {'status':status,'method':method,'report_path':report,'report_sha256':report_hash}
        if status == 'PASS':
            if not isinstance(method, str) or len(method.strip()) < 15:
                errors.append(f'Evidence {key}: descriptive method required')
            if not report or not report_hash:
                errors.append(f'Evidence {key}: report_path/report_sha256 required')
                continue
            path = resolve(root, report)
            if not path.is_file() or sha256(path) != report_hash:
                errors.append(f'Evidence {key}: report missing or SHA256 mismatch')
        elif status not in ('NOT_ESTABLISHED','FAIL','REVIEW_REQUIRED'):
            errors.append(f'Evidence {key}: unsupported status {status}')
    result['limitation'] = 'External evidence is authenticated by SHA256 only; its scientific validity is not independently established by C-11b.'
    return result


SCOPE_LIMITATIONS = [
    'Near-duplicate held-out leakage has not been assessed; exact/normalized hash overlap is not sufficient to establish no near-duplicate leakage.',
    'Project-disjoint evaluation has not been established; do not claim unseen-project generalization.',
    '13,322 negative training candidates were not manually adjudicated, and remain screening-only candidates.',
    'Approval, if explicitly granted, is limited to controlled exploratory training, not ground-truth validation or paper-ready leakage claims.',
]


def check_promoted(c10, promoted):
    """Compare the four promoted dataset files against the immutable C-10 run."""
    compared = {}
    for name in ('data.jsonl', 'train.txt', 'valid.txt', 'negative_candidates.csv'):
        src = c10 / 'python_mixed_screened_candidate' / name
        dst = promoted / name
        if not src.is_file() or not dst.is_file():
            compared[name] = {'status': 'MISSING', 'source': str(src), 'promoted': str(dst)}
        else:
            a, b = sha256(src), sha256(dst)
            compared[name] = {'status': 'PASS' if a == b else 'MISMATCH',
                              'sha256_c10': a, 'sha256_promoted': b}
    return {'directory': str(promoted), 'files': compared,
            'status': 'PASS' if all(x['status'] == 'PASS' for x in compared.values()) else 'NOT_MATCHED'}


def decision(audit, reliability, evidence, issues, args, promoted=None):
    """Scope-limited policy: missing optional evidence is a limitation, never a PASS."""
    if issues:
        return 'AUDIT_FAILED', ['INTEGRITY_OR_EVIDENCE_ERROR'], False
    blockers = []
    if audit.get('status') != 'AUDIT_PASSED_REVIEW_REQUIRED':
        blockers.append('C11A_STATUS_NOT_READY')
    if any(audit.get(k) != 'PASS' for k in
           ('positive_preservation', 'positive_provenance', 'negative_pair_reconciliation')):
        blockers.append('C11A_LABEL_OR_PROVENANCE_FAILED')
    if audit.get('upstream_manifest_sha256') != 'PASS':
        blockers.append('C11A_UPSTREAM_SHA256_FAILED')
    if audit.get('excluded_confirmed_clones') != EXPECTED['excluded']:
        blockers.append('CONFIRMED_CLONE_EXCLUSIONS_MISMATCH')
    splits = audit.get('split_leakage')
    if not isinstance(splits, dict) or set(splits) != {'function_id','exact_code_hash','normalized_code_hash'} or any(type(v) is not int or v != 0 for v in splits.values()):
        blockers.append('TRAIN_VALID_LEAKAGE_OR_UNVERIFIED')
    heldout = audit.get('heldout_leakage', [])
    if len(heldout) != 2 or {r.get('heldout') for r in heldout} != {'java','python'} or any(
        r.get('status') != 'PASS' or r.get('exact_overlap') != 0 or r.get('normalized_overlap') != 0 or r.get('records',0) <= 0 for r in heldout
    ):
        blockers.append('HELDOUT_EXACT_OR_NORMALIZED_LEAKAGE_OR_UNVERIFIED')
    if reliability.get('human_reviewed_nonclone') != EXPECTED['reviewed'] or reliability.get('screening_only_unverified') != EXPECTED['unverified'] or reliability.get('candidate_total') != EXPECTED['negative']:
        blockers.append('NEGATIVE_RELIABILITY_COUNTS_MISMATCH')
    if audit.get('near_duplicate_test_leakage') not in ('NOT_ASSESSED','NOT_ESTABLISHED'):
        blockers.append('UNEXPECTED_C11A_NEAR_DUPLICATE_STATUS')
    if audit.get('project_level_test_leakage') not in ('NOT_ESTABLISHED','NOT_ASSESSED'):
        blockers.append('UNEXPECTED_C11A_PROJECT_STATUS')
    if promoted is None or promoted.get('status') != 'PASS':
        blockers.append('PROMOTED_DATASET_NOT_VERIFIED')
    # External FAIL reports must never be overridden by a scope waiver.
    for key in ('near_duplicate_test','project_independence'):
        if evidence[key]['status'] in ('FAIL','REVIEW_REQUIRED'):
            blockers.append(key.upper() + '_ADVERSE_EVIDENCE')
    if blockers:
        return 'AUDIT_FAILED_OR_DATASET_MISMATCH', blockers, False
    approval_requested = bool(args.approve_controlled)
    if not approval_requested:
        return 'SCOPE_LIMITED_AUDIT_PASS_EXPLICIT_APPROVAL_REQUIRED', ['EXPLICIT_HUMAN_APPROVAL_NOT_PROVIDED'], False
    if not getattr(args, 'accept_scope_limitations', False):
        blockers.append('SCOPE_LIMITATIONS_NOT_ACCEPTED')
    if not args.accept_unverified_negative_risk:
        blockers.append('UNVERIFIED_NEGATIVE_RISK_NOT_ACCEPTED')
    if not args.reviewer.strip() or len(args.rationale.strip()) < 50:
        blockers.append('REVIEWER_AND_SUBSTANTIVE_RATIONALE_REQUIRED')
    if blockers:
        return 'CONTROLLED_TRAINING_NOT_APPROVED_REVIEW_REQUIRED', blockers, False
    return 'APPROVED_FOR_SCOPE_LIMITED_CONTROLLED_TRAINING', [], True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--c11a',type=Path)
    p.add_argument('--c10',type=Path)
    p.add_argument('--evidence',type=Path,help='Optional independent near-duplicate and project-independence evidence JSON')
    p.add_argument('--approve-controlled',action='store_true',help='Explicit human decision for scope-limited exploratory training')
    p.add_argument('--accept-scope-limitations',action='store_true',help='Explicitly acknowledge incomplete near-duplicate and project-level checks')
    p.add_argument('--promoted-dataset',type=Path,default=Path('data/training/clean/python_mixed'))
    p.add_argument('--accept-unverified-negative-risk',action='store_true')
    p.add_argument('--reviewer',default='')
    p.add_argument('--rationale',default='')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args(argv)
    root=args.root.resolve()
    c11a=resolve(root,args.c11a).resolve() if args.c11a else find_run(root,'c-11a','audit.json')
    c10=resolve(root,args.c10).resolve() if args.c10 else find_run(root,'c-10','python_mixed_screened_candidate/train.txt')
    audit=read_json(c11a/'audit.json')
    reliability=read_json(c11a/'negative_reliability.json')
    upstream_manifest=read_json(c11a/'manifest.json')
    errs=[]
    verify_hash_listing(c11a,upstream_manifest.get('output_sha256',{}),'C-11a',errs)
    verify_upstream_hashes(upstream_manifest,c10,errs)
    counts=count_pairs(c10,errs)
    evidence_path=resolve(root,args.evidence).resolve() if args.evidence else None
    evidence=validate_evidence(root,evidence_path,errs)
    promoted_path=resolve(root,args.promoted_dataset).resolve()
    promoted=check_promoted(c10,promoted_path)
    state,blockers,approval=decision(audit,reliability,evidence,errs,args,promoted)
    now=datetime.now(ZoneInfo('America/Chicago')).isoformat()
    result={'version':VERSION,'created_at_chicago':now,'status':state,
            'training_approval':approval,'controlled_training_approval':approval,
            'source_run_c11a':str(c11a),'source_run_c10':str(c10),
            'dataset_counts':counts, 'unverified_negative_pairs':EXPECTED['unverified'],
            'negative_label_risk_accepted':bool(approval and args.accept_unverified_negative_risk),
            'scope_limitations_accepted':bool(approval and args.accept_scope_limitations),
            'approval_scope':'SCOPE_LIMITED_EXPLORATORY_TRAINING_ONLY' if approval else None,
            'promoted_dataset_verification':promoted,
            'reviewer':args.reviewer if approval else None,
            'approval_rationale':args.rationale if approval else None,
            'blockers':blockers,'integrity_errors':errs,
            'evidence':evidence, 'unestablished_evidence':[
                key for key in ('near_duplicate_test','project_independence') if evidence[key]['status'] != 'PASS'],
            'scope_limitations':SCOPE_LIMITATIONS,
            'limitations':['C-11b does not itself run a near-duplicate detector or establish test-project provenance.',
                           'External evidence SHA256 does not verify the scientific appropriateness of the supplied analysis.',
                           '13,322 negative labels remain unreviewed; acceptance does not convert them to ground truth.',
                           'No dataset promotion or model training is performed.']}
    print('C-11b '+json.dumps({'status':state,'approval':approval,'pairs':counts['total'],
              'blockers':blockers,'errors':errs,'promoted':promoted['status']},sort_keys=True))
    if args.dry_run:
        print('DRY RUN: no output created')
        return 2 if errs else 0
    output=resolve(root,args.output) if args.output else root/'results'/'c-11b'/datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
    if output.exists():
        raise ValueError(f'Output exists; will not overwrite: {output}')
    output.mkdir(parents=True)
    (output/'training_decision.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    import io
    stream=io.StringIO()
    writer=csv.writer(stream)
    writer.writerow(['check','status','notes'])
    checks=[
      ('c11a_integrity', 'FAIL' if errs else 'PASS', 'C-11a upstream SHA256 and audit'),
      ('promoted_dataset', promoted['status'], 'Byte-identical to four C-10 dataset files'),
      ('exact_normalized_leakage', 'PASS' if not any('LEAKAGE' in b for b in blockers) else 'FAIL', 'C-11a Java/Python function hashes only'),
      ('near_duplicate_leakage', evidence['near_duplicate_test']['status'], 'Not required for scope-limited approval; not automatically cleared'),
      ('project_independence', evidence['project_independence']['status'], 'Not required for scope-limited approval; cannot claim project-disjoint evaluation'),
      ('unverified_negative_risk', 'ACCEPTED_LIMITATION' if approval else 'UNRESOLVED', '13322 of 13584 negatives unreviewed'),
      ('scope_limitation_acknowledgment', 'ACCEPTED' if approval else 'NOT_ACCEPTED', 'Requires explicit reviewer, flags, and written rationale'),
      ('controlled_training_approval', 'APPROVED' if approval else 'NOT_APPROVED', 'Restricted to exploratory controlled training'),
    ]
    writer.writerows(checks)
    (output/'approval_checklist.csv').write_text(stream.getvalue(), encoding='utf-8')
    (output/'scope_policy.json').write_text(json.dumps({
        'version':VERSION, 'policy':'SCOPE_LIMITED_CONTROLLED_TRAINING',
        'scope_limitations':SCOPE_LIMITATIONS,
        'approval_does_not_imply':['near_duplicate_leakage_clearance', 'project_disjoint_evaluation',
                                   'confirmed_negative_ground_truth', 'unrestricted_training_approval'],
        'approved':approval, 'reviewer':args.reviewer if approval else None,
        'rationale':args.rationale if approval else None,
        'dataset_sha256':{name: info.get('sha256_promoted') for name,info in promoted['files'].items()},
    },indent=2)+'\n',encoding='utf-8')
    hashes={str(f):sha256(f) for f in [c11a/'audit.json',c11a/'manifest.json',c11a/'negative_reliability.json',
      c10/'python_mixed_screened_candidate'/'train.txt',c10/'python_mixed_screened_candidate'/'valid.txt']}
    (output/'manifest.json').write_text(json.dumps({'version':VERSION,'created_at_chicago':now,'status':state,
         'training_approval':approval,'input_sha256':hashes,'output_sha256':{
         f:sha256(output/f) for f in ('training_decision.json','approval_checklist.csv','scope_policy.json')},
         'immutable_inputs':True},indent=2)+'\n',encoding='utf-8')
    print('OUTPUT:',output)
    return 2 if errs else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError,OSError,KeyError,TypeError,json.JSONDecodeError) as e:
        print(f'C-11b ERROR: {e}',file=sys.stderr)
        sys.exit(2)
