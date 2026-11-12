#!/usr/bin/env python3
"""Read-only review of existing F-01c structural truncation proposals.

Does not certify compilation, semantics, labels, or authorization for training.
All configuration comes from required shell-passed CLI arguments.
"""
import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path


def sha256(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def load_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_no}: {exc}') from exc


def assess(proposals, audit, token_limit, max_removed_for_low_risk, max_reduction_for_low_risk):
    records = audit['candidate_records']
    by_key = {(r['id1'], r['id2'], r['label']): r for r in records}
    if len(by_key) != len(records):
        raise ValueError('Duplicate pair IDs in audit')
    rows = []
    for p in proposals:
        key = (p['id1'], p['id2'], p['label'])
        if key not in by_key:
            raise ValueError(f'Proposal not found in audit: {key}')
        a = by_key[key]
        a_before, a_after = p['code_a_before'], p['code_a_after']
        b_before, b_after = p['code_b_before'], p['code_b_after']
        a_unchanged = a_before == a_after
        b_unchanged = b_before == b_after
        any_changed = not (a_unchanged and b_unchanged)
        exactly_one_changed = a_unchanged != b_unchanged
        original_count, changed_count = p['original_tokens'], p['transformed_tokens']
        checks = {
            'any_side_changed': any_changed,
            'original_token_count_matches_audit': original_count == a['original_tokens'],
            'transformed_token_count_matches_audit': changed_count == a.get('transformed_tokens'),
            'token_limit_from_prior_audit': changed_count <= token_limit,
            'reduction_positive': original_count > changed_count,
            'original_code_hashes_match_audit': sha256(a_before) == a['original_code_sha256']['id1'] and sha256(b_before) == a['original_code_sha256']['id2'],
            'processed_code_hashes_match_audit': sha256(a_after) == a.get('processed_code_sha256', {}).get('id1') and sha256(b_after) == a.get('processed_code_sha256', {}).get('id2'),
            'removed_statement_count_matches_audit': len(p['removed_statements']) == a.get('removed_count'),
            'negative_label_unchanged': p['label'] == 0,
            'not_approved_for_training': p.get('approved_for_training') is False and a.get('approved_for_training') is False,
        }
        passed = all(checks.values())
        changed_side = 'A+B' if not a_unchanged and not b_unchanged else ('A' if not a_unchanged else 'B' if not b_unchanged else 'INVALID')
        reduction = original_count - changed_count
        large_edit = len(p['removed_statements']) > max_removed_for_low_risk or reduction > max_reduction_for_low_risk
        # This is a *review-priority* heuristic, never a semantic safety assertion.
        priority = 'EXTENSIVE_EDIT_REVIEW' if large_edit else 'FOCUSED_EDIT_REVIEW'
        if not passed:
            priority = 'INTEGRITY_FAILURE'
        row = {
            'id1': p['id1'], 'id2': p['id2'], 'label': p['label'],
            'changed_side': changed_side,
            'unchanged_side': ('B' if changed_side == 'A' else 'A') if exactly_one_changed else None,
            'original_tokens': original_count, 'transformed_tokens': changed_count,
            'token_reduction': reduction, 'removed_count': len(p['removed_statements']),
            'strategy': p['strategy'], 'checks': checks, 'integrity_pass': passed,
            'review_priority': priority,
            'parser_verified_by_prior_audit': bool(a.get('status') == 'STRUCTURAL_PROPOSAL_MANUAL_REVIEW_REQUIRED'),
            'semantic_equivalence_verified': False, 'negative_label_validity_verified': False,
            'compile_verified': False, 'approved_for_training': False,
        }
        rows.append(row)
    rows.sort(key=lambda r: (r['original_tokens'], r['id1'], r['id2']))
    summary = {
        'status': 'REVIEW_ONLY_INTEGRITY_PASS' if rows and all(r['integrity_pass'] for r in rows) else 'REVIEW_INTEGRITY_FAILURE',
        'proposals_reviewed': len(rows),
        'one_sided_count': sum(r['changed_side'] in ('A','B') for r in rows),
        'two_sided_count': sum(r['changed_side']=='A+B' for r in rows),
        'integrity_pass_count': sum(r['integrity_pass'] for r in rows),
        'focused_edit_count': sum(r['review_priority']=='FOCUSED_EDIT_REVIEW' for r in rows),
        'extensive_edit_count': sum(r['review_priority']=='EXTENSIVE_EDIT_REVIEW' for r in rows),
        'approved_for_training': False,
        'caveats': ['Token lengths and parse status are cross-checked against the prior server audit, not independently recomputed.',
                    'Removing registration, rendering, deserialization or switch branches may change behavior.',
                    'Review priority is not a semantic safety grade.',
                    'Cross-pair endpoint uniqueness and train/validation isolation are not certified.']}
    return summary, rows


def make_console(proposals, rows, pair_filter):
    row_by_key = {(r['id1'], r['id2']):r for r in rows}
    chunks=[]
    matches=0
    for p in proposals:
        if pair_filter != 'all' and pair_filter not in (p['id1'],p['id2']):
            continue
        matches+=1
        r=row_by_key[(p['id1'],p['id2'])]
        chunks.append('=' * 100)
        chunks.append(f"PAIR: {p['id1']} / {p['id2']} | original_label={p['label']} | review={r['review_priority']}")
        chunks.append(f"TOKENS: {r['original_tokens']} -> {r['transformed_tokens']} (-{r['token_reduction']}); CHANGED={r['changed_side']}; REMOVED={r['removed_count']}")
        chunks.append('INTEGRITY CHECK: '+('PASS' if r['integrity_pass'] else 'FAIL'))
        for side in ('a','b'):
            before=p[f'code_{side}_before'];after=p[f'code_{side}_after']
            chunks.extend(['',f'CODE {side.upper()} BEFORE:', before, f'CODE {side.upper()} AFTER:',after])
            chunks.append(f'CODE {side.upper()} DIFF:')
            diff=''.join(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),fromfile='BEFORE',tofile='AFTER'))
            chunks.append(diff if diff else '(UNCHANGED)')
        chunks.extend(['TRAINING APPROVAL: NO; HUMAN SEMANTIC/LABEL REVIEW REQUIRED',''])
    if not matches:
        raise ValueError(f'No proposal matches pair filter: {pair_filter}')
    return '\n'.join(chunks)+'\n'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('proposals','audit','output-dir','pair-filter','log-timestamp','version'):
        ap.add_argument('--'+key, required=True)
    for key in ('max-seq-length','max-removed-for-focused-review','max-token-reduction-for-focused-review'):
        ap.add_argument('--'+key, type=int, required=True)
    args=ap.parse_args()
    if args.max_seq_length <= 0 or args.max_removed_for_focused_review < 0 or args.max_token_reduction_for_focused_review < 0:
        raise ValueError('Invalid review limit')
    proposals=list(load_jsonl(Path(args.proposals)))
    with Path(args.audit).open(encoding='utf-8') as f: audit=json.load(f)
    summary, rows=assess(proposals,audit,args.max_seq_length,args.max_removed_for_focused_review,args.max_token_reduction_for_focused_review)
    summary.update({'version':args.version,'chicago_log_timestamp':args.log_timestamp,'pair_filter':args.pair_filter})
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=False)
    (output/'truncation_review.json').write_text(json.dumps({'summary':summary,'records':rows},indent=2)+'\n',encoding='utf-8')
    rendered=make_console(proposals,rows,args.pair_filter)
    (output/'before_after_console.txt').write_text(rendered,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
    print(rendered)
    if summary['status']=='REVIEW_INTEGRITY_FAILURE':
        raise RuntimeError('Truncation review integrity failure: do not use proposals')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise
