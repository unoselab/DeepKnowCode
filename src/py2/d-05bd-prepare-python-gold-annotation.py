#!/usr/bin/env python3
"""Prepare blinded independent annotation packets from D-05bc v1, no automatic gold labels.

This program only reads the 24 active C-01-only candidates and source repositories.
"""
import argparse
import ast
import collections
import csv
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v1'
POLICY = 'C01_ONLY_INDEPENDENT_DOUBLE_BLIND_ANNOTATION_PREPARATION_NO_GOLD_ASSIGNMENT'
EXPECTED_RECORDS = 3971


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def sha_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def jsonl(path):
    with path.open('r', encoding='utf-8') as f:
        for no, line in enumerate(f, 1):
            if line.strip():
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError(f'Not an object: {path}:{no}')
                yield data


def function_spans(tree):
    results = []
    stack = [(tree, [])]
    while stack:
        node, parents = stack.pop()
        prefix = parents
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min([node.lineno] + [getattr(x, 'lineno', node.lineno) for x in node.decorator_list])
            results.append((node.name, '.'.join(parents + [node.name]), start, node.end_lineno))
            prefix = parents + [node.name]
        elif isinstance(node, ast.ClassDef):
            prefix = parents + [node.name]
        stack.extend((child, prefix) for child in ast.iter_child_nodes(node))
    return results


class Resolver:
    def __init__(self, root):
        self.root = root
        self.cache = {}

    def get(self, rec, side):
        info = rec['v6_source_resolution'][side]
        if info.get('status') != 'EXACT_AST_FUNCTION':
            raise ValueError(f'Unverified AST for {rec["pair_id"]} {side}')
        relative = info.get('repository_file', '')
        project = rec['project']
        if not isinstance(relative, str) or not relative.startswith('data/repositories/') or '\\' in relative:
            raise ValueError('Unsafe source path')
        repo = (self.root / 'data/repositories' / project).resolve()
        path = (self.root / relative).resolve()
        if not path.is_relative_to(repo) or not path.is_file() or path.suffix != '.py':
            raise ValueError('Source path outside active repo or missing: ' + relative)
        if str(path) not in self.cache:
            blob = path.read_bytes()
            if hashlib.sha256(blob).hexdigest() != info.get('repository_file_sha256'):
                raise ValueError('Repository file SHA mismatch: ' + relative)
            text = blob.decode('utf-8')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', SyntaxWarning)
                tree = ast.parse(text, filename=str(path))
            self.cache[str(path)] = (text.splitlines(keepends=True), function_spans(tree))
        lines, spans = self.cache[str(path)]
        start, end = int(info['start_line']), int(info['end_line'])
        qualname = rec.get(side + '_locator', {}).get('function')
        matches = [t for t in spans if t[1] == qualname and t[2] == start and t[3] == end]
        if len(matches) != 1 or start < 1 or end < start or end > len(lines):
            raise ValueError('Source AST span mismatch: ' + relative)
        code = ''.join(lines[start - 1:end])
        if sha_text(code) != info.get('code_sha256'):
            raise ValueError('Source code SHA mismatch: ' + relative)
        return code, info


def source_dir(root, explicit):
    if explicit:
        return explicit.resolve()
    folders = sorted((root / 'results/d-05b').glob('python-near-duplicate-audit-v1-*'))
    if not folders:
        raise ValueError('Missing D-05bc v1 run; supply --source')
    return folders[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--source', type=Path, help='D-05bc v1 audit output directory')
    p.add_argument('--output', type=Path, help='New directory under results/d-05b (must not exist)')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    source = source_dir(root, args.source)
    stamp = datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
    out = (args.output or (root / 'results/d-05b' / ('python-gold-annotation-preparation-v1-' + stamp))).resolve()
    if not source.is_dir() or out.exists() or source == out or source in out.parents or out in source.parents:
        raise ValueError('Missing source, existing output, or overlapping input/output paths')
    mf = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if mf.get('policy') != 'C01_ONLY_NEAR_DUPLICATE_REPORT_ONLY_NO_EXCLUSION':
        raise ValueError('Requires D-05bc C01_ONLY report-only source')
    for name in ('candidates_near_duplicate_audited.jsonl', 'near_duplicate_audit.json'):
        if sha_file(source/name) != mf.get('artifacts_sha256', {}).get(name):
            raise ValueError('D-05bc source SHA256 mismatch: ' + name)
    previous = json.loads((source/'near_duplicate_audit.json').read_text(encoding='utf-8'))
    if previous.get('verified_gold_labels') != 0 or previous.get('candidates_excluded_due_to_near_duplicate') != 0:
        raise ValueError('D-05bc provenance/status mismatch')
    data_path = source / 'candidates_near_duplicate_audited.jsonl'
    resolver = Resolver(root)
    projects = collections.Counter()
    methods = collections.Counter()
    seen = set()
    packets = []
    near = 0
    for rec in jsonl(data_path):
        pair_id = rec.get('pair_id')
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
            raise ValueError('Invalid/duplicate pair_id')
        seen.add(pair_id)
        if rec.get('origin') != 'c01_hybrid_detector' or rec.get('legacy_prediction') is not None:
            raise ValueError('Legacy record forbidden in annotation inputs')
        if rec.get('gold_label') is not None or rec.get('evaluation_eligible') is not False:
            raise ValueError('Gold or eligible record unexpectedly present')
        if rec.get('code_hash_train_overlap_status') != 'NO_MATCH_AGAINST_M2_M3_ACTIVE_PYTHON_TRAIN_VALID_HASHES':
            raise ValueError('Non-heldout code record')
        if rec.get('near_duplicate_leakage_status') not in ('AUDITED_REPORT_ONLY_HIGH_SIMILARITY', 'AUDITED_REPORT_ONLY_BELOW_THRESHOLD'):
            raise ValueError('Missing near duplicate report-only status')
        left_code, left_info = resolver.get(rec, 'left')
        right_code, right_info = resolver.get(rec, 'right')
        project = rec['project']
        projects[project] += 1
        methods[rec.get('detection_source', 'unknown')] += 1
        near += int(bool(rec.get('d05bc_similarity', {}).get('flagged_report_only')))
        # Reviewers see code, not classifier predictions, detection source, similarity flags or labels.
        packets.append({
            'pair_id': pair_id, 'project': project,
            'left': {'code': left_code, 'path': left_info['repository_file'], 'function': left_info['ast_qualname'],
                     'start_line': left_info['start_line'], 'end_line': left_info['end_line'], 'code_sha256': left_info['code_sha256']},
            'right': {'code': right_code, 'path': right_info['repository_file'], 'function': right_info['ast_qualname'],
                      'start_line': right_info['start_line'], 'end_line': right_info['end_line'], 'code_sha256': right_info['code_sha256']},
        })
    if len(packets) != EXPECTED_RECORDS or len(projects) != 24 or projects.get('scikit-learn', 0) == 0 or 'azure-sdk-for-python' in projects:
        raise ValueError(f'Expected 3971 C-01-only records across 24 active projects; got {len(packets)}, {len(projects)}')
    if near != previous.get('counts', {}).get('candidate_near_duplicate_flagged'):
        raise ValueError('Near duplicate count differs from source audit')
    packets.sort(key=lambda r: (hashlib.sha256(r['pair_id'].encode('utf-8')).hexdigest(), r['pair_id']))
    summary = {'status': 'D05BD_V1_ANNOTATION_PREPARED_GOLD_LABELS_NOT_VERIFIED',
               'version': VERSION, 'policy': POLICY,
               'created_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
               'source': str(source), 'source_candidates_sha256': sha_file(data_path),
               'counts': {'annotation_packets': len(packets), 'active_projects': len(projects),
                          'near_duplicate_flagged_report_only': near, 'excluded_due_to_similarity': 0,
                          'verified_gold_labels': 0, 'reviewers_required': 2},
               'project_counts': dict(sorted(projects.items())), 'detection_source_counts_hidden_from_reviewers': dict(sorted(methods.items())),
               'annotation_protocol': {'labels': {'0': 'not a clone', '1': 'clone', 'uncertain': 'needs adjudication'},
                   'reviewers': 'A and B independently label each pair without detector result or near-duplicate scores',
                   'adjudication': 'disagreements and uncertain judgments require resolution; never auto-label',
                   'evaluation': 'no evaluation_eligible or gold_label can be set by this preparation step',
                   'negative_pairs': 'do not assume non-detected pairs are negatives; C-01-only samples may be positive-skewed'},
               'scope': 'C-01 detected-pair candidate annotation only; no representative negative sampling yet'}
    if args.dry_run:
        print(json.dumps({'status':'DRY_RUN', **summary['counts']}, sort_keys=True))
        return 0
    out.mkdir(parents=True, exist_ok=False)
    packet_path = out / 'annotation_packets.jsonl'
    with packet_path.open('w', encoding='utf-8') as f:
        for pkt in packets:
            f.write(json.dumps(pkt, ensure_ascii=False, sort_keys=True) + '\n')
    cols = ['pair_id', 'project', 'clone_label', 'confidence', 'notes']
    for reviewer in ('A', 'B'):
        with (out / f'reviewer_{reviewer.lower()}_template.csv').open('w', encoding='utf-8', newline='') as f:
            w=csv.DictWriter(f, fieldnames=cols);w.writeheader()
            for pkt in packets:
                w.writerow({'pair_id':pkt['pair_id'],'project':pkt['project'], 'clone_label':'', 'confidence':'', 'notes':''})
    with (out/'adjudication_template.csv').open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f,fieldnames=['pair_id','project','reviewer_a_label','reviewer_b_label','resolved_gold_label','resolution_notes','adjudicator']);w.writeheader()
        for pkt in packets:
            w.writerow({'pair_id':pkt['pair_id'],'project':pkt['project']})
    (out/'annotation_preparation_audit.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)+'\n', encoding='utf-8')
    artifacts = ['annotation_packets.jsonl','reviewer_a_template.csv','reviewer_b_template.csv','adjudication_template.csv','annotation_preparation_audit.json']
    manifest = {'version':VERSION, 'created_at_chicago':summary['created_at_chicago'],
                'input_sha256':sha_file(data_path), 'artifacts_sha256':{n:sha_file(out/n) for n in artifacts}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'out':str(out), **summary['counts']}, sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, UnicodeError, SyntaxError) as e:
        print('D-05bd ERROR: ' + str(e), file=sys.stderr)
        sys.exit(2)
