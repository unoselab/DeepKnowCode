#!/usr/bin/env python3
"""Synchronize only the Python evaluation metadata to a verified held-out JSONL."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze_dataset(path, args):
    require(path.is_file(), f'Ground truth not found: {path}')
    counts = Counter()
    projects = set()
    records = 0
    with path.open('r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            require(line.strip(), f'Blank JSONL line {lineno}')
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'Invalid JSON line {lineno}: {e}') from e
            require(isinstance(obj, dict), f'Line {lineno} is not JSON object')
            require(args.label_field in obj, f'Missing {args.label_field}: line {lineno}')
            label = obj[args.label_field]
            require(type(label) is int and label in (0, 1), f'Invalid binary label at line {lineno}: {label!r}')
            require(isinstance(obj.get(args.project_field), str) and obj[args.project_field].strip(),
                    f'Missing project: line {lineno}')
            counts[label] += 1
            projects.add(obj[args.project_field].strip())
            records += 1
    require(records == args.expected_pairs, f'Pair count {records} != {args.expected_pairs}')
    require(counts[1] == args.expected_positive, f'Positive {counts[1]} != {args.expected_positive}')
    require(counts[0] == args.expected_negative, f'Negative {counts[0]} != {args.expected_negative}')
    require(len(projects) == args.expected_projects, f'Projects {len(projects)} != {args.expected_projects}')
    actual_sha = digest(path)
    if args.expected_sha256 != 'AUTO':
        require(actual_sha == args.expected_sha256.lower(),
                f'Ground truth SHA256 mismatch: actual {actual_sha}, expected {args.expected_sha256}')
    return dict(pairs=records, positive_pairs=counts[1], negative_pairs=counts[0],
                projects=len(projects), dataset_sha256=actual_sha)


def load_manifests(paths):
    docs = []
    for path in paths:
        require(path.is_file(), f'Manifest not found: {path}')
        doc = json.loads(path.read_text(encoding='utf-8'))
        require(isinstance(doc.get('languages'), dict), f'Missing languages: {path}')
        require(isinstance(doc['languages'].get('java'), dict), f'Missing java: {path}')
        require(isinstance(doc['languages'].get('python'), dict), f'Missing python: {path}')
        docs.append(doc)
    return docs


def updated(doc, stats, args):
    new = json.loads(json.dumps(doc))
    old = new['languages']['python']
    if 'previous_snapshot' in old:
        previous = old['previous_snapshot']
        require(isinstance(previous, dict), 'Invalid previous_snapshot')
    else:
        legacy_fields = ('projects', 'code_ids', 'code_corpora', 'source_lines_scanned', 'pairs',
                         'positive_pairs', 'negative_pairs')
        previous = {k: old[k] for k in legacy_fields if k in old}
    result = {k: v for k, v in old.items() if k not in (
        'projects', 'pairs', 'positive_pairs', 'negative_pairs', 'code_ids',
        'code_corpora', 'source_lines_scanned', 'previous_snapshot', 'dataset_path',
        'dataset_sha256', 'label_field', 'dataset_provenance', 'reference_document',
        'evaluation_file', 'ground_truth_status')}
    result.update({
        'projects': stats['projects'], 'pairs': stats['pairs'],
        'positive_pairs': stats['positive_pairs'], 'negative_pairs': stats['negative_pairs'],
        'dataset_path': args.dataset_relative_path, 'dataset_sha256': stats['dataset_sha256'],
        'label_field': args.label_field, 'dataset_provenance': args.provenance,
        'reference_document': args.reference_document,
        'ground_truth_status': 'CURRENT_HELD_OUT_EVALUATION_GROUND_TRUTH',
        'previous_snapshot': previous,
    })
    new['languages']['python'] = result
    require(new['languages']['java'] == doc['languages']['java'], 'Java metadata changed unexpectedly')
    require(new.get('source_provenance') == doc.get('source_provenance'), 'Source provenance changed')
    return new


def write_new_file(path, data):
    # Create a temporary file in the same filesystem and sync before atomic replace.
    fd, temp = tempfile.mkstemp(prefix='.python-manifest-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(temp, path.stat().st_mode & 0o777)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def run(args):
    data = Path(args.dataset).resolve(strict=True)
    paths = [Path(args.manifest).resolve(strict=True), Path(args.manifest_details).resolve(strict=True)]
    require(paths[0] != paths[1], 'Manifest files must differ')
    stats = analyze_dataset(data, args)
    docs = load_manifests(paths)
    require(docs[0]['languages']['python'] == docs[1]['languages']['python'],
            'Manifests have different sync states for Python; manual inspection required')
    before_java = [json.dumps(doc['languages']['java'], sort_keys=True) for doc in docs]
    outputs = [updated(doc, stats, args) for doc in docs]
    new_bytes = [(json.dumps(doc, indent=2, ensure_ascii=False) + '\n').encode('utf-8') for doc in outputs]
    originals = [path.read_bytes() for path in paths]
    changed = [a != b for a, b in zip(originals, new_bytes)]
    require(changed[0] == changed[1], 'Manifests are in different sync states; manual inspection required')
    info = {
        'status': 'H10_PYTHON_MANIFEST_PREFLIGHT_PASS' if args.mode == 'preflight' else 'H10_PYTHON_MANIFEST_APPLY_PASS',
        'mode': args.mode, 'version': args.version, 'chicago_time': datetime.now(ZoneInfo(args.timezone)).isoformat(),
        'source_of_truth': args.reference_document, 'ground_truth': str(data),
        'observed': stats, 'manifests': [str(p) for p in paths],
        'manifest_before_sha256': [hashlib.sha256(b).hexdigest() for b in originals],
        'would_change': changed, 'java_unchanged': True, 'code_id_count': 'NOT_REVALIDATED',
        'previous_snapshot': [outputs[i]['languages']['python']['previous_snapshot'] for i in range(2)]}
    if args.mode == 'apply':
        require(any(changed), 'Already synchronized: refusing repeat apply')
        backup_dir = Path(args.backup_dir).resolve()
        audit_dir = Path(args.audit_dir).resolve()
        backup_dir.mkdir(parents=True, exist_ok=True)
        audit_dir.mkdir(parents=True, exist_ok=True)
        tag = datetime.now(ZoneInfo(args.timezone)).strftime('%Y%m%d-%H%M%S-%f') + f'-{os.getpid()}'
        backups = [backup_dir / f'{p.name}.before-python-sync-{tag}' for p in paths]
        audit = audit_dir / f'python-manifest-sync-{tag}.json'
        require(all(not p.exists() for p in backups + [audit]), 'Output collision')
        # Exclusive backups before touching the originals.
        for path, backup, original in zip(paths, backups, originals):
            with backup.open('xb') as f:
                f.write(original)
                f.flush()
                os.fsync(f.fileno())
        try:
            for path, b in zip(paths, new_bytes):
                write_new_file(path, b)
            after = load_manifests(paths)
            require(all(json.dumps(d['languages']['java'], sort_keys=True) == j for d, j in zip(after, before_java)),
                    'Java metadata changed')
            require(all(d['languages']['python']['pairs'] == stats['pairs'] for d in after),
                    'Post-write Python count mismatch')
        except BaseException:
            for path, original in zip(paths, originals):
                write_new_file(path, original)
            raise
        info['backups'] = [str(b) for b in backups]
        info['after_manifest_sha256'] = [digest(p) for p in paths]
        info['audit_file'] = str(audit)
        with audit.open('x', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
            f.write('\n')
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print('H10_PYTHON_MANIFEST_PREFLIGHT_PASS_NO_CHANGES' if args.mode == 'preflight' else 'H10_PYTHON_MANIFEST_APPLY_PASS')


def main():
    parser = argparse.ArgumentParser()
    for arg in ('mode', 'version', 'dataset', 'manifest', 'manifest-details', 'backup-dir',
                'audit-dir', 'expected-pairs', 'expected-positive', 'expected-negative',
                'expected-projects', 'expected-sha256', 'dataset-relative-path',
                'label-field', 'project-field', 'provenance', 'reference-document', 'timezone'):
        parser.add_argument('--' + arg, required=True,
                            type=int if arg in ('expected-pairs', 'expected-positive', 'expected-negative', 'expected-projects') else str)
    args = parser.parse_args()
    require(args.mode in ('preflight', 'apply'), 'Invalid mode')
    require(args.expected_positive + args.expected_negative == args.expected_pairs, 'Count configuration inconsistent')
    require(args.version, 'Version missing')
    run(args)


if __name__ == '__main__':
    main()
