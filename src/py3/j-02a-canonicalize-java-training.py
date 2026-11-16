#!/usr/bin/env python3
"""J-02a v2: enrich Java training/validation JSONL without changing existing records.

Source code identity: java-sha256:sha256(func.encode('utf-8')), no normalization.
Input and experiment configuration are mandatory CLI parameters from the Shell wrapper.
No leakage or held-out evaluation analysis is performed by J-02a.
"""
import argparse
import collections
import hashlib
import itertools
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

EXPERIMENT = 'j-02a'
VERSION = 'v2'
CORPORA = ('java_source', 'java_mixed', 'python_mixed')
BASE_FIELDS = frozenset(('corpus', 'id1', 'id2', 'label', 'tokens'))
JAVA_FIELDS = BASE_FIELDS | frozenset(('left', 'right'))
FUNCTION_PATTERN = re.compile(r'^py-sha256:[0-9a-f]{64}$')


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(piece)
    return digest.hexdigest()


def jsonl(path):
    with path.open('r', encoding='utf-8', newline='') as handle:
        for lineno, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ValueError(f'Blank JSONL record: {path}:{lineno}')
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSONL at {path}:{lineno}: {exc}') from exc
            if not isinstance(item, dict):
                raise ValueError(f'Expected object: {path}:{lineno}')
            yield lineno, item, raw


def index_source(path, corpus):
    if not path.is_file():
        raise ValueError(f'Missing Java {corpus} source: {path}')
    index = {}
    for lineno, item, _ in jsonl(path):
        ident = item.get('idx')
        code = item.get('func')
        if not isinstance(ident, str) or not ident:
            raise ValueError(f'Invalid idx at {path}:{lineno}')
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f'Invalid func at {path}:{lineno} idx={ident!r}')
        if ident in index:
            raise ValueError(f'Duplicate idx in {corpus}: {ident!r} at {path}:{lineno}')
        index[ident] = 'java-sha256:' + hashlib.sha256(code.encode('utf-8')).hexdigest()
    if not index:
        raise ValueError(f'Empty Java source: {path}')
    return index


def java_metadata(corpus, ident, index):
    if ident not in index:
        raise ValueError(f'Unmapped Java Function ID: corpus={corpus!r}, id={ident!r}')
    # Unknown provenance must be null; do not infer repository/location/revision from IDs.
    return dict(language='java', function_id=index[ident], source_id=ident,
                source_dataset=corpus, repository=None, revision=None,
                path=None, start_line=None, end_line=None)


def validate_base(item, path, lineno):
    if not BASE_FIELDS <= item.keys():
        raise ValueError(f'Missing base fields {path}:{lineno}: {sorted(BASE_FIELDS - item.keys())}')
    corpus = item['corpus']
    if corpus not in CORPORA:
        raise ValueError(f'Unknown corpus at {path}:{lineno}: {corpus!r}')
    if not all(isinstance(item[field], str) and item[field] for field in ('id1', 'id2')):
        raise ValueError(f'Invalid id1/id2 at {path}:{lineno}')
    if type(item['label']) is not int or item['label'] not in (0, 1):
        raise ValueError(f'Invalid label at {path}:{lineno}')
    if type(item['tokens']) is not int or item['tokens'] <= 0:
        raise ValueError(f'Invalid tokens at {path}:{lineno}')
    return corpus


def expected_counts(args, split):
    if split == 'train':
        return dict(total=args.expected_train_total,
                    java_source=args.expected_train_java_source,
                    java_mixed=args.expected_train_java_mixed,
                    python_mixed=args.expected_train_python_mixed)
    return dict(total=args.expected_validation_total,
                java_source=args.expected_validation_java_source,
                java_mixed=args.expected_validation_java_mixed,
                python_mixed=args.expected_validation_python_mixed)


def check_expected(counts, expected, split):
    for key, value in expected.items():
        if counts[key] != value:
            raise ValueError(f'{split} {key} count mismatch: actual={counts[key]}, expected={value}')


def inspect_prepared(path, split, indexes, args):
    if not path.is_file():
        raise ValueError(f'Missing canonical v2 {split}: {path}')
    counts = collections.Counter()
    source_ids = {corpus: set() for corpus in ('java_source', 'java_mixed')}
    for lineno, item, _ in jsonl(path):
        corpus = validate_base(item, path, lineno)
        counts['total'] += 1
        counts[corpus] += 1
        if corpus in indexes:
            if set(item) != BASE_FIELDS:
                raise ValueError(f'Java record must have only base fields in canonical v2: {path}:{lineno}')
            for key in ('id1', 'id2'):
                ident = item[key]
                if ident not in indexes[corpus]:
                    raise ValueError(f'Unmapped Java ID at {path}:{lineno}: {corpus}:{ident}')
                source_ids[corpus].add(ident)
        else:
            if set(item) != JAVA_FIELDS:
                raise ValueError(f'Expected enriched Python canonical v2 record: {path}:{lineno}')
            for side, field in (('left', 'id1'), ('right', 'id2')):
                meta = item[side]
                if not isinstance(meta, dict) or meta.get('language') != 'python' or meta.get('source_id') != item[field] or not isinstance(meta.get('function_id'), str) or not FUNCTION_PATTERN.fullmatch(meta['function_id']):
                    raise ValueError(f'Invalid Python canonical metadata: {path}:{lineno} {side}')
    check_expected(counts, expected_counts(args, split), split)
    return dict(records=dict(counts), java_unique_source_ids={k: len(v) for k,v in source_ids.items()})


def enriched_line(item, raw, indexes):
    corpus = item['corpus']
    if corpus == 'python_mixed':
        return raw  # Preserve Python canonical records byte-for-byte.
    extended = dict(item)
    extended['left'] = java_metadata(corpus, item['id1'], indexes[corpus])
    extended['right'] = java_metadata(corpus, item['id2'], indexes[corpus])
    return json.dumps(extended, ensure_ascii=False, separators=(',', ':')) + '\n'


def verify_pair(base_path, converted_path, indexes):
    if not converted_path.is_file():
        raise ValueError(f'Missing output split: {converted_path}')
    totals = collections.Counter()
    for base, derived in itertools.zip_longest(jsonl(base_path), jsonl(converted_path)):
        if base is None or derived is None:
            raise ValueError(f'Row count differs: {base_path} vs {converted_path}')
        lineno, original, raw_before = base
        _, enriched, raw_after = derived
        corpus = original['corpus']
        if corpus == 'python_mixed':
            if raw_before != raw_after or original != enriched:
                raise ValueError(f'Python record changed at {base_path}:{lineno}')
            totals['python_rows_exactly_preserved'] += 1
        else:
            if set(enriched) != JAVA_FIELDS:
                raise ValueError(f'Unexpected Java output fields at {converted_path}:{lineno}')
            if {key: enriched[key] for key in BASE_FIELDS} != original:
                raise ValueError(f'Java original fields changed at {converted_path}:{lineno}')
            for side, field in (('left', 'id1'), ('right', 'id2')):
                expected = java_metadata(corpus, original[field], indexes[corpus])
                if enriched[side] != expected:
                    raise ValueError(f'Incorrect Java {side} metadata at {converted_path}:{lineno}')
            totals['java_rows_enriched'] += 1
        totals['rows_compared'] += 1
    return dict(rows_compared=totals['rows_compared'],
                java_rows_enriched=totals['java_rows_enriched'],
                python_rows_exactly_preserved=totals['python_rows_exactly_preserved'],
                original_fields_equal=True,
                row_order_preserved=True,
                only_java_metadata_added=True,
                status='IDENTICAL_BASE_RECORDS')


def check_paths(args):
    inputs = (args.java_source, args.java_mixed, args.train, args.validation)
    if len(set(inputs)) != len(inputs):
        raise ValueError('Input file paths must be distinct')
    if args.train == args.validation:
        raise ValueError('Train and validation must be different files')
    if args.output in inputs or any(args.output in p.parents or p in args.output.parents for p in inputs):
        raise ValueError('Output overlaps with an input path')
    if args.mode == 'build' and args.output.exists():
        raise ValueError(f'Output already exists; refusing overwrite: {args.output}')
    if args.mode == 'verify' and not args.output.is_dir():
        raise ValueError(f'Output missing for --verify: {args.output}')
    if args.report in inputs or args.report == args.output or args.report in args.output.parents:
        raise ValueError('Report path conflicts with dataset inputs/output')
    if args.report.is_relative_to(args.output):
        raise ValueError('Report must be outside the new dataset output directory')
    for split in ('train', 'validation'):
        exp = expected_counts(args, split)
        if exp['total'] != sum(exp[k] for k in CORPORA):
            raise ValueError(f'Inconsistent expected {split} totals: {exp}')
        if any(type(v) is not int or v <= 0 for v in exp.values()):
            raise ValueError(f'Expected {split} counts must be positive integers')


def export_split(source, target, indexes):
    with target.open('x', encoding='utf-8', newline='') as handle:
        for _, item, raw in jsonl(source):
            handle.write(enriched_line(item, raw, indexes))


def run(args):
    check_paths(args)
    for p in (args.java_source, args.java_mixed, args.train, args.validation):
        if not p.is_file():
            raise ValueError(f'Missing input: {p}')
    indexes = {'java_source': index_source(args.java_source, 'java_source'),
               'java_mixed': index_source(args.java_mixed, 'java_mixed')}
    summary = {'train': inspect_prepared(args.train, 'train', indexes, args),
               'validation': inspect_prepared(args.validation, 'validation', indexes, args)}
    report = dict(experiment=EXPERIMENT, version=VERSION, mode=args.mode,
                  timestamp_chicago=datetime.now(ZoneInfo('America/Chicago')).isoformat(),
                  status='INSPECT_PASS',
                  scope='JAVA_CANONICAL_FORMAT_AND_ORIGINAL_RECORD_INTEGRITY_ONLY',
                  leakage_audit_performed=False,
                  hash_convention='java-sha256:sha256(func.encode(utf-8)); no normalization',
                  input_files={'java_source': str(args.java_source), 'java_mixed': str(args.java_mixed),
                               'canonical_v2_train': str(args.train), 'canonical_v2_validation': str(args.validation)},
                  input_sha256={key: sha256_file(path) for key, path in {
                      'java_source': args.java_source, 'java_mixed': args.java_mixed,
                      'train': args.train, 'validation': args.validation}.items()},
                  source_records={k: len(v) for k, v in indexes.items()},
                  split_summary=summary,
                  output_directory=str(args.output),
                  provenance_policy='Unknown provenance preserved as null; no inferred repository/path/revision',
                  python_policy='Existing canonical v2 Python rows preserved byte-for-byte',
                  original_dataset_modified=False)
    if args.mode == 'build':
        args.output.mkdir(parents=True, exist_ok=False)
        export_split(args.train, args.output / 'train.jsonl', indexes)
        export_split(args.validation, args.output / 'validation.jsonl', indexes)
    if args.mode in ('build', 'verify'):
        report['paired_file_integrity'] = {
            split: verify_pair(getattr(args, split), args.output / f'{split}.jsonl', indexes)
            for split in ('train', 'validation')}
        report['output_sha256'] = {
            split: sha256_file(args.output / f'{split}.jsonl') for split in ('train', 'validation')}
        report['status'] = 'BUILT_AND_VERIFIED' if args.mode == 'build' else 'VERIFY_PASS'
    if args.report.exists():
        raise ValueError(f'Report already exists; refusing overwrite: {args.report}')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'mode': args.mode,
                      'report': str(args.report), 'output': str(args.output),
                      'split_summary': summary,
                      'paired_file_integrity': report.get('paired_file_integrity'),
                      'leakage_audit_performed': False}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mode', 'java-source', 'java-mixed', 'train', 'validation', 'output', 'report'):
        parser.add_argument('--' + name, required=True)
    for split in ('train', 'validation'):
        for corpus in ('total',) + CORPORA:
            parser.add_argument(f'--expected-{split}-{corpus.replace("_", "-")}', required=True, type=int)
    args = parser.parse_args()
    if args.mode not in ('inspect', 'build', 'verify'):
        parser.error('--mode must be inspect, build, or verify')
    for field in ('java_source', 'java_mixed', 'train', 'validation', 'output', 'report'):
        setattr(args, field, Path(getattr(args, field)).resolve())
    try:
        run(args)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(f'J02A_ERROR:{type(exc).__name__}:{exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
