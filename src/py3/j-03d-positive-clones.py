#!/usr/bin/env python3
"""J-03d v6: immutable positive export/verification and H-07/H-08 AST type audit.

All experiment configuration is mandatory and supplied by the standalone Shell.
No input mutation, no negative labels, no train/validation/test partitioning.
"""
import argparse
import csv
from collections import Counter
from decimal import Decimal, InvalidOperation
import functools
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PAIR_FIELDS = (
    'pair_id', 'fragment_a_id', 'repository_a', 'file_a', 'start_line_a',
    'end_line_a', 'commit_a', 'source_sha256_a', 'fragment_b_id',
    'repository_b', 'file_b', 'start_line_b', 'end_line_b', 'commit_b',
    'source_sha256_b', 'same_repository', 'nicad_similarity_raw',
    'nicad_nlines', 'detector', 'config', 'xml_sha256',
)
FRAGMENT_FIELDS = (
    'fragment_id', 'repository', 'file_path', 'start_line', 'end_line',
    'commit', 'source_sha256', 'language',
)
NAME = re.compile(r'^[a-z0-9][a-z0-9_.-]*__[a-z0-9][a-z0-9_.-]*$')
HEX64 = re.compile(r'^[0-9a-f]{64}$')


def fail(message):
    raise RuntimeError(message)


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json_new(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write('\n')


def read_json(path):
    with path.open('r', encoding='utf-8') as stream:
        return json.load(stream)


def timestamp(args):
    return datetime.now(ZoneInfo(args.time_zone)).isoformat(timespec='seconds')


def valid_reljava(value):
    if not value or '\\' in value or '\x00' in value:
        fail(f'INVALID_JAVA_PATH {value!r}')
    path = Path(value)
    if (path.is_absolute() or '..' in path.parts or '.' in path.parts
            or path.suffix != '.java' or path.as_posix() != value):
        fail(f'INVALID_JAVA_PATH {value!r}')
    return path


def strict_source_file(source_root, rel):
    path = source_root / valid_reljava(rel)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(source_root.resolve(strict=True)):
        fail(f'SOURCE_PATH_ESCAPE path={path}')
    if not path.is_file() or path.is_symlink():
        fail(f'SOURCE_NOT_REGULAR path={path}')
    return path


@functools.lru_cache(maxsize=8)
def source_lines(source_path):
    # Match J-03c source hash definition exactly: binary splitlines(keepends=True).
    return Path(source_path).read_bytes().splitlines(keepends=True)


def expected_value(record, field, expected, context):
    if record.get(field) != expected:
        fail(f'{context}_MISMATCH field={field} expected={expected!r} actual={record.get(field)!r}')


def validate_input(args, subject):
    archived = args.input_root / subject
    source = args.source_root / subject
    if not archived.is_dir() or archived.is_symlink():
        fail(f'AFTER_ARCHIVE_REQUIRED path={archived}')
    if not source.is_dir() or source.is_symlink():
        fail(f'PREPROCESSED_SOURCE_REQUIRED path={source}')
    for name in ('clone_pairs.csv', 'clone_fragments.csv', 'archive_manifest.json',
                 'comparison.json', 'run_summary.json', 'preproc_manifest.json'):
        if not (archived / name).is_file() or (archived / name).is_symlink():
            fail(f'AFTER_FILE_REQUIRED path={archived / name}')
    manifest = read_json(archived / 'archive_manifest.json')
    for key, value in (('experiment', args.expected_input_experiment),
                       ('version', args.expected_input_version),
                       ('phase', args.expected_input_phase),
                       ('status', args.expected_input_status),
                       ('subject', subject), ('preproc_version', args.expected_preproc_version),
                       ('config', args.expected_nicad_config),
                       ('threshold', args.expected_nicad_threshold),
                       ('identical_location_pair_set', True)):
        expected_value(manifest, key, value, 'J03C_MANIFEST')
    recorded_hashes = manifest.get('files_sha256')
    if not isinstance(recorded_hashes, dict):
        fail(f'J03C_NO_FILE_HASHES subject={subject}')
    for name in ('clone_pairs.csv', 'clone_fragments.csv', 'comparison.json',
                 'run_summary.json', 'preproc_manifest.json'):
        expected = recorded_hashes.get(name)
        if not isinstance(expected, str) or not HEX64.fullmatch(expected):
            fail(f'J03C_NO_RECORDED_SHA256 subject={subject} file={name}')
        actual = sha_file(archived / name)
        if actual != expected:
            fail(f'J03C_ARCHIVE_SHA256_MISMATCH subject={subject} file={name}')
    comparison = read_json(archived / 'comparison.json')
    for key, value in (('identical_location_pair_set', True),
                       ('baseline_only_pairs', 0), ('preprocessed_only_pairs', 0)):
        expected_value(comparison, key, value, 'J03C_COMPARISON')
    summary = read_json(archived / 'run_summary.json')
    for key, value in (('subject', subject), ('phase', args.expected_input_phase),
                       ('identical_location_pair_set', True)):
        expected_value(summary, key, value, 'J03C_RUN_SUMMARY')
    if not isinstance(manifest.get('clone_pairs'), int) or manifest['clone_pairs'] < 0:
        fail(f'J03C_INVALID_PAIR_COUNT subject={subject}')
    if not isinstance(manifest.get('clone_fragments'), int) or manifest['clone_fragments'] < 0:
        fail(f'J03C_INVALID_FRAGMENT_COUNT subject={subject}')
    if (manifest['clone_pairs'] != comparison.get('preprocessed_pair_rows')
            or manifest['clone_pairs'] != summary.get('exported_unique_pairs')):
        fail(f'J03C_PAIR_COUNT_CONFLICT subject={subject}')
    if manifest['clone_fragments'] != summary.get('unique_fragments'):
        fail(f'J03C_FRAGMENT_COUNT_CONFLICT subject={subject}')
    preproc_file = source / '.j03b-preproc-manifest.json'
    if not preproc_file.is_file() or preproc_file.is_symlink():
        fail(f'J03B_SOURCE_MANIFEST_MISSING subject={subject}')
    preproc_sha = sha_file(preproc_file)
    if preproc_sha != manifest.get('preproc_manifest_sha256'):
        fail(f'J03B_SOURCE_MANIFEST_SHA256_MISMATCH subject={subject}')
    if sha_file(archived / 'preproc_manifest.json') != preproc_sha:
        fail(f'J03C_PREPROC_COPY_SHA256_MISMATCH subject={subject}')
    preproc = read_json(preproc_file)
    for key, value in (('experiment', args.expected_preproc_experiment),
                       ('version', args.expected_preproc_version), ('status', args.expected_preproc_status),
                       ('subject', subject), ('git_head', manifest.get('git_head'))):
        expected_value(preproc, key, value, 'J03B_SOURCE_MANIFEST')
    return archived, source, manifest, recorded_hashes, preproc_sha


def csv_rows(path, mandatory_fields):
    stream = path.open('r', newline='', encoding='utf-8')
    try:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames is None or set(mandatory_fields) - set(reader.fieldnames):
            fail(f'CSV_SCHEMA_MISMATCH path={path} columns={reader.fieldnames!r}')
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            fail(f'CSV_DUPLICATE_HEADER path={path}')
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                fail(f'CSV_MALFORMED_ROW path={path} line={reader.line_num}')
            yield row
    finally:
        stream.close()


def verify_endpoint(row, suffix, stored, subject):
    if row['repository_' + suffix] != subject:
        fail(f'PAIR_CROSS_SUBJECT subject={subject} pair={row["pair_id"]}')
    expected = (row['fragment_' + suffix + '_id'], row['repository_' + suffix],
                row['file_' + suffix], int(row['start_line_' + suffix]),
                int(row['end_line_' + suffix]), row['commit_' + suffix],
                row['source_sha256_' + suffix])
    if stored != expected:
        fail(f'PAIR_FRAGMENT_JOIN_MISMATCH subject={subject} pair={row["pair_id"]} side={suffix}')


def effective_configuration(args):
    """Full effective experiment settings; defined by Shell, recorded and checked."""
    return {
        'expected_subjects': args.expected_subjects,
        'expected_total_pairs': args.expected_total_pairs,
        'expected_input_experiment': args.expected_input_experiment,
        'expected_input_version': args.expected_input_version,
        'expected_input_phase': args.expected_input_phase,
        'expected_input_status': args.expected_input_status,
        'expected_preproc_experiment': args.expected_preproc_experiment,
        'expected_preproc_version': args.expected_preproc_version,
        'expected_preproc_status': args.expected_preproc_status,
        'expected_nicad_config': args.expected_nicad_config,
        'expected_nicad_threshold': args.expected_nicad_threshold,
        'expected_detector': args.expected_detector,
        'expected_language': args.expected_language,
        'positive_label': args.positive_label,
        'code_encoding': args.code_encoding,
        'code_errors': args.code_errors,
        'time_zone': args.time_zone,
        'progress_every_fragments': args.progress_every_fragments,
        'progress_every_pairs': args.progress_every_pairs,
        'project_root': str(args.project_root),
        'input_root': str(args.input_root),
        'expected_input_relative': args.expected_input_relative,
        'source_root': str(args.source_root),
        'output_root': str(args.output_root),
        'expected_output_relative': args.expected_output_relative,
        'run_dir_policy': 'per-run Shell-created result directory',
    }


def export_subject(args, subject):
    archived, source, manifest, source_hashes, preproc_sha = validate_input(args, subject)
    destination = args.output_root / subject
    if destination.exists() or destination.is_symlink():
        verify_output(args, subject)
        print(f'SKIP_VERIFIED subject={subject} path={destination}', flush=True)
        return read_json(destination / 'manifest.json')
    pending = Path(tempfile.mkdtemp(prefix=f'.{subject}.pending-', dir=args.output_root))
    print(f'SUBJECT_START subject={subject} input={archived} source={source} pending={pending}', flush=True)
    pairs_count = fragments_count = 0
    try:
        index_path = pending / '.validation.sqlite'
        db = sqlite3.connect(index_path)
        try:
            db.execute('PRAGMA journal_mode=DELETE')
            db.execute('CREATE TABLE fragments (id TEXT PRIMARY KEY, repo TEXT, file TEXT, '
                       'start INTEGER, finish INTEGER, commit_sha TEXT, sha TEXT)')
            db.execute('CREATE TABLE pair_ids (id TEXT PRIMARY KEY)')
            with (pending / 'fragments.jsonl').open('x', encoding='utf-8') as outf:
                for row in csv_rows(archived / 'clone_fragments.csv', FRAGMENT_FIELDS):
                    if row['repository'] != subject or row['language'] != args.expected_language:
                        fail(f'FRAGMENT_METADATA_INVALID subject={subject} id={row["fragment_id"]}')
                    if row['commit'] != manifest['git_head']:
                        fail(f'FRAGMENT_COMMIT_MISMATCH subject={subject} id={row["fragment_id"]}')
                    if not row['fragment_id'] or not HEX64.fullmatch(row['source_sha256']):
                        fail(f'FRAGMENT_IDENTIFIER_INVALID subject={subject}')
                    start, end = int(row['start_line']), int(row['end_line'])
                    if start < 1 or end < start:
                        fail(f'FRAGMENT_RANGE_INVALID subject={subject} id={row["fragment_id"]}')
                    path = strict_source_file(source, row['file_path'])
                    lines = source_lines(str(path))
                    if end > len(lines):
                        fail(f'FRAGMENT_RANGE_OUT_OF_BOUNDS subject={subject} id={row["fragment_id"]}')
                    code_bytes = b''.join(lines[start - 1:end])
                    if sha_bytes(code_bytes) != row['source_sha256']:
                        fail(f'FRAGMENT_SOURCE_SHA256_MISMATCH subject={subject} id={row["fragment_id"]}')
                    code = code_bytes.decode(args.code_encoding, errors=args.code_errors)
                    record = dict(row)
                    record.update({'start_line': start, 'end_line': end, 'code': code})
                    outf.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + '\n')
                    try:
                        db.execute('INSERT INTO fragments VALUES (?,?,?,?,?,?,?)',
                                   (row['fragment_id'], subject, row['file_path'], start, end,
                                    row['commit'], row['source_sha256']))
                    except sqlite3.IntegrityError as exc:
                        fail(f'DUPLICATE_FRAGMENT_ID subject={subject} id={row["fragment_id"]}: {exc}')
                    fragments_count += 1
                    if fragments_count % args.progress_every_fragments == 0:
                        db.commit()
                        print(f'FRAGMENT_PROGRESS subject={subject} count={fragments_count}', flush=True)
            db.commit()
            if fragments_count != manifest['clone_fragments']:
                fail(f'FRAGMENT_COUNT_MISMATCH subject={subject} actual={fragments_count} expected={manifest["clone_fragments"]}')
            with (pending / 'positive_pairs.csv').open('x', newline='', encoding='utf-8') as outf:
                writer = csv.DictWriter(outf, fieldnames=[*PAIR_FIELDS, 'label'], extrasaction='ignore')
                writer.writeheader()
                for row in csv_rows(archived / 'clone_pairs.csv', PAIR_FIELDS):
                    if (row['same_repository'].lower() != 'true'
                            or row['detector'] != args.expected_detector
                            or row['config'] != args.expected_nicad_config):
                        fail(f'PAIR_DETECTOR_METADATA_MISMATCH subject={subject} pair={row["pair_id"]}')
                    if not row['pair_id'] or row['fragment_a_id'] == row['fragment_b_id']:
                        fail(f'PAIR_SELF_OR_EMPTY subject={subject} pair={row["pair_id"]}')
                    for suffix in ('a', 'b'):
                        stored = db.execute('SELECT * FROM fragments WHERE id=?',
                                            (row['fragment_' + suffix + '_id'],)).fetchone()
                        if stored is None:
                            fail(f'PAIR_UNKNOWN_FRAGMENT subject={subject} pair={row["pair_id"]} side={suffix}')
                        verify_endpoint(row, suffix, stored, subject)
                    try:
                        db.execute('INSERT INTO pair_ids VALUES (?)', (row['pair_id'],))
                    except sqlite3.IntegrityError as exc:
                        fail(f'DUPLICATE_PAIR_ID subject={subject} pair={row["pair_id"]}: {exc}')
                    writer.writerow({**row, 'label': args.positive_label})
                    pairs_count += 1
                    if pairs_count % args.progress_every_pairs == 0:
                        db.commit()
                        print(f'PAIR_PROGRESS subject={subject} count={pairs_count}', flush=True)
            db.commit()
            if pairs_count != manifest['clone_pairs']:
                fail(f'PAIR_COUNT_MISMATCH subject={subject} actual={pairs_count} expected={manifest["clone_pairs"]}')
            for name, expected_sha in (('clone_pairs.csv', source_hashes['clone_pairs.csv']),
                                       ('clone_fragments.csv', source_hashes['clone_fragments.csv'])):
                if sha_file(archived / name) != expected_sha:
                    fail(f'AFTER_INPUT_CHANGED_DURING_EXPORT subject={subject} file={name}')
        finally:
            db.close()
        index_path.unlink()
        artifact_hashes = {name: sha_file(pending / name) for name in ('positive_pairs.csv', 'fragments.jsonl')}
        output_manifest = {
            'experiment': args.experiment, 'version': args.version,
            'timestamp_chicago': timestamp(args), 'subject': subject,
            'status': 'PREPARED_VERIFIED', 'label': args.positive_label,
            'pair_type': 'NiCad-detected positive candidate; not manually adjudicated',
            'input_phase': args.expected_input_phase,
            'input_root': str(args.input_root), 'source_root': str(args.source_root),
            'input_archive': str(archived), 'source_repository': str(source),
            'input_archive_manifest_sha256': sha_file(archived / 'archive_manifest.json'),
            'input_pair_csv_sha256': source_hashes['clone_pairs.csv'],
            'input_fragment_csv_sha256': source_hashes['clone_fragments.csv'],
            'input_comparison_sha256': source_hashes['comparison.json'],
            'source_preproc_manifest_sha256': preproc_sha,
            'source_git_head': manifest['git_head'],
            'detector': args.expected_detector, 'nicad_config': args.expected_nicad_config,
            'nicad_threshold': args.expected_nicad_threshold,
            'source_language': args.expected_language,
            'code_encoding': args.code_encoding, 'code_errors': args.code_errors,
            'pairs': pairs_count, 'fragments': fragments_count,
            'files_sha256': artifact_hashes,
            'configuration': effective_configuration(args),
        }
        write_json_new(pending / 'manifest.json', output_manifest)
        # Shell holds an exclusive flock; refuse any preexisting publication target.
        if destination.exists() or destination.is_symlink():
            fail(f'OUTPUT_TARGET_RACE subject={subject} path={destination}')
        os.rename(pending, destination)
        print(f'SUBJECT_PASS subject={subject} pairs={pairs_count} fragments={fragments_count} path={destination}', flush=True)
        return output_manifest
    except BaseException:
        print(f'SUBJECT_FAIL subject={subject} partial={pending}', file=sys.stderr, flush=True)
        raise
    finally:
        source_lines.cache_clear()


def verify_output(args, subject):
    archived, source, manifest, hashes, source_manifest_sha = validate_input(args, subject)
    dest = args.output_root / subject
    if not dest.is_dir() or dest.is_symlink():
        fail(f'OUTPUT_NOT_FOUND_OR_UNSAFE subject={subject} path={dest}')
    mp = dest / 'manifest.json'
    if not mp.is_file() or mp.is_symlink():
        fail(f'OUTPUT_MANIFEST_REQUIRED subject={subject}')
    m = read_json(mp)
    if m.get('version') not in args.accepted_output_versions:
        fail(f'OUTPUT_VERSION_UNSUPPORTED subject={subject} version={m.get("version")!r} '
             f'accepted={args.accepted_output_versions}')
    checks = {
        'experiment': args.experiment,
        'subject': subject, 'status': 'PREPARED_VERIFIED',
        'label': args.positive_label, 'input_root': str(args.input_root),
        'source_root': str(args.source_root),
        'source_git_head': manifest['git_head'],
        'input_phase': args.expected_input_phase,
        'input_archive_manifest_sha256': sha_file(archived / 'archive_manifest.json'),
        'input_pair_csv_sha256': hashes['clone_pairs.csv'],
        'input_fragment_csv_sha256': hashes['clone_fragments.csv'],
        'input_comparison_sha256': hashes['comparison.json'],
        'source_preproc_manifest_sha256': source_manifest_sha,
        'pairs': manifest['clone_pairs'], 'fragments': manifest['clone_fragments'],
        'detector': args.expected_detector, 'nicad_config': args.expected_nicad_config,
        'nicad_threshold': args.expected_nicad_threshold,
        'source_language': args.expected_language,
        'code_encoding': args.code_encoding, 'code_errors': args.code_errors,
    }
    for key, value in checks.items():
        expected_value(m, key, value, 'OUTPUT_MANIFEST')
    expected_value(m, 'configuration', effective_configuration(args), 'OUTPUT_MANIFEST')
    if not isinstance(m.get('files_sha256'), dict):
        fail(f'OUTPUT_HASHES_MISSING subject={subject}')
    for name in ('positive_pairs.csv', 'fragments.jsonl'):
        file = dest / name
        if not file.is_file() or file.is_symlink() or sha_file(file) != m['files_sha256'].get(name):
            fail(f'OUTPUT_FILE_SHA256_MISMATCH subject={subject} file={name}')
    print(f'VERIFY_PASS subject={subject} pairs={m["pairs"]} fragments={m["fragments"]}', flush=True)
    return m


def subjects(args):
    if not args.input_root.is_dir() or args.input_root.is_symlink():
        fail(f'AFTER_INPUT_DIRECTORY_REQUIRED path={args.input_root}')
    if not args.source_root.is_dir() or args.source_root.is_symlink():
        fail(f'PREPROCESSED_SOURCE_DIRECTORY_REQUIRED path={args.source_root}')
    names = sorted(p.name for p in args.input_root.iterdir() if p.is_dir() and not p.is_symlink())
    if len(names) != args.expected_subjects or any(not NAME.fullmatch(n) for n in names):
        fail(f'SUBJECT_COUNT_OR_NAME_MISMATCH actual={len(names)} expected={args.expected_subjects}')
    # J-03c AFTER archives are authoritative for the experiment subject set.
    # The preprocessing directory may also contain scratch/smoke/other folders;
    # they must not become subjects or prevent a valid 47-subject export.
    sources = {p.name for p in args.source_root.iterdir() if p.is_dir() and not p.is_symlink()}
    missing = set(names) - sources
    if missing:
        fail(f'AFTER_SOURCE_SUBJECT_MISSING missing={sorted(missing)} '
             f'after={len(names)} source={len(sources)}')
    extra = sorted(sources - set(names))
    if extra and args.mode == 'inspect':
        print(f'SOURCE_EXTRA_DIRECTORIES_IGNORED count={len(extra)} names={extra} '
              f'authority={args.input_root}', flush=True)
    return names


def summarize(args, names):
    count = total_pairs = total_fragments = 0
    records = []
    for name in names:
        m = verify_output(args, name)
        count += 1
        total_pairs += m['pairs']
        total_fragments += m['fragments']
        records.append({'subject': name, 'pairs': m['pairs'], 'fragments': m['fragments'],
                        'manifest_sha256': sha_file(args.output_root / name / 'manifest.json')})
    if count != args.expected_subjects or total_pairs != args.expected_total_pairs:
        fail(f'AGGREGATE_COUNT_MISMATCH subjects={count}/{args.expected_subjects} '
             f'pairs={total_pairs}/{args.expected_total_pairs}')
    report = {
        'experiment': args.experiment, 'version': args.version,
        'status': 'PASS', 'timestamp_chicago': timestamp(args),
        'subjects': count, 'pairs': total_pairs, 'fragments': total_fragments,
        'input_root': str(args.input_root), 'source_root': str(args.source_root),
        'output_root': str(args.output_root), 'subject_manifests': records,
        'configuration': effective_configuration(args),
    }
    write_json_new(args.run_dir / 'aggregate_summary.json', report)
    print(f'AGGREGATE_PASS subjects={count} pairs={total_pairs} fragments={total_fragments} '
          f'report={args.run_dir / "aggregate_summary.json"}', flush=True)



def analyze_clonetypes(args, names):
    """Delegate to the H-07/H-08 AST analyzer; no NiCad-only Type proxy labels."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('j03d_ast_clonetypes', args.ast_impl)
    if spec is None or spec.loader is None:
        fail(f'AST_IMPLEMENTATION_MISSING {args.ast_impl}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.analyze(args, names, sys.modules[__name__])


def prepare_fixed_positives(args, names):
    """New v6 positive-only selection; verify original inputs before publishing."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('j03d_fixed_positives', args.positive_impl)
    if spec is None or spec.loader is None:
        fail(f'POSITIVE_IMPLEMENTATION_MISSING {args.positive_impl}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run(args, names, sys.modules[__name__])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('experiment', 'version', 'mode', 'subject', 'project-root',
                 'input-root', 'expected-input-relative', 'source-root',
                 'output-root', 'expected-output-relative', 'run-dir',
                 'expected-subjects', 'expected-total-pairs',
                 'expected-input-experiment', 'expected-input-version',
                 'expected-input-phase', 'expected-input-status',
                 'expected-preproc-experiment', 'expected-preproc-version',
                 'expected-preproc-status', 'expected-nicad-config',
                 'expected-nicad-threshold', 'expected-detector',
                 'expected-language', 'positive-label',
                 'code-encoding', 'code-errors', 'time-zone',
                 'progress-every-fragments', 'progress-every-pairs',
                 'accepted-output-versions', 'clonetype-t1-label',
                 'clonetype-t2-label', 'clonetype-t3-label', 'clonetype-t4-label',
                 'clonetype-min-similarity', 'clonetype-max-similarity',
                 'clonetype-report-name', 'clonetype-progress-every-pairs',
                 'h07-python', 'h08-python', 'ast-impl',
                 'ast-parser-version', 'ast-grammar-version', 'ast-gram-size',
                 'ast-primary-threshold', 'ast-thresholds', 'ast-max-cells',
                 'ast-progress-fragments', 'ast-progress-pairs',
                 'ast-fragment-cache-size', 'ast-score-cache-size',
                 'ast-compression-level', 'ast-sample-per-category',
                 'ast-report-name', 'ast-nicad-band-edges',
                 'positive-impl', 'positive-source-census-name', 'positive-source-census-sha256',
                 'positive-balance-policy', 'positive-balance-types', 'positive-target-type-ratio',
                 'positive-missing-type-policy', 'positive-type2-blind-min', 'positive-sample-seed',
                 'positive-dedup-strategy', 'positive-public-name', 'positive-details-dir',
                 'positive-details-name', 'positive-manifest-name', 'positive-report-name',
                 'positive-progress-pairs', 'positive-sqlite-cache-kib', 'positive-create-outputs'):
        p.add_argument('--' + flag, required=True)
    a = p.parse_args()
    for name in ('project_root', 'input_root', 'source_root', 'output_root', 'run_dir', 'h07_python', 'h08_python', 'ast_impl', 'positive_impl'):
        setattr(a, name, Path(getattr(a, name)).resolve())
    for name in ('expected_subjects', 'expected_total_pairs', 'positive_label',
                 'progress_every_fragments', 'progress_every_pairs',
                 'clonetype_progress_every_pairs', 'ast_gram_size', 'ast_max_cells',
                 'ast_progress_fragments', 'ast_progress_pairs', 'ast_fragment_cache_size',
                 'ast_score_cache_size', 'ast_compression_level', 'ast_sample_per_category',
                 'positive_type2_blind_min', 'positive_sample_seed', 'positive_progress_pairs',
                 'positive_sqlite_cache_kib'):
        try:
            setattr(a, name, int(getattr(a, name)))
        except ValueError:
            p.error(f'{name} must be an integer')
    if a.expected_subjects < 1 or a.expected_total_pairs < 1 or a.positive_label != 1:
        p.error('Expected subjects and pair count must be positive; positive label must be 1')
    if a.progress_every_fragments < 1 or a.progress_every_pairs < 1 or a.clonetype_progress_every_pairs < 1:
        p.error('Progress intervals must be positive')
    if a.mode not in ('inspect', 'export', 'verify', 'summarize', 'clonetypes', 'clonetypes-smoke', 'positiveclone', 'positiveclone-verify'):
        p.error('Invalid mode')
    if a.mode in ('export', 'verify', 'clonetypes-smoke') and not NAME.fullmatch(a.subject):
        p.error('Invalid subject name')
    for rel in ('expected_input_relative', 'expected_output_relative'):
        value = Path(getattr(a, rel))
        if value.is_absolute() or '..' in value.parts or not value.parts:
            p.error(f'Invalid relative path: {rel}')
    if a.input_root != (a.project_root / a.expected_input_relative).resolve():
        p.error(f'WRONG_INPUT_ROOT: expected {a.project_root / a.expected_input_relative}; got {a.input_root}')
    if a.output_root != (a.project_root / a.expected_output_relative).resolve():
        p.error(f'WRONG_OUTPUT_ROOT: expected {a.project_root / a.expected_output_relative}; got {a.output_root}')
    if not a.input_root.is_relative_to(a.project_root) or not a.output_root.is_relative_to(a.project_root):
        p.error('Input and output paths must resolve inside the project root')
    if a.input_root == a.output_root or a.output_root.is_relative_to(a.input_root) or a.output_root.is_relative_to(a.source_root):
        p.error('Input, source, and output directory containment conflict')
    a.accepted_output_versions = tuple(a.accepted_output_versions.split(','))
    if not a.accepted_output_versions or any(not re.fullmatch(r'v[0-9]+', x) for x in a.accepted_output_versions):
        p.error('Invalid accepted output versions')
    a.clonetype_labels = (a.clonetype_t1_label, a.clonetype_t2_label,
                          a.clonetype_t3_label, a.clonetype_t4_label)
    if len(set(a.clonetype_labels)) != 4 or any(not re.fullmatch(r'[A-Z0-9_]+', x) for x in a.clonetype_labels):
        p.error('Invalid or duplicated clone type labels')
    for attr in ('clonetype_min_similarity', 'clonetype_max_similarity'):
        try:
            setattr(a, attr, Decimal(getattr(a, attr)))
        except InvalidOperation:
            p.error(f'Invalid decimal: {attr}')
    if not (Decimal(0) <= a.clonetype_min_similarity < a.clonetype_max_similarity == Decimal(100)):
        p.error('Clone similarity range must satisfy 0 <= min < max == 100')
    try:
        threshold = Decimal(a.expected_nicad_threshold)
    except InvalidOperation:
        p.error('Invalid NiCad threshold')
    if a.clonetype_min_similarity != Decimal(100) * (Decimal(1) - threshold):
        p.error('Clone similarity minimum must equal 100*(1-NiCad threshold)')
    if (Path(a.clonetype_report_name).name != a.clonetype_report_name
            or a.clonetype_report_name in ('.', '..')
            or not a.clonetype_report_name.endswith('.md')):
        p.error('Clone type report name must be a simple .md file name')
    for attr in ('ast_gram_size', 'ast_max_cells', 'ast_progress_fragments',
                 'ast_progress_pairs', 'ast_fragment_cache_size', 'ast_score_cache_size'):
        if getattr(a, attr) < 1:
            p.error(f'Invalid AST setting: {attr} must be >= 1')
    if not 0 <= a.ast_compression_level <= 9 or a.ast_sample_per_category < 0:
        p.error('Invalid AST compression level or sample count')
    try:
        a.ast_primary_threshold = Decimal(a.ast_primary_threshold)
        a.ast_thresholds = tuple(Decimal(x) for x in a.ast_thresholds.split(','))
    except InvalidOperation:
        p.error('Invalid AST thresholds')
    if (not 0 <= a.ast_primary_threshold <= 1
            or not a.ast_thresholds or any(not 0 <= x <= 1 for x in a.ast_thresholds)
            or a.ast_primary_threshold not in a.ast_thresholds
            or len(set(a.ast_thresholds)) != len(a.ast_thresholds)):
        p.error('Invalid AST thresholds; primary must be one of sensitivity thresholds')
    try:
        a.ast_nicad_band_edges = tuple(Decimal(x) for x in a.ast_nicad_band_edges.split(','))
    except InvalidOperation:
        p.error('Invalid NiCad band thresholds')
    if (len(a.ast_nicad_band_edges) < 2
            or any(not x.is_finite() for x in a.ast_nicad_band_edges)
            or a.ast_nicad_band_edges[0] != a.clonetype_max_similarity
            or a.ast_nicad_band_edges[-1] != a.clonetype_min_similarity
            or any(a.ast_nicad_band_edges[i] <= a.ast_nicad_band_edges[i+1]
                   for i in range(len(a.ast_nicad_band_edges)-1))):
        p.error('NiCad band thresholds must descend from max similarity to minimum')
    a.ast_nicad_band_labels = (str(a.ast_nicad_band_edges[0]),) + tuple(
        f'{a.ast_nicad_band_edges[i]}-<{a.ast_nicad_band_edges[i-1]}'
        for i in range(1, len(a.ast_nicad_band_edges)))
    if (Path(a.ast_report_name).name != a.ast_report_name
            or not a.ast_report_name.endswith('.md')):
        p.error('AST report name must be a basename ending .md')
    for path in (a.h07_python, a.h08_python, a.ast_impl, a.positive_impl):
        if not path.is_relative_to(a.project_root):
            p.error('AST source path outside project root')
    if a.positive_balance_policy != 'equal_available_types' or a.positive_missing_type_policy != 'equal_present_types_with_warning':
        p.error('Unsupported positive balance/missing-type policy')
    if a.positive_dedup_strategy != 'unordered_source_sha256_pair':
        p.error('Unsupported positive dedup strategy')
    if a.positive_balance_types != ('TYPE_1_SIGNATURE,TYPE_2_CONSISTENT_SIGNATURE,'
                                     'TYPE_2_BLIND_SIGNATURE,TYPE_3_STRUCTURAL_CANDIDATE'):
        p.error('Unexpected positive candidate labels')
    if a.positive_target_type_ratio != '1,1,1':
        p.error('Only strict equal Type1:Type2:Type3 supported')
    if a.positive_type2_blind_min < 0 or a.positive_progress_pairs < 1 or a.positive_sqlite_cache_kib < 64:
        p.error('Invalid positive sampling configuration')
    if a.positive_sample_seed < 0:
        p.error('Positive sample seed cannot be negative')
    if a.positive_create_outputs != 'true':
        p.error('Positive output publishing configuration must be true')
    if not HEX64.fullmatch(a.positive_source_census_sha256):
        p.error('Invalid positive source census sha256')
    for key in ('positive_source_census_name', 'positive_public_name', 'positive_details_dir',
                'positive_details_name', 'positive_manifest_name', 'positive_report_name'):
        x = getattr(a, key)
        if not x or x in ('.', '..') or Path(x).name != x or '/' in x or '\\' in x:
            p.error(f'Invalid positive output basename: {key}')
    if a.positive_public_name != a.positive_details_name:
        p.error('Public and detailed JSONL basenames must be identical')
    if (not a.positive_public_name.endswith('.jsonl')
            or not a.positive_manifest_name.endswith('.json')
            or not a.positive_report_name.endswith('.md')):
        p.error('Invalid positive output file extensions')
    try:
        ZoneInfo(a.time_zone)
    except Exception as exc:
        p.error(f'Invalid time zone: {exc}')
    if a.mode in ('inspect', 'summarize', 'clonetypes', 'clonetypes-smoke', 'positiveclone', 'positiveclone-verify'):
        names = subjects(a)
        if a.mode == 'inspect':
            for subject in names:
                _, _, manifest, _, _ = validate_input(a, subject)
                print(f'INPUT_PASS subject={subject} after_pairs={manifest["clone_pairs"]}', flush=True)
            print(f'INSPECT_PASS subjects={len(names)} '
                  f'input={a.input_root} source={a.source_root}', flush=True)
        elif a.mode == 'summarize':
            summarize(a, names)
        elif a.mode in ('positiveclone', 'positiveclone-verify'):
            prepare_fixed_positives(a, names)
        else:
            if a.mode == 'clonetypes-smoke':
                if a.subject not in names:
                    fail(f'AST_SMOKE_SUBJECT_NOT_IN_AFTER_INPUT {a.subject}')
                names = [a.subject]
            analyze_clonetypes(a, names)
    else:
        names = subjects(a)
        if a.subject not in names:
            fail(f'SUBJECT_NOT_IN_AFTER_INPUT subject={a.subject}')
        if a.mode == 'export':
            export_subject(a, a.subject)
        else:
            verify_output(a, a.subject)


if __name__ == '__main__':
    main()
