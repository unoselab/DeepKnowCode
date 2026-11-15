#!/usr/bin/env python3
"""J-01b v12: lowercase owner_project ground truth and optional detail CSVs.
All gold labels come from the verified source, never inferred from predictions.
Existing legacy files are archived after validation; nothing is overwritten.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def jsonl_rows(path):
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f'{path}:{line_number}: blank JSONL row')
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_number}: malformed JSON') from exc
            if not isinstance(obj, dict):
                raise ValueError(f'{path}:{line_number}: expected JSON object')
            yield line_number, obj


def nonblank(value, location):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{location}: expected nonblank string')
    return value


def safe_segment(value, location):
    nonblank(value, location)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', value) or value in ('.', '..'):
        raise ValueError(f'{location}: unsafe owner/project component {value!r}')
    return value


def positive_int(value, name):
    try:
        num = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}: must be positive integer') from exc
    if num < 1:
        raise ValueError(f'{name}: must be positive integer')
    return num


def json_config(value, name, expected_type):
    try:
        obj = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f'{name}: invalid JSON') from exc
    if not isinstance(obj, expected_type):
        raise ValueError(f'{name}: expected {expected_type.__name__}')
    return obj


def csv_rows(path, header):
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != header:
            raise ValueError(f'{path}: schema mismatch; expected {header}, observed {reader.fieldnames}')
        for num, row in enumerate(reader, 2):
            if None in row or any(x is None for x in row.values()):
                raise ValueError(f'{path}:{num}: malformed CSV row')
            yield num, row


def make_name(pattern, owner, project):
    if pattern.count('{owner}') != 1 or pattern.count('{project}') != 1:
        raise ValueError('CSV name pattern must use {owner} and {project} exactly once')
    name = pattern.replace('{owner}', owner).replace('{project}', project)
    if not name.endswith('.csv') or '/' in name or '\\' in name or name in ('.', '..'):
        raise ValueError(f'Unsafe CSV name: {name!r}')
    return name.lower()


def legacy_name(pattern, project):
    if pattern.count('{project}') != 1:
        raise ValueError('Legacy CSV pattern must use {project} exactly once')
    name = pattern.replace('{project}', project)
    if not name.endswith('.csv') or '/' in name or '\\' in name:
        raise ValueError(f'Unsafe legacy name {name!r}')
    return name


def collect_function_codes(path, num_expected, id_field, code_field, id_prefix):
    values = {}
    for line_no, row in jsonl_rows(path):
        if set(row) != {id_field, code_field}:
            raise ValueError(f'{path}:{line_no}: function schema differs from {{id,code}}')
        code = nonblank(row.get(code_field), f'{path}:{line_no}:{code_field}')
        identifier = nonblank(row.get(id_field), f'{path}:{line_no}:{id_field}')
        computed = id_prefix + hashlib.sha256(code.encode('utf-8')).hexdigest()
        if identifier != computed or identifier in values:
            raise ValueError(f'{path}:{line_no}: code ID mismatch or duplicate')
        values[identifier] = code
    if len(values) != num_expected:
        raise ValueError(f'Function collection count mismatch: {len(values)} != {num_expected}')
    return values


def source_function(row, side_field, source_code_field, id_prefix, code_map, location):
    side = row.get(side_field)
    if not isinstance(side, dict):
        raise ValueError(f'{location}: missing {side_field} object')
    code = nonblank(side.get(source_code_field), f'{location}:{side_field}.{source_code_field}')
    code_hash = hashlib.sha256(code.encode('utf-8')).hexdigest()
    identifier = id_prefix + code_hash
    if code_map.get(identifier) != code:
        raise ValueError(f'{location}: {side_field} code absent/different in function collection')
    if 'code_sha256' in side and side['code_sha256'] != code_hash:
        raise ValueError(f'{location}: {side_field}.code_sha256 mismatch')
    return identifier


def inspect_legacy(output_dir, details_dir, per_project, args, legacy_header, legacy_detail_header):
    old_files = {p.name: p for p in output_dir.glob('*.csv') if p.is_file()}
    detail_files = {p.name: p for p in details_dir.glob('*.csv') if p.is_file()} if details_dir.is_dir() else {}
    legacy_expected = {
        project: legacy_name(args.legacy_csv_name_pattern, project) for project in per_project
    }
    detail_expected = {
        project: legacy_name(args.legacy_detail_name_pattern, project) for project in per_project
    }
    if not old_files and not detail_files:
        # A stranded, nonempty details directory requires manual inspection.
        if details_dir.exists() and any(details_dir.iterdir()):
            raise FileExistsError(f'{details_dir}: unexpected contents')
        return 'fresh', [], []
    # The user may have removed the 24 official v10 CSVs but left details files.
    # Retain the details by archiving them after validating all pair identities.
    if not old_files and set(detail_files) == set(detail_expected.values()):
        if set(x.name for x in details_dir.iterdir()) != set(detail_expected.values()):
            raise FileExistsError('Unknown items in details/; cannot archive safely')
        for project, original_rows in sorted(per_project.items()):
            detail = list(csv_rows(detail_files[detail_expected[project]], legacy_detail_header))
            if len(detail) != len(original_rows):
                raise ValueError(f'{project}: detail count differs from source')
            for n, ((_, meta), expected) in enumerate(zip(detail, original_rows), 1):
                left, right, gold, pid = expected
                if (meta[args.output_left_field] != left or meta[args.output_right_field] != right or
                        meta[args.pair_id_field] != pid or meta[args.project_field] != project or
                        meta[args.label_field] != str(gold)):
                    raise ValueError(f'{project}:{n}: detail mapping differs from source')
        return 'archive_details_only', [], list(detail_files.values()), []
    if set(old_files) != set(legacy_expected.values()) or set(detail_files) != set(detail_expected.values()):
        raise FileExistsError(
            f'Partial/unknown legacy files: official found={len(old_files)}, '
            f'details found={len(detail_files)}, expected={len(per_project)} each; refuse destructive migration')
    if set(x.name for x in details_dir.iterdir()) != set(detail_expected.values()):
        raise FileExistsError('Unknown items in details/; do not archive until resolved')
    previous = []
    for project, original_rows in sorted(per_project.items()):
        old_path = old_files[legacy_expected[project]]
        detail_path = detail_files[detail_expected[project]]
        before = list(csv_rows(old_path, legacy_header))
        detail = list(csv_rows(detail_path, legacy_detail_header))
        if len(before) != len(original_rows) or len(detail) != len(original_rows):
            raise ValueError(f'{project}: legacy row count mismatch')
        for n, ((_, old), (_, meta), expected) in enumerate(zip(before, detail, original_rows), 1):
            left, right, gold, pair_id = expected
            if old[args.output_left_field] != left or old[args.output_right_field] != right:
                raise ValueError(f'{project}:{n}: old pair ID references disagree with verified source')
            if any(old[c] != args.null_literal for c in legacy_header if c not in (args.output_left_field, args.output_right_field)):
                raise ValueError(f'{project}:{n}: old CSV has non-NULL prediction/refactorability values; preserve and review manually')
            if (meta[args.output_left_field] != left or meta[args.output_right_field] != right or
                    meta[args.pair_id_field] != pair_id or meta[args.project_field] != project or
                    meta[args.label_field] != str(gold)):
                raise ValueError(f'{project}:{n}: old Detail metadata/label disagrees with verified source')
        previous.append({'project': project, 'official_file': old_path.name,
                         'official_sha256': sha256_file(old_path), 'details_file': detail_path.name,
                         'details_sha256': sha256_file(detail_path), 'rows': len(original_rows)})
    return 'archive_v10', list(old_files.values()), list(detail_files.values()), previous


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    fields = (
        'mode', 'source-pairs', 'code-jsonl', 'output-dir', 'run-dir',
        'expected-pairs', 'expected-functions', 'expected-projects',
        'expected-positive', 'expected-negative', 'expected-source-sha256',
        'expected-code-sha256', 'code-id-field', 'code-value-field',
        'source-code-field', 'code-id-prefix', 'pair-id-field',
        'project-field', 'label-field', 'eligible-field', 'left-field',
        'right-field', 'output-left-field', 'output-right-field',
        'output-label-field', 'output-columns-json', 'project-owners-json',
        'owner-map-source-url', 'csv-name-pattern',
        'legacy-csv-name-pattern', 'legacy-detail-name-pattern',
        'legacy-columns-json', 'legacy-detail-columns-json',
        'details-subdir', 'archive-subdir', 'archive-prefix',
        'null-literal', 'report-filename', 'experiment', 'version'
    )
    for field in fields:
        parser.add_argument('--' + field, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode not in ('inspect', 'convert'):
        raise ValueError('Mode must be inspect or convert')
    if not re.fullmatch(r'[a-z]-[0-9]{2}[a-z]', args.experiment):
        raise ValueError('Invalid experiment name')
    if not re.fullmatch(r'v[1-9][0-9]*', args.version):
        raise ValueError('Invalid version')
    expected_pairs = positive_int(args.expected_pairs, 'expected-pairs')
    expected_functions = positive_int(args.expected_functions, 'expected-functions')
    expected_projects = positive_int(args.expected_projects, 'expected-projects')
    positive = positive_int(args.expected_positive, 'expected-positive')
    negative = positive_int(args.expected_negative, 'expected-negative')
    if positive + negative != expected_pairs:
        raise ValueError('Expected labels do not total expected pairs')
    output_header = json_config(args.output_columns_json, 'output-columns-json', list)
    expected_output = [args.output_left_field, args.output_right_field, args.output_label_field]
    if output_header != expected_output or len(set(output_header)) != 3:
        raise ValueError(f'Official CSV must contain exactly 3 columns {expected_output}')
    old_header = json_config(args.legacy_columns_json, 'legacy-columns-json', list)
    old_details_header = json_config(args.legacy_detail_columns_json, 'legacy-detail-columns-json', list)
    if len(old_header) != len(set(old_header)) or len(old_details_header) != len(set(old_details_header)):
        raise ValueError('Duplicate legacy header column')
    owner_map = json_config(args.project_owners_json, 'project-owners-json', dict)
    if not owner_map or len(owner_map) != expected_projects:
        raise ValueError(f'Owner map must contain exactly {expected_projects} projects')
    for key, value in owner_map.items():
        safe_segment(key, 'owner map project')
        safe_segment(value, 'GitHub owner')
    for name in (args.details_subdir, args.archive_subdir):
        safe_segment(name, 'subdirectory')
    if args.details_subdir == args.archive_subdir:
        raise ValueError('Details and Archive directory must differ')
    safe_segment(args.archive_prefix.rstrip('-'), 'archive prefix')
    if not re.fullmatch(r'[a-zA-Z0-9_.-]+\.json', args.report_filename):
        raise ValueError('Unsafe report filename')
    if not args.null_literal or any(c in args.null_literal for c in ('\n', '\r', ',')):
        raise ValueError('Invalid NULL literal')
    source = Path(args.source_pairs).resolve()
    code_jsonl = Path(args.code_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    for path in (source, code_jsonl):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not run_dir.is_dir():
        raise ValueError(f'RUN directory missing: {run_dir}')
    for path, expected_hash in ((source, args.expected_source_sha256),
                                (code_jsonl, args.expected_code_sha256)):
        actual = sha256_file(path)
        if actual != expected_hash:
            raise ValueError(f'{path.name}: SHA256 mismatch: expected {expected_hash}, got {actual}')
    code_map = collect_function_codes(code_jsonl, expected_functions,
                                      args.code_id_field, args.code_value_field,
                                      args.code_id_prefix)
    per_project = defaultdict(list)
    ids = set()
    functions_used = set()
    totals = Counter()
    review_counts = Counter()
    source_meta = {}
    for line_num, pair in jsonl_rows(source):
        loc = f'{source}:{line_num}'
        pair_id = nonblank(pair.get(args.pair_id_field), f'{loc}:{args.pair_id_field}')
        if pair_id in ids:
            raise ValueError(f'{loc}: duplicate pair_id {pair_id}')
        ids.add(pair_id)
        project = safe_segment(pair.get(args.project_field), f'{loc}:{args.project_field}')
        label = pair.get(args.label_field)
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f'{loc}: gold label must be integer 0 or 1')
        if args.eligible_field in pair and pair[args.eligible_field] is not True:
            raise ValueError(f'{loc}: evaluation_eligible must be true')
        totals[label] += 1
        if 'reviewer_a_confirmed' in pair:
            review_counts[f'reviewer_a_confirmed={pair["reviewer_a_confirmed"]}'] += 1
        if 'reviewer_b_confirmed' in pair:
            review_counts[f'reviewer_b_confirmed={pair["reviewer_b_confirmed"]}'] += 1
        left = source_function(pair, args.left_field, args.source_code_field,
                               args.code_id_prefix, code_map, loc)
        right = source_function(pair, args.right_field, args.source_code_field,
                                args.code_id_prefix, code_map, loc)
        functions_used.update((left, right))
        per_project[project].append((left, right, label, pair_id))
        meta = {key: value for key, value in pair.items() if key not in (args.left_field, args.right_field)}
        if args.left_field in pair:
            meta[args.left_field] = {k: v for k, v in pair[args.left_field].items() if k != args.source_code_field}
        if args.right_field in pair:
            meta[args.right_field] = {k: v for k, v in pair[args.right_field].items() if k != args.source_code_field}
        source_meta[pair_id] = {'label_source': str(pair.get('label_source', '')),
                                'metadata_json': json.dumps(meta, ensure_ascii=False, separators=(',', ':'), sort_keys=True)}
    if len(ids) != expected_pairs:
        raise ValueError(f'Pair count mismatch: {len(ids)} != {expected_pairs}')
    if totals != Counter({0: negative, 1: positive}):
        raise ValueError(f'Gold distribution mismatch: {dict(totals)}')
    if len(per_project) != expected_projects:
        raise ValueError(f'Project count mismatch: {len(per_project)} != {expected_projects}')
    if set(per_project) != set(owner_map):
        raise ValueError(f'Owner map/project mismatch; missing={set(per_project)-set(owner_map)}, extra={set(owner_map)-set(per_project)}')
    if functions_used != set(code_map):
        raise ValueError(f'Function coverage mismatch: {len(functions_used)} used, {len(code_map)} supplied')
    file_names = {project: make_name(args.csv_name_pattern, owner_map[project], project)
                  for project in per_project}
    if any(name != name.lower() for name in file_names.values()):
        raise ValueError('All official CSV names must be lowercase')
    if len({name.casefold() for name in file_names.values()}) != expected_projects:
        raise ValueError('Output names collide after case folding')

    if output_dir.is_dir():
        actual_top_csv = {p.name for p in output_dir.glob('*.csv') if p.is_file()}
    elif output_dir.exists():
        raise ValueError(f'Output dir is not a directory: {output_dir}')
    else:
        actual_top_csv = set()
    if set(file_names.values()) & actual_top_csv:
        raise FileExistsError('Canonical owner_project CSV already exists; never overwrite it')
    legacy_expected = {legacy_name(args.legacy_csv_name_pattern, p) for p in per_project}
    unrecognized = actual_top_csv - legacy_expected
    if unrecognized:
        raise FileExistsError(f'Unrecognized CSV in Ground Truth directory: {sorted(unrecognized)}')
    details_dir = output_dir / args.details_subdir
    legacy_status = inspect_legacy(output_dir, details_dir, per_project,
                                   args, old_header, old_details_header)
    archive_mode = legacy_status[0]
    old_files = legacy_status[1]
    detail_files = legacy_status[2]
    previous_sha = legacy_status[3] if archive_mode == 'archive_v10' else []
    archive_path = output_dir / args.archive_subdir / (args.archive_prefix + run_dir.name)
    if archive_path.exists():
        raise FileExistsError(f'Archive collision: {archive_path}')

    report = {
        'experiment': args.experiment, 'version': args.version,
        'status': 'INSPECTED' if args.mode == 'inspect' else 'PASS',
        'mode': args.mode, 'dataset_role': 'GOLD_GROUND_TRUTH',
        'source': str(source), 'source_sha256': sha256_file(source),
        'code_collection': str(code_jsonl), 'code_collection_sha256': sha256_file(code_jsonl),
        'owner_map_source': args.owner_map_source_url,
        'owner_map': {p: owner_map[p] for p in sorted(owner_map)},
        'output_dir': str(output_dir), 'output_columns': output_header,
        'clone_label_source_field': args.label_field,
        'clone_label_semantics': {'0': 'non-clone', '1': 'clone'},
        'pair_count': len(ids), 'function_count': len(functions_used),
        'project_count': len(per_project),
        'label_distribution': {str(label): totals[label] for label in (0, 1)},
        'reviewer_flag_counts': dict(sorted(review_counts.items())),
        'truncation': 0, 'resampling': 0,
        'legacy_file_status': archive_mode,
        'legacy_file_count': len(old_files),
        'legacy_detail_file_count': len(detail_files),
        'legacy_sha256': previous_sha,
        'archive_destination': str(archive_path) if archive_mode != 'fresh' else None,
        'details_columns': old_details_header,
        'details_recreated': True,
        'projects': [
            {'project': project, 'github_owner': owner_map[project],
             'filename': file_names[project], 'pairs': len(per_project[project]),
             'clone': sum(row[2] == 1 for row in per_project[project]),
             'non_clone': sum(row[2] == 0 for row in per_project[project])}
            for project in sorted(per_project)
        ],
    }
    if args.mode == 'inspect':
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    staging = Path(tempfile.mkdtemp(prefix='j01b-v12-stage-', dir=run_dir))
    published = []
    relocated = []
    created_archive = False
    try:
        staged_files = []
        for project in sorted(per_project):
            name = file_names[project]
            entries = per_project[project]
            target = staging / name
            with target.open('x', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream, lineterminator='\n')
                writer.writerow(output_header)
                writer.writerows((left, right, label) for left, right, label, _ in entries)
            actual = list(csv_rows(target, output_header))
            expected = [dict(zip(output_header, (left, right, str(label))))
                        for left, right, label, _ in entries]
            if [row for _, row in actual] != expected:
                raise ValueError(f'CSV round-trip verification failed for {name}')
            staged_files.append({'filename': name, 'rows': len(entries), 'sha256': sha256_file(target)})
        staged_details_dir = staging / args.details_subdir
        staged_details_dir.mkdir()
        detail_records = []
        for project in sorted(per_project):
            name = legacy_name(args.legacy_detail_name_pattern, project)
            target = staged_details_dir / name
            with target.open('x', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream, lineterminator='\n')
                writer.writerow(old_details_header)
                for left, right, label, pair_id in per_project[project]:
                    # This table is keyed to the exact same ordered source pairs as official CSV.
                    writer.writerow((left, right, pair_id, project, label,
                                     source_meta[pair_id]['label_source'],
                                     source_meta[pair_id]['metadata_json']))
            validated = list(csv_rows(target, old_details_header))
            if len(validated) != len(per_project[project]):
                raise ValueError(f'{project}: detail row count mismatch')
            detail_records.append({'filename': name, 'rows': len(validated), 'sha256': sha256_file(target)})
        report['files'] = staged_files
        report['detail_files'] = detail_records
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            if archive_mode != 'fresh':
                (archive_path / 'details_v10').mkdir(parents=True, exist_ok=False)
                if old_files:
                    (archive_path / 'official_v10').mkdir(parents=True, exist_ok=False)
                created_archive = True
                for path in sorted(old_files):
                    dst = archive_path / 'official_v10' / path.name
                    os.rename(path, dst)
                    relocated.append((dst, path))
                for path in sorted(detail_files):
                    dst = archive_path / 'details_v10' / path.name
                    os.rename(path, dst)
                    relocated.append((dst, path))
                # Check that archival preserved original bytes exactly.
                for rec in previous_sha:
                    if sha256_file(archive_path / 'official_v10' / rec['official_file']) != rec['official_sha256']:
                        raise ValueError('Archived official CSV SHA256 mismatch')
                    if sha256_file(archive_path / 'details_v10' / rec['details_file']) != rec['details_sha256']:
                        raise ValueError('Archived detail CSV SHA256 mismatch')
            for rec in staged_files:
                dst = output_dir / rec['filename']
                with dst.open('xb') as out, (staging / rec['filename']).open('rb') as src:
                    published.append(dst)
                    shutil.copyfileobj(src, out)
                if sha256_file(dst) != rec['sha256']:
                    raise ValueError(f'Published CSV SHA256 mismatch: {dst}')
            if details_dir.exists() and any(details_dir.iterdir()):
                raise FileExistsError(f'Unexpected contents in details directory: {details_dir}')
            details_dir.mkdir(exist_ok=True)
            for rec in detail_records:
                dst = details_dir / rec['filename']
                with dst.open('xb') as out, (staged_details_dir / rec['filename']).open('rb') as src:
                    published.append(dst)
                    shutil.copyfileobj(src, out)
                if sha256_file(dst) != rec['sha256']:
                    raise ValueError(f'Published detail CSV SHA256 mismatch: {dst}')
            with (run_dir / args.report_filename).open('x', encoding='utf-8') as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
        except BaseException:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            for dest, original in reversed(relocated):
                os.rename(dest, original)
            if created_archive:
                if (archive_path / 'official_v10').exists():
                    (archive_path / 'official_v10').rmdir()
                (archive_path / 'details_v10').rmdir()
                archive_path.rmdir()
            raise
        # Removing an empty, obsolete directory does not remove any prior artifact.
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(staging)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'J01B_V12_FAIL:{type(exc).__name__}:{exc}', file=sys.stderr)
        sys.exit(1)
