#!/usr/bin/env python3
"""H-02 v3. Stage SHA-verified H-02 v2 curated Java CSVs, without claiming gold labels.

All experiment configuration is required via Shell-provided CLI arguments.
Original Java data, existing H-Series runs, training sets and code map are read-only.
"""
import argparse
import collections
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json_exclusive(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write('\n')


def count_jsonl(path):
    n = 0
    with path.open('r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f, 1):
            require(bool(line.strip()), f'Blank JSONL row {path}:{i}')
            o = json.loads(line)
            require(isinstance(o, dict), f'Expected JSON object {path}:{i}')
            n += 1
            yield o


def verified_run(path, stage, version, required_files):
    require(path.is_dir(), f'Missing input run: {path}')
    m = read_json(path / 'manifest.json')
    require(m.get('stage') == stage and m.get('version') == version,
            f'Wrong H-Series source stage/version: {path}')
    for name in required_files:
        require(Path(name).name == name, f'Invalid artifact basename: {name}')
        meta = m.get('files', {}).get(name)
        file = path / name
        require(isinstance(meta, dict) and file.is_file(), f'Missing attested artifact {file}')
        require(sha256_file(file) == meta['sha256'] and file.stat().st_size == meta['bytes'],
                f'Attested artifact SHA/bytes drift: {file}')
    return m


def legacy_label(left_id, right_id):
    a = left_id.split('_')
    b = right_id.split('_')
    require(len(a) >= 2 and len(b) >= 2 and bool(a[1]) and bool(b[1]),
            f'Unparseable legacy ID {left_id} / {right_id}')
    return int(a[1] == b[1])


def check(a):
    root = a.root.resolve()
    dest = a.destination.resolve()
    require(dest == root / 'data/evaluation/ground_truth/java_new',
            f'Destination must be the exact configured java_new directory under root: {dest}')
    require(not dest.exists(), f'Refusing to overwrite existing destination: {dest}')
    require((root / 'data/evaluation/ground_truth/java').is_dir(),
            'Original Java ground_truth/java must remain present')
    require(a.output_dir.is_dir(), f'Missing Shell-created output directory: {a.output_dir}')
    h02, h03, h04 = (p.resolve() for p in (a.h02_run, a.h03_prepare_run, a.h04_run))
    source_dir = a.original_csv_dir.resolve()
    require(source_dir.is_dir(), 'Missing original 48 RQ4 CSV directory')
    require(a.code_map.is_file() and sha256_file(a.code_map) == a.expected_code_map_sha256,
            'Java code map SHA256 mismatch')

    m02 = verified_run(h02, 'h-02', a.source_version,
                       ('identity_audit.json','pair_records.jsonl','excluded_conflict_rows.jsonl',
                        'conflicting_code_pairs.jsonl','curated_csv_manifest.json'))
    h02_sha = sha256_file(h02 / 'manifest.json')
    require(h02_sha == a.expected_h02_manifest_sha256,
            f'H-02 v2 Manifest SHA drift: {h02_sha}')
    require(m02['summary']['status'] == a.expected_h02_status,
            f'H02 status drift: {m02["summary"]["status"]}')
    m03 = verified_run(h03, 'h-03-prepare', a.source_version,
                       ('annotation_audit.json','annotation_packets.jsonl'))
    m04 = verified_run(h04, 'h-04', a.source_version,
                       ('leakage_audit.json','overlap_examples.jsonl'))
    require(m03['summary']['source_h02_sha256'] == h02_sha,
            'H03 review packet originates from a different H02 run')
    require(m03['summary']['status'] == a.expected_h03_status and
            m03['summary']['gold_label_verified'] is False,
            'Expected H03 review packets ready but not human-verified gold')
    require(m04['sources']['h02_manifest_sha256'] == h02_sha,
            'H04 leakage audit originates from a different H02 run')
    require(m04['summary']['status'] == a.expected_h04_status and
            m04['summary']['exact_normalized_leakage_found'] is False and
            m04['summary']['conflict_count'] == 0 and
            m04['summary']['near_duplicate_semantic_review'] == a.semantic_review_scope,
            'H04 Exact/Normalized isolation status drift')
    require(m04['summary']['G02_prepared_manifest_sha256'] == a.expected_g02_manifest_sha256,
            'H04 checked a different G02 frozen preparation')
    require(m04['summary']['splits']['train']['pairs'] == a.expected_train_pairs and
            m04['summary']['splits']['validation']['pairs'] == a.expected_val_pairs,
            'H04 training/validation row count drift')

    report02 = read_json(h02 / 'identity_audit.json')
    require(report02['code_pair_label_conflicts'] == 0 and
            report02['input_pairs'] == a.expected_pairs and
            report02['excluded_conflict_rows'] == a.expected_excluded_rows and
            report02['eligible_positive'] == a.expected_positive and
            report02['eligible_negative'] == a.expected_negative and
            report02['unique_unordered_code_pairs'] == a.expected_unique_code_pairs,
            'H02 curated identity audit drift')
    require(report02['code_map_sha256'] == a.expected_code_map_sha256,
            'H02 code map source SHA drift')
    exclude = list(count_jsonl(h02 / 'excluded_conflict_rows.jsonl'))
    require(len(exclude) == a.expected_excluded_rows and
            {r['pair_key'] for r in exclude} == {a.expected_excluded_pair_key} and
            {f"{r['source_csv']}:{r['source_row']}" for r in exclude} ==
            set(a.expected_excluded_source_rows.split(',')),
            'Approved H02 excluded pair provenance drift')
    require((h02 / 'conflicting_code_pairs.jsonl').stat().st_size == 0,
            'Remaining H02 conflicting code pair groups')

    m02_curated = read_json(h02 / 'curated_csv_manifest.json')
    csv_meta = m02_curated['files']
    require(m02_curated['curated_pair_rows'] == a.expected_pairs and
            m02_curated['excluded_rows'] == a.expected_excluded_rows and
            m02_curated['projects'] == a.expected_projects and
            len(csv_meta) == a.expected_projects,
            'Curated CSV manifest count drift')
    require(len(list((h02/'curated_csv').glob('pred_*_with_refactorability.csv'))) == a.expected_projects,
            'Actual curated CSV file count differs from manifest')
    require(len(list((h02/'curated_csv').iterdir())) == a.expected_projects,
            'Unexpected non-CSV entries inside H02 curated_csv')
    require(len(list(source_dir.glob('pred_*_with_refactorability.csv'))) == a.expected_projects,
            'Original RQ4 CSV count drift')

    identity_counter = collections.Counter()
    unique_pair_keys = set()
    for row in count_jsonl(h02/'pair_records.jsonl'):
        label = row['id_derived_label']
        require(type(label) is int and label in (0,1), 'Invalid H02 row legacy label')
        require(row['pair_key'] != a.expected_excluded_pair_key, 'Excluded conflict pair found in selected H02 records')
        unique_pair_keys.add(row['pair_key'])
        identity_counter[(row['source_csv'],row['left_id'],row['right_id'],label,row['source_prediction'])] += 1
    require(sum(identity_counter.values()) == a.expected_pairs and
            len(unique_pair_keys) == a.expected_unique_code_pairs,
            'H02 pair_records rows/unique pair count drift')
    project_files = m02['summary']['project_files']
    require(len(project_files) == a.expected_projects and
            len(m02['sources']['csv_sha256']) == a.expected_projects,
            'H02 original project provenance incomplete')

    found = collections.Counter()
    census = collections.Counter()
    for name, info in sorted(csv_meta.items()):
        require(name == Path(name).name and name.startswith('pred_') and
                name.endswith('_with_refactorability.csv'),
                f'Unsafe or unrecognized curated CSV name: {name}')
        file = h02/'curated_csv'/name
        require(file.is_file() and not file.is_symlink(), f'Missing curated CSV: {file}')
        require(sha256_file(file) == info['sha256'] and file.stat().st_size == info['bytes'],
                f'Curated CSV drift: {name}')
        original = source_dir/name
        require(original.is_file() and
                sha256_file(original) == info['source_sha256'],
                f'Original RQ4 source hash drift: {name}')
        project = name[len('pred_'):-len('_with_refactorability.csv')]
        require(project in project_files and project_files[project]['name'] == name and
                m02['sources']['csv_sha256'][project] == info['source_sha256'],
                f'Original CSV provenance mismatch: {name}')
        row_count = 0
        with file.open(encoding='utf-8-sig', newline='') as inp:
            reader = csv.DictReader(inp)
            require({'pair_left_func','pair_right_func','clone_predict'} <= set(reader.fieldnames or []),
                    f'Curated CSV malformed columns: {name}')
            for row in reader:
                left, right = row['pair_left_func'].strip(), row['pair_right_func'].strip()
                pred = row['clone_predict'].strip()
                require(pred in ('0','1'), f'Invalid clone_predict: {name}')
                y = legacy_label(left,right)
                found[(name,left,right,y,int(pred))] += 1
                census[y] += 1
                row_count += 1
        require(row_count == info['rows'], f'Curated CSV row count drift: {name}')
    require(found == identity_counter, 'Curated CSV rows differ from H02 selected pair_records')
    require(sum(census.values()) == a.expected_pairs and
            census[1] == a.expected_positive and census[0] == a.expected_negative,
            f'Curated positive/negative census drift: {census}')
    require(sum(1 for _ in count_jsonl(h03/'annotation_packets.jsonl')) == a.expected_unique_code_pairs,
            'H03 prepared annotation packet count drift')
    require(sum(1 for _ in count_jsonl(h04/'overlap_examples.jsonl')) == 0,
            'H04 contains exact/normalized overlap examples')
    return {
        'status': 'H02_V3_PRECHECK_PASS_NOT_GOLD',
        'at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
        'source_h02_manifest_sha256': h02_sha,
        'source_h03_manifest_sha256': sha256_file(h03/'manifest.json'),
        'source_h04_manifest_sha256': sha256_file(h04/'manifest.json'),
        'code_map_sha256': a.expected_code_map_sha256,
        'projects': a.expected_projects,
        'original_pairs': a.expected_original_pairs,
        'curated_pairs': a.expected_pairs,
        'id_derived_positive': census[1],
        'id_derived_negative': census[0],
        'excluded_conflict_rows': a.expected_excluded_rows,
        'unique_code_pairs': a.expected_unique_code_pairs,
        'contradictory_label_groups': 0,
        'exact_normalized_leakage_g02': False,
        'semantic_near_duplicate_review': a.semantic_review_scope,
        'human_verified_gold': False,
        'data_classification': 'CURATED_LEGACY_ID_DERIVED_EVALUATION_CSV_NOT_HUMAN_VERIFIED_GOLD',
        'destination': str(dest),
        'files': {name: {'sha256': info['sha256'], 'bytes': info['bytes'], 'rows': info['rows']}
                  for name, info in sorted(csv_meta.items())},
        'effective_parameters': {
            key: str(val) if isinstance(val, Path) else val for key, val in vars(a).items()
        },
    }


def stage(a, precheck):
    dest = a.destination
    src = a.h02_run / 'curated_csv'
    require(not dest.exists(), f'Destination already exists (no overwrite): {dest}')
    # Exclusive directory creation prevents overwriting any pre-existing dataset.
    dest.mkdir(exist_ok=False)
    copied = 0
    for name, info in sorted(precheck['files'].items()):
        target = dest / name
        require(not target.exists(), 'Existing target unexpectedly detected: ' + str(target))
        shutil.copyfile(src / name, target)
        require(sha256_file(target) == info['sha256'] and
                target.stat().st_size == info['bytes'],
                f'Staged file did not match its immutable H02 source: {name}')
        copied += 1
    require(copied == a.expected_projects and len(list(dest.iterdir())) == a.expected_projects,
            'Unexpected staged directory entries')
    manifest = {**precheck,
        'status': 'H02_V3_JAVA_NEW_STAGED_PASS_NOT_GOLD',
        'staged_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
        'staged_directory_sha256': hashlib.sha256(
            '\n'.join(f'{k} {v["sha256"]}' for k, v in sorted(precheck['files'].items()))
            .encode('utf-8')).hexdigest(),
        'original_java_directory_preserved': True,
        'file_overwrite_performed': False,
    }
    write_json_exclusive(a.output_dir/'java_new_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root','h02-run','h03-prepare-run','h04-run','destination',
                'original-csv-dir','code-map','output-dir'):
        parser.add_argument('--'+key, type=Path, required=True)
    for key in ('source-version','expected-code-map-sha256','expected-h02-manifest-sha256',
                'expected-h02-status','expected-h03-status','expected-h04-status',
                'expected-excluded-pair-key','expected-excluded-source-rows',
                'expected-g02-manifest-sha256','semantic-review-scope'):
        parser.add_argument('--'+key, required=True)
    for key in ('expected-projects','expected-original-pairs','expected-pairs',
                'expected-positive','expected-negative','expected-unique-code-pairs',
                'expected-excluded-rows','expected-train-pairs','expected-val-pairs'):
        parser.add_argument('--'+key, type=int, required=True)
    parser.add_argument('--mode', choices=('preflight','stage'), required=True)
    a = parser.parse_args()
    require(a.expected_projects > 0 and a.expected_pairs == a.expected_positive + a.expected_negative,
            'Invalid explicit expected counts')
    require(a.expected_original_pairs == a.expected_pairs + a.expected_excluded_rows,
            'Expected source count != curated + excluded')
    result = check(a)
    if a.mode == 'preflight':
        write_json_exclusive(a.output_dir/'preflight.json', result)
    else:
        result = stage(a, result)
    print(json.dumps({k: result[k] for k in ('status','projects','original_pairs',
                       'curated_pairs','id_derived_positive','id_derived_negative',
                       'excluded_conflict_rows','contradictory_label_groups','human_verified_gold',
                       'destination')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
