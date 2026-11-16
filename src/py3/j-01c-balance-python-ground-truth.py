#!/usr/bin/env python3
"""Validate/rename V1 and build a project-stratified 50/50 V2 under python/."""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path, columns):
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != columns:
            raise ValueError(f'{path}: schema mismatch: {reader.fieldnames!r} != {columns!r}')
        items = []
        # Content-addressed function IDs can coincide for distinct source locations.
        # Preserve row multiplicity; identical ID pairs are not proof of bad annotations.
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'{path}:{line}: malformed CSV row')
            left, right, label = (row[column] for column in columns)
            if not re.fullmatch(r'py-sha256:[0-9a-f]{64}', left) or not re.fullmatch(r'py-sha256:[0-9a-f]{64}', right):
                raise ValueError(f'{path}:{line}: invalid function ID')
            if label not in ('0', '1'):
                raise ValueError(f'{path}:{line}: invalid clone_label={label!r}')
            # Do not reject self/duplicate content-hash pairs here: row-level
            # source locations and annotation identities were collapsed by the
            # code-content SHA256 ID representation in the official 3-col CSV.
            items.append(row)
    return items


def selection(rows, project, seed, strategy):
    groups = {'0': [], '1': []}
    for row in rows:
        groups[row['clone_label']].append(row)
    n = min(len(groups['0']), len(groups['1']))
    if not n:
        raise ValueError(f'{project}: zero examples in one class; cannot balance')
    chosen = []
    discarded = []
    for label in ('0', '1'):
        ordered = sorted(groups[label], key=lambda r: (r['pair_left_func'], r['pair_right_func']))
        if strategy == 'seeded-random':
            digest = hashlib.sha256(f'{seed}|{project}|{label}'.encode()).digest()
            rng = random.Random(int.from_bytes(digest, 'big'))
            rng.shuffle(ordered)
        picked = ordered[:n]
        omitted = ordered[n:]
        chosen.extend(picked)
        discarded.extend(omitted)
    chosen.sort(key=lambda r: (r['pair_left_func'], r['pair_right_func'], r['clone_label']))
    discarded.sort(key=lambda r: (r['pair_left_func'], r['pair_right_func'], r['clone_label']))
    return chosen, discarded


def save_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    for k in ('source-dir','output-dir','run-dir','output-columns-json','expected-filenames-json','legacy-filenames-json', 'mode','experiment','version','dataset-version','selection-strategy','manifest-filename','owner-map-source','report-filename'):
        ap.add_argument('--' + k, required=True)
    for k in ('seed','expected-projects','expected-pairs','expected-positive','expected-negative','expected-v2-pairs','expected-v2-per-class'):
        ap.add_argument('--' + k, type=int, required=True)
    args = ap.parse_args()
    source = Path(args.source_dir).resolve()
    output = Path(args.output_dir).resolve()
    run = Path(args.run_dir).resolve()
    columns = json.loads(args.output_columns_json)
    expected_files = json.loads(args.expected_filenames_json)
    legacy_files = json.loads(args.legacy_filenames_json)
    if args.mode not in ('inspect','build','normalize-v1') or args.selection_strategy not in ('seeded-random', 'lexicographic'):
        raise ValueError('Invalid mode or strategy')
    if columns != ['pair_left_func', 'pair_right_func', 'clone_label']:
        raise ValueError('Official V1/V2 CSV schema must be exact 3-column schema')
    if not isinstance(expected_files, list) or len(expected_files) != len(set(expected_files)) or len(expected_files) != args.expected_projects:
        raise ValueError('Expected filenames invalid')
    if any(not re.fullmatch(r'[a-z0-9-]+__[a-z0-9-]+\.csv', name) for name in expected_files):
        raise ValueError('Invalid expected lowercase double-underscore filenames')
    if not isinstance(legacy_files,list) or len(legacy_files)!=len(expected_files) or len(set(legacy_files))!=len(legacy_files):
        raise ValueError('Invalid legacy filename mapping')
    if any(not re.fullmatch(r'[a-z0-9-]+_[a-z0-9-]+\.csv', name) or '__' in name for name in legacy_files):
        raise ValueError('Invalid legacy filename')
    legacy_to_new=dict(zip(legacy_files,expected_files))
    new_to_legacy=dict(zip(expected_files,legacy_files))
    if not source.is_dir() or not run.is_dir():
        raise ValueError('Source or run directory missing')
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('V2 must not overlap V1 directory')
    if source.parent != output.parent or source.name != 'v1' or output.name != 'v2':
        raise ValueError('Require sibling python/v1 and python/v2 directories')
    actual_files = {p.name for p in source.glob('*.csv') if p.is_file()}
    if actual_files==set(expected_files):
        source_layout='double-underscore'
    elif actual_files==set(legacy_files):
        source_layout='legacy-single-underscore'
    else:
        raise ValueError(f'V1 inventory mismatch: found={sorted(actual_files)} expected double={sorted(expected_files)} or legacy={sorted(legacy_files)}')
    if args.mode=='normalize-v1' and source_layout=='double-underscore':
        raise FileExistsError('V1 already has double-underscore names; refusing repeat rename')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    records = {}
    summary = []
    v1_total = Counter()
    v2_total = Counter()
    all_v1_pairs = Counter()
    self_pairs_by_label = Counter()
    repeated_unordered_pairs_by_label = Counter()
    cross_project_repeated_pairs = 0
    first_pair_project = {}
    v1_hashes = {}
    for filename in sorted(expected_files):
        source_name=filename if source_layout=='double-underscore' else new_to_legacy[filename]
        path = source / source_name
        v1_hashes[filename] = sha256(path)
        rows = load_csv(path, columns)
        local_pairs = Counter()
        local_self = Counter()
        local_repeated = Counter()
        for row in rows:
            label = row['clone_label']
            v1_total[label] += 1
            key = tuple(sorted((row['pair_left_func'], row['pair_right_func'])))
            if row['pair_left_func'] == row['pair_right_func']:
                local_self[label] += 1
                self_pairs_by_label[label] += 1
            if local_pairs[key]:
                local_repeated[label] += 1
                repeated_unordered_pairs_by_label[label] += 1
            local_pairs[key] += 1
            if key in first_pair_project and first_pair_project[key] != filename:
                cross_project_repeated_pairs += 1
            first_pair_project.setdefault(key, filename)
            all_v1_pairs[key] += 1
        selected, excluded = selection(rows, filename, args.seed, args.selection_strategy)
        before = Counter(r['clone_label'] for r in rows)
        after = Counter(r['clone_label'] for r in selected)
        v2_total.update(after)
        summary.append({'filename':filename,'v1_pairs':len(rows),'v1_positive':before['1'],'v1_negative':before['0'],
                        'v2_pairs':len(selected),'v2_positive':after['1'],'v2_negative':after['0'],
                        'excluded_pairs':len(excluded),'excluded_positive':before['1']-after['1'],'excluded_negative':before['0']-after['0'],
                        'source_sha256':v1_hashes[filename],
                        'content_identical_id_pairs':dict(local_self),
                        'repeated_unordered_id_pairs':dict(local_repeated)})
        records[filename] = (selected, excluded)
    if (len(expected_files), sum(v1_total.values()), v1_total['1'], v1_total['0']) != (args.expected_projects,args.expected_pairs,args.expected_positive,args.expected_negative):
        raise ValueError(f'Unexpected V1 totals {dict(v1_total)}')
    if (sum(v2_total.values()),v2_total['1'],v2_total['0']) != (args.expected_v2_pairs,args.expected_v2_per_class,args.expected_v2_per_class):
        raise ValueError(f'Unexpected V2 totals {dict(v2_total)}')
    doc = {'experiment':args.experiment,'script_version':args.version,'dataset_version':args.dataset_version,
           'status':'INSPECTED' if args.mode=='inspect' else 'PASS','role':'BALANCED_SUBSET_OF_V1',
           'source_directory':str(source),'output_directory':str(output),'source_layout':source_layout,
           'filename_separator':'__','owner_mapping_source':args.owner_map_source,
           'selection_strategy':args.selection_strategy,'seed':args.seed,
           'schema':columns,'source_projects':args.expected_projects,
           'source_pairs':sum(v1_total.values()),'source_positive':v1_total['1'],'source_negative':v1_total['0'],
           'balanced_pairs':sum(v2_total.values()),'balanced_positive':v2_total['1'],'balanced_negative':v2_total['0'],
           'excluded_pairs':sum(v1_total.values())-sum(v2_total.values()),'no_synthetic_pairs':True,
           'no_relabeling':True,'v1_rows_preserved':True,'excluded_pairs_stored_in_v1_only':True,
           'content_identical_id_pairs_by_label':dict(self_pairs_by_label),
           'repeated_unordered_id_pairs_by_label':dict(repeated_unordered_pairs_by_label),
           'cross_project_repeated_unordered_id_pair_rows':cross_project_repeated_pairs,
           'unique_unordered_content_id_pairs':len(all_v1_pairs),
           'content_hash_caveat':'Equal py-sha256 IDs do not prove the same source-location function. CSVs lack per-pair source-location identity; inspect original verified_pairs.jsonl for provenance.',
           'projects':summary}
    if args.mode == 'normalize-v1':
        # All source rows, global quota and output integrity have been checked above.
        # Preserve content identity by rename only; no source records are changed.
        done=[]
        try:
            for old,new in zip(legacy_files,expected_files):
                oldpath=source/old
                newpath=source/new
                if not oldpath.is_file() or newpath.exists():
                    raise FileExistsError(f'Cannot safely rename {oldpath} to {newpath}')
                os.rename(oldpath,newpath)
                done.append((oldpath,newpath))
            for fname,h in v1_hashes.items():
                if sha256(source/fname)!=h:
                    raise RuntimeError(f'V1 hash changed during filename normalization: {fname}')
        except Exception:
            for oldpath,newpath in reversed(done):
                if newpath.exists() and not oldpath.exists():
                    os.rename(newpath,oldpath)
            raise
        doc['status']='PASS'
        doc['mode']='normalize-v1'
        doc['source_layout_after']='double-underscore'
        doc['renamed_files']=[{'from':old,'to':new,'sha256':v1_hashes[new]} for old,new in zip(legacy_files,expected_files)]
        report=run/args.report_filename
        with report.open('x',encoding='utf-8') as f:
            json.dump(doc,f,indent=2,ensure_ascii=False);f.write('\n')
        print(json.dumps(doc,indent=2),flush=True)
        return
    if args.mode == 'inspect':
        report = run / args.report_filename
        with report.open('x', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
            f.write('\n')
        print(json.dumps(doc, indent=2), flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f'.{output.name}-staging-', dir=output.parent))
    try:
        for filename in sorted(expected_files):
            selected, excluded = records[filename]
            save_csv(temp / filename, columns, selected)
            p = next(r for r in summary if r['filename']==filename)
            p['v2_sha256'] = sha256(temp / filename)
            # Independent read-back guards before final publication.
            if load_csv(temp / filename, columns) != selected:
                raise RuntimeError(f'CSV readback failed {filename}')
        for filename, expected_hash in v1_hashes.items():
            source_name=filename if source_layout=='double-underscore' else new_to_legacy[filename]
            if sha256(source / source_name) != expected_hash:
                raise RuntimeError(f'V1 modified during run: {filename}')
        with (temp / args.manifest_filename).open('x', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
            f.write('\n')
        # Atomic, no-overwrite publication on same filesystem.
        if output.exists():
            raise FileExistsError(f'Output appeared during build: {output}')
        os.rename(temp, output)
    except Exception:
        shutil.rmtree(temp)
        raise
    report = run / args.report_filename
    with report.open('x', encoding='utf-8') as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps(doc, indent=2), flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'J01C_V16_FAIL:{type(e).__name__}:{e}', file=sys.stderr)
        sys.exit(1)
