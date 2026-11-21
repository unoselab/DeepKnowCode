#!/usr/bin/env python3
"""J-04f v1: fail-closed, byte-identical publication of verified J-04e positives.

Publication does not reclassify clones or mutate J-04e/J-04d artifacts.
All experiment configuration is required from the standalone Shell CLI.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from zoneinfo import ZoneInfo

SUBJECT = re.compile(r'^[a-z0-9][a-z0-9_.-]*__[a-z0-9][a-z0-9_.-]*$')
HEX = re.compile(r'^[0-9a-f]{64}$')
RUN_ID = re.compile(r'^[0-9A-Za-z_.-]{1,180}$')
ARTIFACTS = (
    'positive_pairs.csv', 'fragments.jsonl', 'pair_clone_types.csv',
    'ast_clone_type_census.json', 'fixed_positive_clonepairs.jsonl',
    'details/fixed_positive_clonepairs.jsonl',
    'details/fixed_positive_clonepairs.manifest.json', 'manifest.json',
)
TYPES = ('TYPE_1_SIGNATURE', 'TYPE_2_CONSISTENT_SIGNATURE',
         'TYPE_2_BLIND_SIGNATURE', 'TYPE_3_STRUCTURAL_CANDIDATE')


def ensure(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_file(path):
    ensure(path.is_file() and not path.is_symlink(), 'FILE_UNSAFE_OR_MISSING '+str(path))
    return path


def safe_dir(path):
    ensure(path.is_dir() and not path.is_symlink(), 'DIR_UNSAFE_OR_MISSING '+str(path))
    return path


def read_json(path):
    return json.loads(safe_file(path).read_text(encoding='utf-8'))


def write_json_new(path, obj):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, sort_keys=True, ensure_ascii=True, indent=2)
        stream.write('\n')


def stamp(a):
    return datetime.now(ZoneInfo(a.timezone)).isoformat(timespec='microseconds')


def exact_tree(directory):
    safe_dir(directory)
    observed = set()
    for root, dirs, files in os.walk(directory, followlinks=False):
        for item in dirs:
            d = Path(root) / item
            ensure(d.is_dir() and not d.is_symlink(), 'UNSAFE_SUBDIR '+str(d))
        for item in files:
            f = safe_file(Path(root) / item)
            observed.add(f.relative_to(directory).as_posix())
    ensure(observed == set(ARTIFACTS), 'ARTIFACT_FILE_SET_MISMATCH '+str(directory)+' '+str(observed.symmetric_difference(ARTIFACTS)))


def check_subject(a, directory, name, record, verification_record):
    exact_tree(directory)
    manifest = read_json(directory / 'manifest.json')
    detail = read_json(directory / 'details/fixed_positive_clonepairs.manifest.json')
    ensure(sha(directory / 'manifest.json') == record['manifest_sha256'] == verification_record['manifest_sha256'],
           'SUBJECT_MANIFEST_HASH_MISMATCH '+name)
    ensure(manifest.get('schema') == 'deepknowcode-j04e-'+a.source_version and
           manifest.get('experiment') == a.source_experiment and
           manifest.get('version') == a.source_version and
           manifest.get('status') == 'COMPLETED' and
           manifest.get('mode') == 'all' and
           manifest.get('run_id') == a.source_all_run_id and
           manifest.get('subject') == name and
           detail.get('schema') == 'deepknowcode-j04e-fixed-positives-'+a.source_version and
           detail.get('subject') == name and
           detail.get('status') == 'FIXED_POSITIVE_ONLY', 'SOURCE_SUBJECT_IDENTITY_MISMATCH '+name)
    for key, mkey in (('input_pairs','input_pairs'),('input_fragments','input_fragments'),
                      ('selected_positive_pairs','selected_positive_pairs'),
                      ('content_duplicates_removed','content_duplicates_removed')):
        ensure(manifest.get(mkey) == record[key], 'SUBJECT_COUNT_MISMATCH '+name+' '+key)
    ensure(detail.get('selected_positive_pairs') == record['selected_positive_pairs'] and
           detail.get('removed_content_duplicates') == record['content_duplicates_removed'] and
           manifest.get('type_census') == record['type_census'] == verification_record['type_census'] and
           manifest.get('selected_type_counts') == record['selected_type_counts'] and
           detail.get('selected_type_counts') == record['selected_type_counts'] and
           detail.get('candidate_pairs') == record['input_pairs'], 'SUBJECT_TYPE_COUNTS_INVALID '+name)
    hashes = {}
    for rel in ARTIFACTS:
        digest = sha(safe_file(directory / rel))
        if rel != 'manifest.json':
            ensure(manifest.get('files_sha256', {}).get(rel) == digest,
                   'SOURCE_FILE_SHA_MISMATCH '+name+' '+rel)
        if rel not in ('manifest.json', 'details/fixed_positive_clonepairs.manifest.json'):
            ensure(detail.get('files_sha256', {}).get(rel) == digest,
                   'DETAILS_FILE_SHA_MISMATCH '+name+' '+rel)
        hashes[rel] = digest
    # Verify actual ID-only records and one-to-one public/details mapping.
    fragments = {}
    with safe_file(directory / 'fragments.jsonl').open(encoding='utf-8') as inp:
        for line in inp:
            ensure(bool(line.strip()), 'BLANK_FRAGMENT '+name)
            obj = json.loads(line)
            fid = obj['fragment_id']
            ensure(isinstance(fid, str) and fid not in fragments and
                   obj.get('repository') == name and isinstance(obj.get('code'), str) and
                   HEX.fullmatch(obj.get('source_sha256', '')) and
                   hashlib.sha256(obj['code'].encode('utf-8', errors='surrogateescape')).hexdigest() == obj['source_sha256'],
                   'FRAGMENT_INVALID '+name)
            fragments[fid] = obj
    ensure(len(fragments) == record['input_fragments'], 'FRAGMENT_TOTAL_INVALID '+name)
    with safe_file(directory / 'positive_pairs.csv').open(encoding='utf-8', newline='') as inp:
        pairs = list(csv.DictReader(inp))
    ensure(len(pairs) == record['input_pairs'], 'SOURCE_PAIR_TOTAL_INVALID '+name)
    with safe_file(directory / 'pair_clone_types.csv').open(encoding='utf-8', newline='') as inp:
        type_rows = list(csv.DictReader(inp))
    ensure(len(type_rows) == len(pairs), 'TYPE_ROWS_TOTAL_INVALID '+name)
    types = Counter(row['clone_subtype'] for row in type_rows)
    selected = [row for row in type_rows if row['selection'] == 'SELECTED']
    ensure(dict(types) == {k:v for k,v in record['type_census'].items() if v} and
           len(selected) == record['selected_positive_pairs'] and
           all(row['clone_subtype'] in TYPES for row in selected) and
           Counter(row['clone_subtype'] for row in selected) == Counter({k:v for k,v in record['selected_type_counts'].items() if v}),
           'TYPE_SELECTION_INVALID '+name)
    pubids = set()
    seen_content = set()
    selected_by_id = {r['pair_id']:r for r in selected}
    pairs_by_id = {r['pair_id']:r for r in pairs}
    ensure(len(pairs_by_id) == len(pairs) and len(selected_by_id) == len(selected), 'PAIR_IDS_DUPLICATED '+name)
    selected_list = [r for r in type_rows if r['selection'] == 'SELECTED']
    with (directory / 'fixed_positive_clonepairs.jsonl').open(encoding='utf-8') as pub, \
         (directory / 'details/fixed_positive_clonepairs.jsonl').open(encoding='utf-8') as det:
        pub_records = [json.loads(line) for line in pub if line.strip()]
        det_records = [json.loads(line) for line in det if line.strip()]
    ensure(len(pub_records) == len(det_records) == len(selected_list), 'PUBLIC_DETAILS_TOTAL_MISMATCH '+name)
    for record_pub, record_det, selection in zip(pub_records, det_records, selected_list, strict=True):
        ensure(set(record_pub) == {'sample_id','repository','clone_type','label','left','right'} and
               record_pub['label'] == a.positive_label and record_pub['repository'] == name and
               record_pub['sample_id'] not in pubids and
               all(record_pub[k] == record_det[k] for k in record_pub) and
               record_det.get('original_pair_id') == selection['pair_id'] and
               record_det.get('clone_subtype') == selection['clone_subtype'] and
               record_pub['left']['function_id'] == selection['fragment_a_id'] and
               record_pub['right']['function_id'] == selection['fragment_b_id'],
               'PUBLIC_DETAILS_MISMATCH '+name)
        pubids.add(record_pub['sample_id'])
        fa, fb = (fragments.get(record_pub[side]['function_id']) for side in ('left','right'))
        ensure(fa is not None and fb is not None, 'MISSING_FUNCTION_REFERENCE '+name)
        content_key = tuple(sorted((fa['source_sha256'], fb['source_sha256'])))
        ensure(content_key not in seen_content, 'DUPLICATE_CONTENT_PAIR '+name)
        seen_content.add(content_key)
    return hashes


def validate_source(a):
    ensure(a.source_all_root.name == a.source_all_run_id and
           a.source_verify_root.name == a.source_verify_run_id, 'PINNED_RUN_ID_MISMATCH')
    safe_dir(a.source_all_root)
    safe_dir(a.source_verify_root)
    summary = read_json(a.source_all_root / 'batch-summary.json')
    receipt = read_json(a.source_verify_root / 'verification-all.json')
    ensure(summary.get('schema') == 'deepknowcode-j04e-batch-'+a.source_version and
           summary.get('experiment') == a.source_experiment and
           summary.get('version') == a.source_version and
           summary.get('status') == 'COMPLETED' and summary.get('mode') == 'all' and
           summary.get('run_id') == a.source_all_run_id and
           summary.get('subjects_completed') == a.expected_subjects and
           summary.get('input_pairs_total') == a.expected_pairs and
           summary.get('input_fragments_total') == a.expected_fragments and
           summary.get('fixed_positive_total') == a.expected_fixed_positive and
           summary.get('content_duplicates_removed_total') == a.expected_duplicates and
           summary.get('type123_candidates_total') == a.expected_fixed_positive + a.expected_duplicates and
           receipt.get('status') == 'PASS' and receipt.get('version') == a.source_version and
           receipt.get('source_all') == str(a.source_all_root) and
           receipt.get('verified_subjects') == a.expected_subjects and
           receipt.get('input_pairs') == a.expected_pairs and
           receipt.get('input_fragments') == a.expected_fragments and
           receipt.get('fixed_positive_total') == a.expected_fixed_positive and
           receipt.get('source_j04d_batch_sha256') == summary.get('source_j04d_batch_sha256') and
           receipt.get('source_j04d_verification_sha256') == summary.get('source_j04d_verification_sha256'),
           'J04E_BATCH_OR_VERIFY_IDENTITY_INVALID')
    subjects = summary.get('subjects')
    verified = receipt.get('subjects')
    ensure(isinstance(subjects, list) and isinstance(verified, list) and
           len(subjects) == len(verified) == a.expected_subjects, 'SUBJECT_RECORD_COUNTS_INVALID')
    smap = {row['subject']:row for row in subjects}
    vmap = {row['subject']:row for row in verified}
    ensure(len(smap) == len(vmap) == a.expected_subjects and
           set(smap) == set(vmap) and all(SUBJECT.fullmatch(n) for n in smap),
           'SUBJECT_NAMES_INVALID')
    ensure(sum(x['input_pairs'] for x in subjects) == a.expected_pairs and
           sum(x['input_fragments'] for x in subjects) == a.expected_fragments and
           sum(x['selected_positive_pairs'] for x in subjects) == a.expected_fixed_positive and
           sum(x['content_duplicates_removed'] for x in subjects) == a.expected_duplicates,
           'SUBJECT_AGGREGATES_INVALID')
    sroot = safe_dir(a.source_all_root / 'subjects')
    ensure({p.name for p in sroot.iterdir()} == set(smap) and
           all(p.is_dir() and not p.is_symlink() for p in sroot.iterdir()),
           'SOURCE_SUBJECT_DIR_INVENTORY_INVALID')
    data = {}
    for n in sorted(smap):
        row, ver = smap[n], vmap[n]
        ensure(row['status'] == 'COMPLETED' and row['input_pairs'] == ver['input_pairs'] and
               row['input_fragments'] == ver['input_fragments'] and
               row['selected_positive_pairs'] == ver['selected_positive_pairs'],
               'VERIFICATION_SUBJECT_COUNTS_MISMATCH '+n)
        hashes = check_subject(a, sroot/n, n, row, ver)
        data[n] = {'subject':n, 'input_pairs':row['input_pairs'],
                   'input_fragments':row['input_fragments'],
                   'selected_positive_pairs':row['selected_positive_pairs'],
                   'content_duplicates_removed':row['content_duplicates_removed'],
                   'type_census':row['type_census'],
                   'selected_type_counts':row['selected_type_counts'],
                   'files_sha256':hashes}
        print(f'PASS J04F_SOURCE subject={n} candidates={row["input_pairs"]} fixed_positive={row["selected_positive_pairs"]}', flush=True)
    return data, {'batch':sha(a.source_all_root/'batch-summary.json'),
                  'verification':sha(a.source_verify_root/'verification-all.json')}


def ensure_paths(a):
    ensure(a.experiment == 'j-04f' and a.version == 'v1' and
           a.source_experiment == 'j-04e' and a.source_version == 'v1' and
           a.mode in ('inspect','dry-run','copy-result','verify') and
           a.timezone == 'America/Chicago' and a.positive_label == 1,
           'EXPERIMENT_CONFIGURATION_INVALID')
    ensure(all(x > 0 for x in (a.expected_subjects,a.expected_pairs,a.expected_fragments,a.expected_fixed_positive)) and
           a.expected_duplicates >= 0, 'COUNT_PARAMETERS_INVALID')
    ensure(a.expected_python_version == f'{sys.version_info.major}.{sys.version_info.minor}',
           'PYTHON_VERSION_MISMATCH')
    ensure(all(RUN_ID.fullmatch(x) for x in (a.run_id,a.source_all_run_id,a.source_verify_run_id)),
           'RUN_ID_FORMAT_INVALID')
    for p in (a.root,a.source_all_root,a.source_verify_root,a.result_root,a.publish_root,a.run_dir):
        ensure(p.is_absolute(), 'PATH_NOT_ABSOLUTE '+str(p))
    ensure(a.run_dir == a.result_root/a.run_id and
           a.result_root == a.root/'results/j-04f' and
           a.publish_root == a.root/'data/clone-detection/python/3pos-clones' and
           a.source_all_root == a.root/'results/j-04e'/a.source_all_run_id and
           a.source_verify_root == a.root/'results/j-04e'/a.source_verify_run_id,
           'PATH_IDENTITY_MISMATCH')
    for p in (a.root,a.root/'results',a.root/'results/j-04e',a.root/'data',
              a.root/'data/clone-detection',a.root/'data/clone-detection/python'):
        safe_dir(p)
    for p in (a.publish_root,a.result_root):
        ensure(not p.is_symlink(), 'OUTPUT_ROOT_SYMLINK '+str(p))


def verify_published(a,data,hashes):
    root = safe_dir(a.publish_root)
    ensure({p.name for p in root.iterdir()} == set(data) and
           all(p.is_dir() and not p.is_symlink() for p in root.iterdir()),
           'PUBLISHED_ROOT_SUBJECT_SET_INVALID')
    for n, r in sorted(data.items()):
        exact_tree(root/n)
        for rel, digest in r['files_sha256'].items():
            ensure(sha(root/n/rel)==digest, 'PUBLISHED_FILE_SHA256_MISMATCH '+n+' '+rel)
        print(f'PASS J04F_PUBLISHED subject={n} fixed_positive={r["selected_positive_pairs"]}',flush=True)
    return True


def copy_result(a,data,source_hashes):
    ensure(not a.run_dir.exists() and not a.run_dir.is_symlink(), 'RUN_DIRECTORY_EXISTS')
    if a.publish_root.exists():
        safe_dir(a.publish_root)
        ensure(not any(a.publish_root.iterdir()), 'PUBLISH_ROOT_NOT_EMPTY_REFUSE_OVERWRITE')
    else:
        a.publish_root.mkdir(parents=True,exist_ok=False)
    a.run_dir.mkdir(parents=True,exist_ok=False)
    staging = a.publish_root/('.j04f-stage-'+a.run_id)
    ensure(not staging.exists() and not staging.is_symlink(), 'STAGING_EXISTS')
    staging.mkdir(exist_ok=False)
    try:
        for n in sorted(data):
            src = a.source_all_root/'subjects'/n
            dst = staging/n
            dst.mkdir(exist_ok=False)
            (dst/'details').mkdir(exist_ok=False)
            for rel, digest in data[n]['files_sha256'].items():
                with (src/rel).open('rb') as reader, (dst/rel).open('xb') as writer:
                    shutil.copyfileobj(reader,writer)
                ensure(sha(dst/rel)==digest, 'STAGED_FILE_SHA_MISMATCH '+n+' '+rel)
            exact_tree(dst)
        # No replace semantics and no pre-existing published subject allowed.
        for n in sorted(data):
            target=a.publish_root/n
            ensure(not target.exists() and not target.is_symlink(), 'PUBLISH_TARGET_EXISTS '+str(target))
        for n in sorted(data):
            staging.joinpath(n).rename(a.publish_root/n)
            print('PASS J04F_COPY subject='+n,flush=True)
        staging.rmdir()
        verify_published(a,data,source_hashes)
        manifest={'schema':'deepknowcode-j04f-publication-v1','experiment':a.experiment,
                  'version':a.version,'status':'COMPLETED',
                  'timestamp_chicago':stamp(a),'run_id':a.run_id,
                  'source_all':str(a.source_all_root),'source_verify':str(a.source_verify_root),
                  'source_all_run_id':a.source_all_run_id,'source_verify_run_id':a.source_verify_run_id,
                  'source_batch_sha256':source_hashes['batch'],
                  'source_verification_sha256':source_hashes['verification'],
                  'publish_root':str(a.publish_root),
                  'subjects_total':len(data),'candidate_pairs_total':a.expected_pairs,
                  'fragments_total':a.expected_fragments,'fixed_positive_total':a.expected_fixed_positive,
                  'content_duplicates_total':a.expected_duplicates,
                  'subjects':[data[n] for n in sorted(data)],
                  'effective_configuration':vars(a) | {'root':str(a.root),'source_all_root':str(a.source_all_root),
                    'source_verify_root':str(a.source_verify_root),'result_root':str(a.result_root),
                    'publish_root':str(a.publish_root),'run_dir':str(a.run_dir)}}
        write_json_new(a.run_dir/'publication-manifest.json',manifest)
        print('PASS J04F_COPY_RESULT subjects='+str(len(data))+' fixed_positive='+str(a.expected_fixed_positive),flush=True)
    except BaseException:
        # Never remove published data on failure. Staging remains for manual inspection/recovery.
        print('FAIL J04F_PUBLICATION staging_or_partial_publication_preserved='+str(staging),file=sys.stderr,flush=True)
        raise


def verify_result(a,data,source_hashes):
    safe_dir(a.result_root)
    receipts=sorted((p for p in a.result_root.iterdir() if p.is_dir() and not p.is_symlink() and
                    '-copy-result-' in p.name),reverse=True)
    ensure(bool(receipts), 'NO_PUBLICATION_RECEIPT')
    chosen=receipts[0]
    publication=read_json(chosen/'publication-manifest.json')
    ensure(publication.get('schema') == 'deepknowcode-j04f-publication-v1' and
           publication.get('status') == 'COMPLETED' and
           publication.get('run_id') == chosen.name and
           publication.get('source_all') == str(a.source_all_root) and
           publication.get('source_verify') == str(a.source_verify_root) and
           publication.get('source_batch_sha256') == source_hashes['batch'] and
           publication.get('source_verification_sha256') == source_hashes['verification'] and
           publication.get('publish_root') == str(a.publish_root) and
           publication.get('subjects_total') == len(data) and
           publication.get('fixed_positive_total') == a.expected_fixed_positive and
           publication.get('candidate_pairs_total') == a.expected_pairs and
           publication.get('fragments_total') == a.expected_fragments and
           publication.get('content_duplicates_total') == a.expected_duplicates and
           publication.get('subjects') == [data[n] for n in sorted(data)],
           'PUBLICATION_RECEIPT_INVALID')
    verify_published(a,data,source_hashes)
    a.run_dir.mkdir(parents=True,exist_ok=False)
    write_json_new(a.run_dir/'verification-all.json',{
        'schema':'deepknowcode-j04f-verification-v1','status':'PASS','timestamp_chicago':stamp(a),
        'source_publication_run_id':chosen.name,
        'publication_manifest_sha256':sha(chosen/'publication-manifest.json'),
        'verified_subjects':len(data),'candidate_pairs':a.expected_pairs,
        'fixed_positive_total':a.expected_fixed_positive,'fragments':a.expected_fragments,
        'source_batch_sha256':source_hashes['batch'],
        'source_verification_sha256':source_hashes['verification']})
    print('PASS J04F_VERIFY_ALL subjects='+str(len(data))+' fixed_positive='+str(a.expected_fixed_positive),flush=True)


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','version','source-experiment','source-version','source-all-run-id',
                 'source-verify-run-id','run-id','mode','timezone','expected-python-version'):
        p.add_argument('--'+name,required=True)
    for name in ('root','source-all-root','source-verify-root','result-root','publish-root','run-dir'):
        p.add_argument('--'+name,required=True,type=Path)
    for name in ('expected-subjects','expected-pairs','expected-fragments','expected-fixed-positive',
                 'expected-duplicates','positive-label'):
        p.add_argument('--'+name,required=True,type=int)
    return p.parse_args(argv)


def main(argv=None):
    a=parse(argv)
    ensure_paths(a)
    data, hashes=validate_source(a)
    if a.mode in ('inspect','dry-run'):
        ensure(not a.run_dir.exists(), 'RUN_DIRECTORY_EXISTS')
        a.run_dir.mkdir(parents=True,exist_ok=False)
        write_json_new(a.run_dir/'inspection.json',{
            'schema':'deepknowcode-j04f-inspection-v1','status':'PASS',
            'timestamp_chicago':stamp(a),'mode':a.mode,
            'source_batch_sha256':hashes['batch'],
            'source_verification_sha256':hashes['verification'],
            'subjects':len(data),'candidate_pairs':a.expected_pairs,
            'fixed_positive':a.expected_fixed_positive,'fragments':a.expected_fragments,
            'publish_root':str(a.publish_root),
            'publish_root_empty':not a.publish_root.exists() or not any(a.publish_root.iterdir())})
        print('PASS J04F_'+a.mode.upper().replace('-','_')+' subjects='+str(len(data))+' fixed_positive='+str(a.expected_fixed_positive),flush=True)
    elif a.mode == 'copy-result':
        copy_result(a,data,hashes)
    else:
        ensure(not a.run_dir.exists() and not a.run_dir.is_symlink(),'RUN_DIRECTORY_EXISTS')
        verify_result(a,data,hashes)
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except (RuntimeError,ValueError,KeyError,TypeError,IndexError,OSError,UnicodeError) as exc:
        print('FAIL J04F '+type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True)
        sys.exit(2)
