#!/usr/bin/env python3
"""J-02a v3: public-ready 5-field prepared train, deterministic Java IDs, reversible mapping."""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

FIELDS = ('corpus', 'id1', 'id2', 'label', 'tokens')
JAVA = ('java_source', 'java_mixed')
PY_PATTERN = re.compile(r'^py_[a-f0-9]{24}$')
JAVA_HASH = re.compile(r'^java-sha256:[a-f0-9]{64}$')
PY_HASH = re.compile(r'^py-sha256:[a-f0-9]{64}$')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1048576), b''):
            h.update(buf)
    return h.hexdigest()


def java_id(corpus, legacy):
    # Namespace includes corpus: identical textual legacy IDs from different pools are distinct.
    payload = ('j-02a-v3\x00' + corpus + '\x00' + legacy).encode('utf-8')
    return 'ja_' + hashlib.sha256(payload).hexdigest()[:24]


def rows(path):
    with path.open('r', encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f'Blank JSONL {path}:{n}')
            try:
                o = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'Invalid JSON {path}:{n}: {e}') from e
            if not isinstance(o, dict):
                raise ValueError(f'Non-object {path}:{n}')
            yield n, o


def check_base(row, path, n):
    if not set(FIELDS).issubset(row):
        raise ValueError(f'Missing base fields {path}:{n}')
    c = row['corpus']
    if c not in JAVA + ('python_mixed',):
        raise ValueError(f'Unsupported corpus {path}:{n}: {c!r}')
    if type(row['label']) is not int or row['label'] not in (0, 1):
        raise ValueError(f'Invalid label {path}:{n}')
    if type(row['tokens']) is not int or row['tokens'] <= 0:
        raise ValueError(f'Invalid tokens {path}:{n}')
    for side in ('id1', 'id2'):
        if not isinstance(row[side], str) or not row[side]:
            raise ValueError(f'Invalid {side} {path}:{n}')
    if set(row) != set(FIELDS) | {'left', 'right'}:
        raise ValueError(f'Not fully canonical (5 base+left/right) {path}:{n}')
    for side, ident in (('left','id1'),('right','id2')):
        metadata = row[side]
        expected_lang = 'python' if c == 'python_mixed' else 'java'
        pattern = PY_HASH if expected_lang == 'python' else JAVA_HASH
        if not isinstance(metadata, dict) or metadata.get('language') != expected_lang \
                or metadata.get('source_id') != row[ident] or metadata.get('source_dataset') != c \
                or not isinstance(metadata.get('function_id'), str) \
                or not pattern.fullmatch(metadata['function_id']):
            raise ValueError(f'Invalid {side} metadata for {path}:{n}')
        if c == 'python_mixed' and not PY_PATTERN.fullmatch(row[ident]):
            raise ValueError(f'Unexpected Python legacy ID {path}:{n}')
    return c


def generate(train_path, validation_path, expected):
    mappings = {}
    inverted = {}
    compact_train = []
    counts = {}
    for split, path in [('train', train_path), ('validation', validation_path)]:
        tally = Counter()
        for n, row in rows(path):
            c = check_base(row, path, n)
            tally[c] += 1
            out = {field: row[field] for field in FIELDS}
            for side, field in [('left','id1'),('right','id2')]:
                if c not in JAVA:
                    continue
                legacy = row[field]
                new = java_id(c, legacy)
                code_id = row[side]['function_id']
                key = (c, legacy)
                value = (new, code_id)
                if key in mappings and mappings[key] != value:
                    raise ValueError(f'Conflicting source mapping: {key}')
                mappings[key] = value
                if new in inverted and inverted[new] != key:
                    raise ValueError(f'Java compact ID collision: {new}')
                inverted[new] = key
                out[field] = new
            if split == 'train':
                compact_train.append(out)
        counts[split] = dict(tally)
        for corpus, num in expected[split].items():
            if corpus == 'total':
                continue
            if tally[corpus] != num:
                raise ValueError(f'{split} {corpus} count mismatch: {tally[corpus]} != {num}')
        if sum(tally.values()) != expected[split]['total']:
            raise ValueError(f'{split} total mismatch')
    mapping_rows = [dict(corpus=c, legacy_source_id=legacy, compact_id=val[0], function_id=val[1])
                    for (c, legacy), val in sorted(mappings.items())]
    return compact_train, mapping_rows, counts


def atomic_jsonl_create(path, objects):
    # Never overwrite existing artifacts; path is not modified if it already exists.
    with path.open('x', encoding='utf-8', newline='\n') as out:
        for obj in objects:
            out.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')) + '\n')


def compare(path, expected, is_compact=False):
    i = 0
    for n, actual in rows(path):
        if n > len(expected) or (is_compact and list(actual) != list(FIELDS)) or actual != expected[n-1]:
            raise ValueError(f'Unexpected simplified row {path}:{n}')
        i += 1
    if i != len(expected):
        raise ValueError(f'Output row count mismatch {i} != {len(expected)}')


def main():
    ap = argparse.ArgumentParser()
    for item in ('mode','canonical-train','canonical-validation','target-train','mapping-file','backup-dir','report'):
        ap.add_argument('--'+item, required=True)
    for spl in ('train','validation'):
        for c in ('total','java-source','java-mixed','python-mixed'):
            ap.add_argument('--expected-'+spl+'-'+c, required=True, type=int)
    args = ap.parse_args()
    if args.mode not in ('inspect','apply','verify'):
        ap.error('--mode must be inspect, apply, or verify')
    p = lambda attr: Path(getattr(args, attr.replace('-','_'))).resolve()
    canonical_train = p('canonical-train'); canonical_validation = p('canonical-validation')
    target = p('target-train'); mapping_file = p('mapping-file'); backup_dir = p('backup-dir'); report = p('report')
    for f in (canonical_train, canonical_validation, target):
        if not f.is_file():
            raise ValueError(f'Missing input: {f}')
    if len({canonical_train,target,mapping_file,canonical_validation}) != 4:
        raise ValueError('Input/output path collision')
    if not report.parent.is_dir():
        raise ValueError('Missing report parent')
    expected = {}
    for spl in ('train','validation'):
        expected[spl] = {c: getattr(args, f'expected_{spl}_{c.replace("-", "_")}')
                         for c in ('total','java-source','java-mixed','python-mixed')}
        expected[spl] = {k.replace('-','_'):v for k,v in expected[spl].items()}
        if min(expected[spl].values()) <= 0:
            raise ValueError('Expected counts must be positive')
    compact, mapping, counts = generate(canonical_train, canonical_validation, expected)
    result = dict(experiment='j-02a', version='v3', mode=args.mode,
                  timestamp_chicago=datetime.now(ZoneInfo('America/Chicago')).isoformat(),
                  canonical_train=str(canonical_train), target_train=str(target),
                  canonical_train_sha256=sha(canonical_train),
                  train_rows=len(compact), corpus_counts=counts,
                  java_mapping_entries=len(mapping), mapping_file=str(mapping_file),
                  java_id_rule='ja_ + SHA256(UTF8("j-02a-v3\\0" + corpus + "\\0" + legacy_source_id))[:24]',
                  python_id_policy='unchanged', validation_policy='unchanged',
                  leakage_audit_performed=False)
    if args.mode == 'inspect':
        if mapping_file.exists():
            raise ValueError('Mapping output already exists, refusing preflight')
        result['status'] = 'INSPECT_PASS'
    elif args.mode == 'apply':
        # Exact previous state: original target is a byte-for-byte copy of canonical v3.
        if sha(target) != sha(canonical_train):
            raise ValueError('Target is not canonical v3 bytes. Refusing overwrite.')
        if mapping_file.exists():
            raise ValueError('Mapping output exists; refusing overwrite')
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / ('train-canonical-v3-before-simplification-' +
                              datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f') + '.jsonl')
        with target.open('rb') as src, backup.open('xb') as dst:
            for buf in iter(lambda: src.read(1048576), b''):
                dst.write(buf)
            dst.flush(); os.fsync(dst.fileno())
        if sha(backup) != sha(target):
            raise ValueError('Backup mismatch. Original target remains untouched.')
        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_jsonl_create(mapping_file, mapping)
        compare(mapping_file, mapping)
        # Stage on same filesystem, and validate BEFORE atomic replacement.
        stage = target.with_name(target.name + '.j02a-v3-' + str(os.getpid()) + '.tmp')
        try:
            atomic_jsonl_create(stage, compact)
            compare(stage, compact, True)
            if sha(target) != sha(backup):
                raise ValueError('Target changed since backup; refusing replacement')
            os.replace(stage, target)
        finally:
            stage.unlink(missing_ok=True)
        compare(target, compact, True)
        result.update(status='UPDATED_AND_VERIFIED', previous_target_backup=str(backup),
                      previous_target_sha256=sha(backup), updated_target_sha256=sha(target),
                      mapping_sha256=sha(mapping_file))
    elif args.mode == 'verify':
        compare(target, compact, True)
        if not mapping_file.is_file():
            raise ValueError(f'Missing mapping file: {mapping_file}')
        compare(mapping_file, mapping)
        result.update(status='VERIFIED', updated_target_sha256=sha(target),
                      mapping_sha256=sha(mapping_file))
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f'J02A_V3_ERROR:{type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(2)
