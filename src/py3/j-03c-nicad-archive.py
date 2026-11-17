#!/usr/bin/env python3
"""J-03c v1: immutable before/after NiCad archives with comparable clone CSVs.

No experimental configuration defaults: standalone Shell supplies every required setting.
No changes to Git repositories, preprocessed repositories, or prepared datasets.
"""
import argparse
import csv
import functools
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PAIR_FIELDS = [
    'pair_id', 'fragment_a_id', 'repository_a', 'file_a', 'start_line_a', 'end_line_a', 'commit_a',
    'source_sha256_a', 'fragment_b_id', 'repository_b', 'file_b', 'start_line_b', 'end_line_b',
    'commit_b', 'source_sha256_b', 'same_repository', 'nicad_similarity_raw', 'nicad_nlines', 'detector',
    'config', 'xml_sha256'
]
FRAGMENT_FIELDS = ['fragment_id', 'repository', 'file_path', 'start_line', 'end_line', 'commit', 'source_sha256', 'language']
NAME = re.compile(r'^[a-z0-9][a-z0-9_.-]*__[a-z0-9][a-z0-9_.-]*$')


def chicago():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat(timespec='seconds')


def hbytes(data):
    return hashlib.sha256(data).hexdigest()


def hfile(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as fd:
        for buf in iter(lambda: fd.read(1024 * 1024), b''):
            digest.update(buf)
    return digest.hexdigest()


def fresh_json(path, content):
    with Path(path).open('x', encoding='utf-8') as fd:
        json.dump(content, fd, indent=2, sort_keys=True, ensure_ascii=False)
        fd.write('\n')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def git_head(repo, require_clean):
    if not repo.is_dir() or not (repo / '.git').exists():
        raise RuntimeError(f'Raw Git checkout missing: {repo}')
    def call(*args):
        proc = subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True)
        return proc.stdout.strip()
    if require_clean and call('status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError(f'Raw Git checkout dirty: {repo}')
    return call('rev-parse', 'HEAD')


def resolve_within(path, parent):
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(Path(parent).resolve(strict=True)):
        raise RuntimeError(f'Source escapes expected directory: {path} outside {parent}')
    return resolved


def verify_files(folder, manifest):
    for rel, sha in manifest.get('files_sha256', {}).items():
        file = folder / rel
        if not file.is_file() or file.is_symlink() or hfile(file) != sha:
            raise RuntimeError(f'ARCHIVE_FILE_HASH_MISMATCH file={file}')
    if not manifest.get('files_sha256'):
        raise RuntimeError(f'ARCHIVE_EMPTY_MANIFEST {folder}')


def settings(args):
    """Record and verify every effective experimental Shell configuration knob."""
    return {'granularity': args.granularity, 'language': args.language,
            'strict_parse_failures': args.strict_parse, 'require_clean': args.require_clean,
            'unicode_escape_policy': args.unicode_policy,
            'copy_class_xml': args.copy_class_xml,
            'expected_subjects': args.expected_subjects,
            'progress_every_pairs': args.progress_every}


def existing(folder, args, phase, subject, head):
    """Fail closed, or return True only when the existing artifact is fully verified."""
    if not folder.exists() and not folder.is_symlink():
        return False
    if folder.is_symlink() or not folder.is_dir():
        raise RuntimeError(f'ARCHIVE_TARGET_UNSAFE {folder}')
    path = folder / 'archive_manifest.json'
    if not path.is_file():
        raise RuntimeError(f'ARCHIVE_TARGET_EXISTS_UNVERIFIED {folder}')
    m = load_json(path)
    required = {'experiment': args.experiment, 'version': args.version, 'subject': subject,
                'phase': phase, 'git_head': head, 'config': args.config,
                'threshold': args.threshold, 'preproc_version': args.preproc_version}
    for key, value in required.items():
        if m.get(key) != value:
            raise RuntimeError(f'ARCHIVE_MANIFEST_MISMATCH {folder} field={key}: {m.get(key)} != {value}')
    verify_files(folder, m)
    if phase == 'after':
        current_preproc = args.preproc_root / subject / '.j03b-preproc-manifest.json'
        if not current_preproc.is_file() or hfile(current_preproc) != m.get('preproc_manifest_sha256'):
            raise RuntimeError(f'ARCHIVE_PREPROC_PROVENANCE_MISMATCH subject={subject}')
    print(f'ARCHIVE_VERIFIED phase={phase} subject={subject} pairs={m["clone_pairs"]} path={folder}', flush=True)
    return True


def subject_list(args):
    subjects = sorted(p.name for p in args.repo_root.iterdir() if p.is_dir() and (p / '.git').exists())
    if len(subjects) != args.expected_subjects or any(not NAME.fullmatch(s) for s in subjects):
        raise RuntimeError(f'Subject inventory mismatch: found {len(subjects)} expected {args.expected_subjects}')
    return subjects


def original_attempt(args, subject, head):
    root = args.j03a_run / 'subjects' / subject / 'attempts'
    if not root.is_dir():
        raise RuntimeError(f'J03A_ATTEMPT_ROOT_MISSING subject={subject} path={root}')
    rejected = []
    for a in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        try:
            s = load_json(a / 'run_summary.json')
            m = load_json(a / 'run_manifest.json')
            if s.get('status') not in ('NICAD_XML_PARSED', 'PARSE_WARNINGS_REVIEW_REQUIRED'):
                raise RuntimeError('status is not complete')
            if s.get('subject') != subject or m.get('subject') != subject:
                raise RuntimeError('subject mismatch')
            if s.get('version') != args.baseline_version or m.get('version') != args.baseline_version:
                raise RuntimeError('baseline experiment version mismatch')
            if m.get('config') != args.config or m.get('granularity') != args.granularity or m.get('language') != args.language:
                raise RuntimeError('config mismatch')
            candidates = [r for r in m.get('git_repositories', []) if r.get('repository') == subject]
            if len(candidates) != 1 or candidates[0]['head_commit'] != head:
                raise RuntimeError('raw Git HEAD differs')
            for filename, field in [('clone_pairs.csv', 'pair_csv_sha256'),
                                    ('clone_fragments.csv', 'fragment_csv_sha256'),
                                    ('repository_clone_counts.csv', 'repository_clone_counts_sha256')]:
                if hfile(a / filename) != s[field]:
                    raise RuntimeError(f'{filename} hash mismatch')
            xml = resolve_within(s['nicad_pair_xml'], a)
            expected_threshold = format(float(args.threshold) * 100, 'g') + '%'
            systeminfo = next((el.attrib for _, el in ET.iterparse(xml, events=('start',)) if el.tag == 'systeminfo'), None)
            if systeminfo is None or systeminfo.get('threshold') != expected_threshold:
                raise RuntimeError('baseline NiCad threshold mismatch')
            if hfile(xml) != s['nicad_pair_xml_sha256']:
                raise RuntimeError('original NiCad XML hash mismatch')
            if s.get('submitted_repository_count') != 1 or s.get('cross_repository_pairs') != 0:
                raise RuntimeError('not subject independent')
            return a, s, m, xml
        except (OSError, KeyError, ValueError, RuntimeError) as e:
            rejected.append(f'{a.name}: {str(e)[:200]}')
    raise RuntimeError(f'NO_VERIFIED_J03A_ATTEMPT subject={subject} rejected={rejected[:6]}')


def publish_file(src, folder, rel, recorded):
    dst = folder / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, 'rb') as s, open(dst, 'xb') as d:
        shutil.copyfileobj(s, d, length=1024 * 1024)
    digest = hfile(dst)
    if digest != hfile(src):
        raise RuntimeError(f'Copy verification failed: {src} -> {dst}')
    recorded[rel] = digest


def optional_nicad_xml(original_xml, temp, recorded, args):
    """Copy class XMLs from the sibling NiCad output, without large HTML/source listings."""
    if not args.copy_class_xml:
        return
    p = original_xml.parent
    for suffix, dest in [('classes.xml', 'classes.xml'), ('classes-withsource.xml', 'classes-withsource.xml')]:
        matches = sorted(x for x in p.iterdir() if x.is_file() and x.name.endswith('-' + suffix))
        if len(matches) > 1:
            raise RuntimeError(f'Ambiguous NiCad class XML: {matches}')
        if matches:
            publish_file(matches[0], temp, 'nicad/' + dest, recorded)


def atomic_folder(root, subject, operation):
    root.mkdir(parents=True, exist_ok=True)
    out = root / subject
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'Existing target cannot be replaced: {out}')
    pending = Path(tempfile.mkdtemp(dir=root, prefix=f'.{subject}.pending-'))
    try:
        manifest = operation(pending)
        fresh_json(pending / 'archive_manifest.json', manifest)
        if out.exists() or out.is_symlink():
            raise FileExistsError(f'Concurrent target creation: {out}')
        pending.rename(out)
        return out, manifest
    except BaseException:
        print(f'ARCHIVE_PARTIAL_PRESERVED path={pending}', file=sys.stderr, flush=True)
        raise


def publish_before(args, subject):
    head = git_head(args.repo_root / subject, args.require_clean)
    folder = args.before_root / subject
    if existing(folder, args, 'before', subject, head):
        return True
    a, summary, _, xml = original_attempt(args, subject, head)
    def op(temp):
        recorded = {}
        for rel in ('clone_pairs.csv', 'clone_fragments.csv', 'repository_clone_counts.csv', 'run_summary.json'):
            publish_file(a / rel, temp, rel, recorded)
        publish_file(xml, temp, 'nicad/pairs.xml', recorded)
        optional_nicad_xml(xml, temp, recorded, args)
        return {'experiment': args.experiment, 'version': args.version, 'phase': 'before',
                'timestamp_chicago': chicago(), 'subject': subject, 'status': 'ARCHIVED_VERIFIED',
                'git_head': head, 'config': args.config, 'threshold': args.threshold,
                'preproc_version': args.preproc_version, 'baseline_version': args.baseline_version,
                'clone_pairs': summary['exported_unique_pairs'],
                'clone_fragments': summary['unique_fragments'], **settings(args), 'parse_failures': summary['parse_failures_detected_in_nicad_logs'],
                'source_j03a_attempt': str(a), 'source_j03a_summary_sha256': hfile(a / 'run_summary.json'),
                'source_xml_sha256': hfile(xml), 'files_sha256': recorded}
    out, m = atomic_folder(args.before_root, subject, op)
    print(f'BEFORE_PASS subject={subject} pairs={m["clone_pairs"]} path={out}', flush=True)
    return True


def preproc_info(args, subject, head):
    root = args.preproc_root / subject
    mp = root / '.j03b-preproc-manifest.json'
    if not mp.is_file():
        raise RuntimeError(f'PREPROC_NOT_READY subject={subject} path={mp}')
    m = load_json(mp)
    required = {'experiment': 'j-03b', 'version': args.preproc_version, 'status': 'PREPARED',
                'subject': subject, 'git_head': head,
                'java_unicode_escape_policy': args.unicode_policy}
    for key, value in required.items():
        if m.get(key) != value:
            raise RuntimeError(f'PREPROC_POLICY_MISMATCH subject={subject} field={key}')
    audit = Path(m['audit_csv'])
    if not audit.is_file() or hfile(audit) != m.get('audit_sha256'):
        raise RuntimeError(f'PREPROC_AUDIT_INTEGRITY_FAIL subject={subject}')
    return root, mp, m


def normalize_file(file, subject):
    f = file.replace('\\', '/')
    if f.startswith('nicadclones/'):
        prefix = f'nicadclones/{subject}/{subject}/'
        if not f.startswith(prefix):
            raise RuntimeError(f'Unexpected NiCad subject prefix: {file}')
        f = f[len(prefix):]
    else:
        marker = f'/{subject}/'
        if marker not in f:
            raise RuntimeError(f'NiCad source has no verified subject: {file}')
        f = f.split(marker, 1)[1]
    p = Path(f)
    if p.is_absolute() or not p.parts or '..' in p.parts or p.suffix != '.java':
        raise RuntimeError(f'Unsafe NiCad source path: {file}')
    return p.as_posix()


def location_key(p1, p2):
    return '\x1f'.join(sorted((p1, p2)))


def endpoint(file, start, end):
    s, e = int(start), int(end)
    if s < 1 or e < s:
        raise RuntimeError(f'Invalid fragment coordinates: {file}:{start}-{end}')
    return f'{file}:{s}-{e}'


@functools.lru_cache(maxsize=16)
def source_lines(path):
    return Path(path).read_bytes().splitlines(keepends=True)


@functools.lru_cache(maxsize=100000)
def fragment_for(source, rel, start, end, subject, head, language):
    path = source / rel
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f'NiCad fragment source missing: {path}')
    lines = source_lines(str(path))
    if end > len(lines):
        raise RuntimeError(f'Out-of-range NiCad fragment: {rel}:{start}-{end}, total={len(lines)}')
    frag = {'repository': subject, 'file_path': rel, 'start_line': start, 'end_line': end,
            'commit': head, 'source_sha256': hbytes(b''.join(lines[start-1:end])), 'language': language}
    frag_id = 'j03cf_' + hbytes(json.dumps(frag, sort_keys=True).encode('utf-8'))[:24]
    frag['fragment_id'] = frag_id
    return frag


def export_after_xml(args, xml, subject, source, head, target, recorded):
    """Stream NiCad XML to J-03a schema; SQLite dedup avoids materializing millions of pairs."""
    digest = hfile(xml)
    db = sqlite3.connect(target / 'location_pairs.sqlite')
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('CREATE TABLE pairs (k TEXT PRIMARY KEY, pair_id TEXT UNIQUE)')
    db.execute('CREATE TABLE fragments (fragment_id TEXT PRIMARY KEY, repository TEXT, file_path TEXT, start_line INTEGER, end_line INTEGER, commit_sha TEXT, source_sha256 TEXT, language TEXT)')
    total = duplicates = 0
    try:
        with (target / 'clone_pairs.csv').open('x', newline='', encoding='utf-8') as fd:
            writer = csv.DictWriter(fd, fieldnames=PAIR_FIELDS)
            writer.writeheader()
            context = ET.iterparse(xml, events=('start', 'end'))
            _, root = next(context)
            declared_pairs = None
            declared_threshold = None
            for event, elem in context:
                if event != 'end':
                    continue
                if elem.tag == 'systeminfo':
                    declared_threshold = elem.attrib.get('threshold')
                if elem.tag == 'cloneinfo':
                    declared_pairs = int(elem.attrib['npairs'])
                if elem.tag != 'clone':
                    continue
                total += 1
                sources = elem.findall('source')
                if len(sources) != 2:
                    raise RuntimeError(f'Invalid clone #{total}: expected exactly 2 sources')
                frags = []
                for src in sources:
                    rel = normalize_file(src.attrib['file'], subject)
                    start, end = int(src.attrib['startline']), int(src.attrib['endline'])
                    frag = fragment_for(source, rel, start, end, subject, head, args.language)
                    frags.append(frag)
                # stable ordering uses fragment IDs, same as J-03a.
                frags.sort(key=lambda f: f['fragment_id'])
                a, b = frags
                if a['fragment_id'] == b['fragment_id']:
                    raise RuntimeError(f'Self clone at XML clone index {total}')
                loc = location_key(endpoint(a['file_path'], a['start_line'], a['end_line']),
                                   endpoint(b['file_path'], b['start_line'], b['end_line']))
                pair_id = 'j03cp_' + hbytes(('\0'.join((a['fragment_id'], b['fragment_id'])) + '\0' + args.config).encode('utf-8'))[:24]
                try:
                    db.execute('INSERT INTO pairs VALUES (?,?)', (loc, pair_id))
                except sqlite3.IntegrityError:
                    # Duplicate locations are tolerable, but an ID collision across
                    # distinct locations must never be silently dropped.
                    if db.execute('SELECT 1 FROM pairs WHERE k=?', (loc,)).fetchone() is None:
                        raise RuntimeError(f'PAIR_ID_COLLISION subject={subject} pair_id={pair_id}')
                    duplicates += 1
                    elem.clear(); root.clear()
                    continue
                for frag in frags:
                    db.execute('INSERT OR IGNORE INTO fragments VALUES (?,?,?,?,?,?,?,?)',
                               (frag['fragment_id'], frag['repository'], frag['file_path'],
                                frag['start_line'], frag['end_line'], frag['commit'],
                                frag['source_sha256'], frag['language']))
                writer.writerow({
                    'pair_id': pair_id, 'fragment_a_id': a['fragment_id'], 'repository_a': subject,
                    'file_a': a['file_path'], 'start_line_a': a['start_line'], 'end_line_a': a['end_line'],
                    'commit_a': head, 'source_sha256_a': a['source_sha256'],
                    'fragment_b_id': b['fragment_id'], 'repository_b': subject,
                    'file_b': b['file_path'], 'start_line_b': b['start_line'], 'end_line_b': b['end_line'],
                    'commit_b': head, 'source_sha256_b': b['source_sha256'], 'same_repository': 'true',
                    'nicad_similarity_raw': elem.attrib.get('similarity', ''),
                    'nicad_nlines': elem.attrib.get('nlines', ''), 'detector': 'NiCad7/functions/java',
                    'config': args.config, 'xml_sha256': digest
                })
                elem.clear(); root.clear()
                if total % args.progress_every == 0:
                    db.commit()
                    print(f'AFTER_PARSE_PROGRESS subject={subject} xml_nodes={total} duplicates={duplicates}', flush=True)
        if declared_pairs is None or declared_pairs != total:
            raise RuntimeError(f'NiCad cloneinfo count mismatch: declared={declared_pairs} observed={total}')
        expected_percent = format(float(args.threshold) * 100, 'g') + '%'
        if declared_threshold != expected_percent:
            raise RuntimeError(f'NiCad threshold mismatch: XML={declared_threshold} expected={expected_percent}')
        db.commit()
        fragments = db.execute('SELECT COUNT(*) FROM fragments').fetchone()[0]
        pairs = db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]
        with (target / 'clone_fragments.csv').open('x', newline='', encoding='utf-8') as fd:
            w = csv.DictWriter(fd, fieldnames=FRAGMENT_FIELDS)
            w.writeheader()
            for r in db.execute('SELECT * FROM fragments ORDER BY fragment_id'):
                w.writerow(dict(zip(FRAGMENT_FIELDS, r)))
        with (target / 'repository_clone_counts.csv').open('x', newline='', encoding='utf-8') as fd:
            w = csv.writer(fd)
            w.writerow(['repository', 'pair_endpoints', 'within_repo_pairs', 'cross_repo_pair_endpoints'])
            w.writerow([subject, 2 * pairs, pairs, 0])
    finally:
        db.close()
        source_lines.cache_clear()
        fragment_for.cache_clear()
    for rel in ('clone_pairs.csv', 'clone_fragments.csv', 'repository_clone_counts.csv', 'location_pairs.sqlite'):
        recorded[rel] = hfile(target / rel)
    return {'xml_clone_nodes': total, 'duplicate_xml_pairs': duplicates, 'exported_unique_pairs': pairs,
            'unique_fragments': fragments, 'nicad_pair_xml_sha256': digest}


def comparison(before_csv, after_db, output):
    db = sqlite3.connect(after_db)
    try:
        db.execute('CREATE TABLE baseline (k TEXT PRIMARY KEY)')
        old_rows = 0
        with before_csv.open(newline='', encoding='utf-8') as fd:
            reader = csv.DictReader(fd)
            for row in reader:
                if not {'file_a', 'file_b', 'start_line_a', 'end_line_a', 'start_line_b', 'end_line_b'} <= set(row):
                    raise RuntimeError('Unexpected baseline CSV schema')
                k = location_key(endpoint(row['file_a'], row['start_line_a'], row['end_line_a']),
                                 endpoint(row['file_b'], row['start_line_b'], row['end_line_b']))
                db.execute('INSERT OR IGNORE INTO baseline VALUES (?)', (k,))
                old_rows += 1
                if old_rows % 20000 == 0:
                    db.commit()
        db.commit()
        old_unique = db.execute('SELECT COUNT(*) FROM baseline').fetchone()[0]
        new_unique = db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]
        old_only = db.execute('SELECT COUNT(*) FROM baseline LEFT JOIN pairs USING(k) WHERE pairs.k IS NULL').fetchone()[0]
        new_only = db.execute('SELECT COUNT(*) FROM pairs LEFT JOIN baseline USING(k) WHERE baseline.k IS NULL').fetchone()[0]
        sample_old = [x[0] for x in db.execute('SELECT baseline.k FROM baseline LEFT JOIN pairs USING(k) WHERE pairs.k IS NULL LIMIT 15')]
        sample_new = [x[0] for x in db.execute('SELECT pairs.k FROM pairs LEFT JOIN baseline USING(k) WHERE baseline.k IS NULL LIMIT 15')]
        result = {'comparison_scope': 'unordered clone pair of repository-relative file:inclusive_start-end locations',
                  'baseline_pair_rows': old_rows, 'preprocessed_pair_rows': new_unique,
                  'baseline_unique_location_pairs': old_unique, 'preprocessed_unique_location_pairs': new_unique,
                  'baseline_only_pairs': old_only, 'preprocessed_only_pairs': new_only,
                  'sample_baseline_only': sample_old, 'sample_preprocessed_only': sample_new,
                  'identical_location_pair_set': (old_rows == old_unique == new_unique and old_only == new_only == 0),
                  'baseline_csv_sha256': hfile(before_csv), 'timestamp_chicago': chicago()}
        fresh_json(output, result)
        return result
    finally:
        db.close()


def discover_parse_failures(xml):
    # XML parent is a NiCad clone-output directory; the log is a sibling directory entry.
    logs = list(xml.parent.parent.glob('*_functions-clones-*.log'))
    issues = Counter()
    for f in logs:
        txt = f.read_text(encoding='utf-8', errors='replace')
        issues['TXL0191E'] += txt.count('TXL0191E')
        issues['TXL0192E'] += txt.count('TXL0192E')
    return dict(issues), sum(issues.values()), logs


def publish_after(args, subject):
    head = git_head(args.repo_root / subject, args.require_clean)
    dest = args.after_root / subject
    if existing(dest, args, 'after', subject, head):
        m = load_json(dest / 'archive_manifest.json')
        return m.get('identical_location_pair_set') is True
    before = args.before_root / subject
    if not existing(before, args, 'before', subject, head):
        raise RuntimeError(f'BEFORE_ARCHIVE_REQUIRED subject={subject}')
    source, mp, preproc_meta = preproc_info(args, subject, head)
    orig_attempt, orig, original_manifest, _ = original_attempt(args, subject, head)
    old_filter = original_manifest.get('source_filter_policy', {})
    if (preproc_meta.get('skip_dirs') != old_filter.get('excluded_directories')
            or preproc_meta.get('skip_globs') != old_filter.get('excluded_file_globs')
            or preproc_meta.get('in_file_test_policy') != old_filter.get('embedded_test_policy')):
        raise RuntimeError(f'J03A_J03B_TEST_POLICY_MISMATCH subject={subject}')
    xml = Path(args.nicad_xml).resolve(strict=True) if args.nicad_xml else None
    if xml is None or not xml.is_file() or not xml.name.endswith(f'-clones-{args.threshold}.xml'):
        raise RuntimeError(f'NiCad XML missing or threshold mismatch: {xml}')
    # Reused J-03b smoke result may only be accepted with its original recorded XML hash.
    if args.legacy_comparison_file:
        c = load_json(args.legacy_comparison_file)
        if (c.get('subject') != subject or c.get('baseline_sha256') != hfile(before / 'clone_pairs.csv')
                or c.get('preprocessed_nicad_xml_sha256') != hfile(xml)):
            raise RuntimeError('J03B_LEGACY_SMOKE_PROVENANCE_MISMATCH')
    issues, errors, logs = discover_parse_failures(xml)
    if errors and args.strict_parse:
        raise RuntimeError(f'NiCad extraction had {errors} parse failures; strict mode')
    def op(tmp):
        recorded = {}
        publish_file(xml, tmp, 'nicad/pairs.xml', recorded)
        optional_nicad_xml(xml, tmp, recorded, args)
        for ix, log in enumerate(logs):
            publish_file(log, tmp, f'nicad/nicad-run-{ix}.log', recorded)
        publish_file(mp, tmp, 'preproc_manifest.json', recorded)
        fields = export_after_xml(args, xml, subject, source, head, tmp, recorded)
        result = comparison(before / 'clone_pairs.csv', tmp / 'location_pairs.sqlite', tmp / 'comparison.json')
        recorded['comparison.json'] = hfile(tmp / 'comparison.json')
        recorded['location_pairs.sqlite'] = hfile(tmp / 'location_pairs.sqlite')
        summary = {'experiment': args.experiment, 'version': args.version, 'phase': 'after',
                   'subject': subject, 'timestamp_chicago': chicago(),
                   'status': 'NICAD_XML_PARSED' if not errors else 'PARSE_WARNINGS_REVIEW_REQUIRED',
                   'source_preproc_manifest_sha256': hfile(mp), 'source_j03a_summary_sha256': hfile(orig_attempt / 'run_summary.json'),
                   'parse_failure_codes': issues, 'parse_failures_detected_in_nicad_logs': errors,
                   **fields, 'identical_location_pair_set': result['identical_location_pair_set']}
        fresh_json(tmp / 'run_summary.json', summary)
        recorded['run_summary.json'] = hfile(tmp / 'run_summary.json')
        return {'experiment': args.experiment, 'version': args.version, 'phase': 'after',
                'timestamp_chicago': chicago(), 'status': 'ARCHIVED_VERIFIED', 'subject': subject,
                'git_head': head, 'config': args.config, 'threshold': args.threshold,
                'preproc_version': args.preproc_version, 'baseline_version': args.baseline_version,
                'preproc_manifest_sha256': hfile(mp), 'source_nicad_xml': str(xml),
                'source_nicad_xml_sha256': hfile(xml), 'clone_pairs': fields['exported_unique_pairs'],
                'clone_fragments': fields['unique_fragments'], **settings(args), 'parse_failures': errors,
                'baseline_pairs': result['baseline_pair_rows'],
                'baseline_only_pairs': result['baseline_only_pairs'],
                'preprocessed_only_pairs': result['preprocessed_only_pairs'],
                'identical_location_pair_set': result['identical_location_pair_set'],
                'files_sha256': recorded}
    path, m = atomic_folder(args.after_root, subject, op)
    print(f'AFTER_PASS subject={subject} pairs={m["clone_pairs"]} baseline={m["baseline_pairs"]} '
          f'IDENTICAL={str(m["identical_location_pair_set"]).lower()} '
          f'removed={m["baseline_only_pairs"]} added={m["preprocessed_only_pairs"]} path={path}', flush=True)
    return m['identical_location_pair_set']


def summarize(args):
    """Run-local CSV/JSON snapshot of publication coverage and comparison differences."""
    subjects = subject_list(args)
    rows = []
    totals = Counter()
    for subject in subjects:
        rec = {'subject': subject}
        for phase, root in [('before', args.before_root), ('after', args.after_root)]:
            mp = root / subject / 'archive_manifest.json'
            if not mp.is_file():
                rec[phase + '_status'] = 'MISSING'
                rec[phase + '_pairs'] = ''
                continue
            m = load_json(mp)
            if (m.get('experiment'), m.get('subject'), m.get('phase'), m.get('version')) != (args.experiment, subject, phase, args.version):
                rec[phase + '_status'] = 'MANIFEST_MISMATCH'
                rec[phase + '_pairs'] = ''
                continue
            rec[phase + '_status'] = m['status']
            rec[phase + '_pairs'] = m['clone_pairs']
            totals[phase + '_complete'] += 1
            totals[phase + '_pairs'] += m['clone_pairs']
            if phase == 'after':
                rec['baseline_only'] = m['baseline_only_pairs']
                rec['preprocessed_only'] = m['preprocessed_only_pairs']
                rec['identical'] = m['identical_location_pair_set']
                rec['after_parse_failures'] = m['parse_failures']
                totals['identical' if m['identical_location_pair_set'] else 'different'] += 1
                totals['after_parse_failures'] += m['parse_failures']
            else:
                rec['before_parse_failures'] = m['parse_failures']
                totals['before_parse_failures'] += m['parse_failures']
        rows.append(rec)
    fields = ['subject', 'before_status', 'before_pairs', 'before_parse_failures',
              'after_status', 'after_pairs', 'after_parse_failures',
              'identical', 'baseline_only', 'preprocessed_only']
    csvpath = args.report_dir / 'subject_archive_summary.csv'
    with csvpath.open('x', newline='', encoding='utf-8') as fd:
        w = csv.DictWriter(fd, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    report = {'experiment': args.experiment, 'version': args.version,
              'timestamp_chicago': chicago(), 'expected_subjects': len(subjects),
              'before_complete': totals['before_complete'],
              'after_complete': totals['after_complete'],
              'before_clone_pairs': totals['before_pairs'],
              'after_clone_pairs': totals['after_pairs'],
              'identical_subjects': totals['identical'],
              'different_subjects': totals['different'],
              'before_parse_failures': totals['before_parse_failures'],
              'after_parse_failures': totals['after_parse_failures'],
              'summary_csv_sha256': hfile(csvpath),
              'note': 'Coverage summary from immutable archive manifests; full hash verification occurs when publishing/reusing individual archives.'}
    fresh_json(args.report_dir / 'archive_run_summary.json', report)
    print(f'ARCHIVE_SUMMARY before={report["before_complete"]}/{len(subjects)} '
          f'after={report["after_complete"]}/{len(subjects)} '
          f'before_pairs={report["before_clone_pairs"]} after_pairs={report["after_clone_pairs"]} '
          f'identical={report["identical_subjects"]} different={report["different_subjects"]} '
          f'path={args.report_dir}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    required = ('experiment', 'version', 'mode', 'subject', 'repo-root', 'preproc-root', 'j03a-run',
                'before-root', 'after-root', 'report-dir', 'expected-subjects', 'config', 'threshold',
                'granularity', 'language', 'preproc-version', 'baseline-version', 'unicode-policy', 'require-clean',
                'strict-parse', 'copy-class-xml', 'progress-every')
    for name in required:
        p.add_argument('--' + name, required=True)
    p.add_argument('--nicad-xml')
    p.add_argument('--legacy-comparison-file')
    a = p.parse_args()
    for key in ('repo_root', 'preproc_root', 'j03a_run', 'before_root', 'after_root', 'report_dir'):
        setattr(a, key, Path(getattr(a, key)).resolve())
    a.expected_subjects = int(a.expected_subjects)
    a.progress_every = int(a.progress_every)
    for key in ('require_clean', 'strict_parse', 'copy_class_xml'):
        v = getattr(a, key).lower()
        if v not in ('true', 'false'):
            p.error(f'{key} must be true or false')
        setattr(a, key, v == 'true')
    if a.expected_subjects < 1 or a.progress_every < 1:
        p.error('Expected subjects and progress every must be positive')
    if a.mode not in ('inspect', 'preflight-after', 'before', 'after', 'verify-before', 'verify-after', 'summarize'):
        p.error('Invalid mode')
    if a.mode not in ('inspect', 'preflight-after', 'summarize') and not NAME.fullmatch(a.subject):
        p.error('Invalid subject')
    if a.mode == 'inspect':
        names = subject_list(a)
        print(f'INVENTORY_PASS subjects={len(names)} raw={a.repo_root} before={a.before_root} after={a.after_root}', flush=True)
    elif a.mode == 'preflight-after':
        names = subject_list(a)
        for name in names:
            head = git_head(a.repo_root / name, a.require_clean)
            if not existing(a.before_root / name, a, 'before', name, head):
                raise RuntimeError(f'BEFORE_ARCHIVE_REQUIRED subject={name}; run --before-all first')
            preproc_info(a, name, head)
        print(f'PREPROC_AND_BEFORE_PREFLIGHT_PASS subjects={len(names)}', flush=True)
    elif a.mode == 'summarize':
        summarize(a)
    elif a.mode == 'before':
        publish_before(a, a.subject)
    elif a.mode == 'after':
        if not publish_after(a, a.subject):
            sys.exit(3)
    elif a.mode.startswith('verify-'):
        phase = a.mode.split('-', 1)[1]
        head = git_head(a.repo_root / a.subject, a.require_clean)
        target = (a.before_root if phase == 'before' else a.after_root) / a.subject
        if not existing(target, a, phase, a.subject, head):
            raise RuntimeError(f'ARCHIVE_NOT_FOUND {target}')


if __name__ == '__main__':
    main()
