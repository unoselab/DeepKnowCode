#!/usr/bin/env python3
"""J-03b v3: immutable Java test exclusion + byte/line-preserving comment masking and NiCad comparison."""
import argparse
import csv
import fnmatch
import html
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


def now():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat(timespec='seconds')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json_new(path, obj):
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write('\n')


def git(repo, *args):
    p = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, check=True)
    return p.stdout.decode('utf-8', errors='replace').strip()


def java_tracked(repo):
    data = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z', '--', '*.java'])
    for item in data.split(b'\0'):
        if item:
            rel = Path(os.fsdecode(item))
            if rel.is_absolute() or '..' in rel.parts or rel.suffix != '.java':
                raise ValueError(f'Unsafe Git path: {rel}')
            yield rel


def exclude(rel, args, path):
    for part in rel.parts[:-1]:
        if part.casefold() in args.skip_dir:
            return 'skip_dir:' + part.casefold()
    for pat in args.skip_glob:
        if fnmatch.fnmatchcase(rel.name.casefold(), pat.casefold()):
            return 'skip_file_glob:' + pat
    if path.is_symlink():
        return 'symlink'
    return ''


def mask_java_comments(raw, unicode_escape_policy):
    """Mask comments and preserve byte/CR/LF offsets without guessing Java Unicode lexing.

    Java Unicode escapes are translated before tokenization, even inside comments.
    Rather than rewrite ambiguous Java code or silently corrupt it, an explicitly
    configured conservative policy retains the *entire file byte-identical* and
    records an auditable exception. Other files undergo byte/line-preserving masking.
    """
    if unicode_escape_policy not in ('preserve_original_audited', 'fail_closed'):
        raise ValueError('Invalid JAVA_UNICODE_ESCAPE_POLICY')
    # Conservatively match every raw backslash-u sequence, including escapes in
    # documentation, strings and comments. Some aren't JLS-eligible escapes, but
    # retaining extra files is safer than incorrectly stripping a Java token.
    occurrences = len(re.findall(rb'\\u+', raw))
    if occurrences:
        if unicode_escape_policy == 'fail_closed':
            raise ValueError('JAVA_UNICODE_ESCAPE_REQUIRES_LEXICAL_REVIEW')
        return raw, Counter({'unicode_passthrough_files': 1,
                             'unicode_escape_candidates': occurrences})
    a = bytearray(raw)
    i = 0
    mode = 'code'
    counts = Counter()
    n = len(a)

    def blank(j):
        if a[j] not in (10, 13):
            a[j] = 32

    while i < n:
        c = raw[i]
        nxt = raw[i+1] if i+1 < n else -1
        if mode == 'code':
            if c == 47 and nxt == 47:
                mode = 'line'; counts['line_comments'] += 1
                blank(i); blank(i+1); i += 2; continue
            if c == 47 and nxt == 42:
                mode = 'block'; counts['block_comments'] += 1
                blank(i); blank(i+1); i += 2; continue
            if raw[i:i+3] == b'"""':
                mode = 'textblock'; i += 3; continue
            if c == 34:
                mode = 'string'; i += 1; continue
            if c == 39:
                mode = 'char'; i += 1; continue
            i += 1; continue
        if mode == 'line':
            if c in (10, 13):
                mode = 'code'
            else:
                blank(i)
            i += 1; continue
        if mode == 'block':
            if c == 42 and nxt == 47:
                blank(i); blank(i+1); mode = 'code'; i += 2
            else:
                blank(i); i += 1
            continue
        if mode in ('string', 'char', 'textblock'):
            if c == 92:  # escaped character, including escaped quote
                i += 2; continue
            if mode == 'textblock' and raw[i:i+3] == b'"""':
                mode = 'code'; i += 3; continue
            if (mode == 'string' and c == 34) or (mode == 'char' and c == 39):
                mode = 'code'
            # Don't reject malformed code here: NiCad itself reports parse failure.
            i += 1; continue
    if mode == 'block':
        raise ValueError('UNTERMINATED_BLOCK_COMMENT')
    if len(a) != n or bytes(a).count(b'\n') != raw.count(b'\n') or bytes(a).count(b'\r') != raw.count(b'\r'):
        raise AssertionError('Byte/line preservation failure')
    return bytes(a), counts


def inventory(args):
    records = []
    for repo in sorted(args.repo_root.iterdir()):
        if not repo.is_dir() or not (repo / '.git').exists():
            continue
        head = git(repo, 'rev-parse', 'HEAD')
        if args.require_clean and git(repo, 'status', '--porcelain', '--untracked-files=all'):
            raise RuntimeError(f'Git dirty: {repo}')
        counts = Counter()
        for rel in java_tracked(repo):
            reason = exclude(rel, args, repo / rel)
            counts['excluded' if reason else 'included'] += 1
        records.append({'subject': repo.name, 'git_head': head, **dict(counts)})
    if len(records) != args.expected_subjects:
        raise RuntimeError(f'Repository count mismatch: {len(records)} != {args.expected_subjects}')
    totals = dict(sum_counters([Counter({'included': r.get('included', 0), 'excluded': r.get('excluded', 0)}) for r in records]))
    print(f'INVENTORY PASS subjects={len(records)} included={totals.get("included",0)} excluded={totals.get("excluded",0)}', flush=True)
    return records


def sum_counters(cs):
    result = Counter()
    for c in cs:
        result.update(c)
    return result


def prepare(args, subject):
    repo = args.repo_root / subject
    if not repo.is_dir() or not (repo / '.git').exists():
        raise ValueError(f'Unknown subject: {subject}')
    if args.require_clean and git(repo, 'status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError(f'Git dirty: {repo}')
    head = git(repo, 'rev-parse', 'HEAD')
    final = args.preproc_root / subject
    if final.exists() or final.is_symlink():
        raise FileExistsError(f'IMMUTABLE_TARGET_EXISTS: {final}; do not overwrite')
    args.preproc_root.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=f'.{subject}.pending-', dir=args.preproc_root))
    counts = Counter(); audits = []
    try:
        for rel in java_tracked(repo):
            original = repo / rel
            why = exclude(rel, args, original)
            record = {'file': rel.as_posix(), 'status': 'EXCLUDED' if why else 'INCLUDED', 'reason': why,
                      'raw_sha256': '', 'preproc_sha256': '', 'bytes': '', 'comment_line': 0, 'comment_block': 0,
                      'comment_mask_status': 'EXCLUDED' if why else '', 'unicode_escape_candidates': 0}
            if why:
                counts['excluded'] += 1
                counts['skip_' + why] += 1
            else:
                raw = original.read_bytes()
                try:
                    transformed, c = mask_java_comments(raw, args.java_unicode_escape_policy)
                except ValueError as exc:
                    raise ValueError(f'{subject}/{rel}: {exc}') from exc
                if original.read_bytes() != raw:
                    raise RuntimeError(f'Source changed mid-read: {original}')
                target = pending / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, 'xb') as f:
                    f.write(transformed)
                if sha_file(target) != sha(transformed):
                    raise RuntimeError(f'Post-write hash mismatch: {target}')
                passthrough = bool(c['unicode_passthrough_files'])
                mask_status = ('PASSTHROUGH_UNICODE_ESCAPE' if passthrough else
                               'MASKED' if raw != transformed else 'UNCHANGED_NO_COMMENTS')
                record.update({'raw_sha256': sha(raw), 'preproc_sha256': sha(transformed),
                               'bytes': len(raw), 'comment_line': c['line_comments'],
                               'comment_block': c['block_comments'], 'comment_mask_status': mask_status,
                               'unicode_escape_candidates': c['unicode_escape_candidates']})
                counts.update({'included': 1, 'bytes': len(raw), 'line_comments': c['line_comments'],
                               'block_comments': c['block_comments'], 'changed_files': int(raw != transformed),
                               'unicode_passthrough_files': int(passthrough),
                               'unicode_passthrough_bytes': len(raw) if passthrough else 0,
                               'unicode_escape_candidates': c['unicode_escape_candidates']})
                if passthrough:
                    print(f'UNICODE_PASSTHROUGH subject={subject} file={rel.as_posix()} '
                          f'candidates={c["unicode_escape_candidates"]} raw_sha256={sha(raw)}', flush=True)
                if counts['included'] % args.progress_every == 0:
                    print(f'PREPROC_PROGRESS subject={subject} included={counts["included"]}', flush=True)
            audits.append(record)
        if git(repo, 'rev-parse', 'HEAD') != head:
            raise RuntimeError('Git HEAD changed during preprocessing')
        audit_path = args.report_dir / f'{subject}-file-audit.csv'
        with audit_path.open('x', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(audits[0].keys()))
            w.writeheader(); w.writerows(audits)
        manifest = {'experiment': args.experiment, 'version': args.version, 'timestamp_chicago': now(),
                    'subject': subject, 'raw_source_root': str(repo.resolve()), 'preproc_source_root': str(final.resolve()),
                    'git_head': head, 'source_policy': 'Git-tracked Java HEAD only; test path exclusion; comment byte masking with audited Unicode fallback',
                    'comment_policy': 'Byte/line-preserving masking except audited byte-identical Unicode-escape files',
                    'java_unicode_escape_policy': args.java_unicode_escape_policy, 'in_file_test_policy': args.embedded_test_policy,
                    'skip_dirs': args.skip_dir, 'skip_globs': args.skip_glob,
                    'counts': dict(counts), 'audit_csv': str(audit_path.resolve()),
                    'audit_sha256': sha_file(audit_path), 'status': 'PREPARED'}
        write_json_new(args.report_dir / f'{subject}-preproc-manifest.json', manifest)
        write_json_new(pending / '.j03b-preproc-manifest.json', manifest)
        os.rename(pending, final)
        print(f'PREPROC_PASS subject={subject} included={counts["included"]} excluded={counts["excluded"]} changed={counts["changed_files"]} unicode_passthrough={counts["unicode_passthrough_files"]} path={final}', flush=True)
    except BaseException:
        # Avoid erasing diagnostic files; pending tree remains for audit/recovery.
        print(f'PREPROC_FAIL subject={subject} partial={pending}', file=sys.stderr, flush=True)
        raise



def verify_existing(args, subject):
    final = args.preproc_root / subject
    manifest_path = final / '.j03b-preproc-manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError(f'Existing subject has no provenance manifest: {final}')
    m = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = {'experiment': args.experiment, 'version': args.version,
                'java_unicode_escape_policy': args.java_unicode_escape_policy,
                'skip_dirs': args.skip_dir, 'skip_globs': args.skip_glob,
                'in_file_test_policy': args.embedded_test_policy, 'status': 'PREPARED'}
    for k, v in expected.items():
        if m.get(k) != v:
            raise RuntimeError(f'PREPROC_POLICY_MISMATCH subject={subject} field={k} '
                               f'found={m.get(k)!r} required={v!r}; do not overwrite')
    if m.get('subject') != subject or m.get('git_head') != git(args.repo_root / subject, 'rev-parse', 'HEAD'):
        raise RuntimeError(f'PREPROC_GIT_HEAD_MISMATCH subject={subject}')
    # Every preserved file is represented in the original audit for traceability.
    audit_path = Path(m.get('audit_csv', ''))
    if not audit_path.is_file() or sha_file(audit_path) != m.get('audit_sha256'):
        raise RuntimeError(f'PREPROC_AUDIT_MISSING_OR_CHANGED subject={subject}')
    verified = 0
    with audit_path.open(newline='', encoding='utf-8') as f:
        for rec in csv.DictReader(f):
            if rec['status'] != 'INCLUDED':
                continue
            rel = Path(rec['file'])
            if rel.is_absolute() or '..' in rel.parts or rel.suffix != '.java':
                raise RuntimeError(f'PREPROC_AUDIT_INVALID_PATH subject={subject} file={rel}')
            target = final / rel
            if not target.is_file() or target.is_symlink() or sha_file(target) != rec['preproc_sha256']:
                raise RuntimeError(f'PREPROC_FILE_HASH_MISMATCH subject={subject} file={rel}')
            verified += 1
    if verified != m.get('counts', {}).get('included'):
        raise RuntimeError(f'PREPROC_FILE_COUNT_MISMATCH subject={subject} checked={verified}')
    print(f'PREPROC_EXISTING_VERIFIED subject={subject} version={args.version} '
          f'files={verified} path={final}', flush=True)


def canonical_key(a, b):
    return '\x1f'.join(sorted([a, b]))


def endpoint(file, start, end, subject):
    path = file.replace('\\', '/')
    if '/staged_sources/' in path:
        path = path.split('/staged_sources/', 1)[1]
    elif '/nicadclones/' in path:
        path = path.split('/nicadclones/', 1)[1]
        marker = '/' + subject + '/'
        if marker in path:
            path = subject + '/' + path.split(marker, 1)[1]
    else:
        idx = path.find(subject + '/')
        if idx >= 0:
            path = path[idx:]
        elif path.startswith('/'):
            raise ValueError(f'Cannot resolve subject in source: {file}')
        # J-03a CSV file_a/file_b already hold a repository-relative path.
    if path.startswith(subject + '/'):
        path = path[len(subject)+1:]
    elif path.startswith(subject + '/' + subject + '/'):
        path = path[len(subject)*2+2:]
    if path.startswith(subject + '/'):
        path = path[len(subject)+1:]
    if not path or '..' in Path(path).parts:
        raise ValueError(f'Invalid source path: {file}')
    return f'{path}:{int(start)}-{int(end)}'


def sql_db(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE original (k TEXT PRIMARY KEY)')
    db.execute('CREATE TABLE modified (k TEXT PRIMARY KEY)')
    return db


def pair_rows_from_xml(xml, subject):
    for _, node in ET.iterparse(xml, events=('end',)):
        if node.tag != 'clone':
            continue
        refs = node.findall('source')
        if len(refs) != 2:
            raise RuntimeError('Invalid NiCad clone source cardinality')
        a,b = [endpoint(x.attrib['file'], x.attrib['startline'], x.attrib['endline'], subject) for x in refs]
        yield canonical_key(a, b)
        node.clear()


def pair_rows_from_csv(csvpath, subject):
    with open(csvpath, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            a = endpoint(r['file_a'], r['start_line_a'], r['end_line_a'], subject)
            b = endpoint(r['file_b'], r['start_line_b'], r['end_line_b'], subject)
            yield canonical_key(a, b)


def compare(args, subject, nicad_xml, baseline_csv):
    if not baseline_csv.is_file() or not nicad_xml.is_file():
        raise FileNotFoundError(f'Missing baseline or NiCad XML: {baseline_csv} {nicad_xml}')
    # Require that the preprocessed subject and the J-03a baseline refer to the same Git HEAD.
    baseline_attempt = baseline_csv.parent
    original_manifest_path = baseline_attempt / 'run_manifest.json'
    if original_manifest_path.is_file():
        baseline_meta = json.loads(original_manifest_path.read_text(encoding='utf-8'))
        heads = [r['head_commit'] for r in baseline_meta.get('git_repositories', [])
                 if r.get('repository') == subject]
        if len(heads) != 1 or git(args.repo_root / subject, 'rev-parse', 'HEAD') != heads[0]:
            raise RuntimeError('J-03a baseline HEAD differs from active raw repository')
        pol = baseline_meta.get('source_filter_policy', {})
        if pol.get('excluded_directories') != args.skip_dir or pol.get('excluded_file_globs') != args.skip_glob:
            raise RuntimeError('Baseline test filter policy differs from J-03b')
    elif args.expected_subjects != 1:
        raise RuntimeError(f'Missing J-03a per-subject baseline manifest: {original_manifest_path}')
    if not (args.preproc_root / subject).is_dir():
        raise RuntimeError(f'Preprocessed subject does not exist: {subject}')
    preproc_meta_path = args.preproc_root / subject / '.j03b-preproc-manifest.json'
    if not preproc_meta_path.is_file():
        raise RuntimeError(f'Missing immutable J-03b preproc manifest: {preproc_meta_path}')
    preproc_meta = json.loads(preproc_meta_path.read_text(encoding='utf-8'))
    if preproc_meta['git_head'] != git(args.repo_root / subject, 'rev-parse', 'HEAD'):
        raise RuntimeError('Preprocessed source was built from a different Git HEAD')
    if preproc_meta.get('version') != args.version or preproc_meta.get('java_unicode_escape_policy') != args.java_unicode_escape_policy:
        raise RuntimeError('J-03b preprocessed source version / Unicode fallback policy mismatch')
    if preproc_meta['skip_dirs'] != args.skip_dir or preproc_meta['skip_globs'] != args.skip_glob:
        raise RuntimeError('Preprocessed source filter differs from the configured policy')
    dbpath = args.report_dir / f'{subject}-comparison.sqlite'
    db = sql_db(dbpath)
    try:
        orig_total = 0
        batch_old = []
        for k in pair_rows_from_csv(baseline_csv, subject):
            batch_old.append((k,)); orig_total += 1
            if len(batch_old) == 10000:
                db.executemany('INSERT OR IGNORE INTO original VALUES (?)', batch_old)
                db.commit(); batch_old.clear()
        if batch_old:
            db.executemany('INSERT OR IGNORE INTO original VALUES (?)', batch_old); db.commit()
        new_total = 0
        batch = []
        for k in pair_rows_from_xml(nicad_xml, subject):
            batch.append((k,)); new_total += 1
            if len(batch) == 10000:
                db.executemany('INSERT OR IGNORE INTO modified VALUES (?)', batch)
                db.commit(); batch.clear()
        if batch:
            db.executemany('INSERT OR IGNORE INTO modified VALUES (?)', batch); db.commit()
        old_unique = db.execute('SELECT COUNT(*) FROM original').fetchone()[0]
        new_unique = db.execute('SELECT COUNT(*) FROM modified').fetchone()[0]
        only_old = db.execute('SELECT COUNT(*) FROM original LEFT JOIN modified USING(k) WHERE modified.k IS NULL').fetchone()[0]
        only_new = db.execute('SELECT COUNT(*) FROM modified LEFT JOIN original USING(k) WHERE original.k IS NULL').fetchone()[0]
        sample_missing = [r[0] for r in db.execute('SELECT original.k FROM original LEFT JOIN modified USING(k) WHERE modified.k IS NULL LIMIT 12')]
        sample_added = [r[0] for r in db.execute('SELECT modified.k FROM modified LEFT JOIN original USING(k) WHERE original.k IS NULL LIMIT 12')]
        result = {'experiment': args.experiment, 'version': args.version, 'subject': subject,
                  'timestamp_chicago': now(), 'baseline_csv': str(baseline_csv.resolve()),
                  'baseline_sha256': sha_file(baseline_csv), 'preprocessed_nicad_xml': str(nicad_xml.resolve()),
                  'preprocessed_nicad_xml_sha256': sha_file(nicad_xml),
                  'baseline_pair_rows': orig_total, 'preprocessed_pair_rows': new_total,
                  'baseline_unique_location_pairs': old_unique, 'preprocessed_unique_location_pairs': new_unique,
                  'baseline_only_pairs': only_old, 'preprocessed_only_pairs': only_new,
                  'sample_baseline_only': sample_missing, 'sample_preprocessed_only': sample_added,
                  'identical_location_pair_set': (only_old == 0 and only_new == 0 and orig_total == new_total),
                  'comparison_scope': 'Canonical repository-relative file/start/end clone pair locations; independent of source_sha256 and pair_id',
                  'note': 'NiCad native XML contains no reference to existing J-03a source SHA; differences in source hashes are expected'}
        write_json_new(args.report_dir / f'{subject}-comparison.json', result)
        print(f'COMPARE_RESULT subject={subject} IDENTICAL={str(result["identical_location_pair_set"]).lower()} baseline_pairs={orig_total} preproc_pairs={new_total} removed={only_old} added={only_new}', flush=True)
        return result['identical_location_pair_set']
    finally:
        db.close()




JAVA_HTML_KEYWORDS = frozenset('abstract assert boolean break byte case catch char class const continue default do double else enum extends final finally float for goto if implements import instanceof int interface long native new package private protected public record return sealed permits non-sealed short static strictfp super switch synchronized this throw throws transient try var void volatile while yield true false null'.split())
JAVA_HTML_TYPES = frozenset('String Object Integer Long Double Float Boolean Character Byte Short Class Optional List Map Set Collection Iterable Stream Path Files IOException Exception RuntimeException'.split())
JAVA_HTML_TOKEN = re.compile(r'//|/\*|"""|"|\x27|@[A-Za-z_$][\w$]*|\b(?:0[xX][0-9a-fA-F_]+|0[bB][01_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eEpP][+-]?\d+)?[fFdDlL]?)\b|[A-Za-z_$][\w$]*')


def highlight_java_lines(lines):
    """Escape untrusted Java source, decorate tokens, and retain per-line state.

    All highlighting is display-only; byte-preserving preprocessed files are never changed.
    """
    state = 'code'
    rendered = []
    for line in lines:
        parts = []
        i = 0
        while i < len(line):
            if state == 'block':
                end = line.find('*/', i)
                j = len(line) if end < 0 else end + 2
                parts.append('<span class="tok-comment">'+html.escape(line[i:j])+'</span>')
                i = j
                if end >= 0: state = 'code'
                continue
            if state in ('string','char','textblock'):
                terminator = {'string':'"','char':"'",'textblock':'"""'}[state]
                j = i
                while j < len(line):
                    if line[j] == '\\':
                        j += min(2, len(line)-j)
                    elif line.startswith(terminator,j):
                        j += len(terminator)
                        state = 'code'
                        break
                    else:
                        j += 1
                parts.append('<span class="tok-string">'+html.escape(line[i:j])+'</span>')
                i = j
                continue
            match = JAVA_HTML_TOKEN.search(line,i)
            if not match:
                parts.append(html.escape(line[i:]))
                break
            a,b=match.span()
            parts.append(html.escape(line[i:a]))
            token=match.group()
            if token == '//':
                parts.append('<span class="tok-comment">'+html.escape(line[a:])+'</span>')
                i = len(line)
                break
            if token == '/*':
                state='block';i=a;continue
            if token in ('"',"'",'"""'):
                state = {'"':'string',"'":'char','"""':'textblock'}[token]
                parts.append('<span class="tok-string">'+html.escape(token)+'</span>')
                i=b;continue
            if token.startswith('@'):
                cls='tok-annotation'
            elif token in JAVA_HTML_KEYWORDS:
                cls='tok-keyword'
            elif token in JAVA_HTML_TYPES:
                cls='tok-type'
            elif token[0].isdigit():
                cls='tok-number'
            else:
                cls=None
            parts.append('<span class="'+cls+'">'+html.escape(token)+'</span>' if cls else html.escape(token))
            i=b
        if not line:
            parts.append('')
        rendered.append(''.join(parts))
        # A standard string/char cannot continue onto a new Java source line.
        if state in ('string','char'):
            state='code'
    return rendered


def html_inspection(args, subject):
    """Produce immutable, locally browsable side-by-side full-source inspection."""
    final = args.preproc_root / subject
    manifest_file = final / '.j03b-preproc-manifest.json'
    if not manifest_file.is_file():
        raise RuntimeError(f'PREPROC_MANIFEST_MISSING subject={subject}')
    m = json.loads(manifest_file.read_text(encoding='utf-8'))
    if m.get('experiment') != args.experiment or m.get('subject') != subject or m.get('status') != 'PREPARED':
        raise RuntimeError(f'PREPROC_MANIFEST_INVALID subject={subject}')
    if m.get('version') != args.preproc_data_version:
        raise RuntimeError(f'PREPROC_DATA_VERSION_MISMATCH subject={subject}: expected={args.preproc_data_version} got={m.get("version")}')
    if m.get('java_unicode_escape_policy') != args.java_unicode_escape_policy or m.get('skip_dirs') != args.skip_dir or m.get('skip_globs') != args.skip_glob:
        raise RuntimeError(f'PREPROC_POLICY_MISMATCH subject={subject}')
    repo = args.repo_root / subject
    if git(repo, 'rev-parse', 'HEAD') != m.get('git_head'):
        raise RuntimeError(f'PREPROC_GIT_HEAD_MISMATCH subject={subject}')
    audit = Path(m.get('audit_csv', ''))
    if not audit.is_file() or sha_file(audit) != m.get('audit_sha256'):
        raise RuntimeError(f'PREPROC_AUDIT_MISSING_OR_CHANGED subject={subject}')
    with audit.open(newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r.get('status') == 'INCLUDED']
    if len(rows) != m.get('counts', {}).get('included'):
        raise RuntimeError(f'AUDIT_INCLUDED_COUNT_MISMATCH subject={subject}')
    rows.sort(key=lambda r: (-int(r['bytes']), r['file']))
    selected = rows[:args.html_top_n]
    if len(selected) != args.html_top_n:
        raise RuntimeError(f'NOT_ENOUGH_INCLUDED_FILES subject={subject} count={len(selected)} required={args.html_top_n}')
    outdir = args.report_dir / args.html_subdirectory
    outdir.mkdir(parents=True, exist_ok=True)
    outdir = outdir / subject
    outdir.mkdir(parents=True, exist_ok=True)
    payload = []
    audit_records = []
    for rank, r in enumerate(selected, 1):
        rel = Path(r['file'])
        if rel.is_absolute() or '..' in rel.parts or rel.suffix != '.java':
            raise RuntimeError(f'UNSAFE_AUDIT_PATH subject={subject} file={rel}')
        before_path = repo / rel
        after_path = final / rel
        if not before_path.is_file() or before_path.is_symlink() or not after_path.is_file() or after_path.is_symlink():
            raise RuntimeError(f'SOURCE_MISSING_OR_SYMLINK subject={subject} file={rel}')
        before, after = before_path.read_bytes(), after_path.read_bytes()
        if sha(before) != r['raw_sha256'] or sha(after) != r['preproc_sha256']:
            raise RuntimeError(f'SOURCE_SHA256_MISMATCH subject={subject} file={rel}')
        if len(before) != len(after) or len(before) != int(r['bytes']):
            raise RuntimeError(f'BYTE_LENGTH_MISMATCH subject={subject} file={rel}')
        if before.count(b'\n') != after.count(b'\n') or before.count(b'\r') != after.count(b'\r'):
            raise RuntimeError(f'LINE_ENDING_MISMATCH subject={subject} file={rel}')
        if r['comment_mask_status'] == 'PASSTHROUGH_UNICODE_ESCAPE' and before != after:
            raise RuntimeError(f'UNICODE_PASSTHROUGH_CHANGED subject={subject} file={rel}')
        # Decode only for display, not for byte comparison, preserving forensic hashes.
        left = before.decode('utf-8', errors='replace').split('\n')
        right = after.decode('utf-8', errors='replace').split('\n')
        if len(left) != len(right):
            raise RuntimeError(f'DISPLAY_LINE_COUNT_MISMATCH subject={subject} file={rel}')
        def code(lines):
            return ''.join(f'<div class="ln"><span class="num">{i}</span><span class="src">{line}</span></div>\n' for i,line in enumerate(highlight_java_lines(lines),1))
        status = r['comment_mask_status']
        caption = ('Unicode escape: original file preserved (comments may remain)' if status == 'PASSTHROUGH_UNICODE_ESCAPE' else
                   'Comments masked with spaces; byte offsets and line numbers preserved' if status == 'MASKED' else
                   'No comments changed')
        payload.append(f"""<section class="file" id="file-{rank}"><header><h2>#{rank} {html.escape(r['file'])}</h2>
          <p><span class="pill">{html.escape(status)}</span> {html.escape(caption)}</p>
          <p>Original size: {len(before):,} bytes &middot; Lines: {len(left):,} &middot; Line comments: {r['comment_line']} &middot; Block comments: {r['comment_block']}</p>
          </header><div class="duo"><div class="pane"><h3>BEFORE — Git HEAD</h3><div class="codebox">{code(left)}</div></div>
          <div class="pane"><h3>AFTER — J-03b preprocessed</h3><div class="codebox">{code(right)}</div></div></div></section>""")
        audit_records.append({'rank': rank, 'file': r['file'], 'bytes':len(before), 'lines':len(left),
                              'status':status, 'raw_sha256': r['raw_sha256'], 'preproc_sha256':r['preproc_sha256']})
    # Each file gets its own full-height comparison document. Generate all paths
    # before writing anything, and preserve preexisting reports without overwrite.
    filenames = [f'{rank:02d}-before-after.html' for rank in range(1, len(selected)+1)]
    expected = [outdir / name for name in filenames]
    summary_path = outdir / 'inspection_manifest.json'
    collisions = [str(p) for p in expected + [summary_path] if p.exists()]
    if collisions:
        raise FileExistsError(f'HTML_INSPECTION_OUTPUT_EXISTS subject={subject} paths={collisions[:3]}')
    header = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>J-03b source comparison</title>
<style>
*{box-sizing:border-box}html,body{height:100%;margin:0}body{font-family:system-ui,sans-serif;background:#f5f7fa;color:#152238}
main{display:flex;flex-direction:column;height:100vh;padding:12px 18px;gap:8px;min-height:0}
h1{font-size:1.12rem;margin:0}h2{font-size:.95rem;overflow-wrap:anywhere;margin:0 0 5px}
.meta{font-size:.77rem;margin:0;color:#536174}nav{display:flex;gap:6px;flex-wrap:wrap}
nav a{padding:5px 8px;background:#e5ebf4;border-radius:5px;color:#213d65;font-size:.78rem;text-decoration:none}
nav a[aria-current=page]{background:#243e68;color:#fff}.file{background:white;border:1px solid #d7e0eb;border-radius:8px;padding:10px;display:flex;flex-direction:column;flex:1;min-height:0}
.file header{flex-shrink:0}.file header p{font-size:.76rem;margin:3px 0;color:#536174}.pill{background:#edf0ff;padding:3px 6px;border-radius:4px;font-weight:bold}
.duo{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:8px;flex:1;min-height:0;margin-top:8px}
.pane{display:flex;flex-direction:column;min-width:0;min-height:0}.pane h3{font-size:.8rem;background:#e8edf5;padding:8px;margin:0}
.codebox{flex:1;min-height:0;overflow:auto;background:#101c2a;color:#dce8f2;font:13px/1.5 ui-monospace,Consolas,monospace;white-space:pre;tab-size:4}
/* Offline syntax coloring: no CDN, no dependencies, no runtime source modification. */
.tok-keyword{color:#c792ea;font-weight:600}.tok-type{color:#82aaff}.tok-string{color:#c3e88d}.tok-comment{color:#8195a8;font-style:italic}.tok-annotation{color:#ffcb6b}.tok-number{color:#f78c6c}
.ln{display:flex;min-width:max-content}.num{position:sticky;left:0;min-width:55px;text-align:right;padding:0 10px 0 4px;margin-right:8px;color:#8b9bad;background:#1b2c40;user-select:none;border-right:1px solid #40516a}.src{padding-right:16px}
@media(max-width:850px){main{height:100dvh;padding:7px}.duo{grid-template-columns:1fr;grid-template-rows:minmax(0,1fr) minmax(0,1fr)}}
@media print{main{height:auto}.codebox{overflow:visible}.duo{display:block}}
</style></head><body><main>"""
    timestamp = now()
    for rank, (fragment, record, target) in enumerate(zip(payload, audit_records, expected), 1):
        navigation=''.join(f'<a href="{html.escape(fn,quote=True)}"' + (' aria-current="page"' if j == rank else '') + f'>#{j}</a>' for j,fn in enumerate(filenames,1))
        page = header + f'<h1>J-03b {html.escape(args.script_version)} — {html.escape(subject)} | file {rank}/{len(selected)}</h1>' + \
          f'<p class="meta">Preproc policy {html.escape(args.preproc_data_version)} · Git HEAD: <code>{html.escape(m["git_head"])}</code> · Generated: {html.escape(timestamp)} America/Chicago</p>' + \
          f'<nav>{navigation}</nav>' + fragment + \
          """<script>const boxes=document.querySelectorAll('.duo .codebox');let busy=false;boxes.forEach((box,i)=>box.addEventListener('scroll',()=>{if(busy)return;busy=true;const other=boxes[1-i];other.scrollTop=box.scrollTop;other.scrollLeft=box.scrollLeft;requestAnimationFrame(()=>busy=false)}))</script></main></body></html>"""
        with target.open('x', encoding='utf-8') as f:
            f.write(page)
        record['html_filename'] = target.name
        record['html_sha256'] = sha_file(target)
    info={'experiment':args.experiment,'script_version':args.script_version,'preproc_data_version':args.preproc_data_version,
          'subject':subject,'timestamp_chicago':timestamp,'git_head':m['git_head'], 'html_top_n':args.html_top_n,
          'selection_rule':'included Java files sorted by original byte size descending, relative path ascending', 'syntax_highlighting':'offline_java_lexical_v1',
          'output_directory':str(outdir.resolve()),'files':audit_records}
    write_json_new(summary_path,info)
    print(f'HTML_INSPECTION_PASS subject={subject} files={len(selected)} unicode_passthrough={sum(x["status"]=="PASSTHROUGH_UNICODE_ESCAPE" for x in audit_records)} directory={outdir} manifest={summary_path}',flush=True)

def main():
    p = argparse.ArgumentParser()
    for key in ('experiment', 'version', 'mode', 'repo-root', 'preproc-root', 'report-dir', 'expected-subjects',
                'subject', 'require-clean', 'embedded-test-policy', 'progress-every',
                'java-unicode-escape-policy'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--skip-dir', action='append', required=True)
    p.add_argument('--skip-glob', action='append', required=True)
    p.add_argument('--script-version', required=True)
    p.add_argument('--preproc-data-version', required=True)
    p.add_argument('--html-top-n', type=int, required=True)
    p.add_argument('--html-subdirectory', required=True)
    p.add_argument('--baseline-csv', type=Path)
    p.add_argument('--nicad-xml', type=Path)
    args = p.parse_args()
    args.repo_root = Path(args.repo_root).resolve(strict=True)
    args.preproc_root = Path(args.preproc_root).absolute()
    args.report_dir = Path(args.report_dir).resolve(strict=True)
    args.skip_dir = [x.casefold() for x in args.skip_dir]
    args.expected_subjects = int(args.expected_subjects)
    args.progress_every = int(args.progress_every)
    args.require_clean = {'true': True, 'false': False}[args.require_clean.lower()]
    if args.html_top_n <= 0 or args.html_top_n > 100 or not re.fullmatch(r'[A-Za-z0-9_-]+', args.html_subdirectory):
        p.error('Invalid HTML options')
    if args.progress_every <= 0 or args.expected_subjects <= 0:
        p.error('Counts must be positive')
    if args.java_unicode_escape_policy not in ('preserve_original_audited', 'fail_closed'):
        p.error('Unsupported --java-unicode-escape-policy')
    if args.mode == 'inspect':
        inventory(args)
    elif args.mode == 'prepare':
        prepare(args, args.subject)
    elif args.mode == 'verify-existing':
        verify_existing(args, args.subject)
    elif args.mode == 'html':
        html_inspection(args, args.subject)
    elif args.mode == 'compare':
        if not args.baseline_csv or not args.nicad_xml:
            p.error('compare requires --baseline-csv and --nicad-xml')
        identical = compare(args, args.subject, args.nicad_xml, args.baseline_csv)
        if not identical:
            sys.exit(3)
    else:
        p.error('Invalid mode')


if __name__ == '__main__':
    main()
