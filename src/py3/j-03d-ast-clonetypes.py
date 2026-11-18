#!/usr/bin/env python3
"""J-03d v5 H-07/H-08 AST-based clone type *candidate* census.

Imported by j-03d-positive-clones.py. No configuration defaults or CLI here.
The standalone Shell supplies every parameter. H-07/H-08 reference sources
are imported read-only at runtime and their SHA256 are recorded.
"""

import csv
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sqlite3
import tempfile
import time
import zlib
from collections import Counter
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path


TYPE_ORDER = (
    'TYPE_1_SIGNATURE',
    'TYPE_2_CONSISTENT_SIGNATURE',
    'TYPE_2_BLIND_SIGNATURE',
    'TYPE_3_STRUCTURAL_CANDIDATE',
    'SEMANTIC_UNDETERMINED',
    'STRUCTURAL_UNSCORED',
    'INELIGIBLE',
)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def new_json(path, record):
    with path.open('x', encoding='utf-8') as f:
        json.dump(record, f, indent=2, sort_keys=True, ensure_ascii=True)
        f.write('\n')


def load_module(path, name):
    require(path.is_file() and not path.is_symlink(), f'AST_REFERENCE_MISSING_OR_UNSAFE {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f'AST_REFERENCE_IMPORT_INVALID {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependency_preflight(args):
    require(args.time_zone == 'America/Chicago', 'AST_TIMEZONE_MISMATCH')
    for package, expected in (('tree-sitter', args.ast_parser_version),
                              ('tree-sitter-java', args.ast_grammar_version)):
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f'AST_DEPENDENCY_MISSING package={package}; install {package}=={expected}') from exc
        require(actual == expected,
                f'AST_DEPENDENCY_VERSION_MISMATCH package={package} expected={expected} found={actual}')
    from tree_sitter import Language, Parser
    import tree_sitter_java
    parser = Parser(Language(tree_sitter_java.language()))
    h07 = load_module(args.h07_python, 'j03d_h07_reference')
    h08 = load_module(args.h08_python, 'j03d_h08_reference')
    for f in ('parse_strategy', 'direct_targets', 'direct_class_targets', 'pre_order_named'):
        require(callable(getattr(h07, f, None)), f'H07_REFERENCE_INTERFACE_MISSING {f}')
    for f in ('grams', 'jaccard', 'harmonic', 'choose_type'):
        require(callable(getattr(h08, f, None)), f'H08_REFERENCE_INTERFACE_MISSING {f}')
    evidence = {
        'h07_path': str(args.h07_python),
        'h07_sha256': sha_file(args.h07_python),
        'h08_path': str(args.h08_python),
        'h08_sha256': sha_file(args.h08_python),
        'tree_sitter_version': args.ast_parser_version,
        'tree_sitter_java_version': args.ast_grammar_version,
        'gram_size': args.ast_gram_size,
        'primary_threshold': str(args.ast_primary_threshold),
        'sensitivity_thresholds': [str(x) for x in args.ast_thresholds],
        'max_ast_comparison_cells': args.ast_max_cells,
        'fragment_cache_size': args.ast_fragment_cache_size,
        'score_cache_size': args.ast_score_cache_size,
        'nicad_band_edges': [str(x) for x in args.ast_nicad_band_edges],
        'method': 'H-07 exact syntax/consistent/blind signatures; H-08 AST-node-type k-gram Jaccard and SequenceMatcher harmonic mean',
    }
    print('AST_REFERENCE_PASS ' + json.dumps(evidence, sort_keys=True), flush=True)
    return parser, h07, h08, evidence


def retrieve_ast_named_sequence(parser, h07, raw, mode):
    if mode == 'direct_declaration':
        tree = parser.parse(raw)
        targets = h07.direct_targets(tree.root_node)
        require(len(targets) == 1, f'AST_TARGET_RECONSTRUCTION_FAIL mode={mode}')
        target = targets[0]
    elif mode == 'method_in_class_wrapper':
        wrapped = b'class H07SyntheticWrapper {\n' + raw + b'\n}\n'
        tree = parser.parse(wrapped)
        targets = h07.direct_class_targets(tree.root_node)
        require(len(targets) == 1, f'AST_TARGET_RECONSTRUCTION_FAIL mode={mode}')
        target = targets[0]
    else:
        raise RuntimeError(f'AST_UNEXPECTED_PARSE_MODE {mode}')
    return tuple(h07.pre_order_named(target))


def extract_feature(parser, h07, fragment, args):
    ident = fragment.get('fragment_id')
    code = fragment.get('code')
    require(isinstance(ident, str) and ident and isinstance(code, str),
            f'AST_FRAGMENT_SCHEMA_INVALID fragment_id={ident!r}')
    try:
        raw = code.encode(args.code_encoding, errors=args.code_errors)
    except (UnicodeError, LookupError) as e:
        raise RuntimeError(f'AST_FRAGMENT_ENCODING_ERROR id={ident}: {e}') from e
    require(sha_bytes(raw) == fragment.get('source_sha256'),
            f'AST_FRAGMENT_SOURCE_HASH_MISMATCH id={ident}')
    try:
        normalized = raw.decode('utf-8', errors='strict')
    except UnicodeDecodeError:
        # H-07 requires UTF-8 input. Never rewrite invalid bytes to obtain a passing signature.
        return {
            'fragment_id': ident,
            'source_sha256': fragment['source_sha256'],
            'parse_status': 'FAIL', 'parse_mode': 'UNRESOLVED',
            'record_kind': 'INVALID_UTF8_SOURCE', 'function_level_eligible': False,
            'parse_failure_reasons': [{'reason': 'INVALID_UTF8_SOURCE'}],
        }, None
    if not normalized.strip():
        meta = {'parse_status': 'FAIL', 'parse_mode': 'UNRESOLVED',
                'record_kind': 'EMPTY_SOURCE', 'function_level_eligible': False,
                'parse_failure_reasons': [{'reason': 'EMPTY_SOURCE'}]}
    else:
        meta = h07.parse_strategy(parser, normalized)
    feature = {'fragment_id': ident, 'source_sha256': fragment['source_sha256'], **meta}
    seq = None
    if feature.get('function_level_eligible') and feature.get('parse_status') == 'PASS':
        seq = retrieve_ast_named_sequence(parser, h07, normalized.encode('utf-8'), feature['parse_mode'])
        require(seq, f'AST_EMPTY_NAMED_SEQUENCE id={ident}')
        # H-07 hashes the same pre-order named AST node types.
        expected = sha_bytes(json.dumps(list(seq), separators=(',', ':')).encode('utf-8'))
        require(expected == feature['ast_named_preorder_sha256'],
                f'AST_H07_PREORDER_HASH_MISMATCH id={ident}')
    return feature, seq


def build_features(args, subject, parser, h07, run_subject, common):
    """Stream the immutable JSONL once and materialize an on-disk SQLite index."""
    positive_dir = args.output_root / subject
    path = positive_dir / 'fragments.jsonl'
    database = run_subject / 'feature_index.sqlite'
    db = sqlite3.connect(database)
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE features (fragment_id TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL, '
               'parse_status TEXT NOT NULL, record_kind TEXT NOT NULL, eligible INTEGER NOT NULL, '
               'type1 TEXT, type2c TEXT, type2b TEXT, ast_sha TEXT, ast_nodes BLOB)')
    counts = Counter()
    failures = Counter()
    records_path = run_subject / 'fragment_ast_features.jsonl'
    with path.open('r', encoding='utf-8') as inp, records_path.open('x', encoding='utf-8') as out:
        for lineno, line in enumerate(inp, 1):
            require(bool(line.strip()), f'AST_BLANK_FRAGMENT_RECORD subject={subject} line={lineno}')
            frag = json.loads(line)
            feature, seq = extract_feature(parser, h07, frag, args)
            if not feature.get('function_level_eligible'):
                seq = None
            row = (feature['fragment_id'], feature['source_sha256'],
                   feature['parse_status'], feature['record_kind'],
                   int(bool(feature.get('function_level_eligible'))),
                   feature.get('type1_syntax_sha256'), feature.get('type2_consistent_sha256'),
                   feature.get('type2_blind_sha256'), feature.get('ast_named_preorder_sha256'),
                   zlib.compress(json.dumps(seq, separators=(',', ':')).encode('utf-8'),
                                 level=args.ast_compression_level) if seq is not None else None)
            try:
                db.execute('INSERT INTO features VALUES (?,?,?,?,?,?,?,?,?,?)', row)
            except sqlite3.IntegrityError as e:
                raise RuntimeError(f'AST_DUPLICATE_FRAGMENT subject={subject} id={feature["fragment_id"]}') from e
            out.write(json.dumps(feature, sort_keys=True, ensure_ascii=True) + '\n')
            counts['fragments'] += 1
            counts['parse_' + feature['parse_status']] += 1
            counts['kind_' + feature['record_kind']] += 1
            counts['eligible'] += int(bool(feature.get('function_level_eligible')))
            if feature['parse_status'] == 'FAIL':
                for reason in feature.get('parse_failure_reasons', []):
                    failures[str(reason.get('reason') or reason.get('mode') or 'UNKNOWN')] += 1
            if counts['fragments'] % args.ast_progress_fragments == 0:
                db.commit()
                print(f'AST_FRAGMENT_PROGRESS subject={subject} fragments={counts["fragments"]} '
                      f'eligible={counts["eligible"]} failed={counts["parse_FAIL"]}', flush=True)
    db.commit()
    db.execute('CREATE INDEX features_ast_sha_idx ON features(ast_sha)')
    require(counts['fragments'] == common['fragments'],
            f'AST_FRAGMENT_COUNT_MISMATCH subject={subject} actual={counts["fragments"]} expected={common["fragments"]}')
    print(f'AST_FRAGMENT_PASS subject={subject} fragments={counts["fragments"]} '
          f'eligible={counts["eligible"]} failed={counts["parse_FAIL"]}', flush=True)
    return db, counts, failures, sha_file(records_path)


def similarity_band(raw, args):
    try:
        value = Decimal(raw)
    except (InvalidOperation, ValueError):
        raise RuntimeError(f'AST_INVALID_NICAD_SIMILARITY value={raw!r}')
    require(value.is_finite() and args.clonetype_min_similarity <= value <= args.clonetype_max_similarity,
            f'AST_OUT_OF_RANGE_NICAD_SIMILARITY value={raw!r}')
    if value == args.ast_nicad_band_edges[0]:
        return args.ast_nicad_band_labels[0]
    for i in range(1, len(args.ast_nicad_band_edges)):
        if value >= args.ast_nicad_band_edges[i]:
            return args.ast_nicad_band_labels[i]
    raise RuntimeError(f'AST_NICAD_BAND_UNCLASSIFIED value={value}')


def classify_subject(args, subject, db, h08, common, run_subject):
    """H-08 precedence and metric definitions on full subject pair stream."""
    @functools.lru_cache(maxsize=args.ast_fragment_cache_size)
    def info(identifier):
        return db.execute('SELECT * FROM features WHERE fragment_id=?', (identifier,)).fetchone()

    @functools.lru_cache(maxsize=args.ast_fragment_cache_size)
    def named_grams(identifier):
        record = info(identifier)
        require(record is not None, f'AST_UNKNOWN_FRAGMENT id={identifier} subject={subject}')
        require(record[9] is not None, f'AST_NODES_REQUIRED id={identifier}')
        nodes = tuple(json.loads(zlib.decompress(record[9]).decode('utf-8')))
        return nodes, h08.grams(nodes, args.ast_gram_size)

    @functools.lru_cache(maxsize=args.ast_score_cache_size)
    def combined_score(hash_a, hash_b):
        # H-08 score depends only on named AST preorder, not identifier values.
        # Cache by AST digest pair (not fragment IDs) to handle million-pair
        # clone clusters with repeated structural shapes efficiently.
        row_a = db.execute('SELECT fragment_id FROM features WHERE ast_sha=? LIMIT 1',
                           (hash_a,)).fetchone()
        row_b = db.execute('SELECT fragment_id FROM features WHERE ast_sha=? LIMIT 1',
                           (hash_b,)).fetchone()
        require(row_a is not None and row_b is not None,
                f'AST_UNRESOLVED_SCORE_SIGNATURE subject={subject}')
        seq_a, grams_a = named_grams(row_a[0])
        seq_b, grams_b = named_grams(row_b[0])
        if len(seq_a) * len(seq_b) > args.ast_max_cells:
            return None, None, None
        jac = h08.jaccard(grams_a, grams_b)
        ratio = h08.SequenceMatcher(None, seq_a, seq_b, autojunk=False).ratio()
        return jac, ratio, h08.harmonic(jac, ratio)

    totals = Counter()
    threshold_counts = {str(t): Counter() for t in args.ast_thresholds}
    bands = Counter()
    examples = {label: [] for label in TYPE_ORDER}
    start = time.monotonic()
    pairs_path = args.output_root / subject / 'positive_pairs.csv'
    with pairs_path.open('r', encoding='utf-8', newline='') as inp:
        reader = csv.DictReader(inp, strict=True)
        required = {'pair_id', 'fragment_a_id', 'fragment_b_id', 'source_sha256_a',
                    'source_sha256_b', 'label', 'nicad_similarity_raw',
                    'repository_a', 'repository_b', 'detector', 'config'}
        require(required <= set(reader.fieldnames or []), f'AST_PAIR_SCHEMA_MISMATCH subject={subject}')
        for row in reader:
            pair = row['pair_id']
            require(row['label'] == str(args.positive_label)
                    and row['repository_a'] == row['repository_b'] == subject
                    and row['detector'] == args.expected_detector
                    and row['config'] == args.expected_nicad_config,
                    f'AST_PAIR_METADATA_MISMATCH subject={subject} pair={pair}')
            aid, bid = row['fragment_a_id'], row['fragment_b_id']
            require(aid != bid, f'AST_SELF_PAIR subject={subject} pair={pair}')
            left, right = info(aid), info(bid)
            require(left is not None and right is not None,
                    f'AST_MISSING_PAIR_FRAGMENT subject={subject} pair={pair}')
            require(left[1] == row['source_sha256_a'] and right[1] == row['source_sha256_b'],
                    f'AST_PAIR_SOURCE_HASH_CONFLICT subject={subject} pair={pair}')
            band = similarity_band(row['nicad_similarity_raw'], args)
            bands[band] += 1
            eligible = bool(left[4]) and bool(right[4])
            score = None
            if not eligible:
                typ = 'INELIGIBLE'
            elif left[5] == right[5]:
                typ = 'TYPE_1_SIGNATURE'
            elif left[6] == right[6]:
                typ = 'TYPE_2_CONSISTENT_SIGNATURE'
            elif left[7] == right[7]:
                typ = 'TYPE_2_BLIND_SIGNATURE'
            else:
                ha, hb = left[8], right[8]
                require(ha and hb, f'AST_PREORDER_HASH_MISSING subject={subject} pair={pair}')
                if ha == hb:
                    score = 1.0
                else:
                    # The same AST digest pair has the same H-08 structural score.
                    hh1, hh2 = sorted((ha, hb))
                    _, _, score = combined_score(hh1, hh2)
                typ = ('STRUCTURAL_UNSCORED' if score is None else
                       'TYPE_3_STRUCTURAL_CANDIDATE' if score >= float(args.ast_primary_threshold)
                       else 'SEMANTIC_UNDETERMINED')
            totals[typ] += 1
            totals['pairs'] += 1
            if len(examples[typ]) < args.ast_sample_per_category:
                examples[typ].append({'pair_id': pair, 'fragment_a_id': aid,
                                      'fragment_b_id': bid, 'nicad_similarity': row['nicad_similarity_raw'],
                                      'ast_combined': round(score, 8) if score is not None else None})
            for threshold in args.ast_thresholds:
                if typ in ('TYPE_1_SIGNATURE', 'TYPE_2_CONSISTENT_SIGNATURE',
                           'TYPE_2_BLIND_SIGNATURE', 'STRUCTURAL_UNSCORED', 'INELIGIBLE'):
                    cat = typ
                else:
                    cat = 'TYPE_3_STRUCTURAL_CANDIDATE' if score is not None and score >= float(threshold) else 'SEMANTIC_UNDETERMINED'
                threshold_counts[str(threshold)][cat] += 1
            if totals['pairs'] % args.ast_progress_pairs == 0:
                print(f'AST_PAIR_PROGRESS subject={subject} pairs={totals["pairs"]} '
                      f't1={totals["TYPE_1_SIGNATURE"]} '
                      f't2c={totals["TYPE_2_CONSISTENT_SIGNATURE"]} '
                      f't2b={totals["TYPE_2_BLIND_SIGNATURE"]} '
                      f't2_total={totals["TYPE_2_CONSISTENT_SIGNATURE"] + totals["TYPE_2_BLIND_SIGNATURE"]} '
                      f't3={totals["TYPE_3_STRUCTURAL_CANDIDATE"]} '
                      f'unscored={totals["STRUCTURAL_UNSCORED"]} '
                      f'elapsed_sec={int(time.monotonic()-start)}', flush=True)
    require(totals['pairs'] == common['pairs'],
            f'AST_PAIR_COUNT_MISMATCH subject={subject} actual={totals["pairs"]} expected={common["pairs"]}')
    require(sum(totals[k] for k in TYPE_ORDER) == totals['pairs'],
            f'AST_TYPE_COUNTS_DO_NOT_SUM subject={subject}')
    for t, counted in threshold_counts.items():
        require(sum(counted.values()) == totals['pairs'],
                f'AST_THRESHOLD_COUNT_MISMATCH subject={subject} threshold={t}')
    result = {
        'subject': subject, 'pairs': totals['pairs'],
        'counts': {label: totals[label] for label in TYPE_ORDER},
        'type2_total': totals['TYPE_2_CONSISTENT_SIGNATURE'] + totals['TYPE_2_BLIND_SIGNATURE'],
        'nicad_similarity_bands': {band: bands[band] for band in args.ast_nicad_band_labels},
        'threshold_sensitivity': {t: dict(c) for t, c in threshold_counts.items()},
        'examples_by_class': examples,
        'duration_seconds': round(time.monotonic() - start, 3),
        'score_cache_info': str(combined_score.cache_info()),
    }
    print(f'AST_SUBJECT_PASS subject={subject} pairs={totals["pairs"]} '
          f'TYPE_2_TOTAL={result["type2_total"]} '
          + ' '.join(f'{k}={totals[k]}' for k in TYPE_ORDER) +
          f' duration_sec={result["duration_seconds"]}', flush=True)
    return result


def fmt(n):
    return f'{n:,}'


def fixed_width_table(headers, records):
    """Render compact, genuinely fixed-width cells in Markdown code fences.

    GitHub Flavored Markdown tables do not guarantee rendered column widths.
    Each column here has one consistent width across all subjects, determined
    once from the values. No subject name or large count is truncated.
    """
    require(bool(headers) and bool(records), 'AST_FIXED_TABLE_EMPTY')
    require(all(len(row) == len(headers) for row in records), 'AST_FIXED_TABLE_SHAPE')
    widths = [max(len(str(headers[col])), *(len(str(row[col])) for row in records))
              for col in range(len(headers))]

    def line(values):
        return '  '.join((str(value).ljust(widths[col]) if col == 0 else
                          str(value).rjust(widths[col]))
                         for col, value in enumerate(values))

    return ['```text', line(headers),
            '  '.join('-' * width for width in widths),
            *(line(row) for row in records), '```']


def markdown_report(args, results, ast_evidence, timestamp, full):
    c = Counter()
    b = Counter()
    t = {str(x): Counter() for x in args.ast_thresholds}
    for r in results:
        c.update(r['counts'])
        b.update(r['nicad_similarity_bands'])
        for k, v in r['threshold_sensitivity'].items():
            t[k].update(v)
    total = sum(r['pairs'] for r in results)
    md = [
        '# J-03d v5 - AST-based Clone Type Candidate Census',
        '',
        f'Generated (America/Chicago): {timestamp}',
        f'Experiment: `{args.experiment}`; analyzer version: `{args.version}`',
        f'Run mode: `{"all-subject" if full else "smoke-subject"}`',
        f'J-03c AFTER (authoritative input): `{args.expected_input_relative}/`',
        f'J-03d Positive Dataset: `{args.expected_output_relative}/`',
        f'Subjects: **{len(results)}**; Positive pairs: **{fmt(total)}**',
        '',
        '## Classification (H-07 / H-08; candidates, NOT certified gold types)',
        '',
        '- Type 1 Signature: Tree-sitter Syntax Token SHA256 equal, excluding comments/whitespace.',
        '- Type 2 Consistent: after Type 1, H-07 consistent identifier/literal normalization SHA256 equal.',
        '- Type 2 Blind: after previous checks, H-07 blind normalization SHA256 equal.',
        '- Type 2 전체 (total): Type 2 Consistent + Type 2 Blind (derived subtotal, not an extra category).',
        f'- Type 3 Candidate: otherwise H-08 AST Node-type {args.ast_gram_size}-gram Jaccard and '
        f'SequenceMatcher harmonic mean >= {args.ast_primary_threshold}.',
        '- Semantic-undetermined: eligible, scored, below threshold; **not Type 4**.',
        '- Structural-unscored: AST comparison cost above configured cell budget; **not Type 4**.',
        '- Ineligible: invalid/unparseable fragment or non-single-function; **not Type 4**.',
        '- Type 4: N/A (semantic equivalence needs independent evidence).',
        '',
        '## Overall counts',
        '',
        '| Category | Pairs | Share |',
        '|---|---:|---:|',
    ]
    type2_total = c['TYPE_2_CONSISTENT_SIGNATURE'] + c['TYPE_2_BLIND_SIGNATURE']
    for label in TYPE_ORDER:
        n = c[label]
        md.append(f'| `{label}` | {fmt(n)} | {n / total * 100:.2f}% |' if total else
                  f'| `{label}` | 0 | N/A |')
        if label == 'TYPE_2_BLIND_SIGNATURE':
            share = f'{type2_total / total * 100:.2f}%' if total else 'N/A'
            md.append(f'| **Type 2 전체 (subtotal)** | **{fmt(type2_total)}** | **{share}** |')
    md.extend([f'| **TOTAL (7 non-overlapping categories)** | **{fmt(total)}** | **100.00%** |',
               '| Type 4 Semantic Clone | N/A | N/A |', '',
               '**Note:** Type 2 전체 is a subtotal of Consistent and Blind; do not add it to TOTAL.',
               '',
               '## Per-subject candidate counts', '',
               'Fixed-width columns, split into two compact tables for terminal and GitHub readability.',
               'T2-C = Type 2 Consistent; T2-B = Type 2 Blind; **T2-All = Type 2 전체** = T2-C + T2-B.',
               'T3 = Type 3 Candidate. T2-All is a subtotal, not an eighth class.',
               '',
               '**A. Structural clone candidates**', ''])
    main_headers = ('Subject', 'Pairs', 'T1', 'T2-C', 'T2-B', 'T2-All', 'T3')
    main_rows = []
    status_headers = ('Subject', 'Undet', 'Unsc', 'Inelig')
    status_rows = []
    for r in results:
        cnt = r['counts']
        require(cnt['TYPE_2_CONSISTENT_SIGNATURE'] + cnt['TYPE_2_BLIND_SIGNATURE'] == r['type2_total'],
                f'AST_TYPE2_SUBTOTAL_MISMATCH subject={r["subject"]}')
        main_rows.append((r['subject'], fmt(r['pairs']), fmt(cnt['TYPE_1_SIGNATURE']),
                          fmt(cnt['TYPE_2_CONSISTENT_SIGNATURE']), fmt(cnt['TYPE_2_BLIND_SIGNATURE']),
                          fmt(r['type2_total']), fmt(cnt['TYPE_3_STRUCTURAL_CANDIDATE'])))
        status_rows.append((r['subject'], fmt(cnt['SEMANTIC_UNDETERMINED']),
                            fmt(cnt['STRUCTURAL_UNSCORED']), fmt(cnt['INELIGIBLE'])))
    main_rows.append(('TOTAL', fmt(total), fmt(c['TYPE_1_SIGNATURE']),
                      fmt(c['TYPE_2_CONSISTENT_SIGNATURE']), fmt(c['TYPE_2_BLIND_SIGNATURE']),
                      fmt(type2_total), fmt(c['TYPE_3_STRUCTURAL_CANDIDATE'])))
    status_rows.append(('TOTAL', fmt(c['SEMANTIC_UNDETERMINED']),
                        fmt(c['STRUCTURAL_UNSCORED']), fmt(c['INELIGIBLE'])))
    md.extend(fixed_width_table(main_headers, main_rows))
    md.extend(['', '**B. Undetermined / unscored / ineligible**', '',
               'Undet = Semantic-undetermined; Unsc = Structural-unscored; Inelig = Ineligible.', ''])
    md.extend(fixed_width_table(status_headers, status_rows))
    md.extend(['', '## NiCad similarity distribution (independent descriptive metric)', '',
               '| NiCad similarity band | Pairs |', '|---|---:|'])
    for band in args.ast_nicad_band_labels:
        md.append(f'| {band}% | {fmt(b[band])} |')
    md.extend(['', '## Threshold sensitivity (structural candidates only)', '',
               '| AST combined threshold | Type 3 candidates | All Type 1/2 and Type 3 candidates |',
               '|---:|---:|---:|'])
    for threshold in args.ast_thresholds:
        v = t[str(threshold)]
        t3 = v['TYPE_3_STRUCTURAL_CANDIDATE']
        totalcand = t3 + sum(v[k] for k in TYPE_ORDER[:3])
        md.append(f'| {threshold} | {fmt(t3)} | {fmt(totalcand)} |')
    md.extend(['', '## Reproducibility / limitations', '',
               f'- H-07 source SHA256: `{ast_evidence["h07_sha256"]}`.',
               f'- H-08 source SHA256: `{ast_evidence["h08_sha256"]}`.',
               f'- tree-sitter `{args.ast_parser_version}`; tree-sitter-java `{args.ast_grammar_version}`.',
               f'- Comparison max cells: `{args.ast_max_cells}`; fragment cache: `{args.ast_fragment_cache_size}`; '
               f'score cache: `{args.ast_score_cache_size}`.',
               '- Output and source archives are verified by original SHA256 manifests; no positive pair was regenerated.',
               '- H-08 signatures depend on parsing/normalization rules; these are structural candidates only.',
               '- NiCad similarity cannot substitute for AST similarity; AST 0.82 is not a Java gold-label-certified threshold.',
               '- Type 4 remains not assessable here. Do not label unscored or undetermined as semantic clones.',
               '- The H-07/H-08 reference implementations were developed on a different Java Evaluation Dataset; '
               'this report covers J-03d NiCad-derived positives.',
               ''])
    return '\n'.join(md), c, b, t


def publish_no_overwrite(path, content):
    require(not path.is_symlink(), f'AST_REPORT_SYMLINK_REJECTED path={path}')
    if path.exists():
        require(path.is_file(), f'AST_REPORT_UNSAFE path={path}')
        existing = path.read_text(encoding='utf-8')
        # Existing timestamp is retained to allow revalidation without modification.
        prefix = 'Generated (America/Chicago): '
        stamps = [line[len(prefix):] for line in existing.splitlines() if line.startswith(prefix)]
        require(len(stamps) == 1, f'AST_REPORT_EXISTING_TIMESTAMP_INVALID path={path}')
        current = [line[len(prefix):] for line in content.splitlines() if line.startswith(prefix)]
        require(len(current) == 1, f'AST_REPORT_NEW_TIMESTAMP_INVALID path={path}')
        normalized = content.replace(prefix + current[0], prefix + stamps[0], 1)
        require(existing == normalized,
                f'AST_REPORT_EXISTS_DIFFERENT_CONTENT path={path}; old report preserved')
        print(f'AST_REPORT_REUSED_VERIFIED path={path} sha256={sha_file(path)}', flush=True)
        return
    fd, name = tempfile.mkstemp(prefix='.j03d-ast-report-', dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
        os.link(tmp, path)  # Atomic and fails if destination already exists.
    finally:
        tmp.unlink(missing_ok=True)
    print(f'AST_REPORT_CREATED path={path} sha256={sha_file(path)}', flush=True)


def analyze(args, names, original):
    """original is the loaded J-03d module for immutable dataset verification."""
    start = time.monotonic()
    parser, h07, h08, evidence = dependency_preflight(args)
    require(args.run_dir.is_dir() and not args.run_dir.is_symlink(), 'AST_RUN_DIR_INVALID')
    run_subroot = args.run_dir / 'clonetype_ast'
    run_subroot.mkdir(mode=0o755)  # unique Shell-created run dir
    results = []
    is_full = args.mode == 'clonetypes'
    total_pairs = total_fragments = 0
    for idx, subject in enumerate(names, 1):
        print(f'AST_SUBJECT_START subject={subject} index={idx}/{len(names)}', flush=True)
        m = original.verify_output(args, subject)
        path = run_subroot / subject
        path.mkdir(mode=0o755)
        db = None
        try:
            db, fc, failures, features_sha = build_features(args, subject, parser, h07, path, m)
            r = classify_subject(args, subject, db, h08, m, path)
            r.update({
                'fragments': fc['fragments'], 'eligible_fragments': fc['eligible'],
                'parse_failed_fragments': fc['parse_FAIL'],
                'parse_status_counts': dict(fc), 'parse_failure_reasons': dict(failures),
                'features_jsonl_sha256': features_sha,
                'positive_manifest_sha256': sha_file(args.output_root / subject / 'manifest.json'),
                'positive_pairs_sha256': m['files_sha256']['positive_pairs.csv'],
                'positive_fragments_sha256': m['files_sha256']['fragments.jsonl'],
                'j03c_pair_sha256': m['input_pair_csv_sha256'],
            })
            new_json(path / 'subject_summary.json', r)
            total_pairs += m['pairs']
            total_fragments += m['fragments']
            results.append(r)
        finally:
            if db is not None:
                db.close()
            dbfile = path / 'feature_index.sqlite'
            if dbfile.exists():
                dbfile.unlink()  # Ephemeral index of this run, not an existing versioned artifact.
        print(f'AST_SUBJECT_FILES subject={subject} feature_records={path / "fragment_ast_features.jsonl"} '
              f'summary={path / "subject_summary.json"}', flush=True)
    require(total_pairs == sum(r['pairs'] for r in results), 'AST_TOTAL_PAIR_COUNT_MISMATCH')
    if is_full:
        require(len(names) == args.expected_subjects and total_pairs == args.expected_total_pairs,
                f'AST_FULL_COUNT_MISMATCH subjects={len(names)} pairs={total_pairs}')
    stamp = original.timestamp(args)
    final_md, total_counts, total_bands, threshold_counts = markdown_report(args, results, evidence, stamp, is_full)
    # Per-run report is unconditionally new and never conflicts with previous versions.
    md_run = args.run_dir / args.ast_report_name
    with md_run.open('x', encoding='utf-8') as f:
        f.write(final_md)
    if is_full:
        # v3 summary-clonetype.md, if any, remains untouched forever.
        canonical = args.output_root / args.ast_report_name
        publish_no_overwrite(canonical, final_md)
        print(f'AST_CANONICAL_SUMMARY path={canonical}', flush=True)
    else:
        print('AST_SMOKE_NO_ROOT_SUMMARY_WRITTEN', flush=True)
    summary = {
        'experiment': args.experiment, 'version': args.version,
        'status': 'PASS_AST_CANDIDATE_CENSUS' if is_full else 'PASS_AST_SMOKE',
        'timestamp_chicago': stamp, 'full_census': is_full,
        'subjects': len(results), 'pairs': total_pairs, 'fragments': total_fragments,
        'primary_counts': dict(total_counts),
        'type2_total': total_counts['TYPE_2_CONSISTENT_SIGNATURE'] + total_counts['TYPE_2_BLIND_SIGNATURE'],
        'nicad_similarity_bands': dict(total_bands),
        'threshold_sensitivity': {key: dict(v) for key, v in threshold_counts.items()},
        'reference': evidence,
        'inputs': {'j03c_after': str(args.input_root),
                   'j03d_positives': str(args.output_root),
                   'preprocessed_code': str(args.source_root)},
        'output_markdown': str(md_run), 'output_markdown_sha256': sha_file(md_run),
        'dataset_summary': str(args.output_root / args.ast_report_name) if is_full else None,
        'dataset_summary_sha256': sha_file(args.output_root / args.ast_report_name) if is_full else None,
        'legacy_v3_summary_sha256': (sha_file(args.output_root / 'summary-clonetype.md')
                                     if (args.output_root / 'summary-clonetype.md').is_file() else None),
        'per_subject': [{'subject': x['subject'], 'pairs': x['pairs'],
                         'fragment_sha256': x['features_jsonl_sha256'],
                         'summary_sha256': sha_file(run_subroot / x['subject'] / 'subject_summary.json')}
                        for x in results],
        'duration_seconds': round(time.monotonic() - start, 3),
        'effective_configuration': {
            **original.effective_configuration(args),
            'ast_parser_version': args.ast_parser_version,
            'ast_grammar_version': args.ast_grammar_version,
            'ast_gram_size': args.ast_gram_size,
            'ast_primary_threshold': str(args.ast_primary_threshold),
            'ast_thresholds': [str(t) for t in args.ast_thresholds],
            'ast_max_cells': args.ast_max_cells,
            'ast_progress_fragments': args.ast_progress_fragments,
            'ast_progress_pairs': args.ast_progress_pairs,
            'ast_fragment_cache_size': args.ast_fragment_cache_size,
            'ast_score_cache_size': args.ast_score_cache_size,
            'ast_compression_level': args.ast_compression_level,
            'ast_sample_per_category': args.ast_sample_per_category,
            'ast_report_name': args.ast_report_name,
            'ast_nicad_band_edges': [str(e) for e in args.ast_nicad_band_edges],
            'h07_python': str(args.h07_python), 'h08_python': str(args.h08_python),
        },
    }
    new_json(args.run_dir / 'clonetype_ast_summary.json', summary)
    print('AST_CLONETYPE_TERMINAL_REPORT_START\n' + final_md, flush=True)
    print(f'AST_TYPE_2_TOTAL={summary["type2_total"]}', flush=True)
    print(f'AST_CLONETYPE_FINAL_PASS subjects={len(results)} pairs={total_pairs} '
          f'fragments={total_fragments} status={summary["status"]} '
          f'run_report={args.run_dir / "clonetype_ast_summary.json"}', flush=True)
