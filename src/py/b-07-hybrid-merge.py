#!/usr/bin/env python3
"""C-01: read-only, conservative comparison of NiCad and jscpd on staged Salt code."""
import argparse
import ast
import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def fn_key(node):
    return (node['file'], node['start'], node['end'])


def pair_key(a, b):
    return tuple(sorted((a, b)))


def norm_file(raw, known):
    s = str(raw or '').replace('\\', '/')
    while s.startswith('./'):
        s = s[2:]
    if s in known:
        return s
    # NiCad typically prefixes nicadclones/salt/salt/, but only suffixes
    # unambiguously matching a staged path are accepted.
    candidates = [k for k in known if s.endswith('/' + k)]
    return candidates[0] if len(candidates) == 1 else ''


def overlap(a, b, c, d):
    return max(0, min(b, d) - max(a, c) + 1)


def locate(file, start, end, by_file, threshold):
    if not file or start <= 0 or end < start:
        return None, 'bad_range'
    options = []
    for fn in by_file.get(file, ()):
        common = overlap(start, end, fn['start'], fn['end'])
        if not common:
            continue
        ratio_fn = common / (fn['end'] - fn['start'] + 1)
        ratio_fragment = common / (end - start + 1)
        if ratio_fn >= threshold and ratio_fragment >= 0.5:
            options.append((ratio_fn, ratio_fragment, fn))
    if not options:
        return None, 'no_function_match'
    options.sort(key=lambda v: (v[0], v[1]), reverse=True)
    if len(options) > 1 and options[0][:2] == options[1][:2]:
        return None, 'ambiguous_function_match'
    return options[0][2], 'matched'


def collect_functions(pyfiles, stage):
    manifest = []
    by_file = defaultdict(list)
    errors = []
    for path in pyfiles:
        rel = path.relative_to(stage).as_posix()
        b = path.read_bytes()
        manifest.append({'file': rel, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()})
        try:
            tree = ast.parse(b, filename=rel)
        except (SyntaxError, UnicodeError, ValueError) as exc:
            errors.append({'file': rel, 'error': str(exc)[:500]})
            continue
        class Visitor(ast.NodeVisitor):
            def __init__(self):
                self.scope = []
            def visit_ClassDef(self, n):
                self.scope.append(n.name)
                self.generic_visit(n)
                self.scope.pop()
            def visit_FunctionDef(self, n):
                self.add(n)
            def visit_AsyncFunctionDef(self, n):
                self.add(n)
            def add(self, n):
                beginning = min([n.lineno] + [d.lineno for d in n.decorator_list])
                obj = {'file': rel, 'start': beginning, 'end': n.end_lineno,
                       'name': '.'.join(self.scope + [n.name]),
                       'type': 'async' if isinstance(n, ast.AsyncFunctionDef) else 'sync'}
                by_file[rel].append(obj)
                self.scope.append(n.name)
                self.generic_visit(n)
                self.scope.pop()
        Visitor().visit(tree)
    return manifest, by_file, errors



def strict_locate(file, start, end, by_file, fraction=0.90, tolerance=2):
    """Accept only a single whole-function span, not fragments or nested ambiguities."""
    if not file or start < 1 or end < start:
        return None, 'invalid_range'
    candidates = []
    for fn in by_file.get(file, ()):
        fstart, fend = fn['start'], fn['end']
        common = overlap(start, end, fstart, fend)
        if not common:
            continue
        fn_fraction = common / (fend - fstart + 1)
        report_fraction = common / (end - start + 1)
        if (fn_fraction >= fraction and report_fraction >= fraction and
                abs(start - fstart) <= tolerance and abs(end - fend) <= tolerance):
            candidates.append(fn)
    if len(candidates) == 1:
        return candidates[0], 'matched'
    if len(candidates) > 1:
        return None, 'nested_or_ambiguous'
    return None, 'not_whole_function'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--ni-file', type=Path, required=True)
    parser.add_argument('--audit-file', type=Path, required=True)
    parser.add_argument('--js-file', type=Path, required=True)
    parser.add_argument('--ni-threshold', type=float, default=0.65)
    parser.add_argument('--strict-fraction', type=float, default=0.90)
    parser.add_argument('--boundary-tolerance', type=int, default=2)
    args = parser.parse_args()
    if not (0 < args.ni_threshold <= 1 and 0 < args.strict_fraction <= 1 and args.boundary_tolerance >= 0):
        parser.error('Invalid mapping threshold or tolerance')
    root = args.project_root.resolve()
    stage = args.stage.resolve()
    ni_file = args.ni_file.resolve()
    audit_file = args.audit_file.resolve()
    js_file = args.js_file.resolve()
    sources = [ni_file, audit_file, js_file]
    for path in sources:
        if not path.is_file():
            parser.error(f'Required input missing: {path}')
    if not stage.is_dir():
        parser.error(f'Staging directory missing: {stage}')
    out = args.output.resolve()
    if out.exists():
        parser.error(f'Output already exists: {out}')
    pyfiles = sorted(stage.rglob('*.py'))
    if not pyfiles:
        parser.error('No staged Python files')
    manifest, by_file, errors = collect_functions(pyfiles, stage)
    known = {m['file'] for m in manifest}
    index = {fn_key(fn): fn for funcs in by_file.values() for fn in funcs}
    audit = {}
    for item in load_csv(audit_file):
        key = norm_file(item.get('file'), known)
        if key:
            if key in audit:
                parser.error(f'Duplicate audit path: {key}')
            audit[key] = item
    if len(audit) != len(known):
        parser.error(f'Audit alignment failed: {len(audit)}/{len(known)}')
    failed = {file for file, item in audit.items() if item.get('nicad_category') == 'NICAD_PARSE_FAILURE'}
    out.mkdir(parents=True)
    write_csv(out / 'input_manifest.csv', manifest, ['file','bytes','sha256'])
    write_csv(out / 'python_ast_errors.csv', errors, ['file','error'])
    ni_raw, ni_pairs, ni_rejections = 0, set(), Counter()
    ni_rows=[]
    for i, clone in enumerate(ET.parse(ni_file).getroot().findall('clone'), 1):
        ni_raw += 1
        srcs = clone.findall('source')
        if len(srcs) != 2:
            ni_rejections['wrong_endpoint_count'] += 1
            continue
        mapped=[]
        for src in srcs:
            file=norm_file(src.get('file'), known)
            fn,reason=locate(file, as_int(src.get('startline')), as_int(src.get('endline')), by_file, args.ni_threshold)
            if not fn:
                ni_rejections[reason] += 1
            mapped.append(fn)
        if all(mapped) and fn_key(mapped[0]) != fn_key(mapped[1]):
            key=pair_key(fn_key(mapped[0]), fn_key(mapped[1]))
            ni_pairs.add(key)
            ni_rows.append({'record':i,'file_a':key[0][0],'start_a':key[0][1],'end_a':key[0][2],
                            'file_b':key[1][0],'start_b':key[1][1],'end_b':key[1][2],
                            'similarity':clone.get('similarity','')})
    write_csv(out / 'nicad_primary_pairs.csv', ni_rows,
              ['record','file_a','start_a','end_a','file_b','start_b','end_b','similarity'])
    data=json.loads(js_file.read_text(encoding='utf-8'))
    all_similar=0
    rejected=Counter()
    eligible=set()
    candidates=[]
    observed_fail=set()
    for idx,c in enumerate(data.get('duplicates', []),1):
        if c.get('kind') != 'similar' or c.get('method') != 'ast':
            continue
        all_similar+=1
        sides=[]
        for side in ('firstFile','secondFile'):
            frag=c.get(side,{})
            file=norm_file(frag.get('name'),known)
            sides.append((file,as_int(frag.get('start')),as_int(frag.get('end'))))
        has_failed=any(x[0] in failed for x in sides)
        if not has_failed:
            rejected['both_files_nicad_nonfailure']+=1
            continue
        observed_fail.update(x[0] for x in sides if x[0] in failed)
        mapped=[]
        reasons=[]
        for file,start,end in sides:
            fn,reason=strict_locate(file,start,end,by_file,args.strict_fraction,args.boundary_tolerance)
            mapped.append(fn)
            reasons.append(reason)
        if not all(mapped):
            rejected['strict_mapping_failed']+=1
            for reason in reasons:
                if reason != 'matched': rejected['endpoint_'+reason]+=1
            continue
        a,b=fn_key(mapped[0]),fn_key(mapped[1])
        if a==b:
            rejected['same_function']+=1
            continue
        key=pair_key(a,b)
        if key in ni_pairs:
            rejected['already_in_nicad']+=1
            continue
        eligible.add(key)
        candidates.append({'record':idx,'file_a':key[0][0],'start_a':key[0][1],
                          'end_a':key[0][2],'function_a':index[key[0]]['name'],
                          'file_b':key[1][0],'start_b':key[1][1],
                          'end_b':key[1][2],'function_b':index[key[1]]['name'],
                          'similarity':c.get('similarity',''),
                          'failed_a':int(key[0][0] in failed),'failed_b':int(key[1][0] in failed)})
    fields=['record','file_a','start_a','end_a','function_a','file_b','start_b','end_b','function_b','similarity','failed_a','failed_b']
    write_csv(out/'jscpd_fallback_pairs.csv',candidates,fields)
    merged=ni_pairs|eligible
    merged_rows=[]
    for pair in sorted(merged):
        a,b=pair
        merged_rows.append({'pair_id':f'B07-{len(merged_rows)+1:06d}',
                            'file_a':a[0],'start_a':a[1],'end_a':a[2], 'function_a':index[a]['name'],
                            'file_b':b[0],'start_b':b[1],'end_b':b[2], 'function_b':index[b]['name'],
                            'source':'jscpd_ast_fallback' if pair in eligible else 'nicad_primary',
                            'ni_parse_failed_a':int(a[0] in failed),'ni_parse_failed_b':int(b[0] in failed)})
    merged_fields=['pair_id','file_a','start_a','end_a','function_a','file_b','start_b','end_b','function_b',
                   'source','ni_parse_failed_a','ni_parse_failed_b']
    write_csv(out/'hybrid_function_clone_pairs.csv',merged_rows,merged_fields)
    fallback_files={fn[0] for pair in eligible for fn in pair if fn[0] in failed}
    coverage=[]
    for file in sorted(failed):
        coverage.append({'file':file,'python_ast_function_count':len(by_file.get(file,[])),
                         'observed_in_ast_similar_report':int(file in observed_fail),
                         'in_strict_fallback_pair':int(file in fallback_files),
                         'strict_fallback_pair_count':sum(1 for pair in eligible if file in (pair[0][0],pair[1][0]))})
    write_csv(out/'fallback_coverage.csv',coverage,
              ['file','python_ast_function_count','observed_in_ast_similar_report',
               'in_strict_fallback_pair','strict_fallback_pair_count'])
    summary={'experiment':'B-07','version':'v1','created_chicago':datetime.now(__import__('zoneinfo').ZoneInfo('America/Chicago')).isoformat(),
             'staged_files':len(manifest),'python_ast_errors':len(errors),'functions_indexed':len(index),
             'ni_parse_failed_files':len(failed),
             'nicad_raw_reports':ni_raw,'nicad_primary_unique_pairs':len(ni_pairs),
             'jscpd_ast_similar_raw_reports':all_similar,
             'jscpd_strict_fallback_unique_pairs':len(eligible),
             'hybrid_unique_pairs':len(merged),
             'ni_failed_files_in_strict_fallback':len(fallback_files),
             'ni_failed_files_in_any_relevant_ast_similar_report':len(observed_fail),
             'fallback_pair_both_failed':sum(1 for p in eligible if p[0][0] in failed and p[1][0] in failed),
             'fallback_pair_one_failed':sum(1 for p in eligible if (p[0][0] in failed) != (p[1][0] in failed)),
             'ni_mapping_rejections':dict(ni_rejections),'jscpd_rejection_counts':dict(rejected),
             'settings':{'ni_function_coverage_threshold':args.ni_threshold,
                         'jscpd_full_function_fraction':args.strict_fraction,
                         'jscpd_boundary_tolerance_lines':args.boundary_tolerance,
                         'jscpd_filter':'kind=similar AND method=ast AND >=1 NiCad parse-failure file',
                         'jscpd_similarity_threshold':'0.70 as fixed in source pilot'},
             'inputs':[{'path':str(p),'sha256':sha256(p)} for p in sources],
             'limitations':[
                 'NiCad is primary. No jscpd pair with both endpoints in NiCad nonfailed files is included.',
                 'Strict function boundary validation checks line spans; it does not prove semantic equivalence.',
                 'Fallback supports clone discovery, not full AST parser coverage or guaranteed correctness.',
                 'Current staging bytes may not equal the historical NiCad execution input.',
                 'NiCad and jscpd similarity measures are not mathematically equivalent.',
                 'Python AST indexing and NiCad function extraction need not have identical scope or semantics.'
             ]}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=f"""# B-07 v1 — Hybrid NiCad + jscpd AST Fallback

- Staged Python files: {len(manifest)}; functions indexed: {len(index)}; Python AST errors: {len(errors)}.
- NiCad primary: {ni_raw} raw reports, **{len(ni_pairs)} unique mapped function pairs**.
- NiCad parsing-failed files: {len(failed)}.
- jscpd AST similar raw reports: {all_similar}.
- Strict jscpd fallback: **{len(eligible)} unique function pairs** (at least one NiCad-failed file).
- Hybrid total: **{len(merged)} unique function pairs**.
- Strict fallback pairs (failed-failed): {summary['fallback_pair_both_failed']}.
- Strict fallback pairs (failed-nonfailed): {summary['fallback_pair_one_failed']}.
- NiCad-failed files participating in strict fallback pairs: {len(fallback_files)}.

## Rejection counts

{json.dumps(dict(rejected),indent=2)}

## Limitations

"""+'\n'.join('- '+item for item in summary['limitations'])+'\n'
    (out/'report.md').write_text(report,encoding='utf-8')
    print(report)
    print('Output:',out)


if __name__=='__main__':
    main()
