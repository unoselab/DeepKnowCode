#!/usr/bin/env python3
"""C-02 v2: independently audit organized C-01 snapshots against matching B-07 references."""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('c02checker', HERE / 'c-02-check-clone-results.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
KINDS = checker.KINDS


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def commit_from(folder):
    meta = folder / 'project_run.json'
    if meta.is_file():
        return read_json(meta).get('commit')
    return None


def find_b07(project, commit, root, mapping):
    if project in mapping:
        p = Path(mapping[project]).expanduser().resolve()
        return p, 'explicit_mapping', []
    candidates = sorted((p for p in root.glob(f'*/projects/{project}')
                         if (p / 'hybrid/hybrid_function_clone_pairs.csv').is_file()),
                        key=lambda p: str(p))
    matched = [p for p in candidates if commit and commit_from(p) == commit]
    if len(matched) == 1:
        return matched[0], 'unique_commit_match', candidates
    if len(matched) > 1:
        # Do not silently choose among multiple references even with matching commits.
        return None, 'ambiguous_same_commit', candidates
    if len(candidates) == 1 and not commit_from(candidates[0]):
        return candidates[0], 'single_reference_commit_unverified', candidates
    if len(candidates) == 1:
        return candidates[0], 'single_reference_commit_mismatch', candidates
    if not candidates:
        return None, 'no_reference', candidates
    return None, 'ambiguous_or_no_commit_match', candidates


def clean(report):
    for field in ('c01', 'b07'):
        item = report.get(field)
        if item:
            for obj in item['pair_files'].values():
                obj.pop('_keys', None)


def audit(c, b, project, github_slug, reference_selection):
    current = checker.inspect(c)
    reference = checker.inspect(b) if b else None
    findings = list(current['findings']) + (['B07_' + x for x in reference['findings']] if reference else [])
    comparisons = {}
    detail = {}
    equality = bool(reference)
    for kind in KINDS:
        x = current['pair_files'][kind]
        y = reference['pair_files'][kind] if reference else None
        if y is None or '_keys' not in x or '_keys' not in y:
            comparisons[kind] = {'available': False}
            equality = False
            continue
        a, z = x['_keys'], y['_keys']
        only_a, only_z = sorted(a - z), sorted(z - a)
        eq = a == z
        equality &= eq
        comparisons[kind] = {
            'exact_equal': eq, 'intersection': len(a & z),
            'only_c01': len(only_a), 'only_b07': len(only_z),
            'jaccard': round(len(a & z) / len(a | z), 8) if a | z else 1.0,
        }
        detail[kind] = {
            'only_c01': [endpoints(k) for k in only_a],
            'only_b07': [endpoints(k) for k in only_z],
        }
    if reference_selection == 'single_reference_commit_mismatch':
        findings.append('SOURCE_COMMIT_MISMATCH')
    if reference_selection == 'single_reference_commit_unverified':
        findings.append('B07_COMMIT_UNVERIFIED')
    internal = 'PASS' if not current['findings'] else 'FAIL'
    if reference is None:
        reproduction = 'REFERENCE_MISSING'
    elif reference['findings']:
        reproduction = 'REFERENCE_INVALID'
    elif 'SOURCE_COMMIT_MISMATCH' in findings:
        reproduction = 'SOURCE_COMMIT_MISMATCH'
    elif 'B07_COMMIT_UNVERIFIED' in findings:
        reproduction = 'UNVERIFIED_REFERENCE'
    else:
        reproduction = 'PASS' if equality else 'MISMATCH'
    status = 'PASS' if internal == 'PASS' and reproduction == 'PASS' else 'FAIL'
    clean({'c01': current, 'b07': reference})
    return {
        'experiment': 'C-02', 'version': 'v2', 'project': project,
        'github_repository': github_slug, 'status': status,
        'internal_audit_status': internal,
        'reproduction_status': reproduction,
        'reference_selection': reference_selection,
        'c01': current, 'b07': reference,
        'pair_comparison': comparisons,
        'pair_differences': detail,
        'findings': findings,
        'note': 'Read-only audit. Reproduction requires exact equality of all three pair sets and verified matching Git commits.',
    }


def endpoints(pair):
    return [{'file': f, 'start': s, 'end': e} for f, s, e in pair]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path.cwd(), help='DeepKnowCode project root')
    ap.add_argument('--c01-batch', type=Path, help='Organized snapshots root, default results/c-01-batch')
    ap.add_argument('--b07-root', type=Path, help='B-07 run collection root, default results/b-07')
    ap.add_argument('--b07-map', type=Path, help='JSON mapping {"project": "path/to/B-07/projects/project"}')
    ap.add_argument('--projects', nargs='+', help='Filter by original subject project ID (e.g. salt sympy)')
    ap.add_argument('--output', type=Path, help='New, non-existing output directory')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    root = a.root.expanduser().resolve()
    base = (a.c01_batch or root/'results/c-01-batch').expanduser().resolve()
    b07 = (a.b07_root or root/'results/b-07').expanduser().resolve()
    mapping = read_json(a.b07_map) if a.b07_map else {}
    if not isinstance(mapping, dict):
        ap.error('--b07-map must be JSON object')
    if not base.is_dir(): ap.error('Missing organized results: ' + str(base))
    if not b07.is_dir() and not mapping: ap.error('Missing B-07 root: ' + str(b07))
    now = datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
    dest = (a.output or root/'results/c-02-batch'/now).expanduser().resolve()
    if dest.exists(): ap.error('Output already exists: ' + str(dest))
    if dest == base or dest == b07 or base in dest.parents or b07 in dest.parents:
        # default output is not inside either source
        ap.error('Output must not be inside any source tree')
    if root not in dest.parents:
        # External outputs are possible, but avoid accidental source trees.
        pass
    requests = set(a.projects or [])
    chosen = []
    for parent in sorted(base.iterdir()):
        if not parent.is_dir() or '--' not in parent.name: continue
        for snapshot in sorted((parent/'runs').glob('*/project_run.json')):
            meta = read_json(snapshot)
            project = meta.get('project')
            if requests and project not in requests: continue
            if meta.get('status') != 'COMPLETED': continue
            chosen.append((project, parent.name, snapshot.parent, meta))
    if requests and requests - {x[0] for x in chosen}:
        ap.error('Unfound subjects: ' + ', '.join(sorted(requests - {x[0] for x in chosen})))
    if not chosen: ap.error('No completed organized snapshots found')
    # If multiple snapshots exist, make the user specify a run; don't silently switch reference.
    duplicates = {project for project, *_ in chosen if sum(x[0] == project for x in chosen) > 1}
    if duplicates: ap.error('Multiple C-01 snapshots for: ' + ', '.join(sorted(duplicates)) + '; select snapshots by removing older ones from input copy, not from original results')
    all_rows = []
    for project, slug, folder, meta in chosen:
        ref, selection, candidates = find_b07(project, meta.get('commit'), b07, mapping)
        if a.dry_run:
            print(f'{project}: C01={folder} B07={ref or "UNRESOLVED"} selection={selection} candidate_count={len(candidates)}')
            continue
        if ref is not None and not ref.is_dir():
            ref = None; selection = 'mapped_path_missing'
        if ref is not None:
            reference_commit = commit_from(ref)
            if reference_commit and meta.get('commit') and reference_commit != meta['commit']:
                selection = 'single_reference_commit_mismatch'
            elif not reference_commit:
                selection = 'single_reference_commit_unverified'
        report = audit(folder, ref, project, slug.replace('--', '/', 1), selection)
        report['c01_commit'] = meta.get('commit')
        report['b07_commit'] = commit_from(ref) if ref else None
        report['b07_candidate_paths'] = [str(x) for x in candidates]
        if ref is None and selection != 'no_reference':
            report['findings'].append('REFERENCE_SELECTION_' + selection.upper())
        if ref is None:
            report['status'] = 'FAIL'
        row = {'project': project, 'repository': slug, 'c01_run': folder.name,
               'b07_reference': str(ref) if ref else None,
               'internal_audit_status': report['internal_audit_status'],
               'reproduction_status': report['reproduction_status'],
               'status': report['status'],
               'nicad_equal': report['pair_comparison'].get(KINDS[0], {}).get('exact_equal'),
               'fallback_equal': report['pair_comparison'].get(KINDS[1], {}).get('exact_equal'),
               'hybrid_equal': report['pair_comparison'].get(KINDS[2], {}).get('exact_equal'),
               'hybrid_only_c01': report['pair_comparison'].get(KINDS[2], {}).get('only_c01'),
               'hybrid_only_b07': report['pair_comparison'].get(KINDS[2], {}).get('only_b07')}
        target = dest/slug/'runs'/folder.name
        target.mkdir(parents=True, exist_ok=False)
        (target/'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        all_rows.append(row)
        print(f'{project}: internal={row["internal_audit_status"]} reproduction={row["reproduction_status"]} output={target}')
    if a.dry_run:
        print('DRY RUN subjects:', len(chosen))
        return 0
    summary = {'experiment': 'C-02-batch', 'version': 'v2', 'created_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
               'total': len(all_rows), 'passed': sum(x['status']=='PASS' for x in all_rows),
               'failed': sum(x['status']!='PASS' for x in all_rows), 'projects': all_rows}
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    with (dest/'summary.csv').open('w', newline='', encoding='utf-8') as fp:
        writer=csv.DictWriter(fp, fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    print(json.dumps({'status':'PASS' if summary['failed']==0 else 'FAIL', 'output':str(dest), 'passed':summary['passed'], 'failed':summary['failed'], 'total':summary['total']}, indent=2))
    return 0 if summary['failed']==0 else 2


if __name__ == '__main__':
    sys.exit(main())
