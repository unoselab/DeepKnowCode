#!/usr/bin/env python3
"""C-01 v9: organize completed detector artifacts, never run detection or change sources."""
import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

VERSION = 'v9'
REQUIRED = (
    'project_run.json', 'nicad.log', 'jscpd.log', 'merge.log',
    'file_parse_audit.csv', 'filter_summary.json',
    'jscpd/jscpd-report.json', 'hybrid/summary.json',
    'hybrid/nicad_primary_pairs.csv', 'hybrid/jscpd_fallback_pairs.csv',
    'hybrid/hybrid_function_clone_pairs.csv',
)
OPTIONAL = ('filter_audit.csv', 'parse_categories.json', 'hybrid/input_manifest.csv',
            'hybrid/report.md', 'hybrid/fallback_coverage.csv',
            'hybrid/python_ast_errors.csv', 'comparison.log')
SHA = re.compile(r'^[0-9a-f]{40}$')
SEG = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')


def github_slug(remote):
    """Accept GitHub HTTPS, ssh://git@github.com, and git@github.com:owner/repo."""
    remote = (remote or '').strip()
    if remote.startswith('git@github.com:'):
        path = remote.split(':', 1)[1]
    else:
        u = urlparse(remote)
        if u.scheme not in ('http', 'https', 'ssh', 'git') or u.hostname != 'github.com':
            raise ValueError('Not a recognized GitHub remote: ' + remote)
        path = u.path.lstrip('/')
    if path.endswith('.git'):
        path = path[:-4]
    parts = path.strip('/').split('/')
    if len(parts) != 2 or not all(SEG.fullmatch(p) and p not in ('.', '..') for p in parts):
        raise ValueError('Unexpected GitHub owner/repo in remote: ' + remote)
    return parts[0], parts[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def chosen_runs(root, batch_path=None, projects=None):
    selected = {}
    constraints = None
    if batch_path is not None:
        summary = json.loads((batch_path / 'summary.json').read_text(encoding='utf-8'))
        constraints = {}
        for item in summary['results']:
            if item.get('status') not in ('COMPLETED_COMMAND', 'SKIPPED_COMPLETED'):
                raise ValueError(f"Batch entry not completed: {item.get('project')}")
            project, commit = item['project'], item['commit']
            if project in constraints:
                raise ValueError('Duplicated batch project: ' + project)
            constraints[project] = commit
    for file in sorted((root / 'results/c-01').glob('*/project_run.json')):
        try:
            data = json.loads(file.read_text(encoding='utf-8'))
        except (OSError, ValueError) as e:
            raise ValueError(f'Malformed project metadata {file}: {e}') from e
        if data.get('status') != 'COMPLETED':
            continue
        project = data.get('project', '')
        if not SEG.fullmatch(project) or project in ('.', '..'):
            raise ValueError(f'Invalid project ID: {file}')
        if projects and project not in projects:
            continue
        if constraints is not None and constraints.get(project) != data.get('commit'):
            continue
        if not SHA.fullmatch(data.get('commit', '')):
            raise ValueError(f'Invalid Git SHA: {file}')
        # Sort by timestamp run id, so latest completed run for each project wins.
        current = selected.get(project)
        if current is None or file.parent.name > current[0].name:
            selected[project] = (file.parent, data)
    if constraints is not None:
        required = set(constraints) & (set(projects) if projects else set(constraints))
        if missing := sorted(required - set(selected)):
            raise ValueError('Missing matching completed detector results for: ' + ', '.join(missing))
    if projects and (missing := sorted(set(projects) - set(selected))):
        raise ValueError('Requested completed projects not found: '+', '.join(missing))
    if not selected:
        raise ValueError('No completed C-01 project_run.json found')
    return selected


def copy_snapshot(source, dest, data, dry_run=False):
    owner, name = github_slug(data.get('remote'))
    project = data['project']
    # Both project ID and actual Github owner/repository recorded, no inferred ownership.
    label = f'{owner}--{name}'
    run_id = source.name
    target = dest / label / 'runs' / run_id
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'Refusing to overwrite existing snapshot: {target}')
    files = []
    for relative in REQUIRED + OPTIONAL:
        src = source / relative
        if relative in REQUIRED and (not src.is_file() or src.is_symlink()):
            raise FileNotFoundError(f'Missing required artifact: {src}')
        if not src.exists():
            continue
        if not src.is_file() or src.is_symlink():
            raise ValueError(f'Unsafe input artifact: {src}')
        files.append(relative)
    # The XML report file has run-specific basename. Require exactly one.
    xmls = list((source / 'nicad_xml').glob('*functions-blind-clones-0.30.xml'))
    if len(xmls) != 1 or xmls[0].is_symlink():
        raise ValueError(f'Expected one NiCad clones XML under {source / "nicad_xml"}, got {len(xmls)}')
    files.append('nicad_xml/' + xmls[0].name)
    entry = {'project': project, 'github_slug': f'{owner}/{name}', 'github_owner': owner,
             'github_repository': name, 'commit': data['commit'], 'remote': data['remote'],
             'c01_run_id': run_id, 'source_run': str(source), 'snapshot': str(target),
             'file_count': len(files), 'files': {}}
    if dry_run:
        entry['action'] = 'WOULD_COPY'
        return entry
    # Make destination only after all required source files were validated.
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    try:
        for relative in files:
            src = source / relative
            dst = target / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            source_hash, target_hash = digest(src), digest(dst)
            if source_hash != target_hash:
                raise IOError(f'SHA256 mismatch while copying {relative}')
            entry['files'][relative] = {'bytes': dst.stat().st_size, 'sha256': target_hash}
        (target / 'artifact_manifest.json').write_text(json.dumps(entry, indent=2) + '\n', encoding='utf-8')
    except BaseException:
        # Only remove a directory created in this invocation, never the source.
        shutil.rmtree(target)
        raise
    entry['action'] = 'COPIED'
    return entry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--output', type=Path, default=Path('results/c-01-batch'), help='Destination root, alongside existing batch run folders')
    p.add_argument('--batch', type=Path, help='Optional results/c-01-batch/<timestamp> to restrict projects/commits')
    p.add_argument('--projects', nargs='+', help='Only these project IDs')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    output = (args.output if args.output.is_absolute() else root / args.output).resolve()
    original = (root / 'results/c-01').resolve()
    if output == original or original in output.parents:
        p.error('Output must not be inside original C-01 detection results')
    batch = (args.batch if args.batch.is_absolute() else root / args.batch).resolve() if args.batch else None
    if args.projects and len(set(args.projects)) != len(args.projects):
        p.error('Duplicate --projects')
    try:
        selected = chosen_runs(root, batch, set(args.projects) if args.projects else None)
        # Detect owner--repo collisions, ensure every snapshot path is unoccupied before copying.
        planned = []
        targets = set()
        for project, (source, data) in sorted(selected.items()):
            owner, name = github_slug(data.get('remote'))
            target = output / f'{owner}--{name}' / 'runs' / source.name
            if target in targets or target.exists() or target.is_symlink():
                raise ValueError(f'Output collision or existing snapshot: {target}')
            targets.add(target)
            planned.append((source, data))
        entries = []
        for source, data in planned:
            entry = copy_snapshot(source, output, data, args.dry_run)
            entries.append(entry)
            print(f"{entry['action']}: {entry['github_slug']} run={entry['c01_run_id']} files={entry['file_count']}", flush=True)
        status = 'DRY_RUN' if args.dry_run else 'COMPLETED'
        summary = {'version': VERSION, 'status': status, 'output_root': str(output),
                   'selected': len(entries), 'projects': entries}
        if not args.dry_run:
            stamp = datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
            path = output / ('organizer-' + stamp + '.json')
            path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
            print('SUMMARY:', path)
        else:
            print(json.dumps({'status': status, 'selected': len(entries)}, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print('C-01 RESULT ORGANIZER ERROR:', exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
