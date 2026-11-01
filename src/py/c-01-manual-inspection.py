#!/usr/bin/env python3
"""C-01 v4: detector-result inspection (no human-decision controls)."""
import argparse
import csv
import difflib
import html
import json
import random
from pathlib import Path

GROUPS = ('nicad_primary', 'fallback_failed_failed', 'fallback_failed_other')


def group_of(row):
    if row['source'] == 'nicad_primary':
        return 'nicad_primary'
    if int(row['ni_parse_failed_a']) and int(row['ni_parse_failed_b']):
        return 'fallback_failed_failed'
    return 'fallback_failed_other'


def pick(rows, count, seed):
    groups = {g: [] for g in GROUPS}
    for row in rows:
        groups[group_of(row)].append(row)
    if count == 'all':
        return [(group_of(row), row) for row in rows], {g: len(v) for g, v in groups.items()}
    total = int(count)
    if total == 0:
        return [], {g: len(v) for g, v in groups.items()}
    # Balanced stratified sampling; any remainder is allocated to groups with capacity.
    rng = random.Random(seed)
    picks = {g: [] for g in GROUPS}
    capacity = {g: len(groups[g]) for g in GROUPS}
    for group in GROUPS:
        amount = min(total // len(GROUPS), capacity[group])
        picks[group] = rng.sample(groups[group], amount)
    rest = total - sum(map(len, picks.values()))
    while rest:
        candidates = [g for g in GROUPS if len(picks[g]) < capacity[g]]
        if not candidates:
            break
        for group in candidates:
            if not rest:
                break
            available = [r for r in groups[group] if r not in picks[group]]
            picks[group].append(rng.choice(available))
            rest -= 1
    return [(g, row) for g in GROUPS for row in picks[g]], capacity


def read_function(stage, row, side, cache):
    relative = row['file_' + side]
    path = stage / relative
    if not path.resolve().is_relative_to(stage.resolve()) or not path.is_file():
        raise ValueError(f'Invalid staged source: {relative}')
    if relative not in cache:
        cache[relative] = path.read_text(encoding='utf-8', errors='replace').splitlines()
    content = cache[relative]
    begin, end = int(row['start_' + side]), int(row['end_' + side])
    if begin < 1 or end < begin or end > len(content):
        raise ValueError(f'Invalid function range {relative}:{begin}-{end}')
    return [(n, content[n - 1]) for n in range(begin, end + 1)]


def highlights(left, right):
    a = [line for _, line in left]
    b = [line for _, line in right]
    different_a, different_b = set(), set()
    for op, i, j, k, l in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != 'equal':
            different_a.update(range(i, j))
            different_b.update(range(k, l))
    return different_a, different_b


def code_html(lines, changed):
    return '\n'.join(
        f'<span class="line{" changed" if i in changed else ""}">'
        f'<span class="ln">{number:5d}</span>{html.escape(code)}</span>'
        for i, (number, code) in enumerate(lines)
    )


def make_html(chosen, stage, seed, total_counts):
    cache = {}
    counts = {g: 0 for g in GROUPS}
    sections = []
    for group, row in chosen:
        counts[group] += 1
        a = read_function(stage, row, 'a', cache)
        b = read_function(stage, row, 'b', cache)
        ca, cb = highlights(a, b)
        title_a = f"{row['file_a']}:{row['start_a']}-{row['end_a']} · {row['function_a']}"
        title_b = f"{row['file_b']}:{row['start_b']}-{row['end_b']} · {row['function_b']}"
        pair = html.escape(row['pair_id'])
        sections.append(f'''<article class="case" id="{pair}" data-group="{group}" data-search="{html.escape((row['pair_id']+' '+title_a+' '+title_b).lower(), quote=True)}">
<header><a href="#{pair}">{pair}</a><span class="pill">{html.escape(group.replace('_',' / '))}</span><span>NiCad parse failure · A={row['ni_parse_failed_a']} · B={row['ni_parse_failed_b']}</span></header>
<div class="cols"><section><h3>A · {html.escape(title_a)}</h3><pre><code>{code_html(a,ca)}</code></pre></section><section><h3>B · {html.escape(title_b)}</h3><pre><code>{code_html(b,cb)}</code></pre></section></div></article>''')
    stat = ' · '.join(f'{g}: {counts[g]}/{total_counts[g]}' for g in GROUPS)
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C-01 · Function Clone Detector Results</title>
<style>
:root{font:14px/1.5 system-ui,-apple-system,sans-serif;background:#f5f7fb;color:#1a2840}*{box-sizing:border-box}body{max-width:1800px;margin:0 auto;padding:25px}h1{margin:0 0 4px;font-size:25px}p{color:#516079}.toolbar{position:sticky;top:0;z-index:10;background:#f5f7fb;border-bottom:1px solid #d7dfec;padding:12px 0;display:flex;align-items:center;gap:12px;flex-wrap:wrap}input,select{padding:9px;border:1px solid #b8c5d8;border-radius:7px;background:#fff;font:inherit}.case{border:1px solid #d5dfea;border-radius:10px;background:#fff;margin:20px 0;overflow:hidden;box-shadow:0 3px 12px #1f2f4a0b}.case[hidden]{display:none}header{background:#eaf0f8;padding:13px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:13px}header a{font-weight:750;color:#1c4d8e}.pill{border-radius:20px;background:#d9e6fb;padding:2px 10px}.cols{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:1px;background:#d0dae7}.cols section{background:#fff;min-width:0}.cols h3{font-size:13px;margin:0;padding:12px;overflow-wrap:anywhere;border-bottom:1px solid #d7dfeb}pre{margin:0;padding:12px;font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;overflow:auto;max-height:850px;tab-size:4}.line{display:block;min-width:max-content;white-space:pre}.line.changed{background:#fff0ba}.ln{display:inline-block;color:#8995a8;user-select:none;padding-right:14px;border-right:1px solid #e0e6ef;margin-right:12px}.intro{border-left:3px solid #4977b3;padding-left:12px;margin:14px 0 20px}.stats{font-size:12px;color:#64748b} @media(max-width:950px){.cols{grid-template-columns:1fr}}@media print{.toolbar{display:none}pre{max-height:none;white-space:pre-wrap}.case{break-inside:avoid}}
</style></head><body>'''
    page += f'<h1>C-01 · Function Clone Detector Results</h1><p>NiCad primary + jscpd AST fallback · seed {seed} · {len(chosen)} displayed pairs</p><div class="intro">Side-by-side full function source. Yellow highlights line differences, not semantic inequality. These are unverified detector candidates; there are no decision controls or automatic labels.</div><p class="stats">{html.escape(stat)}</p>'
    page += '''<div class="toolbar"><label>Detector group <select id="group"><option value="all">All</option><option value="nicad_primary">NiCad primary</option><option value="fallback_failed_failed">jscpd: failed / failed</option><option value="fallback_failed_other">jscpd: failed / nonfailed</option></select></label><label>Find pair / file / function <input id="search" placeholder="e.g. salt/states or C01-000001"></label><span id="shown"></span></div>'''
    page += '\n'.join(sections)
    page += '''<script>const cases=[...document.querySelectorAll('.case')],group=document.getElementById('group'),search=document.getElementById('search'),shown=document.getElementById('shown');function filter(){const q=search.value.trim().toLowerCase();let count=0;for(const c of cases){const match=(group.value==='all'||c.dataset.group===group.value)&&(!q||c.dataset.search.includes(q));c.hidden=!match;if(match)count++;}shown.textContent=`${count.toLocaleString()} / ${cases.length.toLocaleString()} pairs`;}group.addEventListener('change',filter);search.addEventListener('input',filter);filter();</script></body></html>'''
    return page


def main():
    parser = argparse.ArgumentParser(description='View C-01 hybrid clone pairs in console and HTML (no review decisions).')
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', default='12', help='Total sample count, or "all" for every pair (default: 12).')
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--console-lines', default='12', help='Lines of each function shown in console, or "all" (default: 12).')
    args = parser.parse_args()
    if args.samples != 'all' and (not args.samples.isdecimal()):
        parser.error('--samples must be a nonnegative integer or all')
    if args.console_lines != 'all' and (not args.console_lines.isdecimal()):
        parser.error('--console-lines must be a nonnegative integer or all')
    with (args.output / 'hybrid_function_clone_pairs.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    chosen, totals = pick(rows, args.samples, args.seed)
    dest_html = args.output / 'clone_inspection.html'
    dest_txt = args.output / 'clone_inspection_console.txt'
    dest_html.write_text(make_html(chosen, args.stage, args.seed, totals), encoding='utf-8')
    cache = {}
    parts = [f'C-01 v4 · Detector result inspection · displayed {len(chosen)} / {len(rows)} pairs', f'Selection: {args.samples}; seed: {args.seed}; console-lines: {args.console_lines}', 'Full function code and highlighted line differences: clone_inspection.html', 'No manual verdict is assigned.']
    for group, row in chosen:
        parts.append('\n' + '=' * 78)
        parts.append(f"{row['pair_id']} [{group}]  source={row['source']}")
        for side in ('a', 'b'):
            code = read_function(args.stage, row, side, cache)
            parts.append(f"{side.upper()}: {row['file_'+side]}:{row['start_'+side]}-{row['end_'+side]}  {row['function_'+side]}  ({len(code)} lines)")
            shown = code if args.console_lines == 'all' else code[:int(args.console_lines)]
            parts.extend(f'{n:5d}  {line}' for n, line in shown)
            if len(shown) != len(code):
                parts.append(f'  ... {len(code)-len(shown)} more lines; see HTML (or --console-lines all)')
    content = '\n'.join(parts) + '\n'
    dest_txt.write_text(content, encoding='utf-8')
    print(content)
    path = args.output / 'summary.json'
    summary = json.loads(path.read_text(encoding='utf-8'))
    summary['version'] = 'v4'
    summary['inspection'] = {'displayed_pairs': len(chosen), 'total_pairs': len(rows), 'samples': args.samples, 'seed': args.seed, 'console_lines': args.console_lines, 'group_population': totals, 'html': dest_html.name, 'console': dest_txt.name, 'decisions_collected': False}
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('HTML:', dest_html)
    print('Console:', dest_txt)


if __name__ == '__main__':
    main()
