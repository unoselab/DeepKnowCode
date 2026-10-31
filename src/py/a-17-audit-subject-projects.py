#!/usr/bin/env python3
"""A-17: count original held-out ground-truth pairs by subject project.

This is descriptive data-construction accounting, NOT model predictions.
Positive label exactly matches A-06/A-12/A-14/A-16: the second underscore-
separated component of both function IDs is identical.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ['language','project','pairs','positive_pairs','negative_pairs','positive_pct','negative_pct','unique_function_ids','positive_function_ids','negative_function_ids','source_file','source_sha256']

def pair_label(a, b):
    x, y = a.split('_'), b.split('_')
    if len(x)<2 or len(y)<2 or not x[1] or not y[1]:
        raise ValueError('Malformed function ID(s): %r / %r' % (a,b))
    return int(x[1] == y[1])

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):h.update(block)
    return h.hexdigest()

def write_csv(path,rows,fields):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader();w.writerows(rows)

def project_rows(stage_root):
    results=[]; all_project_labels=set()
    for lang in ('java','python'):
        directory=stage_root/'ground_truth'/lang
        if not directory.is_dir():raise FileNotFoundError('Missing staged ground truth directory: '+str(directory))
        files=sorted(directory.glob('pred_*_with_refactorability.csv'))
        if not files:raise RuntimeError('No project CSV in '+str(directory))
        for path in files:
            name=path.name[len('pred_'):-len('_with_refactorability.csv')]
            if not name or (lang,name) in all_project_labels:raise RuntimeError('Duplicate/blank subject project: '+str(path))
            all_project_labels.add((lang,name))
            ids=set();pos_ids=set();neg_ids=set();counts=Counter()
            with path.open('r',encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f)
                if not reader.fieldnames or not {'pair_left_func','pair_right_func'}.issubset(reader.fieldnames):
                    raise RuntimeError('Missing pair columns in '+str(path))
                for line_no,row in enumerate(reader,2):
                    a=(row.get('pair_left_func') or '').strip(); b=(row.get('pair_right_func') or '').strip()
                    if not a or not b:raise RuntimeError(f'Missing ID: {path}:{line_no}')
                    try: label=pair_label(a,b)
                    except ValueError as e:raise RuntimeError(f'{path}:{line_no}: {e}') from e
                    counts[label]+=1; ids.update((a,b))
                    (pos_ids if label else neg_ids).update((a,b))
            n=counts[0]+counts[1]
            if not n:raise RuntimeError('Empty ground-truth project: '+str(path))
            results.append({'language':lang,'project':name,'pairs':n,'positive_pairs':counts[1],
                            'negative_pairs':counts[0],'positive_pct':round(100*counts[1]/n,4),
                            'negative_pct':round(100*counts[0]/n,4),
                            'unique_function_ids':len(ids),'positive_function_ids':len(pos_ids),
                            'negative_function_ids':len(neg_ids),'source_file':str(path.relative_to(stage_root)),
                            'source_sha256':sha256(path)})
    return results

def aggregate(rows, lang):
    rr=[r for r in rows if lang=='all' or r['language']==lang]
    pairs=sum(r['pairs'] for r in rr); positives=sum(r['positive_pairs'] for r in rr)
    counts=[r['pairs'] for r in rr]
    return {'language':lang,'projects':len(rr),'pairs':pairs,'positive_pairs':positives,
            'negative_pairs':pairs-positives,'positive_pct':round(100*positives/pairs,4),
            'negative_pct':round(100*(pairs-positives)/pairs,4),
            'min_pairs_per_project':min(counts),'max_pairs_per_project':max(counts),
            'median_pairs_per_project':statistics.median(counts),
            'mean_pairs_per_project':round(statistics.mean(counts),4),
            'sd_pairs_per_project_population':round(statistics.pstdev(counts),4),
            'projects_exactly_balanced':sum(r['positive_pairs']==r['negative_pairs'] for r in rr)}

def md_table(rows,headers):
    def esc(x):return str(x).replace('|',r'\|')
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(esc(r[k]) for k in headers)+' |\n' for r in rows)

def render_markdown(projects,summaries):
    shown=[{'language':r['language'],'project':r['project'],'pairs':f"{r['pairs']:,}",
            'positive_pairs':f"{r['positive_pairs']:,}",'negative_pairs':f"{r['negative_pairs']:,}",
            'positive_pct':f"{r['positive_pct']:.2f}%"} for r in projects]
    brief=[{'language':x['language'],'projects':x['projects'],'pairs':f"{x['pairs']:,}",
            'positive_pairs':f"{x['positive_pairs']:,}",'negative_pairs':f"{x['negative_pairs']:,}",
            'positive_pct':f"{x['positive_pct']:.2f}%"} for x in summaries]
    return ('# A-17 Subject-project distribution (held-out evaluation dataset)\n\n'
            '**Scope:** Original staged held-out ground-truth CSV pairs, not the entire training corpus. '
            'Labels reproduce the existing A-06/A-12/A-14/A-16 function-ID matching rule. '
            'Projects reflect ground-truth CSV filenames. Counts are pair counts, not unique code snippets.\n\n'
            '## Summary by language\n\n'+md_table(brief,['language','projects','pairs','positive_pairs','negative_pairs','positive_pct'])+'\n'
            '## Each subject project\n\n'+md_table(shown,['language','project','pairs','positive_pairs','negative_pairs','positive_pct'])+'\n'
            '## Paper-ready paragraph (English; verify before submission)\n\n'
            f"The held-out evaluation corpus comprises {summaries[-1]['pairs']:,} labeled code pairs from "
            f"{summaries[-1]['projects']} subject projects: {summaries[0]['pairs']:,} pairs across "
            f"{summaries[0]['projects']} Java projects and {summaries[1]['pairs']:,} pairs across "
            f"{summaries[1]['projects']} Python projects. There are "
            f"{summaries[-1]['positive_pairs']:,} positive (clone) pairs and "
            f"{summaries[-1]['negative_pairs']:,} negative (non-clone) pairs. "
            'Table A-17 lists the per-project class distribution. '
            'This paragraph describes the held-out evaluation split only, and should not be presented as the training-data composition.\n')

def render_latex(projects,summaries):
    def esc(s):
        out=str(s)
        for c,repl in [('\\',r'\textbackslash{}'),('_',r'\_'),('&',r'\&'),('%',r'\%'),('#',r'\#'),('$',r'\$')]:out=out.replace(c,repl)
        return out
    lines=[r'\begin{longtable}{llrrrr}',r'\caption{Held-out code-pair distribution by subject project. Positive/negative labels are derived from the evaluation ground truth.}\label{tab:subject-project-distribution}\\',r'\toprule',r'Language & Project & All pairs & Clone & Non-clone & Clone (\%) \\',r'\midrule',r'\endfirsthead',r'\toprule',r'Language & Project & All pairs & Clone & Non-clone & Clone (\%) \\',r'\midrule',r'\endhead']
    for r in projects:
        lines.append(f"{esc(r['language'])} & {esc(r['project'])} & {r['pairs']:,} & {r['positive_pairs']:,} & {r['negative_pairs']:,} & {r['positive_pct']:.2f} "+r'\\')
    lines.extend([r'\midrule'])
    for r in summaries:
        lines.append(f"{esc(r['language'].upper())} & {r['projects']} projects & {r['pairs']:,} & {r['positive_pairs']:,} & {r['negative_pairs']:,} & {r['positive_pct']:.2f} "+r'\\')
    lines.extend([r'\bottomrule',r'\end{longtable}','% Requires \\usepackage{booktabs,longtable}\n'])
    return '\n'.join(lines)

def console_table(headers, rows, right_align=()):
    """Dependency-free, stable-width ASCII table for server logs."""
    data = [[str(value) for value in row] for row in rows]
    widths = [max(len(str(h)), *(len(row[i]) for row in data)) for i, h in enumerate(headers)]
    bar = '+-' + '-+-'.join('-' * w for w in widths) + '-+'
    def render(row):
        return '| ' + ' | '.join((value.rjust(widths[i]) if i in right_align else value.ljust(widths[i]))
                                 for i, value in enumerate(row)) + ' |'
    return '\n'.join([bar, render(headers), bar] + [render(row) for row in data] + [bar])


def print_console_report(projects, summaries, output_dir, detail='all'):
    print('\n' + '=' * 82)
    print('A-17  SUBJECT PROJECT DISTRIBUTION | HELD-OUT EVALUATION DATASET')
    print('=' * 82)
    print('SCOPE: Ground-truth evaluation pairs (NOT adaptation training/validation)')
    print('LABEL: Positive iff second underscore-delimited function-ID component matches')
    print('\n[1] LANGUAGE SUMMARY')
    headers = ['Language', 'Projects', 'All pairs', 'Clone (+)', 'Non-clone (-)', 'Clone %', 'Median/project']
    values = [[x['language'].upper(), f"{x['projects']:,}", f"{x['pairs']:,}",
               f"{x['positive_pairs']:,}", f"{x['negative_pairs']:,}",
               f"{x['positive_pct']:.2f}%", f"{x['median_pairs_per_project']:,.1f}"] for x in summaries]
    print(console_table(headers, values, right_align=range(1,len(headers))))
    print('\n[2] PROJECT SIZE DISTRIBUTION')
    values = [[x['language'].upper(), f"{x['min_pairs_per_project']:,}",
               f"{x['max_pairs_per_project']:,}",f"{x['mean_pairs_per_project']:,.2f}",
               f"{x['sd_pairs_per_project_population']:,.2f}",
               f"{x['projects_exactly_balanced']:,} / {x['projects']:,}"] for x in summaries]
    print(console_table(['Language','Min pairs','Max pairs','Mean pairs','Population SD','Balanced projects'], values,
                        right_align=(1,2,3,4,5)))
    print('\n[3] TOP 5 PROJECTS BY PAIRS (PER LANGUAGE)')
    leaders=[]
    for lang in ('java','python'):
        for r in sorted((p for p in projects if p['language']==lang),key=lambda x:(-x['pairs'],x['project']))[:5]:
            leaders.append([lang.upper(), r['project'],f"{r['pairs']:,}",f"{r['positive_pairs']:,}",
                            f"{r['negative_pairs']:,}",f"{r['pairs']/next(x['pairs'] for x in summaries if x['language']==lang)*100:.2f}%"])
    print(console_table(['Language','Project','Pairs','Clone (+)','Non-clone (-)','Language share'], leaders,
                        right_align=(2,3,4,5)))
    if detail=='all':
        print('\n[4] COMPLETE PER-PROJECT DISTRIBUTION (72 SUBJECT PROJECTS)')
        for lang in ('java','python'):
            print('\n' + lang.upper() + ' SUBJECT PROJECTS')
            sorted_rows=sorted((p for p in projects if p['language']==lang),key=lambda x:(-x['pairs'],x['project']))
            values=[[str(i),r['project'],f"{r['pairs']:,}",f"{r['positive_pairs']:,}",
                     f"{r['negative_pairs']:,}",f"{r['positive_pct']:.2f}%",f"{r['unique_function_ids']:,}"]
                    for i,r in enumerate(sorted_rows,1)]
            print(console_table(['#','Project','Pairs','Clone (+)','Non-clone (-)','Clone %','Unique IDs'],values,
                                right_align=(0,2,3,4,5,6)))
    else:
        print('\n[4] Per-project table suppressed (--console-detail summary)')
    print('\n[5] OUTPUT FILES')
    for filename in ('summary.json','project_distribution.csv','language_distribution.csv','paper_table.md','paper_table.tex'):
        print('  ' + str(output_dir / filename))
    print('\nNOTE: Pair counts are not unique-code counts; project-level unique IDs are not additive.')
    print('NOTE: These figures do not describe the adaptation training or validation corpus.')
    print('[A-17] COMPLETE: %d projects, %s pairs\n' % (summaries[-1]['projects'], f"{summaries[-1]['pairs']:,}"),flush=True)


def run(args):
    if args.output_dir.exists():raise FileExistsError('Refusing to overwrite output: '+str(args.output_dir))
    projects=project_rows(args.data_root)
    summaries=[aggregate(projects,l) for l in ('java','python','all')]
    if args.expected_projects and len(projects)!=args.expected_projects:raise RuntimeError('Project count differs from expected')
    if args.expected_pairs and summaries[-1]['pairs']!=args.expected_pairs:raise RuntimeError('Pair count differs from expected')
    for r in summaries:
        assert r['pairs']==r['positive_pairs']+r['negative_pairs']
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir/'project_distribution.csv',projects,FIELDS)
    write_csv(args.output_dir/'language_distribution.csv',summaries,list(summaries[0]))
    (args.output_dir/'paper_table.md').write_text(render_markdown(projects,summaries),encoding='utf-8')
    (args.output_dir/'paper_table.tex').write_text(render_latex(projects,summaries),encoding='utf-8')
    payload={'status':'COMPLETE','scope':'held-out ground truth, per project','project_label_source':'CSV filename',
             'label_definition':'int(left_id.split("_")[1] == right_id.split("_")[1])',
             'created_utc':datetime.now(timezone.utc).isoformat(),'data_root':str(args.data_root),
             'source_files':len(projects),'totals':summaries,
             'warnings':['These are held-out evaluation pair distributions, not training corpus statistics.',
                         'Function IDs may appear in multiple pairs; pair counts are not unique-function counts.',
                         'Unique function ID counts per project are not additive across projects.',
                         'Label interpretation inherits the existing function-ID convention.']}
    (args.output_dir/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print_console_report(projects,summaries,args.output_dir,args.console_detail)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('data/evaluation'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--expected-projects',type=int,default=72)
    p.add_argument('--expected-pairs',type=int,default=39002)
    p.add_argument('--console-detail',choices=('all','summary'),default='all',
                   help='Full 72-project console table (default), or compact summary')
    run(p.parse_args())
if __name__=='__main__':main()
