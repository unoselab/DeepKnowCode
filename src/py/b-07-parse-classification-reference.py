#!/usr/bin/env python3
"""B-03: read-only reclassification of B-02 per-file NiCad probe results."""
import argparse, csv, json, re, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ['file','python_ast','python_error','nicad_status','nicad_category','nicad_failed_count','nicad_extracted_functions','nicad_exit','nicad_error_excerpt','size_bytes']

def integer(value):
    try: return int(value)
    except (TypeError, ValueError): return None

def classify(row):
    msg = row.get('nicad_error_excerpt') or ''
    fail = integer(row.get('nicad_failed_count')) or 0
    extracted = integer(row.get('nicad_extracted_functions'))
    code = integer(row.get('nicad_exit'))
    status = (row.get('nicad_status') or '').strip()
    if fail or re.search(r'source files? failed to parse', msg, re.I):
        return 'NICAD_PARSE_FAILURE'
    if status == 'timeout' or code == 124:
        return 'TIMEOUT'
    if extracted == 0:
        return 'NO_FUNCTIONS_EXTRACTED'
    if code not in (None, 0) and re.search(r'renaming failed', msg, re.I):
        return 'RENAMING_FAILURE'
    if code not in (None, 0):
        return 'OTHER_EXECUTION_ERROR'
    if status == 'not_probed':
        return 'NOT_PROBED'
    if extracted is None:
        return 'UNKNOWN'
    return 'SUCCESS'

def emit_csv(path, records):
    with path.open('w', newline='', encoding='utf-8') as h:
        w = csv.DictWriter(h, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader(); w.writerows(records)

def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    a.add_argument('--input-result', default='results/b-02/20261031-210002-003228')
    a.add_argument('--out', help='New output directory; must not exist')
    args=a.parse_args()
    root=Path(args.root).resolve()
    source=Path(args.input_result)
    if not source.is_absolute(): source=root/source
    source=source.resolve()
    csvpath=source/'file_parse_audit.csv'
    if not csvpath.is_file(): a.error('Missing B-02 input: '+str(csvpath))
    with csvpath.open(newline='',encoding='utf-8') as h:
        reader=csv.DictReader(h)
        needed={'file','python_ast','nicad_status','nicad_failed_count','nicad_extracted_functions','nicad_exit','nicad_error_excerpt'}
        if not needed.issubset(set(reader.fieldnames or [])):
            a.error('Missing required input columns: '+', '.join(sorted(needed-set(reader.fieldnames or []))))
        rows=list(reader)
    for row in rows: row['nicad_category']=classify(row)
    counts=Counter(row['nicad_category'] for row in rows)
    pycounts=Counter(row['python_ast'] for row in rows)
    if args.out:
        out=Path(args.out)
        if not out.is_absolute():out=root/out
    else:
        out=root/'results/b-03'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    out=out.absolute()
    out.mkdir(parents=True,exist_ok=False)
    failed=[r for r in rows if r['nicad_category']=='NICAD_PARSE_FAILURE']
    zero=[r for r in rows if r['nicad_category']=='NO_FUNCTIONS_EXTRACTED']
    emit_csv(out/'file_parse_audit.csv',rows)
    emit_csv(out/'failed_files.csv',failed)
    emit_csv(out/'zero_function_files.csv',zero)
    emit_csv(out/'non_success_files.csv',[r for r in rows if r['nicad_category']!='SUCCESS'])
    with (out/'parse_error_categories.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.writer(h);w.writerow(['category','files']);w.writerows(sorted(counts.items()))
    meta={}
    if (source/'summary.json').is_file():
        meta=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    summary={'status':'COMPLETE','experiment':'B-03','version':'v1','method':'B-02 CSV reclassification (no NiCad rerun)',
             'source_result':str(source),'repo_commit':meta.get('repo_commit'),
             'files_checked':len(rows),'categories':dict(sorted(counts.items())),
             'python_ast_counts':dict(sorted(pycounts.items())),
             'reported_b01_parse_failure_count':282,
             'matches_b01_aggregate':len(failed)==282 if len(rows)==1006 else None,
             'limitations':['Categories use explicit warning and reported counts, not individual grammar diagnosis.',
               'Matching the B-01 aggregate count does not prove exact per-file identity.',
               'No-functions-extracted is not a syntax failure.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# B-03 v1 — NiCad Parse Failure Analysis','',
           f'- Input: `{source}`',f'- Files audited: **{len(rows)}**',
           f'- NiCad explicit parse failures: **{len(failed)}**',
           f'- Zero extracted functions (no parse warning): **{len(zero)}**',
           f'- Python AST failures: **{pycounts.get("fail",0)}**','', '## Categories','']
    lines.extend(f'- {k}: {v}' for k,v in sorted(counts.items()))
    lines += ['','## Limitations','', '- This is classification of recorded B-02 output, not a root-cause grammar diagnosis.', '- B-02 input files and original Salt repository remain unchanged.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('B-03 v1 | B-02 NiCad parse result classification')
    print('SOURCE:',source)
    print('FILES:',len(rows),'| Python AST failed:',pycounts.get('fail',0))
    for k,v in sorted(counts.items()):print(f'  {k:27s} {v:6d}')
    print('[B-03 v1] COMPLETE OUTPUT_DIR='+str(out))
if __name__=='__main__':main()
