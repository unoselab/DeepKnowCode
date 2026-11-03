#!/usr/bin/env python3
"""C-02 v2: independent, read-only audit of C-01 results and optional B-07 reference."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

KINDS = ('nicad_primary_pairs.csv', 'jscpd_fallback_pairs.csv', 'hybrid_function_clone_pairs.csv')
FIELDS = ('file_a','start_a','end_a','file_b','start_b','end_b')

def csv_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rd=csv.DictReader(f)
        if not rd.fieldnames or not set(FIELDS).issubset(rd.fieldnames):
            raise ValueError(f'Missing required pair columns: {path}')
        return list(rd)

def key(row):
    ends=[]
    for side in ('a','b'):
        rel=row['file_'+side].replace('\\','/')
        if not rel or rel.startswith('/') or '..' in Path(rel).parts:
            raise ValueError('Unsafe or empty relative source path: '+repr(rel))
        start,end=int(row['start_'+side]),int(row['end_'+side])
        if start<1 or end<start:raise ValueError(f'Invalid function boundaries: {rel} {start}-{end}')
        ends.append((rel,start,end))
    if ends[0]==ends[1]:raise ValueError('Self pair')
    return tuple(sorted(ends))

def analyze(base):
    out={}
    for kind in KINDS:
        path=base/'hybrid'/kind
        if not path.is_file():
            out[kind]={'status':'MISSING'}
            continue
        rows=csv_rows(path);keys=[];invalid=[]
        for i,row in enumerate(rows,2):
            try:keys.append(key(row))
            except (ValueError,KeyError) as exc:invalid.append({'line':i,'error':str(exc)})
        counts=Counter(keys);duplicates=sum(n-1 for n in counts.values())
        out[kind]={'status':'OK' if not invalid and not duplicates else 'INVALID',
                   'rows':len(rows),'unique_pairs':len(counts),'duplicate_rows':duplicates,
                   'invalid':invalid[:25], 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        out[kind]['_keys']=set(counts)
    return out

def inspect(base):
    result={'folder':str(base),'pair_files':analyze(base),'summary':None,'findings':[]}
    summary=base/'hybrid'/'summary.json'
    if summary.is_file():
        result['summary']=json.loads(summary.read_text(encoding='utf-8'))
        counts={ 'nicad_primary_pairs.csv':'nicad_primary_unique_pairs',
                 'jscpd_fallback_pairs.csv':'jscpd_strict_fallback_unique_pairs',
                 'hybrid_function_clone_pairs.csv':'hybrid_unique_pairs'}
        for fname,field in counts.items():
            item=result['pair_files'][fname]
            if '_keys' in item and field in result['summary'] and item['unique_pairs']!=result['summary'][field]:
                result['findings'].append(f'COUNT_MISMATCH {fname}: {item["unique_pairs"]} != summary {result["summary"][field]}')
    else: result['findings'].append('MISSING_SUMMARY')
    files=result['pair_files']
    if all('_keys' in files[k] for k in KINDS):
        ni,js,hy=(files[k]['_keys'] for k in KINDS)
        if ni&js:result['findings'].append('PRIMARY_FALLBACK_OVERLAP '+str(len(ni&js)))
        if ni|js !=hy:result['findings'].append('HYBRID_UNION_MISMATCH '+str(len((ni|js)^hy)))
    if any(v['status']!='OK' for v in files.values()):result['findings'].append('MISSING_OR_INVALID_PAIR_CSV')
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--c01',required=True,type=Path,help='C-01 single-project output folder')
    ap.add_argument('--b07',type=Path,help='Optional completed B-07 single-project output folder')
    ap.add_argument('--output',type=Path,help='New C-02 output directory; never writes to C-01/B-07')
    a=ap.parse_args(); c=a.c01.resolve(); b=a.b07.resolve() if a.b07 else None
    if not c.is_dir():ap.error('C-01 folder missing: '+str(c))
    for p in ([c]+([b] if b else [])):
        if p is not None and not p.is_dir():ap.error('Folder missing: '+str(p))
    now=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
    output=(a.output or (Path.cwd()/'results'/'c-02'/now)).resolve()
    if output.exists():ap.error('Refusing existing output directory: '+str(output))
    if output==c or (b and output==b):ap.error('Output must be independent')
    current=inspect(c);reference=inspect(b) if b else None
    comparisons={}
    if reference:
        for kind in KINDS:
            x=current['pair_files'][kind];y=reference['pair_files'][kind]
            if '_keys' in x and '_keys' in y:
                cx,bx=x['_keys'],y['_keys']
                comparisons[kind]={'exact_equal':cx==bx,'intersection':len(cx&bx),
                                   'only_c01':len(cx-bx),'only_b07':len(bx-cx),
                                   'jaccard':round(len(cx&bx)/len(cx|bx),8) if cx|bx else 1.0}
            else:comparisons[kind]={'available':False}
    for item in [current,reference]:
        if item:
            for record in item['pair_files'].values():record.pop('_keys',None)
    findings=current['findings'] + (['B07_'+s for s in reference['findings']] if reference else [])
    internal_status='PASS' if not current['findings'] else 'FAIL'
    reference_status='NOT_REQUESTED' if not reference else ('PASS' if not reference['findings'] else 'FAIL')
    reproduction_status=('NOT_REQUESTED' if not reference else
                         ('PASS' if reference_status=='PASS' and all(v.get('exact_equal') is True for v in comparisons.values()) and len(comparisons)==len(KINDS) else 'MISMATCH'))
    status='PASS' if internal_status=='PASS' and reference_status in ('PASS','NOT_REQUESTED') and reproduction_status in ('PASS','NOT_REQUESTED') else 'FAIL'
    if reproduction_status=='MISMATCH': findings.append('PAIR_REPRODUCTION_MISMATCH')
    report={'experiment':'C-02','version':'v2','status':status,
            'created_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
            'internal_audit_status':internal_status, 'reference_audit_status':reference_status,
            'reproduction_status':reproduction_status,
            'c01':current,'b07':reference,'pair_comparison':comparisons,
            'findings':findings,'note':'C-02 reads results only. It does not perform clone detection or alter source results. Exact pair equality needs matching source scope and commits.'}
    output.mkdir(parents=True)
    (output/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':status, 'internal_audit_status':internal_status,
                      'reproduction_status':reproduction_status,
                      'output':str(output),'findings':findings,'pair_comparison':comparisons},indent=2))
    return 0 if status=='PASS' else 2
if __name__=='__main__':sys.exit(main())
