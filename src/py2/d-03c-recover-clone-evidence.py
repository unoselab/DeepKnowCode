#!/usr/bin/env python3
"""D-03c v2: conservative C-10 -> C-03 -> C-01 Python lineage recovery.

Read-only source analysis. Never infer T1/T2/T3/T4 from detector agreement,
textual similarity or NiCad threshold. Pair/project evidence requires a strict
match to a known training pair AND (for C-01) two function-code fingerprints.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

APPROVED = {
    'data.jsonl':'da8fdd6362df692e225a5bf5b72278fd6211cef75e48e6a539dda637b76a1843',
    'train.txt':'e8c8998f2e73a3587190904fb258ac1ccda306f0be62dec499622f4258603693',
    'valid.txt':'33c5a0506fdbbd8ff5f9604a9be1d4fba1b6b6d52943ff11ff53928861ac48a7'
}
SPLITS = (('train','train.txt'),('valid','valid.txt'))
PAIR_FIELDS = [('id_a','id_b'),('id1','id2'),('left_id','right_id'),('function_id_a','function_id_b'),('id_left','id_right')]
PROJECT_FIELDS = ('project_id','project','project_name','repository','repository_name','repo_name')
SKIP_DIRS = {'.git','node_modules','__pycache__','site-packages','venv','.venv','cache','checkpoints','adapter'}


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def fingerprint(code):
    return hashlib.sha256(re.sub(r'\s+','',code).encode('utf-8')).hexdigest()


def canon(a,b): return tuple(sorted((str(a),str(b))))


def rel(p,root):
    try:return str(p.resolve().relative_to(root.resolve()))
    except ValueError:return str(p)


def valid_project(s):
    s=str(s or '').strip()
    return s if s.lower() not in {'','python','java','unknown','none','null','mixed','target'} else ''


def output_csv(path,cols,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)


def load_dataset(root,skip_hash=False):
    base=root/'data/training/clean/python_mixed'
    for filename,expected in APPROVED.items():
        path=base/filename
        if not path.is_file():raise FileNotFoundError(str(path))
        actual=sha(path)
        if actual!=expected:
            raise ValueError(f'APPROVED C-10 hash mismatch {filename}: expected={expected} actual={actual}')
    codes={};hash_ids=defaultdict(set)
    with (base/'data.jsonl').open(encoding='utf-8-sig') as f:
        for ln,line in enumerate(f,1):
            if not line.strip(): continue
            r=json.loads(line);i=str(r['idx']);s=r['func']
            if i in codes or not isinstance(s,str):raise ValueError(f'duplicate/invalid code {ln}')
            codes[i]=s;hash_ids[fingerprint(s)].add(i)
    pairs={};stats=Counter()
    for split,name in SPLITS:
        with (base/name).open(encoding='utf-8-sig') as f:
            for ln,line in enumerate(f,1):
                if not line.strip():continue
                a,b,y=line.split();key=canon(a,b)
                if a not in codes or b not in codes or key in pairs or y not in {'0','1'}:raise ValueError(f'invalid pair {name}:{ln}')
                pairs[key]={'split':split,'label':int(y),'id_a':key[0],'id_b':key[1]};stats[f'{split}_{y}']+=1
    return codes,hash_ids,pairs,dict(stats)


def source_paths(root,sub,max_files):
    base=root/sub;found=[]
    if not base.exists():return []
    for current,dirs,files in os.walk(base,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            p=Path(current)/name
            if p.is_symlink():continue
            if p.suffix.lower() not in {'.csv','.jsonl','.ndjson'}:continue
            if len(found)>=max_files:raise RuntimeError(f'source file limit {max_files} hit while scanning {base}; raise --max-files')
            found.append(p)
    return found


def rows_from_metadata(p,max_rows):
    if p.suffix.lower()=='.csv':
        with p.open(encoding='utf-8-sig',newline='') as f:
            r=csv.DictReader(f)
            for i,row in enumerate(r):
                if i>=max_rows:raise RuntimeError(f'row limit {max_rows} hit for {p}; use --max-rows-per-file 0 for full scan')
                yield row
    else:
        with p.open(encoding='utf-8-sig') as f:
            for i,line in enumerate(f):
                if i>=max_rows:raise RuntimeError(f'row limit {max_rows} hit for {p}')
                if line.strip():
                    obj=json.loads(line)
                    if isinstance(obj,dict): yield obj


def direct_c03(root,pairs,max_files,max_rows):
    matches=[];inventory=[]
    for p in source_paths(root,'results/c-03',max_files):
        n=0;hits=0
        try:
            for row in rows_from_metadata(p,max_rows):
                n+=1;fields=next(((a,b) for a,b in PAIR_FIELDS if a in row and b in row),None)
                if not fields:continue
                key=canon(row.get(fields[0],''),row.get(fields[1],''))
                if key not in pairs:continue
                project=next((valid_project(row.get(k)) for k in PROJECT_FIELDS if valid_project(row.get(k))), '')
                raw_type=str(row.get('clone_type') or '').strip()
                # Type is *not* verified merely because a CSV has a column.
                matches.append({'id_a':key[0],'id_b':key[1],'split':pairs[key]['split'],'label':pairs[key]['label'],
                                'project':project or 'UNKNOWN','detector_source':str(row.get('source') or row.get('detector') or ''),
                                'clone_type':'UNKNOWN','reported_type_unverified':raw_type,'source':rel(p,root),
                                'evidence_status':'CANDIDATE_EXACT_PAIR_ID','match_method':'EXACT_PAIR_ID'})
                hits+=1
        except (UnicodeError,csv.Error,json.JSONDecodeError) as exc:
            inventory.append({'path':rel(p,root),'rows_scanned':n,'matches':hits,'error':str(exc)});continue
        inventory.append({'path':rel(p,root),'rows_scanned':n,'matches':hits,'error':''})
    return matches,inventory


def discover_repos(root):
    base=root/'data/repositories'; names=defaultdict(list)
    if not base.is_dir():return names
    for p in sorted(base.iterdir()):
        if p.is_dir():names[p.name.lower()].append(p)
    return names


def possible_roots(repo_name,repository_dirs):
    # Project path in C-01 is <owner>--<repository>.
    slug=repo_name.split('--')[-1]
    candidates=repository_dirs.get(slug.lower(),[])+repository_dirs.get(repo_name.lower(),[])
    out=[]
    for p in candidates:
        for q in (p,p/slug):
            if q.is_dir() and q not in out:out.append(q)
    return out


def function_fingerprints(path):
    # Compile matched function AST source, with decorators, without mutating source.
    try:
        text=path.read_text(encoding='utf-8-sig',errors='replace')
        tree=ast.parse(text,filename=str(path))
    except (OSError,SyntaxError,ValueError,UnicodeError):return []
    lines=text.splitlines(keepends=True)
    out=[]
    for node in ast.walk(tree):
        if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
        start=min([node.lineno]+[d.lineno for d in node.decorator_list]);end=node.end_lineno
        if not end or end>len(lines):continue
        original=''.join(lines[start-1:end]); undecorated=''.join(lines[node.lineno-1:end])
        # ast.get_source_segment and dedent variants can bridge indentation wrappers.
        import textwrap
        variants={original,undecorated,textwrap.dedent(original),textwrap.dedent(undecorated)}
        out.append((start,end,node.name,{fingerprint(v) for v in variants}))
    return out


def match_endpoint(fnrows, start, end, name,hash_ids):
    try:s=int(float(start));e=int(float(end))
    except (TypeError,ValueError):return set()
    name=str(name or '').split('.')[-1]
    out=set()
    for a,b,n,hashes in fnrows:
        # C-01 function locations can report decorator start or function line.
        if not (a<=s<=b and a<=e<=b):continue
        if name and n!=name:continue
        for h in hashes:out.update(hash_ids.get(h,set()))
    return out


def c01_lineage(root,hash_ids,pairs,max_reports,max_rows):
    report_files=[]
    base=root/'results/c-01-batch'
    if base.exists():
        report_files=sorted(base.glob('*/runs/*/hybrid/hybrid_function_clone_pairs.csv'))
    # Also include explicit project-layout results from B-07, as distinct provenance sources.
    secondary=sorted((root/'results/b-07').glob('*/projects/*/hybrid/hybrid_function_clone_pairs.csv'))
    report_files+=secondary
    if len(report_files)>max_reports:raise RuntimeError(f'C-01/B-07 report limit {max_reports} < {len(report_files)}')
    repo_dirs=discover_repos(root)
    output=[];inventory=[];cache={}
    for report in report_files:
        parts=report.parts
        try:
            if 'c-01-batch' in parts:project=parts[parts.index('c-01-batch')+1]
            else:project=parts[parts.index('projects')+1]
        except (ValueError,IndexError):continue
        roots=possible_roots(project,repo_dirs)
        seen=set();scanned=0;matched=0;unresolved=0
        try:
            for row in rows_from_metadata(report,max_rows):
                scanned+=1
                pa=str(row.get('file_a') or '').strip();pb=str(row.get('file_b') or '').strip()
                if not pa or not pb:continue
                # Require safe relative path under a known repository root.
                if Path(pa).is_absolute() or Path(pb).is_absolute() or '..' in Path(pa).parts or '..' in Path(pb).parts:continue
                for repo in roots:
                    paths=[repo/pa,repo/pb]
                    if not all(x.is_file() and x.resolve().is_relative_to(repo.resolve()) for x in paths):continue
                    for f in paths:
                        if f not in cache:cache[f]=function_fingerprints(f)
                    ia=match_endpoint(cache[paths[0]],row.get('start_a'),row.get('end_a'),row.get('function_a'),hash_ids)
                    ib=match_endpoint(cache[paths[1]],row.get('start_b'),row.get('end_b'),row.get('function_b'),hash_ids)
                    valid={canon(a,b) for a in ia for b in ib if a!=b and canon(a,b) in pairs}
                    if len(valid)!=1:continue
                    key=next(iter(valid))
                    if pairs[key]['label']!=1:continue # detector clone evidence cannot verify negatives
                    identity=(key,project,str(row.get('source') or ''),rel(report,root))
                    if identity in seen:continue
                    seen.add(identity)
                    output.append({'id_a':key[0],'id_b':key[1],'split':pairs[key]['split'],'label':1,
                                   'project':project,'detector_source':str(row.get('source') or ''),
                                   'clone_type':'UNKNOWN','reported_type_unverified':'',
                                   'source':rel(report,root),'evidence_status':'CODE_AND_LOCATION_MATCH',
                                   'match_method':'EXACT_FUNCTION_CODE_PLUS_LOCATION','source_pair_id':str(row.get('pair_id') or '')})
                    matched+=1
                    break
                else:unresolved+=1
        except (OSError,UnicodeError,csv.Error) as e:
            inventory.append({'report':rel(report,root),'project':project,'repository_roots':[rel(x,root) for x in roots],
                              'rows_scanned':scanned,'code_matched_positive_pairs':matched,'unresolved':unresolved,'error':str(e)})
            continue
        inventory.append({'report':rel(report,root),'project':project,'repository_roots':[rel(x,root) for x in roots],
                          'rows_scanned':scanned,'code_matched_positive_pairs':matched,'unresolved':unresolved,'error':''})
    return output,inventory


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--max-files',type=int,default=2500)
    ap.add_argument('--max-reports',type=int,default=1000)
    ap.add_argument('--max-rows-per-file',type=int,default=0,help='0 means full scan')
    ap.add_argument('--skip-c01-code-match',action='store_true',help='Quick ID-only C-03 run')
    args=ap.parse_args()
    root=args.root.resolve();out=args.run_dir.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'nonempty run directory {out}')
    out.mkdir(parents=True,exist_ok=True)
    max_rows=args.max_rows_per_file if args.max_rows_per_file>0 else 10**12
    codes,hash_ids,pairs,stats=load_dataset(root)
    direct,dinv=direct_c03(root,pairs,args.max_files,max_rows)
    matched,rinv=([],[]) if args.skip_c01_code_match else c01_lineage(root,hash_ids,pairs,args.max_reports,max_rows)
    all_rows=direct+matched
    fields=['id_a','id_b','split','label','project','detector_source','clone_type','reported_type_unverified','source',
            'evidence_status','match_method','source_pair_id']
    all_rows.sort(key=lambda r:(r['id_a'],r['id_b'],r['evidence_status'],r['source']))
    output_csv(out/'python_pair_evidence.csv',fields,all_rows)
    output_csv(out/'c03_metadata_inventory.csv',['path','rows_scanned','matches','error'],dinv)
    rinv_rows=[dict(item,repository_roots=json.dumps(item['repository_roots'])) for item in rinv]
    output_csv(out/'c01_report_inventory.csv',['report','project','repository_roots','rows_scanned','code_matched_positive_pairs','unresolved','error'],rinv_rows)
    unique_direct={(r['id_a'],r['id_b']) for r in direct}
    unique_code={(r['id_a'],r['id_b']) for r in matched}
    project_by_pair=defaultdict(set)
    for r in all_rows:
        if r['project']!='UNKNOWN':project_by_pair[(r['id_a'],r['id_b'])].add(r['project'])
    contradictions={k:sorted(v) for k,v in project_by_pair.items() if len(v)>1}
    unique_projects=Counter()
    for key,projs in project_by_pair.items():
        if len(projs)==1:unique_projects[next(iter(projs))]+=1
    output_csv(out/'project_coverage.csv',['project','unique_linked_pairs'],
               [{'project':k,'unique_linked_pairs':n} for k,n in sorted(unique_projects.items())])
    summary={
        'task':'D-03c v2 lineage-based evidence recovery','timestamp_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
        'status':'EVIDENCE_RECOVERY_COMPLETE_NOT_TRAINING_APPROVAL',
        'python_dataset':'data/training/clean/python_mixed (APPROVED C-10)', 'python_sha256':APPROVED,
        'python_training_pairs':len(pairs),'python_function_records':len(codes),'python_split_counts':stats,
        'c03_files_scanned':len(dinv),'c03_exact_id_matches_unique':len(unique_direct),
        'c01_b07_reports_scanned':len(rinv),'code_and_location_matched_positive_pairs_unique':len(unique_code),
        'unique_evidence_pairs':len(unique_direct|unique_code),
        'unique_project_linked_pairs_no_conflict':len(project_by_pair)-len(contradictions),
        'project_conflict_pairs':len(contradictions),'project_coverage':dict(sorted(unique_projects.items())),
        'verified_clone_type_annotations':0,
        'java_target':'NOT_RECOVERED_IN_V2: requires separate A/B-series source mapping',
        'c01_code_match_enabled':not args.skip_c01_code_match,
        'limitations':['A detector-positive match is evidence of detector output, not an independently validated clone type.',
                       'NiCad blind normalization and AST span differences may make strict source-code fingerprints fail, lowering recall.',
                       'Any ambiguous function hash, absent original repository, or contradictory project evidence is excluded from confirmed provenance.',
                       'Neither exact function matching nor file location proves held-out project independence.',
                       'C-10 negative candidate correctness has not been fully manually adjudicated.',
                       'Existing training datasets, adapters and previous results remain unchanged.'],
        'output_sha256':{name:sha(out/name) for name in ('python_pair_evidence.csv','c03_metadata_inventory.csv','c01_report_inventory.csv','project_coverage.csv')}
    }
    (out/'lineage_recovery_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
    print(f'[D-03c v2] C-10 SHA256 VERIFIED pairs={len(pairs)} functions={len(codes)}')
    print(f'[D-03c v2] C-03 exact-ID pairs={len(unique_direct)} C-01/B-07 strict-code pairs={len(unique_code)}')
    print(f'[D-03c v2] project-linked={summary["unique_project_linked_pairs_no_conflict"]} project-conflicts={len(contradictions)} clone_types_confirmed=0')
    print(f'[D-03c v2] outputs={out} (NO TRAINING)')

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('[D-03c v2] ERROR: '+str(e),file=sys.stderr)
        raise
