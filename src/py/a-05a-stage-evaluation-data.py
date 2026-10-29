#!/usr/bin/env python3
"""Prepare portable clone-detection evaluation inputs."""
import argparse, csv, json, hashlib, os, time
from collections import defaultdict
from pathlib import Path

def emit_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def targets(root,language):
    sub='data-java' if language=='java' else 'data-py'
    d=root/language/'data'/sub/'ground_truth_test_refactorability'
    if not d.is_dir(): raise FileNotFoundError(d)
    jobs=[];wanted=set()
    for fp in sorted(d.glob('*.csv')):
        project=fp.name.removeprefix('pred_').removesuffix('_with_refactorability.csv')
        with fp.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
        if not rows: continue
        for row in rows:
            for k in ['pair_left_func','pair_right_func']:
                if not row.get(k): raise ValueError(f'Missing {k}: {fp}')
                wanted.add(row[k].strip())
        jobs.append((project,rows))
    return jobs,wanted

def gt(a,b):
    x=a.split('_');y=b.split('_')
    if len(x)<2 or len(y)<2: raise ValueError(f'Unparseable IDs: {a} {b}')
    return int(x[1]==y[1])

def candidates(rq3,language):
    base=rq3/language
    if not base.is_dir():raise FileNotFoundError(base)
    # Only actual code corpora, not massive model/prediction artifacts.
    results=[]
    for root in [base/'code'/'Dataset-Preparation',base/'data']:
        if root.exists(): results.extend(root.rglob('data.jsonl'))
    return sorted(set(results))

def build_code_map(paths,wanted):
    found={}; conflicts=set();scanned=0; usable=0
    idkeys=('idx','id','index','func_id','function_id')
    codekeys=('func','code','source','source_code','function','body')
    for path in paths:
        with path.open(encoding='utf-8',errors='replace') as f:
            for line in f:
                scanned+=1
                try:d=json.loads(line)
                except (ValueError,TypeError):continue
                if not isinstance(d,dict):continue
                ix=next((str(d[k]).strip() for k in idkeys if k in d and d[k] is not None),'')
                if ix not in wanted:continue
                code=next((d[k] for k in codekeys if isinstance(d.get(k),str) and d[k].strip()),None)
                if not code: continue
                usable+=1
                if ix in found and found[ix]!=code:conflicts.add(ix)
                else:found[ix]=code
    return found,conflicts,scanned,usable


import shutil

def copy_evaluation(args):
    base=args.workspace/'data'/'evaluation'
    stage=base.parent/'.evaluation-staging'
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True)
    summary={'languages':{}, 'source_provenance':{'rq3_root':str(args.rq3_root.resolve()),'rq4_root':str(args.rq4_root.resolve())}}
    for lang in ('java','python'):
        jobs,wanted=targets(args.rq4_root,lang)
        paths=candidates(args.rq3_root,lang)
        codes,conflicts,scanned,usable=build_code_map(paths,wanted)
        missing=wanted-set(codes)
        if missing or conflicts:
            raise RuntimeError(f'{lang}: missing={len(missing)} ambiguous={len(conflicts)}; refusing incomplete dataset')
        gt_dir=stage/'ground_truth'/lang
        gt_dir.mkdir(parents=True)
        sub='data-java' if lang=='java' else 'data-py'
        src=args.rq4_root/lang/'data'/sub/'ground_truth_test_refactorability'
        for file in sorted(src.glob('*.csv')):shutil.copy2(file,gt_dir/file.name)
        code_dir=stage/'code';code_dir.mkdir(exist_ok=True)
        with (code_dir/f'{lang}.jsonl').open('w',encoding='utf-8') as out:
            for idx in sorted(wanted):out.write(json.dumps({'id':idx,'code':codes[idx]},ensure_ascii=False)+'\n')
        summary['languages'][lang]={'projects':len(jobs),'code_ids':len(wanted),'code_corpora':len(paths),'source_lines_scanned':scanned,'pairs':sum(len(rows) for _,rows in jobs)}
        print(f'{lang}: projects={len(jobs)} code_ids={len(wanted)} pairs={summary["languages"][lang]["pairs"]}',flush=True)
    (stage/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    if base.exists():
        if not args.replace:raise FileExistsError(f'{base} exists; set A05A_REPLACE=1 explicitly')
        shutil.rmtree(base)
    stage.rename(base)
    ignore=args.workspace/'.gitignore'
    text=ignore.read_text() if ignore.exists() else ''
    if not any(line.strip() in ('/data/','data/') for line in text.splitlines()):
        with ignore.open('a') as f:f.write(('\n' if text and not text.endswith('\n') else '')+'\n# Local evaluation datasets\n/data/\n')
    print('Portable dataset:',base,'manifest:',base/'manifest.json',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--rq4-root',type=Path,default=Path.home()/'research_dream/2026JournalData/RQ4/c-evaluate_clone')
    ap.add_argument('--rq3-root',type=Path,default=Path.home()/'research_dream/2026JournalData/RQ3')
    ap.add_argument('--replace',action='store_true')
    args=ap.parse_args();copy_evaluation(args)
if __name__=='__main__':main()
