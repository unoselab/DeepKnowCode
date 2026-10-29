#!/usr/bin/env python3
"""A-05: strict code-ID alignment, then StarCoder2-7B label-likelihood inference.
Read-only source data. Never silently score unresolved or ambiguous IDs.
"""
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

def preflight(args):
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    tasks=[];reports=[];bylang={};maps={}
    for lang in ('java','python'):
        jobs,wanted=targets(args.rq4_root,lang)
        paths=candidates(args.rq3_root,lang)
        print(f'[A-05] {lang}: {len(jobs)} projects, {len(wanted)} unique IDs, {len(paths)} code corpora',flush=True)
        code_map,conflict,scanned,usable=build_code_map(paths,wanted)
        missing=wanted-code_map.keys();invalid=missing|conflict
        bylang[lang]={'projects':len(jobs),'requested_ids':len(wanted),'resolved_ids':len(wanted)-len(invalid),'missing_ids':len(missing),'ambiguous_ids':len(conflict),'jsonl_paths':len(paths),'lines_scanned':scanned,'matching_code_records':usable}
        for ix in sorted(invalid):reports.append({'language':lang,'id':ix,'reason':'ambiguous' if ix in conflict else 'missing'})
        maps[lang]=code_map
        for project,rows in jobs:
            for i,row in enumerate(rows):
                a=row['pair_left_func'].strip();b=row['pair_right_func'].strip()
                tasks.append({'language':lang,'project':project,'row':i,'left_id':a,'right_id':b,'label':gt(a,b),'code_a':code_map.get(a),'code_b':code_map.get(b)})
    emit_csv(out/'unresolved_ids.csv',reports,['language','id','reason'])
    (out/'preflight.json').write_text(json.dumps({'languages':bylang,'pairs':len(tasks)},indent=2)+'\n')
    if reports:
        print('[A-05] BLOCKED: missing/ambiguous code IDs. See',out/'unresolved_ids.csv',flush=True)
        raise SystemExit(3)
    # Preserve exact pair list and code in the local output, for stable worker input.
    p=out/'aligned_pairs.jsonl'
    with p.open('w',encoding='utf-8') as f:
        for t in tasks:f.write(json.dumps(t,ensure_ascii=False)+'\n')
    print('[A-05] Alignment complete:',len(tasks),'pairs;',p,flush=True)

def prompt(t):
    return ('Determine whether the following two code fragments are code clones. '
            'Answer YES if they are clones and NO otherwise.\n'
            f'Language: {t["language"]}\nCode A:\n{t["code_a"]}\n'
            f'Code B:\n{t["code_b"]}\nAnswer:')

def infer(args):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.manual_seed(2026)
    opts={'revision':args.revision,'local_files_only':True}
    tokenizer=AutoTokenizer.from_pretrained(args.model_id,**opts)
    model=AutoModelForCausalLM.from_pretrained(args.model_id,dtype=torch.float16,low_cpu_mem_usage=True,device_map={'':0},**opts).eval()
    # Explicit one-token YES/NO choices; abort rather than compare inconsistent multi-token sequences.
    choices={}
    for k in ('YES','NO'):
        ids=tokenizer.encode(' '+k,add_special_tokens=False)
        if len(ids)!=1:raise RuntimeError(f'Label {k} is not single token: {ids}; revise scoring policy')
        choices[k]=ids[0]
    inp=args.output_dir/'aligned_pairs.jsonl'
    dest=args.output_dir/f'predictions-worker-{args.worker_index}.csv'
    fields=['language','project','row','left_id','right_id','true_label','predicted_label','yes_logit','no_logit','worker','input_tokens','truncated']
    print('[A-05] worker',args.worker_index,'labels',choices,flush=True)
    with inp.open(encoding='utf-8') as f, dest.open('w',newline='',encoding='utf-8') as o:
        w=csv.DictWriter(o,fieldnames=fields);w.writeheader()
        count=0;truncated_count=0
        for n,line in enumerate(f):
            if n%args.num_workers!=args.worker_index:continue
            t=json.loads(line)
            # Avoid truncating code silently; abort on oversized pair for transparent reporting.
            encoded=tokenizer(prompt(t),add_special_tokens=False)
            length=len(encoded['input_ids'])
            if length>args.max_input_tokens:
                truncated_count+=1
                if args.oversize=='error':raise RuntimeError(f'Input too long ({length} tokens): {t["language"]}/{t["project"]} row {t["row"]}')
                # Keeping the full prompt prefix and explicit truncation marker makes this non-comparable;
                # recorded separately so skipped pairs never silently enter F1.
                if args.oversize=='skip':
                    w.writerow(dict(language=t['language'],project=t['project'],row=t['row'],left_id=t['left_id'],right_id=t['right_id'],true_label=t['label'],worker=args.worker_index,input_tokens=length,truncated='skipped'))
                    continue
            ids=torch.tensor([encoded['input_ids']],device='cuda:0')
            with torch.inference_mode():logits=model(input_ids=ids).logits[0,-1]
            yes=float(logits[choices['YES']].item());no=float(logits[choices['NO']].item())
            pred=int(yes>no)
            w.writerow(dict(language=t['language'],project=t['project'],row=t['row'],left_id=t['left_id'],right_id=t['right_id'],true_label=t['label'],predicted_label=pred,yes_logit=yes,no_logit=no,worker=args.worker_index,input_tokens=length,truncated='false'))
            count+=1
            if count%args.progress_every==0:
                o.flush();print(f'[A-05] worker {args.worker_index}: {count} pairs',flush=True)
    print(f'[A-05] worker {args.worker_index} complete: evaluated={count}, oversized_skipped={truncated_count}; {dest}',flush=True)

def summarize(args):
    expected=sum(1 for _ in (args.output_dir/'aligned_pairs.jsonl').open())
    rows=[];keys=set()
    for i in range(args.num_workers):
        path=args.output_dir/f'predictions-worker-{i}.csv'
        with path.open(newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                key=(r['language'],r['project'],r['row'])
                if key in keys:raise RuntimeError(f'Duplicate pair {key}')
                keys.add(key);rows.append(r)
    if len(rows)!=expected:raise RuntimeError(f'Expected {expected} pair rows, got {len(rows)}')
    grouped=defaultdict(list)
    for r in rows:grouped[(r['language'],r['project'])].append(r)
    stats=[];incomplete=0
    for (lang,project),items in sorted(grouped.items()):
        valid=[r for r in items if r['predicted_label'] in ('0','1')]
        if len(valid)!=len(items):incomplete+=1
        tp=sum(r['true_label']=='1' and r['predicted_label']=='1' for r in valid)
        fp=sum(r['true_label']=='0' and r['predicted_label']=='1' for r in valid)
        fn=sum(r['true_label']=='1' and r['predicted_label']=='0' for r in valid)
        tn=len(valid)-tp-fp-fn
        precision=tp/(tp+fp) if tp+fp else 0
        recall=tp/(tp+fn) if tp+fn else 0
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
        stats.append(dict(language=lang,project=project,n_expected=len(items),n_scored=len(valid),tp=tp,fp=fp,fn=fn,tn=tn,precision=precision,recall=recall,f1=f1,complete=len(items)==len(valid)))
    emit_csv(args.output_dir/'project_metrics.csv',stats,['language','project','n_expected','n_scored','tp','fp','fn','tn','precision','recall','f1','complete'])
    status='COMPLETE' if not incomplete else 'INCOMPLETE_DO_NOT_COMPARE'
    (args.output_dir/'summary.json').write_text(json.dumps({'status':status,'expected_pairs':expected,'scored_pairs':sum(int(x['n_scored']) for x in stats),'projects':len(stats),'incomplete_projects':incomplete,'model':args.model_id,'revision':args.revision},indent=2)+'\n')
    print('[A-05]',status,'projects=',len(stats),'expected_pairs=',expected,flush=True)
    if incomplete:raise SystemExit(4)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['preflight','infer','summarize'])
    ap.add_argument('--rq4-root',type=Path,default=Path.home()/'research_dream/2026JournalData/RQ4/c-evaluate_clone')
    ap.add_argument('--rq3-root',type=Path,default=Path.home()/'research_dream/2026JournalData/RQ3')
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--model-id',default='bigcode/starcoder2-7b')
    ap.add_argument('--revision',default='bb9afde76d7945da5745592525db122d4d729eb1')
    ap.add_argument('--worker-index',type=int,default=0)
    ap.add_argument('--num-workers',type=int,default=3)
    ap.add_argument('--max-input-tokens',type=int,default=4096)
    ap.add_argument('--oversize',choices=['error','skip'],default='error')
    ap.add_argument('--progress-every',type=int,default=100)
    args=ap.parse_args()
    if args.command=='preflight':preflight(args)
    elif args.command=='infer':infer(args)
    else:summarize(args)
if __name__=='__main__':main()
