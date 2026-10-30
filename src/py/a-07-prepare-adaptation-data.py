#!/usr/bin/env python3
"""Prepare and audit paired code-classification datasets."""
import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

PAIR_KEYS = [('left_id','right_id'),('pair_left_func','pair_right_func'),('func1','func2'),('code1','code2'),('code_a','code_b'),('left','right')]
LABEL_KEYS = ('label','is_clone','clone','ground_truth','gt','target')
CODE_KEYS = ('func','code','source','source_code','function','body')
ID_KEYS = ('idx','id','index','func_id','function_id')


def digest(code):
    return hashlib.sha256(re.sub(r'\s+', '', code).encode('utf-8')).hexdigest()


def read_jsonl(path):
    with path.open(encoding='utf-8-sig') as handle:
        for number,line in enumerate(handle,1):
            if line.strip():
                try: obj=json.loads(line)
                except json.JSONDecodeError as exc: raise ValueError(f'{path}:{number}: {exc}')
                if not isinstance(obj,dict): raise ValueError(f'{path}:{number}: expected object')
                yield obj


def rows_from(path):
    if path.suffix.lower()=='.jsonl':
        yield from read_jsonl(path)
    elif path.suffix.lower()=='.csv':
        with path.open(encoding='utf-8-sig',newline='') as handle:
            yield from csv.DictReader(handle)
    else: raise ValueError(f'Expected CSV/JSONL: {path}')


def key_pair(row):
    return next(((a,b) for a,b in PAIR_KEYS if a in row and b in row),None)


def identify(row):
    return next((str(row[k]).strip() for k in ID_KEYS if k in row and row[k] is not None),'')


def code_of(row):
    return next((row[k] for k in CODE_KEYS if isinstance(row.get(k),str) and row[k].strip()),None)


def label_of(row):
    for k in LABEL_KEYS:
        if k in row:
            value=str(row[k]).strip().lower()
            if value in ('1','true','yes','clone'): return 1
            if value in ('0','false','no','nonclone','non-clone'): return 0
            raise ValueError(f'Unexpected explicit label {row[k]!r} in field {k}; expected binary 0/1')
    raise ValueError('No explicit ground-truth label column; refuse to infer from predicted labels or ID patterns')


def evaluation_index(root):
    index={}; summary={}
    for lang in ('java','python'):
        code_file=root/'code'/f'{lang}.jsonl'
        gt_dir=root/'ground_truth'/lang
        if not code_file.is_file() or not gt_dir.is_dir():raise FileNotFoundError(f'Missing held-out evaluation data for {lang}: {root}')
        codes={}; hashes=set()
        for row in read_jsonl(code_file):
            idx=str(row['id']).strip(); code=row['code']
            if idx in codes:raise ValueError(f'Duplicate evaluation ID {lang}/{idx}')
            codes[idx]=code; hashes.add(digest(code))
        pairs=set();count=0
        files=sorted(gt_dir.glob('*.csv'))
        if not files:raise ValueError(f'No held-out GT CSV for {lang}')
        for file in files:
            for row in rows_from(file):
                a=row['pair_left_func'].strip(); b=row['pair_right_func'].strip()
                if a not in codes or b not in codes:raise ValueError(f'Evaluation code not resolved: {lang}/{file}')
                pairs.add(tuple(sorted((a,b)))); count+=1
        index[lang]={'ids':set(codes),'hashes':hashes,'pairs':pairs}
        summary[lang]={'heldout_pairs':count,'heldout_ids':len(codes),'heldout_code_hashes':len(hashes)}
    return index,summary


def inventory(args,out):
    roots=[p for p in args.search_root if p.exists()]
    found=[]
    for root in roots:
        for file in root.rglob('*'):
            if not file.is_file() or file.suffix.lower() not in ('.csv','.jsonl'): continue
            if any(part in ('results','.git','__pycache__','checkpoints','models') for part in file.parts):continue
            size=file.stat().st_size
            if size>args.max_inventory_bytes: continue
            try:
                it=iter(rows_from(file)); first=next(it,None)
                if first is None:continue
                columns=sorted(first.keys()); kind='pair' if key_pair(first) else 'code' if identify(first) and code_of(first) else 'unknown'
                if kind=='unknown':continue
                found.append({'path':str(file.resolve()),'size_bytes':size,'kind':kind,'columns':','.join(columns),'has_explicit_label':any(k in first for k in LABEL_KEYS)})
            except (ValueError,UnicodeError,OSError,csv.Error):continue
    found.sort(key=lambda x:x['path'])
    write_csv(out/'candidate_inventory.csv',found,['path','size_bytes','kind','columns','has_explicit_label'])
    (out/'inventory.json').write_text(json.dumps({'search_roots':[str(x) for x in roots],'candidates':len(found),'note':'Inventory is informational, not approval of train/test provenance.'},indent=2)+'\n')
    print(f'[A-07] Inventory: {len(found)} candidate pair/code files. See {out/"candidate_inventory.csv"}',flush=True)


def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def load_code_maps(files):
    maps={x:{} for x in ('java','python')}
    for lang,path in files:
        for row in rows_from(path):
            idx=identify(row); code=code_of(row)
            if not idx or not code: raise ValueError(f'Code corpus {path}: missing code or id')
            old=maps[lang].setdefault(idx,code)
            if old !=code:raise ValueError(f'Conflicting code for {lang}/{idx} in {path}')
    return maps


def stage(args,out,heldout):
    if not args.train_pairs: raise ValueError('Provide --train-pairs LANGUAGE:FILE explicitly; inventory alone does not authorize staging')
    if not args.code_corpus: raise ValueError('Provide --code-corpus LANGUAGE:FILE explicitly')
    code_maps=load_code_maps(args.code_corpus)
    pairs=[]; seen={x:set() for x in ('java','python')}; rejected=[]; problems=[]
    for lang,source in args.train_pairs:
        if not source.is_file():raise FileNotFoundError(source)
        for row_number,row in enumerate(rows_from(source),1):
            kp=key_pair(row)
            if not kp:raise ValueError(f'{source}:{row_number}: no recognized pair columns')
            a,b=(str(row[k]).strip() for k in kp)
            label=label_of(row)
            if not a or not b:raise ValueError(f'{source}:{row_number}: blank pair ID')
            ca=code_maps[lang].get(a);cb=code_maps[lang].get(b)
            if ca is None or cb is None:
                problems.append({'language':lang,'source':str(source),'row':row_number,'reason':'unresolved_code_id','left_id':a,'right_id':b});continue
            pair_key=tuple(sorted((a,b)))
            if pair_key in seen[lang]:
                problems.append({'language':lang,'source':str(source),'row':row_number,'reason':'duplicate_training_pair','left_id':a,'right_id':b});continue
            seen[lang].add(pair_key)
            reason=[]
            if a in heldout[lang]['ids'] or b in heldout[lang]['ids']:reason.append('heldout_id')
            if digest(ca) in heldout[lang]['hashes'] or digest(cb) in heldout[lang]['hashes']:reason.append('heldout_exact_code')
            if pair_key in heldout[lang]['pairs']:reason.append('heldout_pair')
            if reason:
                rejected.append({'language':lang,'source':str(source),'row':row_number,'reason':'+'.join(reason),'left_id':a,'right_id':b});continue
            pairs.append({'language':lang,'left_id':a,'right_id':b,'code_a':ca,'code_b':cb,'label':label,'source':str(source),'source_row':row_number,'code_a_sha256':digest(ca),'code_b_sha256':digest(cb)})
    write_csv(out/'excluded_heldout.csv',rejected,['language','source','row','reason','left_id','right_id'])
    write_csv(out/'invalid_training_rows.csv',problems,['language','source','row','reason','left_id','right_id'])
    stats={'candidate_unique_pairs':len(pairs)+len(rejected),'accepted_pairs':len(pairs),'excluded_heldout':len(rejected),'invalid_rows':len(problems),'labels':dict(Counter(str(r['label']) for r in pairs)),'languages':dict(Counter(r['language'] for r in pairs)),'status':'BLOCKED'}
    (out/'stage_audit.json').write_text(json.dumps(stats,indent=2)+'\n')
    if problems:raise RuntimeError(f'{len(problems)} unresolved/duplicate rows; no training dataset written. See invalid_training_rows.csv')
    if not pairs:raise RuntimeError('No leakage-free training pairs found; no training dataset written')
    if not args.confirm_provenance:raise RuntimeError('Training candidates audited, but provenance unconfirmed. Set --confirm-provenance after verifying train/validation split and original source.')
    # A conservative preparation step: model training is intentionally not started here.
    dest=args.workspace/'data'/'training'
    if dest.exists():raise FileExistsError(f'{dest} already exists; refuse overwrite')
    dest.mkdir(parents=True)
    pairs.sort(key=lambda x:(x['language'],x['left_id'],x['right_id']))
    # Deterministic hash split; still requires project/group leakage review before GPU training.
    for split in ('train','validation'):
        with (dest/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
            for row in pairs:
                key=f'{row["language"]}:{row["left_id"]}:{row["right_id"]}'
                is_val=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%10==0
                if (split=='validation')==is_val:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    stats['status']='STAGED_REQUIRES_GROUP_LEAKAGE_REVIEW'
    (dest/'manifest.json').write_text(json.dumps(stats,indent=2)+'\n')
    (out/'stage_audit.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(f'[A-07] Staged {len(pairs)} candidate pairs in {dest}. NOT yet approved for training: group/near-duplicate leakage review required.',flush=True)


def language_path(value):
    lang,sep,path=value.partition(':')
    if not sep or lang not in ('java','python') or not path:raise argparse.ArgumentTypeError('Expected java:/path/to/file or python:/path/to/file')
    return lang,Path(path).expanduser()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--mode',choices=('inventory','stage'),default='inventory')
    ap.add_argument('--search-root',type=Path,action='append',default=[])
    ap.add_argument('--train-pairs',type=language_path,action='append',default=[])
    ap.add_argument('--code-corpus',type=language_path,action='append',default=[])
    ap.add_argument('--confirm-provenance',action='store_true')
    ap.add_argument('--max-inventory-bytes',type=int,default=2_000_000_000)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    index,summary=evaluation_index(args.workspace/'data'/'evaluation')
    (args.output_dir/'heldout_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('[A-07] Held-out evaluation index verified:',summary,flush=True)
    if args.mode=='inventory':inventory(args,args.output_dir)
    else:stage(args,args.output_dir,index)

if __name__=='__main__':main()
