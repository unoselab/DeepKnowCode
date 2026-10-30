#!/usr/bin/env python3
"""Audit staged code-pair corpus provenance and similarity to held-out code."""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def normhash(s):
    return hashlib.sha256(re.sub(r'\s+', '', s).encode('utf-8')).hexdigest()


def tokens(s):
    # Language-agnostic lexical approximation, NOT semantic equivalence.
    return re.findall(r'[A-Za-z_$][\w$]*|\d+|==|!=|<=|>=|=>|->|\S', s)


def fingerprint(ts):
    # token 3-grams; identifiers retained to avoid exaggerating similarity.
    sh = ts if len(ts) < 3 else ['\x1f'.join(ts[i:i+3]) for i in range(len(ts)-2)]
    features = set(sh)
    if not features:
        return 0
    scores = [0] * 64
    for feat in features:
        v = int.from_bytes(hashlib.blake2b(feat.encode('utf-8'), digest_size=8).digest(), 'big')
        for j in range(64):
            scores[j] += 1 if (v >> j) & 1 else -1
    return sum((1 << j) for j, x in enumerate(scores) if x > 0)


def detect_lang(code):
    java = len(re.findall(r'\b(public|private|protected|static|void|implements|throws|new)\b|\bclass\s+\w+\s*\{', code))
    py = len(re.findall(r'(?m)^\s*(def|async def|from\s+\w+\s+import|import\s+\w+|elif|except)\b|\bself\b', code))
    if java >= 2 and java > py * 2:
        return 'likely_java'
    if py >= 2 and py > java * 2:
        return 'likely_python'
    return 'unknown'


def read_code(path, id_field, code_field):
    with path.open(encoding='utf-8-sig') as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            yield str(obj[id_field]), obj[code_field], {k:obj[k] for k in ('repo','repository','project','path','file','source_file','url','origin') if k in obj}


def read_pair(path):
    with path.open(encoding='utf-8-sig') as f:
        for line in f:
            p = line.split()
            if len(p) != 3 or p[2] not in ('0','1'):
                raise ValueError('Unexpected pair row in '+str(path))
            yield p[0],p[1],int(p[2])


def summary_csv(path, rows, fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    root=Path(__file__).resolve().parents[2]
    p.add_argument('--training',type=Path,default=root/'data/training')
    p.add_argument('--evaluation',type=Path,default=root/'data/evaluation')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--near-threshold',type=float,default=0.90)
    p.add_argument('--max-bucket-size',type=int,default=200)
    p.add_argument('--max-candidates-per-test',type=int,default=1500)
    p.add_argument('--max-matches-report',type=int,default=5000)
    a=p.parse_args()
    if not 0 < a.near_threshold <= 1: p.error('near-threshold out of range')
    a.out.mkdir(parents=True,exist_ok=True)
    manifest=a.training/'manifest.json'
    if not manifest.is_file(): raise FileNotFoundError(manifest)
    m=json.loads(manifest.read_text())
    corpora=('java_source','java_mixed','python_mixed')
    datasets={}; code_map={}; pair_data={}; sources={}; ids_global=defaultdict(set)
    for name in corpora:
        base=a.training/'clean'/name
        code_file=base/'data.jsonl'
        codes={}; meta={}
        for idx,code,extra in read_code(code_file,'idx','func'):
            codes[idx]=code
            if extra: meta[idx]=extra
        pairs={split:list(read_pair(base/fn)) for split,fn in (('train','train.txt'),('validation','valid.txt'))}
        ref={split:set(x for aa,bb,_ in rows for x in (aa,bb)) for split,rows in pairs.items()}
        groups=Counter(detect_lang(c) for c in codes.values())
        datasets[name]={'records':len(codes),'labels':{sp:dict(Counter(x[2] for x in rows)) for sp,rows in pairs.items()},'referenced_ids':{sp:len(v) for sp,v in ref.items()},'language_heuristic':dict(groups),'ids_with_provenance_metadata':len(meta),'pairs':{sp:len(rows) for sp,rows in pairs.items()}}
        for idx, code in codes.items():
            ids_global[name].add(idx)
            # Restrict index to training; validation still audited for function reuse below.
            if idx in ref['train']:
                h=normhash(code)
                code_map.setdefault(h, {'text':code,'owners':set(),'ids':[],'length':len(tokens(code))})
                code_map[h]['owners'].add(name)
                if len(code_map[h]['ids']) < 4:code_map[h]['ids'].append(f'{name}:{idx}')
        pair_data[name]=(pairs,ref,codes,meta)
        print(f'[A-09] {name}: functions={len(codes)} train_pairs={len(pairs["train"])} validation_pairs={len(pairs["validation"])} metadata={len(meta)}',flush=True)
    # provenance cannot be conjured from integer IDs or source-only code.
    source_test=[];test_map={};test_counts={}
    for lang in ('java','python'):
        codes={}
        for idx, code, extra in read_code(a.evaluation/'code'/f'{lang}.jsonl','id','code'):
            codes[idx]=code
        test_counts[lang]=len(codes)
        for idx,code in codes.items():
            h=normhash(code)
            test_map.setdefault((lang,h),{'lang':lang,'hash':h,'text':code,'ids':[],'length':len(tokens(code))})['ids'].append(idx)
    # independent function reuse across train/validation within each corpus
    shared={}
    for name,(pairs,ref,codes,meta) in pair_data.items():
        ht={normhash(codes[k]) for k in ref['train']};hv={normhash(codes[k]) for k in ref['validation']}
        shared[name]=len(ht & hv)
    across=set();valid=set()
    for name,(pairs,ref,codes,meta) in pair_data.items():
        across.update(normhash(codes[k]) for k in ref['train'])
        valid.update(normhash(codes[k]) for k in ref['validation'])
    # Simhash LSH; multiple bands to generate candidate keys, followed by exact token set Jaccard.
    print(f'[A-09] Tokenizing {len(code_map)} unique training functions and building near-duplicate index...',flush=True)
    index=defaultdict(list); train=[]; too_short=0
    for h,item in code_map.items():
        tt=tokens(item['text'])
        if len(tt)<8:
            too_short+=1; continue
        f=fingerprint(tt)
        rid=len(train);train.append((h,item['length'],f,set(tt),item))
        for b in range(4):index[(16,b,(f>>(16*b))&65535)].append(rid)
        for b in range(8):index[(8,b,(f>>(8*b))&255)].append(rid)
    crowded=sum(1 for v in index.values() if len(v)>a.max_bucket_size)
    matches=[];exact=0; exceeded=0; skipped_buckets=0; comparisons=0; unmatched_provenance=len(code_map)
    for n,entry in enumerate(test_map.values(),1):
        tt=tokens(entry['text']);h=entry['hash']
        if h in code_map:
            exact+=1; continue
        if len(tt)<8:continue
        f=fingerprint(tt); candidates=set()
        for band in (16,8):
            for b in range(64//band):
                bucket=index.get((band,b,(f>>(band*b))&((1<<band)-1)),())
                if len(bucket)>a.max_bucket_size:
                    skipped_buckets+=1;continue
                candidates.update(bucket)
        if len(candidates)>a.max_candidates_per_test:
            exceeded+=1;continue
        ts=set(tt)
        for rid in candidates:
            th,tl,tf,tset,item=train[rid]
            if min(tl,len(tt))/max(tl,len(tt))<0.65:continue
            if (f^tf).bit_count()>12:continue
            comparisons+=1
            jac=len(ts&tset)/max(1,len(ts|tset))
            if jac>=a.near_threshold:
                if len(matches)<a.max_matches_report:
                    matches.append({'test_language':entry['lang'],'test_id':entry['ids'][0],'test_code_hash':h,'training_code_hash':th,'training_corpora':';'.join(sorted(item['owners'])),'training_id_examples':';'.join(item['ids']),'token_set_jaccard':f'{jac:.5f}','test_tokens':len(tt),'train_tokens':tl,'simhash_hamming':(f^tf).bit_count()})
        if n%5000==0:print(f'[A-09] Checked {n}/{len(test_map)} distinct test code entries',flush=True)
    summary_csv(a.out/'near_duplicate_candidates.csv',matches,['test_language','test_id','test_code_hash','training_code_hash','training_corpora','training_id_examples','token_set_jaccard','test_tokens','train_tokens','simhash_hamming'])
    # Report per language under corpus but avoid claiming certain identification.
    report={'status':'REVIEW_REQUIRED','training_manifest_status':m.get('status'),'methodology':{'fingerprint':'64-bit SimHash of lexical token 3-grams; 4x16 and 8x8 locality-sensitive buckets','verification':'lexical token-set Jaccard','near_threshold':a.near_threshold,'bucket_size_cap':a.max_bucket_size,'candidate_cap':a.max_candidates_per_test,'language_heuristic':'Java/Python heuristic only; uncertain cases unknown','limitations':['Approximate candidate search is NOT exhaustive; near_duplicate_candidates.csv is a shortlist, not a proof of absence.','Matching is language-agnostic and lexical, NOT a semantic clone check.','No project provenance can be inferred from code/IDs when metadata is absent.','Model memorization/pretraining exposure cannot be audited with these artifacts.','Separate project-level split requires repository maps or raw extraction manifests.']},'corpora':datasets,'test_functions':test_counts,'unique_training_code_hashes':len(code_map),'unique_test_code_hashes':len(test_map),'exact_train_test_overlap_unique_test_hashes':exact,'near_duplicate_candidate_rows_written':len(matches),'near_duplicate_rows_capped':len(matches)>=a.max_matches_report,'test_queries_exceeding_candidate_cap':exceeded,'skipped_crowded_bucket_lookups':skipped_buckets,'crowded_index_buckets':crowded,'function_reuse_train_validation':{'per_corpus':shared,'global':len(across&valid)},'near_candidate_comparisons':comparisons,'provenance':{'project_level_isolation_verified':False,'unresolved_unique_training_code_hashes':unmatched_provenance,'required_evidence':['Function-ID-to-repository mapping or original extraction manifests','Repository identifiers for each train/validation pair','Stable project identifier for every held-out evaluation pair']},'training_approval':'NOT_GRANTED_BY_A09'}
    (a.out/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('[A-09] Near-duplicate candidate rows:',len(matches),'exact overlaps:',exact,flush=True)
    print('[A-09] Project provenance NOT VERIFIED; training approval pending.',flush=True)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:
        print(f'[A-09] ERROR: {e}',flush=True)
        raise
