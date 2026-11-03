#!/usr/bin/env python3
"""C-03 v1: read-only C-01 hybrid extraction into A-08-compatible training candidates."""
import argparse
import ast
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import subprocess

VERSION = 'v1'
FIELDS = ('file_a', 'start_a', 'end_a', 'function_a', 'file_b', 'start_b', 'end_b', 'function_b', 'source')

def digest_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def normalized_hash(s):
    return digest_text(re.sub(r'\s+', '', s))

def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()

def within(path, root):
    return path == root or root in path.parents

def read_test_hashes(root, required=True):
    out=set(); detail={}
    for lang in ('java','python'):
        file=root/'code'/f'{lang}.jsonl'
        if not file.is_file():
            if required: raise ValueError(f'Missing held-out test code: {file}')
            detail[lang]={'status':'NOT_FOUND'}
            continue
        n=0
        with file.open(encoding='utf-8-sig') as f:
            for line in f:
                if not line.strip():continue
                d=json.loads(line)
                code=d['code']
                if not isinstance(code,str): raise ValueError(f'Invalid test code at {file}')
                out.add(digest_text(code)); out.add(normalized_hash(code)); n+=1
        detail[lang]={'functions':n,'sha256':sha_file(file)}
    return out,detail

def get_function(lines, file, start, end):
    if not (1 <= start <= end <= len(lines)): raise ValueError(f'Line span outside file: {file}:{start}-{end}')
    code=''.join(lines[start-1:end])
    if not code.strip(): raise ValueError('Empty code span')
    return code

def select_runs(root, projects=None):
    base=root/'results/c-01-batch'
    found=[]; seen=set()
    for repo_dir in sorted(base.iterdir() if base.is_dir() else []):
        if not repo_dir.is_dir() or '--' not in repo_dir.name: continue
        for folder in sorted((repo_dir/'runs').glob('*'), reverse=True):
            meta=folder/'project_run.json'
            pairs=folder/'hybrid/hybrid_function_clone_pairs.csv'
            if not meta.is_file() or not pairs.is_file():continue
            obj=json.loads(meta.read_text(encoding='utf-8'))
            project=obj.get('project')
            if projects and project not in projects:continue
            if project in seen: continue
            if obj.get('status')!='COMPLETED':continue
            found.append((folder,obj,repo_dir.name));seen.add(project)
    if projects and set(projects)!=seen: raise ValueError('Missing C-01 runs: '+str(sorted(set(projects)-seen)))
    if not found: raise ValueError('No completed organized C-01 hybrid runs')
    return found

def read_repo(root, meta, check_commit=True):
    project=meta['project']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',project):raise ValueError('Invalid project name')
    repo=(root/'data/repositories'/project).resolve()
    if not within(repo,(root/'data/repositories').resolve()):raise ValueError('Unsafe repo')
    if not (repo/'.git').exists(): raise ValueError(f'Repository unavailable: {repo}')
    if check_commit:
        head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if head!=meta['commit']:raise ValueError(f'Commit mismatch for {project}: {head} != {meta["commit"]}')
    return repo

def build(args):
    root=args.root.resolve()
    output=(args.output if args.output.is_absolute() else root/args.output).resolve()
    if output.exists() or output.is_symlink():raise ValueError('Output already exists: '+str(output))
    if not within(output,(root/'data/training').resolve()):raise ValueError('Output must be inside data/training')
    old=(root/'data/training/clean/python_mixed_legacy').resolve()
    if within(output,old) or output==old:raise ValueError('Legacy data is protected')
    if args.negative_ratio < 0: raise ValueError('negative-ratio must be nonnegative')
    runs=select_runs(root,set(args.projects) if args.projects else None)
    tests,test_details=read_test_hashes(root/'data/evaluation',not args.allow_missing_test)
    ids={}; corpus={}; codes_hash={}; positive=set(); provenance=[]; audits=Counter(); project_info=[]
    for folder,meta,label in runs:
        repo=read_repo(root,meta)
        cache={}; record_count=0
        with (folder/'hybrid/hybrid_function_clone_pairs.csv').open(newline='',encoding='utf-8-sig') as f:
            reader=csv.DictReader(f)
            missing=set(FIELDS)-set(reader.fieldnames or ())
            if missing:raise ValueError(f'Missing columns {missing}: {folder}')
            for row in reader:
                record_count+=1; end_ids=[]; excluded=False
                for s in ('a','b'):
                    rel=Path(row['file_'+s]); raw=repo/rel
                    if rel.is_absolute() or '..' in rel.parts or not within(raw.resolve(),repo):raise ValueError('Unsafe source path: '+str(rel))
                    if raw not in cache:
                        if not raw.is_file() or raw.is_symlink():raise ValueError('Missing/unsafe source file: '+str(raw))
                        cache[raw]=raw.read_text(encoding='utf-8-sig').splitlines(keepends=True)
                    start,end=int(row['start_'+s]),int(row['end_'+s])
                    code=get_function(cache[raw],raw,start,end)
                    key=(meta['project'],rel.as_posix(),start,end)
                    h=digest_text(code)
                    if h in tests or normalized_hash(code) in tests: excluded=True
                    if key not in ids:
                        uid='py_'+digest_text('|'.join(map(str,key)))[:24]
                        if uid in corpus:raise ValueError('ID collision')
                        ids[key]=uid;corpus[uid]=code;codes_hash[uid]=h
                    end_ids.append(ids[key])
                if excluded:
                    audits['heldout_exact_code_pair_removed']+=1
                    continue
                a,b=end_ids
                if a==b or codes_hash[a]==codes_hash[b]:
                    audits['self_or_same_code_pair_removed']+=1;continue
                pair=tuple(sorted((a,b)))
                if pair in positive: audits['duplicate_positive_removed']+=1;continue
                positive.add(pair)
                provenance.append({'project':meta['project'],'github_repository':label,'run_id':folder.name,
                                   'pair_id':row.get('pair_id',''),'source':row['source'], 'id_a':pair[0],'id_b':pair[1],
                                   'file_a':row['file_a'],'file_b':row['file_b']})
        project_info.append({'project':meta['project'],'github_repository':label,'commit':meta['commit'],
                             'c01_run':folder.name,'pairs_seen':record_count,
                             'pair_csv_sha256':sha_file(folder/'hybrid/hybrid_function_clone_pairs.csv')})
    if not positive:raise ValueError('No usable positive pairs')
    # Dedup by exact unordered CODE identity, not only function-location ID.
    unique={}; removed=0
    for a,b in sorted(positive):
        k=tuple(sorted((codes_hash[a],codes_hash[b])))
        if k not in unique:unique[k]=(a,b)
        else:removed+=1
    positives=list(unique.values());audits['same_code_hash_pair_deduplicated']=removed
    rng=random.Random(args.seed)
    rng.shuffle(positives)
    cut=round(len(positives)*args.validation_fraction)
    if len(positives)>1:cut=max(1,min(len(positives)-1,cut))
    elif cut:raise ValueError('Not enough positives for validation')
    # Enforce function-disjoint split: any pair touching a validation endpoint is assigned
    # to validation; do not silently leak shared functions to both partitions.
    valid_ids=set(x for p in positives[:cut] for x in p)
    train_pos=[];val_pos=[]
    for p in positives:
        if p[0] in valid_ids or p[1] in valid_ids:val_pos.append(p)
        else:train_pos.append(p)
    # Negative candidates are heuristic and ALWAYS flagged as unverified.
    # We permit opt-in only; a missing clone edge is not ground-truth non-clone.
    def negative_candidates(items):
        if not args.sample_negative_candidates or args.negative_ratio==0:return []
        nodes=sorted({n for pair in items for n in pair});used=set(items);negs=set()
        target=min(int(len(items)*args.negative_ratio),len(nodes)*(len(nodes)-1)//2-len(used))
        for _ in range(max(1000,target*50)):
            if len(negs)>=target:break
            if len(nodes)<2:break
            a,b=rng.sample(nodes,2);pair=tuple(sorted((a,b)))
            if pair in used or pair in negs or codes_hash[a]==codes_hash[b]:continue
            negs.add(pair)
        return sorted(negs)
    train_neg=negative_candidates(train_pos);val_neg=negative_candidates(val_pos)
    print(f'Projects={len(project_info)} positive={len(positives)} train={len(train_pos)} valid={len(val_pos)} negatives={len(train_neg)+len(val_neg)}')
    if args.dry_run:return
    # Atomic-ish: directory created only after validations; no destructive rollback of existing data.
    output.mkdir(parents=True,exist_ok=False)
    try:
        split_sets={'train':[(a,b,1) for a,b in train_pos]+[(a,b,0) for a,b in train_neg],
                    'valid':[(a,b,1) for a,b in val_pos]+[(a,b,0) for a,b in val_neg]}
        used={n for rows in split_sets.values() for a,b,_ in rows for n in (a,b)}
        with (output/'data.jsonl').open('w',encoding='utf-8') as f:
            for k in sorted(used):f.write(json.dumps({'idx':k,'func':corpus[k]},ensure_ascii=False)+'\n')
        for split,rows in split_sets.items():
            rng.shuffle(rows)
            with (output/('train.txt' if split=='train' else 'valid.txt')).open('w',encoding='utf-8') as f:
                for a,b,y in rows:f.write(f'{a}\t{b}\t{y}\n')
        with (output/'positive_provenance.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(provenance[0]));writer.writeheader()
            for item in provenance:writer.writerow(item)
        summary={'version':VERSION,'status':'REVIEW_REQUIRED' if args.sample_negative_candidates else 'POSITIVES_ONLY_NOT_TRAINABLE',
                 'schema':'A-08 compatible idx/func JSONL plus train.txt/valid.txt three-column pair labels',
                 'positives':{'train':len(train_pos),'validation':len(val_pos)},
                 'unverified_negative_candidates':{'train':len(train_neg),'validation':len(val_neg)},
                 'negative_candidate_warning':'Unreported pairs are NOT validated non-clones; do not train without human/independent review',
                 'heldout_test_hash_check':test_details,'counts':dict(audits),'project_runs':project_info,
                 'seed':args.seed,'validation_fraction_requested':args.validation_fraction,
                 'existing_python_mixed_legacy_preserved':True,
                 'warning':'Project-level test leakage risk: C-01 repositories may overlap held-out test projects; exact function-code exclusion is not project-disjoint verification.'}
        (output/'manifest.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        (output/'audit.json').write_text(json.dumps({'status':summary['status'],'counts':dict(audits),
            'train_validation_function_overlap':len({x for a,b,_ in split_sets['train'] for x in (a,b)} & {x for a,b,_ in split_sets['valid'] for x in (a,b)}),
            'test_hashes':len(tests),'negative_labels_verified':False},indent=2)+'\n')
        print('OUTPUT:',output,'STATUS:',summary['status'])
    except BaseException:
        # Preserve the attempted output for diagnosis; never touch legacy or detector data.
        print('ERROR: partial new output retained for diagnosis:',output,file=sys.stderr)
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,default=Path('data/training/clean/python_mixed'))
    p.add_argument('--projects',nargs='+')
    p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--validation-fraction',type=float,default=0.1)
    p.add_argument('--sample-negative-candidates',action='store_true',help='Opt-in heuristic non-clone candidate sampling; requires review')
    p.add_argument('--negative-ratio',type=float,default=1.0)
    p.add_argument('--allow-missing-test',action='store_true',help='Unsafe: use for development only; outputs require additional review')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if not (0<a.validation_fraction<1):p.error('validation fraction must be in (0,1)')
    try:build(a);return 0
    except (OSError,ValueError,KeyError,subprocess.CalledProcessError,UnicodeError) as exc:
        print('C-03 ERROR:',exc,file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
