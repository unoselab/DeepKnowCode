#!/usr/bin/env python3
"""C-04 v3: component-aware resplit, provenance reconciliation, candidate negatives."""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

VERSION = 'v3'

def sha(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def normalized(s):
    return re.sub(r'\s+', '', s)

def file_sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def code_records(path):
    out={}
    with path.open(encoding='utf-8-sig') as f:
        for n,line in enumerate(f,1):
            if not line.strip():continue
            obj=json.loads(line)
            idx=obj['idx']; code=obj['func']
            if not isinstance(idx,str) or not isinstance(code,str) or not code.strip():
                raise ValueError(f'Invalid code record: {path}:{n}')
            if idx in out:raise ValueError(f'Duplicate function ID: {idx}')
            out[idx]=code
    return out

def pair_rows(path, ids):
    result=[]; seen=set()
    with path.open(encoding='utf-8-sig') as f:
        for n,line in enumerate(f,1):
            if not line.strip():continue
            cols=line.split()
            if len(cols)!=3:raise ValueError(f'Invalid three-column row at {path}:{n}')
            a,b,label=cols
            if a not in ids or b not in ids:raise ValueError(f'Unknown function ID at {path}:{n}')
            if a==b:raise ValueError(f'Self pair at {path}:{n}')
            if label != '1':raise ValueError(f'Expected C-03 positive-only dataset at {path}:{n}')
            pair=tuple(sorted((a,b)))
            if pair in seen:raise ValueError(f'Duplicate positive pair at {path}:{n}')
            seen.add(pair);result.append(pair)
    return result

def provenance_map(path, positives):
    """Reconcile C-03 provenance against its *post-deduplication* pair files.

    C-03 writes provenance before code-hash pair deduplication; unmatched rows
    must be excluded from C-04 output, not treated as a fatal error.  Retain
    every row corresponding to a surviving positive pair (including multiple
    detector reports for that pair).  Audit the excluded records.
    """
    id_projects=defaultdict(set); pair_project={}; detector=Counter()
    kept=[]; excluded=Counter()
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        columns=list(reader.fieldnames or [])
        needed={'id_a','id_b','project','source'}
        if not needed.issubset(columns):raise ValueError('Missing C-03 provenance fields')
        if 'c04_split' in columns:raise ValueError('Unexpected c04_split in C-03 provenance')
        for row in reader:
            a,b=row['id_a'],row['id_b'];project=row['project']
            if not a or not b or not project:raise ValueError('Empty C-03 provenance endpoint/project')
            pair=tuple(sorted((a,b)))
            if pair not in positives:
                excluded['not_in_final_c03_positive_pairs']+=1
                continue
            if pair in pair_project and pair_project[pair]!=project:
                raise ValueError('Pair assigned to multiple projects')
            pair_project[pair]=project
            id_projects[a].add(project);id_projects[b].add(project)
            detector[row['source']]+=1
            kept.append(row)
    missing=positives-set(pair_project)
    if missing:raise ValueError(f'{len(missing)} positive pairs missing provenance')
    return id_projects,pair_project,detector,columns,kept,dict(excluded)

def component_split(positive_pairs, code_hash, norm_hash, ratio, seed):
    """Group positive connectivity and identical content before any split assignment.

    Function-level and exact/normalized-code-level overlap is forbidden between splits.
    The target fraction is aspirational when components are indivisible.
    """
    parents = {}
    sizes = {}
    def find(v):
        parents.setdefault(v, v)
        if parents[v] != v:
            parents[v] = find(parents[v])
        return parents[v]
    def join(a,b):
        x,y=find(a),find(b)
        if x == y:return
        if sizes.get(x,1)<sizes.get(y,1):x,y=y,x
        parents[y]=x
        sizes[x]=sizes.get(x,1)+sizes.get(y,1)
    ids={i for pair in positive_pairs for i in pair}
    for a,b in sorted(positive_pairs):join(a,b)
    for hashes in (code_hash,norm_hash):
        by_hash={}
        for i in sorted(ids):
            key=hashes[i]
            if key in by_hash:join(i,by_hash[key])
            else:by_hash[key]=i
    groups=defaultdict(list)
    for pair in sorted(positive_pairs):groups[find(pair[0])].append(pair)
    # Preserve the largest components as indivisible units. Balanced global pair counts
    # are approximate because large components and 1-component projects cannot be cut.
    items=sorted(groups.values(),key=lambda g:(-len(g),g[0]))
    rng=random.Random(seed)
    # Shuffle ties deterministically; stable on otherwise identical inputs.
    ties=defaultdict(list)
    for group in items:ties[len(group)].append(group)
    items=[]
    for count in sorted(ties,reverse=True):
        tied=ties[count]
        rng.shuffle(tied)
        items.extend(tied)
    target=int(round(len(positive_pairs)*ratio))
    splits={'train':[], 'valid':[]}
    assigned={}
    for group in items:
        current=len(splits['train'])
        # Place the group wherever it decreases distance from the train fraction;
        # still try to ensure that train/valid both have at least one component.
        if current < target:
            side='train' if abs(target-(current+len(group)))<=abs(target-current) or not splits['train'] else 'valid'
        else:side='valid'
        splits[side].extend(group)
        for i in {i for a,b in group for i in (a,b)}:assigned[i]=side
    if not splits['valid'] and len(items)>1:
        smallest=min(items,key=lambda g:(len(g),g[0]))
        for pair in smallest:splits['train'].remove(pair)
        splits['valid'].extend(smallest)
        for i in {i for a,b in smallest for i in (a,b)}:assigned[i]='valid'
    if not splits['train'] and len(items)>1:
        smallest=min(items,key=lambda g:(len(g),g[0]))
        for pair in smallest:splits['valid'].remove(pair)
        splits['train'].extend(smallest)
        for i in {i for a,b in smallest for i in (a,b)}:assigned[i]='train'
    return splits,assigned,len(groups)

def positive_roots(pairs):
    parent={}
    def find(i):
        parent.setdefault(i,i)
        if parent[i]!=i:parent[i]=find(parent[i])
        return parent[i]
    for a,b in sorted(pairs):
        x,y=find(a),find(b)
        if x!=y:parent[max(x,y)]=min(x,y)
    return {i:find(i) for i in parent}

def test_hashes(root):
    hashes=set(); metadata={}
    for lang in ('java','python'):
        p=root/'data/evaluation/code'/f'{lang}.jsonl'
        if not p.is_file():raise ValueError(f'Missing held-out test code; cannot certify hash audit: {p}')
        count=0
        with p.open(encoding='utf-8-sig') as f:
            for line in f:
                if not line.strip():continue
                obj=json.loads(line); code=obj['code']
                hashes.add(sha(code));hashes.add(sha(normalized(code)));count+=1
        metadata[lang]={'records':count,'sha256':file_sha(p)}
    return hashes,metadata

def tokenize(code):
    return set(re.findall(r'[A-Za-z_][A-Za-z_0-9]*|\d+',code.lower()))

def generate(args):
    root=args.root.resolve()
    inp=(root/args.input).resolve() if not args.input.is_absolute() else args.input.resolve()
    protected=(root/'data/training/clean').resolve()
    if inp != protected/'python_mixed':
        raise ValueError('C-04 v1 expects the established clean/python_mixed directory')
    if args.ratio < 0 or args.ratio > 10:raise ValueError('Negative ratio must be between 0 and 10')
    if not 0 < args.train_fraction < 1:raise ValueError('Train fraction must be in (0,1)')
    if not (0<=args.jaccard_max<=1):raise ValueError('Invalid token Jaccard threshold')
    if not (0<=args.min_length_ratio<=1):raise ValueError('Invalid length ratio')
    for name in ('data.jsonl','train.txt','valid.txt','positive_provenance.csv','manifest.json','audit.json'):
        if not (inp/name).is_file():raise ValueError(f'Missing C-03 input: {inp/name}')
    manifest=json.loads((inp/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='POSITIVES_ONLY_NOT_TRAINABLE':
        raise ValueError('Expected C-03 positive-only manifest')
    corpus=code_records(inp/'data.jsonl')
    train=pair_rows(inp/'train.txt',corpus)
    valid=pair_rows(inp/'valid.txt',corpus)
    if not train or not valid:raise ValueError('Empty training or validation positive split')
    positives=set(train)|set(valid)
    if len(positives)!=len(train)+len(valid):raise ValueError('Train/valid positive pair overlap')
    original_train_ids={n for p in train for n in p}
    original_valid_ids={n for p in valid for n in p}
    original_overlap=len(original_train_ids&original_valid_ids)
    id_projects, pair_project, sources, provenance_cols, provenance_rows, excluded_provenance=provenance_map(
        inp/'positive_provenance.csv', positives)
    all_ids={i for p in positives for i in p}
    ambiguous=[i for i in all_ids if len(id_projects.get(i,set()))!=1]
    if ambiguous:raise ValueError(f'Project attribution is ambiguous for {len(ambiguous)} IDs')
    code_hash={i:sha(corpus[i]) for i in all_ids}
    norm_hash={i:sha(normalized(corpus[i])) for i in all_ids}
    partition,assignment,n_components=component_split(positives,code_hash,norm_hash,args.train_fraction,args.seed)
    train,valid=partition['train'],partition['valid']
    train_ids={n for p in train for n in p};valid_ids={n for p in valid for n in p}
    shared_ids=train_ids&valid_ids
    cross_exact=len({code_hash[i] for i in train_ids}&{code_hash[i] for i in valid_ids})
    cross_norm=len({norm_hash[i] for i in train_ids}&{norm_hash[i] for i in valid_ids})
    heldout, heldout_files=test_hashes(root)
    heldout_hit={i for i in all_ids if code_hash[i] in heldout or norm_hash[i] in heldout}
    if not train or not valid:
        status='BLOCKED_UNSPLITTABLE_COMPONENTS'
    elif shared_ids or cross_exact or cross_norm or heldout_hit:
        status='BLOCKED_SPLIT_OR_TEST_LEAKAGE'
    else:
        status='UNVERIFIED_NEGATIVES_REVIEW_REQUIRED'
    # The true C-03 positive graph defines known-clone components for conservative
    # negative sampling; the split components additionally unite identical code.
    component=assignment
    positive_components=positive_roots(positives)
    selected=set(); rejected=Counter(); rng=random.Random(args.seed)
    tok={i:tokenize(corpus[i]) for i in train_ids|valid_ids}
    created={};neg_provenance=[]
    for split,positive_pairs,nodes in [('train',train,train_ids),('valid',valid,valid_ids)]:
        per_project=defaultdict(list)
        for i in sorted(nodes):per_project[next(iter(id_projects[i]))].append(i)
        goal=round(len(positive_pairs)*args.ratio)
        candidates=[]
        if status!='BLOCKED_SPLIT_OR_TEST_LEAKAGE':
            # Project-local sampling, guarded against global positive and clone-component edges.
            projects=sorted(per_project)
            per_goal={p:round(goal*sum(pair_project[x]==p for x in positive_pairs)/len(positive_pairs)) for p in projects}
            for p in projects:
                members=per_project[p];target=per_goal[p]
                if len(members)<2:rejected['projects_with_fewer_than_two_functions']+=1;continue
                attempts=max(200,target*80)
                seen=set()
                for _ in range(attempts):
                    if len(seen)>=target:break
                    a,b=sorted(rng.sample(members,2));pair=(a,b)
                    if pair in seen or pair in positives:rejected['known_or_duplicate_pair']+=1;continue
                    if component.get(a)!=split or component.get(b)!=split:
                        raise ValueError('Function assigned to incorrect split')
                    # Split components are not necessarily distinct by assignment;
                    # use the true positive-connected roots checked below.
                    if positive_components[a]==positive_components[b]:rejected['same_clone_component']+=1;continue
                    if norm_hash[a]==norm_hash[b]:rejected['identical_code_content']+=1;continue
                    la,lb=len(corpus[a]),len(corpus[b])
                    if min(la,lb)/max(la,lb)<args.min_length_ratio:rejected['length_ratio']+=1;continue
                    union=tok[a]|tok[b]
                    sim=len(tok[a]&tok[b])/len(union) if union else 1.0
                    if sim>args.jaccard_max:rejected['high_token_similarity']+=1;continue
                    seen.add(pair)
                candidates.extend(sorted(seen))
                if len(seen)<target:rejected['sampling_shortfall']+=target-len(seen)
        # no sample may occur in both output splits
        for a,b in candidates:
            if (a,b) in selected:raise ValueError('Cross-split negative duplication')
            selected.add((a,b))
            neg_provenance.append({'split':split,'id_a':a,'id_b':b,'project':next(iter(id_projects[a])),
                                   'label':'0','verification':'UNVERIFIED','selection':'different_clone_component_same_project',
                                   'source_positive_graph':'C-03 hybrid NiCad+jscpd'})
        created[split]={'positive':positive_pairs,'negative':candidates}
    summary={'version':VERSION,'status':status,'source_c03':str(inp),'source_manifest_sha256':file_sha(inp/'manifest.json'),
             'source_file_hashes':{n:file_sha(inp/n) for n in ('data.jsonl','train.txt','valid.txt','positive_provenance.csv')},
             'positives':{'train':len(train),'valid':len(valid)},
             'negative_candidates':{'train':len(created['train']['negative']),'valid':len(created['valid']['negative'])},
             'function_ids':len(train_ids|valid_ids),'train_valid_function_id_overlap':len(shared_ids),
             'original_c03_train_valid_function_id_overlap':original_overlap,
             'positive_components':n_components,'train_fraction_requested':args.train_fraction,
             'train_fraction_actual':round(len(train)/len(positives),8),
             'train_valid_exact_function_hash_overlap':cross_exact,'train_valid_normalized_function_hash_overlap':cross_norm,
             'heldout_exact_or_normalized_function_matches':len(heldout_hit),'heldout_test_source_hashes':heldout_files,
             'repository_count':len({next(iter(id_projects[i])) for i in train_ids|valid_ids}),
             'detector_sources':dict(sources),'positive_provenance_kept_rows':len(provenance_rows),
             'positive_provenance_excluded_rows':sum(excluded_provenance.values()),
             'positive_provenance_exclusion_reasons':excluded_provenance,
             'negative_sampling_rejections':dict(rejected),
             'ratio_requested':args.ratio,'seed':args.seed,'jaccard_max':args.jaccard_max,
             'min_length_ratio':args.min_length_ratio,
             'warnings':['Every negative is a heuristic candidate, NOT a verified non-clone.',
                         'Absence from NiCad/jscpd output does not establish label 0.',
                         'Project-level held-out split and near-duplicate cross-test independence are NOT established.',
                         'Do NOT use these outputs for StarCoder2-7B training without separate label and leakage review.',
                         'C-04 v2 resplits whole positive components and identical code hashes, retaining all C-03 positive pairs.']}
    print(f"C-04 status={status} train_positive={len(train)} valid_positive={len(valid)} "
          f"train_negative_candidates={len(created['train']['negative'])} "
          f"valid_negative_candidates={len(created['valid']['negative'])} "
          f"function_overlap={len(shared_ids)} heldout_code_overlap={len(heldout_hit)}")
    if args.dry_run:return summary,None
    out=args.output
    if out is None:
        timestamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
        out=root/'results/c-04'/timestamp
    elif not out.is_absolute():out=root/out
    out=out.absolute()
    result_root=(root/'results/c-04').resolve()
    if result_root not in out.parents:
        raise ValueError('C-04 output must be a new directory within results/c-04')
    if out.exists() or out.is_symlink():raise ValueError('Refusing to overwrite existing C-04 output')
    out.mkdir(parents=True,exist_ok=False)
    for split in ('train','valid'):
        rows=[(*p,1) for p in created[split]['positive']]+[(*p,0) for p in created[split]['negative']]
        rng.shuffle(rows)
        with (out/f'{split}.txt').open('w',encoding='utf-8') as f:
            for a,b,y in rows:f.write(f'{a}\t{b}\t{y}\n')
    # Same A-08 corpus and split schema, not copied from another dataset.
    with (out/'data.jsonl').open('w',encoding='utf-8') as f:
        for i in sorted(train_ids|valid_ids):f.write(json.dumps({'idx':i,'func':corpus[i]},ensure_ascii=False)+'\n')
    with (out/'negative_candidates.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['split','id_a','id_b','project','label','verification','selection','source_positive_graph'])
        writer.writeheader();writer.writerows(neg_provenance)
    # Input C-03 provenance can contain rows for code-hash-deduplicated pairs.
    # Only surviving positive pairs are emitted, with the *new* C-04 split.
    with (out/'positive_provenance.csv').open('w',newline='',encoding='utf-8') as dst:
        writer=csv.DictWriter(dst,fieldnames=provenance_cols+['c04_split'])
        writer.writeheader()
        for record in provenance_rows:
            record=dict(record)
            record['c04_split']=assignment[record['id_a']]
            writer.writerow(record)
    (out/'audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({'version':VERSION,'status':status,
        'training_approval':False,'source':'C-03 v2 positive-only Python dataset',
        'negative_labels_verified':False,'input_dir':str(inp),'output_dir':str(out)},indent=2)+'\n',encoding='utf-8')
    print('OUTPUT:',out)
    return summary,out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--input',type=Path,default=Path('data/training/clean/python_mixed'))
    p.add_argument('--output',type=Path)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--ratio',type=float,default=1.0)
    p.add_argument('--train-fraction',type=float,default=0.8)
    p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--jaccard-max',type=float,default=0.65)
    p.add_argument('--min-length-ratio',type=float,default=0.30)
    args=p.parse_args()
    try:
        summary,_=generate(args)
        return 2 if summary['status'].startswith('BLOCKED_') else 0
    except (OSError,ValueError,KeyError,json.JSONDecodeError) as exc:
        print('C-04 ERROR:',exc,file=sys.stderr)
        return 3
if __name__=='__main__':sys.exit(main())
