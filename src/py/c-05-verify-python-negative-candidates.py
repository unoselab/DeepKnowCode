#!/usr/bin/env python3
"""C-05 v1: conservative independent screening of C-04 negative candidates.

Never treat a screening pass as a ground-truth negative or training approval.
"""
import argparse
import ast
import csv
import hashlib
import json
import random
import re
import sys
import textwrap
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v2'
TZ = ZoneInfo('America/Chicago')
REQUIRED = ('data.jsonl', 'train.txt', 'valid.txt', 'negative_candidates.csv',
            'positive_provenance.csv', 'audit.json', 'manifest.json')
HEADERS = ['split','id_a','id_b','project','screening','flags','token_jaccard',
           'sequence_similarity','length_ratio','ast_shape_equal','exact_code_equal',
           'normalized_code_equal','same_positive_component','known_positive_pair']

def sha_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def canon(a,b):
    if not a or not b or a==b:
        raise ValueError('Empty or self-linked pair')
    return tuple(sorted((a,b)))

def load_corpus(path):
    out={}
    with path.open(encoding='utf-8-sig') as f:
        for n,line in enumerate(f,1):
            if not line.strip():continue
            obj=json.loads(line)
            idx,code=obj['idx'],obj['func']
            if not isinstance(idx,str) or not isinstance(code,str) or not code.strip() or idx in out:
                raise ValueError(f'Invalid/duplicate code record {path}:{n}')
            out[idx]=code
    if not out:raise ValueError('Empty code dataset')
    return out

def load_pairs(path, corpus):
    out={0:[],1:[]};seen=set()
    with path.open(encoding='utf-8-sig') as f:
        for n,line in enumerate(f,1):
            if not line.strip():continue
            parts=line.split()
            if len(parts)!=3 or parts[2] not in ('0','1'):
                raise ValueError(f'Invalid pair row {path}:{n}')
            a,b,label=parts; pair=canon(a,b)
            if a not in corpus or b not in corpus or pair in seen:
                raise ValueError(f'Unknown IDs or duplicate pair {path}:{n}')
            seen.add(pair);out[int(label)].append(pair)
    return out

def load_provenance(path, positive_set, positive_by_split):
    proj_by_id=defaultdict(set); pair_proj={};counts=Counter();github_by_project=defaultdict(set)
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        req={'id_a','id_b','project','source','github_repository','c04_split'}
        if not req.issubset(reader.fieldnames or []):raise ValueError('Positive provenance schema incomplete')
        for row in reader:
            pair=canon(row['id_a'],row['id_b'])
            p=row['project'];split=row['c04_split']
            if pair not in positive_set:raise ValueError('Unexpected positive provenance pair')
            if pair not in positive_by_split[split]:
                raise ValueError('Positive provenance split disagrees with pair file')
            if not p or split not in ('train','valid') or not row['github_repository']:
                raise ValueError('Incomplete provenance project/split/repository')
            if pair in pair_proj and pair_proj[pair]!=p:raise ValueError('Conflicting positive project')
            pair_proj[pair]=p
            for i in pair:proj_by_id[i].add(p)
            github_by_project[p].add(row['github_repository'])
            counts[row['source']]+=1
    if set(pair_proj)!=positive_set:raise ValueError('Positive provenance missing pairs')
    return proj_by_id,pair_proj,dict(counts),github_by_project

def load_neg_csv(path, negative_by_split, projects):
    seen=set();rows=[]
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        needed={'split','id_a','id_b','project','label','verification'}
        if not needed.issubset(reader.fieldnames or []):raise ValueError('Negative CSV schema incomplete')
        for row in reader:
            pair=canon(row['id_a'],row['id_b']);s=row['split']
            if s not in ('train','valid') or row['label']!='0' or row['verification']!='UNVERIFIED':
                raise ValueError('Negative CSV incorrectly labeled or split')
            if pair in seen or pair not in negative_by_split[s]:
                raise ValueError('Unexpected/duplicate negative provenance pair')
            if row['project'] not in projects:raise ValueError('Unknown negative project')
            seen.add(pair); rows.append((s,pair[0],pair[1],row['project']))
    if seen != set().union(*negative_by_split.values()):
        raise ValueError('Negative CSV missing pairs')
    return rows

def union_find(positives):
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for a,b in sorted(positives):
        x,y=find(a),find(b)
        if x!=y:parent[max(x,y)]=min(x,y)
    return {i:find(i) for i in parent}

def words(code):
    return re.findall(r'[A-Za-z_]\w*|\d+',code.lower())

def normalize_code(code):
    return re.sub(r'\s+','',code)

class ShapeNormalizer(ast.NodeTransformer):
    def visit_Name(self,node):
        node.id='_';return node
    def visit_arg(self,node):
        node.arg='_';return self.generic_visit(node)
    def visit_Attribute(self,node):
        node.attr='_';return self.generic_visit(node)
    def visit_FunctionDef(self,node):
        node.name='_';return self.generic_visit(node)
    def visit_AsyncFunctionDef(self,node):
        node.name='_';return self.generic_visit(node)
    def visit_ClassDef(self,node):
        node.name='_';return self.generic_visit(node)
    def visit_Constant(self,node):
        node.value=None; return node

def ast_shape(code):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree=ast.parse(textwrap.dedent(code))
        return sha_text(ast.dump(ShapeNormalizer().visit(tree),annotate_fields=True,include_attributes=False)),None
    except (SyntaxError,ValueError,TypeError) as exc:
        return None,type(exc).__name__

def load_heldout(root):
    hashes=set(); info={}
    for lang in ('java','python'):
        path=root/'data/evaluation/code'/f'{lang}.jsonl'
        if not path.is_file():raise ValueError(f'Required held-out test code missing: {path}')
        count=0
        with path.open(encoding='utf-8-sig') as f:
            for n,line in enumerate(f,1):
                if not line.strip():continue
                ob=json.loads(line)
                code=ob['code']
                if not isinstance(code,str):raise ValueError(f'Invalid held-out code {path}:{n}')
                hashes.add(sha_text(code));hashes.add(sha_text(normalize_code(code)));count+=1
        if count==0:raise ValueError(f'Empty held-out dataset: {path}')
        info[lang]={'file_sha256':file_sha(path),'records':count}
    return hashes,info

def similarity(a,b):
    # Independent conservative screening: token-set overlap and token-order overlap.
    from difflib import SequenceMatcher
    ta=words(a);tb=words(b);sa=set(ta);sb=set(tb)
    j=len(sa&sb)/len(sa|sb) if sa|sb else 1.
    ratio=SequenceMatcher(None,ta[:400],tb[:400],autojunk=True).ratio() if ta and tb else 1.
    lr=min(len(a),len(b))/max(len(a),len(b))
    return j,ratio,lr

def examine(args):
    root=args.root.resolve()
    input_path=(root/args.input).resolve() if not args.input.is_absolute() else args.input.resolve()
    if not input_path.is_dir():raise ValueError(f'C-04 output directory not found: {input_path}')
    for n in REQUIRED:
        if not (input_path/n).is_file():raise ValueError(f'Missing C-04 file: {n}')
    origin=json.loads((input_path/'audit.json').read_text(encoding='utf-8'))
    oldmanifest=json.loads((input_path/'manifest.json').read_text(encoding='utf-8'))
    if origin.get('status')!='UNVERIFIED_NEGATIVES_REVIEW_REQUIRED' or oldmanifest.get('training_approval') is not False:
        raise ValueError('C-04 audit/manifest status or approval is unexpected')
    if oldmanifest.get('status')!=origin.get('status'):
        raise ValueError('C-04 audit and manifest disagree')
    corpus=load_corpus(input_path/'data.jsonl')
    pairs={s:load_pairs(input_path/f'{s}.txt',corpus) for s in ('train','valid')}
    positives=set(pairs['train'][1])|set(pairs['valid'][1])
    negatives={s:set(pairs[s][0]) for s in ('train','valid')}
    if (set(pairs['train'][1])&set(pairs['valid'][1])) or (negatives['train']&negatives['valid']):
        raise ValueError('Pair appears in both splits')
    if positives & (negatives['train']|negatives['valid']):
        raise ValueError('Label conflict: positive pair is labeled negative')
    for s in ('train','valid'):
        if len(pairs[s][1])!=origin.get('positives',{}).get(s) or len(negatives[s])!=origin.get('negative_candidates',{}).get(s):
            raise ValueError(f'C-04 counts disagree with {s} pair data')
    pos_by_id,pproj,detector,github=load_provenance(input_path/'positive_provenance.csv',positives,
                      {s:set(pairs[s][1]) for s in ('train','valid')})
    negrows=load_neg_csv(input_path/'negative_candidates.csv',negatives,set(pproj.values()))
    used={i for x in positives | negatives['train'] | negatives['valid'] for i in x}
    if set(corpus)!=used:raise ValueError('Unused or missing function records')
    if any(len(pos_by_id[i])!=1 for i in used):raise ValueError('Ambiguous/missing function project')
    neg_proj_errors=0
    for _,a,b,p in negrows:
        if p not in pos_by_id[a] or p not in pos_by_id[b]:neg_proj_errors+=1
    ids={s:{i for label in (0,1) for a,b in pairs[s][label] for i in (a,b)} for s in ('train','valid')}
    split_overlap=ids['train']&ids['valid']
    content_hash={i:sha_text(corpus[i]) for i in used}
    norm_hash={i:sha_text(normalize_code(corpus[i])) for i in used}
    hash_split_overlap=({content_hash[i] for i in ids['train']}&{content_hash[i] for i in ids['valid']})|({norm_hash[i] for i in ids['train']}&{norm_hash[i] for i in ids['valid']})
    heldout,heldout_info=load_heldout(root)
    heldout_matches=sum(1 for i in used if content_hash[i] in heldout or norm_hash[i] in heldout)
    posroot=union_find(positives)
    all_ids=sorted(used)
    asts={i:ast_shape(corpus[i]) for i in all_ids}
    ast_error=sum(1 for _,e in asts.values() if e)
    crossrepo=defaultdict(set)
    for i in all_ids:
        crossrepo[norm_hash[i]].add(next(iter(pos_by_id[i])))
    reused_hashes={key for key,projects in crossrepo.items() if len(projects)>1}
    records=[];ct=Counter();risk_count=Counter();proj_counts=defaultdict(Counter)
    for split,a,b,project in negrows:
        flags=[]
        pair=(a,b)
        j,seq,lr=similarity(corpus[a],corpus[b])
        equal=content_hash[a]==content_hash[b]
        normequal=norm_hash[a]==norm_hash[b]
        same=posroot.get(a)==posroot.get(b)
        known=pair in positives
        ast_equal=bool(asts[a][0] and asts[a][0]==asts[b][0])
        if known:flags.append('KNOWN_POSITIVE_PAIR')
        if same:flags.append('POSITIVE_CONNECTED_COMPONENT')
        if equal or normequal:flags.append('IDENTICAL_SOURCE')
        if ast_equal:flags.append('SAME_NORMALIZED_AST_SHAPE')
        if j>=args.token_jaccard_review:flags.append('HIGH_TOKEN_JACCARD')
        if seq>=args.sequence_review:flags.append('HIGH_TOKEN_SEQUENCE_SIMILARITY')
        if lr<args.min_length_ratio_review:flags.append('LENGTH_IMBALANCE')
        if asts[a][1] or asts[b][1]:flags.append('AST_PARSE_UNAVAILABLE')
        if norm_hash[a] in reused_hashes or norm_hash[b] in reused_hashes:flags.append('CROSS_PROJECT_IDENTICAL_FUNCTION')
        if project not in pos_by_id[a] or project not in pos_by_id[b]:flags.append('PROJECT_PROVENANCE_CONFLICT')
        if content_hash[a] in heldout or norm_hash[a] in heldout or content_hash[b] in heldout or norm_hash[b] in heldout:
            flags.append('HELDOUT_CODE_OVERLAP')
        if any(k in flags for k in ('KNOWN_POSITIVE_PAIR','POSITIVE_CONNECTED_COMPONENT','IDENTICAL_SOURCE','PROJECT_PROVENANCE_CONFLICT','HELDOUT_CODE_OVERLAP')):
            category='REJECT_CONFLICT'
        elif flags:
            category='REVIEW_RISK'
        else:
            category='SCREENED_UNVERIFIED'
        ct[category]+=1;proj_counts[project][category]+=1
        for flag in flags:risk_count[flag]+=1
        records.append({'split':split,'id_a':a,'id_b':b,'project':project,'screening':category,
                        'flags':';'.join(flags),'token_jaccard':f'{j:.6f}',
                        'sequence_similarity':f'{seq:.6f}','length_ratio':f'{lr:.6f}',
                        'ast_shape_equal':int(ast_equal),'exact_code_equal':int(equal),
                        'normalized_code_equal':int(normequal),'same_positive_component':int(same),
                        'known_positive_pair':int(known)})
    if neg_proj_errors!=risk_count.get('PROJECT_PROVENANCE_CONFLICT',0):raise ValueError('Provenance disagreement')
    blockers=[]
    if split_overlap:blockers.append('TRAIN_VALID_FUNCTION_OVERLAP')
    if hash_split_overlap:blockers.append('TRAIN_VALID_CODE_HASH_OVERLAP')
    if heldout_matches:blockers.append('HELDOUT_FUNCTION_CODE_OVERLAP')
    if ct['REJECT_CONFLICT']:blockers.append('NEGATIVE_LABEL_CONFLICT')
    if not records:blockers.append('NO_NEGATIVE_CANDIDATES')
    status='BLOCKED_DATA_INTEGRITY' if blockers else 'SCREENING_COMPLETE_REVIEW_REQUIRED'
    outdir=None
    if not args.dry_run:
        stamp=datetime.now(TZ).strftime('%Y%m%d-%H%M%S-%f')
        outdir=(root/'results/c-05'/stamp) if args.output is None else (root/args.output if not args.output.is_absolute() else args.output)
        outdir=outdir.absolute()
        resultroot=(root/'results/c-05').resolve()
        if resultroot not in outdir.parents:raise ValueError('Output must be inside results/c-05')
        if outdir.exists() or outdir.is_symlink():raise ValueError(f'Output exists; refusing overwrite: {outdir}')
    rng=random.Random(args.seed)
    # Reproducible per-project review sample, drawn from risky candidates first.
    groups=defaultdict(list)
    for item in records:groups[(item['project'],item['screening'])].append(item)
    selected=[]
    for key in sorted(groups):
        group=groups[key];sample=rng.sample(group,min(len(group),args.review_per_project_status))
        selected.extend(sample)
    selected.sort(key=lambda r:(r['project'],r['screening'],r['split'],r['id_a'],r['id_b']))
    manifest={'version':VERSION,'status':status,'training_approval':False,
              'negative_labels_verified':False,'negative_label_policy':'screening_only_manual_or_independent_ground_truth_required',
              'timezone':'America/Chicago','timestamp_chicago':datetime.now(TZ).isoformat(),
              'source_c04':str(input_path),'source_c04_manifest_sha256':file_sha(input_path/'manifest.json'),
              'input_sha256':{n:file_sha(input_path/n) for n in REQUIRED}}
    audit={'version':VERSION,'status':status,'training_approval':False,'source_c04':str(input_path),
           'counts':{'positive_train':len(pairs['train'][1]),'positive_valid':len(pairs['valid'][1]),
                     'negative_train':len(negatives['train']),'negative_valid':len(negatives['valid']),
                     'total_negative_candidates':len(records),'function_ids':len(used)},
           'screening':dict(ct),'flag_counts':dict(risk_count),'project_counts':{p:dict(v) for p,v in sorted(proj_counts.items())},
           'positive_detector_sources':detector,'github_repositories_by_project':{p:sorted(v) for p,v in sorted(github.items())},
           'train_valid_function_overlap':len(split_overlap),'train_valid_code_hash_overlap':len(hash_split_overlap),
           'heldout_function_code_overlap':heldout_matches,'heldout_inputs':heldout_info,'ast_parse_unavailable_functions':ast_error,
           'cross_project_reused_function_code_hashes':len(reused_hashes),
           'blockers':blockers,'review_samples':len(selected),
           'thresholds':{'token_jaccard_review':args.token_jaccard_review,'sequence_similarity_review':args.sequence_review,
                         'min_length_ratio_review':args.min_length_ratio_review},
           'limitations':['SCREENED_UNVERIFIED does NOT establish ground-truth label 0.',
                          'AST shape matching is only a heuristic; different AST shapes do not prove non-clones.',
                          'Token sequence similarity is an approximate first-400-token screening heuristic.',
                          'Near-duplicate overlap against held-out examples is not exhaustively tested.',
                          'Project-level evaluation/train separation is not established by hash overlap alone.',
                          'Manual or independent non-clone adjudication is needed before training promotion.']}
    print('C-05 status={status} total={n} screened={s} review={r} reject={x} split_overlap={o} heldout_overlap={h}'.format(
        status=status,n=len(records),s=ct['SCREENED_UNVERIFIED'],r=ct['REVIEW_RISK'],x=ct['REJECT_CONFLICT'],
        o=len(split_overlap),h=heldout_matches))
    if not args.dry_run:
        outdir.mkdir(parents=True,exist_ok=False)
        with (outdir/'negative_verification.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=HEADERS);w.writeheader();w.writerows(records)
        with (outdir/'review_sample.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=HEADERS);w.writeheader();w.writerows(selected)
        with (outdir/'review_sample_code.jsonl').open('w',encoding='utf-8') as f:
            for r in selected:
                f.write(json.dumps({'split':r['split'],'project':r['project'],'id_a':r['id_a'],'id_b':r['id_b'],
                                    'screening':r['screening'],'flags':r['flags'],
                                    'code_a':corpus[r['id_a']],'code_b':corpus[r['id_b']]},ensure_ascii=False)+'\n')
        (outdir/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        (outdir/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print('OUTPUT:',outdir)
    return audit,outdir

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--input',type=Path,default=Path('results/c-04/20261103-193519-475848'))
    p.add_argument('--output',type=Path)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--token-jaccard-review',type=float,default=0.45)
    p.add_argument('--sequence-review',type=float,default=0.65)
    p.add_argument('--min-length-ratio-review',type=float,default=0.5)
    p.add_argument('--review-per-project-status',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261103)
    args=p.parse_args(argv)
    try:
        for k in ('token_jaccard_review','sequence_review','min_length_ratio_review'):
            x=getattr(args,k)
            if not 0<=x<=1:raise ValueError(f'{k} outside [0,1]')
        if args.review_per_project_status<0:raise ValueError('Negative review sample size')
        audit,_=examine(args)
        return 2 if audit['status'].startswith('BLOCKED') else 0
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError,UnicodeDecodeError) as exc:
        print('C-05 ERROR:',exc,file=sys.stderr)
        return 3

if __name__=='__main__':sys.exit(main())
