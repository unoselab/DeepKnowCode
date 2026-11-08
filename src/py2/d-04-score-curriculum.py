#!/usr/bin/env python3
"""D-04 v2: non-mutating 19,464-pair difficulty audit (NOT M3 training)."""
import argparse
import collections
import datetime
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
from zoneinfo import ZoneInfo

CORPORA = ('java_source', 'java_mixed', 'python_mixed')
DEFAULT_PREPARED = 'data/training/prepared/balanced_multilingual'
DEFAULT_OUTPUT = 'results/d-04/difficulty-audit-v2'
# Original ManageCodeClones weights, retained as a reference only.
W_ORIG = {'cf': .15, 'similarity': .20, 'semantic': .25, 'df': .25, 'architecture': .15}
# Candidate weights, explicitly unvalidated; not a selected curriculum policy.
W_5D = {'cf': .15, 'similarity': .35, 'structural': .20, 'df': .15, 'architecture': .15}
W_4D = {'similarity': .45, 'cf': .25, 'length': .20, 'df': .10}
BRANCH = {'java': {'if_statement','for_statement','enhanced_for_statement','while_statement','do_statement','catch_clause','switch_expression','switch_statement','ternary_expression'},
          'python': {'if_statement','for_statement','while_statement','except_clause','conditional_expression','boolean_operator'}}
COMMENTS = {'comment','line_comment','block_comment'}
IDENTIFIERS = {'identifier'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for buf in iter(lambda: f.read(1024*1024), b''):
            h.update(buf)
    return h.hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write('\n')


def read_jsonl(path):
    with path.open(encoding='utf-8-sig') as f:
        for line_num, line in enumerate(f,1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as e:
                    raise ValueError(f'{path}:{line_num}: {e}') from e


def resolve_input(raw, root):
    p = Path(raw)
    if p.exists(): return p
    # Manifest generated on a different server: retain relative data/training/... tail.
    parts = p.parts
    if 'data' in parts:
        idx = parts.index('data')
        relative = Path(*parts[idx:])
        test = root / relative
        if test.exists(): return test
    raise FileNotFoundError(f'Manifest data dependency missing: {raw}; root={root}')


def read_codes(path):
    codes = {}
    for row in read_jsonl(path):
        k = str(row['idx']).strip()
        body = row['func']
        if not k or k in codes or not isinstance(body,str) or not body.strip():
            raise ValueError(f'Invalid or repeated code idx in {path}: {k!r}')
        codes[k] = body
    return codes


def parser_for(language):
    try:
        from tree_sitter import Language, Parser
        if language == 'java':
            import tree_sitter_java as grammar
        else:
            import tree_sitter_python as grammar
        return Parser(Language(grammar.language()))
    except ImportError as e:
        raise RuntimeError('Install: pip install tree-sitter==0.25.2 tree-sitter-java==0.23.5 tree-sitter-python==0.25.0') from e


def parse_features(parser, code, language):
    """Parse fragments safely; only accept exact ASTs or wrapped Java fragments.

    Wrapper strategy must never relabel an actual erroneous snippet as valid:
    retain parse_mode and first-error evidence for audit. The wrapper's AST
    contains synthetic class nodes; scores are sensitivity proxies, not semantics.
    """
    def parse_once(text):
        return parser.parse(text.encode('utf8'))

    original=parse_once(code)
    tree=original
    parse_mode='direct'
    raw_error=bool(original.root_node.has_error)
    if raw_error and language=='java':
        # Java code can be method declarations without an enclosing class, or
        # statement fragments requiring a method and class context.
        candidates=[
          ('wrapped_class', 'class D04SyntheticClass {\n'+code+'\n}', 'class_body'),
          ('wrapped_method', 'class D04SyntheticClass { void d04SyntheticMethod() {\n'+code+'\n} }', 'block')
        ]
        for mode,text,extract_type in candidates:
            candidate=parse_once(text)
            if candidate.root_node.has_error:continue
            # Keep only nodes corresponding to original body; synthetic wrappers
            # are excluded from lexical/AST metrics as much as possible.
            holder=None; stack=[candidate.root_node]
            while stack:
                n=stack.pop()
                if n.type==extract_type:
                    holder=n;break
                stack.extend(reversed(n.children))
            if holder is None:continue
            tree=candidate; parse_mode=mode
            break
    if tree.root_node.has_error:
        errors=[]
        stack=[original.root_node]
        while stack and len(errors)<6:
            n=stack.pop()
            if n.type=='ERROR' or n.is_missing:
                errors.append({'node_type':n.type,'start_row':n.start_point.row,
                               'start_col':n.start_point.column,'missing':bool(n.is_missing)})
            stack.extend(reversed(n.children))
        return {'error':'AST_HAS_ERROR','has_error':True,'parse_mode':'failed',
                'original_has_error':raw_error,'error_nodes':errors}

    # Parse the original code separately through a wrapper and restrict the
    # feature computation to code-spanning children of the artificial container.
    root=tree.root_node
    if parse_mode!='direct':
        needle='class_body' if parse_mode=='wrapped_class' else 'block'
        stack=[root]
        holder=None
        while stack:
            n=stack.pop()
            if n.type==needle: holder=n; break
            stack.extend(reversed(n.children))
        # Exclude synthetic opening/closing braces; keep code-bearing nodes.
        roots=[n for n in holder.named_children if n.type not in {'ERROR'}] if holder else []
    else:
        roots=[root]
    types, leaves, identifiers=[],[],[]
    branches=0
    todo=list(reversed(roots))
    while todo:
        node=todo.pop()
        if node.type in COMMENTS:continue
        types.append(node.type)
        if node.type in BRANCH[language]: branches+=1
        if node.type in IDENTIFIERS and not node.children:
            identifiers.append(node.text.decode('utf8',errors='replace'))
        if not node.children:
            token=node.text.decode('utf8',errors='replace').strip()
            if token:leaves.append(token)
        else:todo.extend(reversed(node.children))
    return {'error':None,'has_error':False,'parse_mode':parse_mode,
            'original_has_error':raw_error,'error_nodes':[],
            'types':types,'leaves':leaves,'identifiers':identifiers,
            'branches':branches,'complexity':branches+1,'token_count':len(leaves)}


def jaccard(x,y):
    a,b=set(x),set(y)
    return len(a&b)/len(a|b) if a or b else 0.0


def harmonic(a,b):
    return 2*a*b/(a+b) if a+b else 0.0


def clip01(v): return min(1.0,max(0.0,v))


def percentile(values, pct):
    if not values:return None
    vs=sorted(values); pos=(len(vs)-1)*pct; i=int(pos); j=min(i+1,len(vs)-1)
    return vs[i]+(vs[j]-vs[i])*(pos-i)


def normalize(v,bounds):
    if v is None:return None
    lo,hi=bounds
    if hi <= lo:return 0.0
    return clip01((v-lo)/(hi-lo))


def pair_metrics(a,b,language,parser,feature_cache,ida,idb,corpus):
    def feature(fid,code):
        key=(corpus,fid)
        if key not in feature_cache:feature_cache[key]=parse_features(parser,code,language)
        return feature_cache[key]
    f1,f2=feature(ida,a),feature(idb,b)
    parse_modes=[f1['parse_mode'],f2['parse_mode']]
    errors=[f1.get('error_nodes',[]),f2.get('error_nodes',[])]
    if f1['error'] or f2['error']:
        return {'status':'AST_PARSE_FAILED','cf_raw':None,'df_raw':None,'length_raw':None,
                'lexical_similarity':None,'structural_similarity':None,'similarity':None,
                'structural_divergence':None,'architecture':None,
                'structural_similarity_autojunk_false':None,
                'similarity_autojunk_false':None,
                'parse_modes':parse_modes,'parse_error_details':errors,
                'metric_warnings':[f1['error'],f2['error']]}
    lx=jaccard(f1['leaves'],f2['leaves'])
    st=difflib.SequenceMatcher(None,f1['types'],f2['types'],autojunk=True).ratio()
    # Evaluation only: compare to autojunk=False without changing primary
    # similarity definition. This can be substantially slower on repeated AST.
    # Deterministic, bounded sensitivity sample to avoid pathological O(n*m)
    # runtime of autojunk=False on repeated AST token sequences.
    sample_id=(corpus+'|'+ida+'|'+idb).encode('utf8')
    sample_selected=int.from_bytes(hashlib.sha256(sample_id).digest()[:4],'big') % 12 == 0
    sample_selected=sample_selected and max(len(f1['types']),len(f2['types'])) <= 1500
    st_nj=(difflib.SequenceMatcher(None,f1['types'],f2['types'],autojunk=False).ratio()
           if sample_selected else None)
    sim=harmonic(lx,st)
    sim_nj=harmonic(lx,st_nj) if st_nj is not None else None
    cf=max(f1['complexity'],f2['complexity'])+.5*abs(f1['complexity']-f2['complexity'])
    d1,d2=len(set(f1['identifiers'])),len(set(f2['identifiers']))
    df=math.hypot(d1,d2)
    return {'status':'OK','cf_raw':cf,'df_raw':df,
            'length_raw':max(f1['token_count'],f2['token_count']),
            'lexical_similarity':lx,'structural_similarity':st,
            'structural_similarity_autojunk_false':st_nj,
            'similarity':sim,'similarity_autojunk_false':sim_nj,
            'structural_divergence':1-st,'architecture':None,
            'parse_modes':parse_modes,'parse_error_details':errors,
            'metric_warnings':['DF_IDENTIFIER_DIVERSITY_PROXY','ARCHITECTURAL_PROVENANCE_UNAVAILABLE']}


def composite(row, weights):
    if any(row.get(k) is None for k in weights):return None
    return round(sum(row[k]*w for k,w in weights.items()),7)


def compute_similarity_sensitivity(scores):
    groups=collections.defaultdict(list)
    for r in scores:
        if r['composite_similarity'] is not None and r['composite_similarity_autojunk_false'] is not None:
            groups[(r['corpus'],r['label'])].append(r)
    output={}
    for (c,y),items in sorted(groups.items()):
        diffs=[r['composite_similarity_autojunk_false']-r['composite_similarity'] for r in items]
        output[f'{c}/label={y}']={
            'count':len(items),'mean_similarity_autojunk_true':statistics.mean(r['composite_similarity'] for r in items),
            'mean_similarity_autojunk_false':statistics.mean(r['composite_similarity_autojunk_false'] for r in items),
            'mean_delta':statistics.mean(diffs),
            'p95_abs_delta':percentile([abs(d) for d in diffs],.95),
            'delta_gt_0_10_count':sum(abs(d)>.10 for d in diffs)}
    return output


def do_score(args):
    root=Path(args.root).resolve(); prepared=Path(args.prepared)
    if not prepared.is_absolute(): prepared=root/prepared
    out=Path(args.output)
    if not out.is_absolute(): out=root/out
    if out.resolve() == prepared.resolve() or prepared.resolve() in out.resolve().parents:
        raise ValueError('Output must not overwrite the prepared dataset')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output already contains files (do not overwrite): {out}')
    manifest=json.loads((prepared/'manifest.json').read_text(encoding='utf8'))
    if manifest.get('status')!='PREPARED_NOT_TRAINED': raise ValueError('Unexpected preparation status')
    train_file=prepared/'train.jsonl'; valid_file=prepared/'validation.jsonl'
    for split,file in [('train',train_file),('validation',valid_file)]:
        exp=manifest['sample'][split]['sha256']; got=sha256(file)
        if exp != got:raise ValueError(f'{split} SHA256 mismatch: {got} vs {exp}')
    train=list(read_jsonl(train_file)); valid=list(read_jsonl(valid_file))
    if len(train)!=19464 or len(valid)!=512: raise ValueError('Unexpected dataset size')
    def key(r):return (r['corpus'],str(r['id1']),str(r['id2']),int(r['label']))
    if len(set(map(key,train)))!=len(train):raise ValueError('Duplicate training pair detected')
    if set(map(key,train))&set(map(key,valid)):raise ValueError('Train/valid pair overlap')
    counts=collections.Counter((r['corpus'],int(r['label'])) for r in train)
    expected={('java_source',0):6000,('java_source',1):6000,('java_mixed',0):1866,
              ('java_mixed',1):1866,('python_mixed',0):1866,('python_mixed',1):1866}
    if counts!=expected:raise ValueError(f'Training counts unexpected: {counts}')
    codes={}; hashes={}
    for corpus in CORPORA:
        raw=manifest['data_paths'][corpus]['data.jsonl']
        p=resolve_input(raw,root)
        actual=sha256(p); expected_hash=manifest['data_sha256'][corpus]['data.jsonl']
        if actual!=expected_hash:raise ValueError(f'Code corpus hash changed: {corpus}')
        codes[corpus]=read_codes(p); hashes[corpus]=actual
    parsers={'java':parser_for('java'),'python':parser_for('python')}
    cache={}; rows=[]; dist=collections.defaultdict(list); missing=collections.Counter()
    for i,r in enumerate(train):
        corpus=r['corpus']; language='python' if corpus=='python_mixed' else 'java'
        id1,id2=str(r['id1']),str(r['id2'])
        if id1 not in codes[corpus] or id2 not in codes[corpus]:
            raise ValueError(f'Unmatched function ID: {corpus}/{id1}/{id2}')
        metrics=pair_metrics(codes[corpus][id1],codes[corpus][id2],language,parsers[language],cache,id1,id2,corpus)
        row={**r,**metrics,'row_index':i,'language':language}
        if row['status']=='OK':
            for metric in ('cf_raw','df_raw','length_raw'):
                dist[(corpus,metric)].append(row[metric])
        else:missing[(corpus,'AST_PARSE_FAILED')]+=1
        rows.append(row)
        if (i+1)%2000==0:print(f'[D-04] SCORED {i+1}/{len(train)}',flush=True)
    # Training-only, group-specific robust bounds to prevent language scale artifacts.
    bounds={}
    for c in CORPORA:
        bounds[c]={}
        for m in ('cf_raw','df_raw','length_raw'):
            vals=dist[(c,m)]; bounds[c][m]=[percentile(vals,.05),percentile(vals,.95)] if vals else None
    scores=[]
    for row in rows:
        c=row['corpus'];label=int(row['label']);sim=row['similarity']
        def norm(name):
            b=bounds[c][name]
            return normalize(row[name],b) if b is not None else None
        cf,df,length=norm('cf_raw'),norm('df_raw'),norm('length_raw')
        detection_sim=(1-sim if label==1 else sim) if sim is not None else None
        # Original Baseline only if proven architecture AND a defensible 1-4 semantic score.
        # Neither exists here. Explicit NULL instead of manufacturing results.
        original=None
        metrics5={'cf':cf,'similarity':detection_sim,'structural':row['structural_divergence'],
                  'df':df,'architecture':row['architecture']}
        metrics4={'similarity':detection_sim,'cf':cf,'length':length,'df':df}
        r5=composite(metrics5,W_5D)
        r4=composite(metrics4,W_4D)
        output={'row_index':row['row_index'],'corpus':c,'language':row['language'],
                'id1':str(row['id1']),'id2':str(row['id2']),'label':label,
                'tokens':row.get('tokens'),'status':row['status'],
                'lexical_similarity':row['lexical_similarity'],
                'parse_modes':row['parse_modes'],'parse_error_details':row['parse_error_details'],
                'structural_similarity_autojunk_false':row['structural_similarity_autojunk_false'],
                'composite_similarity_autojunk_false':row['similarity_autojunk_false'],
                'structural_similarity':row['structural_similarity'],
                'composite_similarity':sim,'label_aware_similarity_difficulty':detection_sim,
                'cf_raw':row['cf_raw'],'cf_normalized':cf,
                'df_identifier_proxy_raw':row['df_raw'],'df_proxy_normalized':df,
                'length_raw':row['length_raw'],'length_normalized':length,
                'architectural_distance':None,'structural_divergence':row['structural_divergence'],
                'original_5d_score':original,'detection_5d_score':r5,'reduced_4d_score':r4,
                'warnings':row['metric_warnings']}
        scores.append(output)
    # Publish score records only; no curriculum order generated before user selects scoring policy.
    out.mkdir(parents=True,exist_ok=True)
    sf=out/'difficulty_scores.jsonl'
    with sf.open('w',encoding='utf8') as f:
        for r in scores:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
    groups=collections.defaultdict(list)
    for r in scores:
        groups[(r['corpus'],r['label'])].append(r)
    group_stats={}
    for (corpus,label), recs in sorted(groups.items()):
        vals=[r['reduced_4d_score'] for r in recs if r['reduced_4d_score'] is not None]
        group_stats[f'{corpus}/label={label}']={'count':len(recs),'scored_4d':len(vals),
            'missing_4d':len(recs)-len(vals),'mean_4d':statistics.mean(vals) if vals else None,
            'p05_4d':percentile(vals,.05),'median_4d':percentile(vals,.5),'p95_4d':percentile(vals,.95)}
    audit={'status':'AUDIT_ONLY_NO_CURRICULUM_SELECTED','created_chicago':datetime.datetime.now(ZoneInfo('America/Chicago')).isoformat(),
           'prepared_dir':str(prepared),'train_sha256':sha256(train_file),'validation_sha256':sha256(valid_file),
           'clean_data_sha256':hashes,'training_pairs':len(train),'validation_pairs':len(valid),
           'unique_training_pairs':len(set(map(key,train))),
           'counts':{f'{a}/label={b}':n for (a,b),n in sorted(counts.items())},
           'normalization_training_only_p05_p95':bounds,'group_stats':group_stats,
           'parse_failures':{f'{a}/{b}':n for (a,b),n in missing.items()},
           'parser_mode_counts':dict(collections.Counter(mode for r in scores for mode in r['parse_modes'])),
           'sensitivity_autojunk':compute_similarity_sensitivity(scores),
           'autojunk_sensitivity_method':'deterministic SHA256 hash modulo 12; only AST lengths <=1500; sampled sensitivity, not entire corpus',
           'metrics_status':{'original_5d':'UNAVAILABLE_NO_ARCH_AND_UNVALIDATED_SEMANTIC_TAXONOMY',
                             'detection_5d':'UNAVAILABLE_NO_ARCH','reduced_4d':'PROXY_REQUIRES_REVIEW'},
           'limitations':['DF proxy measures identifier diversity, not true def-use/external coupling.',
                          'Architecture unavailable without verified per-function file paths.',
                          'Similarity and structural divergence share AST features.',
                          'Python negative candidates may contain mislabeled clones.',
                          'Incomplete/invalid AST scores remain NULL; no hidden fallback or pair deletion.',
                          'This audit does not prove measured difficulty predicts model loss or errors.',
                          'Primary score uses SequenceMatcher autojunk=True for v1 comparability; autojunk=False is sensitivity-only.',
                          'Java wrapper AST parsing is a heuristic; method/class wrapped features require manual audit.',
                          'No M3 training or curriculum ordering has been performed.']}
    dump(out/'difficulty_audit.json',audit)
    dump(out/'manifest.json',{'component':'D-04a/b','version':'v2','source_manifest_sha256':sha256(prepared/'manifest.json'),
         'train_sha256':sha256(train_file),'validation_sha256':sha256(valid_file),
         'difficulty_scores_sha256':sha256(sf),'difficulty_audit_sha256':sha256(out/'difficulty_audit.json'),
         'weights_original_5d':W_ORIG,'weights_detection_5d_candidate':W_5D,
         'weights_reduced_4d_candidate':W_4D,'status':'AUDIT_ONLY'})
    print(json.dumps({'status':'AUDIT_ONLY','out':str(out),'pairs':len(scores),
                      'valid_4d':sum(r['reduced_4d_score'] is not None for r in scores),
                      'scores_sha256':sha256(sf)},sort_keys=True),flush=True)


def do_verify(args):
    root=Path(args.root).resolve(); prepared=Path(args.prepared)
    if not prepared.is_absolute():prepared=root/prepared
    manifest=json.loads((prepared/'manifest.json').read_text(encoding='utf8'))
    train=prepared/'train.jsonl';valid=prepared/'validation.jsonl'
    for split,path in [('train',train),('validation',valid)]:
        if sha256(path)!=manifest['sample'][split]['sha256']:raise ValueError(f'{split} checksum mismatch')
    train_rows=list(read_jsonl(train));valid_rows=list(read_jsonl(valid))
    print(json.dumps({'status':'PREPARED_VERIFIED','train':len(train_rows),'validation':len(valid_rows),
                      'train_sha256':sha256(train),'validation_sha256':sha256(valid)},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify','score'])
    parser.add_argument('--root',default='.')
    parser.add_argument('--prepared',default=DEFAULT_PREPARED)
    parser.add_argument('--output',default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    if args.command=='verify':do_verify(args)
    else:do_score(args)

if __name__=='__main__':main()
