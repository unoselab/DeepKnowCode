#!/usr/bin/env python3
"""F-01e v4: D-04 v2-equivalent AST difficulty on frozen F-01c effective code.

Reuses D-04 v2 metric definitions, normalization and v5 explicit median fallback.
Never alters prepared data; writes evidence to a fresh directory only.
"""
import argparse
import collections
import difflib
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

CORPORA=('java_source','java_mixed','python_mixed')
BRANCH={'java':{'if_statement','for_statement','enhanced_for_statement','while_statement','do_statement','catch_clause','switch_expression','switch_statement','ternary_expression'},'python':{'if_statement','for_statement','while_statement','except_clause','conditional_expression','boolean_operator'}}
COMMENTS={'comment','line_comment','block_comment'}
IDENTIFIERS={'identifier'}
VERSION='v4'

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def rows(p):
    with Path(p).open('rb') as f:
        for i,raw in enumerate(f):
            if not raw.strip():raise ValueError(f'EMPTY_RECORD {p}:{i+1}')
            yield i,json.loads(raw),sha_bytes(raw.rstrip(b'\r\n'))
def identity(r):return (str(r['corpus']),str(r['id1']),str(r['id2']),int(r['label']))
def percentile(vals,pct):
    if not vals:return None
    s=sorted(vals);pos=(len(s)-1)*pct;i=int(pos);j=min(i+1,len(s)-1)
    return s[i]+(s[j]-s[i])*(pos-i)
def normalize(v,bounds):
    if v is None:return None
    lo,hi=bounds
    if hi<=lo:return 0.0
    return min(1.0,max(0.0,(v-lo)/(hi-lo)))
def jaccard(a,b):
    a,b=set(a),set(b)
    return len(a&b)/len(a|b) if a or b else 0.0
def harmonic(a,b):return 2*a*b/(a+b) if a+b else 0.0

def parser_for(language):
    try:
        from tree_sitter import Language,Parser
        if language=='java':import tree_sitter_java as grammar
        else:import tree_sitter_python as grammar
        return Parser(Language(grammar.language()))
    except ImportError as e:
        raise RuntimeError('D04_GRAMMAR_MISSING: pip install tree-sitter==0.25.2 tree-sitter-java==0.23.5 tree-sitter-python==0.25.0') from e

def parse_features(parser,code,language):
    # Mirrors D-04 v2 parse_features(): Java method/statement wrappers, AST
    # traversal excluding comments and artificial wrapper container nodes.
    original=parser.parse(code.encode('utf8'));tree=original;mode='direct'
    if original.root_node.has_error and language=='java':
        candidates=[('wrapped_class','class D04SyntheticClass {\n'+code+'\n}','class_body'),
                    ('wrapped_method','class D04SyntheticClass { void d04SyntheticMethod() {\n'+code+'\n} }','block')]
        for name,wrapped,typ in candidates:
            test=parser.parse(wrapped.encode('utf8'))
            if test.root_node.has_error:continue
            todo=[test.root_node];holder=None
            while todo:
                node=todo.pop()
                if node.type==typ:holder=node;break
                todo.extend(reversed(node.children))
            if holder is not None:tree=test;mode=name;break
    if tree.root_node.has_error:return {'error':'AST_HAS_ERROR','parse_mode':'failed'}
    if mode=='direct':roots=[tree.root_node]
    else:
        typ='class_body' if mode=='wrapped_class' else 'block'
        todo=[tree.root_node];holder=None
        while todo:
            node=todo.pop()
            if node.type==typ:holder=node;break
            todo.extend(reversed(node.children))
        roots=[n for n in holder.named_children if n.type!='ERROR'] if holder else []
    types=[];leaves=[];ids=[];branches=0;todo=list(reversed(roots))
    while todo:
        node=todo.pop()
        if node.type in COMMENTS:continue
        types.append(node.type)
        if node.type in BRANCH[language]:branches+=1
        if node.type in IDENTIFIERS and not node.children:ids.append(node.text.decode('utf8',errors='replace'))
        if not node.children:
            word=node.text.decode('utf8',errors='replace').strip()
            if word:leaves.append(word)
        else:todo.extend(reversed(node.children))
    return {'error':None,'parse_mode':mode,'types':types,'leaves':leaves,'identifiers':ids,'complexity':branches+1,'token_count':len(leaves)}

def pair_metrics(f1,f2):
    if f1['error'] or f2['error']:return None
    lexical=jaccard(f1['leaves'],f2['leaves'])
    structural=difflib.SequenceMatcher(None,f1['types'],f2['types'],autojunk=True).ratio()
    sim=harmonic(lexical,structural)
    cf=max(f1['complexity'],f2['complexity'])+0.5*abs(f1['complexity']-f2['complexity'])
    d1,d2=len(set(f1['identifiers'])),len(set(f2['identifiers']))
    return {'sim':sim,'cf_raw':cf,'df_raw':math.hypot(d1,d2),
            'length_raw':max(f1['token_count'],f2['token_count'])}

def resolve_path(raw,root):
    p=Path(raw)
    if p.is_file():return p.resolve()
    if not p.is_absolute() and (root/p).is_file():return (root/p).resolve()
    if 'data' in p.parts:
        p2=root/Path(*p.parts[p.parts.index('data'):])
        if p2.is_file():return p2.resolve()
    raise FileNotFoundError(f'PINNED_CORPUS_MISSING {raw}')

def load_codes(p):
    result={}
    for _,r,_ in rows(p):
        idx=str(r['idx']);code=r['func']
        if idx in result or not isinstance(code,str) or not code.strip():raise ValueError(f'CORPUS_CODE_INVALID {idx}')
        result[idx]=code
    return result

def cli():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','frozen-train','frozen-manifest','output-dir'):
        p.add_argument('--'+key,required=True,type=Path)
    for key in ('frozen-train-sha256','frozen-manifest-sha256','component-policy','score-provenance','ast-fallback-policy','parser-version-policy','normalization-policy','similarity-policy'):
        p.add_argument('--'+key,required=True)
    for key in ('expected-count','expected-variant-pairs','max-seq-length','progress-interval'):
        p.add_argument('--'+key,required=True,type=int)
    for key in ('percentile-low','percentile-high'):
        p.add_argument('--'+key,required=True,type=float)
    p.add_argument('--score-weights',nargs=4,type=float,required=True)
    return p.parse_args()

def main():
    a=cli();root=a.root.resolve()
    if a.component_policy!='D04_VERIFIED_EFFECTIVE_CODE_COMPONENTS' or a.ast_fallback_policy!='d04-v5-corpus-label-median-explicit':raise ValueError('POLICY_MISMATCH')
    if a.parser_version_policy!='d04-v2-tree-sitter-wrap' or a.normalization_policy!='d04-v2-corpus-train-p05-p95' or a.similarity_policy!='d04-v2-sequencematcher-autojunk-true':raise ValueError('D04_ALGORITHM_POLICY_MISMATCH')
    if len(a.score_weights)!=4 or any(not math.isfinite(w) or w<0 for w in a.score_weights) or abs(sum(a.score_weights)-1)>1e-12:raise ValueError('SCORE_WEIGHT_INVALID')
    if a.score_provenance in ('','UNVERIFIED') or not (0<=a.percentile_low<a.percentile_high<=1) or min(a.expected_count,a.max_seq_length,a.progress_interval)<1:raise ValueError('CONFIG_INVALID')
    if a.output_dir.exists():raise FileExistsError(f'OUTPUT_COLLISION {a.output_dir}')
    if sha(a.frozen_train)!=a.frozen_train_sha256 or sha(a.frozen_manifest)!=a.frozen_manifest_sha256:raise ValueError('FROZEN_SHA256_MISMATCH')
    m=json.loads(a.frozen_manifest.read_text())
    if m['sample']['train']['sha256']!=a.frozen_train_sha256:raise ValueError('MANIFEST_TRAIN_LINK_MISMATCH')
    if m['sample']['train']['total']!=a.expected_count:raise ValueError('MANIFEST_TRAIN_COUNT_MISMATCH')
    source=m['dataset']; paths=source['paths']; hashes=source['sha256']
    corpus={};corpus_hashes={}
    for group in CORPORA:
        path=resolve_path(paths[group]['data.jsonl'],root)
        digest=sha(path)
        if digest!=hashes[group]['data.jsonl']:raise ValueError('CORPUS_SHA256_MISMATCH '+group)
        corpus[group]=load_codes(path);corpus_hashes[group]=digest
    parsers={'java':parser_for('java'),'python':parser_for('python')}
    feature_cache={};data=[];stats=collections.defaultdict(list);seen=set();variants=[];failures=[]
    for i,r,linehash in rows(a.frozen_train):
        k=identity(r)
        if k in seen:raise ValueError('FROZEN_DUPLICATE '+str(k))
        seen.add(k)
        g,x,y,label=k
        if g not in CORPORA or label not in (0,1):raise ValueError('ROW_IDENTITY_INVALID')
        if type(r.get('tokens')) is not int or not 0<r['tokens']<=a.max_seq_length:raise ValueError('TOKEN_CAP_INVALID')
        try:orig_a=corpus[g][x];orig_b=corpus[g][y]
        except KeyError as e:raise ValueError('SOURCE_CODE_ID_MISSING '+str(k)) from e
        ca=r.get('code_a_override',orig_a);cb=r.get('code_b_override',orig_b)
        if not isinstance(ca,str) or not ca.strip() or not isinstance(cb,str) or not cb.strip():raise ValueError('EFFECTIVE_CODE_EMPTY '+str(k))
        is_variant=('code_a_override' in r or 'code_b_override' in r)
        if is_variant:
            if g!='java_mixed' or label!=0 or r.get('origin')!='structural-v13':raise ValueError('VARIANT_PROVENANCE_INVALID '+str(k))
            def instance(parent,code,original):return parent if code==original else parent+'@trunc-'+sha_bytes(code.encode('utf-8'))
            if r.get('code_instance_id1')!=instance(x,ca,orig_a) or r.get('code_instance_id2')!=instance(y,cb,orig_b):raise ValueError('VARIANT_INSTANCE_SHA_MISMATCH '+str(k))
            if (ca,cb)==(orig_a,orig_b):raise ValueError('VARIANT_NO_CHANGE '+str(k))
            variants.append({'pair':list(k),'effective_code_a_sha256':sha_bytes(ca.encode()),'effective_code_b_sha256':sha_bytes(cb.encode()),'changed_sides':r.get('changed_sides',[])})
        language='python' if g=='python_mixed' else 'java'
        def feat(code):
            key=(language,sha_bytes(code.encode('utf-8')))
            if key not in feature_cache:feature_cache[key]=parse_features(parsers[language],code,language)
            return feature_cache[key]
        fa,fb=feat(ca),feat(cb);raw=pair_metrics(fa,fb)
        if raw is None:failures.append({'pair':list(k),'parse_modes':[fa['parse_mode'],fb['parse_mode']]})
        else:
            for field in ('cf_raw','df_raw','length_raw'):stats[g,field].append(raw[field])
        data.append({'key':k,'raw':raw,'source_row_sha256':linehash,
            'effective_code_a_sha256':sha_bytes(ca.encode()),'effective_code_b_sha256':sha_bytes(cb.encode())})
        if (i+1)%a.progress_interval==0:print(json.dumps({'step':'D04_SCORE','processed':i+1,'total':a.expected_count,'chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat()}),flush=True)
    if len(data)!=a.expected_count or len(variants)!=a.expected_variant_pairs:raise ValueError(f'FROZEN_COUNT_OR_VARIANT_MISMATCH count={len(data)} variants={len(variants)}')
    bounds={g:{f:[percentile(stats[g,f],a.percentile_low),percentile(stats[g,f],a.percentile_high)] for f in ('cf_raw','df_raw','length_raw')} for g in CORPORA}
    groups=collections.defaultdict(list)
    for d in data:
        g,_,_,label=d['key'];raw=d['raw'];components=None
        if raw is not None:
            similarity=(1-raw['sim'] if label==1 else raw['sim'])
            components={'d_sim':similarity,'d_cf':normalize(raw['cf_raw'],bounds[g]['cf_raw']),
                'd_len':normalize(raw['length_raw'],bounds[g]['length_raw']),
                'd_df':normalize(raw['df_raw'],bounds[g]['df_raw'])}
            if any(v is None or not math.isfinite(v) or not 0<=v<=1 for v in components.values()):raise ValueError('COMPONENT_INVALID '+str(d['key']))
            groups[g,label].append(components)
        d['components']=components
    # D-04 v5 imputes the *composite* 4D score, not each component's median.
    # Reconstruct its exact median composite with the midpoint of the two
    # median-ranked observed component vectors (weighted sum is linear).
    # These vectors are explicit imputed proxies, NEVER direct AST measurements.
    median_vectors={}
    def weighted(v):return sum(w*v[k] for w,k in zip(a.score_weights,('d_sim','d_cf','d_len','d_df')))
    for group in ((g,l) for g in CORPORA for l in (0,1)):
        values=groups[group]
        if not values:raise ValueError('NO_VALID_AST_GROUP '+str(group))
        ranked=sorted(enumerate(values),key=lambda item:(weighted(item[1]),item[0]))
        mid=len(ranked)//2
        left=ranked[(len(ranked)-1)//2][1]
        right=ranked[mid][1]
        median_vectors[group]={field:(left[field]+right[field])/2 for field in ('d_sim','d_cf','d_len','d_df')}
    output=[];imputed=[]
    for d in data:
        k=d['key'];c=d['components']
        if c is None:
            c=median_vectors[k[0],k[3]]
            imputed.append(list(k))
        output.append({**dict(zip(('corpus','id1','id2','label'),k)),**c,
            'source_row_sha256':d['source_row_sha256'],'score_provenance':a.score_provenance,
            'component_policy':a.component_policy,
            'effective_code_a_sha256':d['effective_code_a_sha256'],
            'effective_code_b_sha256':d['effective_code_b_sha256']})
    # All validation completed before output creation; do not modify frozen input.
    a.output_dir.mkdir(parents=True,exist_ok=False)
    component_path=a.output_dir/'verified_d04_components.jsonl'
    with component_path.open('x',encoding='utf-8') as f:
        for row in output:f.write(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n')
    report={'status':'F01E_V4_D04_COMPONENTS_RECOMPUTED_WITH_EXPLICIT_PROXY_LIMITATIONS','version':VERSION,
        'created_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
        'score_provenance':a.score_provenance,'frozen_train_sha256':a.frozen_train_sha256,
        'frozen_manifest_sha256':a.frozen_manifest_sha256,'corpus_data_sha256':corpus_hashes,
        'component_file_sha256':sha(component_path),'pair_count':len(data),'variants_count':len(variants),
        'variant_pairs':variants,'ast_parse_failures':failures,'median_fallback_pairs':imputed,
        'fallback_policy':a.ast_fallback_policy,'fallback_components_are_imputed_proxies':True,'score_weights':a.score_weights,'normalization_training_only_p05_p95':bounds,
        'method':'D-04 v2 parse/lexical-Jaccard/AST-SequenceMatcher autojunk=True/harmonic-similarity/CF/identifier-diversity/AST-length; D-04 v5 explicit missing-score median within corpus+label',
        'limitations':['DF is identifier-diversity proxy, not true def-use','Java AST wrappers are heuristics','Median composite-score imputation follows D-04 v5; imputed components are midpoint vectors, not independently measured difficulty','This is reproducible code-derived evidence, not human or model-based validation of difficulty','No M3 GPU training performed']}
    with (a.output_dir/'component_audit.json').open('x',encoding='utf-8') as f:json.dump(report,f,sort_keys=True,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'status':report['status'],'component_file':str(component_path),'component_sha256':report['component_file_sha256'],'train_pairs':len(data),'variants':len(variants),'ast_fallbacks':len(imputed)},sort_keys=True),flush=True)

if __name__=='__main__':main()
