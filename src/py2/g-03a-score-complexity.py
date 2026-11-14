#!/usr/bin/env python3
"""G-03a v1: D-04 Reduced-4D scoring for frozen GPT-OSS G-02 mixed pairs.

Mirrors reviewed F-01e v4 D-04 v2 metrics and D-04 v5 median fallback.
Reads original F-01c effective code, verifies G-02 row and SHA256 lineage.
Never alters prepared data; never trains; writes evidence in a new run only.
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
VERSION='v1'

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


# G-03a v1 independent GPT-OSS source-pinned complexity scorer.
# Mathematical components above are mirrored from the reviewed F-01e v4
# implementation of D-04 v2/v5: do not silently change their definitions.
import importlib.metadata
import sys


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def write_exclusive(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def canonical_hash(obj):
    return sha_bytes(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                separators=(',', ':')).encode('utf-8'))


def timestamp():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def cli():
    p=argparse.ArgumentParser(description='G-03a v1: independently score G-02 v2 frozen mixed pairs, no training')
    p.add_argument('--mode', choices=['score', 'verify'], required=True)
    for name in ('root','g02-pipeline','f01c-prepared','m1-adapter','run-dir','log-file'):
        p.add_argument('--'+name,required=True,type=Path)
    for name in ('version','model-id','model-revision','g02-prepared-manifest-sha256',
                 'g02-train-sha256','g02-validation-sha256','g02-standard-order-sha256','g02-config-fingerprint',
                 'f01c-manifest-sha256','f01c-train-sha256','f01c-validation-sha256',
                 'm1-adapter-sha256','component-policy','score-policy','score-provenance',
                 'ast-fallback-policy','parser-version-policy','normalization-policy',
                 'similarity-policy','tree-sitter-version','tree-sitter-java-version',
                 'tree-sitter-python-version','manual-label-review-policy','score-field'):
        p.add_argument('--'+name,required=True)
    for name in ('expected-train','expected-validation','expected-variants',
                 'max-seq-length','progress-interval','global-batch','optimizer-steps','seed'):
        p.add_argument('--'+name,required=True,type=int)
    for name in ('source-train','java-train','python-train','source-validation',
                 'java-validation','python-validation'):
        p.add_argument('--'+name,required=True,type=int)
    for name in ('percentile-low','percentile-high','score-tolerance'):
        p.add_argument('--'+name,required=True,type=float)
    p.add_argument('--score-weights',required=True,nargs=4,type=float)
    return p.parse_args()


def validate_config(a, check_parsers=True):
    require(a.version == VERSION, 'VERSION_MISMATCH')
    require(a.component_policy == 'D04_VERIFIED_EFFECTIVE_CODE_COMPONENTS', 'COMPONENT_POLICY_INVALID')
    require(a.score_policy == 'reduced-4d-externally-verified-v1', 'SCORE_POLICY_INVALID')
    require(a.score_field == 'reduced_4d_score', 'SCORE_FIELD_INVALID')
    require(a.ast_fallback_policy == 'd04-v5-corpus-label-median-explicit', 'FALLBACK_POLICY_INVALID')
    require(a.parser_version_policy == 'd04-v2-tree-sitter-wrap', 'PARSER_POLICY_INVALID')
    require(a.normalization_policy == 'd04-v2-corpus-train-p05-p95', 'NORMALIZATION_POLICY_INVALID')
    require(a.similarity_policy == 'd04-v2-sequencematcher-autojunk-true', 'SIMILARITY_POLICY_INVALID')
    require(a.manual_label_review_policy == 'PENDING_AND_NOT_INDEPENDENTLY_VERIFIED', 'LABEL_REVIEW_POLICY_MISMATCH')
    require(a.score_provenance == 'G03A_V1_RECOMPUTED_D04_EFFECTIVE_CODE_ON_G02_FROZEN', 'SCORE_PROVENANCE_INVALID')
    for key in ('g02_prepared_manifest_sha256','g02_train_sha256','g02_validation_sha256',
                'g02_standard_order_sha256',
                'f01c_manifest_sha256','f01c_train_sha256','f01c_validation_sha256',
                'm1_adapter_sha256','g02_config_fingerprint'):
        value=getattr(a,key)
        require(isinstance(value,str) and len(value)==64 and
                all(c in '0123456789abcdef' for c in value), 'INVALID_SHA256 '+key)
    require(len(a.model_revision)==40 and all(c in '0123456789abcdef' for c in a.model_revision),
            'INVALID_MODEL_REVISION')
    require(a.expected_train==a.source_train+a.java_train+a.python_train==
            a.global_batch*a.optimizer_steps, 'TRAIN_BUDGET_MISMATCH')
    require(a.expected_validation==a.source_validation+a.java_validation+a.python_validation,
            'VALIDATION_BUDGET_MISMATCH')
    require(all(v>0 and v%2==0 for v in (a.source_train,a.java_train,a.python_train,
               a.source_validation,a.java_validation,a.python_validation)), 'BAD_DOMAIN_COUNT')
    require(all(v>0 for v in (a.expected_variants,a.max_seq_length,a.progress_interval)), 'BAD_POSITIVE_CONFIG')
    require(0<=a.percentile_low<a.percentile_high<=1, 'PERCENTILE_BOUNDS_INVALID')
    require(all(math.isfinite(w) and w>=0 for w in a.score_weights) and
            abs(sum(a.score_weights)-1)<1e-12, 'WEIGHTS_INVALID')
    require(math.isfinite(a.score_tolerance) and 0<a.score_tolerance<=1e-4, 'TOLERANCE_INVALID')
    require(a.root.resolve() == Path(a.root).resolve(), 'ROOT_RESOLUTION_FAILED')
    if check_parsers:
        for pkg, expected in (('tree-sitter',a.tree_sitter_version),
                              ('tree-sitter-java',a.tree_sitter_java_version),
                              ('tree-sitter-python',a.tree_sitter_python_version)):
            try: actual=importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError as e:
                raise RuntimeError(f'REQUIRED_PARSER_MISSING {pkg}=={expected}') from e
            require(actual==expected, f'PARSER_VERSION_MISMATCH {pkg}: {actual}!={expected}')


def bound(path, expected, what):
    require(path.is_file(), f'MISSING_FILE {what}: {path}')
    actual=sha(path)
    require(actual==expected, f'SHA256_MISMATCH {what}: actual={actual} expected={expected}')
    return actual


def inputs(a):
    """Read-only binding to both the original F-01c pair bytes and GPT-OSS G-02 rows."""
    gprep=a.g02_pipeline/'00-prepare'/'prepared'
    gp={'train':gprep/'train.jsonl','validation':gprep/'validation.jsonl','manifest':gprep/'manifest.json'}
    fp={'train':a.f01c_prepared/'train.jsonl','validation':a.f01c_prepared/'validation.jsonl',
        'manifest':a.f01c_prepared/'manifest.json'}
    bound(gprep/'order.jsonl',a.g02_standard_order_sha256,'G02_STANDARD_ORDER')
    for group,paths,pins in (('G02',gp,(a.g02_train_sha256,a.g02_validation_sha256,a.g02_prepared_manifest_sha256)),
                             ('F01C',fp,(a.f01c_train_sha256,a.f01c_validation_sha256,a.f01c_manifest_sha256))):
        for kind,digest in zip(('train','validation','manifest'),pins):
            bound(paths[kind],digest,group+'_'+kind)
    bound(a.m1_adapter,a.m1_adapter_sha256,'G01D_M1_WEIGHTS')
    m=json.loads(gp['manifest'].read_text(encoding='utf-8'))
    f=json.loads(fp['manifest'].read_text(encoding='utf-8'))
    require(m.get('status')=='G02_V2_PREPARE_PASS_NOT_TRAINED', 'G02_PREPARED_STATUS_INVALID')
    require(m.get('config_fingerprint')==a.g02_config_fingerprint, 'G02_FINGERPRINT_MISMATCH')
    require(m.get('model',{}).get('model_id')==a.model_id and
            m.get('model',{}).get('revision')==a.model_revision, 'G02_MODEL_IDENTITY_MISMATCH')
    require(m.get('m1_initial_adapter',{}).get('adapter_sha256')==a.m1_adapter_sha256,
            'G02_M1_INITIALIZATION_MISMATCH')
    require(m.get('files',{}).get('train',{}).get('sha256')==a.g02_train_sha256 and
            m['files']['validation']['sha256']==a.g02_validation_sha256 and
            m['files']['order']['sha256']==a.g02_standard_order_sha256 and
            m['files']['order']['rows']==a.expected_train, 'G02_PREPARED_MANIFEST_HASH_LINK_INVALID')
    require(m.get('frozen_mixed',{}).get('f01c_frozen_sha256')==
            {'train':a.f01c_train_sha256,'validation':a.f01c_validation_sha256,
             'manifest':a.f01c_manifest_sha256}, 'G02_F01C_FROZEN_PROVENANCE_INVALID')
    require(m['frozen_mixed'].get('m1_source_pair_identity')=='TRAIN_EXACT_MATCH',
            'M1_SOURCE_IDENTITY_NOT_VERIFIED')
    require(m['frozen_mixed'].get('manual_label_review_status')==a.manual_label_review_policy,
            'LABEL_REVIEW_STATUS_UNEXPECTED')
    require(m.get('domain_counts',{}).get('train')==
            {'java_source':a.source_train,'java_mixed':a.java_train,'python_mixed':a.python_train},
            'G02_TRAIN_COUNTS_MISMATCH')
    require(m['domain_counts'].get('validation')==
            {'java_source':a.source_validation,'java_mixed':a.java_validation,
             'python_mixed':a.python_validation}, 'G02_VALIDATION_COUNTS_MISMATCH')
    require(m.get('training_budget',{}).get('optimizer_steps')==a.optimizer_steps and
            m['training_budget'].get('examples')==a.expected_train, 'G02_BUDGET_MISMATCH')
    require(m.get('effective_parameters',{}).get('seed')==a.seed and
            m['effective_parameters'].get('max_seq_length')==a.max_seq_length,
            'G02_SEED_OR_SEQUENCE_LENGTH_DRIFT')
    require(m.get('tokenizer_identity',{}).get('no_input_truncation') is True, 'G02_TOKEN_TRUNCATION_POLICY_INVALID')
    require(f.get('sample',{}).get('train',{}).get('sha256')==a.f01c_train_sha256 and
            f['sample'].get('validation',{}).get('sha256')==a.f01c_validation_sha256,
            'F01C_MANIFEST_SAMPLE_LINK_MISMATCH')
    require(f['sample']['train'].get('total')==a.expected_train and
            f['sample']['validation'].get('total')==a.expected_validation, 'F01C_COUNT_MISMATCH')
    require(f.get('status')=='PREPARED_PENDING_LABEL_REVIEW', 'F01C_LABEL_STATUS_DRIFT')
    # This M3 branch must use a common M1 ancestor, not the trained M2 adapter.
    final=json.loads((a.g02_pipeline/'pipeline_result.json').read_text(encoding='utf-8'))
    require(final.get('status')=='G02_V2_FULL_M2_TRAINING_COMPLETE', 'G02_M2_NOT_COMPLETE')
    summary=json.loads((a.g02_pipeline/'03-train'/'training_summary.json').read_text(encoding='utf-8'))
    require(summary.get('status')=='G02_V2_M2_STANDARD_MIXED_TRAINING_COMPLETE' and
            summary.get('optimizer_steps')==a.optimizer_steps and
            summary.get('train_sample_exposures')==a.expected_train and
            summary.get('initial_m1_adapter_sha256')==a.m1_adapter_sha256 and
            summary.get('prepared_manifest_sha256')==a.g02_prepared_manifest_sha256,
            'G02_TRAINING_SUMMARY_INVALID')
    gpins=m['frozen_mixed']['code_sha256']
    codes={}; paths={}
    for g in CORPORA:
        rawpath=Path(m['frozen_mixed']['code_paths'][g])
        path=resolve_path(str(rawpath),a.root.resolve())
        original=resolve_path(str(f['dataset']['paths'][g]['data.jsonl']),a.root.resolve())
        require(path==original, 'G02_F01C_CORPUS_PATH_DIFFER '+g)
        expected=f['dataset']['sha256'][g]['data.jsonl']
        require(expected==gpins[g], 'G02_F01C_CORPUS_PIN_DIFFER '+g)
        bound(path,expected,'CORPUS_'+g)
        paths[g]=str(path)
        codes[g]=load_codes(path)
    # Compare G02 records to exactly the F01C bytes, in the same order.
    train=[];seen=set();count=collections.Counter();variants=0
    src=rows(fp['train']);dst=rows(gp['train'])
    from itertools import zip_longest
    for matched in zip_longest(src,dst):
        require(all(item is not None for item in matched), 'G02_F01C_ROW_COUNT_DIFFER')
        (ix,fr,fraw),(jx,gr,graw)=matched
        require(ix==jx, 'G02_F01C_ROW_INDEX_DRIFT')
        k=identity(fr)
        require(identity(gr)==k and k not in seen, 'G02_F01C_IDENTITY_OR_ORDER_DRIFT '+str(ix))
        seen.add(k)
        require(gr.get('original_row_sha256')==canonical_hash(fr), 'G02_F01C_CANONICAL_ROW_HASH_DRIFT '+str(ix))
        require(type(gr.get('n_tokens')) is int and 0<gr['n_tokens']<=a.max_seq_length,
                'G02_TOKEN_LENGTH_INVALID '+str(ix))
        require(type(fr.get('tokens')) is int and fr['tokens']>0, 'F01C_TOKEN_LENGTH_INVALID '+str(ix))
        g,x,y,label=k
        try: oa=codes[g][x];ob=codes[g][y]
        except KeyError as exc:raise RuntimeError('F01C_CODE_ID_NOT_FOUND '+str(k)) from exc
        ca=fr.get('code_a_override',oa);cb=fr.get('code_b_override',ob)
        require(isinstance(ca,str) and ca.strip() and isinstance(cb,str) and cb.strip(),
                'EFFECTIVE_CODE_INVALID '+str(ix))
        require(gr.get('code1_sha256')==sha_bytes(ca.encode('utf-8')) and
                gr.get('code2_sha256')==sha_bytes(cb.encode('utf-8')),
                'G02_EFFECTIVE_CODE_SHA256_MISMATCH '+str(ix))
        for field in ('code_a_override','code_b_override','origin'):
            require(fr.get(field)==gr.get(field), 'G02_F01C_OVERRIDE_ORIGIN_MISMATCH '+str(ix)+' '+field)
        override=('code_a_override' in fr or 'code_b_override' in fr)
        if override:
            require(g=='java_mixed' and label==0 and fr.get('origin')=='structural-v13',
                    'STRUCTURAL_VARIANT_PROVENANCE_INVALID '+str(ix))
            def inst(id,body,original):
                return id if body==original else id+'@trunc-'+sha_bytes(body.encode('utf-8'))
            require(fr.get('code_instance_id1')==inst(x,ca,oa) and
                    fr.get('code_instance_id2')==inst(y,cb,ob) and
                    (ca,cb)!=(oa,ob), 'F01C_STRUCTURAL_INSTANCE_MISMATCH '+str(ix))
            variants+=1
        count[g,label]+=1
        train.append({'index':ix,'key':k,'f01c_row_sha256':fraw,'g02_row_sha256':graw,
                      'g02_original_row_sha256':gr['original_row_sha256'],
                      'code_a':ca,'code_b':cb,
                      'effective_code_a_sha256':sha_bytes(ca.encode('utf-8')),
                      'effective_code_b_sha256':sha_bytes(cb.encode('utf-8')),
                      'is_variant':override})
    require(len(train)==a.expected_train and variants==a.expected_variants,
            f'TRAIN_COUNT_VARIANT_MISMATCH count={len(train)} variants={variants}')
    for g,expect in (('java_source',a.source_train),('java_mixed',a.java_train),
                     ('python_mixed',a.python_train)):
        require(count[g,0]==count[g,1]==expect//2, 'CLASS_BALANCE_INVALID '+g)
    return train, {'g02_manifest':m,'f01c_manifest':f,'code_paths':paths,'code_sha256':gpins,
                   'variant_count':variants,'counts':{g:{str(l):count[g,l] for l in (0,1)} for g in CORPORA}}


def compute_components(train, a, parser_factory=None, feature_fn=None):
    parser_factory = parser_factory or parser_for
    feature_fn = feature_fn or parse_features
    parsers={'java':parser_factory('java'),'python':parser_factory('python')}
    cache={}; raw=[]; stats=collections.defaultdict(list); fail=[]
    for r in train:
        g=r['key'][0]; lang='python' if g=='python_mixed' else 'java'
        def feature(body):
            k=(lang,sha_bytes(body.encode('utf-8')))
            if k not in cache:cache[k]=feature_fn(parsers[lang],body,lang)
            return cache[k]
        fa=feature(r['code_a']);fb=feature(r['code_b'])
        metric=pair_metrics(fa,fb)
        if metric is None:
            fail.append({'row_index':r['index'],'pair':list(r['key']),
                         'parse_modes':[fa['parse_mode'],fb['parse_mode']]})
        else:
            for field in ('cf_raw','df_raw','length_raw'):stats[g,field].append(metric[field])
        raw.append(metric)
        if (r['index']+1)%a.progress_interval==0:
            print(json.dumps({'event':'G03A_AST_PROGRESS','processed':r['index'],
                  'total':a.expected_train,'at_chicago':timestamp()}),flush=True)
    bounds={g:{f:[percentile(stats[g,f],a.percentile_low),
                   percentile(stats[g,f],a.percentile_high)]
                for f in ('cf_raw','df_raw','length_raw')} for g in CORPORA}
    grouped=collections.defaultdict(list)
    components=[]
    for r,metric in zip(train,raw):
        g,_,_,label=r['key'];c=None
        if metric is not None:
            c={'d_sim':1-metric['sim'] if label==1 else metric['sim'],
               'd_cf':normalize(metric['cf_raw'],bounds[g]['cf_raw']),
               'd_len':normalize(metric['length_raw'],bounds[g]['length_raw']),
               'd_df':normalize(metric['df_raw'],bounds[g]['df_raw'])}
            require(all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1
                        for v in c.values()), 'COMPONENT_OUT_OF_BOUNDS '+str(r['index']))
            grouped[g,label].append(c)
        components.append(c)
    def weighted(c):return sum(w*c[k] for k,w in zip(('d_sim','d_cf','d_len','d_df'),a.score_weights))
    medians={}
    for g in CORPORA:
        for label in (0,1):
            options=grouped[g,label]
            require(bool(options), f'NO_VALID_AST_GROUP {g}/{label}')
            ranked=sorted(enumerate(options),key=lambda x:(weighted(x[1]),x[0]))
            left=ranked[(len(ranked)-1)//2][1];right=ranked[len(ranked)//2][1]
            medians[g,label]={k:(left[k]+right[k])/2 for k in ('d_sim','d_cf','d_len','d_df')}
    comps=[];scores=[];proxies=[]
    for r,c in zip(train,components):
        g,x,y,label=r['key'];imp=c is None
        if imp:
            c=medians[g,label]
            proxies.append({'row_index':r['index'],'pair':list(r['key'])})
        base={'row_index':r['index'],'corpus':g,'id1':x,'id2':y,'label':label,
              'source_row_sha256':r['f01c_row_sha256'],
              'g02_row_sha256':r['g02_row_sha256'],
              'g02_original_row_sha256':r['g02_original_row_sha256'],
              'effective_code_a_sha256':r['effective_code_a_sha256'],
              'effective_code_b_sha256':r['effective_code_b_sha256'],
              'imputed_proxy':imp,'score_provenance':a.score_provenance,
              'component_policy':a.component_policy}
        component_record={**base,**c}
        score_record={**component_record,'score_policy':a.score_policy,'reduced_4d_score':weighted(c)}
        comps.append(component_record);scores.append(score_record)
    return comps,scores,{'bounds':bounds,'ast_parse_failures':fail,
                         'median_fallback_pairs':proxies,
                         'cached_effective_code_functions':len(cache)}


def jsonl(path, items):
    with path.open('x',encoding='utf-8') as f:
        for obj in items:
            f.write(json.dumps(obj,sort_keys=True,ensure_ascii=False,
                               separators=(',',':'),allow_nan=False)+'\n')


def effective(a):
    return {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}


def score(a):
    validate_config(a,check_parsers=True)
    require(not a.run_dir.exists(), 'OUTPUT_ALREADY_EXISTS '+str(a.run_dir))
    train,context=inputs(a)
    comps,scores,metrics=compute_components(train,a)
    # Run directory is created only after full scoring succeeds.
    a.run_dir.mkdir(parents=True,exist_ok=False)
    component_file=a.run_dir/'verified_d04_components.jsonl'
    score_file=a.run_dir/'verified_scores.jsonl'
    jsonl(component_file,comps)
    jsonl(score_file,scores)
    comp_sha=sha(component_file);score_sha=sha(score_file)
    report={'version':VERSION,'status':'G03A_V1_D04_COMPONENTS_RECOMPUTED',
            'created_at_chicago':timestamp(),'score_provenance':a.score_provenance,
            'source_train_rows':len(train),'f01c_frozen_train_sha256':a.f01c_train_sha256,
            'g02_prepared_train_sha256':a.g02_train_sha256,
            'g02_prepared_manifest_sha256':a.g02_prepared_manifest_sha256,
            'g02_config_fingerprint':a.g02_config_fingerprint,
            'effective_code_sha256_required':True,'variants_count':context['variant_count'],
            'domain_label_counts':context['counts'],
            'ast_parse_failures':metrics['ast_parse_failures'],
            'median_fallback_pairs':metrics['median_fallback_pairs'],
            'median_fallback_values_are_imputed_proxies':True,
            'normalization_training_only_p05_p95':metrics['bounds'],
            'code_sha256':context['code_sha256'],
            'component_file_sha256':comp_sha,
            'cached_effective_code_functions':metrics['cached_effective_code_functions'],
            'method':'D-04 v2 tree-sitter parse wrappers; lexical Jaccard; AST type SequenceMatcher autojunk=True; harmonic similarity; branch count CF; AST leaf length; identifier diversity DF proxy; D-04 v5 corpus-label median composite fallback',
            'limitations':['DF is identifier diversity proxy, not true def-use',
                           'AST parser wrappers are heuristics, not verified semantic equivalence',
                           'Fallback components are imputed, not independently measured',
                           'Complexity scores are deterministic heuristics, not validated task difficulty',
                           'Pending Java-mixed negative label review is inherited, not resolved',
                           'No M3 order selection or GPU training performed']}
    write_exclusive(a.run_dir/'component_audit.json',report)
    values=[r['reduced_4d_score'] for r in scores]
    summary={'count':len(scores),'minimum_score':min(values),'median_score':statistics.median(values),
             'p95_score':percentile(values,0.95),'maximum_score':max(values),
             'average_score':statistics.fmean(values),
             'imputed_pairs':len(metrics['median_fallback_pairs']),
             'domain_label_counts':context['counts']}
    write_exclusive(a.run_dir/'complexity_summary.json',summary)
    manifest={'version':VERSION,'status':'G03A_V1_COMPLEXITY_SCORE_PASS_NOT_M3_TRAINING',
              'created_at_chicago':timestamp(),'model_id':a.model_id,'model_revision':a.model_revision,
              'g02_pipeline':str(a.g02_pipeline.resolve()),
              'm1_initial_adapter':str(a.m1_adapter.resolve()),
              'm1_initial_adapter_sha256':a.m1_adapter_sha256,
              'g02_prepared_manifest_sha256':a.g02_prepared_manifest_sha256,
              'g02_prepared_train_sha256':a.g02_train_sha256,
              'g02_prepared_validation_sha256':a.g02_validation_sha256,
              'g02_standard_mixed_order_sha256':a.g02_standard_order_sha256,
              'f01c_frozen_manifest_sha256':a.f01c_manifest_sha256,
              'f01c_frozen_train_sha256':a.f01c_train_sha256,
              'f01c_frozen_validation_sha256':a.f01c_validation_sha256,
              'g02_config_fingerprint':a.g02_config_fingerprint,
              'component_policy':a.component_policy,'score_policy':a.score_policy,
              'score_provenance':a.score_provenance,'weights':a.score_weights,
              'percentile_low':a.percentile_low,'percentile_high':a.percentile_high,
              'ast_fallback_policy':a.ast_fallback_policy,
              'normalization_policy':a.normalization_policy,
              'similarity_policy':a.similarity_policy,
              'manual_label_review_status':a.manual_label_review_policy,
              'train_count':len(scores),'validation_count':a.expected_validation,
              'variants_count':context['variant_count'],
              'imputed_pairs':len(metrics['median_fallback_pairs']),
              'verified_d04_components_sha256':comp_sha,
              'verified_scores_sha256':score_sha,
              'component_audit_sha256':sha(a.run_dir/'component_audit.json'),
              'complexity_summary_sha256':sha(a.run_dir/'complexity_summary.json'),
              'effective_parameters':effective(a),
              'approved_for_training':False,
              'train_order_generated':False,
              'validation_data_modified':False,
              'm2_adapter_used_as_initial_m3_weights':False}
    write_exclusive(a.run_dir/'score_manifest.json',manifest)
    print(json.dumps({'status':manifest['status'],'run':str(a.run_dir),
       'train_pairs':len(scores),'validation_pairs_unchanged':a.expected_validation,
       'components_sha256':comp_sha,'scores_sha256':score_sha,
       'imputed_pairs':len(metrics['median_fallback_pairs']),
       'approved_for_m3_training':False,'at_chicago':timestamp()}),flush=True)


def verify(a):
    validate_config(a,check_parsers=False)
    require(a.run_dir.is_dir(), 'SCORE_RUN_MISSING')
    meta=json.loads((a.run_dir/'score_manifest.json').read_text(encoding='utf-8'))
    require(meta['status']=='G03A_V1_COMPLEXITY_SCORE_PASS_NOT_M3_TRAINING' and
            meta.get('approved_for_training') is False, 'SCORE_RUN_STATUS_INVALID')
    # Verify the exact same Shell configuration except operation-specific run path.
    prior=meta['effective_parameters'];current=effective(a)
    allowed={'mode','run_dir','log_file'}
    changed=[k for k in prior if k not in allowed and prior[k]!=current.get(k)]
    require(not changed, 'SCORE_CONFIG_DRIFT '+repr(changed))
    train,_=inputs(a)
    require(meta['train_count']==len(train)==a.expected_train, 'SCORE_ROW_COUNT_DRIFT')
    component=a.run_dir/'verified_d04_components.jsonl'
    scored=a.run_dir/'verified_scores.jsonl'
    bound(component,meta['verified_d04_components_sha256'],'D04_COMPONENTS')
    bound(scored,meta['verified_scores_sha256'],'REDUCED4D_SCORES')
    bound(a.run_dir/'component_audit.json',meta['component_audit_sha256'],'COMPONENT_AUDIT')
    bound(a.run_dir/'complexity_summary.json',meta['complexity_summary_sha256'],'SCORE_SUMMARY')
    n=0
    from itertools import zip_longest
    for items in zip_longest(rows(component),rows(scored),train):
        require(all(v is not None for v in items), 'OUTPUT_RECORD_LENGTH_MISMATCH')
        (_,c,_),(_,s,_),r=items
        n+=1
        g,x,y,label=r['key']
        require(c['row_index']==s['row_index']==r['index'] and
                tuple(s[k] for k in ('corpus','id1','id2','label'))==(g,x,y,label),
                'OUTPUT_IDENTITY_MISMATCH '+str(n))
        require(c['g02_row_sha256']==s['g02_row_sha256']==r['g02_row_sha256'] and
                c['source_row_sha256']==s['source_row_sha256']==r['f01c_row_sha256'] and
                s['effective_code_a_sha256']==r['effective_code_a_sha256'] and
                s['effective_code_b_sha256']==r['effective_code_b_sha256'],
                'OUTPUT_PROVENANCE_DRIFT '+str(n))
        for field in ('d_sim','d_cf','d_len','d_df'):
            v=s[field]
            require(isinstance(v,(int,float)) and not isinstance(v,bool) and
                    math.isfinite(v) and 0<=v<=1 and v==c[field],
                    'OUTPUT_COMPONENT_INVALID '+field+' '+str(n))
        expected=sum(w*s[k] for k,w in zip(('d_sim','d_cf','d_len','d_df'),a.score_weights))
        require(math.isclose(expected,s[a.score_field],rel_tol=0,abs_tol=a.score_tolerance),
                'OUTPUT_SCORE_ARITHMETIC_MISMATCH '+str(n))
    require(n==a.expected_train, 'OUTPUT_TOTAL_INCORRECT')
    print(json.dumps({'event':'G03A_V1_SCORE_VERIFY_PASS','run':str(a.run_dir),
        'train_pairs':n,'scores_sha256':meta['verified_scores_sha256'],
        'at_chicago':timestamp()}),flush=True)


def main():
    a=cli()
    try:
        if a.mode=='score':score(a)
        else:verify(a)
    except Exception as e:
        print(json.dumps({'event':'G03A_FAIL','error_type':type(e).__name__,
                          'error':str(e),'at_chicago':timestamp()}),flush=True)
        raise

if __name__=='__main__':main()
