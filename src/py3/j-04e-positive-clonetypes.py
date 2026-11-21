#!/usr/bin/env python3
"""J-04e v1: Python AST clone-type census and full-coverage fixed positives.

Consumes an explicitly pinned, independently verified J-04d v4 ALL run.
Read-only upstream; no --copy-result (reserved for J-04f).
All operational/experiment parameters are required CLI arguments supplied by Shell.
"""
from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from zoneinfo import ZoneInfo

PAIR_COLUMNS = (
    'pair_id','fragment_a_id','repository_a','file_a','start_line_a','end_line_a',
    'commit_a','source_sha256_a','fragment_b_id','repository_b','file_b',
    'start_line_b','end_line_b','commit_b','source_sha256_b','same_repository',
    'nicad_similarity_raw','nicad_nlines','detector','config','xml_sha256',
    'upstream_pair_id','function_a','function_b','ni_parse_failed_a',
    'ni_parse_failed_b','label',
)
TYPE_COLUMNS = ('pair_id','fragment_a_id','fragment_b_id','detector','clone_subtype',
                'ast_combined_similarity','selection','selected_pair_id','reason')
T1 = 'TYPE_1_SIGNATURE'
T2C = 'TYPE_2_CONSISTENT_SIGNATURE'
T2B = 'TYPE_2_BLIND_SIGNATURE'
T3 = 'TYPE_3_STRUCTURAL_CANDIDATE'
UNDET = 'SEMANTIC_UNDETERMINED'
UNSCORED = 'STRUCTURAL_UNSCORED'
INELIGIBLE = 'INELIGIBLE'
CATEGORIES = (T1,T2C,T2B,T3,UNDET,UNSCORED,INELIGIBLE)
SELECTABLE = (T1,T2C,T2B,T3)
HEX64 = re.compile(r'^[0-9a-f]{64}$')
SUBJECT_RE = re.compile(r'^[a-z0-9][a-z0-9_.-]*__[a-z0-9][a-z0-9_.-]*$')
ID_FIELDS = {
    'Name': frozenset(('id',)), 'arg':frozenset(('arg',)),
    'Attribute':frozenset(('attr',)), 'FunctionDef':frozenset(('name',)),
    'AsyncFunctionDef':frozenset(('name',)), 'ClassDef':frozenset(('name',)),
    'alias':frozenset(('name','asname')), 'keyword':frozenset(('arg',)),
    'ExceptHandler':frozenset(('name',)), 'Global':frozenset(('names',)),
    'Nonlocal':frozenset(('names',)), 'MatchAs':frozenset(('name',)),
    'MatchStar':frozenset(('name',)), 'MatchMapping':frozenset(('rest',)),
    'MatchClass':frozenset(('kwd_attrs',)),
    'TypeVar':frozenset(('name',)), 'TypeVarTuple':frozenset(('name',)),
    'ParamSpec':frozenset(('name',)),
}
OUTPUT_FILES = frozenset((
    'positive_pairs.csv','fragments.jsonl','pair_clone_types.csv',
    'ast_clone_type_census.json','fixed_positive_clonepairs.jsonl',
    'details/fixed_positive_clonepairs.jsonl',
    'details/fixed_positive_clonepairs.manifest.json','manifest.json',
))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for piece in iter(lambda:f.read(1024*1024),b''):
            h.update(piece)
    return h.hexdigest()


def json_line(record):
    return json.dumps(record,ensure_ascii=True,sort_keys=True,separators=(',',':'))+'\n'


def stamp(a):
    return datetime.now(ZoneInfo(a.timezone)).isoformat(timespec='microseconds')


def safe_regular(path):
    require(path.is_file() and not path.is_symlink(),'REQUIRED_REGULAR_FILE '+str(path))
    return path


def json_read(path):
    return json.loads(safe_regular(path).read_text(encoding='utf-8'))


def json_new(path,record):
    with path.open('x',encoding='utf-8') as f:
        json.dump(record,f,indent=2,sort_keys=True,ensure_ascii=True)
        f.write('\n')


def jsonl_read(path):
    records=[]
    with safe_regular(path).open('r',encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            require(bool(line.strip()),'EMPTY_JSONL_LINE path='+str(path)+' line='+str(n))
            obj=json.loads(line)
            require(isinstance(obj,dict),'JSONL_NOT_OBJECT path='+str(path))
            records.append(obj)
    return records


def csv_read(path,fields):
    with safe_regular(path).open('r',encoding='utf-8',newline='') as f:
        rd=csv.DictReader(f,strict=True)
        require(tuple(rd.fieldnames or ())==fields,'CSV_SCHEMA_MISMATCH '+str(path))
        return list(rd)


def read_dataset(a):
    """Require exact verified J-04d run, never silently fall back to older data."""
    base=a.source_all_root
    ver=a.source_verify_root
    require(base.name==a.source_all_run_id and ver.name==a.source_verify_run_id,
            'PINNED_RUN_ID_MISMATCH')
    require(base.is_dir() and not base.is_symlink(),'SOURCE_J04D_ALL_RUN_MISSING '+str(base))
    require(ver.is_dir() and not ver.is_symlink(),'SOURCE_J04D_VERIFY_RUN_MISSING '+str(ver))
    summary=json_read(base/a.source_batch_name)
    verified=json_read(ver/a.source_verification_name)
    require(summary.get('schema')=='deepknowcode-j04d-batch-'+a.source_version and
            summary.get('experiment')==a.source_experiment and
            summary.get('version')==a.source_version and
            summary.get('status')=='COMPLETED' and summary.get('mode')=='all' and
            summary.get('run_id')==a.source_all_run_id and
            summary.get('subjects_completed')==a.expected_subjects and
            summary.get('positive_pairs_total')==a.expected_pairs and
            summary.get('unique_fragments_total')==a.expected_fragments and
            summary.get('input_files_total')==a.expected_input_files,
            'J04D_ALL_SUMMARY_IDENTITY_OR_COUNT_MISMATCH')
    require(verified.get('status')=='PASS' and
            verified.get('version')==a.source_version and
            verified.get('source_all')==str(base) and
            verified.get('verified_subjects')==a.expected_subjects and
            verified.get('positive_pairs')==a.expected_pairs and
            verified.get('source_copy_manifest_sha256')==summary.get('source_copy_manifest_sha256'),
            'J04D_INDEPENDENT_VERIFICATION_INVALID')
    entries=summary.get('subjects')
    v_entries=verified.get('subjects')
    require(isinstance(entries,list) and isinstance(v_entries,list) and
            len(entries)==len(v_entries)==a.expected_subjects,
            'J04D_SUBJECT_COUNTS_INVALID')
    by_name={r['subject']:r for r in entries}
    by_verify={r['subject']:r for r in v_entries}
    require(len(by_name)==len(by_verify)==a.expected_subjects and
            set(by_name)==set(by_verify) and
            all(SUBJECT_RE.fullmatch(n) for n in by_name),
            'J04D_SUBJECT_IDENTITY_INVALID')
    require(sum(r['positive_pairs'] for r in entries)==a.expected_pairs and
            sum(r['unique_fragments'] for r in entries)==a.expected_fragments and
            sum(r['input_files'] for r in entries)==a.expected_input_files,
            'J04D_SUBJECT_AGGREGATES_INVALID')
    for name,r in by_name.items():
        require(r['status']=='COMPLETED' and
                r['positive_pairs']==by_verify[name]['pairs'] and
                r['unique_fragments']==by_verify[name]['fragments'] and
                HEX64.fullmatch(r['manifest_sha256']),
                'J04D_VERIFIED_SUBJECT_MISMATCH '+name)
    subject_dir=base/'subjects'
    require(subject_dir.is_dir() and not subject_dir.is_symlink() and
            {p.name for p in subject_dir.iterdir()}==set(by_name),
            'J04D_SUBJECT_DIRECTORY_SET_MISMATCH')
    return summary,verified,by_name,sha_file(base/a.source_batch_name),sha_file(ver/a.source_verification_name)


def load_subject(a,subject,entry):
    directory=a.source_all_root/'subjects'/subject
    require(directory.is_dir() and not directory.is_symlink(),'J04D_SOURCE_SUBJECT_MISSING '+subject)
    require({p.name for p in directory.iterdir()}==set(('positive_pairs.csv','fragments.jsonl','manifest.json')) and
            all(not p.is_symlink() for p in directory.iterdir()),
            'J04D_SUBJECT_FILE_SET_INVALID '+subject)
    manifest_path=directory/'manifest.json'
    require(sha_file(manifest_path)==entry['manifest_sha256'],
            'J04D_SOURCE_MANIFEST_SHA256_MISMATCH '+subject)
    manifest=json_read(manifest_path)
    require(manifest.get('schema')=='deepknowcode-j04d-'+a.source_version and
            manifest.get('experiment')==a.source_experiment and
            manifest.get('version')==a.source_version and
            manifest.get('status')=='COMPLETED' and
            manifest.get('subject')==subject and
            manifest.get('positive_pairs')==entry['positive_pairs'] and
            manifest.get('unique_fragments')==entry['unique_fragments'] and
            manifest.get('input_hybrid_pairs')==entry['positive_pairs'] and
            manifest.get('source_files')==entry['input_files'] and
            manifest.get('input_version')==a.source_input_version,
            'J04D_SOURCE_MANIFEST_FIELDS_MISMATCH '+subject)
    for filename in ('positive_pairs.csv','fragments.jsonl'):
        require(sha_file(directory/filename)==manifest['files_sha256'].get(filename),
                'J04D_SOURCE_FILE_SHA256_MISMATCH '+subject+':'+filename)
    fragments={}
    for fragment in jsonl_read(directory/'fragments.jsonl'):
        fid=fragment.get('fragment_id')
        require(isinstance(fid,str) and bool(fid) and fid not in fragments and
                fragment.get('repository')==subject and fragment.get('language')==a.language and
                isinstance(fragment.get('code'),str) and
                isinstance(fragment.get('source_sha256'),str) and
                HEX64.fullmatch(fragment['source_sha256']),
                'J04D_FRAGMENT_SCHEMA_INVALID '+subject+':'+str(fid))
        require(sha_bytes(fragment['code'].encode(a.code_encoding,errors=a.code_errors))==fragment['source_sha256'],
                'J04D_FRAGMENT_CODE_SHA256_MISMATCH '+subject+':'+fid)
        fragments[fid]=fragment
    require(len(fragments)==entry['unique_fragments'],'J04D_FRAGMENT_COUNT_MISMATCH '+subject)
    rows=csv_read(directory/'positive_pairs.csv',PAIR_COLUMNS)
    require(len(rows)==entry['positive_pairs'],'J04D_PAIR_COUNT_MISMATCH '+subject)
    pair_ids=set(); references=set(); detectors=Counter()
    for row in rows:
        pid=row['pair_id']
        require(pid not in pair_ids and bool(pid),'J04D_DUPLICATE_PAIR_ID '+subject+':'+pid)
        pair_ids.add(pid)
        require(row['repository_a']==row['repository_b']==subject and
                row['label']==str(a.positive_label) and row['same_repository']=='1' and
                row['detector'] in a.allowed_detectors,
                'J04D_PAIR_METADATA_INVALID '+subject+':'+pid)
        detectors[row['detector']]+=1
        fa=fragments.get(row['fragment_a_id']);fb=fragments.get(row['fragment_b_id'])
        require(fa is not None and fb is not None and fa['fragment_id']!=fb['fragment_id'],
                'J04D_PAIR_FRAGMENT_MISSING_OR_SELF '+subject+':'+pid)
        references.update((fa['fragment_id'],fb['fragment_id']))
        for side,frag in (('a',fa),('b',fb)):
            require(row['source_sha256_'+side]==frag['source_sha256'] and
                    row['file_'+side]==frag['file_path'] and
                    row['start_line_'+side]==str(frag['start_line']) and
                    row['end_line_'+side]==str(frag['end_line']) and
                    row['function_'+side]==frag['function_name'] and
                    row['commit_'+side]==frag['commit'],
                    'J04D_PAIR_FRAGMENT_METADATA_MISMATCH '+subject+':'+pid)
    require(references==set(fragments),'J04D_ORPHAN_OR_UNREFERENCED_FRAGMENT '+subject)
    require(dict(detectors)==manifest.get('source_detector_counts'),
            'J04D_DETECTOR_COUNTS_INVALID '+subject)
    return directory,manifest,rows,fragments


def scalar_key(v):
    return (type(v).__name__,repr(v))


def signature(node,mode):
    """AST signatures: exact, consistent token mapping, blind token mapping.

    Python AST attributes (line/column offsets) and comments are excluded.
    Source spelling is never rewritten; normalization is for signatures only.
    """
    ids={}; literals={}
    def transform(value,identifier=False,literal=False):
        if isinstance(value,ast.AST):
            return [type(value).__name__,[
                [field,transform(v,identifier=(field in ID_FIELDS.get(type(value).__name__,())),
                                 literal=isinstance(value,ast.Constant) and field=='value')]
                for field,v in ast.iter_fields(value)]]
        if isinstance(value,(tuple,list)):
            return [transform(x,identifier=identifier,literal=literal) for x in value]
        if identifier and isinstance(value,str):
            if mode=='exact':return ['identifier',value]
            if mode=='blind':return ['identifier','ID']
            if value not in ids:ids[value]='ID'+str(len(ids))
            return ['identifier',ids[value]]
        if literal:
            k=scalar_key(value)
            if mode=='exact':return ['constant',k[0],k[1]]
            if mode=='blind':return ['constant',k[0],'LITERAL']
            if k not in literals:literals[k]='LIT'+str(len(literals))
            return ['constant',k[0],literals[k]]
        return ['scalar',type(value).__name__,repr(value)]
    data=json.dumps(transform(node),ensure_ascii=True,separators=(',',':')).encode('utf-8')
    return sha_bytes(data)


def preorder_types(node):
    out=[]; pending=[node]
    while pending:
        top=pending.pop()
        out.append(type(top).__name__)
        pending.extend(reversed(list(ast.iter_child_nodes(top))))
    return tuple(out)


def feature_extract(code, a):
    # A J-04d fragment can start at nested function/method indentation.
    try:
        # An indented method/nested function is valid only in its original
        # enclosing suite.  Wrap it instead of textwrap.dedent(): dedent can
        # silently alter the contents of triple-quoted string literals.
        first=next((line for line in code.splitlines() if line.strip()),'')
        indented=first.startswith((' ','\t'))
        src=('if True:\n'+code) if indented else code
        module=ast.parse(src,filename='<j04e-fragment>',type_comments=False)
        top=module.body[0] if len(module.body)==1 else None
        if indented:
            if not isinstance(top,ast.If) or len(top.body)!=1:
                return {'eligible':False,'reason':'NOT_SINGLE_WRAPPED_FUNCTION'}
            node=top.body[0]
        else:
            node=top
        if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            return {'eligible':False,'reason':'NOT_SINGLE_FUNCTION_DEFINITION'}
        seq=preorder_types(node)
        if not seq:return {'eligible':False,'reason':'EMPTY_AST_SEQUENCE'}
        return {'eligible':True,'reason':'',
                't1':signature(node,'exact'),
                't2c':signature(node,'consistent'),
                't2b':signature(node,'blind'),
                'ast_sha':sha_bytes(json.dumps(seq,separators=(',',':')).encode('ascii')),
                'seq':seq}
    except (SyntaxError,UnicodeError,ValueError,TypeError,RecursionError,MemoryError) as exc:
        return {'eligible':False,'reason':'AST_PARSE_OR_SIGNATURE_ERROR_'+type(exc).__name__}


def grams(seq,k):
    if len(seq)<k:return {seq}
    return set(tuple(seq[i:i+k]) for i in range(len(seq)-k+1))


def structure_score(left,right,a,score_cache):
    key=tuple(sorted((left['ast_sha'],right['ast_sha'])))
    if key in score_cache:return score_cache[key]
    sa,sb=left['seq'],right['seq']
    if len(sa)*len(sb)>a.ast_max_cells:
        return None
    ga,gb=grams(sa,a.ast_gram_size),grams(sb,a.ast_gram_size)
    jac=len(ga&gb)/len(ga|gb) if ga or gb else 1.0
    ratio=SequenceMatcher(None,sa,sb,autojunk=False).ratio()
    score=0.0 if not jac or not ratio else (2*jac*ratio/(jac+ratio))
    # A score depends only on the unordered pair of AST preorder signatures.
    score_cache[key]=score
    return score


def classify(a,subject,rows,fragments):
    features={}; cache={}; failure_reasons=Counter()
    for i,(fid,frag) in enumerate(fragments.items(),1):
        raw_sha=frag['source_sha256']
        if raw_sha not in cache:cache[raw_sha]=feature_extract(frag['code'],a)
        features[fid]=cache[raw_sha]
        if not features[fid]['eligible']:failure_reasons[features[fid]['reason']]+=1
        if i%a.fragment_progress_every==0:
            print(f'[{stamp(a)}] J04E_AST_PROGRESS subject={subject} fragments={i}/{len(fragments)}',flush=True)
    counts=Counter(); detector_types=defaultdict(Counter)
    sensitivity={str(t):Counter() for t in a.sensitivity_thresholds}
    scores={}; records=[]; example=defaultdict(list)
    for i,row in enumerate(rows,1):
        left,right=features[row['fragment_a_id']],features[row['fragment_b_id']]
        score=None; reason=''
        if not left['eligible'] or not right['eligible']:
            typ=INELIGIBLE
            reason=left.get('reason','') if not left['eligible'] else right.get('reason','')
        elif left['t1']==right['t1']:
            typ=T1
        elif left['t2c']==right['t2c']:
            typ=T2C
        elif left['t2b']==right['t2b']:
            typ=T2B
        else:
            score=structure_score(left,right,a,scores)
            typ=UNSCORED if score is None else (T3 if score >= float(a.primary_threshold) else UNDET)
            reason='AST_COMPARISON_CELLS_EXCEEDED' if typ==UNSCORED else ''
        counts[typ]+=1
        detector_types[row['detector']][typ]+=1
        for threshold in a.sensitivity_thresholds:
            if typ in (T1,T2C,T2B,INELIGIBLE,UNSCORED): t=typ
            else:t=T3 if score is not None and score>=float(threshold) else UNDET
            sensitivity[str(threshold)][t]+=1
        rec={'row':row,'clone_subtype':typ,'score':score,'reason':reason}
        records.append(rec)
        if len(example[typ])<a.examples_per_category:
            example[typ].append({'pair_id':row['pair_id'],
                                 'ast_combined_similarity':round(score,8) if score is not None else None})
        if i%a.pair_progress_every==0:
            print(f'[{stamp(a)}] J04E_PAIR_PROGRESS subject={subject} pairs={i}/{len(rows)}',flush=True)
    require(sum(counts.values())==len(rows),'TYPE_CENSUS_PAIR_LOSS '+subject)
    return records,counts,detector_types,sensitivity,failure_reasons,example


def content_pair_key(row):
    return ':'.join(sorted((row['source_sha256_a'],row['source_sha256_b'])))


def select(a,subject,records):
    """All available Type 1/2/3, unbiased by source order; one code pair per subject."""
    winners={}; category_for={}; total_candidates=0
    for rec in records:
        typ=rec['clone_subtype']
        if typ not in SELECTABLE:continue
        total_candidates+=1
        row=rec['row']; key=content_pair_key(row)
        if key in category_for:
            require(category_for[key]==typ,'CONTENT_PAIR_CLONE_TYPE_CONFLICT '+subject+':'+row['pair_id'])
        else:category_for[key]=typ
        rank=sha_bytes((str(a.sample_seed)+'\0'+subject+'\0'+row['pair_id']).encode('utf-8'))
        if key not in winners or (rank,row['pair_id'])<winners[key][0]:
            winners[key]=((rank,row['pair_id']),row['pair_id'])
    selected_ids={v[1] for v in winners.values()}
    for rec in records:
        typ=rec['clone_subtype']
        if typ not in SELECTABLE:
            rec['selection']='EXCLUDED';rec['selected_pair_id']=''
        else:
            winner=winners[content_pair_key(rec['row'])][1]
            rec['selected_pair_id']=winner
            rec['selection']='SELECTED' if rec['row']['pair_id']==winner else 'CONTENT_DUPLICATE'
    require(sum(x['selection']=='SELECTED' for x in records)==len(winners) and
            len(selected_ids)==len(winners),'DEDUP_SELECTION_INCONSISTENCY '+subject)
    return selected_ids,total_candidates


def public_item(a,subject,row,typ):
    return {'sample_id':'pos-'+sha_bytes((subject+'\0'+row['pair_id']).encode('utf-8'))[:24],
            'repository':subject,'clone_type':'Type 1' if typ==T1 else 'Type 3' if typ==T3 else 'Type 2',
            'label':a.positive_label,
            'left':{'function_id':row['fragment_a_id']},
            'right':{'function_id':row['fragment_b_id']}}


def details_item(a,subject,rec,fragments):
    row=rec['row']
    item=public_item(a,subject,row,rec['clone_subtype'])
    item.update({'clone_subtype':rec['clone_subtype'],
                 'original_pair_id':row['pair_id'],
                 'upstream_pair_id':row['upstream_pair_id'],
                 'nicad_similarity_percent':row['nicad_similarity_raw'],
                 'ast_combined_similarity':round(rec['score'],8) if rec['score'] is not None else None,
                 'ast_primary_threshold':str(a.primary_threshold),
                 'detector':row['detector'],'detector_config':row['config']})
    for side,name in (('a','source_a'),('b','source_b')):
        frag=fragments[row['fragment_'+side+'_id']]
        item[name]={'fragment_id':frag['fragment_id'],
                    'file':frag['file_path'],'start_line':frag['start_line'],
                    'end_line':frag['end_line'],'commit':frag['commit'],
                    'source_sha256':frag['source_sha256']}
    return item


def config(a):
    return {
        'experiment':a.experiment,'version':a.version,'source_experiment':a.source_experiment,
        'source_version':a.source_version,'source_input_version':a.source_input_version,
        'source_all_run_id':a.source_all_run_id,'source_verify_run_id':a.source_verify_run_id,
        'source_all_root':str(a.source_all_root),'source_verify_root':str(a.source_verify_root),
        'source_batch_name':a.source_batch_name,'source_verification_name':a.source_verification_name,
        'result_root':str(a.result_root),'expected_subjects':a.expected_subjects,
        'expected_input_files':a.expected_input_files,'expected_pairs':a.expected_pairs,
        'expected_fragments':a.expected_fragments,'primary_threshold':str(a.primary_threshold),
        'sensitivity_thresholds':[str(x) for x in a.sensitivity_thresholds],
        'ast_gram_size':a.ast_gram_size,'ast_max_cells':a.ast_max_cells,
        'sample_seed':a.sample_seed,'selection_policy':a.selection_policy,
        'dedup_policy':a.dedup_policy,'positive_label':a.positive_label,
        'allowed_detectors':list(a.allowed_detectors),
        'language':a.language,'code_encoding':a.code_encoding,'code_errors':a.code_errors,
        'python_version':a.expected_python_version,'timezone':a.timezone,
        'smoke_subject':a.smoke_subject,'pair_progress_every':a.pair_progress_every,
        'fragment_progress_every':a.fragment_progress_every,
        'examples_per_category':a.examples_per_category,
    }


def census_item(a,subject,records,counts,detector_types,sensitivity,failures,examples,selected_ids,candidates):
    selected_counts=Counter(r['clone_subtype'] for r in records if r['selection']=='SELECTED')
    return {
        'schema':'deepknowcode-j04e-type-census-'+a.version,'subject':subject,
        'timestamp_chicago':stamp(a),'source_pairs':len(records),
        'counts':{k:counts[k] for k in CATEGORIES},
        'type2_all':counts[T2C]+counts[T2B],
        'structural_candidates_before_dedup':candidates,
        'selected_unique_positive_pairs':len(selected_ids),
        'content_duplicates_excluded':candidates-len(selected_ids),
        'excluded_not_type_123':len(records)-candidates,
        'selected_counts':{k:selected_counts[k] for k in SELECTABLE},
        'detector_by_clone_type':{d:{k:v[k] for k in CATEGORIES} for d,v in sorted(detector_types.items())},
        'sensitivity_thresholds':{k:{x:v[x] for x in CATEGORIES} for k,v in sensitivity.items()},
        'ineligible_fragment_reasons':dict(sorted(failures.items())),
        'examples':{k:examples[k] for k in CATEGORIES},
        'classification_method':{
            'type1':'Exact Python AST canonical syntax excluding location attributes and comments',
            'type2_consistent':'Consistent renaming of identifier and literal tokens in AST',
            'type2_blind':'Blind renaming of identifier and literal tokens in AST',
            'type3':'Harmonic mean of AST node-type k-gram Jaccard and SequenceMatcher ratio',
            'classification_precedence':list(CATEGORIES),
            'parser':'Python stdlib ast.parse; indentation-preserving enclosing if-wrapper for nested functions',
            'min_ast_score':str(a.primary_threshold),
            'gram_size':a.ast_gram_size,'max_ast_comparison_cells':a.ast_max_cells},
        'note':'Structural detector candidates only; no semantic equivalence certification.'}


def emit_subject(a,subject,source_dir,source_manifest,rows,fragments,out,source_hashes):
    require(not out.exists() and not out.is_symlink(),'J04E_OUTPUT_ALREADY_EXISTS '+str(out))
    out.mkdir(parents=True,exist_ok=False)
    (out/'details').mkdir(exist_ok=False)
    for filename in ('positive_pairs.csv','fragments.jsonl'):
        with (source_dir/filename).open('rb') as inp,(out/filename).open('xb') as dest:
            shutil.copyfileobj(inp,dest)
        require(sha_file(out/filename)==source_manifest['files_sha256'][filename],
                'J04E_COPIED_J04D_DATA_MISMATCH '+subject+':'+filename)
    records,counts,detector_types,sensitivity,failures,examples=classify(a,subject,rows,fragments)
    selected_ids,candidates=select(a,subject,records)
    census=census_item(a,subject,records,counts,detector_types,sensitivity,failures,examples,
                       selected_ids,candidates)
    json_new(out/'ast_clone_type_census.json',census)
    with (out/'pair_clone_types.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=TYPE_COLUMNS,lineterminator='\n')
        w.writeheader()
        for rec in records:
            row=rec['row']
            w.writerow({'pair_id':row['pair_id'],'fragment_a_id':row['fragment_a_id'],
                        'fragment_b_id':row['fragment_b_id'],'detector':row['detector'],
                        'clone_subtype':rec['clone_subtype'],
                        'ast_combined_similarity':f"{rec['score']:.8f}" if rec['score'] is not None else '',
                        'selection':rec['selection'],'selected_pair_id':rec['selected_pair_id'],
                        'reason':rec['reason']})
    pub=out/'fixed_positive_clonepairs.jsonl'
    detail=out/'details/fixed_positive_clonepairs.jsonl'
    with pub.open('x',encoding='utf-8') as fp,detail.open('x',encoding='utf-8') as fd:
        for rec in records:
            if rec['selection']!='SELECTED':continue
            fp.write(json_line(public_item(a,subject,rec['row'],rec['clone_subtype'])))
            fd.write(json_line(details_item(a,subject,rec,fragments)))
    listed={p:sha_file(out/p) for p in sorted(OUTPUT_FILES-{'manifest.json','details/fixed_positive_clonepairs.manifest.json'})}
    dmanifest={
        'schema':'deepknowcode-j04e-fixed-positives-'+a.version,
        'experiment':a.experiment,'version':a.version,'status':'FIXED_POSITIVE_ONLY',
        'timestamp_chicago':stamp(a),'subject':subject,
        'source_j04d_manifest_sha256':source_hashes['source_manifest_sha256'],
        'source_j04d_verification_sha256':source_hashes['source_verification_sha256'],
        'candidate_pairs':len(rows),'selected_positive_pairs':len(selected_ids),
        'removed_content_duplicates':candidates-len(selected_ids),
        'excluded_non_type123':len(rows)-candidates,
        'type_census':{k:counts[k] for k in CATEGORIES},
        'selected_type_counts':census['selected_counts'],
        'files_sha256':listed,'effective_configuration':config(a),
        'function_id_reference':'left.function_id/right.function_id => fragments.jsonl:fragment_id',
        'label_semantics':'structural clone candidate; not independently certified semantic gold',
    }
    json_new(out/'details/fixed_positive_clonepairs.manifest.json',dmanifest)
    file_hashes={**listed,
                 'details/fixed_positive_clonepairs.manifest.json':sha_file(out/'details/fixed_positive_clonepairs.manifest.json')}
    manifest={'schema':'deepknowcode-j04e-'+a.version,'experiment':a.experiment,
              'version':a.version,'status':'COMPLETED',
              'timestamp_chicago':stamp(a),'mode':a.mode,'subject':subject,'run_id':a.run_id,
              'source_j04d_manifest_sha256':source_hashes['source_manifest_sha256'],
              'source_j04d_batch_sha256':source_hashes['source_batch_sha256'],
              'source_j04d_verification_sha256':source_hashes['source_verification_sha256'],
              'input_pairs':len(rows),'input_fragments':len(fragments),
              'selected_positive_pairs':len(selected_ids),'type_census':census['counts'],
              'selected_type_counts':census['selected_counts'],
              'content_duplicates_removed':candidates-len(selected_ids),
              'files_sha256':file_hashes,'effective_configuration':config(a)}
    json_new(out/'manifest.json',manifest)
    print('PASS J04E_SUBJECT subject='+subject+' input_pairs='+str(len(rows))+
          ' fragments='+str(len(fragments))+' type123_candidates='+str(candidates)+
          ' fixed_positive='+str(len(selected_ids))+' content_duplicates='+
          str(candidates-len(selected_ids))+' output='+str(out),flush=True)
    return {'subject':subject,'status':'COMPLETED','input_pairs':len(rows),
            'input_fragments':len(fragments),'type_census':census['counts'],
            'selected_type_counts':census['selected_counts'],
            'type123_candidates':candidates,'selected_positive_pairs':len(selected_ids),
            'content_duplicates_removed':candidates-len(selected_ids),
            'manifest_sha256':sha_file(out/'manifest.json')}


def verify_subject(a,subject,entry,out,source_dir,source_manifest,rows,fragments,source_hashes):
    require(out.is_dir() and not out.is_symlink(),'J04E_OUTPUT_DIR_MISSING '+str(out))
    observed=set()
    for base,dirs,files in os.walk(out,followlinks=False):
        for name in dirs:
            require(not (Path(base)/name).is_symlink(),'J04E_SYMLINK_FORBIDDEN '+str(Path(base)/name))
        for name in files:
            path=Path(base)/name
            require(path.is_file() and not path.is_symlink(),'J04E_UNSAFE_FILE '+str(path))
            observed.add(path.relative_to(out).as_posix())
    require(observed==OUTPUT_FILES,'J04E_OUTPUT_FILE_SET_MISMATCH '+subject)
    manifest=json_read(out/'manifest.json')
    dm=json_read(out/'details/fixed_positive_clonepairs.manifest.json')
    require(manifest.get('schema')=='deepknowcode-j04e-'+a.version and
            manifest.get('version')==a.version and manifest.get('subject')==subject and
            manifest.get('status')=='COMPLETED' and
            manifest.get('input_pairs')==len(rows) and manifest.get('input_fragments')==len(fragments) and
            manifest.get('source_j04d_manifest_sha256')==source_hashes['source_manifest_sha256'] and
            manifest.get('source_j04d_batch_sha256')==source_hashes['source_batch_sha256'] and
            manifest.get('source_j04d_verification_sha256')==source_hashes['source_verification_sha256'] and
            manifest.get('effective_configuration')==config(a),
            'J04E_MANIFEST_IDENTITY_OR_CONFIG_INVALID '+subject)
    require(dm.get('schema')=='deepknowcode-j04e-fixed-positives-'+a.version and
            dm.get('status')=='FIXED_POSITIVE_ONLY' and dm.get('subject')==subject and
            dm.get('effective_configuration')==config(a),
            'J04E_DETAILS_MANIFEST_INVALID '+subject)
    for rel in OUTPUT_FILES-{'manifest.json'}:
        require(sha_file(out/rel)==manifest['files_sha256'].get(rel),
                'J04E_OUTPUT_SHA256_MISMATCH '+subject+':'+rel)
        if rel!='details/fixed_positive_clonepairs.manifest.json':
            require(sha_file(out/rel)==dm['files_sha256'].get(rel),
                    'J04E_DETAILS_SHA256_MISMATCH '+subject+':'+rel)
    for rel in ('positive_pairs.csv','fragments.jsonl'):
        require(sha_file(out/rel)==sha_file(source_dir/rel),
                'J04E_SOURCE_COPIED_BYTES_CHANGED '+subject+':'+rel)
    records,counts,dt,sen,failures,examples=classify(a,subject,rows,fragments)
    selected_ids,candidates=select(a,subject,records)
    generated=census_item(a,subject,records,counts,dt,sen,failures,examples,selected_ids,candidates)
    actual_census=json_read(out/'ast_clone_type_census.json')
    # Timestamp is output metadata, not part of deterministic AST classification.
    for k,v in generated.items():
        if k!='timestamp_chicago':
            require(actual_census.get(k)==v,'J04E_CENSUS_SEMANTIC_MISMATCH '+subject+':'+k)
    observed_types=csv_read(out/'pair_clone_types.csv',TYPE_COLUMNS)
    require(len(observed_types)==len(records),'J04E_CENSUS_ROW_COUNT_MISMATCH '+subject)
    for got,rec in zip(observed_types,records,strict=True):
        row=rec['row']
        expected={'pair_id':row['pair_id'],'fragment_a_id':row['fragment_a_id'],
                  'fragment_b_id':row['fragment_b_id'],'detector':row['detector'],
                  'clone_subtype':rec['clone_subtype'],
                  'ast_combined_similarity':f"{rec['score']:.8f}" if rec['score'] is not None else '',
                  'selection':rec['selection'],'selected_pair_id':rec['selected_pair_id'],
                  'reason':rec['reason']}
        require(got==expected,'J04E_CLONE_TYPE_ROW_MISMATCH '+subject+':'+row['pair_id'])
    public=jsonl_read(out/'fixed_positive_clonepairs.jsonl')
    details=jsonl_read(out/'details/fixed_positive_clonepairs.jsonl')
    chosen=[rec for rec in records if rec['selection']=='SELECTED']
    require(len(chosen)==len(public)==len(details)==len(selected_ids),
            'J04E_FIXED_POSITIVE_COUNT_MISMATCH '+subject)
    ids=set(); content=set()
    for rec,pub,det in zip(chosen,public,details,strict=True):
        expected_pub=public_item(a,subject,rec['row'],rec['clone_subtype'])
        expected_det=details_item(a,subject,rec,fragments)
        require(pub==expected_pub and det==expected_det,
                'J04E_PUBLIC_DETAILS_ID_OR_PROVENANCE_MISMATCH '+subject+':'+rec['row']['pair_id'])
        require(set(pub)=={'sample_id','repository','clone_type','label','left','right'} and
                pub['sample_id'] not in ids and
                pub['left']['function_id'] in fragments and pub['right']['function_id'] in fragments,
                'J04E_PUBLIC_SCHEMA_OR_ID_ERROR '+subject)
        ids.add(pub['sample_id'])
        key=content_pair_key(rec['row'])
        require(key not in content,'J04E_DUPLICATE_SOURCE_CONTENT_PAIR '+subject)
        content.add(key)
    require(manifest.get('type_census')==generated['counts'] and
            manifest.get('selected_type_counts')==generated['selected_counts'] and
            manifest.get('selected_positive_pairs')==len(chosen) and
            manifest.get('content_duplicates_removed')==candidates-len(chosen) and
            dm.get('type_census')==generated['counts'] and
            dm.get('selected_type_counts')==generated['selected_counts'] and
            dm.get('selected_positive_pairs')==len(chosen) and
            dm.get('candidate_pairs')==len(records) and
            dm.get('removed_content_duplicates')==candidates-len(chosen) and
            dm.get('excluded_non_type123')==len(records)-candidates,
            'J04E_MANIFEST_CENSUS_COUNTS_MISMATCH '+subject)
    print('PASS J04E_VERIFY subject='+subject+' pairs='+str(len(rows))+
          ' selected='+str(len(chosen))+' fragments='+str(len(fragments)),flush=True)
    return {'subject':subject,'input_pairs':len(rows),'input_fragments':len(fragments),
            'selected_positive_pairs':len(chosen),'type_census':generated['counts'],
            'manifest_sha256':sha_file(out/'manifest.json')}


def args_parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','version','source-experiment','source-version',
                 'source-input-version','source-all-run-id','source-verify-run-id',
                 'source-batch-name','source-verification-name','run-id','mode','subject',
                 'smoke-subject','timezone','language','code-encoding','code-errors',
                 'expected-python-version','allowed-detectors','primary-threshold',
                 'sensitivity-thresholds','selection-policy','dedup-policy'):
        p.add_argument('--'+name,required=True)
    for name in ('source-all-root','source-verify-root','result-root','run-dir'):
        p.add_argument('--'+name,required=True,type=Path)
    for name in ('expected-subjects','expected-input-files','expected-pairs','expected-fragments',
                 'positive-label','ast-gram-size','ast-max-cells','sample-seed',
                 'pair-progress-every','fragment-progress-every','examples-per-category'):
        p.add_argument('--'+name,required=True,type=int)
    a=p.parse_args(argv)
    require(a.experiment=='j-04e' and re.fullmatch(r'v[0-9]+',a.version) and
            a.source_experiment=='j-04d' and re.fullmatch(r'v[0-9]+',a.source_version) and
            a.mode in ('inspect','smoke','all','verify','subject') and
            a.timezone=='America/Chicago' and a.language=='python' and
            a.positive_label==1,'J04E_EXPERIMENT_IDENTITY_INVALID')
    require(a.selection_policy=='all_available_types' and
            a.dedup_policy=='unordered_source_sha256_pair' and
            a.expected_subjects>0 and a.expected_input_files>0 and
            a.expected_pairs>0 and a.expected_fragments>0 and
            a.ast_gram_size>=1 and a.ast_max_cells>=1 and a.sample_seed>=0 and
            a.pair_progress_every>0 and a.fragment_progress_every>0 and
            a.examples_per_category>=0,'J04E_CONFIGURATION_INVALID')
    require(a.expected_python_version==f'{sys.version_info.major}.{sys.version_info.minor}',
            'PYTHON_VERSION_MISMATCH expected='+a.expected_python_version+
            ' actual='+f'{sys.version_info.major}.{sys.version_info.minor}')
    require(re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',a.run_id) is not None and
            re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',a.source_all_run_id) is not None and
            re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',a.source_verify_run_id) is not None,
            'J04E_RUN_IDS_INVALID')
    require(SUBJECT_RE.fullmatch(a.smoke_subject) is not None,'J04E_SMOKE_SUBJECT_INVALID')
    if a.mode in ('smoke','subject'):
        require(SUBJECT_RE.fullmatch(a.subject) is not None,'J04E_SUBJECT_INVALID')
    a.allowed_detectors=tuple(a.allowed_detectors.split(','))
    require(set(a.allowed_detectors)=={'nicad_primary','jscpd_ast_fallback'},
            'J04E_DETECTOR_ALLOWLIST_INVALID')
    try:
        a.primary_threshold=Decimal(a.primary_threshold)
        a.sensitivity_thresholds=tuple(Decimal(v) for v in a.sensitivity_thresholds.split(','))
    except (InvalidOperation,ValueError) as exc:
        raise RuntimeError('J04E_THRESHOLD_INVALID') from exc
    require(a.primary_threshold.is_finite() and Decimal(0)<=a.primary_threshold<=Decimal(1) and
            len(a.sensitivity_thresholds)>0 and
            all(x.is_finite() and Decimal(0)<=x<=Decimal(1) for x in a.sensitivity_thresholds) and
            a.primary_threshold in a.sensitivity_thresholds,
            'J04E_THRESHOLD_OUT_OF_RANGE_OR_PRIMARY_NOT_INCLUDED')
    for name in ('source_all_root','source_verify_root','result_root','run_dir'):
        setattr(a,name,getattr(a,name).absolute())
    require(a.run_dir.parent==a.result_root and a.source_all_root!=a.source_verify_root and
            a.result_root!=a.source_all_root and a.result_root!=a.source_verify_root and
            not a.result_root.is_relative_to(a.source_all_root) and
            not a.result_root.is_relative_to(a.source_verify_root) and
            not a.source_all_root.is_relative_to(a.result_root) and
            not a.source_verify_root.is_relative_to(a.result_root),
            'J04E_INPUT_OUTPUT_PATH_OVERLAP')
    return a


def most_recent_all(a):
    require(a.result_root.is_dir() and not a.result_root.is_symlink(),'J04E_RESULT_ROOT_MISSING')
    directories=sorted((p for p in a.result_root.iterdir() if p.is_dir() and not p.is_symlink() and
                        '-all-' in p.name),key=lambda p:p.name,reverse=True)
    require(directories,'J04E_ALL_RUN_NOT_FOUND')
    root=directories[0]
    summary=json_read(root/'batch-summary.json')
    require(summary.get('status')=='COMPLETED' and summary.get('mode')=='all' and
            summary.get('version')==a.version and summary.get('experiment')==a.experiment and
            summary.get('run_id')==root.name and summary.get('subjects_completed')==a.expected_subjects and
            summary.get('input_pairs_total')==a.expected_pairs and
            summary.get('input_fragments_total')==a.expected_fragments,
            'J04E_LATEST_ALL_INCOMPLETE_OR_WRONG_VERSION')
    return root,summary


def main(argv=None):
    a=args_parse(argv)
    summary,verification,by_name,batch_sha,verification_sha=read_dataset(a)
    names=sorted(by_name)
    source_hashes_base={'source_batch_sha256':batch_sha,'source_verification_sha256':verification_sha}
    if a.mode=='inspect':
        for name in names:
            _dir,manifest,rows,fragments=load_subject(a,name,by_name[name])
            print(f'PASS J04E_INPUT subject={name} pairs={len(rows)} fragments={len(fragments)}',flush=True)
        a.run_dir.mkdir(parents=True,exist_ok=False)
        json_new(a.run_dir/'inspection.json',{'status':'PASS','timestamp_chicago':stamp(a),
                 'subjects':len(names),'pairs':a.expected_pairs,'fragments':a.expected_fragments,
                 'source_all':str(a.source_all_root),'source_verification':str(a.source_verify_root),
                 **source_hashes_base})
        print(f'PASS J04E_INSPECT subjects={len(names)} pairs={a.expected_pairs} fragments={a.expected_fragments}',flush=True)
        return 0
    if a.mode=='verify':
        all_root,all_summary=most_recent_all(a)
        require(all_summary.get('source_j04d_batch_sha256')==batch_sha and
                all_summary.get('source_j04d_verification_sha256')==verification_sha and
                {r['subject'] for r in all_summary.get('subjects',[])}==set(names),
                'J04E_VERIFICATION_INPUT_OR_SUBJECT_DRIFT')
        records=[]
        for index,name in enumerate(names,1):
            source_dir,source_manifest,rows,fragments=load_subject(a,name,by_name[name])
            hashes={**source_hashes_base,'source_manifest_sha256':sha_file(source_dir/'manifest.json')}
            check=verify_subject(a,name,by_name[name],all_root/'subjects'/name,
                                 source_dir,source_manifest,rows,fragments,hashes)
            require(check['manifest_sha256']==next(r['manifest_sha256'] for r in all_summary['subjects']
                                                  if r['subject']==name),
                    'J04E_BATCH_SUBJECT_MANIFEST_HASH_MISMATCH '+name)
            records.append(check)
        require(sum(x['input_pairs'] for x in records)==a.expected_pairs and
                sum(x['input_fragments'] for x in records)==a.expected_fragments and
                sum(x['selected_positive_pairs'] for x in records)==all_summary['fixed_positive_total'],
                'J04E_VERIFICATION_TOTAL_MISMATCH')
        a.run_dir.mkdir(parents=True,exist_ok=False)
        json_new(a.run_dir/'verification-all.json',{'status':'PASS','version':a.version,
                 'timestamp_chicago':stamp(a),'source_all':str(all_root),
                 'source_j04d_batch_sha256':batch_sha,'source_j04d_verification_sha256':verification_sha,
                 'verified_subjects':len(records),'input_pairs':a.expected_pairs,
                 'input_fragments':a.expected_fragments,'fixed_positive_total':sum(x['selected_positive_pairs'] for x in records),
                 'subjects':records})
        print('PASS J04E_VERIFY_ALL subjects='+str(len(records))+' pairs='+str(a.expected_pairs)+
              ' fixed_positive='+str(sum(x['selected_positive_pairs'] for x in records)),flush=True)
        return 0
    if a.mode=='all':chosen=names
    else:
        chosen=[a.smoke_subject if a.mode=='smoke' else a.subject]
        require(chosen[0] in by_name,'J04E_UNKNOWN_SUBJECT '+chosen[0])
    a.run_dir.mkdir(parents=True,exist_ok=False)
    (a.run_dir/'subjects').mkdir(exist_ok=False)
    records=[]
    for i,name in enumerate(chosen,1):
        print(f'[{stamp(a)}] START J04E_SUBJECT {i}/{len(chosen)} subject={name}',flush=True)
        source_dir,source_manifest,rows,fragments=load_subject(a,name,by_name[name])
        hashes={**source_hashes_base,'source_manifest_sha256':sha_file(source_dir/'manifest.json')}
        output=a.run_dir/'subjects'/name
        rec=emit_subject(a,name,source_dir,source_manifest,rows,fragments,output,hashes)
        verify_subject(a,name,by_name[name],output,source_dir,source_manifest,rows,fragments,hashes)
        records.append(rec)
    if a.mode=='all':
        require(len(records)==a.expected_subjects and
                sum(r['input_pairs'] for r in records)==a.expected_pairs and
                sum(r['input_fragments'] for r in records)==a.expected_fragments,
                'J04E_FULL_RUN_INPUT_CONSERVATION_FAILURE')
    total_census={k:sum(r['type_census'][k] for r in records) for k in CATEGORIES}
    selected_counts={k:sum(r['selected_type_counts'][k] for r in records) for k in SELECTABLE}
    out_summary={'schema':'deepknowcode-j04e-batch-'+a.version,
                 'experiment':a.experiment,'version':a.version,'mode':a.mode,
                 'status':'COMPLETED','run_id':a.run_id,'timestamp_chicago':stamp(a),
                 'source_all':str(a.source_all_root),'source_verification':str(a.source_verify_root),
                 'source_j04d_batch_sha256':batch_sha,'source_j04d_verification_sha256':verification_sha,
                 'subjects_completed':len(records),'input_pairs_total':sum(r['input_pairs'] for r in records),
                 'input_fragments_total':sum(r['input_fragments'] for r in records),
                 'type_census':total_census,'selected_type_counts':selected_counts,
                 'type123_candidates_total':sum(r['type123_candidates'] for r in records),
                 'fixed_positive_total':sum(r['selected_positive_pairs'] for r in records),
                 'content_duplicates_removed_total':sum(r['content_duplicates_removed'] for r in records),
                 'effective_configuration':config(a),'subjects':records}
    json_new(a.run_dir/'batch-summary.json',out_summary)
    print('PASS J04E_SUMMARY mode='+a.mode+' subjects='+str(len(records))+
          ' input_pairs='+str(out_summary['input_pairs_total'])+
          ' fixed_positive='+str(out_summary['fixed_positive_total'])+
          ' duplicates='+str(out_summary['content_duplicates_removed_total']),flush=True)
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except (RuntimeError,ValueError,TypeError,KeyError,IndexError,UnicodeError,
            SyntaxError,OSError,OverflowError,RecursionError,MemoryError) as exc:
        print('FAIL J04E '+type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True)
        sys.exit(2)
