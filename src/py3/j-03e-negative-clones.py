#!/usr/bin/env python3
"""J-03e v3 script (v2 dataset schema): per-subject AST-low-similarity negative *candidates*, never semantic gold.

All experiment configuration must be passed by the shell. No data is changed in 3pos-clones.
"""
import argparse
import csv
from collections import Counter
from datetime import datetime
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import sys
import time
from zoneinfo import ZoneInfo


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):
            h.update(data)
    return h.hexdigest()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pair_key(hex_a,hex_b):
    a,b=sorted((bytes.fromhex(hex_a),bytes.fromhex(hex_b)))
    return hashlib.sha256(a+b).digest()


def json_line(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True)+'\n'


def clock(a):
    return datetime.now(ZoneInfo(a.time_zone)).isoformat(timespec='seconds')


def emit(a,event,**kwargs):
    print(f'[{clock(a)}] {event} '+' '.join(f'{k}={v}' for k,v in kwargs.items()),flush=True)


def parser_args():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('experiment','version','mode','subject','time-zone','positive-root',
              'negative-root','run-dir','reference-summary-name','reference-summary-sha256',
              'public-name','detail-dir','detail-name','manifest-name','report-name',
              'h07-python','h08-python','parser-version','grammar-version',
              'code-encoding','code-errors','positive-policy','positive-schema',
              'h07-expected-sha256','h08-expected-sha256',
              'source-manifest-name','fragment-name','original-positive-csv-name',
              'script-version','legacy-positive-root','legacy-negative-root',
              'fixed-positive-public-name','fixed-positive-details-name',
              'fixed-positive-manifest-name','positive-details-dir'):
        p.add_argument('--'+n,required=True)
    for n in ('expected-subjects','expected-total-positives','positive-label','negative-label',
              'sample-seed','ast-gram-size','ast-max-cells','ast-score-cache-size',
              'attempts-per-target','attempt-limit-cap','feasibility-probe-pairs',
              'progress-every','sqlite-cache-kib'):
        p.add_argument('--'+n,required=True,type=int)
    p.add_argument('--negative-similarity-max',required=True,type=float)
    p.add_argument('--positive-ast-threshold',required=True,type=float)
    a=p.parse_args()
    for n in ('positive_root','negative_root','legacy_positive_root','legacy_negative_root','run_dir','h07_python','h08_python'):
        setattr(a,n,Path(getattr(a,n)).expanduser().resolve())
    need(a.mode in ('negativeclone-feasibility','negativeclone','negativeclone-verify'),'INVALID_MODE')
    need(a.time_zone=='America/Chicago','TIMEZONE_MISMATCH')
    need(a.experiment=='j-03e' and a.version=='v2' and a.script_version=='v3',
         'EXPERIMENT_SCRIPT_OR_DATASET_VERSION_MISMATCH')
    need(a.positive_root==a.legacy_positive_root.parent/'java'/a.legacy_positive_root.name and
         a.negative_root==a.legacy_negative_root.parent/'java'/a.legacy_negative_root.name,
         'INVALID_JAVA_LAYOUT_RELOCATION')
    need(a.positive_label==1 and a.negative_label==0,'LABEL_CONFIGURATION_INVALID')
    need(a.positive_policy=='all_available_types' and a.positive_schema=='function_refs_v8',
         'POSITIVE_INPUT_POLICY_OR_SCHEMA_UNEXPECTED')
    need(0 < a.negative_similarity_max < 1 and a.negative_similarity_max < a.positive_ast_threshold <= 1,
         'INVALID_NEGATIVE_SIMILARITY_MAX')
    for n in ('expected_subjects','expected_total_positives','ast_gram_size','ast_max_cells',
              'ast_score_cache_size','attempts_per_target','attempt_limit_cap',
              'feasibility_probe_pairs','progress_every','sqlite_cache_kib'):
        need(getattr(a,n)>0,'CONFIG_MUST_BE_POSITIVE '+n)
    need(a.positive_root.is_dir() and not a.positive_root.is_symlink(),'POSITIVE_ROOT_MISSING_OR_UNSAFE')
    need(a.run_dir.is_dir() and not a.run_dir.is_symlink(),'RUN_DIR_MISSING_OR_UNSAFE')
    need(a.negative_root.is_dir() and not a.negative_root.is_symlink(),'NEGATIVE_ROOT_MISSING_OR_UNSAFE')
    need(bool(re.fullmatch('[a-z0-9][a-z0-9_.-]*(?:__[a-z0-9_.-]+)?',a.subject)) or a.subject=='ALL',
         'INVALID_SUBJECT_NAME')
    return a


def read_json(path):
    need(path.is_file() and not path.is_symlink(),'REQUIRED_FILE_MISSING_OR_SYMLINK '+str(path))
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def rows_jsonl(path):
    need(path.is_file() and not path.is_symlink(),'REQUIRED_JSONL_MISSING_OR_SYMLINK '+str(path))
    with path.open(encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            need(bool(line.strip()),f'BLANK_JSONL {path}:{i}')
            obj=json.loads(line)
            need(isinstance(obj,dict),f'INVALID_JSONL {path}:{i}')
            yield obj


def load_summary(a):
    p=a.positive_root/a.reference_summary_name
    need(p.is_file() and not p.is_symlink(),'REFERENCE_SUMMARY_MISSING '+str(p))
    need(sha_file(p)==a.reference_summary_sha256,'REFERENCE_SUMMARY_SHA256_MISMATCH')
    raw=p.read_text(encoding='utf-8')
    need('## A. Fixed structural positive clone pairs' in raw,'REFERENCE_SECTION_A_MISSING')
    seg=raw.split('## A. Fixed structural positive clone pairs',1)[1].split('```text',1)[1].split('```',1)[0]
    result={}
    for line in seg.strip().splitlines():
        fields=line.split()
        if len(fields)!=8 or fields[0] in ('Subject','---------------------------') or fields[0].startswith('--'):
            continue
        need(all(x.replace(',','').isdigit() for x in fields[1:]),'REFERENCE_INVALID_NUMERIC_ROW '+line)
        need(fields[0] not in result,'REFERENCE_DUPLICATE_SUBJECT '+fields[0])
        values=[int(x.replace(',','')) for x in fields[1:]]
        need(values[4]==values[2]+values[3] and values[6]==values[1]+values[4]+values[5],
             'REFERENCE_TYPE_TOTAL_MISMATCH '+fields[0])
        result[fields[0]]=values[6]
    need(len(result)==a.expected_subjects+1 and 'TOTAL' in result,'REFERENCE_SUBJECT_COUNT_MISMATCH')
    need(result.pop('TOTAL')==a.expected_total_positives,'REFERENCE_TOTAL_MISMATCH')
    need(sum(result.values())==a.expected_total_positives,'REFERENCE_SUM_INCONSISTENT')
    dirs={p.name for p in a.positive_root.iterdir() if p.is_dir() and not p.is_symlink()}
    # The 3pos root may have archived auxiliary directories; only summary-defined subjects are used.
    need(set(result)<=dirs,'REFERENCE_SUBJECT_DIRECTORY_MISSING')
    if a.subject=='ALL':
        return dict(sorted(result.items()))
    need(a.subject in result,'SUBJECT_NOT_IN_REFERENCE '+a.subject)
    return {a.subject:result[a.subject]}


def load_ref(path,name):
    need(path.is_file() and not path.is_symlink(),'AST_REFERENCE_REQUIRED '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    need(spec is not None and spec.loader is not None,'AST_REFERENCE_SPEC_INVALID '+str(path))
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ast_setup(a):
    for package,expected in (('tree-sitter',a.parser_version),('tree-sitter-java',a.grammar_version)):
        try: actual=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError('AST_DEPENDENCY_MISSING '+package+' install '+package+'=='+expected) from exc
        need(actual==expected,f'AST_DEPENDENCY_VERSION_MISMATCH {package} {actual}!={expected}')
    from tree_sitter import Parser,Language
    import tree_sitter_java
    need(sha_file(a.h07_python)==a.h07_expected_sha256,'H07_REFERENCE_SHA256_MISMATCH')
    need(sha_file(a.h08_python)==a.h08_expected_sha256,'H08_REFERENCE_SHA256_MISMATCH')
    h07=load_ref(a.h07_python,'j03e_h07')
    h08=load_ref(a.h08_python,'j03e_h08')
    for attr in ('parse_strategy','direct_targets','direct_class_targets','pre_order_named'):
        need(callable(getattr(h07,attr,None)),'H07_FUNCTION_MISSING '+attr)
    for attr in ('grams','jaccard','harmonic'):
        need(callable(getattr(h08,attr,None)),'H08_FUNCTION_MISSING '+attr)
    need(hasattr(h08,'SequenceMatcher'),'H08_SEQUENCEMATCHER_MISSING')
    emit(a,'AST_REFERENCE_PASS',h07_sha256=sha_file(a.h07_python),h08_sha256=sha_file(a.h08_python),
         gram_size=a.ast_gram_size,threshold=a.negative_similarity_max)
    return Parser(Language(tree_sitter_java.language())),h07,h08


def ast_sequence(parser,h07,raw,mode):
    if mode=='direct_declaration':
        root=parser.parse(raw).root_node
        nodes=h07.direct_targets(root)
    elif mode=='method_in_class_wrapper':
        root=parser.parse(b'class H07SyntheticWrapper {\n'+raw+b'\n}\n').root_node
        nodes=h07.direct_class_targets(root)
    else:raise RuntimeError('AST_PARSE_MODE_UNSUPPORTED '+mode)
    need(len(nodes)==1,'AST_TARGET_RECONSTRUCTION_FAILED')
    return tuple(h07.pre_order_named(nodes[0]))


def read_fragments(a,subject,parser,h07,h08):
    path=a.positive_root/subject/a.fragment_name
    records={};eligible=[];invalid=Counter()
    for row in rows_jsonl(path):
        fid,code,stored_sha=row.get('fragment_id'),row.get('code'),row.get('source_sha256')
        need(isinstance(fid,str) and fid and isinstance(code,str) and isinstance(stored_sha,str) and len(stored_sha)==64,
             'INVALID_FRAGMENT_SCHEMA '+subject)
        need(fid not in records,'DUPLICATE_FRAGMENT_ID '+subject+' '+fid)
        raw=code.encode(a.code_encoding,errors=a.code_errors)
        need(digest(raw)==stored_sha,'FRAGMENT_SOURCE_HASH_MISMATCH '+subject+' '+fid)
        item={'id':fid,'sha':stored_sha,'file':row.get('file_path'),
              'start_line':row.get('start_line'),'end_line':row.get('end_line'),
              'commit':row.get('commit'),'eligible':False}
        try: normalized=raw.decode('utf-8',errors='strict')
        except UnicodeDecodeError:
            invalid['INVALID_UTF8']+=1
            records[fid]=item
            continue
        meta=h07.parse_strategy(parser,normalized) if normalized.strip() else {}
        if meta.get('parse_status')=='PASS' and meta.get('function_level_eligible'):
            seq=ast_sequence(parser,h07,normalized.encode('utf-8'),meta['parse_mode'])
            need(seq and digest(json.dumps(list(seq),separators=(',',':')).encode())==meta['ast_named_preorder_sha256'],
                 'AST_NAMED_PREORDER_HASH_MISMATCH '+subject+' '+fid)
            item.update({'eligible':True,'ast_sha':meta['ast_named_preorder_sha256'],
                         'seq':seq,'grams':h08.grams(seq,a.ast_gram_size),
                         't1':meta['type1_syntax_sha256'],'t2c':meta['type2_consistent_sha256'],
                         't2b':meta['type2_blind_sha256']})
            eligible.append(item)
        else: invalid[str(meta.get('record_kind','UNRESOLVED'))]+=1
        records[fid]=item
        if len(records)%a.progress_every==0:
            emit(a,'NEGATIVE_FRAGMENT_PROGRESS',subject=subject,fragments=len(records),eligible=len(eligible))
    # Count every unique source SHA, including invalid UTF-8 sources, for manifest provenance.
    all_sha={x['sha'] for x in records.values()}
    need(records and len(eligible)>=2,'INSUFFICIENT_ELIGIBLE_FUNCTIONS '+subject)
    emit(a,'NEGATIVE_FRAGMENT_PASS',subject=subject,fragments=len(records),eligible=len(eligible),
         distinct_code_hashes=len(all_sha),invalid=dict(invalid))
    return records,eligible,invalid


def validate_positive(a,subject,npos,records):
    pos=a.positive_root/subject
    source=read_json(pos/a.source_manifest_name)
    pm=read_json(pos/a.positive_details_dir/a.fixed_positive_manifest_name)
    need(source.get('files_sha256',{}).get(a.fragment_name)==sha_file(pos/a.fragment_name),
         'SOURCE_FRAGMENT_PROVENANCE_MISMATCH '+subject)
    need(source.get('files_sha256',{}).get(a.original_positive_csv_name)==sha_file(pos/a.original_positive_csv_name),
         'SOURCE_PAIRS_PROVENANCE_MISMATCH '+subject)
    need(pm.get('status')=='FIXED_POSITIVE_ONLY' and pm.get('subject')==subject
         and pm.get('fixed_positive_pairs')==npos,'J03D_POSITIVE_MANIFEST_MISMATCH '+subject)
    cfg=pm.get('effective_configuration',{})
    need(cfg.get('policy')==a.positive_policy and cfg.get('record_schema')==a.positive_schema,
         'J03D_POSITIVE_SCHEMA_OR_POLICY_MISMATCH '+subject)
    need(pm.get('h07_source_sha256')==a.h07_expected_sha256 and
         pm.get('h08_source_sha256')==a.h08_expected_sha256,
         'J03D_AST_REFERENCE_SHA256_MISMATCH '+subject)
    need(pm.get('source_manifest_sha256')==sha_file(pos/a.source_manifest_name) and
         pm.get('source_fragments_sha256')==source['files_sha256'][a.fragment_name] and
         pm.get('source_pairs_sha256')==source['files_sha256'][a.original_positive_csv_name],
         'J03D_POSITIVE_SOURCE_MISMATCH '+subject)
    pub=pos/a.fixed_positive_public_name
    details=pos/a.positive_details_dir/a.fixed_positive_details_name
    need(pm.get('files_sha256',{}).get(a.fixed_positive_public_name)==sha_file(pub) and
         pm.get('files_sha256',{}).get(a.positive_details_dir+'/'+a.fixed_positive_details_name)==sha_file(details),
         'J03D_POSITIVE_ARTIFACT_HASH_MISMATCH '+subject)
    return source,pm


def make_blocked_index(a,subject,records,runsubject,source,target):
    db=sqlite3.connect(runsubject/'pair_index.sqlite')
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=NORMAL')
    db.execute(f'PRAGMA cache_size=-{a.sqlite_cache_kib}')
    db.execute('CREATE TABLE blocked (key BLOB PRIMARY KEY) WITHOUT ROWID')
    db.execute('CREATE TABLE selected (key BLOB PRIMARY KEY) WITHOUT ROWID')
    path=a.positive_root/subject/a.original_positive_csv_name
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f,strict=True)
        required={'fragment_a_id','fragment_b_id','source_sha256_a','source_sha256_b',
                  'repository_a','repository_b','label'}
        need(required<=set(reader.fieldnames or []),'ORIGINAL_NICAD_COLUMNS_MISSING '+subject)
        count=0
        for row in reader:
            a_id,b_id=row['fragment_a_id'],row['fragment_b_id']
            need(a_id in records and b_id in records and a_id!=b_id,'NICAD_FRAGMENT_ID_INVALID '+subject)
            need(row['repository_a']==row['repository_b']==subject and row['label']==str(a.positive_label),
                 'NICAD_METADATA_INVALID '+subject)
            need(records[a_id]['sha']==row['source_sha256_a'] and records[b_id]['sha']==row['source_sha256_b'],
                 'NICAD_SOURCE_HASH_INVALID '+subject)
            db.execute('INSERT OR IGNORE INTO blocked VALUES (?)',
                       (pair_key(row['source_sha256_a'],row['source_sha256_b']),))
            count+=1
            if count%a.progress_every==0:
                db.commit()
                emit(a,'NEGATIVE_BLOCKED_PROGRESS',subject=subject,original_positive_pairs=count)
    db.commit()
    need(count==source['pairs'],'ORIGINAL_NICAD_COUNT_MISMATCH '+subject)
    fixed_count=0
    for obj in rows_jsonl(a.positive_root/subject/a.fixed_positive_public_name):
        need(obj.get('label')==a.positive_label and obj.get('repository')==subject,
             'J03D_POSITIVE_RECORD_INVALID '+subject)
        ids=(obj.get('left',{}).get('function_id'),obj.get('right',{}).get('function_id'))
        need(all(fid in records for fid in ids) and ids[0]!=ids[1],
             'J03D_POSITIVE_FUNCTION_ID_UNRESOLVED '+subject)
        k=pair_key(records[ids[0]]['sha'],records[ids[1]]['sha'])
        need(db.execute('SELECT 1 FROM blocked WHERE key=?',(k,)).fetchone() is not None,
             'FIXED_POSITIVE_NOT_PRESENT_IN_ORIGINAL_NICAD '+subject)
        fixed_count+=1
        if fixed_count%a.progress_every==0:
            emit(a,'NEGATIVE_FIXED_POSITIVE_CHECK_PROGRESS',subject=subject,pairs=fixed_count)
    need(fixed_count==target,'J03D_POSITIVE_COUNT_MISMATCH '+subject)
    emit(a,'NEGATIVE_BLOCKED_PASS',subject=subject,original_positive_pairs=count,
         fixed_positive_pairs=fixed_count,
         distinct_blocked_source_pairs=db.execute('SELECT COUNT(*) FROM blocked').fetchone()[0])
    return db


def unrank_pair(index,n):
    need(n>=2 and 0<=index<n*(n-1)//2,'PAIR_INDEX_INVALID')
    lo,hi=0,n-2
    while lo<hi:
        mid=(lo+hi+1)//2
        prefix=mid*(2*n-mid-1)//2
        if prefix<=index:lo=mid
        else:hi=mid-1
    i=lo;prefix=i*(2*n-i-1)//2
    return i,i+1+index-prefix


def pair_permutation(n,seed,subject):
    total=n*(n-1)//2
    need(total>0,'NO_UNIQUE_FRAGMENT_PAIRS')
    rng=random.Random(f'{seed}:{subject}')
    offset=rng.randrange(total)
    if total==1: stride=1
    else:
        stride=rng.randrange(1,total)
        while math.gcd(stride,total)!=1:stride=rng.randrange(1,total)
    return total,offset,stride


class Scorer:
    def __init__(self,a,h08,eligible):
        self.a=a;self.h08=h08;self.by_ast={}
        for f in eligible:self.by_ast.setdefault(f['ast_sha'],f)
        self._cached=functools.lru_cache(maxsize=a.ast_score_cache_size)(self._combined)
    def _combined(self,ha,hb):
        x,y=self.by_ast[ha],self.by_ast[hb]
        if len(x['seq'])*len(y['seq'])>self.a.ast_max_cells:return None
        jac=self.h08.jaccard(x['grams'],y['grams'])
        ratio=self.h08.SequenceMatcher(None,x['seq'],y['seq'],autojunk=False).ratio()
        return jac,ratio,self.h08.harmonic(jac,ratio)
    def check(self,x,y):
        if x['t1']==y['t1'] or x['t2c']==y['t2c'] or x['t2b']==y['t2b']:
            return 'SIGNATURE_CLONE',None
        if x['ast_sha']==y['ast_sha']:
            return 'SIMILARITY_HIGH',1.0
        key=sorted((x['ast_sha'],y['ast_sha']))
        result=self._cached(key[0],key[1])
        if result is None:return 'UNSCORED',None
        score=result[2]
        if score>=self.a.negative_similarity_max:return 'SIMILARITY_HIGH',score
        return 'ACCEPT',score
    def stats(self):
        x=self._cached.cache_info()
        return {'score_cache_hits':x.hits,'score_cache_misses':x.misses,'score_cache_maxsize':x.maxsize}


def evaluate_index(a,subject,eligible,db,scorer,target,probe_only,stage=None):
    total,offset,stride=pair_permutation(len(eligible),a.sample_seed,subject)
    max_attempts=min(total,a.feasibility_probe_pairs if probe_only else
                     min(a.attempt_limit_cap,target*a.attempts_per_target))
    counts=Counter();start=time.monotonic()
    if not probe_only:
        need(stage is not None,'MISSING_GENERATION_STAGE')
        (stage/a.detail_dir).mkdir(parents=True,exist_ok=False)
        pub=(stage/a.public_name).open('x',encoding='utf-8')
        detail=(stage/a.detail_dir/a.detail_name).open('x',encoding='utf-8')
    else:pub=detail=None
    try:
        for attempt in range(max_attempts):
            idx=(offset+attempt*stride)%total
            i,j=unrank_pair(idx,len(eligible))
            x,y=eligible[i],eligible[j]
            counts['attempts']+=1
            if x['sha']==y['sha']:
                counts['SAME_CODE']+=1;continue
            key=pair_key(x['sha'],y['sha'])
            if db.execute('SELECT 1 FROM blocked WHERE key=?',(key,)).fetchone():
                counts['KNOWN_NICAD_POSITIVE']+=1;continue
            if db.execute('SELECT 1 FROM selected WHERE key=?',(key,)).fetchone():
                counts['DUPLICATE_NEGATIVE_CONTENT']+=1;continue
            status,score=scorer.check(x,y)
            if status!='ACCEPT':counts[status]+=1;continue
            counts['accepted']+=1
            db.execute('INSERT INTO selected VALUES (?)',(key,))
            if not probe_only:
                left,right=x['id'],y['id']
                sample_id='neg-'+digest((subject+'\0'+left+'\0'+right).encode())[:24]
                common={'sample_id':sample_id,'repository':subject,'clone_type':'Non-clone',
                        'label':a.negative_label,'left':{'function_id':left},'right':{'function_id':right}}
                more={**common,'ast_combined_similarity':round(score,12),
                      'ast_similarity_threshold_exclusive':a.negative_similarity_max,
                      'source_a':{'fragment_id':left,'file':x['file'],'start_line':x['start_line'],
                                  'end_line':x['end_line'],'commit':x['commit'],'source_sha256':x['sha']},
                      'source_b':{'fragment_id':right,'file':y['file'],'start_line':y['start_line'],
                                  'end_line':y['end_line'],'commit':y['commit'],'source_sha256':y['sha']}}
                pub.write(json_line(common));detail.write(json_line(more))
            if counts['attempts']%a.progress_every==0 or counts['accepted']%a.progress_every==0 and counts['accepted']:
                db.commit()
                emit(a,'NEGATIVE_SAMPLE_PROGRESS',subject=subject,attempts=counts['attempts'],
                     accepted=counts['accepted'],target=target)
            if not probe_only and counts['accepted']==target:break
        db.commit()
    finally:
        if pub is not None:pub.close()
        if detail is not None:detail.close()
    counts['elapsed_seconds']=round(time.monotonic()-start,3)
    counts['candidate_pairs_by_function_id']=total
    counts['max_attempts']=max_attempts
    counts.update(scorer.stats())
    if probe_only:
        counts['estimated_accepted_from_probe']=round(total*counts['accepted']/counts['attempts']) if counts['attempts'] else 0
        counts['estimate_is_not_a_capacity_guarantee']=True
    return dict(counts)


def effective_config(a):
    return {'experiment':a.experiment,'version':a.version,'mode':a.mode,
            'positive_root':str(a.positive_root),'negative_root':str(a.negative_root),
            'reference_summary_name':a.reference_summary_name,
            'reference_summary_sha256':a.reference_summary_sha256,
            'positive_selection_policy':a.positive_policy,'positive_record_schema':a.positive_schema,
            'negative_similarity_exclusive':a.negative_similarity_max,
            'sampling_strategy':'seeded_coprime_affine_permutation_of_unordered_function_pairs',
            'sample_seed':a.sample_seed,'ast_gram_size':a.ast_gram_size,
            'positive_ast_threshold':a.positive_ast_threshold,
            'h07_expected_sha256':a.h07_expected_sha256,
            'h08_expected_sha256':a.h08_expected_sha256,
            'ast_max_cells':a.ast_max_cells,'ast_score_cache_size':a.ast_score_cache_size,
            'attempts_per_target':a.attempts_per_target,'attempt_limit_cap':a.attempt_limit_cap,
            'feasibility_probe_pairs':a.feasibility_probe_pairs,'progress_every':a.progress_every,
            'sqlite_cache_kib':a.sqlite_cache_kib,'time_zone':a.time_zone,
            'parser_version':a.parser_version,'grammar_version':a.grammar_version,
            'code_encoding':a.code_encoding,'code_errors':a.code_errors,
            'h07_sha256':sha_file(a.h07_python),'h08_sha256':sha_file(a.h08_python),
            'public_name':a.public_name,'detail_dir':a.detail_dir,'detail_name':a.detail_name,
            'source_manifest_name':a.source_manifest_name,'fragment_name':a.fragment_name,
            'original_positive_csv_name':a.original_positive_csv_name,
            'fixed_positive_public_name':a.fixed_positive_public_name,
            'fixed_positive_details_name':a.fixed_positive_details_name,
            'fixed_positive_manifest_name':a.fixed_positive_manifest_name,
            'positive_details_dir':a.positive_details_dir,
            'manifest_name':a.manifest_name,'report_name':a.report_name,
            'positive_label':a.positive_label,'negative_label':a.negative_label,
            'expected_subjects':a.expected_subjects,'expected_total_positives':a.expected_total_positives}


def recorded_configuration_identity(a, recorded, current):
    """Permit only the two-root relocation; reject any other config drift."""
    if recorded == current:
        return 'CURRENT'
    legacy = dict(current)
    legacy['positive_root'] = str(a.legacy_positive_root)
    legacy['negative_root'] = str(a.legacy_negative_root)
    if recorded == legacy:
        return 'LEGACY_RELOCATED'
    need(False, 'NEGATIVE_EFFECTIVE_CONFIGURATION_DRIFT')


def expected_paths(a,subject):
    root=a.negative_root/subject
    return root,root/a.public_name,root/a.detail_dir/a.detail_name,root/a.detail_dir/a.manifest_name


def verify_output(a,subject,target,records,db,scorer,pm,config,root_override=None):
    root,pub,detail,mp=expected_paths(a,subject)
    if root_override is not None:
        root=root_override
        pub=root/a.public_name
        detail=root/a.detail_dir/a.detail_name
        mp=root/a.detail_dir/a.manifest_name
    m=read_json(mp)
    need(m.get('status')=='FIXED_NEGATIVE_CANDIDATES' and m.get('subject')==subject,
         'NEGATIVE_MANIFEST_INVALID '+subject)
    layout_identity = recorded_configuration_identity(
        a, m.get('effective_configuration'), {**config, 'mode':'negativeclone'})
    if layout_identity == 'LEGACY_RELOCATED':
        emit(a, 'NEGATIVE_RELOCATED_MANIFEST_PASS', subject=subject,
             old_root=a.legacy_negative_root, current_root=a.negative_root)
    need(m.get('positive_manifest_sha256')==sha_file(a.positive_root/subject/a.positive_details_dir/a.fixed_positive_manifest_name),
         'NEGATIVE_INPUT_POSITIVE_MANIFEST_DRIFT '+subject)
    need(m.get('positive_fixed_pairs')==target and m.get('fixed_negative_pairs')==target,
         'NEGATIVE_COUNT_MANIFEST_INVALID '+subject)
    need(m['files_sha256'].get(a.public_name)==sha_file(pub) and
         m['files_sha256'].get(a.detail_dir+'/'+a.detail_name)==sha_file(detail),
         'NEGATIVE_FILE_SHA256_INVALID '+subject)
    count=0;ids=set()
    with pub.open(encoding='utf-8') as f1,detail.open(encoding='utf-8') as f2:
        for line1,line2 in zip(f1,f2):
            x=json.loads(line1);y=json.loads(line2)
            need(set(x)=={'sample_id','repository','clone_type','label','left','right'},
                 'NEGATIVE_PUBLIC_SCHEMA_INVALID '+subject)
            need(all(x.get(k)==y.get(k) for k in x) and x['label']==a.negative_label and
                 x['repository']==subject and x['clone_type']=='Non-clone',
                 'NEGATIVE_PUBLIC_DETAILS_JOIN_FAIL '+subject)
            left=x['left']['function_id'];right=x['right']['function_id']
            need(left in records and right in records and left!=right and
                 records[left]['eligible'] and records[right]['eligible'],
                 'NEGATIVE_FUNCTION_ID_INVALID '+subject)
            fl,fr=records[left],records[right]
            need(fl['sha']!=fr['sha'],'NEGATIVE_EQUAL_CODE_CONTENT '+subject)
            key=pair_key(fl['sha'],fr['sha'])
            need(db.execute('SELECT 1 FROM blocked WHERE key=?',(key,)).fetchone() is None,
                 'NEGATIVE_OVERLAPS_ORIGINAL_NICAD '+subject)
            need(key not in ids,'NEGATIVE_DUPLICATE_CONTENT_PAIR '+subject)
            ids.add(key)
            status,score=scorer.check(fl,fr)
            need(status=='ACCEPT' and score is not None and score<a.negative_similarity_max,
                 'NEGATIVE_SIMILARITY_VERIFICATION_FAIL '+subject)
            need(y.get('ast_combined_similarity')==round(score,12),
                 'NEGATIVE_RECORDED_SIMILARITY_DRIFT '+subject)
            need(y.get('ast_similarity_threshold_exclusive')==a.negative_similarity_max,
                 'NEGATIVE_RECORDED_THRESHOLD_DRIFT '+subject)
            need(y['source_a']['source_sha256']==fl['sha'] and y['source_b']['source_sha256']==fr['sha'],
                 'NEGATIVE_SOURCE_PROVENANCE_DRIFT '+subject)
            need(x['sample_id']=='neg-'+digest((subject+'\0'+left+'\0'+right).encode())[:24],
                 'NEGATIVE_SAMPLE_ID_DRIFT '+subject)
            count+=1
            if count%a.progress_every==0:emit(a,'NEGATIVE_VERIFY_PROGRESS',subject=subject,pairs=count)
        need(f1.readline()=='' and f2.readline()=='','NEGATIVE_DETAIL_PUBLIC_LENGTH_MISMATCH '+subject)
    need(count==target,'NEGATIVE_VERIFY_COUNT_MISMATCH '+subject)
    emit(a,'NEGATIVE_VERIFY_PASS',subject=subject,positive=target,negative=count,threshold=a.negative_similarity_max)
    return m


def subject_task(a,subject,target,parser,h07,h08,config):
    started=time.monotonic();runsubject=a.run_dir/subject
    root,pub,detail,mp=expected_paths(a,subject)
    existing_now=[p.exists() or p.is_symlink() for p in (pub,detail,mp)]
    if a.mode=='negativeclone-verify':
        need(all(existing_now),'NEGATIVE_OUTPUT_MISSING_OR_PARTIAL '+subject)
    if a.mode=='negativeclone' and (any(existing_now) and not all(existing_now) or
                                    root.exists() and not all(existing_now) or root.is_symlink()):
        raise RuntimeError('NEGATIVE_PARTIAL_OR_UNSAFE_OUTPUT_EXISTS '+subject)
    runsubject.mkdir(mode=0o755,exist_ok=False)
    records,eligible,invalid=read_fragments(a,subject,parser,h07,h08)
    source,pm=validate_positive(a,subject,target,records)
    db=make_blocked_index(a,subject,records,runsubject,source,target)
    scorer=Scorer(a,h08,eligible)
    try:
        _,pub,detail,mp=expected_paths(a,subject)
        paths=[pub,detail,mp]
        existing=[p.exists() or p.is_symlink() for p in paths]
        if a.mode=='negativeclone-verify' or (a.mode=='negativeclone' and all(existing)):
            need(all(existing),'NEGATIVE_OUTPUT_MISSING_OR_PARTIAL '+subject)
            m=verify_output(a,subject,target,records,db,scorer,pm,config)
            return {'subject':subject,'target':target,'accepted':target,
                    'status':'VERIFIED_EXISTING','duration_seconds':round(time.monotonic()-started,3),
                    'manifest_sha256':sha_file(mp)}
        if a.mode=='negativeclone':
            need(not any(existing),'NEGATIVE_PARTIAL_OUTPUT_EXISTS '+subject)
        ncontent=len({f['sha'] for f in eligible})
        upper=ncontent*(ncontent-1)//2
        if upper<target and a.mode=='negativeclone':
            raise RuntimeError(f'NEGATIVE_CONTENT_CAPACITY_IMPOSSIBLE subject={subject} '
                               f'target={target} distinct_content_pairs_upper_bound={upper}')
        stage=runsubject/'publish' if a.mode=='negativeclone' else None
        if stage:stage.mkdir(mode=0o755)
        result=evaluate_index(a,subject,eligible,db,scorer,target,
                              a.mode=='negativeclone-feasibility',stage)
        result.update({'subject':subject,'target':target,'eligible_fragments':len(eligible),
                       'distinct_eligible_content_hashes':ncontent,
                       'distinct_content_pairs_upper_bound':upper,
                       'source_fragments_sha256':source['files_sha256'][a.fragment_name],
                       'positive_manifest_sha256':sha_file(a.positive_root/subject/a.positive_details_dir/a.fixed_positive_manifest_name)})
        if a.mode=='negativeclone-feasibility':
            result['status']='UPPER_BOUND_IMPOSSIBLE' if upper<target else 'PROBE_ONLY_NOT_CERTIFIED'
            (runsubject/'feasibility.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
            emit(a,'NEGATIVE_FEASIBILITY_PASS',subject=subject,probe_accepted=result['accepted'],
                 probe_attempts=result['attempts'],content_pair_upper_bound=upper,target=target)
            return result
        if result['accepted']!=target:
            result['status']='FAIL_INSUFFICIENT_CANDIDATES_OR_ATTEMPT_BUDGET'
            (runsubject/'failure_diagnostics.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
            raise RuntimeError(f'NEGATIVE_SAMPLING_EXHAUSTED subject={subject} '
                               f'accepted={result["accepted"]} target={target} '
                               f'attempts={result["attempts"]} max_attempts={result["max_attempts"]}; '
                               f'diagnostics={runsubject / "failure_diagnostics.json"}')
        manifest={'experiment':a.experiment,'version':a.version,
                  'script_version':a.script_version,'subject':subject,
                  'status':'FIXED_NEGATIVE_CANDIDATES','generated_at_chicago':clock(a),
                  'effective_configuration':{**config,'mode':'negativeclone'},
                  'positive_fixed_pairs':target,'fixed_negative_pairs':result['accepted'],
                  'source_manifest_sha256':sha_file(a.positive_root/subject/a.source_manifest_name),
                  'positive_manifest_sha256':result['positive_manifest_sha256'],
                  'source_fragments_sha256':source['files_sha256'][a.fragment_name],
                  'original_nicad_positive_pairs':source['pairs'],
                  'sampling_statistics':result,
                  'files_sha256':{a.public_name:sha_file(stage/a.public_name),
                     a.detail_dir+'/'+a.detail_name:sha_file(stage/a.detail_dir/a.detail_name)},
                  'not_semantic_gold':True}
        (stage/a.detail_dir/a.manifest_name).write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        root,_,_,_=expected_paths(a,subject)
        need(not root.exists() and not root.is_symlink(),'NEGATIVE_PUBLICATION_DEST_ALREADY_EXISTS '+subject)
        # Atomic directory move on same filesystem. Stage lives below results/ and output under data/.
        # Source/target must reside on same filesystem; otherwise fail closed.
        need(stage.stat().st_dev==a.negative_root.stat().st_dev,'STAGING_AND_OUTPUT_DIFFERENT_FILESYSTEM')
        verify_output(a,subject,target,records,db,scorer,pm,config,root_override=stage)
        os.rename(stage,root)
        emit(a,'NEGATIVE_OUTPUT_PUBLISHED',subject=subject,path=root)
        result['status']='PASS_FIXED_NEGATIVE_CANDIDATES'
        result['manifest_sha256']=sha_file(root/a.detail_dir/a.manifest_name)
        return result
    finally:db.close()


def write_summary(a,results,subject_targets):
    summary={'experiment':a.experiment,'version':a.version,
             'script_version':a.script_version,'mode':a.mode,
             'generated_at_chicago':clock(a),
             'status':('FEASIBILITY_HARD_BLOCKER' if any(r.get('status')=='UPPER_BOUND_IMPOSSIBLE' for r in results)
                       else 'FEASIBILITY_PROBE_ONLY') if a.mode=='negativeclone-feasibility' else 'PASS',
             'subjects':len(results),'expected_subjects':a.expected_subjects,
             'target_negatives':sum(subject_targets.values()),
             'accepted_negatives':(0 if a.mode=='negativeclone-feasibility' else sum(r.get('accepted',0) for r in results)),
             'probe_accepted':(sum(r.get('accepted',0) for r in results) if a.mode=='negativeclone-feasibility' else 0),
             'reference_summary_sha256':a.reference_summary_sha256,
             'effective_configuration':effective_config(a),
             'subject_results':results}
    filename='summary-'+a.mode+'.json'
    p=a.run_dir/filename
    with p.open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2,sort_keys=True);f.write('\n')
    if a.mode!='negativeclone-feasibility':
        md=['# J-03e v2 — Fixed Negative Clone Candidates','',
            f'Generated (America/Chicago): {clock(a)}','',
            f'Command: `--{a.mode}`','',
            f'Subjects in this run: {len(results)}','',
            '| Subject | N-pos | N-neg | Status |','|---|---:|---:|---|']
        for r in results:
            md.append(f'| {r["subject"]} | {r["target"]:,} | {r["accepted"]:,} | {r["status"]} |')
        md+=['',f'**Run totals: N-pos={summary["target_negatives"]:,}, N-neg={summary["accepted_negatives"]:,}.**','',
             f'All accepted candidates satisfy H-08 AST combined similarity < {a.negative_similarity_max}.',
             'These are automatically constructed negative candidates, not human-certified semantic gold.',
             'No training/validation/test splits or positive files have been modified.','']
        report=a.run_dir/a.report_name
        report.write_text('\n'.join(md),encoding='utf-8')
        if a.subject=='ALL' and a.mode=='negativeclone':
            destination=a.negative_root/a.report_name
            if not destination.exists() and not destination.is_symlink():
                os.link(report,destination)  # fail closed if a concurrent writer published it
    emit(a,'NEGATIVE_FINAL_PASS',mode=a.mode,subjects=len(results),
         target=summary['target_negatives'],accepted=summary['accepted_negatives'],
         probe_accepted=summary['probe_accepted'],status=summary['status'],report=p)


def main():
    a=parser_args();start=time.monotonic()
    targets=load_summary(a)
    parser,h07,h08=ast_setup(a)
    config=effective_config(a)
    emit(a,'NEGATIVE_RUN_START',mode=a.mode,subjects=len(targets),root=a.negative_root,
         run=a.run_dir,seed=a.sample_seed,threshold=a.negative_similarity_max)
    results=[]
    for i,(subject,target) in enumerate(targets.items(),1):
        emit(a,'NEGATIVE_SUBJECT_START',index=f'{i}/{len(targets)}',subject=subject,target=target)
        try:r=subject_task(a,subject,target,parser,h07,h08,config)
        except Exception as err:
            emit(a,'NEGATIVE_SUBJECT_FAIL',subject=subject,reason=repr(str(err)),run=a.run_dir)
            raise
        results.append(r)
        emit(a,'NEGATIVE_SUBJECT_PASS',subject=subject,status=r['status'],
             accepted=r['accepted'],target=target)
    write_summary(a,results,targets)
    if a.mode=='negativeclone-feasibility' and any(x.get('status')=='UPPER_BOUND_IMPOSSIBLE' for x in results):
        raise RuntimeError('FEASIBILITY_HARD_BLOCKER: exact N-neg=N-pos impossible for at least one subject')
    emit(a,'PASS',experiment=a.experiment,version=a.version,mode=a.mode,
         elapsed_sec=round(time.monotonic()-start,3))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('NEGATIVE_FINAL_FAIL '+repr(str(exc)),file=sys.stderr,flush=True)
        raise
