#!/usr/bin/env python3
"""J-04d v4: verified J-04c AFTER hybrid pairs -> function-linked positives.

Only Steps 1-3: verify source, join every candidate, extract and hash source.
No AST clone-type classification, positive selection, deduplication or publishing.
All experiment configuration must come from the standalone Shell through CLI.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

PAIR_COLUMNS = (
    'pair_id','fragment_a_id','repository_a','file_a','start_line_a','end_line_a',
    'commit_a','source_sha256_a','fragment_b_id','repository_b','file_b',
    'start_line_b','end_line_b','commit_b','source_sha256_b','same_repository',
    'nicad_similarity_raw','nicad_nlines','detector','config','xml_sha256',
    'upstream_pair_id','function_a','function_b','ni_parse_failed_a',
    'ni_parse_failed_b','label',
)
UPSTREAM_COLUMNS = (
    'subject','pair_id','file_a','start_a','end_a','function_a',
    'file_b','start_b','end_b','function_b','source',
    'ni_parse_failed_a','ni_parse_failed_b',
)
INVENTORY_COLUMNS = ('file','bytes','sha256')
SUBJECT_RE = re.compile(r'^[a-z0-9][a-z0-9_.-]*__[a-z0-9][a-z0-9_.-]*$')
HEX_RE = re.compile(r'^[0-9a-f]{64}$')


def check(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def digest_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def timestamp(a):
    return datetime.now(ZoneInfo(a.timezone)).isoformat()


def read_json(path):
    check(path.is_file() and not path.is_symlink(), 'MISSING_OR_UNSAFE_JSON '+str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, obj):
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write('\n')


def read_csv(path, columns):
    check(path.is_file() and not path.is_symlink(), 'MISSING_OR_UNSAFE_CSV '+str(path))
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        reader = csv.DictReader(f)
        check(tuple(reader.fieldnames or ())==tuple(columns), 'BAD_CSV_SCHEMA '+str(path))
        return list(reader)


def safe_rel(value, suffix=None):
    check(isinstance(value,str) and bool(value) and '\\' not in value and '\x00' not in value,
          'INVALID_RELATIVE_PATH '+repr(value))
    p=PurePosixPath(value)
    check(not p.is_absolute() and str(p)==value and
          all(part not in ('','.','..') for part in value.split('/')) and
          (suffix is None or p.suffix==suffix), 'INVALID_RELATIVE_PATH '+repr(value))
    return Path(*p.parts)


def regular(path, root):
    check(root.is_dir() and not root.is_symlink(),'UNSAFE_ROOT '+str(root))
    check(root==path or root in path.parents, 'PATH_ESCAPES_ROOT '+str(path))
    cur=path
    while cur!=root:
        check(not cur.is_symlink(),'SYMLINK_FORBIDDEN '+str(cur))
        cur=cur.parent
    check(path.is_file() and not path.is_symlink(),'MISSING_FILE '+str(path))
    return path


def files_under(root):
    check(root.is_dir() and not root.is_symlink(), 'UNSAFE_SUBJECT_ROOT '+str(root))
    found={}
    for base,dirs,files in os.walk(root,followlinks=False):
        b=Path(base)
        for name in dirs:
            check(not (b/name).is_symlink(),'SYMLINK_DIR '+str(b/name))
        for name in files:
            path=b/name
            check(path.is_file() and not path.is_symlink(),'UNSAFE_FILE '+str(path))
            rel=path.relative_to(root).as_posix()
            check(rel not in found,'DUPLICATE_PATH '+rel)
            found[rel]=path
    return found


def load_source(a):
    root=a.input_root
    check(root.is_dir() and not root.is_symlink(),'J04C_PUBLISHED_ROOT_REQUIRED')
    receipt_path=root/a.copy_manifest_name
    batch_path=root/a.batch_summary_name
    receipt=read_json(receipt_path)
    batch=read_json(batch_path)
    check(receipt.get('schema')=='deepknowcode-j04c-copy-'+a.input_version and
          receipt.get('experiment')==a.input_experiment and
          receipt.get('version')==a.input_version and
          receipt.get('status')=='COMPLETED' and
          receipt.get('source_data_version')==a.source_data_version and
          receipt.get('published_subjects')==a.expected_subjects and
          receipt.get('source_files')==a.expected_files and
          receipt.get('clone_pairs_total')==a.expected_pairs,
          'PUBLISH_RECEIPT_IDENTITY_OR_COUNTS_MISMATCH')
    check(receipt.get('published_root')==str(root) and
          receipt.get('source_all_result')==batch.get('result_root') and
          batch.get('publish_root')==str(root),
          'J04C_PUBLISH_PROVENANCE_MISMATCH')
    check(batch.get('experiment')==a.input_experiment and
          batch.get('version')==a.input_version and
          batch.get('batch_mode')=='all' and
          batch.get('status')=='COMPLETED' and
          batch.get('source_data_version')==a.source_data_version and
          batch.get('completed_subjects')==a.expected_subjects and
          batch.get('input_files_total')==a.expected_files and
          batch.get('clone_pairs_total')==a.expected_pairs and
          digest_file(batch_path)==receipt.get('source_batch_summary_sha256'),
          'J04C_BATCH_SUMMARY_MISMATCH')
    records=receipt.get('subjects')
    batch_records=batch.get('subjects')
    check(isinstance(records,list) and isinstance(batch_records,list) and
          len(records)==len(batch_records)==a.expected_subjects,
          'J04C_SUBJECT_RECORD_COUNT_MISMATCH')
    names=[r.get('subject') for r in records]
    check(all(isinstance(n,str) and SUBJECT_RE.fullmatch(n) for n in names) and
          len(set(names))==a.expected_subjects and
          {r.get('subject') for r in batch_records}==set(names),
          'J04C_SUBJECT_SET_MISMATCH')
    check(set(d.name for d in root.iterdir() if d.is_dir())==set(names),
          'J04C_SUBJECT_DIRECTORY_SET_MISMATCH')
    by_batch={r['subject']:r for r in batch_records}
    check(sum(int(r['source_files']) for r in records)==a.expected_files and
          sum(int(r['clone_pairs']) for r in records)==a.expected_pairs,
          'J04C_SUBJECT_AGGREGATES_MISMATCH')
    for r in records:
        b=by_batch[r['subject']]
        check(b.get('source_files')==r['source_files'] and
              b.get('clone_pairs')==r['clone_pairs'] and
              b.get('status')=='COMPLETED', 'BATCH_RECEIPT_SUBJECT_MISMATCH '+r['subject'])
    return receipt,batch,{r['subject']:r for r in records},digest_file(receipt_path)


def validate_subject(a,subject,entry,*,full_hash=True):
    """Validate J04c publish/subject invariants, including exact published file inventory."""
    where=a.input_root/subject
    check(where.is_dir() and not where.is_symlink(),'J04C_SUBJECT_MISSING '+subject)
    expected=entry.get('copy_file_sha256')
    check(isinstance(expected,dict) and bool(expected) and
          len(expected)==entry.get('files_copied'), 'COPY_HASH_TABLE_INVALID '+subject)
    for rel,sha in expected.items():
        safe_rel(rel)
        check(isinstance(sha,str) and HEX_RE.fullmatch(sha), 'COPY_SHA_INVALID '+subject+':'+rel)
    actual=files_under(where)
    check(set(actual)==set(expected),'PUBLISHED_FILE_SET_MISMATCH '+subject)
    if full_hash:
        for idx,(rel,file) in enumerate(sorted(actual.items()),1):
            check(digest_file(file)==expected[rel],'PUBLISHED_SHA256_MISMATCH '+subject+':'+rel)
            if idx%a.hash_progress_every==0:
                print(f'[{timestamp(a)}] HASH_PROGRESS subject={subject} files={idx}/{len(expected)}',flush=True)
    run=read_json(where/a.source_run_manifest_name)
    check(run.get('schema')=='deepknowcode-j04c-'+a.input_version and
          run.get('experiment')==a.input_experiment and
          run.get('version')==a.input_version and
          run.get('status')=='COMPLETED' and run.get('subject')==subject and
          run.get('input_data_version')==a.source_data_version and
          run.get('source_files')==entry['source_files'] and
          run.get('metrics',{}).get('hybrid_unique_pairs')==entry['clone_pairs'] and
          digest_file(where/a.source_run_manifest_name)==entry['j04c_run_sha256'],
          'J04C_RUN_MANIFEST_MISMATCH '+subject)
    inv=read_csv(where/a.input_inventory_name,INVENTORY_COLUMNS)
    check(len(inv)==entry['source_files'], 'J04C_INVENTORY_COUNT_MISMATCH '+subject)
    stage_root=where/a.stage_dir_name
    check(stage_root.is_dir() and not stage_root.is_symlink(),'J04C_STAGE_MISSING '+subject)
    children=list(stage_root.iterdir())
    check(len(children)==1 and children[0].is_dir() and not children[0].is_symlink(),
          'J04C_STAGE_LAYOUT_MISMATCH '+subject)
    stage=children[0]
    observed={}
    for row in inv:
        rel=row['file']
        safe_rel(rel,'.py')
        check(rel not in observed and HEX_RE.fullmatch(row['sha256']) and
              int(row['bytes'])>=0,'J04C_INPUT_INVENTORY_INVALID '+subject+':'+rel)
        observed[rel]=row
        path=regular(stage/safe_rel(rel,'.py'),stage)
        check(path.stat().st_size==int(row['bytes']) and
              expected.get((Path(a.stage_dir_name)/stage.name/safe_rel(rel)).as_posix())==row['sha256'],
              'J04C_STAGED_INPUT_IDENTITY_MISMATCH '+subject+':'+rel)
    check({p.relative_to(stage).as_posix() for p in files_under(stage).values()}==set(observed),
          'J04C_STAGED_INPUT_SET_MISMATCH '+subject)
    pairs=read_csv(where/a.source_pair_csv_name,UPSTREAM_COLUMNS)
    check(len(pairs)==entry['clone_pairs'],'J04C_PAIR_COUNT_MISMATCH '+subject)
    seen=set()
    for row in pairs:
        check(row['subject']==subject and row['source'] in a.allowed_detector_sources and
              row['file_a'] in observed and row['file_b'] in observed,
              'J04C_PAIR_SOURCE_INVALID '+subject+':'+row['pair_id'])
        check(row['pair_id'] not in seen and bool(row['pair_id']),
              'J04C_DUPLICATE_PAIR_ID '+subject+':'+row['pair_id'])
        seen.add(row['pair_id'])
    summary=read_json(where/a.source_hybrid_summary_name)
    check(summary.get('hybrid_unique_pairs')==entry['clone_pairs'] and
          summary.get('staged_files')==entry['source_files'] and
          summary.get('python_ast_errors')==0,
          'J04C_HYBRID_SUMMARY_MISMATCH '+subject)
    return where,stage,inv,pairs,run


def ast_index(raw,rel):
    tree=ast.parse(raw,filename=rel)
    result={}
    stack=[(tree,())]
    while stack:
        node,scope=stack.pop()
        child_scope=scope
        if isinstance(node,ast.ClassDef):
            child_scope=scope+(node.name,)
        elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            beginning=min([node.lineno]+[d.lineno for d in node.decorator_list])
            key=(beginning,node.end_lineno,'.'.join(scope+(node.name,)))
            check(key not in result,'AST_DUPLICATE_FUNCTION_KEY '+rel+':'+str(key))
            result[key]=node
            child_scope=scope+(node.name,)
        for child in reversed(list(ast.iter_child_nodes(node))):
            stack.append((child,child_scope))
    return result


def fragment_for(a,subject,stage,git_head,file,begin,end,name,cache):
    key=(file,begin,end,name)
    if key in cache['fragments']:
        return cache['fragments'][key]
    path=regular(stage/safe_rel(file,'.py'),stage)
    if file not in cache['files']:
        raw=path.read_bytes()
        index=ast_index(raw,file)
        cache['files'][file]=(raw.splitlines(keepends=True),index,digest_bytes(raw))
        if len(cache['files'])>a.source_cache_files:
            cache['files'].popitem(last=False)
    else:
        cache['files'].move_to_end(file)
    lines,index,file_sha=cache['files'][file]
    check((begin,end,name) in index,'FUNCTION_AST_IDENTITY_MISMATCH '+subject+':'+file+':'+str((begin,end,name)))
    check(1<=begin<=end<=len(lines),'FUNCTION_LINE_RANGE_INVALID '+subject+':'+file)
    raw=b''.join(lines[begin-1:end])
    check(bool(raw),'EMPTY_FUNCTION_SOURCE '+subject+':'+file)
    fid='fn-'+digest_bytes((subject+'\0'+file+'\0'+str(begin)+'\0'+str(end)+'\0'+name).encode())[:24]
    code=raw.decode(a.code_encoding,errors=a.code_errors)
    check(code.encode(a.code_encoding,errors=a.code_errors)==raw,
          'SOURCE_ENCODING_ROUNDTRIP_FAILED '+subject+':'+file)
    record={'fragment_id':fid,'repository':subject,'file_path':file,
            'start_line':begin,'end_line':end,'commit':git_head,
            'source_sha256':digest_bytes(raw),'language':a.language,
            'function_name':name,'file_sha256':file_sha,'code':code}
    cache['fragments'][key]=record
    return record


def produce_subject(a,subject,entry,out):
    _,stage,inventory,pairs,run=validate_subject(a,subject,entry)
    check(not out.exists() and not out.is_symlink(),'J04D_OUTPUT_EXISTS '+str(out))
    out.mkdir(parents=True,exist_ok=False)
    cache={'files':OrderedDict(),'fragments':{}}
    rows=[]
    seen_pairs=set()
    commit=run['subject_git_head']
    artifacts=run['artifacts_sha256']
    xml_hashes=[sha for rel,sha in artifacts.items() if rel.startswith('nicad_xml/') and rel.endswith('.xml')]
    check(len(xml_hashes)==1 and all(HEX_RE.fullmatch(x) for x in xml_hashes),
          'J04C_NICAD_XML_PROVENANCE_MISSING '+subject)
    jscpd_report_sha=artifacts.get('jscpd/jscpd-report.json')
    check(isinstance(jscpd_report_sha,str) and HEX_RE.fullmatch(jscpd_report_sha),
          'J04C_JSCPD_PROVENANCE_MISSING '+subject)
    for number,pair in enumerate(pairs,1):
        a_rel,b_rel=pair['file_a'],pair['file_b']
        a_begin,a_end=int(pair['start_a']),int(pair['end_a'])
        b_begin,b_end=int(pair['start_b']),int(pair['end_b'])
        fa=fragment_for(a,subject,stage,commit,a_rel,a_begin,a_end,pair['function_a'],cache)
        fb=fragment_for(a,subject,stage,commit,b_rel,b_begin,b_end,pair['function_b'],cache)
        check(fa['fragment_id']!=fb['fragment_id'],'J04D_SELF_CLONE '+subject)
        ident='J04D-'+digest_bytes((subject+'\0'+pair['pair_id']).encode())[:24]
        check(ident not in seen_pairs,'J04D_PAIR_ID_COLLISION '+subject)
        seen_pairs.add(ident)
        rows.append({
            'pair_id':ident,'fragment_a_id':fa['fragment_id'],'repository_a':subject,
            'file_a':a_rel,'start_line_a':a_begin,'end_line_a':a_end,
            'commit_a':commit,'source_sha256_a':fa['source_sha256'],
            'fragment_b_id':fb['fragment_id'],'repository_b':subject,
            'file_b':b_rel,'start_line_b':b_begin,'end_line_b':b_end,
            'commit_b':commit,'source_sha256_b':fb['source_sha256'],
            'same_repository':1,'nicad_similarity_raw':'','nicad_nlines':'',
            'detector':pair['source'],'config':run['settings']['nicad']['config'],
            'xml_sha256':xml_hashes[0], 'upstream_pair_id':pair['pair_id'],
            'function_a':pair['function_a'],'function_b':pair['function_b'],
            'ni_parse_failed_a':pair['ni_parse_failed_a'],
            'ni_parse_failed_b':pair['ni_parse_failed_b'],'label':a.positive_label,
        })
        if number%a.pair_progress_every==0:
            print(f'[{timestamp(a)}] PAIR_PROGRESS subject={subject} {number}/{len(pairs)}',flush=True)
    check(len(rows)==entry['clone_pairs'],'J04D_PAIR_CONSERVATION_FAILED '+subject)
    fragment_list=sorted(cache['fragments'].values(),key=lambda r:r['fragment_id'])
    file_csv=out/'positive_pairs.csv'
    with file_csv.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=PAIR_COLUMNS,lineterminator='\n')
        w.writeheader();w.writerows(rows)
    fragment_jsonl=out/'fragments.jsonl'
    with fragment_jsonl.open('x',encoding='utf-8') as f:
        for row in fragment_list:
            f.write(json.dumps(row,ensure_ascii=True,sort_keys=True,separators=(',',':'))+'\n')
    count=Counter(p['source'] for p in pairs)
    manifest={
        'schema':'deepknowcode-j04d-'+a.version,'experiment':a.experiment,'version':a.version,
        'status':'COMPLETED','timestamp_chicago':timestamp(a),'mode':a.mode,
        'subject':subject,'run_id':a.run_id,'input_experiment':a.input_experiment,
        'input_version':a.input_version,'input_data_version':a.source_data_version,
        'source_published_root':str(a.input_root),'source_subject':str(a.input_root/subject),
        'source_j04c_run_sha256':digest_file(a.input_root/subject/a.source_run_manifest_name),
        'source_pairs_sha256':digest_file(a.input_root/subject/a.source_pair_csv_name),
        'source_inventory_sha256':digest_file(a.input_root/subject/a.input_inventory_name),
        'source_stage_name':stage.name,'source_git_head':commit,
        'implementation_sha256':digest_file(Path(__file__)),
        'source_nicad_xml_sha256':xml_hashes[0],
        'source_jscpd_report_sha256':jscpd_report_sha,
        'source_files':entry['source_files'],'input_hybrid_pairs':len(pairs),
        'positive_pairs':len(rows),'unique_fragments':len(fragment_list),
        'source_detector_counts':dict(count),
        'files_sha256':{'positive_pairs.csv':digest_file(file_csv),
                        'fragments.jsonl':digest_file(fragment_jsonl)},
        'effective_configuration':{
            'experiment':a.experiment,'version':a.version,'mode':a.mode,
            'input_root':str(a.input_root),'result_root':str(a.result_root),
            'expected_subjects':a.expected_subjects,'expected_files':a.expected_files,
            'expected_pairs':a.expected_pairs,'input_experiment':a.input_experiment,
            'input_version':a.input_version,'source_data_version':a.source_data_version,
            'positive_label':a.positive_label,'language':a.language,
            'code_encoding':a.code_encoding,'code_errors':a.code_errors,
            'allowed_detector_sources':list(a.allowed_detector_sources),
            'timezone':a.timezone,'source_pair_csv_name':a.source_pair_csv_name,
            'source_run_manifest_name':a.source_run_manifest_name,
            'source_hybrid_summary_name':a.source_hybrid_summary_name,
            'input_inventory_name':a.input_inventory_name,
            'stage_dir_name':a.stage_dir_name,
            'pair_progress_every':a.pair_progress_every,
            'hash_progress_every':a.hash_progress_every,
            'expected_python_version':a.expected_python_version,
            'source_cache_files':a.source_cache_files,
        },
        'limitations':[
            'J-04d exports all J-04c hybrid candidates without deduplication.',
            'No clone-type classification or structural-positive selection (reserved for J-04e).',
            'No publishing to 3pos-clones (reserved for J-04f).',
            'The positive label is a detector candidate label, not independently certified gold.',
        ],
    }
    write_json(out/'manifest.json',manifest)
    print(f'PASS J04D_SUBJECT subject={subject} input_files={entry["source_files"]} '
          f'pairs={len(rows)} fragments={len(fragment_list)} output={out}',flush=True)
    return {'subject':subject,'status':'COMPLETED','input_files':entry['source_files'],
            'positive_pairs':len(rows),'unique_fragments':len(fragment_list),
            'nicad_primary_pairs':count['nicad_primary'],
            'jscpd_fallback_pairs':count['jscpd_ast_fallback'],
            'manifest_sha256':digest_file(out/'manifest.json')}


def verify_output(a, subject, entry, location):
    check(location.is_dir() and not location.is_symlink(),'J04D_OUTPUT_MISSING '+subject)
    m=read_json(location/'manifest.json')
    check(m.get('schema')=='deepknowcode-j04d-'+a.version and
          m.get('version')==a.version and m.get('subject')==subject and
          m.get('status')=='COMPLETED' and
          m.get('input_hybrid_pairs')==m.get('positive_pairs')==entry['clone_pairs'] and
          m.get('source_files')==entry['source_files'] and
          m.get('source_j04c_run_sha256')==digest_file(a.input_root/subject/a.source_run_manifest_name) and
          m.get('source_pairs_sha256')==digest_file(a.input_root/subject/a.source_pair_csv_name) and
          m.get('source_inventory_sha256')==digest_file(a.input_root/subject/a.input_inventory_name),
          'J04D_MANIFEST_MISMATCH '+subject)
    for name in ('positive_pairs.csv','fragments.jsonl'):
        check(digest_file(regular(location/name,location))==m['files_sha256'][name],
              'J04D_OUTPUT_SHA256_MISMATCH '+subject+':'+name)
    check(set(files_under(location))=={'manifest.json','positive_pairs.csv','fragments.jsonl'},
          'J04D_UNEXPECTED_OUTPUT_FILES '+subject)
    source=read_csv(a.input_root/subject/a.source_pair_csv_name,UPSTREAM_COLUMNS)
    check(len(source)==entry['clone_pairs'],'J04D_SOURCE_CHANGED '+subject)
    src_byid={r['pair_id']:r for r in source}
    check(len(src_byid)==len(source),'J04D_UPSTREAM_DUPLICATE_IDS '+subject)
    f_byid={}
    with (location/'fragments.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            fid=row['fragment_id']
            check(fid not in f_byid,'J04D_FRAGMENT_DUPLICATE '+subject)
            check(row['repository']==subject and row['language']==a.language and
                  digest_bytes(row['code'].encode(a.code_encoding,errors=a.code_errors))==row['source_sha256'],
                  'J04D_FRAGMENT_INTEGRITY '+subject)
            f_byid[fid]=row
    check(len(f_byid)==m['unique_fragments'],'J04D_FRAGMENT_COUNT_MISMATCH '+subject)
    rows=read_csv(location/'positive_pairs.csv',PAIR_COLUMNS)
    check(len(rows)==entry['clone_pairs'],'J04D_PAIR_COUNT_MISMATCH '+subject)
    seen_ids=set();used=set();referenced_fragments=set();detector_counts=Counter()
    for row in rows:
        key=row['upstream_pair_id']
        check(key in src_byid and key not in used,'J04D_PAIR_REFERENTIAL_ERROR '+subject)
        used.add(key)
        orig=src_byid[key]
        detector_counts[row['detector']]+=1
        check(row['pair_id']=='J04D-'+digest_bytes((subject+'\0'+key).encode())[:24],
              'J04D_PAIR_ID_INVALID '+subject)
        check(row['pair_id'] not in seen_ids,'J04D_PAIR_ID_DUPLICATE '+subject)
        seen_ids.add(row['pair_id'])
        for suffix,short in (('a','a'),('b','b')):
            frag=f_byid.get(row['fragment_'+suffix+'_id'])
            check(frag is not None,'J04D_FRAGMENT_REFERENCE_MISSING '+subject)
            referenced_fragments.add(frag['fragment_id'])
            check(frag['fragment_id']=='fn-'+digest_bytes((subject+'\0'+frag['file_path']+
                  '\0'+str(frag['start_line'])+'\0'+str(frag['end_line'])+'\0'+
                  frag['function_name']).encode())[:24],
                  'J04D_FRAGMENT_IDENTITY_INVALID '+subject)
            check(row['file_'+suffix]==orig['file_'+short] and
                  row['file_'+suffix]==frag['file_path'] and
                  row['start_line_'+suffix]==orig['start_'+short]==str(frag['start_line']) and
                  row['end_line_'+suffix]==orig['end_'+short]==str(frag['end_line']) and
                  row['source_sha256_'+suffix]==frag['source_sha256'] and
                  row['function_'+suffix]==orig['function_'+short]==frag['function_name'] and
                  row['commit_'+suffix]==frag['commit'],
                  'J04D_PAIR_SOURCE_JOIN_MISMATCH '+subject)
        check(row['ni_parse_failed_a']==orig['ni_parse_failed_a'] and
              row['ni_parse_failed_b']==orig['ni_parse_failed_b'] and
              row['xml_sha256']==m['source_nicad_xml_sha256'] and
              row['detector']==orig['source'] and row['label']==str(a.positive_label) and
              row['repository_a']==row['repository_b']==subject and
              row['same_repository']=='1','J04D_PAIR_METADATA_MISMATCH '+subject)
    check(len(used)==len(src_byid),'J04D_PAIR_COVERAGE_INCOMPLETE '+subject)
    check(referenced_fragments==set(f_byid),'J04D_UNUSED_OR_MISSING_FRAGMENT '+subject)
    check(dict(detector_counts)==m['source_detector_counts'],
          'J04D_DETECTOR_COUNT_MISMATCH '+subject)
    # Verify extracted source bytes against the immutable published J-04c staging.
    stage_dir=a.input_root/subject/a.stage_dir_name/m['source_stage_name']
    verify_cache=OrderedDict()
    for frag in f_byid.values():
        relative=frag['file_path']
        if relative not in verify_cache:
            path=regular(stage_dir/safe_rel(relative,'.py'),stage_dir)
            raw=path.read_bytes()
            verify_cache[relative]=(raw.splitlines(keepends=True),digest_bytes(raw))
            if len(verify_cache)>a.source_cache_files:
                verify_cache.popitem(last=False)
        else:
            verify_cache.move_to_end(relative)
        lines,source_file_sha=verify_cache[relative]
        start,end=frag['start_line'],frag['end_line']
        check(1<=start<=end<=len(lines) and
              frag['file_sha256']==source_file_sha and
              digest_bytes(b''.join(lines[start-1:end]))==frag['source_sha256'],
              'J04D_FRAGMENT_STAGE_MISMATCH '+subject+':'+frag['fragment_id'])
    print(f'PASS J04D_VERIFY subject={subject} pairs={len(rows)} fragments={len(f_byid)}',flush=True)
    return {'subject':subject,'pairs':len(rows),'fragments':len(f_byid)}


def get_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('experiment','version','input-experiment','input-version','source-data-version',
                'expected-python-version','run-id','mode','subject','timezone','language',
                'code-encoding','code-errors','copy-manifest-name','batch-summary-name',
                'source-run-manifest-name','source-pair-csv-name','source-hybrid-summary-name',
                'input-inventory-name','stage-dir-name','allowed-detector-sources'):
        p.add_argument('--'+key,required=True)
    for key in ('input-root','result-root','run-dir'):
        p.add_argument('--'+key,required=True,type=Path)
    for key in ('expected-subjects','expected-files','expected-pairs','positive-label',
                'pair-progress-every','hash-progress-every','source-cache-files'):
        p.add_argument('--'+key,required=True,type=int)
    a=p.parse_args(argv)
    check(a.mode in ('inspect','smoke','subject','all','verify'),'INVALID_MODE')
    check(a.experiment=='j-04d' and re.fullmatch(r'v[0-9]+',a.version) and
          a.timezone=='America/Chicago' and a.positive_label==1 and
          a.language=='python','INVALID_EXPERIMENT_IDENTITY')
    check(a.expected_subjects>0 and a.expected_files>0 and a.expected_pairs>0 and
          a.pair_progress_every>0 and a.hash_progress_every>0 and a.source_cache_files>0,
          'INVALID_EXPECTATIONS')
    check(re.fullmatch(r'[A-Za-z0-9_.-]{1,120}',a.run_id),'INVALID_RUN_ID')
    check(a.expected_python_version==f'{sys.version_info.major}.{sys.version_info.minor}',
          'PYTHON_VERSION_MISMATCH expected='+a.expected_python_version+
          ' actual='+f'{sys.version_info.major}.{sys.version_info.minor}')
    check(a.mode not in ('smoke','subject') or SUBJECT_RE.fullmatch(a.subject),
          'INVALID_SUBJECT')
    a.allowed_detector_sources=tuple(a.allowed_detector_sources.split(','))
    check(set(a.allowed_detector_sources)=={'nicad_primary','jscpd_ast_fallback'},
          'INVALID_DETECTOR_SOURCE_POLICY')
    a.input_root=a.input_root.absolute()
    a.result_root=a.result_root.absolute()
    a.run_dir=a.run_dir.absolute()
    check(a.run_dir.parent==a.result_root and
          a.input_root not in a.result_root.parents and
          a.result_root not in a.input_root.parents and
          a.input_root!=a.result_root,'INPUT_RESULT_OVERLAP')
    return a


def latest_all(a):
    check(a.result_root.is_dir() and not a.result_root.is_symlink(),'RESULT_ROOT_MISSING')
    folders=sorted((p for p in a.result_root.iterdir() if p.is_dir() and not p.is_symlink() and
                    '-all-' in p.name),key=lambda p:p.name,reverse=True)
    check(folders,'J04D_ALL_RUN_REQUIRED')
    folder=folders[0]
    summary=read_json(folder/'batch-summary.json')
    check(summary.get('status')=='COMPLETED' and summary.get('mode')=='all' and
          summary.get('experiment')==a.experiment and summary.get('version')==a.version and
          summary.get('subjects_completed')==a.expected_subjects and
          summary.get('input_files_total')==a.expected_files and
          summary.get('positive_pairs_total')==a.expected_pairs,
          'J04D_LATEST_ALL_NOT_COMPLETED')
    return folder,summary


def main(argv=None):
    a=get_args(argv)
    receipt,batch,subjects,receipt_hash=load_source(a)
    names=sorted(subjects)
    if a.mode=='inspect':
        for name in names:
            validate_subject(a,name,subjects[name],full_hash=True)
            print(f'PASS J04D_INPUT subject={name} files={subjects[name]["source_files"]} '
                  f'pairs={subjects[name]["clone_pairs"]}',flush=True)
        a.run_dir.mkdir(parents=True,exist_ok=False)
        write_json(a.run_dir/'inspection.json',{'status':'PASS','timestamp_chicago':timestamp(a),
                   'source_manifest_sha256':receipt_hash,
                   'subjects':len(names),'files':a.expected_files,'pairs':a.expected_pairs})
        print(f'PASS J04D_INSPECT subjects={len(names)} files={a.expected_files} '
              f'pairs={a.expected_pairs}',flush=True)
        return 0
    if a.mode=='verify':
        folder,prior=latest_all(a)
        check(prior.get('source_copy_manifest_sha256')==receipt_hash,
              'J04D_ORIGINAL_SOURCE_RECEIPT_CHANGED')
        check({r['subject'] for r in prior.get('subjects',[])}==set(names),
              'J04D_VERIFY_SUBJECT_SET_MISMATCH')
        checks=[]
        for name in names:
            validate_subject(a,name,subjects[name],full_hash=True)
            checks.append(verify_output(a,name,subjects[name],folder/'subjects'/name))
        check(sum(c['pairs'] for c in checks)==a.expected_pairs,'J04D_VERIFY_TOTAL_MISMATCH')
        a.run_dir.mkdir(parents=True,exist_ok=False)
        write_json(a.run_dir/'verification-all.json',{'status':'PASS','version':a.version,
                   'timestamp_chicago':timestamp(a),'source_all':str(folder),
                   'source_copy_manifest_sha256':receipt_hash,
                   'verified_subjects':len(checks),'positive_pairs':sum(c['pairs'] for c in checks),
                   'subjects':checks})
        print(f'PASS J04D_VERIFY_ALL subjects={len(checks)} pairs={a.expected_pairs}',flush=True)
        return 0
    if a.mode=='all': selected=names
    elif a.mode in ('smoke','subject'):
        check(a.subject in subjects,'SUBJECT_UNKNOWN '+a.subject)
        selected=[a.subject]
    else:raise RuntimeError('UNKNOWN_MODE')
    a.run_dir.mkdir(parents=True,exist_ok=False)
    (a.run_dir/'subjects').mkdir(exist_ok=False)
    records=[]
    for index,name in enumerate(selected,1):
        print(f'[{timestamp(a)}] START J04D_SUBJECT {index}/{len(selected)} subject={name}',flush=True)
        records.append(produce_subject(a,name,subjects[name],a.run_dir/'subjects'/name))
        verify_output(a,name,subjects[name],a.run_dir/'subjects'/name)
    check(a.mode!='all' or (len(records)==a.expected_subjects and
          sum(r['input_files'] for r in records)==a.expected_files and
          sum(r['positive_pairs'] for r in records)==a.expected_pairs),
          'J04D_ALL_AGGREGATE_MISMATCH')
    summary={'schema':'deepknowcode-j04d-batch-'+a.version,'experiment':a.experiment,
             'version':a.version,'status':'COMPLETED','mode':a.mode,
             'run_id':a.run_id,'timestamp_chicago':timestamp(a),
             'source_copy_manifest_sha256':receipt_hash,
             'input_root':str(a.input_root),'subjects_completed':len(records),
             'input_files_total':sum(r['input_files'] for r in records),
             'positive_pairs_total':sum(r['positive_pairs'] for r in records),
             'unique_fragments_total':sum(r['unique_fragments'] for r in records),
             'effective_configuration':{
               'expected_subjects':a.expected_subjects,'expected_files':a.expected_files,
               'expected_pairs':a.expected_pairs,'input_experiment':a.input_experiment,
               'input_version':a.input_version,'source_data_version':a.source_data_version,
               'python_version':a.expected_python_version,'source_cache_files':a.source_cache_files,
               'label':a.positive_label,'code_encoding':a.code_encoding,
               'code_errors':a.code_errors,'timezone':a.timezone,
               'detector_sources':list(a.allowed_detector_sources)},
             'subjects':records}
    write_json(a.run_dir/'batch-summary.json',summary)
    print(f'PASS J04D_SUMMARY mode={a.mode} subjects={len(records)} '
          f'pairs={summary["positive_pairs_total"]} fragments={summary["unique_fragments_total"]}',flush=True)
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except (RuntimeError,ValueError,TypeError,KeyError,UnicodeError,SyntaxError,
            OSError,OverflowError,RecursionError) as exc:
        print(f'FAIL J04D {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        sys.exit(2)
