#!/usr/bin/env python3
"""F-01c v6 read-only discovery of external BigCloneBench/CodeXGLUE train pairs.

Only explicit, local shell-configured roots are accessed. External labels remain
UNVERIFIED; this audit neither updates the corpus nor authorizes training.
"""
import hashlib
import heapq
import json
from pathlib import Path

import f01c_shared as s
from f01c_variants import contextual_continuation


def pair_fingerprint(a, b):
    return tuple(sorted((s.code_hash(a), s.code_hash(b))))


def locate_sources(roots, patterns, max_files, approved_java_dir):
    found = set()
    missing = []
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            missing.append(str(root)); continue
        for pattern in patterns:
            if not pattern.startswith('**/') or '..' in pattern or pattern.count('*') > 4:
                raise ValueError('Unsafe BCB discovery pattern: ' + pattern)
            for p in root.glob(pattern):
                if not p.is_file() or p.is_symlink():
                    continue
                p = p.resolve()
                if not p.is_relative_to(root):
                    continue
                lower = [x.lower() for x in p.parts]
                if any(x in ('test','valid','validation','heldout','evaluation') for x in lower):
                    continue
                name = p.name.lower()
                if name not in ('train.txt', 'train_pairs.txt'):
                    continue
                if p.parent == Path(approved_java_dir).resolve():
                    continue
                found.add(p)
                if len(found) > max_files:
                    raise ValueError('BCB discovery max-files exceeded; narrow shell roots')
    return sorted(found), missing


def local_codes_for(train_file):
    # A BCB/CodeXGLUE pair file must have an associated local Java code map.
    # Never reinterpret foreign IDs with the approved Java Target map.
    for code in (train_file.parent / 'data.jsonl', train_file.parent / 'code.jsonl',
                 train_file.parent.parent / 'data.jsonl'):
        if code.is_file() and not code.is_symlink():
            return code.resolve()
    return None


def split_fingerprints(folder, codes, max_lines):
    forbidden=set(); forbidden_endpoints=set(); summary={}; complete=True
    for name in ('valid.txt','validation.txt','test.txt'):
        path=folder/name
        if not path.is_file():
            continue
        count=0
        with path.open(encoding='utf-8-sig') as stream:
            for line in stream:
                if not line.strip():continue
                count += 1
                if count>max_lines:
                    complete=False; break
                row=line.split()
                if len(row)!=3 or row[2] not in ('0','1') or row[0] not in codes or row[1] not in codes:
                    continue
                forbidden.add(pair_fingerprint(codes[row[0]],codes[row[1]]))
                forbidden_endpoints.update((s.code_hash(codes[row[0]]),s.code_hash(codes[row[1]])))
        summary[name]={'rows_examined':min(count,max_lines),'sha256':s.sha256(path),'scan_complete':count<=max_lines}
    return forbidden,forbidden_endpoints,summary,complete


def audit_file(path, a, current_fingerprints, forbidden_fingerprints, excluded_functions,
               tok, answer_ids):
    code_path=local_codes_for(path)
    rec={'train_file':str(path),'train_sha256':s.sha256(path),'code_map':str(code_path) if code_path else None,
         'label_provenance':'EXTERNAL_TRAIN_SPLIT_UNVERIFIED','approved_for_training':False}
    if code_path is None:
        rec.update({'status':'NO_ASSOCIATED_CODE_MAP','eligible_shortlisted':0});return rec,[]
    if code_path.stat().st_size > a.bcb_max_code_map_bytes:
        rec.update({'status':'CODE_MAP_SIZE_LIMIT','eligible_shortlisted':0});return rec,[]
    try:
        codes=s.load_codes(code_path)
    except (ValueError,KeyError,TypeError,UnicodeError,json.JSONDecodeError) as exc:
        rec.update({'status':'UNSUPPORTED_CODE_MAP','error':str(exc)[:200],'eligible_shortlisted':0});return rec,[]
    local_excluded,split_endpoints,split_meta,split_complete=split_fingerprints(path.parent,codes,a.bcb_max_rows_per_file)
    # If any disallowed split was not fully scanned, no candidates are considered safe.
    if not split_complete:
        rec.update({'status':'SPLIT_SCAN_INCOMPLETE','split_sources':split_meta,'eligible_shortlisted':0});return rec,[]
    excluded=forbidden_fingerprints | local_excluded
    counts={'rows_read':0,'negative_rows':0,'unresolved_ids':0,'malformed_rows':0,
            'source_or_heldout_function':0,'baseline_or_forbidden_pair':0,'same_code':0}
    # Deterministic min-hash shortlist; no random state and no silently incomplete scan.
    heap=[];seen=set();complete=True
    with path.open(encoding='utf-8-sig') as src:
        for raw in src:
            if not raw.strip():continue
            counts['rows_read']+=1
            if counts['rows_read']>a.bcb_max_rows_per_file:
                complete=False;break
            fields=raw.split()
            if len(fields)!=3 or fields[2] not in ('0','1'):
                counts['malformed_rows']+=1;continue
            x,y,label=fields
            if label!='0':continue
            counts['negative_rows']+=1
            if x==y or x not in codes or y not in codes:
                counts['unresolved_ids']+=1;continue
            fa,fb=s.code_hash(codes[x]),s.code_hash(codes[y])
            if fa==fb:
                counts['same_code']+=1;continue
            if fa in excluded_functions or fb in excluded_functions or fa in split_endpoints or fb in split_endpoints:
                counts['source_or_heldout_function']+=1;continue
            fingerprint=tuple(sorted((fa,fb)))
            if fingerprint in current_fingerprints or fingerprint in excluded:
                counts['baseline_or_forbidden_pair']+=1;continue
            if fingerprint in seen:continue
            seen.add(fingerprint)
            rank=int.from_bytes(hashlib.sha256((str(a.seed)+'|'+fingerprint[0]+'|'+fingerprint[1]).encode()).digest()[:16],'big')
            entry=(-rank,x,y,fingerprint)
            if len(heap)<a.bcb_shortlist_size:
                heapq.heappush(heap,entry)
            elif entry>heap[0]:
                heapq.heapreplace(heap,entry)
    previews=[]
    for neg_rank,x,y,fingerprint in sorted(heap,reverse=True):
        ca,cb=codes[x],codes[y]
        prompt=a.prompt_prefix+ca+a.prompt_b_separator+cb+a.prompt_suffix
        try:
            prefix,token=contextual_continuation(tok,prompt,a.answer_no)
        except ValueError:
            continue
        if token!=answer_ids[0] or len(prefix)+1>a.max_seq_length:
            continue
        previews.append({'id1':x,'id2':y,'label':0,'tokens':len(prefix)+1,
            'code_hashes':list(fingerprint),'train_file':str(path),'code_map':str(code_path),
            'code_map_sha256':s.sha256(code_path),'train_sha256':rec['train_sha256'],
            'provenance_approved':False,'train_split_validated':False,
            'integrity_status':'PRELIMINARY_REVIEW_ONLY', 'approved_for_training':False})
    rec.update(counts)
    rec.update({'status':'PRELIMINARY_CANDIDATES_FOUND' if previews else 'NO_PRELIMINARY_CANDIDATES',
        'complete_train_scan':complete,'split_sources':split_meta,
        'code_map_sha256':s.sha256(code_path),'code_map_entries':len(codes),
        'unique_pre_tokenization_candidates':len(seen),
        'shortlist_examined':len(heap),'eligible_shortlisted':len(previews),
        'shortlist_is_exhaustive':len(seen)<=a.bcb_shortlist_size,
        'limitations':['External BCB labels and split provenance are not certified by this audit.',
                       'Missing external valid/test split files do not prove absence of overlap.',
                       'Selection uses a deterministic bounded shortlist, not an exhaustive token audit.']})
    return rec, previews


def inventory_archives(roots, max_files):
    records=[]
    for root in roots:
        root=Path(root).expanduser().resolve()
        if not root.is_dir():continue
        for pattern in ('**/*bigclone*.zip','**/*bcb*.zip','**/*clone*bench*.zip'):
            for item in root.glob(pattern):
                if not item.is_file() or item.is_symlink(): continue
                if item.resolve().is_relative_to(root) and str(item) not in records:
                    records.append(str(item))
                if len(records)>=max_files:return sorted(records)
    return sorted(records)


def run(a,codes,source_hash,heldout_hashes,tok,answer_ids):
    files,missing=locate_sources(a.bcb_discovery_roots,a.bcb_discovery_patterns,
                                  a.bcb_max_files,a.java_target_dir)
    java=codes['java_mixed']
    existing=set();validation=set();validation_functions=set()
    for source, target in ((a.java_target_dir/'train.txt',existing),(a.java_target_dir/'valid.txt',validation)):
        for x,y,label in s.pair_rows(source,java):
            target.add(pair_fingerprint(java[x],java[y]))
            if target is validation:
                validation_functions.update((s.code_hash(java[x]),s.code_hash(java[y])))
    all_seen=set();file_reports=[];previews=[]
    for path in files:
        rec,rows=audit_file(path,a,existing,validation,source_hash|heldout_hashes|validation_functions,tok,answer_ids)
        file_reports.append(rec)
        for row in rows:
            key=tuple(row['code_hashes'])
            if key in all_seen:continue
            all_seen.add(key)
            if len(previews)<a.bcb_max_report_candidates:
                previews.append(row)
    return {'files_discovered':len(files),'missing_roots':missing,
       'archives_not_extracted':inventory_archives(a.bcb_discovery_roots,a.bcb_max_files),
       'files':file_reports,
       'preliminary_eligible_count':len(all_seen), 'report_candidates':previews,
       'candidate_report_capped':len(all_seen)>a.bcb_max_report_candidates,
       'any_incomplete_scan':any(x.get('complete_train_scan') is False for x in file_reports),
       'limitations':['No files are downloaded or extracted automatically.',
        'Only supported local train.txt + associated data.jsonl/code.jsonl pairs are evaluated.',
        'ZIP archives and differently structured BCB data are inventoried only by external tools.',
        'No external pair is approved without trusted provenance, full split/endpoint isolation, and leakage/label review.',
        'No training data, labels, tokenizer configuration or artifacts are modified.']}
