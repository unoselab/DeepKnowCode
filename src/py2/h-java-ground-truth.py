#!/usr/bin/env python3
"""H-Series v2: evidence-driven Java held-out GT conflict exclusion; never train or rewrite input.

H-01 inventory; H-02 code-pair identity; H-03 human review packets + validation;
H-04 G-02/G-03 frozen data isolation; H-05 conditionally freeze; H-06 verify.
"""
import argparse
import collections
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

VERSION = 'v2'


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def chicago():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def sha_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def sha_string(s):
    return sha_bytes(s.encode('utf-8'))


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n'


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(dump(value))


def records(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        for i, line in enumerate(f, 1):
            need(bool(line.strip()), f'Blank JSONL line: {path}:{i}')
            row = json.loads(line)
            need(isinstance(row, dict), f'Nonobject JSONL: {path}:{i}')
            yield row


def jsonl(path, entries):
    with Path(path).open('x', encoding='utf-8') as f:
        for item in entries:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n')


def read_json(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def marker(stage, out, files, summary, sources=None, a=None):
    mf = dict(version=VERSION, stage=stage, at_chicago=chicago(), summary=summary,
              files={name: dict(sha256=sha_file(out/name), bytes=(out/name).stat().st_size)
                     for name in files}, sources=sources or {},
              effective_parameters=({k: str(v) if isinstance(v,Path) else v
                                     for k,v in vars(a).items()} if a is not None else {}))
    write_json(out/'manifest.json', mf)
    return mf


def verified_run(path, stage, expected_files):
    path = Path(path).resolve()
    need(path.is_dir(), f'Missing prior run: {path}')
    manifest = read_json(path/'manifest.json')
    need(manifest.get('stage') == stage and manifest.get('version') == VERSION,
         f'Wrong version/stage: {path}')
    files = manifest.get('files', {})
    for name in expected_files:
        need(name in files and Path(name).name == name and (path/name).is_file(),
             f'Prior run artifact missing: {path/name}')
        meta = files[name]
        need(sha_file(path/name) == meta['sha256'] and
             (path/name).stat().st_size == meta['bytes'],
             f'Prior run SHA/size drift: {path/name}')
    return path, manifest


def all_csv_files(a):
    root = Path(a.source_dir).resolve()
    need(root.is_dir(), f'Missing Java RQ4 reference CSV directory: {root}')
    files = sorted(root.glob('pred_*_with_refactorability.csv'))
    need(len(files) == a.expected_projects,
         f'Project file count mismatch: {len(files)} != {a.expected_projects}')
    need(len(files) == len({p.name for p in files}), 'Duplicate reference filenames')
    return files


def load_codes(a):
    path = Path(a.code_map).resolve()
    need(path.is_file(), f'Missing Evaluation Java code map: {path}')
    digest = sha_file(path)
    need(digest == a.expected_code_sha256, f'Evaluation code map SHA mismatch: {digest}')
    ids, exact, norm = {}, collections.defaultdict(list), collections.defaultdict(list)
    count = 0
    for obj in records(path):
        count += 1
        need(set(obj) == {'id', 'code'}, 'Unexpected Java code map schema')
        idx = str(obj['id']); src = obj['code']
        need(idx and idx not in ids and isinstance(src, str) and src.strip(),
             f'Invalid/duplicate/empty Java function: {idx}')
        ids[idx] = src
        exact[sha_string(src)].append(idx)
        norm[sha_string(''.join(src.split()))].append(idx)
    need(count == a.expected_code_rows and len(ids) == count,
         f'Java code map row count mismatch {count}')
    return ids, exact, norm, digest


def src_label(a, b):
    aa = a.split('_'); bb = b.split('_')
    need(len(aa) >= 2 and len(bb) >= 2 and bool(aa[1]) and bool(bb[1]),
         'Unparseable legacy ID for Label: '+a+' / '+b)
    return int(aa[1] == bb[1])


def pair_key(h1, h2):
    return sha_string('\0'.join(sorted((h1, h2))))


def source_records(a, codes):
    stats = collections.Counter()
    names = {}; rows = []; ids = set(); cross_id=set(); row_keys=set()
    for p in all_csv_files(a):
        project = p.name[len('pred_'):-len('_with_refactorability.csv')]
        need(project not in names, f'Duplicate project filename {project}')
        names[project] = {'name':p.name,'sha256':sha_file(p), 'bytes':p.stat().st_size}
        within=set(); count=0
        with p.open(encoding='utf-8-sig', newline='') as f:
            rdr = csv.DictReader(f)
            need({'pair_left_func','pair_right_func','clone_predict'} <= set(rdr.fieldnames or []),
                 f'Invalid reference CSV columns {p}')
            for lineno,row in enumerate(rdr, 2):
                la=(row.get('pair_left_func') or '').strip()
                rb=(row.get('pair_right_func') or '').strip()
                need(la in codes and rb in codes,
                     f'Missing Code ID at {p.name}:{lineno}: {la}/{rb}')
                label=src_label(la,rb)
                v=(row.get('clone_predict') or '').strip()
                need(v in ('0','1'), f'Invalid clone_predict (prediction) {p.name}:{lineno}')
                ikey=tuple(sorted((la,rb)))
                need(ikey not in within, f'Duplicate pair IDs within project {p.name}:{lineno}')
                within.add(ikey)
                eh1=sha_string(codes[la]);eh2=sha_string(codes[rb])
                key=pair_key(eh1,eh2)
                r={'pair_key':key,'project':project,'source_csv':p.name,'source_row':lineno,
                   'left_id':la,'right_id':rb,'left_code_sha256':eh1,'right_code_sha256':eh2,
                   'id_derived_label':label,'source_prediction':int(v)}
                rows.append(r); stats[label] += 1; count+=1
                ids.update((la,rb));cross_id.add((project,ikey))
                row_keys.add((project,ikey))
        names[project]['pairs']=count
    need(stats[0] == a.expected_negative and stats[1] == a.expected_positive and
         len(rows) == a.expected_pairs and len(ids) == a.expected_code_rows,
         f'Pair census mismatch: n={len(rows)} labels={dict(stats)} IDs={len(ids)}')
    return rows,names,stats


def curated_records(a,codes):
    """Exclude *all* rows from the single explicitly pinned contradictory code-pair group.

    Never mutate the original 48 CSVs. Exclusion is a versioned, fully audited
    evaluation candidate policy, not a human verification or label correction.
    """
    original,names,original_stats=source_records(a,codes)
    by_key=collections.defaultdict(list)
    for record in original:
        by_key[record['pair_key']].append(record)
    conflicts={key:group for key,group in by_key.items()
               if len({r['id_derived_label'] for r in group})>1}
    need(len(conflicts)==a.expected_conflict_groups_before_exclusion,
         f'Original conflicting-group count drift: {len(conflicts)}')
    need(set(conflicts)=={a.exclude_conflict_pair_key},
         f'Unexpected conflict key(s): {sorted(conflicts)}')
    excluded=conflicts[a.exclude_conflict_pair_key]
    refs=sorted(f"{r['source_csv']}:{r['source_row']}" for r in excluded)
    want=sorted(a.exclude_conflict_source_rows.split(','))
    need(refs==want and len(excluded)==a.expected_excluded_rows,
         f'Conflicting rows differ from approved exclusion: {refs}')
    need({r['id_derived_label'] for r in excluded}=={0,1},
         'Excluded pair group no longer contains both labels')
    kept=[r for r in original if r['pair_key'] != a.exclude_conflict_pair_key]
    labels=collections.Counter(r['id_derived_label'] for r in kept)
    keys=collections.defaultdict(set)
    for r in kept:
        keys[r['pair_key']].add(r['id_derived_label'])
    need(not any(len(v)>1 for v in keys.values()),
         'Other conflicting pair group remains after exclusion')
    need(len(kept)==a.expected_curated_pairs and
         labels[1]==a.expected_curated_positive and
         labels[0]==a.expected_curated_negative and
         len(keys)==a.expected_curated_unique_code_pairs,
         f'Curated pair census differs from explicit Shell policy: rows={len(kept)} labels={dict(labels)} unique={len(keys)}')
    return kept,names,labels,excluded,original,original_stats


def exclusion_summary(a,excluded,original):
    return dict(policy='EXCLUDE_ALL_OCCURRENCES_OF_EXPLICIT_CONFLICTING_EXACT_CODE_PAIR',
                reason='CONTRADICTORY_ID_DERIVED_LABELS_SAME_UNORDERED_CODE_PAIR',
                pair_key=a.exclude_conflict_pair_key,
                source_rows=sorted(f"{r['source_csv']}:{r['source_row']}" for r in excluded),
                excluded_rows=len(excluded),raw_input_pairs=len(original),
                raw_csv_files_preserved=True,label_reassignment_performed=False,
                human_gold_label_established=False)


def inventory(a):
    codes,exact,norm,digest=load_codes(a)
    kept,names,stats,excluded,original,original_stats=curated_records(a,codes)
    report=dict(status='H01_V2_SOURCE_INVENTORY_AND_CONFLICT_EXCLUSION_POLICY_PASS_NOT_GOLD',
                at_chicago=chicago(),projects=len(names),
                pairs=len(original),positive=original_stats[1],negative=original_stats[0],
                curated_pairs=len(kept),curated_positive=stats[1],curated_negative=stats[0],
                excluded_conflict_rows=len(excluded),resolved_ids=len(codes),missing_ids=0,
                code_map_sha256=digest,exclusion=exclusion_summary(a,excluded,original),
                legacy_paper_reference={'projects':a.reference_projects,
                'positive':a.reference_positive,'negative':a.reference_negative,
                'same_as_staged':len(original)==a.reference_positive+a.reference_negative},
                project_files=names,
                cautions=['ID-derived labels are not independently verified gold.',
                          'clone_predict is prediction, never gold.',
                          'Conflicting rows excluded from candidate evaluation; original source preserved.'])
    write_json(a.out_dir/'inventory.json',report)
    marker('h-01',a.out_dir,['inventory.json'],report,
           sources={'code_map_sha256':digest},a=a)
    print(dump({k:report[k] for k in ('status','projects','pairs','positive','negative',
            'curated_pairs','curated_positive','curated_negative','excluded_conflict_rows','resolved_ids')}))


def write_curated_csv_snapshot(a, names, excluded):
    """Write 48 new CSVs, excluding exactly two approved source CSV records.

    The source CSVs are only read. The derived CSVs retain the original columns
    and all non-excluded row values. Each CSV hash is saved for readback.
    """
    directory=a.out_dir/'curated_csv'
    directory.mkdir(exist_ok=False)
    refs={(r['source_csv'],r['source_row']) for r in excluded}
    need(len(refs)==a.expected_excluded_rows,'Duplicate excluded row identities')
    meta={};total=0
    for project, entry in sorted(names.items()):
        filename=entry['name']
        source=Path(a.source_dir)/filename
        dest=directory/filename
        kept=0;removed=0
        with source.open(encoding='utf-8-sig',newline='') as inp, dest.open('x',encoding='utf-8',newline='') as out:
            reader=csv.DictReader(inp)
            need(bool(reader.fieldnames),'CSV header missing: '+filename)
            writer=csv.DictWriter(out,fieldnames=reader.fieldnames)
            writer.writeheader()
            for row_no,row in enumerate(reader,2):
                if (filename,row_no) in refs:
                    removed+=1
                    continue
                writer.writerow(row)
                kept+=1
        meta[filename]=dict(rows=kept,excluded_rows=removed,sha256=sha_file(dest),
                            bytes=dest.stat().st_size,source_sha256=entry['sha256'])
        need(kept+removed==entry['pairs'],'Curated CSV source count drift: '+filename)
        total+=kept
    need(total==a.expected_curated_pairs and
         sum(x['excluded_rows'] for x in meta.values())==a.expected_excluded_rows and
         len(meta)==a.expected_projects, 'Curated CSV census drift')
    write_json(a.out_dir/'curated_csv_manifest.json',dict(
        status='H02_V2_CURATED_SOURCE_CSV_COPY_COMPLETE_NOT_GOLD',
        original_source_preserved=True,curated_pair_rows=total,projects=len(meta),
        excluded_rows=a.expected_excluded_rows,files=meta))


def identity(a):
    codes,exact,norm, code_hash = load_codes(a)
    items,names,stats,excluded,original,original_stats=curated_records(a,codes)
    grouped=collections.defaultdict(list)
    for row in items:
        grouped[row['pair_key']].append(row)
    conflicts=[];duplicates=[]
    for k,part in sorted(grouped.items()):
        labels={r['id_derived_label'] for r in part}
        if len(labels)>1:
            conflicts.append(dict(pair_key=k, code_hash_pair=sorted((part[0]['left_code_sha256'],part[0]['right_code_sha256'])),
                                  conflicting_labels=sorted(labels), occurrences=len(part), rows=part))
        if len(part)>1:
            duplicates.append(dict(pair_key=k, occurrences=len(part), labels=sorted(labels),
                                   projects=sorted({r['project'] for r in part}),
                                   records=[{'project':r['project'],'source_csv':r['source_csv'],
                                             'source_row':r['source_row'],'left_id':r['left_id'],
                                             'right_id':r['right_id'],'id_derived_label':r['id_derived_label']}
                                            for r in part]))
    report=dict(status='H02_V2_UNRESOLVED_CONFLICT_REVIEW_REQUIRED' if conflicts else
                'H02_V2_CONFLICT_GROUP_EXCLUDED_LEGACY_LABELS_UNVERIFIED',
                at_chicago=chicago(),project_files=names,code_map_sha256=code_hash,
                code_rows=len(codes),unique_exact_code_hashes=len(exact),
                unique_normalized_code_hashes=len(norm),
                exact_hash_shared_by_multiple_ids=sum(len(x)>1 for x in exact.values()),
                normalized_hash_shared_by_multiple_ids=sum(len(x)>1 for x in norm.values()),
                original_input_pairs=len(original),input_pairs=len(items),
                eligible_positive=stats[1],eligible_negative=stats[0],
                excluded_conflict_rows=len(excluded),exclusion=exclusion_summary(a,excluded,original),
                unique_unordered_code_pairs=len(grouped),
                redundant_code_pair_occurrences=len(items)-len(grouped),
                repeated_unordered_code_pairs=len(duplicates),
                code_pair_label_conflicts=len(conflicts),
                gold_label_verified=False, freeze_allowed=False,
                limitations=['Repeated function body across IDs may be legitimate Type-1 clone.',
                             'Both contradictory source records are excluded, not relabeled; raw CSVs stay untouched.',
                             'Unordered code-hash collisions are NOT independent gold adjudication.',
                             'Legacy ID-derived labels not semantically validated.'])
    need(len(conflicts)==0 and len(grouped)==a.expected_curated_unique_code_pairs,
         'H02 selected dataset has an unexpected contradiction or unique-pair count')
    jsonl(a.out_dir/'pair_records.jsonl',items)
    jsonl(a.out_dir/'excluded_conflict_rows.jsonl',excluded)
    jsonl(a.out_dir/'conflicting_code_pairs.jsonl',conflicts)
    jsonl(a.out_dir/'duplicate_code_pairs.jsonl',duplicates)
    write_json(a.out_dir/'identity_audit.json',report)
    write_curated_csv_snapshot(a,names,excluded)
    marker('h-02',a.out_dir,['pair_records.jsonl','excluded_conflict_rows.jsonl',
                              'conflicting_code_pairs.jsonl',
                              'duplicate_code_pairs.jsonl','identity_audit.json','curated_csv_manifest.json'],report,
           sources={'code_map_sha256':code_hash, 'csv_sha256':{k:v['sha256'] for k,v in names.items()}},a=a)
    print(dump({k:report[k] for k in ('status','original_input_pairs','input_pairs','eligible_positive','eligible_negative',
            'excluded_conflict_rows','unique_unordered_code_pairs',
            'redundant_code_pair_occurrences','code_pair_label_conflicts',
            'exact_hash_shared_by_multiple_ids','normalized_hash_shared_by_multiple_ids')}))


def verify_h02(a):
    path,m = verified_run(a.h02_run,'h-02',['pair_records.jsonl',
            'excluded_conflict_rows.jsonl','conflicting_code_pairs.jsonl',
            'duplicate_code_pairs.jsonl','identity_audit.json','curated_csv_manifest.json'])
    code_sha=sha_file(a.code_map)
    need(code_sha==m['sources']['code_map_sha256']==a.expected_code_sha256,
         'Java evaluation code-map frozen SHA mismatch since H02')
    for project, h in m['sources']['csv_sha256'].items():
        p=Path(a.source_dir)/m['summary']['project_files'][project]['name']
        need(p.is_file() and sha_file(p)==h, f'Source CSV drift since H02 {p}')
    curated=read_json(path/'curated_csv_manifest.json')
    need(curated['curated_pair_rows']==a.expected_curated_pairs and
         curated['excluded_rows']==a.expected_excluded_rows and
         len(curated['files'])==a.expected_projects,
         'Prior H02 curated CSV census mismatch')
    for filename,info in curated['files'].items():
        need(Path(filename).name==filename and
             (path/'curated_csv'/filename).is_file() and
             sha_file(path/'curated_csv'/filename)==info['sha256'] and
             (path/'curated_csv'/filename).stat().st_size==info['bytes'],
             'H02 derived CSV snapshot drift: '+filename)
    need(m['summary']['exclusion']['pair_key']==a.exclude_conflict_pair_key
         and m['summary']['excluded_conflict_rows']==a.expected_excluded_rows
         and m['summary']['code_pair_label_conflicts']==0,
         'Prior H02 did not implement the current explicit conflict-exclusion policy')
    return path,m


def review_load_rows(a):
    h02,m=verify_h02(a)
    items=list(records(h02/'pair_records.jsonl'))
    grouped=collections.defaultdict(list)
    for r in items:
        grouped[r['pair_key']].append(r)
    need(len(items)==m['summary']['input_pairs'] and len(grouped)==m['summary']['unique_unordered_code_pairs'],
         'H02 row identity changed')
    return h02,m,items,grouped


def prepare_review(a):
    h02,m,items,grouped=review_load_rows(a)
    codes,_,_,_=load_codes(a)
    packet=[]
    for key, group in sorted(grouped.items()):
        chosen=sorted(group,key=lambda r:(r['project'],r['source_csv'],r['source_row'],r['left_id'],r['right_id']))[0]
        label_set=sorted(set(r['id_derived_label'] for r in group))
        packet.append({'pair_key':key,'project':chosen['project'],'legacy_labels':label_set,
                       'legacy_pair_occurrences':len(group),
                       'left_id':chosen['left_id'],'right_id':chosen['right_id'],
                       'left_code':codes[chosen['left_id']], 'right_code':codes[chosen['right_id']],
                       'left_code_sha256':chosen['left_code_sha256'],
                       'right_code_sha256':chosen['right_code_sha256'],
                       'source_rows':[{'project':r['project'],'source_csv':r['source_csv'],
                                       'source_row':r['source_row'],'id_derived_label':r['id_derived_label']}
                                      for r in group]})
    jsonl(a.out_dir/'annotation_packets.jsonl',packet)
    for who in ('a','b'):
        with (a.out_dir/('reviewer_'+who+'_template.csv')).open('x',encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(['pair_key','project','id_derived_labels','reviewer_label','reviewer_id','evidence_notes'])
            for x in packet:
                w.writerow([x['pair_key'],x['project'],'/'.join(map(str,x['legacy_labels'])),'','',''])
    state=dict(status='H03_V2_REVIEW_PACKETS_READY_NOT_GOLD',
               source_h02_sha256=sha_file(h02/'manifest.json'),candidate_pairs=len(items),
               unique_code_pairs=len(packet),conflicts=m['summary']['code_pair_label_conflicts'],
               reviewer_identity_authenticated=False,independent_blind_annotation_established=False,
               gold_label_verified=False,
               message='Two complete independently submitted reviewer CSVs required. Templates contain NO presumed labels.')
    write_json(a.out_dir/'annotation_audit.json',state)
    marker('h-03-prepare',a.out_dir,['annotation_packets.jsonl',
             'reviewer_a_template.csv','reviewer_b_template.csv','annotation_audit.json'],state,a=a)
    print(dump(state))


def reviewer_csv(path,expected):
    result={};identities=collections.Counter()
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f)
        need({'pair_key','reviewer_label','reviewer_id','evidence_notes'} <= set(r.fieldnames or []),
             f'Missing reviewer CSV fields: {path}')
        for n,row in enumerate(r,2):
            key=(row.get('pair_key') or '').strip()
            need(key in expected and key not in result,
                 f'Unexpected/duplicate review pair at {path}:{n}')
            label=(row.get('reviewer_label') or '').strip()
            rid=(row.get('reviewer_id') or '').strip()
            need(label in ('0','1') and rid,
                 f'Incomplete/uncertain reviewer label or reviewer ID at {path}:{n}')
            result[key]=int(label);identities[rid]+=1
    need(set(result)==set(expected), f'Reviewer coverage incomplete {path}: {len(result)} / {len(expected)}')
    need(len(identities)==1, f'A single reviewer file must have a consistent reviewer ID {path}')
    return result,next(iter(identities))


def validate_review(a):
    h02,m,items,grouped=review_load_rows(a)
    prepare,prior=verified_run(a.h03_prepare_run,'h-03-prepare',
                               ['annotation_packets.jsonl','annotation_audit.json'])
    packet=list(records(prepare/'annotation_packets.jsonl'))
    need(len(packet)==len(grouped),'H03 prepare packet coverage drift')
    need(prior['summary']['source_h02_sha256']==sha_file(h02/'manifest.json'),
         'H03 prepared packet was not derived from this H02 frozen run')
    want={x['pair_key'] for x in packet}
    need(want==set(grouped),'H03 reviewer IDs differ from H02 source')
    left,right=Path(a.reviewer_a),Path(a.reviewer_b)
    need(left.resolve()!=right.resolve() and left.is_file() and right.is_file(),
         'Two distinct complete reviewer CSVs required')
    a1,n1=reviewer_csv(left,want);a2,n2=reviewer_csv(right,want)
    need(n1!=n2,'Two distinct reviewer identifiers required')
    disagreements=[k for k in sorted(want) if a1[k]!=a2[k]]
    jsonl(a.out_dir/'reviewer_disagreements.jsonl',[{'pair_key':k,'reviewer_a':a1[k],
                        'reviewer_b':a2[k]} for k in disagreements])
    labels=[]
    if not disagreements:
        labels=[{'pair_key':k,'gold_label':a1[k],
                 'label_source':'TWO_REVIEWER_CSV_REPORTED_CONSENSUS',
                 'reviewer_identity_authenticated':False,
                 'independent_blind_review_verified':False} for k in sorted(want)]
    jsonl(a.out_dir/'reviewed_pair_labels.jsonl',labels)
    report=dict(status='H03_V2_REVIEWER_CSV_CONSENSUS_RECORDED_IDENTITY_NOT_AUTHENTICATED'
                if not disagreements else 'H03_V2_REVIEWER_DISAGREEMENT_BLOCKED',
                reviewed_unique_pairs=len(labels),expected_unique_pairs=len(want),
                disagreements=len(disagreements),reviewer_a_id=n1,reviewer_b_id=n2,
                reviewer_a_sha256=sha_file(left),reviewer_b_sha256=sha_file(right),
                source_h02_sha256=sha_file(h02/'manifest.json'),
                source_h03_prepare_sha256=sha_file(prepare/'manifest.json'),
                review_convention='EXPLICIT_0_OR_1_ONLY_NO_BLANK_DEFAULTS',
                reviewer_identity_authenticated=False,
                independent_blind_annotation_established=False,
                false_independent_claim_blocked=True)
    write_json(a.out_dir/'annotation_audit.json',report)
    marker('h-03-validate',a.out_dir,['reviewed_pair_labels.jsonl',
               'reviewer_disagreements.jsonl','annotation_audit.json'],report,a=a)
    print(dump(report))
    need(not disagreements,'Reviewer disagreements unresolved')


def load_g02_codes(man, required_ids):
    # Stream each SHA-pinned training code corpus, retain ONLY the ~40k endpoints
    # referenced by the frozen G02 train/validation rows (not full BigCloneBench).
    fp=man['frozen_mixed']
    need(set(required_ids)==set(fp['code_paths']),
         'Mismatch between frozen G02 consumed corpora and source code maps')
    base={}
    for corpus,path in fp['code_paths'].items():
        code_path=Path(path)
        need(code_path.is_file() and sha_file(code_path)==fp['code_sha256'][corpus],
             f'G02 corpus code source SHA drift: {code_path}')
        wanted=required_ids[corpus];d={}
        for row in records(code_path):
            idx=str(row['idx'])
            if idx in wanted:
                src=row['func']
                need(idx not in d and isinstance(src,str),f'Duplicate/missing G02 code {idx}')
                d[idx]=src
        need(set(d)==wanted, f'Missing SHA-pinned G02 code endpoints: {corpus}={len(wanted-set(d))}')
        base[corpus]=d
    return base


def leakage(a):
    h02,m,rows,grouped=review_load_rows(a)
    codes,_,_,_ = load_codes(a)
    eval_exact=set();eval_norm=set();eval_exact_pairs=set();eval_norm_pairs=set()
    for idx in {id for r in rows for id in (r['left_id'],r['right_id'])}:
        x=codes[idx]
        eval_exact.add(sha_string(x));eval_norm.add(sha_string(''.join(x.split())))
    for r in rows:
        eval_exact_pairs.add(tuple(sorted((r['left_code_sha256'],r['right_code_sha256']))))
        eval_norm_pairs.add(tuple(sorted((sha_string(''.join(codes[r['left_id']].split())),
                                          sha_string(''.join(codes[r['right_id']].split()))))))
    prep=Path(a.g02_prepared).resolve()
    need(sha_file(prep/'manifest.json') == a.expected_g02_prepared_manifest_sha256,
         'G02 frozen prepared manifest SHA mismatch')
    manifest=read_json(prep/'manifest.json')
    need(manifest.get('status')=='G02_V2_PREPARE_PASS_NOT_TRAINED', 'G02 frozen manifest status unexpected')
    need(manifest['files']['train']['rows']==a.expected_g02_train_rows and
         manifest['files']['validation']['rows']==a.expected_g02_validation_rows,
         'G02 train/validation record budget drift')
    need(sha_file(prep/'train.jsonl') == a.expected_g02_train_sha256 == manifest['files']['train']['sha256'],
         'G02 training input SHA mismatch')
    need(sha_file(prep/'validation.jsonl') == a.expected_g02_validation_sha256 == manifest['files']['validation']['sha256'],
         'G02 validation input SHA mismatch')
    frozen_rows={split:list(records(prep/(split+'.jsonl'))) for split in ('train','validation')}
    required_ids=collections.defaultdict(set)
    for split_rows in frozen_rows.values():
        for row in split_rows:
            required_ids[row['corpus']].update((str(row['id1']),str(row['id2'])))
    bases=load_g02_codes(manifest,required_ids)
    summaries={};hits=[];total=0
    for split in ('train','validation'):
        count=collections.Counter()
        for r in frozen_rows[split]:
            total+=1; count['pairs']+=1
            corpus=r['corpus'];mapping=bases[corpus]
            l=r.get('code_a_override',mapping[str(r['id1'])]);
            rr=r.get('code_b_override',mapping[str(r['id2'])])
            h1=sha_string(l);h2=sha_string(rr)
            need(h1==r['code1_sha256'] and h2==r['code2_sha256'],
                 f'G02 consumption code content mismatch {split}:{corpus}:{r["id1"]}:{r["id2"]}')
            e1,e2=(h1,h2); n1,n2=(sha_string(''.join(l.split())),sha_string(''.join(rr.split())))
            checks={
              'function_exact':(e1 in eval_exact)+(e2 in eval_exact),
              'function_normalized':(n1 in eval_norm)+(n2 in eval_norm),
              'pair_exact':int(tuple(sorted((e1,e2))) in eval_exact_pairs),
              'pair_normalized':int(tuple(sorted((n1,n2))) in eval_norm_pairs)}
            for typ,v in checks.items():
                if v:count[typ+'_endpoint_or_pair_incidence']+=v
            if any(checks.values()) and len(hits)<a.max_overlap_examples:
                hits.append(dict(split=split,corpus=corpus,id1=str(r['id1']),id2=str(r['id2']),
                                 exact_code_hashes=[e1,e2],normalized_code_hashes=[n1,n2],findings=checks))
        need(count['pairs']==manifest['files'][split]['rows'],f'G02 {split} row census mismatch')
        summaries[split]=dict(count)
    blocked=any(v for d in summaries.values() for k,v in d.items() if k!='pairs')
    report=dict(status='H04_V2_EXACT_OR_NORMALIZED_LEAKAGE_BLOCKED' if blocked else
                  'H04_V2_EXACT_NORMALIZED_ISOLATION_PASS',
                at_chicago=chicago(), G02_prepared_manifest_sha256=sha_file(prep/'manifest.json'),
                G02_train_sha256=sha_file(prep/'train.jsonl'),
                G02_validation_sha256=sha_file(prep/'validation.jsonl'),
                G02_consumed_pairs=total, splits=summaries, conflict_count=m['summary']['code_pair_label_conflicts'],
                exact_normalized_leakage_found=blocked, near_duplicate_semantic_review='OUT_OF_SCOPE_BY_PROTOCOL',
                note='M3 inherits G02 frozen pair contents; test code vs M1 source prior training must be independently confirmed by G01 record; not an automatic assertion of global no leakage.')
    write_json(a.out_dir/'leakage_audit.json',report)
    jsonl(a.out_dir/'overlap_examples.jsonl',hits)
    marker('h-04',a.out_dir,['leakage_audit.json','overlap_examples.jsonl'],report,
           sources={'h02_manifest_sha256':sha_file(h02/'manifest.json')},a=a)
    print(dump(report))
    need(not blocked,'H04 blocked on exact/normalized training/validation leakage')


def freeze(a):
    h02,m,items,grouped=review_load_rows(a)
    r3,v3=verified_run(a.h03_run,'h-03-validate',
                       ['reviewed_pair_labels.jsonl','annotation_audit.json'])
    r4,v4=verified_run(a.h04_run,'h-04',['leakage_audit.json','overlap_examples.jsonl'])
    need(v3['summary']['status']=='H03_V2_REVIEWER_CSV_CONSENSUS_RECORDED_IDENTITY_NOT_AUTHENTICATED'
         and v3['summary']['disagreements']==0,'H03 two-reviewer consensus missing')
    need(v3['summary']['source_h02_sha256']==sha_file(h02/'manifest.json'),
         'H03 not derived from selected H02')
    need(v4['sources']['h02_manifest_sha256']==sha_file(h02/'manifest.json') and
         not v4['summary']['exact_normalized_leakage_found'] and
         v4['summary']['status']=='H04_V2_EXACT_NORMALIZED_ISOLATION_PASS',
         'H04 leakage clearance missing')
    need(a.authorize_dedup=='true' and a.accept_exact_normalized_scope=='true',
         'Explicit Shell approval required for dedup and exact/normalized-only leakage scope')
    labels={r['pair_key']:r for r in records(r3/'reviewed_pair_labels.jsonl')}
    need(set(labels)==set(grouped),'H03 review pair coverage differs from H02')
    codes,_,_,_=load_codes(a)
    gold=[]; removed=[]
    excluded_conflicts=list(records(h02/'excluded_conflict_rows.jsonl'))
    need(len(excluded_conflicts)==a.expected_excluded_rows,'Excluded H02 conflict evidence missing')
    for key,part in sorted(grouped.items()):
        chosen=sorted(part,key=lambda r:(r['project'],r['source_csv'],r['source_row'],r['left_id'],r['right_id']))[0]
        label=labels[key]['gold_label']
        need(type(label) is int and label in (0,1),'Unapproved adjudicated label')
        src1=codes[chosen['left_id']];src2=codes[chosen['right_id']]
        legacy=sorted({r['id_derived_label'] for r in part})
        gold.append(dict(pair_id=key,project=chosen['project'],language='java',gold_label=label,
                         evaluation_eligible=True,legacy_id_derived_labels=legacy,
                         original_candidate_occurrences=len(part),
                         reviewer_identity_authenticated=False,
                         independent_blind_annotation_established=False,
                         label_source=labels[key]['label_source'],
                         left=dict(source_code=src1,code_sha256=sha_string(src1),original_id=chosen['left_id']),
                         right=dict(source_code=src2,code_sha256=sha_string(src2),original_id=chosen['right_id'])))
        for r in part:
            if r is not chosen:
                removed.append(dict(pair_key=key,reason='DUPLICATE_UNORDERED_CODE_PAIR',source=r))
    target=Path(a.frozen_dir).resolve()
    need(not target.exists(),'Cannot overwrite frozen Java GT: '+str(target))
    need(target != Path(a.source_dir).resolve() and target != Path(a.code_map).resolve().parent,
         'Unsafe GT target')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.mkdir(exist_ok=False)
    pos=[r for r in gold if r['gold_label']==1];neg=[r for r in gold if r['gold_label']==0]
    jsonl(target/'verified_pairs.jsonl',gold)
    jsonl(target/'positive_verified_pairs.jsonl',pos)
    jsonl(target/'negative_verified_pairs.jsonl',neg)
    jsonl(target/'excluded_duplicate_code_pairs.jsonl',removed)
    jsonl(target/'excluded_conflict_rows.jsonl',excluded_conflicts)
    audit=dict(status='H05_V2_FROZEN_JAVA_REVIEWER_REPORTED_GROUND_TRUTH',
               at_chicago=chicago(),original_candidate_pairs=len(items)+len(excluded_conflicts),
               reviewed_candidate_occurrences=len(items),
               excluded_conflict_occurrences=len(excluded_conflicts),
               frozen_unique_pairs=len(gold),
               dropped_duplicate_occurrences=len(removed),positives=len(pos),negatives=len(neg),
               reviewer_identity_authenticated=False,independent_blind_annotation_established=False,
               conflict_policy='EXCLUDE_BOTH_CONTRADICTORY_ROWS_NO_LABEL_REASSIGNMENT',
               duplicate_policy='EXPLICIT_USER_AUTHORIZED_UNORDERED_CODE_HASH_PAIR_DEDUPLICATION',
               leakage_scope='EXACT_AND_WHITESPACE_NORMALIZED_ONLY_USER_ACCEPTED',
               upstream_label_conflicts_excluded=m['summary']['excluded_conflict_rows'],
               source_h02_manifest_sha256=sha_file(h02/'manifest.json'),
               source_h03_manifest_sha256=sha_file(r3/'manifest.json'),
               source_h04_manifest_sha256=sha_file(r4/'manifest.json'),
               limitations=['Review CSV identity and independent blind annotation not authenticated.',
                            'Legacy paper Pair counts not reproduced by deduplicated frozen evaluation.',
                            'Only exact/normalized leakage is in scope for this evaluation protocol.'])
    write_json(target/'annotation_audit.json',audit)
    write_json(target/'leakage_audit.json',v4['summary'])
    marker('h-05-frozen',target,['verified_pairs.jsonl','positive_verified_pairs.jsonl',
            'negative_verified_pairs.jsonl','excluded_duplicate_code_pairs.jsonl',
            'excluded_conflict_rows.jsonl','annotation_audit.json','leakage_audit.json'],audit,a=a)
    write_json(a.out_dir/'freeze_result.json',
               dict(status=audit['status'],frozen_dir=str(target),
                    frozen_manifest_sha256=sha_file(target/'manifest.json'),audit=audit))
    marker('h-05',a.out_dir,['freeze_result.json'],audit,a=a)
    print(dump(dict(status=audit['status'],frozen_dir=str(target),
                    verified_pairs=len(gold),positive=len(pos),negative=len(neg))))


def verify(a):
    target,m=verified_run(a.frozen_dir,'h-05-frozen',
           ['verified_pairs.jsonl','positive_verified_pairs.jsonl',
            'negative_verified_pairs.jsonl','excluded_duplicate_code_pairs.jsonl',
            'excluded_conflict_rows.jsonl','annotation_audit.json','leakage_audit.json'])
    source=read_json(target/'annotation_audit.json')
    orig=collections.Counter(); key_count=set(); positive=[];negative=[]
    for row in records(target/'verified_pairs.jsonl'):
        label=row['gold_label'];need(type(label) is int and label in (0,1),'Invalid binary gold label')
        need(row['language']=='java' and row['evaluation_eligible'] is True, 'Invalid frozen record status')
        need(row['pair_id'] not in key_count,'Repeated frozen pair')
        code_hashes=[]
        for side in ('left','right'):
            x=row[side]
            need(isinstance(x['source_code'],str) and x['source_code'].strip(), 'Empty frozen Java code')
            actual=sha_string(x['source_code'])
            need(actual==x['code_sha256'], 'Frozen effective Java code SHA mismatch')
            code_hashes.append(actual)
        need(row['pair_id']==pair_key(*code_hashes),'Frozen pair key/hash mismatch')
        key_count.add(row['pair_id']); orig[label]+=1
        (positive if label else negative).append(row)
    def same_set(a,b):
        return collections.Counter(json.dumps(x,sort_keys=True,ensure_ascii=False)
                                   for x in a)==collections.Counter(json.dumps(x,sort_keys=True,ensure_ascii=False) for x in b)
    need(same_set(positive,list(records(target/'positive_verified_pairs.jsonl'))) and
         same_set(negative,list(records(target/'negative_verified_pairs.jsonl'))),
         'Merged row set differs from class-specific JSONL')
    need(len(key_count)==source['frozen_unique_pairs'] and
         orig[1]==source['positives'] and orig[0]==source['negatives'] and
         source['original_candidate_pairs']==len(key_count)+source['dropped_duplicate_occurrences']+source['excluded_conflict_occurrences']
         and source['excluded_conflict_occurrences']==len(list(records(target/'excluded_conflict_rows.jsonl'))),
         'Frozen count/duplicate provenance drift')
    report=dict(status='H06_V2_FROZEN_JAVA_GT_INTEGRITY_PASS_PROVENANCE_LIMITATIONS_RETAINED',
                at_chicago=chicago(),frozen_dir=str(target),manifest_sha256=sha_file(target/'manifest.json'),
                total=len(key_count),positive=orig[1],negative=orig[0],
                reviewer_identity_authenticated=False,independent_blind_annotation_established=False,
                near_duplicate_semantic_review='OUT_OF_SCOPE_BY_PROTOCOL',
                evaluation_eligibility_scope='REVIEWER_REPORTED_GOLD_EXACT_NORMALIZED_ISOLATION_ONLY')
    write_json(a.out_dir/'verification.json',report)
    marker('h-06',a.out_dir,['verification.json'],report,a=a)
    print(dump(report))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['h-01','h-02','h-03-prepare','h-03-validate',
                                    'h-04','h-05','h-06'])
    for name in ('root','out-dir','source-dir','code-map','expected-code-sha256',
                 'g02-prepared','expected-g02-train-sha256','expected-g02-validation-sha256',
                 'expected-g02-prepared-manifest-sha256',
                 'h02-run','h03-prepare-run','h03-run','h04-run','reviewer-a','reviewer-b',
                 'frozen-dir','authorize-dedup','accept-exact-normalized-scope',
                 'exclude-conflict-pair-key','exclude-conflict-source-rows'):
        p.add_argument('--'+name,required=True)
    for name in ('expected-projects','reference-projects','expected-pairs','expected-positive',
                 'expected-negative','reference-positive','reference-negative','expected-code-rows',
                 'expected-g02-train-rows','expected-g02-validation-rows',
                 'max-overlap-examples','expected-conflict-groups-before-exclusion',
                 'expected-excluded-rows','expected-curated-pairs','expected-curated-positive',
                 'expected-curated-negative','expected-curated-unique-code-pairs'):
        p.add_argument('--'+name,required=True,type=int)
    a=p.parse_args()
    a.root=Path(a.root).resolve();a.out_dir=Path(a.out_dir).resolve()
    need(a.out_dir.is_dir() and not any(a.out_dir.iterdir()), 'New stage output directory must exist and be empty')
    need(a.expected_projects>0 and a.expected_pairs==a.expected_positive+a.expected_negative and
         a.expected_positive>0 and a.expected_negative>0 and a.expected_code_rows>0,
         'Invalid explicit Shell census configuration')
    need(a.expected_excluded_rows>0 and a.expected_conflict_groups_before_exclusion==1 and
         a.expected_curated_pairs==a.expected_pairs-a.expected_excluded_rows and
         a.expected_curated_positive+a.expected_curated_negative==a.expected_curated_pairs,
         'Inconsistent explicit Shell exclusion and curated pair counts')
    need(a.authorize_dedup in ('true','false') and a.accept_exact_normalized_scope in ('true','false'),
         'Invalid explicit Shell authorization flags')
    stages={'h-01':inventory,'h-02':identity,'h-03-prepare':prepare_review,
            'h-03-validate':validate_review,'h-04':leakage,'h-05':freeze,'h-06':verify}
    stages[a.stage](a)


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print(json.dumps(dict(status='H_SERIES_FAIL',error_type=type(e).__name__,
                              error=str(e),at_chicago=chicago()),sort_keys=True),file=sys.stderr)
        raise
