#!/usr/bin/env python3
"""J-03d v6: deterministic, content-deduplicated positive-only subject balancing.

Runtime configuration comes only from the standalone J-03d Shell wrapper.
No negative samples, training splits, or changes to original verified CSV/JSONL.
"""
import csv
from collections import Counter
from datetime import datetime
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import time
from zoneinfo import ZoneInfo

T1 = 'TYPE_1_SIGNATURE'
T2C = 'TYPE_2_CONSISTENT_SIGNATURE'
T2B = 'TYPE_2_BLIND_SIGNATURE'
T3 = 'TYPE_3_STRUCTURAL_CANDIDATE'
CATEGORIES = (T1, T2C, T2B, T3)
CENSUS_CATS = ('SEMANTIC_UNDETERMINED', 'STRUCTURAL_UNSCORED', 'INELIGIBLE')


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def stamp(args):
    return datetime.now(ZoneInfo(args.time_zone)).isoformat(timespec='seconds')


def output_line(obj):
    # Compact one JSON object per line; source code newlines remain escaped.
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(',', ':')) + '\n'


def parse_fixed_table(md, label, expected_cols):
    require(md.count(label) == 1, 'POSITIVE_CENSUS_TABLE_MISSING_OR_DUPLICATED '+label)
    remainder = md.split(label, 1)[1]
    require('```text' in remainder, 'POSITIVE_CENSUS_TABLE_FENCE_MISSING '+label)
    block = remainder.split('```text', 1)[1].split('```', 1)[0]
    lines = [line.split() for line in block.splitlines() if line.strip()]
    require(len(lines) > 3 and len(lines[0]) == expected_cols,
            'POSITIVE_CENSUS_TABLE_INVALID_HEADER '+label)
    result = {}
    for columns in lines[1:]:
        if len(columns) != expected_cols or columns[0].startswith('--'):
            continue
        name = columns[0]
        require(name not in result, 'POSITIVE_CENSUS_DUPLICATED_SUBJECT '+name)
        require(all(c.replace(',', '').isdigit() for c in columns[1:]),
                'POSITIVE_CENSUS_NOT_NUMERIC '+name)
        result[name] = [int(c.replace(',', '')) for c in columns[1:]]
    require('TOTAL' in result, 'POSITIVE_CENSUS_TOTAL_MISSING '+label)
    return result


def load_census(args, names, original):
    path = args.output_root / args.positive_source_census_name
    require(path.is_file() and not path.is_symlink(),
            'POSITIVE_CENSUS_REQUIRED path='+str(path))
    actual_sha = original.sha_file(path)
    require(actual_sha == args.positive_source_census_sha256,
            'POSITIVE_CENSUS_SHA256_MISMATCH expected='+args.positive_source_census_sha256+
            ' actual='+actual_sha)
    md = path.read_text(encoding='utf-8')
    a = parse_fixed_table(md, '**A. Structural clone candidates**', 7)
    b = parse_fixed_table(md, '**B. Undetermined / unscored / ineligible**', 4)
    require(set(a) == set(b) == set(names) | {'TOTAL'},
            'POSITIVE_CENSUS_SUBJECT_SET_MISMATCH')
    summary = {}
    for name in [*names, 'TOTAL']:
        pairs, t1, t2c, t2b, t2all, t3 = a[name]
        undet, unsc, inelig = b[name]
        require(t2c+t2b == t2all and pairs == t1+t2all+t3+undet+unsc+inelig,
                'POSITIVE_CENSUS_SUBJECT_COUNT_INVALID '+name)
        summary[name] = {'pairs':pairs,'counts': {T1:t1,T2C:t2c,T2B:t2b,T3:t3,
                     CENSUS_CATS[0]:undet,CENSUS_CATS[1]:unsc,CENSUS_CATS[2]:inelig}}
    require(summary['TOTAL']['pairs'] == args.expected_total_pairs,
            'POSITIVE_CENSUS_EXPECTED_TOTAL_MISMATCH')
    require(all(sum(summary[n]['counts'][t] for n in names)==summary['TOTAL']['counts'][t]
                for t in (*CATEGORIES,*CENSUS_CATS)),
            'POSITIVE_CENSUS_SUMMED_TYPE_MISMATCH')
    require(sum(summary[n]['pairs'] for n in names) == summary['TOTAL']['pairs'],
            'POSITIVE_CENSUS_SUMMED_PAIR_MISMATCH')
    print(f'POSITIVE_CENSUS_PASS subjects={len(names)} pairs={summary["TOTAL"]["pairs"]} '
          f'sha256={actual_sha} report={path}', flush=True)
    return summary


def make_pool(path, args):
    require(not path.exists() and not path.is_symlink(), 'POSITIVE_POOL_ALREADY_EXISTS')
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=NORMAL')
    db.execute(f'PRAGMA cache_size=-{args.positive_sqlite_cache_kib}')
    db.execute('CREATE TABLE candidates (code_pair_key TEXT PRIMARY KEY, rank TEXT NOT NULL, '
               'pair_id TEXT NOT NULL, category TEXT NOT NULL, score REAL)')
    return db


def consume_pairs(args, subject, pool):
    """Deduplicate location-repeated source content pairs using unordered hashes."""
    observed = Counter()
    writes = [0]
    insert = ('INSERT INTO candidates(code_pair_key,rank,pair_id,category,score) '
              'VALUES (?,?,?,?,?) ON CONFLICT(code_pair_key) DO UPDATE SET '
              'rank=excluded.rank,pair_id=excluded.pair_id,score=excluded.score '
              'WHERE excluded.rank < candidates.rank AND excluded.category=candidates.category')

    def callback(row, typ, score):
        observed[typ] += 1
        if typ in CATEGORIES:
            a, b = sorted((row['source_sha256_a'], row['source_sha256_b']))
            require(len(a)==64 and len(b)==64, 'POSITIVE_SOURCE_HASH_INVALID')
            code_pair_key = sha256((a+':'+b).encode('ascii')).hexdigest()
            rank = sha256((str(args.positive_sample_seed)+'\0'+subject+'\0'+
                           row['pair_id']).encode('utf-8')).hexdigest()
            pool.execute(insert, (code_pair_key,rank,row['pair_id'],typ,score))
            writes[0] += 1
        if sum(observed.values()) % args.positive_progress_pairs == 0:
            pool.commit()
            print(f'[{stamp(args)}] POSITIVE_SCAN_PROGRESS subject={subject} '
                  f'pairs={sum(observed.values())} eligible_candidates={writes[0]}',flush=True)
    return callback, observed


def quotas_from_counts(available, args, subject):
    t1, t2c, t2b, t3 = (available[k] for k in CATEGORIES)
    t2 = t2c+t2b
    present = [x for x in (t1,t2,t3) if x > 0]
    require(len(present) >= 2, 'POSITIVE_INSUFFICIENT_CLONE_TYPES subject='+subject)
    q = min(present)
    target_t1 = q if t1 else 0
    target_t2 = q if t2 else 0
    target_t3 = q if t3 else 0
    blind = 0
    if target_t2:
        blind = (target_t2*t2b + t2//2)//t2  # round half up
        min_possible = max(0, target_t2-t2c)
        max_possible = min(target_t2, t2b)
        if t2b and t2c and target_t2 >= 2:
            min_possible = max(min_possible, min(args.positive_type2_blind_min,
                                                 target_t2-1, t2b))
        blind = max(min_possible, min(blind, max_possible))
    selected = {T1:target_t1,T2C:target_t2-blind,T2B:blind,T3:target_t3}
    require(all(selected[k] <= available[k] for k in CATEGORIES),
            'POSITIVE_QUOTA_EXCEEDS_POOL subject='+subject)
    require(sum(selected.values()) > 0, 'POSITIVE_EMPTY_SELECTION subject='+subject)
    warning = []
    if not t1:warning.append('MISSING_TYPE_1')
    if not t2:warning.append('MISSING_TYPE_2')
    if not t3:warning.append('MISSING_TYPE_3')
    require(not warning or warning == ['MISSING_TYPE_1'],
            'POSITIVE_ONLY_TYPE1_ABSENCE_SUPPORTED subject='+subject)
    return selected, q, warning


def choose_pairs(pool, quotas):
    pool.execute('CREATE INDEX candidates_category_rank ON candidates(category,rank,pair_id)')
    result={}
    for category in CATEGORIES:
        qty = quotas[category]
        if not qty:continue
        for pair_id,typ,score in pool.execute('SELECT pair_id,category,score FROM candidates '
                    'WHERE category=? ORDER BY rank,pair_id LIMIT ?', (category,qty)):
            require(pair_id not in result, 'POSITIVE_DUPLICATE_SELECTED_PAIR_ID '+pair_id)
            result[pair_id] = (typ,score)
    require(len(result) == sum(quotas.values()), 'POSITIVE_SELECTED_QUOTA_MISMATCH')
    return result


def load_selected_rows(args, subject, selected, original):
    rows = []
    fragments = set()
    path = args.output_root/subject/'positive_pairs.csv'
    fields = set(original.PAIR_FIELDS)|{'label'}
    for row in original.csv_rows(path, fields):
        if row['pair_id'] not in selected:continue
        cat, score=selected[row['pair_id']]
        require(row['label'] == str(args.positive_label) and
                row['repository_a'] == row['repository_b'] == subject,
                'POSITIVE_SELECTED_METADATA_MISMATCH subject='+subject)
        require(row['fragment_a_id'] != row['fragment_b_id'],
                'POSITIVE_SELECTED_SELF_PAIR subject='+subject)
        rows.append((row,cat,score))
        fragments.add(row['fragment_a_id']);fragments.add(row['fragment_b_id'])
    require(len(rows) == len(selected), 'POSITIVE_SELECTED_PAIR_MISSING subject='+subject)
    fragment_codes = {}
    with (args.output_root/subject/'fragments.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():raise RuntimeError('POSITIVE_BLANK_FRAGMENT_RECORD subject='+subject)
            r=json.loads(line)
            id_=r['fragment_id']
            if id_ not in fragments:continue
            require(id_ not in fragment_codes, 'POSITIVE_DUPLICATE_FRAGMENT_SELECTED '+id_)
            raw = r['code'].encode(args.code_encoding, errors=args.code_errors)
            require(original.sha_bytes(raw)==r['source_sha256'],
                    'POSITIVE_SELECTED_CODE_HASH_MISMATCH '+id_)
            fragment_codes[id_] = r
    require(set(fragment_codes)==fragments, 'POSITIVE_SELECTED_FRAGMENT_MISSING subject='+subject)
    return rows, fragment_codes


def render_outputs(args, subject, rows, fragments, stage):
    pub = stage / args.positive_public_name
    detail = stage / args.positive_details_dir / args.positive_details_name
    detail.parent.mkdir(mode=0o755)
    ids=set(); content_pairs=set(); counts=Counter()
    with pub.open('x',encoding='utf-8') as out_pub, detail.open('x',encoding='utf-8') as out_details:
        for row, category, score in rows:
            a, b = fragments[row['fragment_a_id']],fragments[row['fragment_b_id']]
            require(a['source_sha256']==row['source_sha256_a'] and
                    b['source_sha256']==row['source_sha256_b'],
                    'POSITIVE_SOURCE_METADATA_JOIN_MISMATCH '+row['pair_id'])
            keys=tuple(sorted((a['source_sha256'],b['source_sha256'])))
            require(keys not in content_pairs, 'POSITIVE_DUPLICATE_CODE_PAIR '+row['pair_id'])
            content_pairs.add(keys)
            sample_id='pos-'+sha256((subject+'\0'+row['pair_id']).encode()).hexdigest()[:24]
            require(sample_id not in ids,'POSITIVE_SAMPLE_ID_COLLISION')
            ids.add(sample_id)
            clone_type='Type 1' if category==T1 else 'Type 3' if category==T3 else 'Type 2'
            public={'sample_id':sample_id,'repository':subject,'clone_type':clone_type,
                    'label':args.positive_label,'code_a':a['code'],'code_b':b['code']}
            detail_item={**public,
                'clone_subtype':category,
                'original_pair_id':row['pair_id'],
                'nicad_similarity_percent':row['nicad_similarity_raw'],
                'ast_combined_similarity':round(score,8) if score is not None else None,
                'ast_primary_threshold':str(args.ast_primary_threshold),
                'detector':row['detector'],'detector_config':row['config'],
                'source_a':{'fragment_id':a['fragment_id'],'file':a['file_path'],
                    'start_line':a['start_line'],'end_line':a['end_line'],
                    'commit':a['commit'],'source_sha256':a['source_sha256']},
                'source_b':{'fragment_id':b['fragment_id'],'file':b['file_path'],
                    'start_line':b['start_line'],'end_line':b['end_line'],
                    'commit':b['commit'],'source_sha256':b['source_sha256']}}
            out_pub.write(output_line(public))
            out_details.write(output_line(detail_item))
            counts[category]+=1
    require(len(rows)==len(ids)==len(content_pairs) and len(rows)==sum(counts.values()),
            'POSITIVE_OUTPUT_PAIR_COUNT_INVALID subject='+subject)
    return pub,detail,counts


def compare_or_publish(src, dst, original):
    require(not dst.is_symlink(), 'POSITIVE_OUTPUT_SYMLINK_FORBIDDEN '+str(dst))
    if dst.exists():
        require(dst.is_file() and original.sha_file(src)==original.sha_file(dst),
                'POSITIVE_EXISTING_OUTPUT_DIFFERS path='+str(dst))
        print('POSITIVE_OUTPUT_REUSED_VERIFIED path='+str(dst), flush=True)
        return
    try:
        os.link(src,dst)  # atomic no-replace for each artifact
    except FileExistsError:
        require(original.sha_file(src)==original.sha_file(dst),
                'POSITIVE_RACE_OUTPUT_DIFFERS '+str(dst))
    print('POSITIVE_OUTPUT_PUBLISHED path='+str(dst),flush=True)


def manifest_effective(args):
    return {
        'version':args.version,'experiment':args.experiment,
        'seed':args.positive_sample_seed,
        'policy':args.positive_balance_policy,
        'target_type_ratio':args.positive_target_type_ratio,
        'missing_type_policy':args.positive_missing_type_policy,
        'type2_blind_min':args.positive_type2_blind_min,
        'dedup_strategy':args.positive_dedup_strategy,
        'primary_ast_threshold':str(args.ast_primary_threshold),
        'gram_size':args.ast_gram_size,
        'max_ast_comparison_cells':args.ast_max_cells,
        'positive_label':args.positive_label,
        'code_encoding':args.code_encoding,'code_errors':args.code_errors,
        'public_name':args.positive_public_name,
        'details_dir':args.positive_details_dir,
        'details_name':args.positive_details_name,
        'manifest_name':args.positive_manifest_name,
        'report_name':args.positive_report_name,
        'census_name':args.positive_source_census_name,
        'census_sha256':args.positive_source_census_sha256,
        'timezone':args.time_zone,
    }


def verify_published(args, subject, census, original):
    dest=args.output_root/subject
    pub=dest/args.positive_public_name
    details=dest/args.positive_details_dir/args.positive_details_name
    manifest=dest/args.positive_details_dir/args.positive_manifest_name
    for path in (pub,details,manifest):
        require(path.is_file() and not path.is_symlink(),
                'POSITIVE_PUBLISHED_FILE_REQUIRED '+str(path))
    m=original.read_json(manifest)
    require(m['subject']==subject and m['status']=='FIXED_POSITIVE_ONLY' and
            m['effective_configuration']==manifest_effective(args),
            'POSITIVE_MANIFEST_METADATA_MISMATCH '+subject)
    require(m['census_counts']==census['counts'] and m['original_pairs']==census['pairs'],
            'POSITIVE_MANIFEST_CENSUS_MISMATCH '+subject)
    source_m=dest/'manifest.json'
    require(m['source_manifest_sha256']==original.sha_file(source_m),
            'POSITIVE_SOURCE_MANIFEST_MODIFIED '+subject)
    src=original.read_json(source_m)
    require(m['source_pairs_sha256']==src['files_sha256']['positive_pairs.csv'] and
            m['source_fragments_sha256']==src['files_sha256']['fragments.jsonl'],
            'POSITIVE_SOURCE_HASH_MISMATCH '+subject)
    require(m['h07_source_sha256']==original.sha_file(args.h07_python) and
            m['h08_source_sha256']==original.sha_file(args.h08_python),
            'POSITIVE_AST_REFERENCE_CHANGED '+subject)
    expected_selected, _, warnings=quotas_from_counts(m['unique_content_pair_counts'],args,subject)
    require(m['selected_counts']==expected_selected and m['warnings']==warnings,
            'POSITIVE_MANIFEST_BALANCE_QUOTAS_CHANGED '+subject)
    require(m['files_sha256'][args.positive_public_name]==original.sha_file(pub) and
            m['files_sha256'][args.positive_details_dir+'/'+args.positive_details_name]==original.sha_file(details),
            'POSITIVE_PUBLISHED_SHA256_MISMATCH '+subject)
    count=0;typecounts=Counter(); ids=set(); codekeys=set()
    with pub.open(encoding='utf-8') as f1, details.open(encoding='utf-8') as f2:
        for ln,(p,d) in enumerate(zip(f1,f2,strict=True),1):
            x=json.loads(p); y=json.loads(d)
            require(x['sample_id']==y['sample_id'] and x['repository']==subject and
                    x['label']==args.positive_label and x['clone_type'] in ('Type 1','Type 2','Type 3'),
                    'POSITIVE_OUTPUT_JOIN_INVALID '+subject+':'+str(ln))
            for key in ('code_a','code_b','clone_type'):
                require(x[key]==y[key], 'POSITIVE_PUBLIC_DETAIL_MISMATCH '+subject+':'+str(ln))
            hashes=tuple(sorted((original.sha_bytes(x['code_a'].encode(args.code_encoding,errors=args.code_errors)),
                                 original.sha_bytes(x['code_b'].encode(args.code_encoding,errors=args.code_errors)))))
            require(hashes==(tuple(sorted((y['source_a']['source_sha256'],y['source_b']['source_sha256'])))),
                    'POSITIVE_OUTPUT_CODE_HASH_MISMATCH '+subject+':'+str(ln))
            require(x['sample_id'] not in ids and hashes not in codekeys,
                    'POSITIVE_OUTPUT_DUPLICATE_PAIR '+subject+':'+str(ln))
            ids.add(x['sample_id']);codekeys.add(hashes)
            require(y['clone_subtype'] in CATEGORIES,'POSITIVE_OUTPUT_BAD_SUBTYPE')
            expected_type = ('Type 1' if y['clone_subtype']==T1 else
                             'Type 3' if y['clone_subtype']==T3 else 'Type 2')
            require(x['clone_type']==expected_type,'POSITIVE_OUTPUT_TYPE_INCONSISTENCY')
            typecounts[y['clone_subtype']]+=1;count+=1
    require(count==m['fixed_positive_pairs'] and dict(typecounts)=={k:v for k,v in m['selected_counts'].items() if v},
            'POSITIVE_OUTPUT_COUNTS_MISMATCH '+subject)
    require(count == m['fixed_type1'] + m['fixed_type2_total'] + m['fixed_type3'] and
            m['fixed_type2_total'] == m['selected_counts'][T2C]+m['selected_counts'][T2B],
            'POSITIVE_OUTPUT_SUBTOTAL_MISMATCH '+subject)
    if m['type_balance_status']=='ALL_THREE_TYPES_BALANCED':
        require(m['fixed_type1']==m['fixed_type2_total']==m['fixed_type3'],
                'POSITIVE_TYPE_BALANCE_MISMATCH '+subject)
    else:
        require(m['type_balance_status']=='MISSING_TYPE_1_BALANCED_T2_T3' and
                m['fixed_type1']==0 and m['fixed_type2_total']==m['fixed_type3'],
                'POSITIVE_MISSING_TYPE_BALANCE_INVALID '+subject)
    print(f'[{stamp(args)}] POSITIVE_VERIFY_PASS subject={subject} '
          f'original={census["pairs"]} fixed={count} '
          f'T1={m["fixed_type1"]} T2={m["fixed_type2_total"]} T3={m["fixed_type3"]}',flush=True)
    return m


def classify_and_prepare(args, subject, census, original, ast, parser, h07, h08, runroot):
    original_manifest=original.verify_output(args,subject)
    output_dir=args.output_root/subject
    published=[output_dir/args.positive_public_name,
               output_dir/args.positive_details_dir/args.positive_details_name,
               output_dir/args.positive_details_dir/args.positive_manifest_name]
    existing=[p.exists() or p.is_symlink() for p in published]
    if all(existing):
        print(f'[{stamp(args)}] POSITIVE_ALREADY_PRESENT_VERIFY subject={subject}',flush=True)
        return verify_published(args,subject,census,original)
    require(not any(existing), 'POSITIVE_INCOMPLETE_EXISTING_ARTIFACTS_REVIEW_REQUIRED subject='+subject)
    runsubject=runroot/subject
    runsubject.mkdir(mode=0o755)
    pool=make_pool(runsubject/'positive_pool.sqlite',args)
    db=None
    start=time.monotonic()
    try:
        db,fragcounts,failures,features_sha=ast.build_features(
            args,subject,parser,h07,runsubject,original_manifest)
        callback,observed=consume_pairs(args,subject,pool)
        counts=ast.classify_subject(args,subject,db,h08,original_manifest,runsubject,callback)
        require(counts['counts']==census['counts'] and counts['pairs']==census['pairs'],
                'POSITIVE_AST_V5_CENSUS_MISMATCH subject='+subject+' expected='+
                repr(census['counts'])+' actual='+repr(counts['counts']))
        with (runsubject/'subject_summary.json').open('x',encoding='utf-8') as f:
            json.dump(counts,f,sort_keys=True,indent=2,ensure_ascii=True)
            f.write('\n')
        pool.commit()
        unique=Counter(dict(pool.execute('SELECT category,COUNT(*) FROM candidates GROUP BY category')))
        available={c:unique[c] for c in CATEGORIES}
        selected_counts,q,warnings=quotas_from_counts(available,args,subject)
        selected=choose_pairs(pool,selected_counts)
        print(f'[{stamp(args)}] POSITIVE_POOL_PASS subject={subject} '
              f'raw_candidates={sum(census["counts"][c] for c in CATEGORIES)} '
              f'unique_candidates={sum(available.values())} '
              f'fixed={len(selected)} min_class_quota={q} missing={warnings}',flush=True)
        rows, fragments=load_selected_rows(args,subject,selected,original)
        pub,detail,finalcounts=render_outputs(args,subject,rows,fragments,runsubject)
        require(all(finalcounts[k]==selected_counts[k] for k in CATEGORIES),
                'POSITIVE_FINAL_CATEGORY_COUNT_MISMATCH '+subject)
        source_root=args.output_root/subject
        src=original.read_json(source_root/'manifest.json')
        record={
            'experiment':args.experiment,'version':args.version,'subject':subject,
            'status':'FIXED_POSITIVE_ONLY','generated_at_chicago':stamp(args),
            'effective_configuration':manifest_effective(args),
            'original_pairs':census['pairs'],'census_counts':census['counts'],
            'eligible_candidate_pairs':sum(census['counts'][c] for c in CATEGORIES),
            'unique_content_pair_counts':available,
            'removed_content_duplicate_candidates':sum(census['counts'][c] for c in CATEGORIES)-sum(available.values()),
            'selected_counts':selected_counts,
            'fixed_type1':selected_counts[T1],
            'fixed_type2_total':selected_counts[T2C]+selected_counts[T2B],
            'fixed_type3':selected_counts[T3],
            'fixed_positive_pairs':len(rows),
            'type_balance_status':('ALL_THREE_TYPES_BALANCED' if not warnings else
                                   'MISSING_TYPE_1_BALANCED_T2_T3'),
            'warnings':warnings,
            'fragment_count':fragcounts['fragments'],
            'eligible_fragments':fragcounts['eligible'],
            'parse_failure_reasons':failures,
            'source_manifest_sha256':original.sha_file(source_root/'manifest.json'),
            'source_pairs_sha256':src['files_sha256']['positive_pairs.csv'],
            'source_fragments_sha256':src['files_sha256']['fragments.jsonl'],
            'h07_source_sha256':original.sha_file(args.h07_python),
            'h08_source_sha256':original.sha_file(args.h08_python),
            'files_sha256':{
                args.positive_public_name:original.sha_file(pub),
                args.positive_details_dir+'/'+args.positive_details_name:original.sha_file(detail),
            },
            'duration_seconds':round(time.monotonic()-start,3),
            'stage_fragment_features_sha256':features_sha,
        }
        manifest_stage=runsubject/args.positive_details_dir/args.positive_manifest_name
        with manifest_stage.open('x',encoding='utf-8') as f:
            json.dump(record,f,ensure_ascii=True,sort_keys=True,indent=2)
            f.write('\n')
        dest_details=source_root/args.positive_details_dir
        require(not dest_details.is_symlink(),'POSITIVE_UNSAFE_DETAILS_DIRECTORY '+str(dest_details))
        dest_details.mkdir(exist_ok=True)
        for x,y in ((pub,source_root/args.positive_public_name),
                    (detail,dest_details/args.positive_details_name),
                    (manifest_stage,dest_details/args.positive_manifest_name)):
            compare_or_publish(x,y,original)
        return verify_published(args,subject,census,original)
    finally:
        if db is not None:db.close()
        pool.close()
        for p in (runsubject/'feature_index.sqlite',runsubject/'positive_pool.sqlite'):
            if p.is_file():p.unlink()


def fixed_width_table(head, rows):
    widths=[max(len(str(head[i])), *(len(str(r[i])) for r in rows)) for i in range(len(head))]
    def fmt(row):
        return '  '.join(str(v).ljust(widths[i]) if i==0 else str(v).rjust(widths[i])
                         for i,v in enumerate(row))
    return '\n'.join(['```text',fmt(head),'  '.join('-'*w for w in widths),
                      *(fmt(row) for row in rows),'```'])


def report(args, names, manifests, census, original):
    main=[]; status=[]
    for n in names:
        m=manifests[n]; counts=m['selected_counts'];u=m['unique_content_pair_counts']
        main.append((n,m['original_pairs'],counts[T1],counts[T2C],counts[T2B],
                    m['fixed_type2_total'],counts[T3],m['fixed_positive_pairs']))
        status.append((n,u[T1],u[T2C]+u[T2B],u[T3],
                       m['removed_content_duplicate_candidates'],
                       '3/3' if not m['warnings'] else '2/3 (T1=0)'))
    total=[sum(row[i] for row in main) for i in range(1,8)]
    u_total=[sum(row[i] for row in status) for i in range(1,5)]
    fmt=lambda n:f'{n:,}'
    rows_a=[(r[0],*(fmt(x) for x in r[1:])) for r in main]
    rows_a.append(('TOTAL',*(fmt(x) for x in total)))
    rows_b=[(r[0],*(fmt(x) for x in r[1:5]),r[5]) for r in status]
    complete=sum(not manifests[n]['warnings'] for n in names)
    rows_b.append(('TOTAL',*(fmt(x) for x in u_total),f'{complete}/{len(names)} 3-Type'))
    text='\n'.join([
        '# J-03d v6 — Subject-wise Fixed Positive Clone Pairs',
        '',f'Generated (America/Chicago): {stamp(args)}',
        f'Reference census: `{args.positive_source_census_name}` (SHA256 `{args.positive_source_census_sha256}`)',
        f'Subjects: **{len(names)}**; original NiCad positives: **{fmt(total[0])}**; '
        f'fixed positive pairs: **{fmt(total[-1])}**.',
        '', '## Selection policy', '',
        '- Type 1, Type 2 (Consistent+Blind), Type 3 candidates only; no Type 4 or Undetermined.',
        '- De-duplicate unordered exact **source code content** pairs before sampling.',
        '- Equal Type 1:Type 2:Type 3 per subject where all three types exist.',
        '- For missing Type 1, balance available Type 2:Type 3 1:1 and flag the limitation.',
        '- Deterministic seed and hash-ranked selection; retain Type 2 subtypes.',
        '- These are NiCad-positive structural clone candidates, not semantic-certified labels.',
        '- This stage creates NO negatives and NO train/valid/test splits.',
        '', '## A. Fixed structural positive clone pairs', '',
        'T2-C = Type 2 Consistent; T2-B = Type 2 Blind; T2-All = Type 2 total (subtotal).',
        '',fixed_width_table(('Subject','Orig','T1','T2-C','T2-B','T2-All','T3','N-pos'),rows_a),
        '', '## B. Unique candidate pool and missing-type status', '',
        'Unique-T1/T2/T3 are counts **after** duplicate source-code pair removal; DupRm is removed candidate pairs.',
        '',fixed_width_table(('Subject','Unique-T1','Unique-T2','Unique-T3','DupRm','Type status'),rows_b),
        '', '## Verification / next phases', '',
        '- Original CSV/JSONL and all previous manifests and clone-type reports remain immutable.',
        '- Per-subject public JSONL, detailed JSONL, and immutable manifest are verified by SHA256.',
        '- Subject-wise N-pos is fixed only for **this v6 balanced positive selection**.',
        '- Future negative generation must produce N-neg = N-pos for every subject without label overlap.',
        '- Future split design must prevent fragment and connected-component leakage; NOT completed here.',
        '- Public redistribution of source code requires reviewing upstream software licenses.',
        '' ])
    return text


def run(args, names, original):
    require(len(names)==args.expected_subjects,'POSITIVE_SUBJECT_COUNT_MISMATCH')
    census=load_census(args,names,original)
    runroot=args.run_dir/'fixed_positive'
    runroot.mkdir(mode=0o755)
    results={}
    if args.mode=='positiveclone':
        # Type classification is recomputed from immutable fragments, not inferred
        # from NiCad similarity and not silently treated as certified ground truth.
        ast_spec=importlib.util.spec_from_file_location('j03d_ast_for_positive',args.ast_impl)
        require(ast_spec is not None and ast_spec.loader is not None,'POSITIVE_AST_IMPL_MISSING')
        ast=importlib.util.module_from_spec(ast_spec)
        ast_spec.loader.exec_module(ast)
        parser,h07,h08,evidence=ast.dependency_preflight(args)
        for idx,name in enumerate(names,1):
            print(f'[{stamp(args)}] POSITIVE_SUBJECT_START index={idx}/{len(names)} subject={name}',flush=True)
            results[name]=classify_and_prepare(args,name,census[name],original,ast,parser,h07,h08,runroot)
    else:
        for idx,name in enumerate(names,1):
            original.verify_output(args,name)
            print(f'[{stamp(args)}] POSITIVE_VERIFY_START index={idx}/{len(names)} subject={name}',flush=True)
            results[name]=verify_published(args,name,census[name],original)
    require(sum(r['fixed_positive_pairs'] for r in results.values())>0,'POSITIVE_ALL_EMPTY')
    md=report(args,names,results,census,original)
    perrun=args.run_dir/args.positive_report_name
    with perrun.open('x',encoding='utf-8') as f:f.write(md)
    summary={'experiment':args.experiment,'version':args.version,'mode':args.mode,
             'status':'PASS_FIXED_POSITIVE_ONLY',
             'generated_at_chicago':stamp(args),
             'census_sha256':args.positive_source_census_sha256,
             'subject_count':len(names), 'original_positive_pairs':census['TOTAL']['pairs'],
             'fixed_positive_pairs':sum(r['fixed_positive_pairs'] for r in results.values()),
             'type1':sum(r['fixed_type1'] for r in results.values()),
             'type2':sum(r['fixed_type2_total'] for r in results.values()),
             'type3':sum(r['fixed_type3'] for r in results.values()),
             'effective_configuration':manifest_effective(args),
             'report_sha256':original.sha_file(perrun),
             'per_subject':[{'subject':n,'fixed_positive_pairs':results[n]['fixed_positive_pairs'],
                              'type_balance_status':results[n]['type_balance_status'],
                              'manifest_sha256':original.sha_file(args.output_root/n/args.positive_details_dir/args.positive_manifest_name)}
                             for n in names]}
    final=args.run_dir/'fixed_positive_summary.json'
    with final.open('x',encoding='utf-8') as f:
        json.dump(summary,f,indent=2,sort_keys=True,ensure_ascii=True); f.write('\n')
    canonical=args.output_root/args.positive_report_name
    if args.mode=='positiveclone':
        # The timestamp-free normalized content can be reproduced and reused.
        if canonical.exists():
            require(not canonical.is_symlink(),'POSITIVE_REPORT_SYMLINK')
            old=canonical.read_text(encoding='utf-8')
            prefix='Generated (America/Chicago): '
            oldtime=next((l for l in old.splitlines() if l.startswith(prefix)),None)
            newtime=next((l for l in md.splitlines() if l.startswith(prefix)),None)
            require(oldtime is not None and newtime is not None and
                    old==md.replace(newtime,oldtime,1),
                    'POSITIVE_CANONICAL_REPORT_DIFFERENT_PRESERVED')
            print('POSITIVE_REPORT_REUSED_VERIFIED '+str(canonical),flush=True)
        else:
            os.link(perrun,canonical)
            print('POSITIVE_REPORT_CREATED '+str(canonical),flush=True)
    print('POSITIVE_TERMINAL_REPORT_START\n'+md,flush=True)
    print(f'POSITIVE_FINAL_PASS subjects={len(names)} original_pairs={summary["original_positive_pairs"]} '
          f'fixed_pairs={summary["fixed_positive_pairs"]} status={summary["status"]} '
          f'report={perrun}',flush=True)
