#!/usr/bin/env python3
"""D-05bb v9: C-01-only heldout filtering from verified D-05ba v6 audit.

Do not read legacy directories or emit legacy candidates. C-01 detector outputs are
annotation candidates, never gold labels. No clone detection or model inference.
"""
import argparse
import collections
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v9'
STATUS = 'D05BB_V9_C01_ONLY_HELDOUT_CANDIDATES_GOLD_NOT_VERIFIED'
NO_MATCH = 'NO_MATCH_AGAINST_M2_M3_ACTIVE_PYTHON_TRAIN_VALID_HASHES'
OVERLAP = 'EXACT_OR_NORMALIZED_ACTIVE_TRAINING_CODE_HASH_OVERLAP'
REQUIRED_FILES = ('candidates_source_audited.jsonl','source_leakage_audit.json','annotation_queue.csv')


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for part in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_jsonl(path):
    with path.open('r',encoding='utf-8') as f:
        for line_no,line in enumerate(f,1):
            if line.strip():
                item=json.loads(line)
                if not isinstance(item,dict):
                    raise ValueError(f'Expected JSON object: {path}:{line_no}')
                yield item


def validate_input(source):
    if not source.is_dir(): raise ValueError(f'Input v6 directory not found: {source}')
    mp=source/'manifest.json'
    if not mp.is_file():raise ValueError('Missing v6 manifest.json')
    manifest=json.loads(mp.read_text(encoding='utf-8'))
    if manifest.get('version')!='v6' or manifest.get('status')!='D05BA_V6_ACTIVE_TRAINING_HASH_AUDIT_GOLD_NOT_VERIFIED':
        raise ValueError('Input is not a recognized D-05ba v6 output')
    checks=manifest.get('files',{})
    for name in REQUIRED_FILES:
        f=source/name
        if not f.is_file():raise ValueError('Missing v6 artifact: '+str(f))
        expected=checks.get(name,{})
        if expected.get('sha256')!=sha256_file(f) or expected.get('bytes')!=f.stat().st_size:
            raise ValueError('Input SHA256/size mismatch: '+name)
    audit=json.loads((source/'source_leakage_audit.json').read_text(encoding='utf-8'))
    if audit.get('version')!='v6' or audit.get('training_reference_complete') is not True:
        raise ValueError('Training function mapping not complete; no filtration authorized')
    ta=audit.get('active_training_audit',{})
    if ta.get('active_only_legacy_training_excluded') is not True:
        raise ValueError('Active-only training data provenance not established')
    if len(audit.get('active_repositories',[]))!=24 or 'scikit-learn' not in audit['active_repositories'] or 'azure-sdk-for-python' in audit['active_repositories']:
        raise ValueError('24-project active roster mismatch')
    if audit.get('counts',{}).get('pairs_both_code_resolved')!=14406:
        raise ValueError('Expected fully source-resolved v6 candidate set of 14406')
    return manifest,audit


def classify(o):
    if o.get('gold_label') is not None or o.get('evaluation_eligible') is not False:
        raise ValueError('Unexpected gold/evaluation label in v6 input')
    sides=o.get('v6_source_resolution',{})
    if not all(isinstance(sides.get(side),dict) and sides[side].get('code_sha256') and sides[side].get('normalized_code_sha256') for side in ('left','right')):
        raise ValueError('Unresolved source: '+str(o.get('pair_id')))
    matches=o.get('v6_code_leakage_matches_by_split',{})
    active=False
    for split in ('train','validation'):
        m=matches.get(split)
        if not isinstance(m,dict):raise ValueError('Missing split leakage evidence: '+str(o.get('pair_id')))
        for k in ('exact_sides','normalized_sides'):
            if not isinstance(m.get(k),list) or any(x not in ('left','right') for x in m[k]):
                raise ValueError('Invalid split leakage sides: '+str(o.get('pair_id')))
        if not isinstance(m.get('exact_code_pair_overlap'),bool):
            raise ValueError('Invalid pair overlap flag: '+str(o.get('pair_id')))
        active=active or bool(m['exact_sides'] or m['normalized_sides'] or m['exact_code_pair_overlap'])
    status=o.get('code_hash_train_overlap_status')
    if status not in (NO_MATCH,OVERLAP) or (status==OVERLAP)!=active:
        raise ValueError('Inconsistent leakage status: '+str(o.get('pair_id')))
    # Independent re-computation: within test, un-ordered exact code-hash pair, across all projects.
    key=tuple(sorted((sides['left']['code_sha256'],sides['right']['code_sha256'])))
    pair_hash=hashlib.sha256('\0'.join(key).encode('ascii')).hexdigest()
    if o.get('v6_code_pair_sha256')!=pair_hash:
        raise ValueError('Inconsistent v6 pair hash: '+str(o.get('pair_id')))
    return ('quarantine' if active else 'candidate'),pair_hash


def run(args):
    source=args.source.resolve();dest=args.output.resolve()
    if source==dest or dest in source.parents or source in dest.parents:
        raise ValueError('Input/output directory nesting prohibited')
    if dest.exists():raise ValueError('Output exists; refusing overwrite: '+str(dest))
    _,input_audit=validate_input(source)
    seen_ids=set(); buckets=collections.defaultdict(list); quarantined=[]
    excluded_legacy=collections.Counter()
    stats=collections.Counter();projects=collections.defaultdict(collections.Counter)
    for o in read_jsonl(source/'candidates_source_audited.jsonl'):
        pid=o.get('pair_id');proj=o.get('project')
        if not isinstance(pid,str) or not pid or pid in seen_ids:raise ValueError('Duplicate/invalid pair_id: '+str(pid))
        if proj not in input_audit['active_repositories']:raise ValueError('Unexpected project: '+str(proj))
        seen_ids.add(pid)
        # STRICT PROVENANCE GATE: only C-01 detector results may reach outputs.
        # Legacy CSV records are counted, not filtered, annotated, or exported.
        origin=o.get('origin')
        if origin!='c01_hybrid_detector':
            if origin!='legacy_prediction_csv':
                raise ValueError('Unknown, non-C-01 origin: '+str(origin))
            excluded_legacy[proj]+=1
            stats['excluded_legacy_prediction_records']+=1
            continue
        if o.get('legacy_prediction') is not None:
            raise ValueError('C-01 record contaminated with legacy prediction: '+str(pid))
        if not isinstance(o.get('source_reference'),dict):
            raise ValueError('C-01 record missing source_reference: '+str(pid))
        ref_file=str(o['source_reference'].get('file',''))
        if not (ref_file.startswith('results/c-01/') and ref_file.endswith('/hybrid/hybrid_function_clone_pairs.csv')):
            raise ValueError('C-01 record has unexpected source: '+ref_file)
        category,pair_hash=classify(o)
        stats['input_records']+=1;projects[proj]['input_records']+=1
        if category=='quarantine':
            quarantined.append(o);stats['quarantined_train_or_validation_overlap']+=1
            projects[proj]['quarantined_train_or_validation_overlap']+=1
        else:
            buckets[pair_hash].append(o);stats['no_hash_overlap_records']+=1
            projects[proj]['no_hash_overlap_records']+=1
    expected=input_audit.get('counts',{})
    if stats['input_records']+stats['excluded_legacy_prediction_records']!=expected.get('pairs_both_code_resolved'):
        raise ValueError('v6 source audit count mismatch')
    if stats['excluded_legacy_prediction_records']!=458:
        raise ValueError('Unexpected legacy record count in pinned v6: '+str(stats['excluded_legacy_prediction_records']))
    if stats['input_records']!=13948:
        raise ValueError('Expected 13,948 C-01 records; found '+str(stats['input_records']))
    if stats['quarantined_train_or_validation_overlap']!=9900:
        raise ValueError('Expected 9,900 C-01 overlaps; found '+str(stats['quarantined_train_or_validation_overlap']))
    if stats['no_hash_overlap_records']!=4048:
        raise ValueError('Expected 4,048 C-01 non-overlap records; found '+str(stats['no_hash_overlap_records']))
    retained=[];duplicate_rows=[]
    for pair_hash,records in sorted(buckets.items()):
        # C-01 only: stable deterministic choice by project and pair ID.
        records.sort(key=lambda o:(o['project'],o['pair_id']))
        chosen=records[0]
        out=dict(chosen)
        out['v9_selection']='C01_ONLY_HELDOUT_HASH_FILTERED_ANNOTATION_PENDING'
        out['v9_global_exact_pair_sha256']=pair_hash
        out['v9_redundant_records']=len(records)-1
        out['gold_label']=None;out['evaluation_eligible']=False
        out['annotation_status']='pending'
        out['near_duplicate_leakage_status']='NOT_CHECKED'
        retained.append(out)
        projects[out['project']]['unique_retained_pairs']+=1
        if len(records)>1:
            stats['duplicate_groups_in_no_overlap']+=1
            for other in records[1:]:
                duplicate_rows.append({'pair_id':other['pair_id'],'retained_pair_id':chosen['pair_id'],
                                       'project':other['project'],'retained_project':chosen['project'],
                                       'v9_global_exact_pair_sha256':pair_hash,'origin':other['origin']})
                projects[other['project']]['duplicate_records_removed']+=1
    stats['exact_pair_duplicate_records_removed']=len(duplicate_rows)
    stats['heldout_candidates_after_dedup']=len(retained)
    stats['verified_gold_labels']=0
    if stats['heldout_candidates_after_dedup']!=3971:
        raise ValueError('Expected 3,971 C-01-only heldout records; found '+str(stats['heldout_candidates_after_dedup']))
    if stats['exact_pair_duplicate_records_removed']!=77:
        raise ValueError('Expected 77 C-01-only duplicate records; found '+str(stats['exact_pair_duplicate_records_removed']))
    if any(o.get('origin')!='c01_hybrid_detector' for o in retained+quarantined):
        raise ValueError('Legacy provenance escaped C-01-only selection')
    if stats['input_records']!=stats['quarantined_train_or_validation_overlap']+stats['heldout_candidates_after_dedup']+stats['exact_pair_duplicate_records_removed']:
        raise ValueError('Accounting invariant failed')
    if set(projects)!=set(input_audit['active_repositories']):raise ValueError('Project coverage changed')
    if stats['input_records']+stats['excluded_legacy_prediction_records']!=14406:
        raise ValueError('Total audit accounting mismatch')
    audit={'version':VERSION,'status':STATUS,'created_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
           'source_v6_directory':str(source),'source_v6_manifest_sha256':sha256_file(source/'manifest.json'),
           'source_v6_candidates_sha256':sha256_file(source/'candidates_source_audited.jsonl'),
           'training_reference_complete':True,'active_training_only':True,'legacy_training_used':False,
           'c01_clone_detection_reexecuted':False,'active_projects':sorted(projects),
           'candidate_source_policy':'C01_ONLY; no source files under ground_truth/python_legacy or archived prediction CSV are read',
           'excluded_legacy_by_project':dict(sorted(excluded_legacy.items())),
           'c01_only_output_asserted':True,
           'counts':dict(stats),'per_project':{p:dict(c) for p,c in sorted(projects.items())},
           'dedup_key':'unordered pair of exact source-code SHA256 hashes; global across projects',
           'deterministic_preference':'Only C-01 records; then lexical project and pair_id',
           'heldout_definition':'C-01-only; neither side matches exact/whitespace-normalized code SHA256 in active M2/M3 Python training OR validation',
           'limitations':['These are heldout candidate pairs only, not independently gold-labeled test examples.',
                          'Near-duplicate leakage NOT CHECKED; semantic/AST similarity leakage NOT CHECKED.',
                          'M1 historical training exposure separately NOT ESTABLISHED; M2/M3 active Python train and validation only.',
                          'M0-M3 heldout eligibility is not yet certified; near-duplicate and label audits are pending.',
                          'Legacy prediction records are excluded from ALL emitted candidate and quarantine files; legacy directories never read.',
                          'C-01 detection positives are unverified; negatives must be independently constructed and annotated.',
                          'Exact code hash identity does not establish semantic clone truth.']}
    result={'status':'DRY_RUN' if args.dry_run else STATUS,'counts':dict(stats),
            'active_projects':len(projects),'training_reference_complete':True,'source_policy':'C01_ONLY'}
    if args.dry_run:
        print(json.dumps(result,sort_keys=True));return
    dest.parent.mkdir(parents=True,exist_ok=True);dest.mkdir(exist_ok=False)
    def dump_jsonl(name,items):
        with (dest/name).open('x',encoding='utf-8') as f:
            for obj in items:f.write(json.dumps(obj,sort_keys=True,ensure_ascii=False)+'\n')
    dump_jsonl('heldout_candidates.jsonl',retained)
    dump_jsonl('quarantined_train_validation_overlap.jsonl',quarantined)
    with (dest/'exact_pair_duplicates.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['pair_id','retained_pair_id','project','retained_project','v9_global_exact_pair_sha256','origin']);writer.writeheader();writer.writerows(duplicate_rows)
    with (dest/'annotation_queue.csv').open('x',newline='',encoding='utf-8') as f:
        fields=['pair_id','project','origin','left_repository_file','right_repository_file','left_start_line','left_end_line','right_start_line','right_end_line','training_code_overlap','near_duplicate_status','gold_label','annotation_status']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for o in retained:
            s=o['v6_source_resolution']
            w.writerow({'pair_id':o['pair_id'],'project':o['project'],'origin':o['origin'],
                        'left_repository_file':s['left'].get('repository_file',''),'right_repository_file':s['right'].get('repository_file',''),
                        'left_start_line':s['left'].get('start_line',''),'left_end_line':s['left'].get('end_line',''),
                        'right_start_line':s['right'].get('start_line',''),'right_end_line':s['right'].get('end_line',''),
                        'training_code_overlap':'NO_EXACT_OR_NORMALIZED_MATCH','near_duplicate_status':'NOT_CHECKED',
                        'gold_label':'','annotation_status':'pending'})
    with (dest/'filter_audit.json').open('x',encoding='utf-8') as f:json.dump(audit,f,indent=2,sort_keys=True);f.write('\n')
    manifest={'version':VERSION,'status':STATUS,'created_at_chicago':audit['created_at_chicago'],
              'files':{p.name:{'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in sorted(dest.iterdir()) if p.is_file()}}
    with (dest/'manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2,sort_keys=True);f.write('\n')
    result['out']=str(dest)
    print(json.dumps(result,sort_keys=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--source',type=Path,help='Explicit D-05ba v6 result directory')
    parser.add_argument('--output',type=Path,help='New v9 result directory; no overwrites')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    root=args.root.resolve()
    if args.source is None:
        candidates=sorted((root/'results/d-05b').glob('python-source-leakage-audit-v6-*/manifest.json'))
        if len(candidates)!=1:
            parser.error('Specify --source: expected exactly one v6 result, found '+str(len(candidates)))
        args.source=candidates[0].parent
    if args.output is None:
        stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
        args.output=root/'results/d-05b'/('python-heldout-c01-only-v9-'+stamp)
    try:run(args)
    except (ValueError,OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
        print('D-05bb v9 ERROR: '+str(exc),file=sys.stderr);return 2
    return 0

if __name__=='__main__':sys.exit(main())
