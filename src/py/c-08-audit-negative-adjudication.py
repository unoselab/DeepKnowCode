#!/usr/bin/env python3
"""C-08: strict human-adjudication audit for C-06 risk-stratified negative pairs."""
import argparse
import collections
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

EXPECTED_FIELDS = ('review_id','project','split','id_a','id_b','review_group')
LABELS = {'CONFIRMED_CLONE','CONFIRMED_NONCLONE','UNCERTAIN'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f'Empty CSV: {path}')
        names=[n.strip().strip('*') for n in reader.fieldnames]
        if len(names)!=len(set(names)):
            raise ValueError(f'Duplicate columns in {path}')
        reader.fieldnames=names
        return list(reader),set(names)


def summarize(review_csv, queue_csv, blank_means_nonclone=True):
    references, refcols=read_csv(queue_csv)
    reviews, columns=read_csv(review_csv)
    for field in EXPECTED_FIELDS:
        if field not in columns or field not in refcols:
            raise ValueError(f'Missing required {field} field')
    if 'adjudication' not in columns:
        raise ValueError('Missing adjudication column')
    lookup={}
    for ref in references:
        key=ref['review_id'].strip()
        if not key or key in lookup: raise ValueError(f'Invalid or duplicate queue review_id: {key!r}')
        lookup[key]=ref
    found={}
    errors=[]
    for index,row in enumerate(reviews, start=2):
        key=(row.get('review_id') or '').strip()
        if not key or key in found:
            errors.append(f'Row {index}: missing/duplicate review_id {key!r}')
            continue
        if key not in lookup:
            errors.append(f'Row {index}: unrecognized review_id {key}')
            continue
        bad=[field for field in EXPECTED_FIELDS if (row.get(field) or '').strip() != (lookup[key].get(field) or '').strip()]
        if bad:
            errors.append(f'Row {index}: queue mismatch for {key}: {bad}')
        found[key]=row
    missing=sorted(set(lookup)-set(found))
    if missing:
        errors.append(f'Missing {len(missing)} queue entries; examples: {missing[:5]}')
    records=[]
    label_counts=collections.Counter()
    group_counts=collections.defaultdict(collections.Counter)
    project_counts=collections.defaultdict(collections.Counter)
    blank_count=0; metadata_missing=collections.Counter()
    for key in sorted(set(found)&set(lookup)):
        row=found[key]
        raw=(row.get('adjudication') or '').strip().upper()
        if not raw:
            blank_count+=1
            label='CONFIRMED_NONCLONE' if blank_means_nonclone else 'UNREVIEWED'
        elif raw in LABELS:
            label=raw
        else:
            errors.append(f'Invalid adjudication {raw!r} for {key}')
            label='INVALID'
        item={field:(lookup[key].get(field) or '').strip() for field in EXPECTED_FIELDS}
        item.update(adjudication=label,original_adjudication=raw,reviewer=(row.get('reviewer') or '').strip(),confidence=(row.get('confidence') or '').strip(),review_notes=(row.get('review_notes') or '').strip(),reviewed_at_chicago=(row.get('reviewed_at_chicago') or '').strip(),label_origin='EXPLICIT_CSV' if raw else 'BLANK_AS_NONCLONE_USER_POLICY')
        for field in ('reviewer','confidence','reviewed_at_chicago'):
            if not item[field]:metadata_missing[field]+=1
        records.append(item)
        label_counts[label]+=1
        group_counts[item['review_group']][label]+=1
        project_counts[item['project']][label]+=1
    if len(reviews)!=len(references):
        errors.append(f'CSV row count {len(reviews)} does not equal queue size {len(references)}')
    # Do not interpret absence of metadata as proof that a blank label was manually inspected.
    warnings=[]
    if blank_count:
        warnings.append(f'{blank_count} blank adjudications interpreted as CONFIRMED_NONCLONE by USER-SPECIFIED policy; CSV itself cannot demonstrate that each was inspected.')
    if metadata_missing['reviewer']==len(records):
        warnings.append('No reviewer identifiers: manual review provenance cannot be independently confirmed.')
    if metadata_missing['reviewed_at_chicago']==len(records):
        warnings.append('No review timestamps: review chronology cannot be verified from CSV.')
    if metadata_missing['confidence']==len(records):
        warnings.append('No confidence values recorded.')
    warnings.append('Sample is risk-stratified, not a simple random sample: raw error fraction is not a population prevalence estimate.')
    warnings.append('Labels for 274 reviewed pairs cannot be automatically extended to 13,322 unreviewed negative candidates.')
    status='AUDIT_PASS_REVIEWED_SUBSET_ONLY' if not errors else 'AUDIT_FAILED'
    summary=dict(version='v1',status=status,review_rows=len(reviews),queue_rows=len(references),matched_rows=len(records),blank_interpretation='CONFIRMED_NONCLONE' if blank_means_nonclone else 'UNREVIEWED',blank_rows=blank_count,labels=dict(sorted(label_counts.items())),metadata_missing=dict(sorted(metadata_missing.items())),group_labels={k:dict(sorted(v.items())) for k,v in sorted(group_counts.items())},project_labels={k:dict(sorted(v.items())) for k,v in sorted(project_counts.items())},errors=errors,warnings=warnings,training_approval=False,negative_population_verified=False)
    return summary, records


def save_csv(path, records, fields):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(records)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--review-csv',type=Path,required=True)
    ap.add_argument('--c06',type=Path,required=True)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--blank-is-unreviewed',action='store_true',help='Override user-provided blank=NONCLONE rule')
    args=ap.parse_args()
    review=args.review_csv.expanduser().resolve()
    queue=(args.c06.expanduser().resolve()/'review_queue.csv')
    if not review.is_file() or not queue.is_file():
        ap.error('Review CSV or C-06 review_queue.csv does not exist')
    summary,rows=summarize(review,queue,not args.blank_is_unreviewed)
    print('C-08',json.dumps({k:summary[k] for k in ('status','review_rows','matched_rows','blank_rows','labels','training_approval')},ensure_ascii=False,sort_keys=True))
    if args.dry_run:
        print('DRY RUN: no output created')
        if summary['errors']:
            for err in summary['errors'][:15]:print('ERROR:',err)
        return 0 if not summary['errors'] else 2
    if summary['errors']:
        for err in summary['errors'][:15]:print('ERROR:',err)
        print('ERROR: refusing to write labels due to failed audit')
        return 2
    chicago=datetime.now(ZoneInfo('America/Chicago'))
    default=Path.cwd()/'results'/'c-08'/chicago.strftime('%Y%m%d-%H%M%S-%f')
    output=(args.output or default).expanduser().resolve()
    if output.exists():
        ap.error(f'Refusing overwrite: {output}')
    output.mkdir(parents=True)
    fields=list(EXPECTED_FIELDS)+['adjudication','original_adjudication','label_origin','reviewer','confidence','review_notes','reviewed_at_chicago']
    files={
        'adjudicated_review_subset.csv':rows,
        'confirmed_clone_in_review.csv':[r for r in rows if r['adjudication']=='CONFIRMED_CLONE'],
        'confirmed_nonclone_in_review.csv':[r for r in rows if r['adjudication']=='CONFIRMED_NONCLONE'],
        'uncertain_in_review.csv':[r for r in rows if r['adjudication'] in ('UNCERTAIN','UNREVIEWED')]
    }
    for name,data in files.items():save_csv(output/name,data,fields)
    summary['created_at_chicago']=chicago.isoformat()
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    manifest=dict(version='v1',created_at_chicago=chicago.isoformat(),input_review_csv=str(review),input_review_sha256=digest(review),input_c06_queue_csv=str(queue),input_c06_queue_sha256=digest(queue),output_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},blank_adjudication_policy=summary['blank_interpretation'],training_approval=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('OUTPUT:',output)
    print('NOTE: do not promote remaining unreviewed negative labels or begin model training based only on this audit.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
