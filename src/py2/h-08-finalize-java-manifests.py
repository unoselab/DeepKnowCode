#!/usr/bin/env python3
"""Read-only preflight and atomic manifest update for H-08 curated Java dataset."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def load_json_with_trailing_comma_fix(path):
    raw=path.read_text(encoding='utf-8')
    try:return json.loads(raw),False
    except json.JSONDecodeError:
        chars=[];in_string=False;escaped=False;removed=0
        for i,c in enumerate(raw):
            if in_string:
                chars.append(c)
                if escaped:escaped=False
                elif c=='\\':escaped=True
                elif c=='"':in_string=False
            else:
                if c=='"':in_string=True;chars.append(c);continue
                if c==',':
                    j=i+1
                    while j<len(raw) and raw[j].isspace():j+=1
                    if j<len(raw) and raw[j] in '}]':removed+=1;continue
                chars.append(c)
        if not removed:raise
        return json.loads(''.join(chars)),True


def json_bytes(obj):return (json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode('utf-8')


def ensure(cond,msg):
    if not cond:raise ValueError(msg)


def compare(a,b,label):ensure(a==b,f'{label}: expected {b!r}, found {a!r}')


def audit(args):
    root=args.root.resolve();base=root/args.dataset_dir
    compact=root/args.compact_manifest;detail=root/args.detailed_manifest
    code=root/args.code_map
    for p in [base,compact,detail,code]:ensure(p.exists(),f'missing input: {p}')
    compare(sha(code),args.expected_code_sha256,'CODE_MAP_SHA256')
    c,c_fixed=load_json_with_trailing_comma_fix(compact)
    d,d_fixed=load_json_with_trailing_comma_fix(detail)
    cj=c['languages']['java'];dj=d['languages']['java']
    existing_hashes=dj['csv_file_sha256']
    compare(len(existing_hashes),args.expected_csv_files,'MANIFEST_CSV_COUNT')
    paths=sorted(base.glob('*.csv'))
    compare(len(paths),args.expected_csv_files,'DATASET_CSV_COUNT')
    compare({p.name for p in paths},set(existing_hashes),'DATASET_FILENAMES')
    compare(sha(base/args.target_csv),args.expected_target_sha256,'TARGET_CSV_SHA256')
    compare(dj['code_map_sha256'],args.expected_code_sha256,'MANIFEST_CODE_SHA256')
    before_target=existing_hashes[args.target_csv]
    changed={p.name:(sha(p),existing_hashes[p.name]) for p in paths if sha(p)!=existing_hashes[p.name]}
    compare(set(changed),{args.target_csv},'EXACT_CHANGED_CSV_SET')
    compare(changed[args.target_csv][0],args.expected_target_sha256,'EXPECTED_UPDATED_TARGET')
    label_counts=Counter();ids=set();pairs=set();ids_by_code={};conflicts=defaultdict(set);target_found=[];old_found=[];per_file={}
    required={'pair_left_func','pair_right_func','clone_predict'}
    for p in paths:
        n=0
        with p.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            ensure(reader.fieldnames is not None and required<=set(reader.fieldnames),f'bad schema: {p}')
            for rownum,row in enumerate(reader,start=2):
                n+=1
                a=row['pair_left_func'].strip();b=row['pair_right_func'].strip();label=row['clone_predict'].strip()
                ensure(a and b and a!=b,f'bad pair at {p.name}:{rownum}')
                ensure(label in {'0','1'},f'bad label at {p.name}:{rownum}')
                # Existing manifest's label rule: compare second underscore-delimited token.
                aa=a.split('_');bb=b.split('_')
                ensure(len(aa)>=3 and len(bb)>=3,f'invalid ID at {p.name}:{rownum}')
                expected='1' if aa[1]==bb[1] else '0'
                ensure(label==expected,f'ID-derived label mismatch at {p.name}:{rownum}')
                key=tuple(sorted((a,b)))
                ensure(key not in pairs,f'duplicate unordered ID pair: {key}')
                pairs.add(key);ids.update((a,b));label_counts[label]+=1
                conflicts[key].add(label)
                if key==tuple(sorted((args.new_left,args.new_right))):target_found.append((p.name,rownum,row))
                if key==tuple(sorted((args.old_left,args.old_right))):old_found.append((p.name,rownum,row))
        per_file[p.name]=n
    compare(sum(label_counts.values()),args.expected_pairs,'PAIR_COUNT')
    compare(label_counts['1'],args.expected_positive,'POSITIVE_COUNT')
    compare(label_counts['0'],args.expected_negative,'NEGATIVE_COUNT')
    compare(len(target_found),1,'NEW_PAIR_OCCURRENCES')
    compare(target_found[0][0],args.target_csv,'NEW_PAIR_SOURCE_CSV')
    compare(target_found[0][2]['clone_predict'],'0','NEW_PAIR_LABEL')
    compare(len(old_found),0,'OLD_PAIR_OCCURRENCES')
    compare(per_file[args.target_csv],args.expected_target_rows,'TARGET_ROWS')
    compare(sum(1 for v in conflicts.values() if len(v)>1),0,'CONFLICTING_ID_PAIR_LABELS')
    code_ids=set();code_rows=0
    with code.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            obj=json.loads(line);code_rows+=1
            ensure(obj.get('id') and obj.get('code') is not None,'code map id/code invalid')
            ensure(obj['id'] not in code_ids,'duplicate code id')
            code_ids.add(obj['id'])
    compare(code_rows,args.expected_code_rows,'CODE_ROWS')
    ensure(ids<=code_ids,f'unknown code IDs: {sorted(ids-code_ids)[:10]}')
    compare(len(ids),args.expected_referenced_code_ids,'REFERENCED_CODE_IDS')
    for field,value in [('projects',args.expected_csv_files),('pairs',args.expected_pairs),('positive_pairs',args.expected_positive),('negative_pairs',args.expected_negative),('code_ids',args.expected_code_rows),('referenced_code_ids',args.expected_referenced_code_ids)]:
        for name,item in [('compact',cj),('detailed',dj)]:compare(item[field],value,f'{name}.{field}')
    ensure(not dj.get('human_verified_gold',True),'unexpected human_verified_gold true')
    report={'status':'H08_MANIFEST_PREFLIGHT_PASS','version':args.version,'timestamp_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),'mode':args.mode,'counts':{'csv_files':len(paths),'pairs':len(pairs),'positive':label_counts['1'],'negative':label_counts['0'],'code_rows':code_rows,'referenced_code_ids':len(ids),'target_csv_rows':per_file[args.target_csv]},'target_csv':args.target_csv,'previous_csv_sha256':before_target,'updated_csv_sha256':args.expected_target_sha256,'changed_files':sorted(changed),'removed_pair':[args.old_left,args.old_right],'added_negative_pair':[args.new_left,args.new_right],'compact_json_trailing_comma_fixed':c_fixed,'detailed_json_trailing_comma_fixed':d_fixed,'prior_h02_h04_h07_h08_results':'HISTORICAL_SNAPSHOT_NOT_REVALIDATED','fresh_h04_h08_status':'REQUIRES_RERUN'}
    dj['csv_file_sha256'][args.target_csv]=args.expected_target_sha256
    curation={'timestamp_chicago':report['timestamp_chicago'],'event':'H08_MANUAL_NEGATIVE_REPLACEMENT','removed_pair':[args.old_left,args.old_right],'added_pair':[args.new_left,args.new_right],'label':'0','target_csv':args.target_csv,'new_csv_sha256':args.expected_target_sha256,'prior_csv_sha256':before_target,'manual_review_status':'ACCEPT_NEGATIVE','h04_h08_reevaluation':'PENDING'}
    dj.setdefault('manual_curation_history',[]).append(curation)
    dj['current_snapshot_h04_audit_status']='REVALIDATION_REQUIRED'
    dj['current_snapshot_h08_analysis_status']='REVALIDATION_REQUIRED'
    # Keep prior audit status and references intact, they describe older dataset snapshot.
    return c,d,report


def write_exclusive(p,contents):
    fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as f:f.write(contents);f.flush();os.fsync(f.fileno())


def apply(args,c,d,report):
    root=args.root.resolve();compact=root/args.compact_manifest;detail=root/args.detailed_manifest
    archive=root/args.backup_dir;results=root/args.result_dir
    archive.mkdir(parents=True,exist_ok=True);results.mkdir(parents=True,exist_ok=True)
    tag=args.run_id
    backups=[archive/(p.name+'.before-'+tag) for p in (compact,detail)]
    for b in backups:ensure(not b.exists(),f'backup already exists: {b}')
    for p,b in zip((compact,detail),backups):write_exclusive(b,p.read_bytes())
    target_contents=[json_bytes(c),json_bytes(d)]
    temps=[];replaced=[]
    try:
        for p,data in zip((compact,detail),target_contents):
            fd,temp=tempfile.mkstemp(prefix='.'+p.name+'.',dir=p.parent)
            temps.append(Path(temp))
            with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
            os.chmod(temp,p.stat().st_mode & 0o777)
        for p,temp in zip((compact,detail),temps):os.replace(temp,p);replaced.append(p)
        for p,expected in zip((compact,detail),target_contents):ensure(p.read_bytes()==expected,f'post-write mismatch: {p}')
    except BaseException:
        for p,b in zip((compact,detail),backups):
            if p in replaced:shutil.copy2(b,p)
        raise
    finally:
        for temp in temps:
            if temp.exists():temp.unlink()
    report['status']='H08_MANIFEST_APPLY_PASS';report['backups']=[str(x) for x in backups]
    report['updated_manifest_sha256']={p.name:sha(p) for p in (compact,detail)}
    audit_file=results/('manifest-finalization-'+tag+'.json')
    write_exclusive(audit_file,json_bytes(report))
    report['audit_file']=str(audit_file)
    return report


def main():
    p=argparse.ArgumentParser()
    names=['root','dataset_dir','compact_manifest','detailed_manifest','code_map','target_csv','expected_code_sha256','expected_target_sha256','old_left','old_right','new_left','new_right','backup_dir','result_dir','run_id','version','mode']
    for name in names:p.add_argument('--'+name.replace('_','-'),required=True)
    for name in ['expected_csv_files','expected_pairs','expected_positive','expected_negative','expected_target_rows','expected_code_rows','expected_referenced_code_ids']:
        p.add_argument('--'+name.replace('_','-'),required=True,type=int)
    a=p.parse_args();a.root=Path(a.root)
    ensure(a.mode in {'preflight','apply'},'invalid mode')
    c,d,report=audit(a)
    if a.mode=='apply':report=apply(a,c,d,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('H08_MANIFEST_'+('APPLY_PASS' if a.mode=='apply' else 'PREFLIGHT_PASS_NO_CHANGES'))

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError,json.JSONDecodeError) as e:
        print(f'H08_MANIFEST_FAIL: {e}',file=sys.stderr);sys.exit(1)
