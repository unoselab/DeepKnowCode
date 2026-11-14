#!/usr/bin/env python3
"""Safely add a manually reviewed negative pair, sourcing per-function metadata from existing CSV rows."""
import argparse, csv, hashlib, json, os, shutil, sys, tempfile
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

FIELDS = ('pair_left_func','pair_right_func','clone_predict','pair_left_in','pair_right_in',
          'pair_left_out','pair_right_out','pair_inType_ok','pair_outType_ok','pair_type_ok',
          'pair_left_extractable','pair_right_extractable','pair_refactorable')
PER_SIDE = ('in','out','extractable')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def truth(s):
    if s not in ('True','False'): raise ValueError('Expected True or False, got '+repr(s))
    return s

def load_csv(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f)
        if r.fieldnames != list(FIELDS):
            raise ValueError(f'Unexpected CSV schema in {path}: {r.fieldnames}')
        return list(r)

def key(a,b): return tuple(sorted((a,b)))

def generate(args):
    target=Path(args.target).resolve(); directory=Path(args.ground_truth_dir).resolve()
    archive=Path(args.archive_dir).resolve(); journal=Path(args.journal_dir).resolve()
    files=sorted(directory.glob('pred_*_with_refactorability.csv'))
    if len(files)!=args.expected_csv_files: raise ValueError('CSV count mismatch')
    if target not in files: raise ValueError('Target not among CSV files')
    rows_by_file={p:load_csv(p) for p in files}
    counts=Counter(); seen=set(); observations=defaultdict(lambda: defaultdict(set))
    for file,rows in rows_by_file.items():
        for row in rows:
            a,b=(row['pair_left_func'].strip(),row['pair_right_func'].strip())
            if a==b: raise ValueError('Self pair in '+str(file))
            k=key(a,b)
            if k in seen: raise ValueError('Duplicate unordered pair already present: '+str(k))
            seen.add(k)
            pred=row['clone_predict'].strip()
            if pred not in ('0','1'):raise ValueError('Invalid source label')
            derived=int(a.split('_')[1]==b.split('_')[1])
            if derived!=int(pred):raise ValueError('ID derived label disagreement in '+str(file))
            counts[pred]+=1
            for side,fid in (('left',a),('right',b)):
                for suffix in PER_SIDE:
                    observations[fid][suffix].add(row[f'pair_{side}_{suffix}'].strip())
    if counts['0'] != args.expected_negative_before or counts['1'] != args.expected_positive:
        raise ValueError(f'Current counts mismatch: {counts}')
    if sum(counts.values())!=args.expected_pairs_before:raise ValueError('Total count mismatch')
    a,b=args.left_id,args.right_id
    if key(a,b) in seen:raise ValueError('Replacement pair already exists')
    if a.split('_')[0]!=args.project or b.split('_')[0]!=args.project:raise ValueError('Project mismatch')
    if a.split('_')[1]==b.split('_')[1]:raise ValueError('Target IDs are positive under ID-derived rule')
    if args.original_code_sha256 != sha(Path(args.code_map).resolve()):raise ValueError('Code map SHA mismatch')
    code_ids=set()
    with open(args.code_map,encoding='utf-8') as inp:
        for line in inp:
            if not line.strip():raise ValueError('Blank code map line')
            d=json.loads(line)
            if d['id'] in (a,b):code_ids.add(d['id'])
    if code_ids!={a,b}:raise ValueError('Missing code IDs')
    new={'pair_left_func':a,'pair_right_func':b,'clone_predict':'0'}
    for side,fid in (('left',a),('right',b)):
        for suffix in PER_SIDE:
            found=observations[fid][suffix]
            if len(found)!=1 or '' in found:
                raise ValueError(f'Cannot establish {fid}/{suffix} from existing rows: {sorted(found)}; no insertion')
            new[f'pair_{side}_{suffix}']=next(iter(found))
    for k,v in [('pair_inType_ok',args.in_type_ok),('pair_outType_ok',args.out_type_ok),
                ('pair_type_ok',args.type_ok),('pair_refactorable',args.refactorable)]:new[k]=truth(v)
    if args.type_ok != ('True' if args.in_type_ok=='True' and args.out_type_ok=='True' else 'False'):
        raise ValueError('pair_type_ok inconsistent with in/out type flags')
    if args.refactorable=='True' and args.type_ok=='False':raise ValueError('Refactorable true with incompatible types')
    if len(rows_by_file[target]) != args.expected_target_rows_before:raise ValueError('Target CSV count mismatch')
    chicago=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
    summary={'event':'H08_NEGATIVE_REPLACEMENT','version':args.version,'chicago_time':chicago,
             'target_csv':str(target),'before_sha256':sha(target),'before_counts':dict(counts),
             'new_pair':new,'metadata_provenance':{'per_function':'observed consistently in existing 48 CSVs',
             'pair_specific':'explicit shell-reviewed settings'},'mode':args.mode,
             'manifest_status':'REGENERATION_REQUIRED','previous_h07_h08_outputs':'PRESERVED_STALE_DATASET_REFERENCE'}
    if args.mode=='preflight':
        print(json.dumps(summary,indent=2));print('PREFLIGHT_PASS_NO_CHANGES');return
    archive.mkdir(parents=True,exist_ok=True);journal.mkdir(parents=True,exist_ok=True)
    backup=archive / f'{target.stem}-before-h08-negative-replacement-{chicago}.csv'
    audit=journal / f'negative-replacement-{chicago}.json'
    if backup.exists() or audit.exists():raise ValueError('Backup/audit path exists')
    with backup.open('xb') as out, target.open('rb') as inp:shutil.copyfileobj(inp,out)
    if sha(backup)!=summary['before_sha256']:raise ValueError('Backup integrity mismatch')
    # Atomic CSV replacement: new row is appended; existing rows remain in original order.
    fd,tmp=tempfile.mkstemp(prefix='.'+target.stem+'-',suffix='.csv',dir=str(target.parent))
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='') as out:
            wr=csv.DictWriter(out,fieldnames=FIELDS,lineterminator='\n')
            wr.writeheader();wr.writerows(rows_by_file[target]);wr.writerow(new)
            out.flush();os.fsync(out.fileno())
        os.chmod(tmp,target.stat().st_mode & 0o777)
        reread=load_csv(Path(tmp))
        if len(reread)!=args.expected_target_rows_before+1 or reread[-1]!=new:
            raise ValueError('Temporary target verification failed')
        os.replace(tmp,target)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    summary.update({'backup_csv':str(backup),'after_sha256':sha(target),
                    'after_counts':{'positive':counts['1'],'negative':counts['0']+1,
                                    'pairs':sum(counts.values())+1},
                    'target_rows_after':len(rows_by_file[target])+1})
    with audit.open('x',encoding='utf-8') as out:
        json.dump(summary,out,indent=2);out.write('\n')
    print(json.dumps(summary,indent=2));print('JAVA_NEGATIVE_REPLACEMENT_PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['target','ground-truth-dir','archive-dir','journal-dir','code-map','original-code-sha256',
              'left-id','right-id','project','in-type-ok','out-type-ok','type-ok','refactorable','version','mode']:
        p.add_argument('--'+k,required=True)
    for k in ['expected-csv-files','expected-positive','expected-negative-before','expected-pairs-before','expected-target-rows-before']:
        p.add_argument('--'+k,required=True,type=int)
    a=p.parse_args();
    if a.mode not in ('preflight','apply'):p.error('mode must be preflight or apply')
    try:generate(a)
    except Exception as e:
        print(f'FAIL_NO_UNVERIFIED_ADDITION: {e}',file=sys.stderr)
        raise SystemExit(1)
if __name__=='__main__':main()
