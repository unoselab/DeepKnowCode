#!/usr/bin/env python3
"""J-01g v5: read-only Python content-ID leakage audit; never modifies inputs."""
import argparse
import collections
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HASH_RE = re.compile(r'^py-sha256:[0-9a-f]{64}$')


def rows(path):
    with path.open('r', encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f'Blank row: {path}:{n}')
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'Not JSON object: {path}:{n}')
            yield n, row


def check_hash(value, where):
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise ValueError(f'Invalid canonical Function Content-ID at {where}: {str(value)[:100]}')
    return value


def unordered(a, b):
    return tuple(sorted((a, b)))


def read_canonical(path, expected_total, expected_python, split):
    funcs, pairs, counts = set(), set(), collections.Counter()
    for n, row in rows(path):
        counts['total'] += 1
        if row.get('corpus') != 'python_mixed':
            if 'left' in row or 'right' in row:
                raise ValueError(f'Unexpected non-Python enrichment: {path}:{n}')
            continue
        counts['python_mixed'] += 1
        for side, source_field in (('left', 'id1'), ('right', 'id2')):
            info = row.get(side)
            if not isinstance(info, dict):
                raise ValueError(f'Missing canonical {side} metadata: {path}:{n}')
            fid = check_hash(info.get('function_id'), f'{path}:{n}:{side}')
            if info.get('language') != 'python' or info.get('source_id') != row.get(source_field):
                raise ValueError(f'Mismatched source identity: {path}:{n}:{side}')
            funcs.add(fid)
        pairs.add(unordered(row['left']['function_id'], row['right']['function_id']))
    if counts['total'] != expected_total or counts['python_mixed'] != expected_python:
        raise ValueError(f'{split} record count mismatch: {dict(counts)} expected={expected_total},{expected_python}')
    if not funcs or not pairs:
        raise ValueError(f'{split}: No recognized Python content IDs')
    return funcs, pairs, dict(counts)


def audit_csv(directory, train_funcs, train_pairs, val_funcs, val_pairs, expected_rows, expected_files, expected_pos, expected_neg, examples_limit, csv_left_field, csv_right_field, csv_label_field):
    files = sorted(directory.glob('*.csv'))
    if len(files) != expected_files:
        raise ValueError(f'{directory}: expected {expected_files} CSVs, got {len(files)}')
    counts = collections.Counter()
    fids, pairset = set(), set()
    examples = {'train_function': [], 'train_pair': [], 'validation_function': [], 'validation_pair': []}
    per_project = []
    for file in files:
        if file.stem.count('__') != 1 or any(not section for section in file.stem.split('__')):
            raise ValueError(f'{file}: filename must be owner__project.csv')
        project_count = collections.Counter()
        with file.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != [csv_left_field, csv_right_field, csv_label_field]:
                raise ValueError(f'{file}: CSV schema mismatch; expected={[csv_left_field, csv_right_field, csv_label_field]!r}; actual={reader.fieldnames!r}')
            for line, row in enumerate(reader, 2):
                a = check_hash(row.get(csv_left_field), f'{file}:{line}:left')
                b = check_hash(row.get(csv_right_field), f'{file}:{line}:right')
                label = row.get(csv_label_field)
                if label not in ('0', '1'):
                    raise ValueError(f'{file}:{line}: invalid {csv_label_field}={label!r}')
                pair = unordered(a, b)
                if pair in pairset:
                    raise ValueError(f'Duplicate evaluation unordered pair across CSVs: {file}:{line}')
                pairset.add(pair); fids.update((a,b))
                counts['pairs'] += 1
                counts['positive' if label == '1' else 'negative'] += 1
                project_count['pairs'] += 1
                for prefix, functions, pairs in (('train',train_funcs,train_pairs),('validation',val_funcs,val_pairs)):
                    overlap = (a in functions) + (b in functions)
                    if overlap:
                        counts[prefix+'_pairs_with_function_overlap'] += 1
                        counts[prefix+'_function_occurrences_overlap'] += overlap
                        project_count[prefix+'_pairs_with_function_overlap'] += 1
                        if len(examples[prefix+'_function']) < examples_limit:
                            examples[prefix+'_function'].append({'csv':file.name,'line':line,'left':a,'right':b,'label':int(label)})
                    if pair in pairs:
                        counts[prefix+'_pair_overlap'] += 1
                        project_count[prefix+'_pair_overlap'] += 1
                        if len(examples[prefix+'_pair']) < examples_limit:
                            examples[prefix+'_pair'].append({'csv':file.name,'line':line,'left':a,'right':b,'label':int(label)})
        per_project.append({'file':file.name, **dict(project_count)})
    if (counts['pairs'],counts['positive'],counts['negative']) != (expected_rows,expected_pos,expected_neg):
        raise ValueError(f'{directory}: evaluation totals mismatch {dict(counts)}')
    result = {'csv_directory':str(directory),'csv_files':len(files),
              'pair_count':counts['pairs'],'positive':counts['positive'],'negative':counts['negative'],
              'unique_function_content_ids':len(fids),'unique_unordered_pairs':len(pairset),
              'train_overlapping_unique_function_ids':len(fids & train_funcs),
              'validation_overlapping_unique_function_ids':len(fids & val_funcs),
              'train_pairs_with_function_overlap':counts['train_pairs_with_function_overlap'],
              'validation_pairs_with_function_overlap':counts['validation_pairs_with_function_overlap'],
              'train_exact_unordered_pair_overlap':counts['train_pair_overlap'],
              'validation_exact_unordered_pair_overlap':counts['validation_pair_overlap'],
              'per_project':per_project,'examples':examples}
    return result


def run(args):
    inputs = [args.train,args.validation]
    if any(not p.is_file() for p in inputs):
        raise ValueError('Canonical training/validation input missing')
    for d in (args.v1_dir,args.v2_dir):
        if not d.is_dir():
            raise ValueError(f'Missing evaluation directory: {d}')
    if args.output.exists():
        raise ValueError(f'No-overwrite: audit report already exists: {args.output}')
    tf,tp,tc=read_canonical(args.train,args.expected_train_rows,args.expected_train_python_rows,'train')
    vf,vp,vc=read_canonical(args.validation,args.expected_validation_rows,args.expected_validation_python_rows,'validation')
    res={'experiment':'j-01g','version':args.version,'timestamp_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
         'status':'UNDETERMINED','scope':'EXACT_PYTHON_FUNCTION_CONTENT_IDS_AND_UNORDERED_PAIRS',
         'canonical':{'train_file':str(args.train),'validation_file':str(args.validation),
                      'train':tc,'validation':vc,
                      'train_distinct_function_content_ids':len(tf),'train_distinct_unordered_pairs':len(tp),
                      'validation_distinct_function_content_ids':len(vf),'validation_distinct_unordered_pairs':len(vp),
                      'shared_train_validation_content_ids':len(tf & vf),'shared_train_validation_unordered_pairs':len(tp & vp)},
         'evaluation':{}, 'csv_fields':{'left':args.csv_left_field,'right':args.csv_right_field,'label':args.csv_label_field}, 'limitations':[
          'Exact UTF-8 source-code content hash overlap only; transformed/renamed/semantic equivalents may escape detection.',
          'Only python_mixed training/validation records carry canonical Python content IDs. Java rows are out of scope.',
          'Identical source-code content at distinct locations has the same content ID; overlap is not proof of same file occurrence.',
          'Zero overlap does not independently prove absence of pretraining contamination or non-exact leakage.' ]}
    for version, path, n, pos, neg in (('v1',args.v1_dir,args.expected_v1_rows,args.expected_v1_positive,args.expected_v1_negative),('v2',args.v2_dir,args.expected_v2_rows,args.expected_v2_positive,args.expected_v2_negative)):
        res['evaluation'][version]=audit_csv(path,tf,tp,vf,vp,n,args.expected_csv_files,pos,neg,args.max_examples,args.csv_left_field,args.csv_right_field,args.csv_label_field)
    problems=[]
    for version,d in res['evaluation'].items():
        for k in ('train_overlapping_unique_function_ids','validation_overlapping_unique_function_ids','train_exact_unordered_pair_overlap','validation_exact_unordered_pair_overlap'):
            if d[k]: problems.append(f'{version}.{k}={d[k]}')
    if res['canonical']['shared_train_validation_content_ids'] or res['canonical']['shared_train_validation_unordered_pairs']:
        problems.append('canonical.train_validation_content_overlap')
    res['finding_flags']=problems
    res['status']='OVERLAP_DETECTED' if problems else 'NO_EXACT_OVERLAP_DETECTED'
    return res


def parse():
    p=argparse.ArgumentParser()
    for name in ('version','train','validation','v1-dir','v2-dir','output','csv-left-field','csv-right-field','csv-label-field'):
        p.add_argument('--'+name,required=True)
    for name in ('expected-train-rows','expected-train-python-rows','expected-validation-rows','expected-validation-python-rows','expected-v1-rows','expected-v1-positive','expected-v1-negative','expected-v2-rows','expected-v2-positive','expected-v2-negative','expected-csv-files','max-examples'):
        p.add_argument('--'+name,type=int,required=True)
    a=p.parse_args()
    fields = [a.csv_left_field, a.csv_right_field, a.csv_label_field]
    if len(set(fields)) != 3 or any(not v or '\n' in v or '\r' in v for v in fields):
        p.error('CSV fields must be three nonempty distinct column names')
    for name in ('train','validation','v1_dir','v2_dir','output'):
        setattr(a,name,Path(getattr(a,name)).expanduser().resolve())
    if any(getattr(a,name)<0 for name in ('expected_train_rows','expected_train_python_rows','expected_validation_rows','expected_validation_python_rows','expected_v1_rows','expected_v2_rows','expected_csv_files','max_examples')):
        p.error('Expected counts must be nonnegative')
    return a


def main():
    args=parse()
    try:
        result=run(args)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as f:
            json.dump(result,f,sort_keys=True,indent=2,ensure_ascii=False)
            f.write('\n')
        print(json.dumps({'report':str(args.output),'status':result['status'],
                          'canonical':result['canonical'],'findings':result['finding_flags'],
                          'evaluation_summary':{k:{name:v[name] for name in ('pair_count','train_overlapping_unique_function_ids','validation_overlapping_unique_function_ids','train_exact_unordered_pair_overlap','validation_exact_unordered_pair_overlap')} for k,v in result['evaluation'].items()}},indent=2))
        return 3 if result['status']=='OVERLAP_DETECTED' else 0
    except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError,csv.Error) as e:
        print(f'J01G_AUDIT_ERROR:{type(e).__name__}: {e}',file=sys.stderr)
        return 2

if __name__=='__main__':
    sys.exit(main())
