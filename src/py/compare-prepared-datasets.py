#!/usr/bin/env python3
"""Read-only, domain-aware pair/manifest comparison for prepared JSONL datasets."""
import argparse
import collections
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def canon(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def load(path):
    data = {}
    dup = []
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}')
    for n, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            raise ValueError(f'Blank line at {path}:{n}')
        x = json.loads(line)
        if not isinstance(x, dict) or not all(z in x for z in ('id1','id2','corpus','label')):
            raise ValueError(f'Invalid pair at {path}:{n}, required id1,id2,corpus,label')
        key = (str(x['corpus']), *sorted((str(x['id1']), str(x['id2']))))
        if key in data: dup.append({'pair': list(key), 'line': n})
        else: data[key] = {'row': x, 'line': n}
    return data, dup


def groups(a, b, max_examples):
    ak,bk=set(a),set(b)
    cats = sorted(set(k[0] for k in ak|bk))
    counts={}
    for cat in cats:
        x={k for k in ak if k[0]==cat};y={k for k in bk if k[0]==cat}
        counts[cat]={'frozen':len(x),'current':len(y),'shared':len(x&y),
                     'frozen_only':len(x-y),'current_only':len(y-x),
                     'shared_percent_of_frozen':round(100*len(x&y)/len(x),3) if x else None,
                     'frozen_only_examples':[list(z) for z in sorted(x-y)[:max_examples]],
                     'current_only_examples':[list(z) for z in sorted(y-x)[:max_examples]]}
    return counts


def compare_split(a,b,cap):
    ak,bk=set(a),set(b)
    common=ak&bk
    changes=collections.Counter()
    examples={}
    label_diffs=[]
    for k in sorted(common):
        x=a[k]['row'];y=b[k]['row']
        fields=[f for f in sorted(set(x)|set(y)) if f not in x or f not in y or x[f]!=y[f]]
        if fields:
            for f in fields: changes[f]+=1
            if len(examples)<cap:
                examples[canon(k)]={'frozen_line':a[k]['line'],'current_line':b[k]['line'],'changed_fields':fields}
            if x.get('label')!=y.get('label'):
                label_diffs.append({'pair':list(k),'frozen':x.get('label'),'current':y.get('label')})
    return {'frozen_rows':len(a),'current_rows':len(b),'common_pairs':len(common),
            'frozen_only':len(ak-bk),'current_only':len(bk-ak),
            'unchanged_pair_rows':len(common)-sum(bool([f for f in set(a[k]['row'])|set(b[k]['row']) if f not in a[k]['row'] or f not in b[k]['row'] or a[k]['row'][f]!=b[k]['row'][f]]) for k in common),
            'changed_field_counts':dict(changes),'label_change_count':len(label_diffs),
            'label_changes':label_diffs[:cap],'corpus':groups(a,b,cap),
            'changed_examples':examples}


def flatten_diffs(a,b,path='',out=None):
    if out is None:out=[]
    if isinstance(a,dict) and isinstance(b,dict):
        for k in sorted(set(a)|set(b)):
            p=path+'.'+k if path else k
            if k not in a:out.append({'path':p,'type':'only_current','current':b[k]})
            elif k not in b:out.append({'path':p,'type':'only_frozen','frozen':a[k]})
            else:flatten_diffs(a[k],b[k],p,out)
    elif a!=b:
        out.append({'path':path,'frozen':a,'current':b})
    return out


def pair_presence(keys, splitmaps):
    a,b=keys
    matches={}
    for name,mp in splitmaps.items():
        exact=[{'pair':list(k),'line':v['line'],'label':v['row'].get('label'),
                'override_fields':[f for f in ('code_a_override','code_b_override','changed_sides','removed_count','origin') if f in v['row']]}
               for k,v in mp.items() if k[1:]==tuple(sorted((a,b)))]
        matches[name]=exact
    return matches


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-dir','current-dir','report-file','max-examples','focus-pair'):
        p.add_argument('--'+name,required=name!='focus-pair',action='append' if name=='focus-pair' else 'store')
    args=p.parse_args();cap=int(args.max_examples)
    if cap<0:p.error('max-examples must be >=0')
    dirs={'frozen':Path(args.frozen_dir).expanduser().resolve(),'current':Path(args.current_dir).expanduser().resolve()}
    pairs={};summaries={};dups={};file_info={}
    for split in ('train','validation'):
        mp={};ds={};fi={}
        for side in dirs:
            path=dirs[side]/f'{split}.jsonl'
            mp[side],ds[side]=load(path)
            fi[side]={'path':str(path),'sha256':sha(path),'rows':len(mp[side])}
        pairs[split]=mp;summaries[split]=compare_split(mp['frozen'],mp['current'],cap)
        dups[split]=ds;file_info[split]=fi
    frozen_train=set(pairs['train']['frozen']);current_train=set(pairs['train']['current'])
    frozen_val=set(pairs['validation']['frozen']);current_val=set(pairs['validation']['current'])
    cross={'frozen_train_in_current_validation':sorted(frozen_train&current_val),
           'current_train_in_frozen_validation':sorted(current_train&frozen_val),
           'within_frozen_train_validation':sorted(frozen_train&frozen_val),
           'within_current_train_validation':sorted(current_train&current_val)}
    cross={k:{'count':len(v),'examples':[list(x) for x in v[:cap]]} for k,v in cross.items()}
    manifests={}; manifest_files={}
    for side in dirs:
        path=dirs[side]/'manifest.json'
        if path.is_file():
            manifests[side]=json.loads(path.read_text(encoding='utf-8'))
            manifest_files[side]={'sha256':sha(path),'path':str(path)}
        else:manifests[side]={};manifest_files[side]={'missing':True}
    differences=flatten_diffs(manifests['frozen'],manifests['current'])
    focused={}
    for item in args.focus_pair or []:
        ids=item.split('::')
        if len(ids)!=2:p.error(f'--focus-pair requires ID1::ID2, got {item}')
        names={f'{split}_{side}':pairs[split][side] for split in pairs for side in ('frozen','current')}
        focused[item]=pair_presence(ids,names)
    result={'at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
            'inputs':{k:str(v) for k,v in dirs.items()},'pair_identity':'(corpus, unordered id1/id2), excludes label',
            'files':file_info,'splits':summaries,'cross_split':cross,'duplicates':dups,
            'manifest_files':manifest_files,'manifest_differences_count':len(differences),
            'manifest_differences':differences,'focus_pairs':focused,
            'interpretation_warning':'Pair mismatch does not establish whether truncation caused resampling; verify manifests and preparation version. Inherited heldout PASS may refer to older evaluation snapshots.'}
    target=Path(args.report_file).expanduser().resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError('Refusing to overwrite comparison report: '+str(target))
    with target.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    summary={'report':str(target),'train':{k:result['splits']['train'][k] for k in ('common_pairs','frozen_only','current_only')},
             'validation':{k:result['splits']['validation'][k] for k in ('common_pairs','frozen_only','current_only')},
             'per_corpus_train':{k:{z:v[z] for z in ('frozen','current','shared','frozen_only','current_only')} for k,v in result['splits']['train']['corpus'].items()},
             'per_corpus_validation':{k:{z:v[z] for z in ('frozen','current','shared','frozen_only','current_only')} for k,v in result['splits']['validation']['corpus'].items()},
             'cross_split':{k:v['count'] for k,v in cross.items()},
             'manifest_difference_count':len(differences),
             'focus_presence':{k:{n:len(v) for n,v in x.items()} for k,x in focused.items()}}
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
