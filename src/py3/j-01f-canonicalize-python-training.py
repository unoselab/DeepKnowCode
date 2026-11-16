#!/usr/bin/env python3
"""J-01f v2: non-destructive schema-aware canonical Python training pair export."""
import argparse
import collections
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v2'
EXPECTED_KEYS = {'corpus', 'id1', 'id2', 'label', 'tokens'}
METADATA_KEYS = ('language','function_id','source_id','source_dataset','repository','revision','path','start_line','end_line')

def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def records(path):
    with path.open('r', encoding='utf-8-sig') as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f'Blank JSONL row {path}:{lineno}')
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f'Expected JSON object {path}:{lineno}')
            yield lineno, obj

def index_clean(path):
    lookup={}
    for line, obj in records(path):
        ident, code = obj.get('idx'), obj.get('func')
        if not isinstance(ident,str) or not ident or not isinstance(code,str) or not code.strip():
            raise ValueError(f'Invalid idx/func {path}:{line}')
        if ident in lookup:
            raise ValueError(f'Duplicate idx {path}:{line}: {ident}')
        lookup[ident]='py-sha256:'+hashlib.sha256(code.encode('utf-8')).hexdigest()
    if not lookup: raise ValueError('Empty source corpus')
    return lookup

def metadata(ident, lookup):
    return dict(language='python',function_id=lookup[ident],source_id=ident,
                source_dataset='python_mixed',repository=None,revision=None,
                path=None,start_line=None,end_line=None)

def run(args):
    inputs=[args.clean,args.train,args.validation]
    for path in inputs:
        if not path.is_file(): raise ValueError(f'Missing input: {path}')
    out=args.output
    if out.exists():raise ValueError(f'Output exists, refusing overwrite: {out}')
    if any(out==path or out in path.parents or path in out.parents for path in inputs):
        raise ValueError('Output/input path overlap')
    lookup=index_clean(args.clean)
    summaries={}; output_records={}; seen_ids={}
    for split,path in (('train',args.train),('validation',args.validation)):
        counter=collections.Counter(); used=set(); result=[]
        for lineno,obj in records(path):
            if set(obj)!=EXPECTED_KEYS:
                raise ValueError(f'Unexpected prepared schema {path}:{lineno}: {sorted(obj)}')
            corpus=obj['corpus']; a=obj['id1']; b=obj['id2']
            if not isinstance(corpus,str) or not isinstance(a,str) or not isinstance(b,str) or not a or not b or obj['label'] not in (0,1) or type(obj['label']) is not int or type(obj['tokens']) is not int or obj['tokens']<=0:
                raise ValueError(f'Invalid prepared pair fields {path}:{lineno}')
            counter['total']+=1;counter['corpus:'+corpus]+=1
            row=dict(obj)
            if corpus=='python_mixed':
                if a not in lookup or b not in lookup:
                    raise ValueError(f'Unmapped python_mixed ID {path}:{lineno}: {a!r}, {b!r}')
                row['left']=metadata(a,lookup);row['right']=metadata(b,lookup)
                counter['python_mixed_enriched']+=1
                used.update((a,b))
            result.append(row)
        summaries[split]=dict(counter);output_records[split]=result;seen_ids[split]=used
    if not summaries['train'].get('python_mixed_enriched') or not summaries['validation'].get('python_mixed_enriched'):
        raise ValueError('Python corpus missing in training or validation split')
    audit={'experiment':'j-01f','version':VERSION,'timestamp_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
           'status':'INSPECT_PASS' if args.mode=='inspect' else 'BUILT',
           'policy':'DERIVED_ONLY_ORIGINAL_FROZEN_PREPARED_UNMODIFIED',
           'input_sha256':{str(p):sha_file(p) for p in inputs},
           'source_records':len(lookup),'split_counts':summaries,
           'shared_source_ids_between_splits':len(seen_ids['train'] & seen_ids['validation']),
           'hash_convention':'sha256(func.encode(utf-8)); no normalization',
           'source_input_policy':'RAW_PYTHON_MIXED_IDX_FUNC_VALIDATED_AGAINST_PREPARED_PAIR_IDS',
           'provenance_policy':'Unknown repository/location/revision represented as null; no inferred provenance',
           'unchanged_non_python_records':True}
    if args.mode=='build':
        out.mkdir(parents=True,exist_ok=False)
        for split,rs in output_records.items():
            with (out/(split+'.jsonl')).open('x',encoding='utf-8') as stream:
                for r in rs: stream.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
        audit['output_sha256']={name:sha_file(out/name) for name in ('train.jsonl','validation.jsonl')}
        (out/'manifest.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2,sort_keys=True))

def main():
    p=argparse.ArgumentParser()
    for k in ('mode','clean','train','validation','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args()
    if a.mode not in ('inspect','build'):p.error('--mode must be inspect or build')
    for attr in ('clean','train','validation','output'):setattr(a,attr,Path(getattr(a,attr)).resolve())
    try:run(a)
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
        print(f'J01F_FAIL:{type(exc).__name__}:{exc}',file=sys.stderr);return 1
    return 0
if __name__=='__main__':sys.exit(main())
