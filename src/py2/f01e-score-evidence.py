#!/usr/bin/env python3
"""F-01e v2: fail-closed, byte-bound Reduced-4D component evidence materializer.

This script DOES NOT compute D-04 components from code. It requires audited
component measurements on the exact CodeLlama-effective code of every frozen
training pair; only the published weighted aggregation is computed here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

VERSION = 'v2'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def lines(path):
    with Path(path).open('rb') as stream:
        for i, raw in enumerate(stream, 1):
            if not raw.strip():
                raise ValueError(f'EMPTY_RECORD {path}:{i}')
            yield i, json.loads(raw), hashlib.sha256(raw.rstrip(b'\r\n')).hexdigest()

def ident(obj):
    return (str(obj['corpus']), str(obj['id1']), str(obj['id2']), int(obj['label']))

def valid_hash(v):
    return isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)

def cli():
    p=argparse.ArgumentParser(description=__doc__)
    for s in ('frozen-train','components-file','output-dir'):
        p.add_argument('--'+s, type=Path, required=True)
    for s in ('train-sha256','components-sha256','provenance','score-policy','component-policy','score-field'):
        p.add_argument('--'+s, required=True)
    p.add_argument('--component-fields', nargs=4, required=True)
    p.add_argument('--weights', nargs=4, type=float, required=True)
    p.add_argument('--tolerance', type=float, required=True)
    p.add_argument('--expected-count', type=int, required=True)
    p.add_argument('--max-seq-length', type=int, required=True)
    p.add_argument('--require-effective-code-sha256', required=True, choices=('true','false'))
    return p.parse_args()

def main():
    a=cli()
    if a.provenance in ('', 'UNVERIFIED', 'INSPECT_ONLY'):
        raise ValueError('PROVENANCE_REQUIRED: independent Code/AST computation evidence')
    if a.score_policy!='reduced-4d-externally-verified-v1':
        raise ValueError('SCORE_POLICY_INVALID')
    if a.component_policy!='D04_VERIFIED_EFFECTIVE_CODE_COMPONENTS':
        raise ValueError('COMPONENT_POLICY_INVALID')
    if a.component_fields != ['d_sim','d_cf','d_len','d_df'] or a.score_field != 'reduced_4d_score':
        raise ValueError('D04_COMPONENT_FIELDS_INVALID')
    if not all(math.isfinite(x) and x>=0 for x in a.weights) or abs(sum(a.weights)-1)>1e-12:
        raise ValueError('D04_WEIGHT_INVALID')
    if not (0<a.tolerance<=1e-4) or a.expected_count<=0 or a.max_seq_length<=0:
        raise ValueError('BOUNDS_INVALID')
    for v in (a.train_sha256, a.components_sha256):
        if not valid_hash(v): raise ValueError('SHA256_FORMAT_INVALID')
    if digest(a.frozen_train)!=a.train_sha256:
        raise ValueError('FROZEN_TRAIN_SHA256_MISMATCH')
    if digest(a.components_file)!=a.components_sha256:
        raise ValueError('COMPONENT_INPUT_SHA256_MISMATCH')
    frozen={}
    for rowno,row,rawhash in lines(a.frozen_train):
        k=ident(row)
        if k in frozen: raise ValueError('FROZEN_DUPLICATE_IDENTITY '+str(k))
        t=row['tokens']
        if type(t) is not int or not (0<t<=a.max_seq_length):
            raise ValueError('FROZEN_TOKEN_LENGTH_INVALID')
        frozen[k]=rawhash
    if len(frozen)!=a.expected_count:
        raise ValueError(f'FROZEN_COUNT_MISMATCH {len(frozen)} != {a.expected_count}')
    outputs={}
    for lineno,row,_ in lines(a.components_file):
        k=ident(row)
        if k not in frozen: raise ValueError('UNKNOWN_COMPONENT_PAIR '+str(k))
        if k in outputs: raise ValueError('DUPLICATE_COMPONENT_PAIR '+str(k))
        if row.get('source_row_sha256')!=frozen[k]:
            raise ValueError('SOURCE_ROW_SHA256_MISMATCH '+str(k))
        if row.get('score_provenance')!=a.provenance:
            raise ValueError('COMPONENT_PROVENANCE_MISMATCH '+str(k))
        if row.get('component_policy')!=a.component_policy:
            raise ValueError('COMPONENT_POLICY_MISMATCH '+str(k))
        if a.require_effective_code_sha256=='true':
            for key in ('effective_code_a_sha256','effective_code_b_sha256'):
                if not valid_hash(row.get(key)):
                    raise ValueError('EFFECTIVE_CODE_SHA_MISSING '+key+' '+str(k))
        vals=[]
        for name in a.component_fields:
            v=row.get(name)
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=1:
                raise ValueError('COMPONENT_INVALID '+name+' '+str(k))
            vals.append(float(v))
        score=sum(w*v for w,v in zip(a.weights,vals))
        out={name:row[name] for name in a.component_fields}
        out.update({'corpus':k[0],'id1':k[1],'id2':k[2],'label':k[3],
          'source_row_sha256':frozen[k], 'score_provenance':a.provenance,
          'score_policy':a.score_policy, a.score_field:score})
        for key in ('effective_code_a_sha256','effective_code_b_sha256'):
            if key in row: out[key]=row[key]
        outputs[k]=out
    if set(outputs)!=set(frozen):
        missing=sorted(set(frozen)-set(outputs))
        raise ValueError('COMPONENT_COVERAGE_MISSING '+str(len(missing))+' first='+str(missing[:1]))
    # Work completes in memory before any output path is created.
    a.output_dir.mkdir(parents=True,exist_ok=False)
    target=a.output_dir/'verified_scores.jsonl'
    with target.open('x',encoding='utf-8') as f:
        for k in frozen: # deterministic frozen training order
            f.write(json.dumps(outputs[k],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n')
    manifest={
      'status':'F01E_V2_SCORE_AGGREGATION_PASS_EXTERNAL_COMPONENTS',
      'created_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
      'version':VERSION,'frozen_train_sha256':a.train_sha256,
      'components_sha256':a.components_sha256,'score_file_sha256':digest(target),
      'component_policy':a.component_policy,'score_policy':a.score_policy,
      'score_provenance':a.provenance,'component_fields':a.component_fields,
      'weights':a.weights,'score_tolerance':a.tolerance,
      'effective_code_sha256_required':a.require_effective_code_sha256=='true',
      'score_count':len(outputs),
      'limitation':'No independent recomputation of Code/AST difficulty components; hashes alone do not verify correctness of source code or score computation.'}
    with (a.output_dir/'score_manifest.json').open('x',encoding='utf-8') as f:
        json.dump(manifest,f,indent=2,ensure_ascii=False,sort_keys=True);f.write('\n')
    print(json.dumps({'status':manifest['status'],'score_path':str(target.resolve()),
        'score_sha256':manifest['score_file_sha256'],'pair_count':len(outputs)},sort_keys=True),flush=True)

if __name__=='__main__':
    main()
