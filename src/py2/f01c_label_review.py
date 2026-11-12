#!/usr/bin/env python3
"""F-01c v18: offline label review export and strict evidence verifier.

The output is NOT training approval. A human must review and annotate all eight pairs.
All experiment configuration is passed as required CLI arguments by the Shell wrapper.
"""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')


def load(path):
    with path.open('r',encoding='utf-8') as h: return json.load(h)


def exclusive_json(path,obj):
    with path.open('x',encoding='utf-8') as h: json.dump(obj,h,ensure_ascii=False,indent=2,sort_keys=True);h.write('\n')


def inspect(prepared, expected_train, expected_val, expected_manifest, expected_pairs, max_tokens):
    folder=prepared/'prepared'
    paths={n:folder/n for n in ('manifest.json','train.jsonl','validation.jsonl')}
    for p in paths.values():
        if not p.is_file(): raise ValueError('Missing prepared input: '+str(p))
    actual={n:sha(p.read_bytes()) for n,p in paths.items()}
    for name,expect in [('train.jsonl',expected_train),('validation.jsonl',expected_val),('manifest.json',expected_manifest)]:
        if actual[name]!=expect: raise ValueError('Prepared SHA256 mismatch: '+name)
    manifest=load(paths['manifest.json'])
    if manifest.get('status')!='PREPARED_PENDING_LABEL_REVIEW' or manifest.get('approved_for_training') is not False:
        raise ValueError('Expected read-only pending-label-review prepared manifest')
    samples=manifest.get('sample',{})
    if samples.get('train',{}).get('sha256')!=actual['train.jsonl'] or samples.get('validation',{}).get('sha256')!=actual['validation.jsonl']:
        raise ValueError('Manifest dataset hashes do not match files')
    train=[json.loads(line) for line in paths['train.jsonl'].read_text(encoding='utf-8').splitlines() if line.strip()]
    val=[json.loads(line) for line in paths['validation.jsonl'].read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(train)!=19464 or len(val)!=512: raise ValueError('Unexpected prepared dataset counts')
    if any(row['tokens']>max_tokens for row in train+val): raise ValueError('Overlength sample')
    sel=[r for r in train if r.get('origin')=='structural-v13']
    if len(sel)!=expected_pairs or len({(r['id1'],r['id2']) for r in sel})!=len(sel): raise ValueError('Transformed pair count/duplicate')
    for r in sel:
        if (r['label']!=0 or not r.get('code_a_override') or not r.get('code_b_override') or
                len(r.get('changed_sides',[])) not in (1,2)): raise ValueError('Malformed transformed row')
        for k,ikey in [('code_a_override','code_instance_id1'),('code_b_override','code_instance_id2')]:
            if not r.get(ikey): raise ValueError('Missing code instance identity')
    return manifest,sel,actual


def keys(r):
    return {'id1':r['id1'],'id2':r['id2'],'label':r['label'],
            'changed_sides':r['changed_sides'],'tokens':r['tokens'],
            'code_instance_id1':r['code_instance_id1'],'code_instance_id2':r['code_instance_id2'],
            'code_a_sha256':sha(r['code_a_override'].encode('utf-8')),
            'code_b_sha256':sha(r['code_b_override'].encode('utf-8'))}


def produce(prepared,output,expected_train,expected_val,expected_manifest,expected_pairs,max_tokens):
    manifest,selected,hashes=inspect(prepared,expected_train,expected_val,expected_manifest,expected_pairs,max_tokens)
    output.mkdir(parents=True,exist_ok=False)
    records=sorted((keys(r) for r in selected), key=lambda r:(r['id1'],r['id2']))
    bundle={'schema':'f01c-v18-label-review-v1','prepared_run':str(prepared),'prepared_sha256':hashes,
            'train_pair_ids_sha256':manifest['sample']['train']['pair_ids_sha256'],
            'required_pair_count':expected_pairs,'records':records}
    exclusive_json(output/'review_inventory.json',bundle)
    decision={'schema':bundle['schema'],'prepared_sha256':hashes,
              'train_pair_ids_sha256':bundle['train_pair_ids_sha256'],
              'reviewer':'REPLACE_WITH_REVIEWER_NAME','review_basis':'Full effective Code A/B; negative clone relation after truncation',
              'decisions':[dict(r,decision='PENDING',reason='') for r in records]}
    exclusive_json(output/'label_review_decisions.json',decision)
    with (output/'label_review_full_code.md').open('x',encoding='utf-8') as out:
        out.write('# F-01c v18 — transformed negative pair label review\n\n')
        out.write('Training is NOT approved. Review whether effective Code A and Code B remain non-clones.\n\n')
        for r in sorted(selected,key=lambda r:(r['id1'],r['id2'])):
            out.write('## '+r['id1']+' / '+r['id2']+'\n\n')
            out.write('Changed: '+','.join(r['changed_sides'])+'; Tokens: '+str(r['tokens'])+'; removed: '+str(r.get('removed_count'))+'\n\n')
            out.write('### Effective Code A\n\n```java\n'+r['code_a_override']+'\n```\n\n')
            out.write('### Effective Code B\n\n```java\n'+r['code_b_override']+'\n```\n\n')
            out.write('**Human decision:** PENDING; **Rationale:** ______\n\n---\n\n')
    print(json.dumps({'status':'F01C_V18_REVIEW_EXPORTED_NOT_APPROVED','pairs':len(records),'output':str(output),'approved_for_training':False}))


def verify(prepared,decisions_path,output,expected_train,expected_val,expected_manifest,expected_pairs,max_tokens):
    manifest,selected,hashes=inspect(prepared,expected_train,expected_val,expected_manifest,expected_pairs,max_tokens)
    d=load(decisions_path)
    if d.get('schema')!='f01c-v18-label-review-v1' or d.get('prepared_sha256')!=hashes or d.get('train_pair_ids_sha256')!=manifest['sample']['train']['pair_ids_sha256']:
        raise ValueError('Decision document is not bound to pinned prepared dataset')
    reviewer=d.get('reviewer','').strip()
    if not reviewer or reviewer=='REPLACE_WITH_REVIEWER_NAME':raise ValueError('Reviewer identity required')
    actual=d.get('decisions',[])
    expected=sorted((keys(r) for r in selected),key=lambda r:(r['id1'],r['id2']))
    if len(actual)!=expected_pairs:raise ValueError('Every transformed pair requires an independent review')
    valid=[]
    for e,record in zip(expected,actual):
        if any(record.get(k)!=v for k,v in e.items()):raise ValueError('Altered/reordered candidate identity '+e['id1'])
        if record.get('decision') not in ('NON_CLONE','CLONE','UNSURE'):
            raise ValueError('Incomplete or invalid decision: '+e['id1']+'/'+e['id2'])
        if len(record.get('reason','').strip())<12:raise ValueError('Specific review rationale required for '+e['id1'])
        valid.append(record)
    all_pass=all(r['decision']=='NON_CLONE' for r in valid)
    output.mkdir(parents=True,exist_ok=False)
    result={'status':'F01C_V18_LABEL_REVIEW_ALL_NON_CLONE' if all_pass else 'F01C_V18_LABEL_REVIEW_BLOCKED',
            'version':'v18','created_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
            'reviewer':reviewer,'prepared_sha256':hashes,'decisions_sha256':sha(decisions_path.read_bytes()),
            'selected_pairs':len(valid),'non_clone_count':sum(r['decision']=='NON_CLONE' for r in valid),
            'uncertain_or_clone_count':sum(r['decision']!='NON_CLONE' for r in valid),
            'label_review_complete':True,'approved_for_training':False,
            'note':'This verifies human review record integrity; DOES NOT authorize GPU training or modify prepared files.'}
    exclusive_json(output/'label_review_verification.json',result)
    print(json.dumps({'status':result['status'],'report':str(output/'label_review_verification.json'),'approved_for_training':False}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('export','verify'))
    p.add_argument('--prepared-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--decisions',type=Path,required=True)
    for n in ('expected-train-sha256','expected-val-sha256','expected-manifest-sha256'):
        p.add_argument('--'+n,required=True)
    p.add_argument('--expected-pairs',type=int,required=True)
    p.add_argument('--max-tokens',type=int,required=True)
    a=p.parse_args()
    if a.expected_pairs<=0 or a.max_tokens<=0:raise ValueError('Invalid shell-configured bounds')
    if a.mode=='export':produce(a.prepared_run.resolve(),a.output.resolve(),a.expected_train_sha256,a.expected_val_sha256,a.expected_manifest_sha256,a.expected_pairs,a.max_tokens)
    else:verify(a.prepared_run.resolve(),a.decisions.resolve(),a.output.resolve(),a.expected_train_sha256,a.expected_val_sha256,a.expected_manifest_sha256,a.expected_pairs,a.max_tokens)

if __name__=='__main__':main()
