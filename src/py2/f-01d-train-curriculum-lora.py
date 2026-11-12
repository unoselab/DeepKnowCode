#!/usr/bin/env python3
"""F-01d v1: immutable M2 input verification and D-04 curriculum feasibility audit.

No model training or difficulty-score selection is performed in v1.
"""
import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION='v1'

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def now():return datetime.now(ZoneInfo('America/Chicago')).isoformat()

def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','audit'])
    for x in ['root','run-dir','prepared-run','m1-adapter','log-file']:
        p.add_argument('--'+x,required=True,type=Path)
    for x in ['train-sha256','validation-sha256','manifest-sha256','m1-sha256','model-id','model-revision','model-mode','gpu-ids','precision','difficulty-policy','score-input-policy','log-prefix','log-timestamp-format']:
        p.add_argument('--'+x,required=True)
    for x in ['seed','world-size','max-seq-length','global-batch','optimizer-steps','train-count','validation-count','source-count','java-target-count','python-target-count']:
        p.add_argument('--'+x,required=True,type=int)
    for x in ['phase-names']:
        p.add_argument('--'+x,nargs=3,required=True)
    for x in ['phase-steps','phase-source','phase-target']:
        p.add_argument('--'+x,nargs=3,required=True,type=int)
    return p

def validate(a):
    if a.difficulty_policy!='PENDING_D04_REDUCED_4D_SCORE_AUDIT' or a.score_input_policy!='required_before_gpu_training':
        raise ValueError('Unverified difficulty-scoring policy cannot be silently accepted')
    if a.model_id!='meta-llama/CodeLlama-7b-hf' or a.model_mode!='codellama-7b':raise ValueError('Unexpected model')
    if a.world_size!=len(a.gpu_ids.split(',')) or a.global_batch%a.world_size:raise ValueError('GPU/global batch mismatch')
    if a.optimizer_steps*a.global_batch!=a.train_count:raise ValueError('Training budget mismatch')
    if sum(a.phase_steps)!=a.optimizer_steps:raise ValueError('Phase step budget mismatch')
    if sum(a.phase_source)!=a.source_count or sum(a.phase_target)!=a.java_target_count+a.python_target_count:raise ValueError('Phase corpus budget mismatch')
    for i in range(3):
        if (a.phase_source[i]+a.phase_target[i])!=a.phase_steps[i]*a.global_batch:raise ValueError('Phase step/sample count mismatch')
        if a.phase_source[i]%2 or a.phase_target[i]%4:raise ValueError('Phase cannot balance corpus/label counts')
    if a.max_seq_length<1:raise ValueError('Invalid token limit')
    for name in ['train_sha256','validation_sha256','manifest_sha256','m1_sha256']:
        x=getattr(a,name)
        if len(x)!=64 or any(c not in '0123456789abcdef' for c in x):raise ValueError('Invalid pinned SHA256 '+name)

def inventory(file, max_tokens):
    counts=Counter(); pairs=set(); maxseen=0; token_over=0; n=0
    with file.open(encoding='utf-8') as f:
        for lineno,line in enumerate(f,1):
            x=json.loads(line);corpus=x['corpus'];label=x['label'];tokens=x['tokens']
            if corpus not in ('java_source','java_mixed','python_mixed') or label not in (0,1):raise ValueError('Unexpected corpus/label')
            if not isinstance(tokens,int) or tokens<1:raise ValueError('Invalid token count')
            maxseen=max(maxseen,tokens)
            if tokens>max_tokens:token_over+=1
            key=(corpus,str(x['id1']),str(x['id2']),label)
            if key in pairs:raise ValueError('Duplicate pair identity')
            pairs.add(key);counts[f'{corpus}/{label}']+=1;n+=1
    if token_over:raise ValueError(f'Token cap violated by {token_over} pairs')
    return {'total':n,'corpus_label_counts':dict(sorted(counts.items())),'maximum_tokens':maxseen,'duplicate_pairs':0,'over_token_limit':token_over}

def check(a):
    validate(a)
    prepared=a.prepared_run/'prepared'
    pinned={'train.jsonl':a.train_sha256,'validation.jsonl':a.validation_sha256,'manifest.json':a.manifest_sha256}
    for name,digest in pinned.items():
        actual=sha256(prepared/name)
        if actual!=digest:raise ValueError(f'FROZEN_SHA256_MISMATCH {name}: {actual}')
    if sha256(a.m1_adapter)!=a.m1_sha256:raise ValueError('FROZEN_M1_ADAPTER_HASH_MISMATCH')
    m=json.loads((prepared/'manifest.json').read_text())
    if m.get('sample',{}).get('train',{}).get('sha256')!=a.train_sha256 or m['sample']['validation']['sha256']!=a.validation_sha256:raise ValueError('Manifest links wrong files')
    train=inventory(prepared/'train.jsonl',a.max_seq_length)
    val=inventory(prepared/'validation.jsonl',a.max_seq_length)
    if train['total']!=a.train_count or val['total']!=a.validation_count:raise ValueError('Frozen sample count mismatch')
    expected={'java_source':a.source_count,'java_mixed':a.java_target_count,'python_mixed':a.python_target_count}
    for c,n in expected.items():
        for lab in [0,1]:
            if train['corpus_label_counts'].get(f'{c}/{lab}')!=n//2:raise ValueError('Frozen corpus/label balance mismatch')
    return pinned,train,val

def atomic_create(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,sort_keys=True,ensure_ascii=False);f.write('\n')

def main():
    a=args_parser().parse_args()
    pinned,train,val=check(a)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='command'}
    phases=[]
    for i,name in enumerate(a.phase_names):
        n=a.phase_source[i];t=a.phase_target[i]
        phases.append({'name':name,'steps':a.phase_steps[i],'total':n+t,
         'java_source':{'label_0':n//2,'label_1':n//2},
         'java_mixed':{'label_0':t//4,'label_1':t//4},
         'python_mixed':{'label_0':t//4,'label_1':t//4}})
    report={'version':VERSION,'status':'F01D_V1_CURRICULUM_FEASIBILITY_PASS_ORDER_PENDING',
       'created_at_chicago':now(),'approved_for_gpu_training':False,
       'frozen_sha256':pinned,'initial_m1_adapter_sha256':a.m1_sha256,
       'training_inventory':train,'validation_inventory':val,
       'phases':phases,'difficulty_policy':a.difficulty_policy,
       'note':'This establishes feasible phase budgets only. Does not select a verified Reduced-4D score or implement DDP curriculum order.',
       'effective_config':config}
    if a.command=='prepare':
        a.run_dir.mkdir(parents=True,exist_ok=False)
        atomic_create(a.run_dir/'curriculum_feasibility.json',report)
    else:
        prev=json.loads((a.run_dir/'curriculum_feasibility.json').read_text())
        for k in ['frozen_sha256','initial_m1_adapter_sha256','training_inventory','validation_inventory','phases','difficulty_policy']:
            if prev.get(k)!=report[k]:raise ValueError('Prepared curriculum audit drift: '+k)
    print(json.dumps({'status':report['status'],'run':str(a.run_dir),'phase_steps':a.phase_steps,
       'train':train['total'],'val':val['total'],'gpu_training_enabled':False}),flush=True)

if __name__=='__main__':main()
