#!/usr/bin/env python3
"""F-01c v1 M2 mixed-source LoRA: independent multi-LLM core.

Model choices, all experiment parameters, paths, budgets, seeding, scoring IDs
and training settings are exclusively Shell -> required CLI. No silent defaults.
"""
import hashlib
import json
import os
import platform
import random
import re
from collections import Counter
from pathlib import Path

import f01c_shared as s
from f01c_variants import TokenPolicy, checked_training_tokens, resolve_answer_ids

VERSION = 'v1'
CORPORA = ('java_source', 'java_mixed', 'python_mixed')


def all_paths(a):
    return {'java_source': a.source_dir, 'java_mixed': a.java_target_dir,
            'python_mixed': a.python_target_dir}


def arg_validate(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != a.gpu_ids:
        raise ValueError('Shell GPU list differs from CUDA_VISIBLE_DEVICES')
    if a.model_mode not in ('starcoder2-7b','starcoder2-15b','codellama-7b'):
        raise ValueError('Unapproved model mode')
    expected = {'starcoder2-7b':('starcoder2','standalone-one-token'),
                'starcoder2-15b':('starcoder2','standalone-one-token'),
                'codellama-7b':('llama','contextual-one-token')}
    family, policy = expected[a.model_mode]
    if (a.model_family,a.expected_model_type,a.token_policy)!=(family,family,policy):
        raise ValueError('Model family and tokenizer policy inconsistent')
    if a.m1_source_identity_policy!='strict-full-only':
        raise ValueError('M2 source 12,000 pair identity must be checked against M1')
    if a.train_order != 'shuffled' or a.prepared_policy != 'fresh-frozen-m2':
        raise ValueError('F-01c M2 requires shuffled training and immutable prepared dataset')
    if a.sample_policy != 'balanced' or a.source_corpus != 'java_source':
        raise ValueError('Unapproved balanced source dataset settings')
    if (a.source_val+a.java_target_val+a.python_target_val)!=a.max_val_examples:
        raise ValueError('Validation domain budget mismatch')
    counts=(a.max_train_examples,a.java_target_train,a.python_target_train,
            a.source_val,a.java_target_val,a.python_target_val)
    if any(v<2 or v%2 for v in counts):
        raise ValueError('All source/target train and validation counts must be positive even integers')
    if a.java_target_seed_offset<0 or a.python_target_seed_offset<0:
        raise ValueError('Domain sampling seed offsets must be >=0')
    if a.max_seq_length<256 or a.expected_world_size<1 or a.per_device_batch_size<1 or a.gradient_accumulation_steps<1:
        raise ValueError('Invalid sequence length or batch settings')
    global_batch=a.per_device_batch_size*a.gradient_accumulation_steps*a.expected_world_size
    train_total=sum(counts[:3])
    if a.purpose=='train' and train_total!=global_batch*a.max_steps:
        raise ValueError('M2 must consume exactly one full mixed epoch (train pairs == global batch * steps)')
    if a.max_steps<1 or a.eval_steps<1 or a.max_steps%a.eval_steps or not 0<=a.warmup_steps<a.max_steps:
        raise ValueError('Invalid steps/warmup, schedule must align with evaluation')
    if a.precision!='bf16' or a.lora_rank<1 or a.lora_alpha<1 or not 0<=a.lora_dropout<1:
        raise ValueError('Unapproved precision or LoRA hyperparameters')
    if a.verify_holdout_overlap!='strict':
        raise ValueError('Held-out overlap check cannot be disabled in M2')
    if a.max_overlap_examples<1 or a.max_train_examples<2 or a.max_val_examples<2:
        raise ValueError('Invalid count or overlap settings')
    if a.purpose not in ('preflight','prepare','smoke','train'):
        raise ValueError('Invalid execution purpose')
    if a.adapter_dir!=a.run_dir/'adapter' or a.prepared_dir!=a.run_dir/'prepared' or a.checkpoint_dir!=a.run_dir/'checkpoints':
        raise ValueError('Artifact paths must be under this run directory')
    if not re.fullmatch(r'[0-9a-f]{64}',a.initial_adapter_sha256):
        raise ValueError('M1 adapter SHA256 must be pinned full 64-hex')
    if len(a.dataset_c10_expected_sha256.split(':'))!=3 or any(not re.fullmatch(r'[0-9a-f]{64}',v) for v in a.dataset_c10_expected_sha256.split(':')):
        raise ValueError('C-10 corpus SHA256 pin malformed')
    if 'Language: java\n' not in a.prompt_prefix:
        raise ValueError('Prompt template lacks explicit Java language line')
    if a.source_pair_ids_sha256!='unverified' and not re.fullmatch(r'[0-9a-f]{64}',a.source_pair_ids_sha256):
        raise ValueError('Source pair identity pin invalid')
    if a.expected_yes_token!='derive' and not a.expected_yes_token.isdecimal():
        raise ValueError('YES token ID malformed')
    if a.expected_no_token!='derive' and not a.expected_no_token.isdecimal():
        raise ValueError('NO token ID malformed')
    if a.token_policy=='contextual-one-token' and ('derive' in (a.expected_yes_token,a.expected_no_token)):
        raise ValueError('CodeLlama must have pinned contextual YES/NO IDs')
    if a.greater_is_better or a.use_cache or a.save_total_limit!=0:
        raise ValueError('Unapproved checkpoint/persistence strategy')


def implementation_sha(a):
    return {p.name:s.sha256(s.require_file(p)) for p in (
        a.root/'src/py2/f-01c-train-standard-mixed-lora.py',
        a.root/'src/py2/f01c_shared.py',
        a.root/'src/py2/f01c_core.py',
        a.root/'src/py2/f01c_variants.py',
        a.root/'src/sh2/f-01c-train-standard-mixed-lora.sh')}


def audit_m1(a):
    path=Path(a.initial_adapter).resolve()
    cfg=s.require_file(path/'adapter_config.json')
    weight=s.require_file(path/'adapter_model.safetensors')
    digest=s.sha256(weight)
    if digest!=a.initial_adapter_sha256:
        raise RuntimeError(f'BLOCKED: M1 adapter SHA256 mismatch {digest} != {a.initial_adapter_sha256}')
    c=s.read_json(cfg)
    if c.get('base_model_name_or_path') not in (a.model_id,str(a.hf_home/'hub'/('models--'+a.model_id.replace('/','--'))/'snapshots'/a.model_revision)):
        raise RuntimeError('M1 adapter base model ID differs from selected LLM')
    if c.get('r')!=a.lora_rank or c.get('lora_alpha')!=a.lora_alpha or abs(float(c.get('lora_dropout',-1))-a.lora_dropout)>1e-10:
        raise RuntimeError('M1 LoRA rank/alpha/dropout mismatch')
    if set(c.get('target_modules',[]))!=set(a.target_modules.split(',')):
        raise RuntimeError('M1 LoRA targets differ from M2 planned modules')
    m=s.require_file(a.m1_manifest_file)
    obj=s.read_json(m)
    hashes=obj.get('adapter_sha256') or obj.get('adapter_files_sha256') or {}
    expected=hashes.get('adapter_model.safetensors')
    if expected is None or expected!=digest:
        raise RuntimeError('M1 manifest does not certify pinned adapter weight SHA256')
    mid=obj.get('model_id') or obj.get('model',{}).get('id')
    if mid and mid!=a.model_id:
        raise RuntimeError('M1 manifest model differs')
    rev=obj.get('model_revision') or obj.get('revision') or obj.get('model',{}).get('revision')
    if rev and rev!=a.model_revision:
        raise RuntimeError('M1 manifest revision differs')
    return {'adapter':str(path),'adapter_model_sha256':digest,'adapter_config_sha256':s.sha256(cfg),
            'm1_manifest':str(m),'m1_manifest_sha256':s.sha256(m),'m1_model_id':mid,'m1_revision':rev}


def validate_datasets(a):
    data={}; codes={}; paths={}; source_hash=set()
    for corpus, root in all_paths(a).items():
        paths[corpus]={name:s.require_file(root/name) for name in ('data.jsonl','train.txt','valid.txt')}
        codes[corpus]=s.load_codes(paths[corpus]['data.jsonl'])
        data[corpus]={name:s.sha256(path) for name,path in paths[corpus].items()}
    for key,expected in zip(('data.jsonl','train.txt','valid.txt'),a.dataset_c10_expected_sha256.split(':')):
        if data['python_mixed'][key]!=expected:
            raise RuntimeError(f'C-10 approved Python corpus mismatch: {key} expected={expected} got={data["python_mixed"][key]}')
    s.require_file(a.staging_manifest)
    if s.read_json(a.staging_manifest).get('status')!='STAGED_CLEANED_NOT_APPROVED_FOR_TRAINING':
        raise RuntimeError('Source staging manifest status changed')
    if s.sha256(s.require_file(a.ground_truth))!=a.expected_ground_truth_sha256:
        raise RuntimeError('Frozen held-out SHA256 mismatch')
    source_hash={s.code_hash(code) for code in codes['java_source'].values()}
    if a.a10_audit_file:
        auditfile=s.require_file(a.a10_audit_file)
    else:
        if a.a10_discovery!='newest-valid':raise ValueError('A10 audit path missing')
        candidates=sorted((a.root/'results/a-10').glob('*/audit.json'),key=lambda p:(p.stat().st_mtime_ns,str(p)),reverse=True)
        auditfile=next((p for p in candidates if s.read_json(p).get('status')=='EXACT_SPLIT_CHECK_PASSED_REVIEW_PENDING' and not s.read_json(p).get('blocking_findings')),None)
        if auditfile is None:raise RuntimeError('No approved A10 audit')
    audit=s.read_json(auditfile)
    if audit.get('status')!='EXACT_SPLIT_CHECK_PASSED_REVIEW_PENDING' or audit.get('test',{}).get('pair_rows')!=a.expected_a10_test_pairs or audit.get('blocking_findings'):
        raise RuntimeError('A-10 approved/test panel conditions failed')
    for name in ('train_test_function_overlap_exact','train_test_function_overlap_normalized','validation_test_function_overlap_exact','validation_test_function_overlap_normalized','train_test_pair_overlap_exact','train_test_pair_overlap_normalized','validation_test_pair_overlap_normalized'):
        if audit.get('corpora',{}).get('java_source',{}).get(name)!=0:
            raise RuntimeError(f'A-10 source leakage finding {name}')
    source_audit=audit.get('corpora',{}).get('java_source',{})
    for split,filename in (('train','train.txt'),('validation','valid.txt')):
        counts=Counter(y for _,_,y in s.pair_rows(paths['java_source'][filename],codes['java_source']))
        if sum(counts.values())!=source_audit.get(split+'_pairs'):
            raise RuntimeError(f'A-10 source {split} total differs from staged corpus')
        expected_labels=source_audit.get(split+'_labels',{})
        if {str(y):counts[y] for y in (0,1)}!={str(k):v for k,v in expected_labels.items()}:
            raise RuntimeError(f'A-10 source {split} label distribution differs')
    info={'sha256':data,'paths':{c:{k:str(v) for k,v in x.items()} for c,x in paths.items()},
          'a10_sha256':s.sha256(auditfile),'a10_path':str(auditfile),
          'staging_sha256':s.sha256(a.staging_manifest),
          'ground_truth_sha256':s.sha256(a.ground_truth)}
    return info,codes,source_hash


def signature(a, model, data, m1):
    cfg=s.configuration(a)
    for key in ('purpose','run_dir','prepared_dir','adapter_dir','checkpoint_dir','log_file','log_dir','log_prefix','log_timestamp','log_timestamp_format',
                'max_train_examples','java_target_train','python_target_train','max_val_examples','source_val','java_target_val','python_target_val',
                'max_steps','eval_steps','log_steps','warmup_steps'):
        cfg.pop(key,None)
    return s.canonical_hash({'config':cfg,'model':{'id':a.model_id,'revision':a.model_revision,'tokens':model['answer_ids']},
                             'dataset':data['sha256'],'m1_adapter':m1['adapter_model_sha256'],
                             'm1_manifest_sha256':m1['m1_manifest_sha256'],'implementation':implementation_sha(a)})


def validated(a):
    arg_validate(a)
    m1=audit_m1(a)
    data,codes,source_hash=validate_datasets(a)
    model,tokenizer=s.cached_model(a)
    sig=signature(a,model,data,m1)
    return m1,data,codes,source_hash,model,tokenizer,sig


def validate_args(a):arg_validate(a)


def run_preflight(a):
    m1,data,_,_,model,_,sig=validated(a)
    a.run_dir.mkdir(parents=True,exist_ok=False)
    report={'status':'F01C_V1_PREFLIGHT_PASS_NOT_TRAINED','version':VERSION,'created_at_chicago':s.chicago_now(),
            'configuration':s.configuration(a),'m1':m1,'dataset':data,'model':model,
            'signature':sig,'implementation_sha256':implementation_sha(a),
            'limitations':['No GPU training in preflight','Training eligibility and held-out overlap require full preparation']}
    s.exclusive_json(a.run_dir/'preflight.json',report)
    print(json.dumps({'status':report['status'],'signature':sig,'run':str(a.run_dir)}),flush=True)


def target_pool(path,codes,source_hash):
    pools={0:[],1:[]}
    fingerprints={idx:s.code_hash(body) for idx,body in codes.items()}
    for x,y,label in s.pair_rows(path,codes):
        if fingerprints[x] not in source_hash or fingerprints[y] not in source_hash:
            pools[label].append((x,y,label))
    return pools


def quotas(n):return {0:n//2,1:n//2}


def select_domain(a, corpus, split, amount, codes, source_hash, tok, answers, banned):
    path=all_paths(a)[corpus]/('train.txt' if split=='train' else 'valid.txt')
    rng=random.Random(a.seed+(0 if split=='train' else a.sample_val_seed_offset) + {'java_source':0,'java_mixed':a.java_target_seed_offset,'python_mixed':a.python_target_seed_offset}[corpus])
    if corpus=='java_source':
        pool={0:[],1:[]}
        # Deterministic stratified reservoir with oversampling capacity.
        count=Counter(); cap={y:amount//2+max(a.sample_oversample_min,amount//(2*a.sample_oversample_divisor)) for y in (0,1)}
        per={y:random.Random(a.seed+a.sample_rng_label_stride*y+(0 if split=='train' else a.sample_val_seed_offset)) for y in (0,1)}
        for row in s.pair_rows(path,codes):
            y=row[2];count[y]+=1
            if len(pool[y])<cap[y]:pool[y].append(row)
            else:
                ix=per[y].randrange(count[y])
                if ix<cap[y]:pool[y][ix]=row
        for y in (0,1):per[y].shuffle(pool[y])
    else:
        pool=target_pool(path,codes,source_hash)
        for y in (0,1):rng.shuffle(pool[y])
    result=[];accepted=Counter();overlength=Counter();overlap_skips=Counter();selected_ids=set()
    language='python' if corpus=='python_mixed' else 'java'
    for y in (0,1):
        for x,z,label in pool[y]:
            if x in banned or z in banned or x in selected_ids or z in selected_ids:
                # Endpoint-level isolation applies to validation against training; selecting train
                # may include reused endpoints, matching original D-03 selection semantics.
                if x in banned or z in banned:
                    overlap_skips[y]+=1;continue
            text=a.prompt_prefix.replace('Language: java\n',f'Language: {language}\n')+codes[x]+a.prompt_b_separator+codes[z]+a.prompt_suffix
            payload=checked_training_tokens(tok,text,a.answer_yes if y else a.answer_no,answers[y],a.max_seq_length)
            if payload is None:
                overlength[y]+=1;continue
            result.append({'corpus':corpus,'id1':x,'id2':z,'label':label,'tokens':len(payload['input_ids'])})
            selected_ids.update((x,z));accepted[y]+=1
            if accepted[y]==amount//2:break
        if accepted[y]!=amount//2:
            raise RuntimeError(f'Insufficient token eligible {corpus}/{split} label={y}: {accepted[y]}/{amount//2}; overlength={overlength[y]}')
    return result,{'count':len(result),'labels':dict(accepted),'overlength_skipped':dict(overlength),
                   'train_endpoint_overlap_skipped':dict(overlap_skips),'raw_target_candidate_count':{str(y):len(pool[y]) for y in (0,1)}}


def selected_data(a,codes,source_hash,tok,answers):
    quantities={'train':{'java_source':a.max_train_examples,'java_mixed':a.java_target_train,'python_mixed':a.python_target_train},
                'validation':{'java_source':a.source_val,'java_mixed':a.java_target_val,'python_mixed':a.python_target_val}}
    selected={};stats={}; train_ids={c:set() for c in CORPORA}
    for split in ('train','validation'):
        result=[];stats[split]={}
        for corpus in CORPORA:
            rows,meta=select_domain(a,corpus,split,quantities[split][corpus],codes[corpus],source_hash,tok,answers,
                                    train_ids[corpus] if split=='validation' else set())
            result.extend(rows);stats[split][corpus]=meta
            if split=='train':
                for x in rows:train_ids[corpus].update((x['id1'],x['id2']))
        random.Random(a.seed+(0 if split=='train' else a.sample_val_seed_offset)+a.sample_order_seed_offset).shuffle(result)
        selected[split]=result
        stats[split]['total']=len(result)
        stats[split]['pair_ids_sha256']=s.canonical_hash([(x['corpus'],x['id1'],x['id2'],x['label']) for x in result])
    if len(selected['train'])!=sum(quantities['train'].values()) or len(selected['validation'])!=a.max_val_examples:
        raise RuntimeError('Mixed train/validation count mismatch')
    return selected,stats


def check_heldout(a,selected,codes):
    heldout,detail=s.read_eval_fingerprints(a)
    failures=[]
    for split,rows in selected.items():
        for row in rows:
            for key in ('id1','id2'):
                if s.code_hash(codes[row['corpus']][row[key]]) in heldout:
                    failures.append((split,row['corpus'],row[key]))
                    if len(failures)>=a.max_overlap_examples:
                        raise RuntimeError('BLOCKED held-out overlap: '+repr(failures))
    if failures:raise RuntimeError('BLOCKED held-out normalized function overlap: '+repr(failures))
    return {'status':'PASS','overlap_count':0,'heldout_code':detail}


def require_smoke(a,sig):
    home=a.root/'results/f-01c'/a.model_mode
    for file in reversed(sorted(home.glob('*-smoke/smoke_success.json'))):
        try:
            obj=s.read_json(file)
            if obj.get('status')=='F01C_V1_SMOKE_PASS' and obj.get('compatibility_signature')==sig and obj.get('initial_adapter_sha256')==a.initial_adapter_sha256:
                # Revalidate recorded adapter integrity, not merely presence of a JSON proof.
                ad=Path(obj['adapter'])
                hashes=obj.get('adapter_sha256',{})
                if {'adapter_config.json','adapter_model.safetensors'}<=set(hashes) and all(s.sha256(s.require_file(ad/name))==digest for name,digest in hashes.items()):
                    return str(file)
        except (ValueError,OSError,KeyError,TypeError):continue
    raise RuntimeError('No matching F-01c smoke proof, execute --smoke before --train')


def check_m1_source_identity(a,selected):
    # Smoke intentionally uses 48 source examples; full M2 must preserve M1 source IDs.
    if a.purpose=='smoke':
        return {'status':'SMOKE_NOT_APPLICABLE'}
    path=s.require_file(Path(a.m1_manifest_file).parent/'prepared/train.jsonl')
    original=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    initial={(row['id1'],row['id2'],int(row['label'])) for row in original}
    current={(row['id1'],row['id2'],int(row['label'])) for row in selected['train'] if row['corpus']=='java_source'}
    if len(original)!=a.max_train_examples or len(initial)!=a.max_train_examples:
        raise RuntimeError('M1 source selected 12,000 pairs missing or duplicated')
    if initial!=current:
        raise RuntimeError(f'BLOCKED: M2 source pairs differ from M1 Source Adapter training data: missing={len(initial-current)} extra={len(current-initial)}')
    return {'status':'PASS','m1_prepared_train_path':str(path),'m1_prepared_train_sha256':s.sha256(path),
            'm1_source_pair_set_sha256':s.canonical_hash(sorted(initial)),'source_pair_count':len(initial)}


def run_prepare(a):
    m1,data,codes,source_hash,model,tok,sig=validated(a)
    smoke=require_smoke(a,sig) if a.purpose=='train' else None
    selected,stats=selected_data(a,codes,source_hash,tok,model['answer_ids'])
    leakage=check_heldout(a,selected,codes)
    m1_source_identity=check_m1_source_identity(a,selected)
    if a.source_pair_ids_sha256!='unverified' and a.source_pair_ids_sha256!=stats['train']['pair_ids_sha256']:
        raise RuntimeError('Pinned pair identity differs from current M2 prepared dataset')
    a.run_dir.mkdir(parents=True,exist_ok=False)
    a.prepared_dir.mkdir(exist_ok=False)
    for split in ('train','validation'):
        file=a.prepared_dir/(split+'.jsonl')
        with file.open('x',encoding='utf-8') as f:
            for row in selected[split]:f.write(json.dumps(row,sort_keys=True)+'\n')
        stats[split]['sha256']=s.sha256(file)
    manifest={'status':'PREPARED_NOT_TRAINED','version':VERSION,'task':'F-01c M2 Standard Mixed LoRA from frozen M1',
        'created_at_chicago':s.chicago_now(),'model':model,'initial_m1':m1,'dataset':data,
        'configuration':s.configuration(a),'training_plan':{'order':'standard-mixed-shuffled','train':{'java_source':a.max_train_examples,'java_mixed':a.java_target_train,'python_mixed':a.python_target_train},
                          'validation':{'java_source':a.source_val,'java_mixed':a.java_target_val,'python_mixed':a.python_target_val},
                          'global_batch':a.expected_world_size*a.per_device_batch_size*a.gradient_accumulation_steps,'steps':a.max_steps},
        'sample':stats,'heldout_overlap':leakage,'m1_source_identity':m1_source_identity,'answer_ids':{str(k):v for k,v in model['answer_ids'].items()},
        'compatibility_signature':sig,'implementation_sha256':implementation_sha(a),'validated_smoke':smoke,
        'limitations':['Negative candidates in C-10 corpus may be unadjudicated','Project-level independence not established',
                       'M3 MUST use byte-identical prepared files to compare order only',
                       'Model tokenization may prevent exact Pair ID matching across different LLMs; hashes are recorded']}
    s.exclusive_json(a.prepared_dir/'manifest.json',manifest)
    print(json.dumps({'status':'F01C_V1_PREPARED_NOT_TRAINED','run':str(a.run_dir),'train':len(selected['train']),
                      'validation':len(selected['validation']),'train_pair_ids_sha256':stats['train']['pair_ids_sha256'],
                      'manifest_sha256':s.sha256(a.prepared_dir/'manifest.json')}),flush=True)


class MixedDataset:
    def __init__(self,file,codes,tokenizer,answers,a):
        self.rows=[json.loads(x) for x in file.read_text(encoding='utf-8').splitlines() if x.strip()]
        self.codes=codes;self.tokenizer=tokenizer;self.answers=answers;self.a=a;self.cache={}
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        if i not in self.cache:
            row=self.rows[i]; corpus=row['corpus'];lang='python' if corpus=='python_mixed' else 'java'
            text=self.a.prompt_prefix.replace('Language: java\n',f'Language: {lang}\n')+self.codes[corpus][row['id1']]+self.a.prompt_b_separator+self.codes[corpus][row['id2']]+self.a.prompt_suffix
            data=checked_training_tokens(self.tokenizer,text,self.a.answer_yes if row['label'] else self.a.answer_no,
                                         self.answers[row['label']],self.a.max_seq_length)
            if data is None or len(data['input_ids'])!=row['tokens']:
                raise RuntimeError(f'Prepared pair token drift index={i}')
            self.cache[i]=data
        return self.cache[i]


def verify_prepared(a):
    path=s.require_file(a.prepared_dir/'manifest.json');m=s.read_json(path)
    if m.get('status')!='PREPARED_NOT_TRAINED':raise RuntimeError('Invalid M2 prepared manifest')
    if m['implementation_sha256']!=implementation_sha(a):raise RuntimeError('Implementation changed since dataset preparation')
    baseline=m['configuration'].copy(); actual=s.configuration(a)
    if a.command=='verify':
        for field in ('log_file','log_timestamp'):
            baseline.pop(field,None);actual.pop(field,None)
    if baseline!=actual:raise RuntimeError('Shell/CLI values differ from prepared manifest: '+repr([k for k in set(baseline)|set(actual) if baseline.get(k)!=actual.get(k)]))
    if m['initial_m1']!=audit_m1(a):raise RuntimeError('M1 Adapter was replaced or modified')
    if s.sha256(s.require_file(a.staging_manifest))!=m['dataset']['staging_sha256']:
        raise RuntimeError('Source staging manifest drift')
    if s.sha256(s.require_file(m['dataset']['a10_path']))!=m['dataset']['a10_sha256']:
        raise RuntimeError('A-10 leakage audit drift')
    if s.sha256(s.require_file(a.ground_truth))!=m['dataset']['ground_truth_sha256']:
        raise RuntimeError('Held-out ground-truth drift')
    for c,files in m['dataset']['paths'].items():
        for name,p in files.items():
            if s.sha256(s.require_file(p))!=m['dataset']['sha256'][c][name]:raise RuntimeError(f'Dataset drift {c}/{name}')
    if m.get('m1_source_identity',{}).get('status')=='PASS':
        check=s.require_file(m['m1_source_identity']['m1_prepared_train_path'])
        if s.sha256(check)!=m['m1_source_identity']['m1_prepared_train_sha256']:
            raise RuntimeError('M1 original selected source samples changed')
    for split in ('train','validation'):
        if s.sha256(s.require_file(a.prepared_dir/(split+'.jsonl')))!=m['sample'][split]['sha256']:
            raise RuntimeError('Prepared selected pairs changed')
    return m,path


def run_worker(a):
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
    from peft import PeftModel
    world=int(os.environ.get('WORLD_SIZE','-1'));local=int(os.environ.get('LOCAL_RANK','-1'))
    if world!=a.expected_world_size or not 0<=local<world:raise RuntimeError('torchrun DDP world/rank mismatch')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA/BF16 unavailable')
    torch.cuda.set_device(local);torch.manual_seed(a.seed);random.seed(a.seed)
    m,manifest_path=verify_prepared(a)
    if a.adapter_dir.exists() or (a.run_dir/'training_summary.json').exists():raise FileExistsError('Refusing overwrite of M2 adapter')
    cached=Path(m['model']['snapshot'])
    tokenizer=AutoTokenizer.from_pretrained(str(cached),local_files_only=True,trust_remote_code=False,use_fast=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    ans,_=resolve_answer_ids(tokenizer,s.prompt(a,a.probe_code_a,a.probe_code_b),
         TokenPolicy(a.token_policy,a.answer_yes,a.answer_no,a.expected_yes_token,a.expected_no_token))
    if {str(k):v for k,v in ans.items()}!=m['answer_ids']:raise RuntimeError('Answer Token drift')
    codes={c:s.load_codes(Path(m['dataset']['paths'][c]['data.jsonl'])) for c in CORPORA}
    train=MixedDataset(a.prepared_dir/'train.jsonl',codes,tokenizer,ans,a)
    val=MixedDataset(a.prepared_dir/'validation.jsonl',codes,tokenizer,ans,a)
    for dataset in (train,val):
        for i in range(len(dataset)):dataset[i]
    if len(train)!=a.max_train_examples+a.java_target_train+a.python_target_train or len(val)!=a.max_val_examples:
        raise RuntimeError('Prepared lengths differ')
    base=AutoModelForCausalLM.from_pretrained(str(cached),local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,low_cpu_mem_usage=True)
    if base.config.model_type!=a.expected_model_type:raise RuntimeError('Model architecture drift')
    base.config.use_cache=a.use_cache
    model=PeftModel.from_pretrained(base,a.initial_adapter,is_trainable=True,local_files_only=True)
    if not any(p.requires_grad for p in model.parameters()):raise RuntimeError('M1 adapter not trainable')
    if hasattr(model,'enable_input_require_grads'):model.enable_input_require_grads()
    targs=s.make_training_arguments(a,a.checkpoint_dir)
    import inspect
    kwargs=dict(model=model,args=targs,train_dataset=train,eval_dataset=val,
                data_collator=lambda batch:s.collate(batch,tokenizer.pad_token_id))
    signature=inspect.signature(Trainer.__init__).parameters
    if 'processing_class' in signature:kwargs['processing_class']=tokenizer
    elif 'tokenizer' in signature:kwargs['tokenizer']=tokenizer
    else:raise RuntimeError('Unsupported Trainer tokenizer API')
    trainer=Trainer(**kwargs)
    if local==0:
        print(f'[F-01c v1] M2 standard mixed DDP world={world} train={len(train)} val={len(val)} steps={a.max_steps} initial M1={a.initial_adapter}',flush=True)
        model.print_trainable_parameters()
    outcome=trainer.train()
    if trainer.state.global_step!=a.max_steps:raise RuntimeError('Training did not finish expected steps')
    trainer.save_model(str(a.adapter_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(a.adapter_dir)
        s.exclusive_json(a.run_dir/'training_history.json',trainer.state.log_history)
        adapter_files={p.name:s.sha256(p) for p in a.adapter_dir.iterdir() if p.is_file()}
        for needed in ('adapter_model.safetensors','adapter_config.json'):
            if needed not in adapter_files:raise RuntimeError('Missing final LoRA adapter component '+needed)
        summary={'version':VERSION,'status':'TRAINING_COMPLETE','finished_at_chicago':s.chicago_now(),
                 'model_mode':a.model_mode,'model_id':a.model_id,'revision':a.model_revision,
                 'global_step':trainer.state.global_step,'train_loss':outcome.training_loss,
                 'best_checkpoint':trainer.state.best_model_checkpoint,'best_metric':trainer.state.best_metric,
                 'initial_m1_sha256':m['initial_m1']['adapter_model_sha256'],
                 'prepared_manifest_sha256':s.sha256(manifest_path),'compatibility_signature':m['compatibility_signature'],
                 'effective_configuration':s.configuration(a),'trainer_arguments':targs.to_dict(),
                 'adapter_sha256':adapter_files,
                 'versions':{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__}}
        s.exclusive_json(a.run_dir/'training_summary.json',summary)
        print('[F-01c v1] TRAINING_COMPLETE adapter='+str(a.adapter_dir),flush=True)
    if torch.distributed.is_initialized():torch.distributed.barrier();torch.distributed.destroy_process_group()


def run_finalize(a):
    m,path=verify_prepared(a)
    summ=s.read_json(s.require_file(a.run_dir/'training_summary.json'))
    if summ.get('status')!='TRAINING_COMPLETE' or summ.get('global_step')!=a.max_steps:
        raise RuntimeError('Incomplete full M2 training')
    if summ['prepared_manifest_sha256']!=s.sha256(path) or summ['compatibility_signature']!=m['compatibility_signature']:
        raise RuntimeError('Summary/Prepared Manifest signature drift')
    for name,digest in summ['adapter_sha256'].items():
        if s.sha256(s.require_file(a.adapter_dir/name))!=digest:raise RuntimeError('Adapter weight/checksum mismatch '+name)
    if summ['initial_m1_sha256']!=m['initial_m1']['adapter_model_sha256']:
        raise RuntimeError('M1 adapter SHA changed')
    status='F01C_V1_SMOKE_PASS' if a.purpose=='smoke' else 'F01C_V1_M2_TRAINING_COMPLETE'
    rec={'version':VERSION,'status':status,'created_at_chicago':s.chicago_now(),
         'model_id':a.model_id,'model_revision':a.model_revision,'model_mode':a.model_mode,
         'initial_adapter_sha256':m['initial_m1']['adapter_model_sha256'],
         'initial_adapter':m['initial_m1']['adapter'],
         'compatibility_signature':m['compatibility_signature'],'prepared_manifest_sha256':s.sha256(path),
         'training_summary_sha256':s.sha256(a.run_dir/'training_summary.json'),
         'adapter_sha256':summ['adapter_sha256'],'adapter':str(a.adapter_dir),'run':str(a.run_dir),
         'train_pair_ids_sha256':m['sample']['train']['pair_ids_sha256'],
         'validation_pair_ids_sha256':m['sample']['validation']['pair_ids_sha256'],
         'config':s.configuration(a)}
    final=a.run_dir/('smoke_success.json' if a.purpose=='smoke' else 'f01c_m2_manifest.json')
    s.exclusive_json(final,rec)
    print(json.dumps({'status':status,'manifest':str(final),'signature':m['compatibility_signature']}),flush=True)


def run_verify(a):
    m,path=verify_prepared(a)
    result=a.run_dir/('smoke_success.json' if a.purpose=='smoke' else 'f01c_m2_manifest.json')
    r=s.read_json(s.require_file(result))
    if r['prepared_manifest_sha256']!=s.sha256(path) or r['compatibility_signature']!=m['compatibility_signature']:
        raise RuntimeError('Prepared identity mismatch')
    if r['training_summary_sha256']!=s.sha256(s.require_file(a.run_dir/'training_summary.json')):
        raise RuntimeError('Training summary hash mismatch')
    for name,digest in r['adapter_sha256'].items():
        if s.sha256(s.require_file(a.adapter_dir/name))!=digest:raise RuntimeError('Adapter hash mismatch '+name)
    print(json.dumps({'status':'F01C_V1_VERIFY_PASS','run':str(a.run_dir),'adapter':str(a.adapter_dir)}),flush=True)
