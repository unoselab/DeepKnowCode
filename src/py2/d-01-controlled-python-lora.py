#!/usr/bin/env python3
"""D-01: independently gated Python clone-classification LoRA training."""
import argparse
import inspect
import collections
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MODEL = 'bigcode/starcoder2-7b'
REVISION = 'bb9afde76d7945da5745592525db122d4d729eb1'
EXPECTED = {'train': {'1':10877,'0':10866}, 'valid': {'1':2719,'0':2718}}
FILES = ('data.jsonl','train.txt','valid.txt','negative_candidates.csv')


def timestamp():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,sort_keys=True)+'\n',encoding='utf-8')
    tmp.replace(path)


def choose_approval(root, explicit):
    if explicit is not None:
        file=(explicit if explicit.is_absolute() else root/explicit)
        if file.is_dir(): file=file/'training_decision.json'
        return file
    candidates=list((root/'results/c-11b').glob('*/training_decision.json'))
    approved=[]
    for f in candidates:
        try:
            x=load_json(f)
            if x.get('status')=='APPROVED_FOR_SCOPE_LIMITED_CONTROLLED_TRAINING' and x.get('training_approval') is True:
                approved.append(f)
        except (ValueError,OSError): pass
    if not approved:
        raise ValueError('No C-11b v2 approved training_decision.json; set --approval')
    return max(approved,key=lambda p:p.parent.name)


def verify_approval(root, dataset, approval):
    if not approval.is_file():raise ValueError(f'Approval missing: {approval}')
    d=load_json(approval)
    if d.get('status')!='APPROVED_FOR_SCOPE_LIMITED_CONTROLLED_TRAINING' or d.get('training_approval') is not True:
        raise ValueError('Training approval is not granted')
    if d.get('approval_scope')!='SCOPE_LIMITED_EXPLORATORY_TRAINING_ONLY':
        raise ValueError('Unexpected approval scope')
    if d.get('scope_limitations_accepted') is not True or d.get('negative_label_risk_accepted') is not True:
        raise ValueError('Residual risk has not been explicitly accepted')
    reviewer=str(d.get('reviewer') or '').strip()
    if reviewer.upper() in ('','YOUR_REVIEWER_ID','ACTUAL_REVIEWER_ID'):
        raise ValueError('An actual reviewer is required')
    original=d.get('promoted_dataset_verification',{})
    if original.get('status')!='PASS':raise ValueError('Promoted dataset was not verified')
    ref=original.get('files') or {}
    hashes={}
    for name in FILES:
        item=ref.get(name) or {}
        expected=item.get('sha256_promoted')
        if item.get('status')!='PASS' or not isinstance(expected,str) or len(expected)!=64:
            raise ValueError(f'Approval hash missing/failed: {name}')
        path=dataset/name
        if not path.is_file():raise ValueError(f'Dataset file missing: {path}')
        value=digest(path)
        if value!=expected:raise ValueError(f'Dataset SHA256 mismatch: {name}')
        hashes[name]=value
    counts=d.get('dataset_counts',{}).get('by_split',{})
    for split in EXPECTED:
        if any(int(counts.get(split,{}).get({'0':'negative','1':'positive'}[label],-1)) != count
               for label,count in EXPECTED[split].items()):
            raise ValueError('Approved counts mismatch D-01 expected baseline')
    return d,hashes


def load_corpus(path):
    data={}
    with path.open(encoding='utf-8-sig') as f:
        for line_no,line in enumerate(f,1):
            if not line.strip():continue
            ob=json.loads(line)
            idx=ob.get('idx'); code=ob.get('func')
            if not isinstance(idx,str) or not idx or not isinstance(code,str) or not code.strip() or idx in data:
                raise ValueError(f'Invalid code record {path}:{line_no}')
            data[idx]=code
    return data


def load_pairs(path,corpus):
    rows=[];seen=set();labels=collections.Counter()
    with path.open(encoding='utf-8-sig') as f:
        for line_no,line in enumerate(f,1):
            if not line.strip():continue
            parts=line.split()
            if len(parts)!=3:raise ValueError(f'Invalid pair columns: {path}:{line_no}')
            a,b,y=parts
            if y not in ('0','1') or a==b or a not in corpus or b not in corpus:
                raise ValueError(f'Invalid pair IDs/label: {path}:{line_no}')
            key=tuple(sorted((a,b)))
            if key in seen:raise ValueError(f'Duplicate pair: {path}:{line_no}')
            seen.add(key);rows.append((a,b,int(y)));labels[y]+=1
    return rows,dict(labels),seen


def verify_dataset(dataset):
    corpus=load_corpus(dataset/'data.jsonl')
    train,tc,tkeys=load_pairs(dataset/'train.txt',corpus)
    valid,vc,vkeys=load_pairs(dataset/'valid.txt',corpus)
    if tkeys & vkeys:raise ValueError('Train/valid pair identity overlap')
    for split,counts in [('train',tc),('valid',vc)]:
        if counts!=EXPECTED[split]:raise ValueError(f'Unexpected {split} label distribution: {counts}')
    tr_ids={x for a,b,_ in train for x in (a,b)}
    va_ids={x for a,b,_ in valid for x in (a,b)}
    if tr_ids & va_ids:raise ValueError('Train/valid function ID overlap')
    hashes=lambda ids:{hashlib.sha256(re.sub(r'\s+','',corpus[i]).encode()).hexdigest() for i in ids}
    if hashes(tr_ids)&hashes(va_ids):raise ValueError('Train/valid normalized code overlap')
    return corpus,{'train':train,'valid':valid},{'train':tc,'valid':vc}


def make_prompt(a,b):
    return ('Determine whether the following two code fragments are code clones. '
            'Answer YES if they are clones and NO otherwise.\n'
            f'Language: python\nCode A:\n{a}\nCode B:\n{b}\nAnswer:')


def tokenizer_for(args):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.model_id,revision=args.revision,
                                       trust_remote_code=False,local_files_only=not args.allow_download)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    return tok


def prepare(args,run,corpus,pairs,approval_file,hashes):
    tok=tokenizer_for(args)
    answers={}
    for y,word in [(0,' NO'),(1,' YES')]:
        ids=tok.encode(word,add_special_tokens=False)
        if len(ids)!=1:raise ValueError(f'Answer token must be single-token: {word}: {ids}')
        answers[y]=ids[0]
    root=run/'prepared'
    if root.exists():raise ValueError(f'Prepared directory already exists: {root}')
    root.mkdir(parents=True)
    output={};counts={}
    for split,rows in pairs.items():
        rejected=collections.Counter();accepted=collections.Counter()
        out=root/f'{split}.jsonl'
        with out.open('w',encoding='utf-8') as f:
            for a,b,y in rows:
                ids=tok.encode(make_prompt(corpus[a],corpus[b]),add_special_tokens=False)
                if not ids or len(ids)+1>args.max_seq_length:
                    rejected[str(y)]+=1;continue
                record={'input_ids':ids+[answers[y]],'label':int(y),'id_a':a,'id_b':b}
                f.write(json.dumps(record,separators=(',',':'))+'\n')
                accepted[str(y)]+=1
        if not accepted['0'] or not accepted['1']:
            raise ValueError(f'No usable pairs for one class in {split}')
        output[split]=str(out)
        counts[split]={'accepted':dict(accepted),'over_length_rejected':dict(rejected)}
    info={'created_at_chicago':timestamp(),'status':'PREPARED_NOT_TRAINED','model_id':args.model_id,
          'revision':args.revision,'approval_path':str(approval_file),'approval_sha256':digest(approval_file),
          'dataset_hashes':hashes,'max_seq_length':args.max_seq_length,'pairs':counts,
          'prepared_sha256':{f'{s}.jsonl':digest(root/f'{s}.jsonl') for s in pairs},
          'limitations':['Unreviewed negatives remain','Near-duplicate and project-disjoint independence not established']}
    dump_json(root/'manifest.json',info)
    print('D-01 PREPARED',json.dumps(counts,sort_keys=True),'OUTPUT:',root,flush=True)


def train(args,run,approval_file,hashes):
    prepared=run/'prepared'
    manifest=load_json(prepared/'manifest.json')
    if manifest['approval_sha256']!=digest(approval_file) or manifest['dataset_hashes']!=hashes:
        raise ValueError('Approval or dataset changed after preparation')
    for name,value in manifest['prepared_sha256'].items():
        if digest(prepared/name)!=value:raise ValueError(f'Prepared data hash changed: {name}')
    if (run/'adapter').exists():raise ValueError('Adapter output already exists')
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, Trainer,TrainingArguments,
                              DataCollatorForSeq2Seq,BitsAndBytesConfig)
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,PeftModel,TaskType
    if not torch.cuda.is_available():raise RuntimeError('Training requires CUDA GPU')
    world=int(os.environ.get('WORLD_SIZE','1'))
    if world!=args.expected_world_size:raise ValueError(f'Expected {args.expected_world_size} DDP workers; got {world}')
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    def load_prepared(name):
        items=[]
        for line in (prepared/f'{name}.jsonl').open(encoding='utf-8'):
            ob=json.loads(line); items.append({'input_ids':ob['input_ids'],
                 'attention_mask':[1]*len(ob['input_ids']),
                 'labels':[-100]*(len(ob['input_ids'])-1)+[ob['input_ids'][-1]]})
        return Dataset.from_list(items)
    train_ds=load_prepared('train'); val_ds=load_prepared('valid')
    tok=tokenizer_for(args)
    use_bf16=torch.cuda.is_bf16_supported()
    qconf=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16)
    model=AutoModelForCausalLM.from_pretrained(args.model_id,revision=args.revision,
            quantization_config=qconf,device_map={'':local_rank},
            trust_remote_code=False,local_files_only=not args.allow_download)
    model.config.use_cache=False
    model=prepare_model_for_kbit_training(model)
    if args.initial_adapter:
        initial=Path(args.initial_adapter).expanduser()
        if not initial.is_dir():raise ValueError(f'Missing initial LoRA adapter: {initial}')
        model=PeftModel.from_pretrained(model,str(initial),is_trainable=True)
    else:
        lora=LoraConfig(r=args.lora_rank,lora_alpha=args.lora_alpha,
             lora_dropout=args.lora_dropout,target_modules=['q_proj','k_proj','v_proj','o_proj'],
             bias='none',task_type=TaskType.CAUSAL_LM)
        model=get_peft_model(model,lora)
    out=run/'training'
    eval_key='eval_strategy' if 'eval_strategy' in inspect.signature(TrainingArguments).parameters else 'evaluation_strategy'
    train_kwargs={eval_key:'steps'}
    ta=TrainingArguments(output_dir=str(out),max_steps=args.max_steps,
           per_device_train_batch_size=args.batch_size,
           per_device_eval_batch_size=args.batch_size,
           gradient_accumulation_steps=args.grad_accum,
           learning_rate=args.lr,logging_steps=args.log_steps,eval_steps=max(1,min(100,args.max_steps)),save_strategy='no',
           report_to='none',bf16=use_bf16,fp16=not use_bf16,
           seed=args.seed,ddp_find_unused_parameters=False,remove_unused_columns=False,**train_kwargs)
    trainer=Trainer(model=model,args=ta,train_dataset=train_ds,eval_dataset=val_ds,
                    data_collator=DataCollatorForSeq2Seq(tokenizer=tok,padding=True,label_pad_token_id=-100))
    trainer.train()
    metrics=trainer.evaluate()
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        (run/'adapter').mkdir(parents=True,exist_ok=False)
        model.save_pretrained(str(run/'adapter'))
        dump_json(run/'training_summary.json',{'status':'TRAINING_COMPLETE','created_at_chicago':timestamp(),
            'metrics':metrics,'steps':args.max_steps,'world_size':world,'approval_sha256':digest(approval_file),
            'dataset_hashes':hashes,'initial_adapter':args.initial_adapter})
        print('D-01 TRAINING_COMPLETE',run/'adapter',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['audit','prepare','train'])
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--dataset',type=Path,default=Path('data/training/clean/python_mixed'))
    p.add_argument('--approval',type=Path)
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--model-id',default=MODEL)
    p.add_argument('--revision',default=REVISION)
    p.add_argument('--allow-download',action='store_true')
    p.add_argument('--max-seq-length',type=int,default=2048)
    p.add_argument('--initial-adapter')
    p.add_argument('--expected-world-size',type=int,default=3)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--max-steps',type=int,default=500)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--grad-accum',type=int,default=8)
    p.add_argument('--log-steps',type=int,default=10)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--lora-rank',type=int,default=16)
    p.add_argument('--lora-alpha',type=int,default=32)
    p.add_argument('--lora-dropout',type=float,default=0.05)
    args=p.parse_args()
    root=args.root.resolve()
    dataset=args.dataset if args.dataset.is_absolute() else root/args.dataset
    approval=choose_approval(root,args.approval)
    d,hashes=verify_approval(root,dataset,approval)
    corpus,pairs,counts=verify_dataset(dataset)
    print('D-01 GATE_PASS',json.dumps({'approval':str(approval),'reviewer':d['reviewer'],
          'counts':counts,'dataset_sha256':hashes},sort_keys=True),flush=True)
    if args.command=='audit':return
    if not args.run_dir:raise ValueError('--run-dir required for prepare/train')
    run=args.run_dir if args.run_dir.is_absolute() else root/args.run_dir
    if args.command=='prepare':prepare(args,run,corpus,pairs,approval,hashes)
    else:train(args,run,approval,hashes)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'D-01 ERROR: {exc}',file=sys.stderr,flush=True)
        sys.exit(1)
