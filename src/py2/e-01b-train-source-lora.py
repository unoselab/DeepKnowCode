#!/usr/bin/env python3
"""E-01b v1: audited 15B M1 launch, reusing original A-11 Python training implementation.

The supervised sample selection, tokenization, prompt, LoRA, DDP and Trainer
logic are NOT reimplemented. prepare/train delegate to the existing A-11 Python.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MODEL = 'bigcode/starcoder2-15b'
REV = '46d44742909c03ac8cee08eb03fdebce02e193ec'
VERSION = 'v1'
ORIGINAL = 'src/py/a-11-train-source-lora.py'

def now():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def dump(p,obj):
    if p.exists():raise FileExistsError('Refusing overwrite: '+str(p))
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,sort_keys=True)+'\n',encoding='utf-8')

def provenance(root):
    p=root/ORIGINAL
    if not p.is_file():raise FileNotFoundError('Required existing A-11 Python implementation not found: '+str(p))
    actual=sha(p)
    source_git_blob=hashlib.sha1(('blob '+str(p.stat().st_size)+'\0').encode()+p.read_bytes()).hexdigest()
    expected=os.environ.get('E01B_EXPECT_A11_GIT_BLOB','').strip()
    if expected and source_git_blob!=expected:
        raise RuntimeError('A-11 source Git blob does not match E01B_EXPECT_A11_GIT_BLOB')
    return {'path':str(p),'sha256':actual,'git_blob_sha1':source_git_blob}

def cached_model():
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig,AutoTokenizer
    snapshot=Path(snapshot_download(repo_id=MODEL,revision=REV,local_files_only=True))
    if snapshot.name!=REV:raise RuntimeError('Snapshot revision mismatch')
    index=snapshot/'model.safetensors.index.json'
    if not index.is_file():raise FileNotFoundError('Missing safetensors index')
    idx=read(index)
    names=set(idx['weight_map'].values())
    missing=[x for x in names if not (snapshot/x).is_file() or (snapshot/x).stat().st_size==0]
    if missing:raise RuntimeError('Incomplete shards: '+str(missing))
    config=AutoConfig.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False)
    tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,use_fast=True)
    answer={key:tok.encode(' '+key,add_special_tokens=False) for key in ('YES','NO')}
    if any(len(ids)!=1 for ids in answer.values()):
        raise RuntimeError('YES/NO are not single tokens: '+str(answer))
    return {'snapshot':str(snapshot),'revision':REV,'model_type':config.model_type,
            'tokenizer_fast':bool(tok.is_fast),'answer_token_ids':answer,
            'weight_shards':len(names),'index_sha256':sha(index),'tokenizer_sha256':sha(snapshot/'tokenizer.json')}

def hardware():
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if torch.cuda.device_count()!=3:raise RuntimeError('Expected exactly three visible GPUs')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 unavailable')
    return [{'index':i,'name':torch.cuda.get_device_name(i),
             'vram_gib':round(torch.cuda.get_device_properties(i).total_memory/2**30,3)} for i in range(3)]

def controls(a):
    return {'seed':a.seed,'sample_policy':'balanced','max_train_examples':a.train_examples,
            'max_val_examples':a.val_examples,'max_seq_length':a.max_seq_length,
            'max_steps':a.max_steps,'eval_steps':a.eval_steps,'warmup_steps':a.warmup_steps,
            'per_device_batch_size':1,'per_device_eval_batch_size':1,
            'gradient_accumulation_steps':8,'global_batch':24,
            'learning_rate':0.0002,'lora_rank':16,'lora_alpha':32,'lora_dropout':0.05,
            'expected_world_size':3,'precision':'bf16','train_dataset':'java_source'}

def validate(a):
    root=a.root.resolve()
    src=provenance(root)
    for rel in ('data/training/clean/java_source/data.jsonl',
                'data/training/clean/java_source/train.txt',
                'data/training/clean/java_source/valid.txt',
                'data/training/manifest.json',
                'data/evaluation/ground_truth/python/manifest.json'):
        if not (root/rel).is_file():raise FileNotFoundError('Missing required data: '+rel)
    model=cached_model()
    gpus=hardware()
    return {'version':VERSION,'status':'E01B_V1_PREFLIGHT_PASS_NOT_TRAINED',
            'created_at_chicago':now(),'model_id':MODEL,'revision':REV,
            'original_training_implementation':src,'model_cache':model,
            'gpus':gpus,'controls':controls(a),
            'ground_truth_manifest_sha256':sha(root/'data/evaluation/ground_truth/python/manifest.json'),
            'java_source_sha256':{name:sha(root/'data/training/clean/java_source'/name)
                                 for name in ('data.jsonl','train.txt','valid.txt')},
            'limitations':['Cache/index validation is not tensor checksum verification',
                           'BF16/DDP memory feasibility must be tested with --smoke',
                           'E-01b reuses existing A-11 Python training logic; E-01c/d must use frozen M1 weights']}

def original_args(a,command,run):
    return [sys.executable,str(a.root.resolve()/ORIGINAL),command,
            '--root',str(a.root.resolve()),'--run-dir',str(run),
            '--model-id',MODEL,'--revision',REV,
            '--seed',str(a.seed),'--sample-policy','balanced',
            '--max-train-examples',str(a.train_examples),
            '--max-val-examples',str(a.val_examples),
            '--max-seq-length',str(a.max_seq_length),
            '--max-steps',str(a.max_steps),'--eval-steps',str(a.eval_steps),
            '--log-steps',str(a.log_steps),'--warmup-steps',str(a.warmup_steps),
            '--per-device-batch-size','1','--per-device-eval-batch-size','1',
            '--gradient-accumulation-steps','8','--expected-world-size','3',
            '--learning-rate','0.0002','--lora-rank','16','--lora-alpha','32',
            '--lora-dropout','0.05']

def run_command(cmd, env=None):
    print('E-01b EXECUTE', ' '.join(cmd),flush=True)
    subprocess.run(cmd,check=True,env=env)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['preflight','prepare','smoke','train'])
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--train-examples',type=int,default=12000)
    p.add_argument('--val-examples',type=int,default=512)
    p.add_argument('--max-seq-length',type=int,default=2048)
    p.add_argument('--max-steps',type=int,default=500)
    p.add_argument('--eval-steps',type=int,default=100)
    p.add_argument('--warmup-steps',type=int,default=20)
    p.add_argument('--log-steps',type=int,default=10)
    a=p.parse_args()
    if a.mode=='smoke':
        a.train_examples=96;a.val_examples=32;a.max_steps=2;a.eval_steps=1;a.warmup_steps=0
    if a.max_steps<1 or a.eval_steps<1 or a.max_steps%a.eval_steps!=0:raise ValueError('Invalid steps')
    if not 0<=a.warmup_steps<a.max_steps:raise ValueError('Warmup exceeds training length')
    info=validate(a)
    print(json.dumps(info,ensure_ascii=False,sort_keys=True),flush=True)
    if a.mode=='preflight':return
    if a.run_dir is None:raise ValueError('--run-dir required')
    run=a.run_dir.resolve()
    if run.exists():raise FileExistsError('Run directory exists: '+str(run))
    if a.mode=='train':
        if (a.train_examples,a.val_examples,a.max_steps,a.eval_steps,a.warmup_steps)!=(12000,512,500,100,20):
            raise RuntimeError('Full training protocol differs from A-11 default. Refusing unreviewed change.')
        smoke=list((a.root.resolve()/'results/e-01b').glob('*/smoke_success.json'))
        eligible=[]
        for s in smoke:
            try:
                rec=read(s)
                if rec['revision']==REV and rec['original_training_implementation']['sha256']==info['original_training_implementation']['sha256']:
                    eligible.append(s)
            except (KeyError,ValueError,OSError):pass
        if not eligible:raise RuntimeError('No matching successful 15B smoke test. Run --smoke first.')
        info['validated_smoke_run']=str(sorted(eligible)[-1])
    run.mkdir(parents=True,exist_ok=False)
    dump(run/'e01b_preflight.json',info)
    run_command(original_args(a,'prepare',run))
    if a.mode=='prepare':
        print(json.dumps({'status':'E01B_V1_PREPARED_NOT_TRAINED','run':str(run)}),flush=True)
        return
    env=dict(os.environ)
    env['TOKENIZERS_PARALLELISM']='false'
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=3']
    cmd+=original_args(a,'train',run)[1:]
    run_command(cmd,env=env)
    summary=run/'training_summary.json'
    if not summary.is_file() or read(summary).get('status')!='TRAINING_COMPLETE':
        raise RuntimeError('A-11 training did not produce TRAINING_COMPLETE summary')
    adapter=run/'adapter'
    required=[adapter/'adapter_model.safetensors',adapter/'adapter_config.json']
    if not all(x.is_file() and x.stat().st_size>0 for x in required):
        raise RuntimeError('Adapter files incomplete')
    result={'version':VERSION,'created_at_chicago':now(),'revision':REV,
            'model_id':MODEL,'run':str(run),'adapter':str(adapter),
            'original_training_implementation':info['original_training_implementation'],
            'adapter_sha256':{x.name:sha(x) for x in required},
            'training_summary_sha256':sha(summary),
            'status':'E01B_V1_SMOKE_PASS_NOT_FINAL_MODEL' if a.mode=='smoke' else 'E01B_V1_M1_TRAINING_COMPLETE'}
    dump(run/'smoke_success.json' if a.mode=='smoke' else run/'e01b_m1_manifest.json',result)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('E-01b v1 ERROR:',str(e),file=sys.stderr,flush=True)
        raise
