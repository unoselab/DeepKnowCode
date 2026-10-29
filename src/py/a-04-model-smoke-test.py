#!/usr/bin/env python3
"""A-04: load a base code model and run a minimal, NON-SCORED clone-pair probe.

This stage does not access A-02 test data, compute F1, or fine-tune models.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

MODELS = {"starcoder2-7b": "bigcode/starcoder2-7b"}

EXAMPLES = [
    {"language":"java","expected":1,
     "left":"int sum(int a, int b) { return a + b; }",
     "right":"int add(int x, int y) { return x + y; }"},
    {"language":"python","expected":0,
     "left":"def square(x):\n    return x * x",
     "right":"def greet(name):\n    return 'Hello ' + name"},
    {"language":"java","expected":1,"left":"boolean isEven(int n) { return n % 2 == 0; }", "right":"boolean parity(int x) { return x % 2 == 0; }"},
    {"language":"java","expected":0,"left":"int inc(int x) { return x + 1; }", "right":"String hello() { return \"hello\"; }"},
    {"language":"python","expected":1,"left":"def mul(a,b): return a*b", "right":"def product(x,y): return x*y"},
    {"language":"python","expected":0,"left":"def upper(s): return s.upper()", "right":"def length(seq): return len(seq)"}
]

def make_prompt(ex):
    return ("Task: decide whether two code fragments are clones. "
            "Respond with exactly one token: YES or NO.\n"
            f"Language: {ex['language']}\nCode A:\n{ex['left']}\n"
            f"Code B:\n{ex['right']}\nAnswer:")

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model', choices=MODELS, default='starcoder2-7b')
    p.add_argument('--worker-index', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=1)
    p.add_argument('--download-only', action='store_true')
    p.add_argument('--output-dir', default='results/a-04')
    p.add_argument('--max-new-tokens', type=int, default=8)
    p.add_argument('--local-files-only', action='store_true',help='Do not download model weights')
    p.add_argument('--revision', default=None, help='Optional frozen Hugging Face commit SHA')
    args=p.parse_args()
    if args.num_workers < 1 or not 0 <= args.worker_index < args.num_workers:
        p.error('invalid worker index or worker count')
    if args.download_only:
        from huggingface_hub import snapshot_download
        print('[A-04] Downloading or verifying cached StarCoder2-7B snapshot', flush=True)
        print('[A-04] Snapshot:', snapshot_download(repo_id=MODELS[args.model], revision=args.revision, local_files_only=args.local_files_only), flush=True)
        return
    if args.max_new_tokens < 1 or args.max_new_tokens > 32:
        p.error('--max-new-tokens must be between 1 and 32')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; A-04 requires a GPU')
    model_id=MODELS[args.model]
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid())
    result={"run_id":run_id,"model":args.model,"model_id":model_id,"revision_requested":args.revision,
            "gpu_visible":os.environ.get('CUDA_VISIBLE_DEVICES',''),"device":torch.cuda.get_device_name(0),
            "local_files_only":args.local_files_only,"status":"started","worker_index":args.worker_index,"num_workers":args.num_workers,"examples":[]}
    target=out/f'{args.model}-{run_id}.json'
    try:
        print('[A-04] Initial GPU free/total GiB:',tuple(round(x/1024**3,2) for x in torch.cuda.mem_get_info(0)),flush=True)
        options={"local_files_only":args.local_files_only}
        if args.revision: options['revision']=args.revision
        start=time.time()
        tokenizer=AutoTokenizer.from_pretrained(model_id,**options)
        model=AutoModelForCausalLM.from_pretrained(model_id,dtype=torch.float16,
                                                    low_cpu_mem_usage=True,device_map={"":0},**options)
        model.eval()
        result['load_seconds']=round(time.time()-start,2)
        result['actual_revision']=getattr(model.config,'_commit_hash',None)
        result['peak_allocated_gib_after_load']=round(torch.cuda.max_memory_allocated(0)/1024**3,2)
        print('[A-04] Model loaded in seconds:',result['load_seconds'],flush=True)
        for ix,ex in enumerate(EXAMPLES):
            if ix % args.num_workers != args.worker_index:
                continue
            prompt=make_prompt(ex)
            encoded=tokenizer(prompt,return_tensors='pt',truncation=True,max_length=1024).to('cuda:0')
            with torch.inference_mode():
                generated=model.generate(**encoded,max_new_tokens=args.max_new_tokens,do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)
            completion=tokenizer.decode(generated[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True)
            parsed=re.match(r'^\s*(YES|NO)\b',completion,re.I)
            record={"example":ix,"language":ex['language'],"expected_for_diagnostic_only":ex['expected'],
                    "completion":completion,"parsed_prediction":(1 if parsed.group(1).upper()=='YES' else 0) if parsed else None,
                    "parseable":bool(parsed)}
            result['examples'].append(record)
            print('[A-04] Pilot:',json.dumps(record,ensure_ascii=False),flush=True)
        result['status']='completed'
        result['peak_allocated_gib']=round(torch.cuda.max_memory_allocated(0)/1024**3,2)
        print('[A-04] NON-SCORED diagnostic only. No F1 or RCP computed.',flush=True)
    except Exception as exc:
        result['status']='failed';result['error']=repr(exc)
        traceback.print_exc()
        raise
    finally:
        target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        print('[A-04] Result:',target,flush=True)

if __name__=='__main__':
    main()
