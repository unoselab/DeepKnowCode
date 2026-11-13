#!/usr/bin/env python3
"""F-02a v2: Frozen curriculum audit, 3-GPU NCCL DataLoader smoke, and actual 2-step M1-initialized LoRA DDP training smoke.

Full 811-step training is gated by v2 smoke evidence and immutable F-01c/F-01e pins; Shell supplies settings.
"""
import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v3'
CORPORA = ('java_source', 'java_mixed', 'python_mixed')


def now():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def jsonl(path):
    with Path(path).open('rb') as f:
        for n, raw in enumerate(f, 1):
            if not raw.strip():
                raise ValueError('EMPTY_JSONL_RECORD ' + str(path) + ':' + str(n))
            yield json.loads(raw), hashlib.sha256(raw.rstrip(b'\r\n')).hexdigest()


def key(r):
    return (str(r['corpus']), str(r['id1']), str(r['id2']), int(r['label']))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('audit', 'worker', 'finalize', 'train-worker', 'train-finalize', 'full-preflight', 'full-train-worker', 'full-train-finalize'))
    for v in ('root', 'run-dir', 'frozen-prepared-run', 'curriculum-run', 'm1-adapter-file', 'log-file', 'smoke-proof-run'):
        p.add_argument('--' + v, type=Path, required=True)
    for v in ('frozen-train-sha256', 'frozen-validation-sha256', 'frozen-manifest-sha256',
              'curriculum-manifest-sha256', 'curriculum-order-sha256', 'm1-adapter-sha256',
              'model-id', 'model-revision', 'gpu-ids', 'precision', 'logging-prefix',
              'logging-timestamp-format', 'optimizer', 'lr-scheduler', 'learning-rate',
              'lora-dropout', 'lora-bias', 'target-modules', 'weight-decay', 'max-grad-norm',
              'gradient-checkpointing', 'gradient-checkpoint-use-reentrant', 'ddp-find-unused-parameters',
              'train-order-policy', 'expected-smoke-run-name', 'full-save-adapter', 'full-gradient-checkpointing', 'hf-home', 'model-type', 'token-policy', 'answer-yes', 'answer-no', 'expected-yes-token', 'expected-no-token', 'prompt-prefix', 'prompt-b-separator', 'prompt-suffix', 'save-smoke-adapter', 'eval-strategy', 'save-strategy', 'load-best-model-at-end'):
        p.add_argument('--' + v, required=True)
    for v in ('seed', 'world-size', 'per-device-batch', 'gradient-accumulation',
              'global-batch', 'optimizer-steps', 'smoke-steps', 'train-count',
              'validation-count', 'source-count', 'java-target-count', 'python-target-count',
              'max-seq-length', 'lora-rank', 'lora-alpha', 'warmup-steps', 'log-steps', 'eval-steps', 'full-eval-steps', 'full-save-steps', 'full-log-steps', 'full-warmup-steps', 'smoke-warmup-steps', 'smoke-eval-steps', 'smoke-log-steps', 'eval-batch', 'dataloader-workers'):
        p.add_argument('--' + v, type=int, required=True)
    p.add_argument('--lora-dropout-float', required=True, type=float)
    p.add_argument('--weight-decay-float', required=True, type=float)
    p.add_argument('--learning-rate-float', required=True, type=float)
    p.add_argument('--max-grad-norm-float', required=True, type=float)
    p.add_argument('--phase-names', nargs=3, required=True)
    for v in ('phase-steps', 'phase-source', 'phase-java', 'phase-python'):
        p.add_argument('--' + v, nargs=3, type=int, required=True)
    return p


def require_config(a):
    if a.world_size != 3 or len(set(a.gpu_ids.split(','))) != a.world_size:
        raise ValueError('GPU_CONFIGURATION_MISMATCH')
    if a.per_device_batch != 1 or a.gradient_accumulation != 8 or a.global_batch != 24:
        raise ValueError('M2_BATCH_CONFIG_DRIFT')
    if a.world_size * a.per_device_batch * a.gradient_accumulation != a.global_batch:
        raise ValueError('GLOBAL_BATCH_MISMATCH')
    if a.optimizer_steps * a.global_batch != a.train_count or a.smoke_steps < 1 or a.smoke_steps > a.optimizer_steps:
        raise ValueError('STEP_BUDGET_MISMATCH')
    if sum(a.phase_steps) != a.optimizer_steps:
        raise ValueError('PHASE_STEP_MISMATCH')
    if sum(a.phase_source) != a.source_count or sum(a.phase_java) != a.java_target_count or sum(a.phase_python) != a.python_target_count:
        raise ValueError('PHASE_COUNT_MISMATCH')
    for i in range(3):
        if a.phase_steps[i]*a.global_batch != a.phase_source[i]+a.phase_java[i]+a.phase_python[i]:
            raise ValueError('PHASE_BUDGET_MISMATCH')
    if a.train_count != a.source_count+a.java_target_count+a.python_target_count:
        raise ValueError('TRAIN_CORPUS_COUNT_MISMATCH')
    if a.train_order_policy != 'f01e-v4-explicit-global-step-rank-major-v1':
        raise ValueError('TRAIN_ORDER_POLICY_UNSUPPORTED')
    if a.model_id != 'meta-llama/CodeLlama-7b-hf' or a.precision != 'bf16':
        raise ValueError('MODEL_PRECISION_MISMATCH')
    if a.model_type != 'llama' or a.token_policy != 'contextual-one-token':
        raise ValueError('F01C_MODEL_TOKEN_POLICY_DRIFT')
    if a.smoke_steps != 2 or a.smoke_warmup_steps != 0 or a.smoke_eval_steps != 1 or a.smoke_log_steps != 1:
        raise ValueError('F01C_V24_SMOKE_SCHEDULE_DRIFT')
    if a.target_modules != 'q_proj,k_proj,v_proj,o_proj' or a.lora_rank != 16 or a.lora_alpha != 32:
        raise ValueError('F01C_LORA_SHAPE_DRIFT')
    if a.optimizer != 'adamw_torch' or a.lr_scheduler != 'cosine':
        raise ValueError('F01C_OPTIMIZER_DRIFT')
    if a.save_smoke_adapter not in ('true', 'false'):
        raise ValueError('INVALID_SAVE_SMOKE_ADAPTER')
    for field in ('frozen_train_sha256','frozen_validation_sha256','frozen_manifest_sha256',
                  'curriculum_manifest_sha256','curriculum_order_sha256','m1_adapter_sha256'):
        h = getattr(a, field)
        if len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
            raise ValueError('INVALID_HASH '+field)


def audit_frozen_order(a):
    require_config(a)
    f = a.frozen_prepared_run/'prepared'
    frozen_files = [('train.jsonl',a.frozen_train_sha256),('validation.jsonl',a.frozen_validation_sha256),('manifest.json',a.frozen_manifest_sha256)]
    for name, expected in frozen_files:
        if digest(f/name) != expected:
            raise ValueError('FROZEN_SHA_MISMATCH '+name)
    if digest(a.m1_adapter_file) != a.m1_adapter_sha256:
        raise ValueError('M1_ADAPTER_SHA_MISMATCH')
    fm = json.loads((f/'manifest.json').read_text())
    if fm['sample']['train']['sha256'] != a.frozen_train_sha256 or fm['sample']['validation']['sha256'] != a.frozen_validation_sha256:
        raise ValueError('FROZEN_MANIFEST_LINK_MISMATCH')
    raw_train = list(jsonl(f/'train.jsonl'))
    raw_val = list(jsonl(f/'validation.jsonl'))
    if len(raw_train) != a.train_count or len(raw_val) != a.validation_count:
        raise ValueError('FROZEN_COUNTS_MISMATCH')
    f01e = a.curriculum_run
    if digest(f01e/'manifest.json') != a.curriculum_manifest_sha256:
        raise ValueError('CURRICULUM_MANIFEST_SHA_MISMATCH')
    if digest(f01e/'curriculum_order.jsonl') != a.curriculum_order_sha256:
        raise ValueError('CURRICULUM_ORDER_SHA_MISMATCH')
    m = json.loads((f01e/'manifest.json').read_text())
    if m['status'] != 'F01E_V1_CURRICULUM_ORDER_VERIFIED' or m.get('gpu_training_enabled') is not False:
        raise ValueError('F01E_STATUS_INVALID')
    if m['order_sha256'] != a.curriculum_order_sha256 or m['m1_adapter_sha256'] != a.m1_adapter_sha256:
        raise ValueError('F01E_ORDER_OR_M1_PIN_DRIFT')
    if m['frozen_data_sha256'] != dict(frozen_files):
        raise ValueError('F01E_FROZEN_PIN_DRIFT')
    cfg=m['effective_config']
    comparisons={'model_id':a.model_id,'model_revision':a.model_revision,'seed':a.seed,
                 'world_size':a.world_size,'per_device_batch':a.per_device_batch,
                 'gradient_accumulation':a.gradient_accumulation,'global_batch':a.global_batch,
                 'optimizer_steps':a.optimizer_steps,'max_seq_length':a.max_seq_length,
                 'train_count':a.train_count,'validation_count':a.validation_count,
                 'source_count':a.source_count,'java_target_count':a.java_target_count,
                 'python_target_count':a.python_target_count,'precision':a.precision,
                 'phase_names':a.phase_names,'phase_steps':a.phase_steps,
                 'phase_source':a.phase_source,'phase_java':a.phase_java,'phase_python':a.phase_python}
    for k,v in comparisons.items():
        if cfg.get(k)!=v: raise ValueError('F01E_EFFECTIVE_CONFIG_DRIFT '+k)
    rows=[r for r,_ in jsonl(f01e/'curriculum_order.jsonl')]
    if len(rows)!=a.train_count:raise ValueError('ORDER_COUNT_MISMATCH')
    seen=set(); byphase=Counter(); bystep=Counter(); bylabel=Counter()
    boundaries={}; cumulative=0
    for name, steps in zip(a.phase_names,a.phase_steps):
        boundaries[name]=(cumulative+1,cumulative+steps)
        cumulative+=steps
    for n,r in enumerate(rows):
        source_index=r['source_index']
        if not isinstance(source_index,int) or not 0<=source_index<a.train_count:
            raise ValueError('BAD_SOURCE_INDEX')
        source,linehash=raw_train[source_index]
        if (key(source)!=key(r) or linehash!=r['source_row_sha256'] or
            source['tokens']!=r['tokens']):
            raise ValueError('FROZEN_ORDER_IDENTITY_DRIFT '+str(n))
        if r['position']!=n or r['global_step']!=n//a.global_batch+1 or r['batch_position']!=n%a.global_batch:
            raise ValueError('ORDER_POSITION_DRIFT '+str(n))
        if r['phase'] not in boundaries or not boundaries[r['phase']][0] <= r['global_step'] <= boundaries[r['phase']][1]:
            raise ValueError('PHASE_ASSIGNMENT_DRIFT '+str(n))
        if source_index in seen:raise ValueError('ORDER_DUPLICATE_INDEX')
        seen.add(source_index)
        byphase[r['phase'],r['corpus'],r['label']]+=1
        bystep[r['global_step']]+=1
        bylabel[r['global_step'],r['label']]+=1
    if len(seen)!=a.train_count:raise ValueError('ORDER_MISSING_INDICES')
    for step in range(1,a.optimizer_steps+1):
        if bystep[step]!=a.global_batch or (bylabel[step,0],bylabel[step,1])!=(12,12):
            raise ValueError('ORDER_BATCH_LABEL_DRIFT '+str(step))
    for i,name in enumerate(a.phase_names):
        for corpus, expected in [('java_source',a.phase_source[i]),('java_mixed',a.phase_java[i]),('python_mixed',a.phase_python[i])]:
            if byphase[name,corpus,0]!=expected//2 or byphase[name,corpus,1]!=expected//2:
                raise ValueError('PHASE_CORPUS_LABEL_MISMATCH '+name+'/'+corpus)
    return rows,{'status':'F02A_V1_FROZEN_AND_ORDER_AUDIT_PASS','at_chicago':now(),
                 'frozen_train_sha256':a.frozen_train_sha256,'frozen_validation_sha256':a.frozen_validation_sha256,
                 'frozen_manifest_sha256':a.frozen_manifest_sha256,
                 'm1_adapter_sha256':a.m1_adapter_sha256,
                 'curriculum_manifest_sha256':a.curriculum_manifest_sha256,
                 'curriculum_order_sha256':a.curriculum_order_sha256,
                 'pair_count':len(rows),'optimizer_steps':a.optimizer_steps,'global_batch':a.global_batch,
                 'all_batches_balanced':True,'pair_missing':0,'pair_duplicates':0,
                 'phase_budgets_verified':True,'training_enabled':False}


def write_json_new(path,obj):
    path=Path(path)
    with path.open('x',encoding='utf-8') as h:
        json.dump(obj,h,indent=2,sort_keys=True,ensure_ascii=False)
        h.write('\n')


def step_rank_records(rows, step, rank, a):
    """Rank-major 8 microsteps: each rank receives its own contiguous block of 8.

    This is an explicit NEW sampler policy, not equivalent to implicit Trainer shuffling.
    """
    width=a.per_device_batch*a.gradient_accumulation
    start=(step-1)*a.global_batch+rank*width
    return rows[start:start+width]


def worker(a):
    import torch
    import torch.distributed as dist
    from torch.utils.data import Dataset, DataLoader
    if 'LOCAL_RANK' not in os.environ or 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        raise ValueError('TORCHRUN_DDP_ENV_REQUIRED')
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); size=int(os.environ['WORLD_SIZE'])
    if size!=a.world_size or not 0<=rank<size or not 0<=local<size:
        raise ValueError('DDP_WORLD_MISMATCH')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=a.world_size or not torch.cuda.is_bf16_supported():
        raise ValueError('THREE_GPU_BF16_REQUIRED')
    torch.cuda.set_device(local)
    dist.init_process_group(backend='nccl', init_method='env://')
    try:
        rows,_=audit_frozen_order(a)
        class FixedOrderDataset(Dataset):
            def __init__(self,positions):self.positions=positions
            def __len__(self):return len(self.positions)
            def __getitem__(self,n):
                pos=self.positions[n]
                r=rows[pos]
                return {'global_position':pos,'source_index':r['source_index'],
                        'label':r['label'],'global_step':r['global_step']}
        all_rank_events=[]
        for step in range(1,a.smoke_steps+1):
            local_rows=step_rank_records(rows,step,rank,a)
            positions=[r['position'] for r in local_rows]
            loader=DataLoader(FixedOrderDataset(positions),batch_size=a.per_device_batch,
                              shuffle=False,num_workers=0,drop_last=False)
            observed=[]
            for batch in loader:
                pos=batch['global_position'].tolist()
                observed.extend(pos)
                # Real CUDA consumption check; not a model forward/backward.
                cuda_pos=batch['global_position'].to(f'cuda:{local}',non_blocking=False)
                if cuda_pos.cpu().tolist()!=pos:
                    raise ValueError('CUDA_BATCH_TRANSFER_MISMATCH')
            if observed!=positions or len(observed)!=a.gradient_accumulation*a.per_device_batch:
                raise ValueError('ACTUAL_DATALOADER_ORDER_DRIFT')
            all_rank_events.append({'step':step,'rank':rank,'positions':observed,
                                    'source_indices':[rows[i]['source_index'] for i in observed]})
        gathered=[None for _ in range(size)]
        dist.all_gather_object(gathered,all_rank_events)
        if rank==0:
            step_summaries=[]
            for step in range(1,a.smoke_steps+1):
                observed=[p for rank_events in gathered for ev in rank_events if ev['step']==step for p in ev['positions']]
                expected=list(range((step-1)*a.global_batch,step*a.global_batch))
                if sorted(observed)!=expected or len(set(observed))!=a.global_batch:
                    raise ValueError('DDP_GLOBAL_BATCH_ORDER_COVERAGE_FAIL')
                for r,rank_events in enumerate(gathered):
                    e=rank_events[step-1]
                    if e['positions']!=[x['position'] for x in step_rank_records(rows,step,r,a)]:
                        raise ValueError('DDP_RANK_CONSUMPTION_MISMATCH '+str(r))
                labels=Counter(rows[i]['label'] for i in observed)
                if labels[0]!=a.global_batch//2 or labels[1]!=a.global_batch//2:
                    raise ValueError('DDP_BATCH_LABEL_IMBALANCE')
                step_summaries.append({'step':step,'global_positions_sha256':canonical_hash(expected),
                                       'label_0':labels[0],'label_1':labels[1],
                                       'rank_major_order_policy':a.train_order_policy})
                print(json.dumps({'event':'DDP_CONSUMPTION_STEP_PASS','step':step,'at_chicago':now()},sort_keys=True),flush=True)
            report={'status':'F02A_V1_3GPU_DDP_DATALOADER_SMOKE_PASS', 'version':VERSION,
                    'created_at_chicago':now(),'device_count':torch.cuda.device_count(),
                    'world_size':size,'backend':'nccl','steps_consumed':a.smoke_steps,
                    'global_batch':a.global_batch,'rank_microbatches':a.gradient_accumulation,
                    'frozen_train_sha256':a.frozen_train_sha256,
                    'curriculum_order_sha256':a.curriculum_order_sha256,
                    'rank_events':gathered,'step_summaries':step_summaries,
                    'model_forward_backward_performed':False,'optimizer_steps_performed':0,
                    'full_811_step_consumption_verified':False,'approved_for_full_training':False}
            write_json_new(a.run_dir/'ddp_consumption_smoke.json',report)
        dist.barrier()
    finally:
        dist.destroy_process_group()



def effective_training_example(row, codes, tokenizer, answer_ids, a):
    """Precisely mirror F-01c MixedDataset's token construction and override semantics."""
    from f01c_variants import checked_training_tokens
    corpus=row['corpus']
    language='python' if corpus=='python_mixed' else 'java'
    prefix=a.prompt_prefix.replace('Language: java\n', 'Language: '+language+'\n')
    code_a=row.get('code_a_override', codes[corpus][row['id1']])
    code_b=row.get('code_b_override', codes[corpus][row['id2']])
    prompt=prefix+code_a+a.prompt_b_separator+code_b+a.prompt_suffix
    result=checked_training_tokens(tokenizer,prompt,a.answer_yes if row['label'] else a.answer_no,
                                   answer_ids[row['label']],a.max_seq_length)
    if result is None or len(result['input_ids'])!=row['tokens']:
        raise ValueError('FROZEN_TRAIN_TOKEN_DRIFT '+str((corpus,row['id1'],row['id2'])))
    return result


def code_files_from_frozen_manifest(a):
    manifest=json.loads((a.frozen_prepared_run/'prepared'/'manifest.json').read_text())
    dataset=manifest['dataset']
    paths=dataset['paths']; hashes=dataset['sha256']
    result={}
    for corpus in CORPORA:
        path=Path(paths[corpus]['data.jsonl'])
        if digest(path)!=hashes[corpus]['data.jsonl']:
            raise ValueError('FROZEN_RAW_CODE_SHA_DRIFT '+corpus)
        result[corpus]=path
    return result


def training_smoke_worker(a):
    """2 real NCCL/LoRA optimizer steps, explicit rank-major curriculum, no full training."""
    import math
    import random
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
    from peft import PeftModel
    import f01c_shared as shared
    from f01c_variants import TokenPolicy, resolve_answer_ids

    require_config(a)
    if a.smoke_steps != 2: raise ValueError('FULL_TRAINING_BLOCKED_V2')
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
    if world!=a.world_size or rank<0 or rank>=world or local<0 or local>=world:
        raise ValueError('DDP_ENV_INVALID')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=world or not torch.cuda.is_bf16_supported():
        raise RuntimeError('THREE_GPU_BF16_REQUIRED')
    torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    torch.manual_seed(a.seed);random.seed(a.seed)
    # NCCL must initialize before wrapping trainable LoRA parameters in DDP.
    dist.init_process_group('nccl', init_method='env://', device_id=device)
    try:
        rows, audit=audit_frozen_order(a)
        fm=json.loads((a.frozen_prepared_run/'prepared'/'manifest.json').read_text())
        if fm['model']['revision'] != a.model_revision or fm['model']['model_type'] != a.model_type:
            raise ValueError('FROZEN_MODEL_IDENTITY_DRIFT')
        snapshot=Path(a.hf_home)/'hub'/('models--'+a.model_id.replace('/','--'))/'snapshots'/a.model_revision
        if not snapshot.is_dir():raise FileNotFoundError('PINNED_MODEL_SNAPSHOT_MISSING '+str(snapshot))
        tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,use_fast=True)
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        probe=a.prompt_prefix+'public int a(int x) { return x + 1; }'+a.prompt_b_separator+'public int b(int x) { return x + 1; }'+a.prompt_suffix
        ids,_=resolve_answer_ids(tokenizer,probe,TokenPolicy(a.token_policy,a.answer_yes,a.answer_no,
                                                            a.expected_yes_token,a.expected_no_token))
        if {str(k):v for k,v in ids.items()} != fm['answer_ids']:
            raise ValueError('FROZEN_ANSWER_TOKEN_DRIFT')
        code_paths=code_files_from_frozen_manifest(a)
        codes={c:shared.load_codes(path) for c,path in code_paths.items()}
        frozen=[r for r,_ in jsonl(a.frozen_prepared_run/'prepared'/'train.jsonl')]
        examples=[]; rank_events=[]
        for step in range(1,a.smoke_steps+1):
            for record in step_rank_records(rows,step,rank,a):
                sample=frozen[record['source_index']]
                examples.append(effective_training_example(sample,codes,tokenizer,ids,a))
                rank_events.append({'step':step,'rank':rank,'global_position':record['position'],
                                    'source_index':record['source_index'],'label':record['label'],
                                    'tokens':sample['tokens']})
        # Fail before allocating 7B model if any selected pair is incorrectly tokenized.
        if len(examples)!=a.smoke_steps*a.gradient_accumulation*a.per_device_batch:
            raise ValueError('TRAINING_MICROBATCH_COUNT_DRIFT')
        base=AutoModelForCausalLM.from_pretrained(str(snapshot),local_files_only=True,
                  trust_remote_code=False,dtype=torch.bfloat16,low_cpu_mem_usage=True)
        if base.config.model_type!=a.model_type:raise ValueError('MODEL_ARCHITECTURE_DRIFT')
        base.config.use_cache=False
        initial_adapter=a.m1_adapter_file.parent
        if not (initial_adapter/'adapter_config.json').is_file():
            raise FileNotFoundError('M1_ADAPTER_CONFIG_MISSING')
        model=PeftModel.from_pretrained(base,str(initial_adapter),is_trainable=True,local_files_only=True)
        if not any(p.requires_grad for p in model.parameters()):
            raise ValueError('M1_ADAPTER_HAS_NO_TRAINABLE_PARAMETERS')
        config=model.peft_config[model.active_adapter]
        if (config.r!=a.lora_rank or config.lora_alpha!=a.lora_alpha or
            abs(config.lora_dropout-a.lora_dropout_float)>1e-9 or config.bias!=a.lora_bias or
            set(config.target_modules)!=set(a.target_modules.split(','))):
            raise ValueError('M1_ADAPTER_LORA_HYPERPARAMETER_DRIFT')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        if hasattr(model,'enable_input_require_grads'):model.enable_input_require_grads()
        model.to(device)
        ddp=DDP(model,device_ids=[local],output_device=local,find_unused_parameters=False,
                broadcast_buffers=False)
        trainable=[p for p in ddp.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(trainable,lr=a.learning_rate_float,weight_decay=a.weight_decay_float)
        scheduler=get_scheduler(a.lr_scheduler,optimizer=optimizer,
                                num_warmup_steps=a.smoke_warmup_steps,num_training_steps=a.smoke_steps)
        optimizer.zero_grad(set_to_none=True)
        history=[]
        from contextlib import nullcontext
        for step in range(1,a.smoke_steps+1):
            local_losses=[]
            for micro in range(a.gradient_accumulation):
                index=(step-1)*a.gradient_accumulation+micro
                data=examples[index]
                batch={k:torch.tensor([v],device=device,dtype=torch.long) for k,v in data.items()}
                # Only the eighth microbatch synchronizes gradients across GPUs.
                sync=(micro==a.gradient_accumulation-1)
                with (nullcontext() if sync else ddp.no_sync()):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=ddp(**batch).loss
                    if not torch.isfinite(loss):raise ValueError('NONFINITE_TRAINING_LOSS')
                    (loss/a.gradient_accumulation).backward()
                    local_losses.append(float(loss.detach().item()))
            grad_norm=torch.nn.utils.clip_grad_norm_(trainable,a.max_grad_norm_float)
            if not torch.isfinite(grad_norm):raise ValueError('NONFINITE_GRAD_NORM')
            rate=optimizer.param_groups[0]['lr']
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
            values=torch.tensor([sum(local_losses)/len(local_losses),float(grad_norm)],device=device)
            dist.all_reduce(values,op=dist.ReduceOp.SUM)
            values/=world
            rec={'step':step,'loss':float(values[0].item()),'mean_rank_grad_norm':float(values[1].item()),
                 'learning_rate':rate,'at_chicago':now()}
            history.append(rec)
            if rank==0: print(json.dumps({'event':'REAL_LORA_OPTIMIZER_STEP_PASS',**rec}),flush=True)
        gathered=[None for _ in range(world)]
        dist.all_gather_object(gathered,rank_events)
        if rank==0:
            for step in range(1,a.smoke_steps+1):
                actual=[e['global_position'] for rank_records in gathered for e in rank_records if e['step']==step]
                expected=list(range((step-1)*a.global_batch,step*a.global_batch))
                if sorted(actual)!=expected or len(set(actual))!=len(expected):
                    raise ValueError('TRAINING_DDP_ORDER_DRIFT')
                if Counter(rows[p]['label'] for p in actual)!=Counter({0:a.global_batch//2,1:a.global_batch//2}):
                    raise ValueError('TRAINING_DDP_LABEL_BALANCE_DRIFT')
                for r, rank_records in enumerate(gathered):
                    observed=[e['global_position'] for e in rank_records if e['step']==step]
                    intended=[x['position'] for x in step_rank_records(rows,step,r,a)]
                    if observed!=intended:raise ValueError('RANK_LOCAL_ORDER_DRIFT')
            adapter_hashes={}
            if a.save_smoke_adapter=='true':
                adapter_dir=a.run_dir/'adapter'
                if adapter_dir.exists():raise FileExistsError('SMOKE_ADAPTER_ALREADY_EXISTS')
                ddp.module.save_pretrained(str(adapter_dir),safe_serialization=True)
                for file in adapter_dir.iterdir():
                    if file.is_file():adapter_hashes[file.name]=digest(file)
                if 'adapter_model.safetensors' not in adapter_hashes:raise ValueError('SMOKE_ADAPTER_NOT_SAVED')
            write_json_new(a.run_dir/'training_smoke_evidence.json',{
                'status':'F02A_V2_REAL_3GPU_LORA_SMOKE_PASS','version':VERSION,'at_chicago':now(),
                'training_steps':len(history),'rank_events':gathered,'step_history':history,
                'optimizer_steps_performed':a.smoke_steps,'model_forward_backward_performed':True,
                'ddp_world_size':world,'frozen_train_sha256':a.frozen_train_sha256,
                'frozen_validation_sha256':a.frozen_validation_sha256,
                'm1_adapter_sha256':a.m1_adapter_sha256,'curriculum_order_sha256':a.curriculum_order_sha256,
                'adapter_hashes':adapter_hashes,'gradient_checkpointing':True,
                'full_training_approved':False,'full_811_step_consumption_verified':False,
                'effective_configuration':{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}})
        dist.barrier(device_ids=[local])
    finally:
        dist.destroy_process_group()


def training_finalize(a):
    require_config(a)
    path=a.run_dir/'training_smoke_evidence.json'
    data=json.loads(path.read_text())
    if data.get('status')!='F02A_V2_REAL_3GPU_LORA_SMOKE_PASS' or data.get('optimizer_steps_performed')!=a.smoke_steps:
        raise ValueError('TRAINING_SMOKE_NOT_PASS')
    if data['frozen_train_sha256']!=a.frozen_train_sha256 or data['curriculum_order_sha256']!=a.curriculum_order_sha256:
        raise ValueError('SMOKE_PROVENANCE_MISMATCH')
    for name,value in data['adapter_hashes'].items():
        if digest(a.run_dir/'adapter'/name)!=value:raise ValueError('SMOKE_ADAPTER_HASH_DRIFT')
    summary={'status':'F02A_V2_REAL_TRAINING_SMOKE_VERIFIED_FULL_TRAINING_BLOCKED',
             'at_chicago':now(),'model_training_run':True,'optimizer_steps':a.smoke_steps,
             'full_training_approved':False,'training_evidence_sha256':digest(path),
             'curriculum_order_sha256':a.curriculum_order_sha256,
             'frozen_train_sha256':a.frozen_train_sha256,'initial_m1_sha256':a.m1_adapter_sha256,
             'effective_configuration':data['effective_configuration']}
    write_json_new(a.run_dir/'training_smoke_summary.json',summary)
    print(json.dumps({'status':summary['status'],'run':str(a.run_dir)},sort_keys=True),flush=True)

def verify_full_training_gate(a, allow_existing_run=False):
    """Never launch 811-step training unless the v2 real 3-GPU smoke proof is intact."""
    import math
    require_config(a)
    _, audit = audit_frozen_order(a)
    if (a.full_eval_steps != a.optimizer_steps or a.full_save_steps != a.optimizer_steps or
        a.full_log_steps != a.log_steps or a.full_warmup_steps != a.warmup_steps or
        a.full_save_adapter != 'true' or a.full_gradient_checkpointing != 'true'):
        raise ValueError('FULL_M2_TRAINING_CONFIGURATION_DRIFT')
    if a.smoke_proof_run.name != a.expected_smoke_run_name:
        raise ValueError('UNPINNED_SMOKE_RUN')
    ep=a.smoke_proof_run/'training_smoke_evidence.json'
    sp=a.smoke_proof_run/'training_smoke_summary.json'
    ev=json.loads(ep.read_text(encoding='utf-8'))
    su=json.loads(sp.read_text(encoding='utf-8'))
    if su.get('status')!='F02A_V2_REAL_TRAINING_SMOKE_VERIFIED_FULL_TRAINING_BLOCKED':
        raise ValueError('SMOKE_SUMMARY_STATUS_INVALID')
    if ev.get('status')!='F02A_V2_REAL_3GPU_LORA_SMOKE_PASS':
        raise ValueError('SMOKE_EVIDENCE_STATUS_INVALID')
    if digest(ep)!=su.get('training_evidence_sha256'):
        raise ValueError('SMOKE_EVIDENCE_SHA_DRIFT')
    if ev.get('model_forward_backward_performed') is not True or ev.get('optimizer_steps_performed')!=2 or ev.get('ddp_world_size')!=a.world_size:
        raise ValueError('SMOKE_REAL_TRAINING_NOT_VERIFIED')
    for field, value in (('frozen_train_sha256',a.frozen_train_sha256),
                         ('frozen_validation_sha256',a.frozen_validation_sha256),
                         ('m1_adapter_sha256',a.m1_adapter_sha256),
                         ('curriculum_order_sha256',a.curriculum_order_sha256)):
        if ev.get(field)!=value: raise ValueError('SMOKE_PIN_DRIFT '+field)
    events=ev.get('rank_events')
    if not isinstance(events,list) or len(events)!=a.world_size:
        raise ValueError('SMOKE_RANK_EVENTS_INVALID')
    for step in (1,2):
        positions=[event['global_position'] for rank in events for event in rank if event['step']==step]
        if sorted(positions)!=list(range((step-1)*a.global_batch,step*a.global_batch)):
            raise ValueError('SMOKE_CONSUMPTION_DRIFT')
    for rec in ev.get('step_history',[]):
        if not all(math.isfinite(float(rec[field])) for field in ('loss','mean_rank_grad_norm','learning_rate')):
            raise ValueError('SMOKE_NONFINITE_STEP')
    if len(ev.get('step_history',[]))!=2:raise ValueError('SMOKE_STEP_HISTORY_INVALID')
    if a.run_dir.exists() and not allow_existing_run:raise FileExistsError('FULL_TRAIN_RUN_EXISTS')
    return {'status':'F02A_V3_FULL_TRAINING_PREFLIGHT_PASS','at_chicago':now(),
            'smoke_summary_sha256':digest(sp),'smoke_evidence_sha256':digest(ep),
            'curriculum_order_sha256':a.curriculum_order_sha256,'frozen_train_sha256':a.frozen_train_sha256,
            'initial_m1_sha256':a.m1_adapter_sha256,'full_steps':a.optimizer_steps,
            'full_warmup_steps':a.full_warmup_steps,
            'validation_count':a.validation_count,'curriculum_audit':audit}


def full_training_worker(a):
    """DDP 811-step exact rank-major curriculum, M1 initialization, full validation, checkpoint."""
    import math
    import random
    from contextlib import nullcontext
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
    from peft import PeftModel
    import f01c_shared as shared
    from f01c_variants import TokenPolicy, resolve_answer_ids

    gate=verify_full_training_gate(a, allow_existing_run=True)
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=a.world_size or not 0<=rank<world or not 0<=local<world: raise ValueError('BAD_DDP_RANKS')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=world or not torch.cuda.is_bf16_supported():
        raise ValueError('THREE_GPU_BF16_REQUIRED')
    torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    torch.manual_seed(a.seed); random.seed(a.seed)
    dist.init_process_group('nccl',init_method='env://',device_id=device)
    try:
        rows, _=audit_frozen_order(a)
        fm=json.loads((a.frozen_prepared_run/'prepared'/'manifest.json').read_text())
        if fm['model']['revision']!=a.model_revision or fm['model']['model_type']!=a.model_type:
            raise ValueError('FROZEN_MODEL_IDENTITY_DRIFT')
        snapshot=Path(a.hf_home)/'hub'/('models--'+a.model_id.replace('/','--'))/'snapshots'/a.model_revision
        if not snapshot.is_dir():raise FileNotFoundError('MODEL_SNAPSHOT_NOT_CACHED '+str(snapshot))
        tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,use_fast=True)
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        probe=a.prompt_prefix+'public int a(int x) { return x + 1; }'+a.prompt_b_separator+'public int b(int x) { return x + 1; }'+a.prompt_suffix
        answers,_=resolve_answer_ids(tokenizer,probe,TokenPolicy(a.token_policy,a.answer_yes,a.answer_no,a.expected_yes_token,a.expected_no_token))
        if {str(k):v for k,v in answers.items()}!=fm['answer_ids']:
            raise ValueError('FROZEN_ANSWER_TOKEN_DRIFT')
        code_paths=code_files_from_frozen_manifest(a)
        codes={c:shared.load_codes(path) for c,path in code_paths.items()}
        frozen=[record for record,_ in jsonl(a.frozen_prepared_run/'prepared'/'train.jsonl')]
        validation=[record for record,_ in jsonl(a.frozen_prepared_run/'prepared'/'validation.jsonl')]
        # Pre-tokenize full rank-local curriculum and rank-local validation before allocating model.
        all_rank_samples=[]; consumed=[]
        for step in range(1,a.optimizer_steps+1):
            for item in step_rank_records(rows,step,rank,a):
                sample=frozen[item['source_index']]
                all_rank_samples.append(effective_training_example(sample,codes,tokenizer,answers,a))
                consumed.append((item['position'],item['source_index'],int(item['label'])))
        val_local=[(i,effective_training_example(record,codes,tokenizer,answers,a))
                   for i,record in enumerate(validation) if i%world==rank]
        if len(all_rank_samples)!=a.optimizer_steps*a.gradient_accumulation*a.per_device_batch:
            raise ValueError('LOCAL_TRAIN_SAMPLES_INCOMPLETE')
        base=AutoModelForCausalLM.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,
                                                  dtype=torch.bfloat16,low_cpu_mem_usage=True)
        if base.config.model_type!=a.model_type:raise ValueError('MODEL_ARCH_DRIFT')
        base.config.use_cache=False
        model=PeftModel.from_pretrained(base,str(a.m1_adapter_file.parent),is_trainable=True,local_files_only=True)
        if not any(p.requires_grad for p in model.parameters()):raise ValueError('M1_NOT_TRAINABLE')
        lora=model.peft_config[model.active_adapter]
        if (lora.r!=a.lora_rank or lora.lora_alpha!=a.lora_alpha or
            abs(lora.lora_dropout-a.lora_dropout_float)>1e-9 or lora.bias!=a.lora_bias or
            set(lora.target_modules)!=set(a.target_modules.split(','))):
            raise ValueError('LORA_CONFIG_DRIFT')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        if hasattr(model,'enable_input_require_grads'):model.enable_input_require_grads()
        model.to(device)
        ddp=DDP(model,device_ids=[local],output_device=local,find_unused_parameters=False,broadcast_buffers=False)
        params=[p for p in ddp.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=a.learning_rate_float,weight_decay=a.weight_decay_float)
        scheduler=get_scheduler(a.lr_scheduler,optimizer,num_warmup_steps=a.full_warmup_steps,
                                num_training_steps=a.optimizer_steps)
        optimizer.zero_grad(set_to_none=True)
        history=[]; step_digests=[]
        for step in range(1,a.optimizer_steps+1):
            losses=[]
            for micro in range(a.gradient_accumulation):
                sample=all_rank_samples[(step-1)*a.gradient_accumulation+micro]
                batch={k:torch.tensor([v],device=device,dtype=torch.long) for k,v in sample.items()}
                sync=(micro==a.gradient_accumulation-1)
                with (nullcontext() if sync else ddp.no_sync()):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=ddp(**batch).loss
                    if not torch.isfinite(loss):raise ValueError('NONFINITE_LOSS')
                    (loss/a.gradient_accumulation).backward()
                    losses.append(float(loss.detach().item()))
            grad_norm=torch.nn.utils.clip_grad_norm_(params,a.max_grad_norm_float)
            if not torch.isfinite(grad_norm):raise ValueError('NONFINITE_GRADIENT')
            lr=optimizer.param_groups[0]['lr']
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
            # Fingerprint rank-local order for each step, all three ranks.
            local_events=consumed[(step-1)*a.gradient_accumulation:step*a.gradient_accumulation]
            gathered=[None]*world
            dist.all_gather_object(gathered,local_events)
            if rank==0:
                positions=[entry[0] for events in gathered for entry in events]
                if sorted(positions)!=list(range((step-1)*a.global_batch,step*a.global_batch)):
                    raise ValueError('FULL_CONSUMPTION_GLOBAL_STEP_DRIFT '+str(step))
                for r,events in enumerate(gathered):
                    expected=[x['position'] for x in step_rank_records(rows,step,r,a)]
                    if [x[0] for x in events]!=expected:raise ValueError('FULL_CONSUMPTION_RANK_DRIFT')
                step_digests.append({'step':step,'positions_sha256':canonical_hash(gathered),
                                     'negative':sum(x[2]==0 for events in gathered for x in events),
                                     'positive':sum(x[2]==1 for events in gathered for x in events)})
            vals=torch.tensor([sum(losses)/len(losses),float(grad_norm)],device=device)
            dist.all_reduce(vals,op=dist.ReduceOp.SUM)
            vals/=world
            entry={'step':step,'loss':float(vals[0].item()),'mean_rank_grad_norm':float(vals[1].item()),
                   'learning_rate':lr,'at_chicago':now()}
            history.append(entry)
            if rank==0 and (step==1 or step%a.full_log_steps==0 or step==a.optimizer_steps):
                print(json.dumps({'event':'F02A_V3_FULL_TRAINING_STEP',**entry}),flush=True)
        # Complete distributed evaluation on all 512 immutable validation samples.
        ddp.eval()
        sum_loss=0.0; count=0
        with torch.no_grad():
            for _,sample in val_local:
                batch={k:torch.tensor([v],device=device,dtype=torch.long) for k,v in sample.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=ddp.module(**batch).loss
                if not torch.isfinite(loss):raise ValueError('NONFINITE_EVAL_LOSS')
                sum_loss+=float(loss.item());count+=1
        scores=torch.tensor([sum_loss,float(count)],device=device,dtype=torch.float64)
        dist.all_reduce(scores,op=dist.ReduceOp.SUM)
        if int(scores[1].item())!=a.validation_count:raise ValueError('VALIDATION_COVERAGE_DRIFT')
        eval_loss=float((scores[0]/scores[1]).item())
        if not math.isfinite(eval_loss):raise ValueError('NONFINITE_EVAL_MEAN')
        if rank==0:
            if len(step_digests)!=a.optimizer_steps or any(x['negative']!=12 or x['positive']!=12 for x in step_digests):
                raise ValueError('FULL_CONSUMPTION_LABEL_AUDIT_FAIL')
            adapter_dir=a.run_dir/'adapter'
            checkpoint_dir=a.run_dir/'checkpoints'/f'checkpoint-{a.optimizer_steps}'
            for directory in (adapter_dir,checkpoint_dir):
                if directory.exists():raise FileExistsError('OUTPUT_ALREADY_EXISTS '+str(directory))
            ddp.module.save_pretrained(str(checkpoint_dir),safe_serialization=True)
            tokenizer.save_pretrained(str(checkpoint_dir))
            ddp.module.save_pretrained(str(adapter_dir),safe_serialization=True)
            tokenizer.save_pretrained(str(adapter_dir))
            hashes={str(path.relative_to(a.run_dir)):digest(path)
                    for directory in (checkpoint_dir,adapter_dir) for path in directory.iterdir() if path.is_file()}
            if not (checkpoint_dir/'adapter_model.safetensors').is_file():
                raise ValueError('CHECKPOINT_ADAPTER_MISSING')
            write_json_new(a.run_dir/'full_training_evidence.json',{
                'status':'F02A_V3_FULL_811_STEP_M3_TRAINED_PENDING_FINALIZE','version':VERSION,'at_chicago':now(),
                'optimizer_steps':a.optimizer_steps,'world_size':world,'pair_count':len(consumed)*world,
                'validation_count':a.validation_count,'validation_loss':eval_loss,
                'rank_major_step_consumption':step_digests,'history':history,
                'initial_m1_sha256':a.m1_adapter_sha256,'frozen_train_sha256':a.frozen_train_sha256,
                'frozen_validation_sha256':a.frozen_validation_sha256,'curriculum_order_sha256':a.curriculum_order_sha256,
                'smoke_evidence_sha256':gate['smoke_evidence_sha256'],'output_hashes':hashes,
                'best_checkpoint':str(checkpoint_dir),'best_metric':eval_loss,
                'checkpoint_selection_policy':'final-only-eval-811','full_training_implementation':'custom-ddp-loop-not-transformers-trainer',
                'effective_configuration':{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}})
            print(json.dumps({'event':'F02A_V3_FULL_TRAINING_EVIDENCE_WRITTEN','steps':a.optimizer_steps,
                              'eval_loss':eval_loss,'checkpoint':str(checkpoint_dir),'at_chicago':now()}),flush=True)
        dist.barrier(device_ids=[local])
    finally:
        dist.destroy_process_group()


def full_training_finalize(a):
    gate=verify_full_training_gate_finalize(a)
    evidence=a.run_dir/'full_training_evidence.json'
    data=json.loads(evidence.read_text(encoding='utf-8'))
    if data.get('status')!='F02A_V3_FULL_811_STEP_M3_TRAINED_PENDING_FINALIZE' or data.get('optimizer_steps')!=a.optimizer_steps:
        raise ValueError('FULL_TRAINING_INCOMPLETE')
    if data.get('curriculum_order_sha256')!=a.curriculum_order_sha256 or data.get('initial_m1_sha256')!=a.m1_adapter_sha256:
        raise ValueError('FULL_TRAINING_PROVENANCE_DRIFT')
    if len(data.get('rank_major_step_consumption',[]))!=a.optimizer_steps or data.get('pair_count')!=a.train_count:
        raise ValueError('FULL_CONSUMPTION_INCOMPLETE')
    for name,sha in data['output_hashes'].items():
        if digest(a.run_dir/name)!=sha:raise ValueError('FULL_OUTPUT_SHA_MISMATCH '+name)
    summary={'status':'F02A_V3_FULL_M3_TRAINING_COMPLETE','version':VERSION,'at_chicago':now(),
             'optimizer_steps':a.optimizer_steps,'validation_loss':data['validation_loss'],
             'full_pair_consumption_verified':True,'training_evidence_sha256':digest(evidence),
             'best_checkpoint':data['best_checkpoint'],'initial_m1_sha256':a.m1_adapter_sha256,
             'frozen_train_sha256':a.frozen_train_sha256,'frozen_validation_sha256':a.frozen_validation_sha256,
             'curriculum_order_sha256':a.curriculum_order_sha256,
             'implementation_difference':'Custom DDP loop; not bitwise equivalent to F-01c Hugging Face Trainer',
             'effective_configuration':data['effective_configuration']}
    write_json_new(a.run_dir/'full_training_summary.json',summary)
    print(json.dumps({'status':summary['status'],'run':str(a.run_dir),'summary_sha256':digest(a.run_dir/'full_training_summary.json')},sort_keys=True),flush=True)


def verify_full_training_gate_finalize(a):
    if not a.run_dir.is_dir():raise ValueError('FULL_RUN_MISSING')
    return verify_full_training_gate(a, allow_existing_run=True)



def main():
    a=parser().parse_args()
    if a.command=='audit':
        rows,report=audit_frozen_order(a)
        a.run_dir.mkdir(parents=True,exist_ok=False)
        report['configuration']={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}
        write_json_new(a.run_dir/'frozen_order_audit.json',report)
        print(json.dumps({'status':report['status'],'pairs':len(rows),'run':str(a.run_dir)},sort_keys=True),flush=True)
    elif a.command=='worker':
        worker(a)
    elif a.command=='train-worker':
        training_smoke_worker(a)
    elif a.command=='train-finalize':
        training_finalize(a)
    elif a.command=='full-preflight':
        report=verify_full_training_gate(a)
        print(json.dumps({'status':report['status'],'smoke_evidence_sha256':report['smoke_evidence_sha256'],'at_chicago':now()},sort_keys=True),flush=True)
    elif a.command=='full-train-worker':
        full_training_worker(a)
    elif a.command=='full-train-finalize':
        full_training_finalize(a)
    else:
        report=json.loads((a.run_dir/'ddp_consumption_smoke.json').read_text())
        if report['status']!='F02A_V1_3GPU_DDP_DATALOADER_SMOKE_PASS':raise ValueError('SMOKE_REPORT_NOT_PASS')
        if report['curriculum_order_sha256']!=a.curriculum_order_sha256:raise ValueError('SMOKE_ORDER_DRIFT')
        write_json_new(a.run_dir/'smoke_summary.json',{'status':'F02A_V1_SMOKE_COMPLETE_TRAINING_BLOCKED',
           'created_at_chicago':now(),'ddp_order_smoke_pass':True,'model_training_run':False,
           'full_training_approved':False,'configuration':{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}})
        print(json.dumps({'status':'F02A_V1_SMOKE_COMPLETE_TRAINING_BLOCKED','run':str(a.run_dir)},sort_keys=True),flush=True)

if __name__=='__main__':main()
