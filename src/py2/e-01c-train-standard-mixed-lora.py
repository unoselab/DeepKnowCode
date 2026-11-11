#!/usr/bin/env python3
"""E-01c v2 implementation. Experiment settings are REQUIRED CLI arguments from shell.

Uses immutable 7B D-03a prepared pairs, validates every pair against the 15B tokenizer,
then builds a 15B provenance manifest without resampling or changing pair files.
Training delegates to the unmodified D-03a standard-mixed Python algorithm.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v2'   # artifact format version, not an experiment parameter
CORPORA = ('java_source', 'java_mixed', 'python_mixed')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def now():return datetime.now(ZoneInfo('America/Chicago')).isoformat()
def stamp():return datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
def fail(s):raise RuntimeError(s)
def write_new(p,obj):
    p=Path(p)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,sort_keys=True,ensure_ascii=False);f.write('\n')
def check_sha(p,want):
    p=Path(p)
    if not p.is_file():fail(f'Missing file: {p}')
    got=sha(p)
    if got!=want:fail(f'SHA256 MISMATCH {p}: actual={got} expected={want}')

def arguments(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['preflight','prepare','train','verify'])
    names={'root':Path,'original-m2':Path,'model-id':str,'revision':str,'m1-run-dir':Path,'m1-expected-steps':int,
      'm1-prepared-sha256':str,'m1-config-sha256':str,'m1-weights-sha256':str,'m1-source-sha256':str,
      'ground-truth-sha256':str,'frozen-dir':Path,'frozen-model-id':str,'frozen-revision':str,
      'frozen-train-sha256':str,'frozen-validation-sha256':str,'frozen-train-examples':int,
      'frozen-validation-examples':int,'source-train':int,'java-target-train':int,'python-target-train':int,
      'source-val':int,'java-target-val':int,'python-target-val':int,'budget-policy':str,'seed':int,
      'max-seq-length':int,'eval-steps':int,'log-steps':int,'warmup-steps':int,'per-device-batch-size':int,
      'per-device-eval-batch-size':int,'gradient-accumulation-steps':int,'expected-world-size':int,
      'learning-rate':float,'lora-rank':int,'lora-alpha':int,'lora-dropout':float,'gpus':str,
      'answer-no':int,'answer-yes':int,'python-mixed-data-sha256':str,'python-mixed-train-sha256':str,
      'python-mixed-valid-sha256':str}
    for name,typ in names.items():ap.add_argument('--'+name,type=typ,required=True)
    ap.add_argument('--run-dir',type=Path) # only for selecting new run directory / verification
    a=ap.parse_args(argv)
    for key in ('root','original_m2','m1_run_dir','frozen_dir'):
        setattr(a,key,getattr(a,key).expanduser().resolve())
    if a.run_dir is not None:a.run_dir=a.run_dir.expanduser().resolve()
    if a.budget_policy!='fixed-source':fail('budget-policy must match frozen fixed-source plan')
    if a.max_seq_length<256:fail('max-seq-length must be >=256')
    if any(getattr(a,k)<1 for k in ('source_train','java_target_train','python_target_train','source_val','java_target_val','python_target_val','frozen_train_examples','frozen_validation_examples','per_device_batch_size','per_device_eval_batch_size','gradient_accumulation_steps','expected_world_size','eval_steps','log_steps','lora_rank','lora_alpha')):fail('Expected positive counts/steps/settings')
    if a.seed<0 or a.warmup_steps<0 or a.learning_rate<=0 or not 0<=a.lora_dropout<1:fail('Invalid seed, warmup, LR, or dropout')
    if a.source_train+a.java_target_train+a.python_target_train!=a.frozen_train_examples:fail('Training corpus quotas do not sum to frozen train examples')
    if a.source_val+a.java_target_val+a.python_target_val!=a.frozen_validation_examples:fail('Validation corpus quotas do not sum to frozen validation examples')
    if a.java_target_train!=a.python_target_train or any(getattr(a,k)%2 for k in ('source_train','java_target_train','python_target_train','source_val','java_target_val','python_target_val')):fail('Expected even balanced quotas and equal targets')
    batch=a.per_device_batch_size*a.gradient_accumulation_steps*a.expected_world_size
    if a.frozen_train_examples%batch:fail('Frozen train size not divisible by global batch')
    a.max_steps_expected=a.frozen_train_examples//batch
    if a.max_steps_expected%a.eval_steps:fail('Expected eval steps to divide max steps')
    if a.warmup_steps>=a.max_steps_expected:fail('warmup must be less than max_steps')
    ids=a.gpus.split(',')
    if len(ids)!=a.expected_world_size or len(set(ids))!=len(ids) or any(re.fullmatch(r'\d+',x) is None for x in ids):fail('GPU count/list disagreement')
    if a.answer_no==a.answer_yes:fail('YES and NO token IDs must differ')
    for key in vars(a):
        if key.endswith('sha256') and re.fullmatch('[a-f0-9]{64}',getattr(a,key)) is None:fail(f'Invalid {key}')
    return a

def effective(a):
    return {key:(str(value) if isinstance(value,Path) else value) for key,value in vars(a).items()
            if key not in ('command','run_dir')}

def preflight(a,weights=True):
    if not a.original_m2.is_file():fail('Original D-03a implementation missing')
    source=a.original_m2.read_text(encoding='utf-8')
    for marker in ('PeftModel.from_pretrained','standard shuffled mixing','C10_HASHES','training_plan'):
        if marker not in source:fail('Original D-03a implementation marker missing: '+marker)
    m1=a.m1_run_dir
    check_sha(m1/'prepared/manifest.json',a.m1_prepared_sha256)
    check_sha(m1/'adapter/adapter_config.json',a.m1_config_sha256)
    if weights:check_sha(m1/'adapter/adapter_model.safetensors',a.m1_weights_sha256)
    elif not (m1/'adapter/adapter_model.safetensors').is_file():fail('M1 weights missing')
    check_sha(a.root/'src/py/a-11-train-source-lora.py',a.m1_source_sha256)
    m=read(m1/'e01b_m1_manifest.json');s=read(m1/'training_summary.json')
    if m.get('status')!='E01B_V1_M1_TRAINING_COMPLETE' or s.get('status')!='TRAINING_COMPLETE' or s.get('step')!=a.m1_expected_steps:fail('Full E-01b M1 training provenance invalid')
    for o in (m,s):
        if (o.get('model_id'),o.get('revision'))!=(a.model_id,a.revision):fail('M1 model/revision mismatch')
    if s.get('prepared_manifest_sha256')!=a.m1_prepared_sha256:fail('M1 preparation provenance mismatch')
    if s.get('adapter_files_sha256',{}).get('adapter_model.safetensors')!=a.m1_weights_sha256:fail('M1 weight summary mismatch')
    if m.get('adapter_sha256',{}).get('adapter_model.safetensors')!=a.m1_weights_sha256:fail('M1 weight manifest mismatch')
    if read(m1/'adapter/adapter_config.json').get('base_model_name_or_path')!=a.model_id:fail('Wrong base for M1 adapter')
    cfg=read(m1/'adapter/adapter_config.json')
    if (cfg.get('r'),cfg.get('lora_alpha'),cfg.get('lora_dropout'))!=(a.lora_rank,a.lora_alpha,a.lora_dropout):fail('M1 LoRA settings do not match shell configuration')
    check_sha(a.root/'data/evaluation/ground_truth/python/manifest.json',a.ground_truth_sha256)
    for name,digest in [('data.jsonl',a.python_mixed_data_sha256),('train.txt',a.python_mixed_train_sha256),('valid.txt',a.python_mixed_valid_sha256)]:
        check_sha(a.root/'data/training/clean/python_mixed'/name,digest)
    f=a.frozen_dir
    check_sha(f/'train.jsonl',a.frozen_train_sha256)
    check_sha(f/'validation.jsonl',a.frozen_validation_sha256)
    fm=read(f/'manifest.json')
    if fm.get('model')!={'id':a.frozen_model_id,'revision':a.frozen_revision}:fail('Frozen 7B model/revision mismatch')
    if fm.get('sample',{}).get('train',{}).get('sha256')!=a.frozen_train_sha256 or fm.get('sample',{}).get('validation',{}).get('sha256')!=a.frozen_validation_sha256:fail('Frozen manifest data digest mismatch')
    plan=fm.get('training_plan',{})
    expected={'java_source':a.source_train,'java_mixed':a.java_target_train,'python_mixed':a.python_target_train}
    if plan.get('train_counts')!=expected or plan.get('global_batch')!=a.per_device_batch_size*a.gradient_accumulation_steps*a.expected_world_size or plan.get('max_steps')!=a.max_steps_expected or plan.get('seed')!=a.seed or plan.get('eval_steps')!=a.eval_steps:fail('Frozen training plan does not match shell')
    if fm.get('max_seq_length')!=a.max_seq_length or fm.get('answer_token_ids')!={'0':a.answer_no,'1':a.answer_yes}:fail('Frozen prompt/answer constraints mismatch')
    for split,quotas in [('train',expected),('validation',{'java_source':a.source_val,'java_mixed':a.java_target_val,'python_mixed':a.python_target_val})]:
        if fm['sample'][split]['count']!=(a.frozen_train_examples if split=='train' else a.frozen_validation_examples):fail(f'Frozen {split} count mismatch')
        for corpus,n in quotas.items():
            if fm['sample'][split]['by_corpus'][corpus]['selected']!=n:fail(f'Frozen {split} corpus {corpus} quota mismatch')
    for corpus in CORPORA:
        for file,digest in fm['data_sha256'][corpus].items():check_sha(Path(fm['data_paths'][corpus][file]),digest)
    for lang in ('java','python'):
        if not (a.root/'data/evaluation/code'/f'{lang}.jsonl').is_file():fail('Missing heldout evaluation code file')
    return {'status':'E01C_V2_PREFLIGHT_PASS_NOT_TRAINED','version':VERSION,'timestamp_chicago':now(),
        'm1_adapter_sha256':a.m1_weights_sha256,'frozen_train_sha256':a.frozen_train_sha256,
        'frozen_validation_sha256':a.frozen_validation_sha256,'original_m2_source_sha256':sha(a.original_m2),
        'frozen_manifest_sha256':sha(f/'manifest.json'),'effective_parameters':effective(a),
        'weight_verified':weights}

def tokenizer_audit(a,manifest):
    """Validate each frozen pair under actual 15B tokenizer; do not resample/replace rows."""
    # Import existing D-03a implementation to keep prompt/tokenization identical.
    import importlib.util
    spec=importlib.util.spec_from_file_location('d03a_original',a.original_m2)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tok=mod.load_tokenizer(a.model_id,a.revision)
    ids=mod.encode_answer_ids(tok)
    if ids!={0:a.answer_no,1:a.answer_yes}:fail('15B tokenizer YES/NO ID mismatch')
    codes={c:mod.read_codes(Path(manifest['data_paths'][c]['data.jsonl'])) for c in CORPORA}
    outcome={}
    for split in ('train','validation'):
        counts=Counter();mismatch=[];overlong=[];total=0
        with (a.frozen_dir/f'{split}.jsonl').open(encoding='utf-8') as stream:
            for line_no,line in enumerate(stream,1):
                if not line.strip():fail(f'Blank row in {split}:{line_no}')
                r=json.loads(line);c=r['corpus'];label=r['label'];left=r['id1'];right=r['id2']
                if c not in CORPORA or label not in (0,1):fail(f'Bad corpus/label {split}:{line_no}')
                language='python' if c=='python_mixed' else 'java'
                item=mod.build_tokens(codes[c][left],codes[c][right],label,tok,ids,a.max_seq_length,language)
                if item is None:
                    if len(overlong)<10:overlong.append({'line':line_no,'corpus':c,'id1':left,'id2':right})
                elif r.get('tokens')!=len(item['input_ids']):
                    if len(mismatch)<10:mismatch.append({'line':line_no,'corpus':c,'old':r.get('tokens'),'new':len(item['input_ids'])})
                counts[c]+=1;total+=1
        if overlong or mismatch:fail(f'15B TOKEN COMPATIBILITY FAIL {split}: overlong={overlong}, token_count_mismatch={mismatch}. No training.')
        if total!=manifest['sample'][split]['count']:fail(f'{split} row count changed')
        outcome[split]={'rows':total,'corpus_counts':dict(counts),'sha256':sha(a.frozen_dir/f'{split}.jsonl'),
                        'overlong':0,'token_length_mismatch':0}
    return {'status':'E01C_V2_15B_TOKENIZER_COMPATIBILITY_PASS','timestamp_chicago':now(),
            'model_id':a.model_id,'revision':a.revision,'answer_token_ids':ids,'splits':outcome}

def prepare(a,run,report):
    frozen=read(a.frozen_dir/'manifest.json')
    audit=tokenizer_audit(a,frozen)
    p=run/'prepared';p.mkdir()
    # Exact byte-identical copies of 7B frozen samples.
    for split,digest in [('train',a.frozen_train_sha256),('validation',a.frozen_validation_sha256)]:
        shutil.copyfile(a.frozen_dir/f'{split}.jsonl',p/f'{split}.jsonl')
        check_sha(p/f'{split}.jsonl',digest)
    manifest=dict(frozen)
    manifest['model']={'id':a.model_id,'revision':a.revision}
    manifest['initial_adapter']=str((a.m1_run_dir/'adapter').resolve())
    manifest['initial_adapter_sha256']={'adapter_config.json':a.m1_config_sha256,'adapter_model.safetensors':a.m1_weights_sha256}
    manifest['task']='D-03 M2 standard mixed-domain BF16 LoRA continuation from 15B M1; 7B frozen pair identity preserved'
    manifest['e01c_v2_provenance']={'timestamp_chicago':now(),'source_7b_frozen_manifest_sha256':sha(a.frozen_dir/'manifest.json'),
        'frozen_sample_identity':'byte identical','tokenizer_audit_status':audit['status'],
        'effective_parameters':effective(a),'source_implementation_sha256':sha(a.original_m2)}
    write_new(p/'manifest.json',manifest)
    write_new(run/'e01c_tokenizer_audit.json',audit)
    write_new(run/'e01c_effective_config.json',{'version':VERSION,'timestamp_chicago':now(),'effective_parameters':effective(a),'prepared_manifest_sha256':sha(p/'manifest.json')})
    return audit

def native_args(a,run):
    """EVERY original D-03a experiment option explicitly forwarded from shell CLI."""
    return ['--root',str(a.root),'--run-dir',str(run),'--prepared-dir',str(run/'prepared'),
      '--model-id',a.model_id,'--revision',a.revision,'--initial-adapter',str(a.m1_run_dir/'adapter'),
      '--budget-policy',a.budget_policy,'--fixed-source',str(a.source_train),
      '--target-per-language',str(a.java_target_train),'--fixed-total',str(a.frozen_train_examples),
      '--source-val',str(a.source_val),'--java-target-val',str(a.java_target_val),'--python-target-val',str(a.python_target_val),
      '--seed',str(a.seed),'--max-seq-length',str(a.max_seq_length),'--eval-steps',str(a.eval_steps),
      '--log-steps',str(a.log_steps),'--warmup-steps',str(a.warmup_steps),'--per-device-batch-size',str(a.per_device_batch_size),
      '--per-device-eval-batch-size',str(a.per_device_eval_batch_size),'--gradient-accumulation-steps',str(a.gradient_accumulation_steps),
      '--expected-world-size',str(a.expected_world_size),'--learning-rate',str(a.learning_rate),
      '--lora-rank',str(a.lora_rank),'--lora-alpha',str(a.lora_alpha),'--lora-dropout',str(a.lora_dropout)]

def verify(a,run,weight=True):
    cfg=read(run/'e01c_effective_config.json')
    if cfg.get('effective_parameters')!=effective(a):fail('Effective shell configuration mismatch with run')
    prep=read(run/'prepared/manifest.json')
    if cfg.get('prepared_manifest_sha256')!=sha(run/'prepared/manifest.json'):fail('Prepared manifest digest mismatch')
    if prep.get('model')!={'id':a.model_id,'revision':a.revision}:fail('Prepared 15B model mismatch')
    for split,digest in [('train',a.frozen_train_sha256),('validation',a.frozen_validation_sha256)]:
        check_sha(run/'prepared'/f'{split}.jsonl',digest)
        if prep['sample'][split]['sha256']!=digest:fail('Manifest sample identity mismatch')
    if prep.get('initial_adapter_sha256',{}).get('adapter_model.safetensors')!=a.m1_weights_sha256:fail('M1 adapter mismatch')
    summary=read(run/'training_summary.json')
    if summary.get('status')!='TRAINING_COMPLETE' or summary.get('step')!=a.max_steps_expected:fail('M2 training incomplete')
    if summary.get('model_id')!=a.model_id or summary.get('revision')!=a.revision:fail('M2 model mismatch')
    if summary.get('prepared_manifest_sha256')!=sha(run/'prepared/manifest.json'):fail('Trained manifest digest mismatch')
    if summary.get('initial_adapter_sha256')!=prep['initial_adapter_sha256']:fail('Trained initial adapter mismatch')
    if summary.get('training_plan')!=prep['training_plan']:fail('Training plan mismatch')
    expected_cfg={
        'budget_policy':a.budget_policy,'fixed_source':a.source_train,'fixed_total':a.frozen_train_examples,
        'target_per_language':a.java_target_train,'source_val':a.source_val,'java_target_val':a.java_target_val,
        'python_target_val':a.python_target_val,'seed':a.seed,'max_seq_length':a.max_seq_length,
        'eval_steps':a.eval_steps,'log_steps':a.log_steps,'warmup_steps':a.warmup_steps,
        'per_device_batch_size':a.per_device_batch_size,'per_device_eval_batch_size':a.per_device_eval_batch_size,
        'gradient_accumulation_steps':a.gradient_accumulation_steps,'expected_world_size':a.expected_world_size,
        'learning_rate':a.learning_rate,'lora_rank':a.lora_rank,'lora_alpha':a.lora_alpha,'lora_dropout':a.lora_dropout,
        'max_steps':a.max_steps_expected,'model_id':a.model_id,'revision':a.revision}
    for k,v in expected_cfg.items():
        if summary.get('training_config',{}).get(k)!=v:fail(f'Actual training configuration mismatch: {k}: {summary.get("training_config",{}).get(k)} != {v}')
    if weight:
        for name,digest in summary.get('adapter_files_sha256',{}).items():check_sha(run/'adapter'/name,digest)
    return {'status':'E01C_V2_M2_VERIFIED','timestamp_chicago':now(),'run':str(run),'training_steps':a.max_steps_expected,
         'frozen_train_sha256':a.frozen_train_sha256,'frozen_validation_sha256':a.frozen_validation_sha256,
         'initial_adapter_sha256':a.m1_weights_sha256,'adapter_files_sha256':summary['adapter_files_sha256'],
         'prepared_manifest_sha256':sha(run/'prepared/manifest.json'),'training_summary_sha256':sha(run/'training_summary.json'),
         'effective_parameters':effective(a)}

def main(argv=None):
    a=arguments(argv)
    if a.command=='verify':
        if a.run_dir is None:fail('--run-dir is required for --verify')
        report=verify(a,a.run_dir)
        print(json.dumps(report,indent=2,sort_keys=True));return
    report=preflight(a,weights=True)
    if a.command=='preflight':print(json.dumps(report,indent=2,sort_keys=True));return
    run=a.run_dir or a.root/'results/e-01c'/f'{stamp()}-{a.command}'
    if run.exists():fail(f'Refusing overwrite existing output directory: {run}')
    run.parent.mkdir(parents=True,exist_ok=True);run.mkdir()
    write_new(run/'e01c_preflight.json',report)
    audit=prepare(a,run,report)
    print(json.dumps({'status':'E01C_V2_PREPARED_NOT_TRAINED','run':str(run),'tokenizer_audit':audit['status'],
                      'train':a.frozen_train_examples,'val':a.frozen_validation_examples,'steps':a.max_steps_expected},sort_keys=True),flush=True)
    if a.command=='prepare':return
    # Original M2 train() uses the 15B-adjusted immutable prepared manifest; no resampling.
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=a.gpus;env['TOKENIZERS_PARALLELISM']='false'
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node='+str(a.expected_world_size),
         str(a.original_m2),'train']+native_args(a,run)
    print('E-01c v2 FULL TRAIN', ' '.join(cmd),flush=True)
    subprocess.run(cmd,check=True,cwd=a.root,env=env)
    receipt=verify(a,run)
    receipt['status']='E01C_V2_M2_TRAINING_COMPLETE_VERIFIED'
    write_new(run/'e01c_m2_manifest.json',receipt)
    print(json.dumps(receipt,indent=2,sort_keys=True),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('E01C_V2_ERROR:',exc,file=sys.stderr,flush=True)
        raise
