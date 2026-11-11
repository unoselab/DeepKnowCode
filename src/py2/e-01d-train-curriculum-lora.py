#!/usr/bin/env python3
"""E-01d v2 fail-closed launcher: D-04 v7 original Trainer order; Shell owns all parameters."""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Refusing to overwrite ' + str(path))
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')


def verify_file(path, expected):
    if not Path(path).is_file():
        raise FileNotFoundError(str(path))
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'SHA256 mismatch {path}: got {actual} expected {expected}')


def effective(a):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items() if k != 'command'}


def checks(a):
    if a.train_count != a.steps * a.world_size * a.batch_size * a.accum:
        raise ValueError('Training rows / DDP schedule mismatch')
    if a.source_train + a.java_train + a.python_train != a.train_count:
        raise ValueError('Corpus budget mismatch')
    if a.source_val + a.java_val + a.python_val != a.val_count:
        raise ValueError('Validation budget mismatch')
    if a.java_train != a.python_train or a.java_train % 2:
        raise ValueError('Unbalanced target domains')
    if a.eval_steps != a.steps:
        raise ValueError('D-04 v7 last-step eval required')
    if len(a.gpus.split(',')) != a.world_size or len(set(a.gpus.split(','))) != a.world_size or not all(x.isdigit() for x in a.gpus.split(',')):
        raise ValueError('GPU IDs must be three distinct integers')
    if a.audit_steps < 1 or a.audit_steps > a.steps:
        raise ValueError('Invalid audit steps')
    for n in ('m1_weights_hash', 'm1_config_hash', 'train_hash', 'val_hash', 'order_hash', 'original_d04_hash', 'e01c_manifest_hash'):
        v = getattr(a, n)
        if len(v) != 64 or any(c not in '0123456789abcdef' for c in v):
            raise ValueError(f'Invalid SHA256 {n}')
    verify_file(a.original_d04, a.original_d04_hash)
    verify_file(a.m1_dir / 'adapter_model.safetensors', a.m1_weights_hash)
    verify_file(a.m1_dir / 'adapter_config.json', a.m1_config_hash)
    verify_file(a.e01c_prepared / 'train.jsonl', a.train_hash)
    verify_file(a.e01c_prepared / 'validation.jsonl', a.val_hash)
    verify_file(a.e01c_prepared / 'manifest.json', a.e01c_manifest_hash)
    verify_file(a.e01c_summary, a.e01c_summary_hash)
    verify_file(a.original_candidate / 'curriculum_order_candidate.jsonl', a.order_hash)
    cm = read(a.original_candidate / 'manifest.json')
    if cm.get('status') != a.candidate_status:
        raise ValueError('D-04 candidate status changed')
    if cm.get('curriculum_order_candidate_sha256') != a.order_hash:
        raise ValueError('D-04 candidate SHA differs from Shell')
    pm = read(a.e01c_prepared / 'manifest.json')
    if pm['model'] != {'id': a.model_id, 'revision': a.revision}:
        raise ValueError('E-01c prepared model mismatch')
    if pm['initial_adapter_sha256'] != {'adapter_config.json': a.m1_config_hash, 'adapter_model.safetensors': a.m1_weights_hash}:
        raise ValueError('Initial adapter provenance mismatch')
    if Path(pm['initial_adapter']).resolve() != a.m1_dir.resolve():
        raise ValueError('E-01c initial adapter path mismatch')
    if pm['sample']['train']['sha256'] != a.train_hash or pm['sample']['validation']['sha256'] != a.val_hash:
        raise ValueError('E-01c prepared sample hash mismatch')
    ref = read(a.e01c_summary)
    if ref['status'] != 'TRAINING_COMPLETE' or ref['step'] != a.steps or ref['model_id'] != a.model_id or ref['revision'] != a.revision:
        raise ValueError('E-01c reference training not complete or mismatched')
    if ref['prepared_manifest_sha256'] != a.e01c_manifest_hash:
        raise ValueError('E-01c training summary prepared manifest mismatch')
    expected_ref = {'budget_policy': a.budget_policy, 'target_per_language': a.java_train, 'fixed_source': a.source_train,
        'max_steps': a.steps, 'eval_steps': a.eval_steps, 'seed': a.seed, 'max_seq_length': a.max_seq_length,
        'learning_rate': a.lr, 'warmup_steps': a.warmup_steps, 'log_steps': a.log_steps,
        'per_device_batch_size': a.batch_size, 'per_device_eval_batch_size': a.eval_batch_size,
        'gradient_accumulation_steps': a.accum, 'expected_world_size': a.world_size,
        'lora_rank': a.lora_rank, 'lora_alpha': a.lora_alpha, 'lora_dropout': a.lora_dropout,
        'source_val': a.source_val, 'java_target_val': a.java_val, 'python_target_val': a.python_val}
    for key, value in expected_ref.items():
        if ref['training_config'].get(key) != value:
            raise ValueError(f'E-01d vs E-01c optimizer/protocol mismatch: {key}')
    plan = pm['training_plan']
    if plan['max_steps'] != a.steps or plan['eval_steps'] != a.eval_steps or plan['global_batch'] != a.world_size * a.batch_size * a.accum:
        raise ValueError('E-01c training plan mismatch')
    if plan['train_counts'] != {'java_source': a.source_train, 'java_mixed': a.java_train, 'python_mixed': a.python_train}:
        raise ValueError('E-01c corpus plan mismatch')
    original_d04_hash = sha(a.original_d04)
    return {'status': 'E01D_V2_PREFLIGHT_PASS_NOT_TRAINED', 'timestamp_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
            'original_d04_sha256': original_d04_hash, 'original_curriculum_order_sha256': a.order_hash,
            'original_e01c_prepared_manifest_sha256': a.e01c_manifest_hash,
            'initial_adapter_weights_sha256': a.m1_weights_hash, 'effective_parameters': effective(a)}


def prepare(a):
    report = checks(a)
    if a.prepared_dir.exists() or a.candidate_dir.exists():
        raise FileExistsError('Prepared or candidate output exists; never overwrite')
    a.prepared_dir.mkdir(parents=True, exist_ok=False)
    for filename in ('train.jsonl', 'validation.jsonl', 'manifest.json'):
        shutil.copyfile(a.e01c_prepared / filename, a.prepared_dir / filename)
    verify_file(a.prepared_dir / 'train.jsonl', a.train_hash)
    verify_file(a.prepared_dir / 'validation.jsonl', a.val_hash)
    verify_file(a.prepared_dir / 'manifest.json', a.e01c_manifest_hash)
    a.candidate_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(a.original_candidate / 'curriculum_order_candidate.jsonl', a.candidate_dir / 'curriculum_order_candidate.jsonl')
    cm = read(a.original_candidate / 'manifest.json')
    # Only rebind provenance from 7B prepared manifest to byte-identical 15B prepared manifest.
    cm['source_prepared_manifest_sha256'] = a.e01c_manifest_hash
    cm['e01d_v2_provenance'] = {'source_d04_manifest_sha256': sha(a.original_candidate / 'manifest.json'),
        'source_d04_order_sha256': a.order_hash, 'rebound_to_15b_prepared_manifest_sha256': a.e01c_manifest_hash,
        'reason': 'Same train/validation bytes; 15B model and M1 adapter in prepared manifest'}
    write(a.candidate_dir / 'manifest.json', cm)
    write(a.prepared_dir / 'e01d_effective_config.json', effective(a))
    write(a.prepared_dir / 'e01d_prepare_report.json', report)
    print(json.dumps({'status': 'E01D_V2_PREPARED_NOT_TRAINED', 'train': a.train_count, 'validation': a.val_count, 'steps': a.steps,
                      'prepared_dir': str(a.prepared_dir), 'candidate_dir': str(a.candidate_dir)}, indent=2), flush=True)


def audit_inputs(a):
    checks(a)
    if not (a.prepared_dir / 'e01d_effective_config.json').is_file():
        raise FileNotFoundError('Run --prepare first')
    if read(a.prepared_dir / 'e01d_effective_config.json') != effective(a):
        raise ValueError('Effective parameter drift since preparation')
    verify_file(a.prepared_dir / 'train.jsonl', a.train_hash)
    verify_file(a.prepared_dir / 'validation.jsonl', a.val_hash)
    verify_file(a.prepared_dir / 'manifest.json', a.e01c_manifest_hash)
    verify_file(a.candidate_dir / 'curriculum_order_candidate.jsonl', a.order_hash)
    cm = read(a.candidate_dir / 'manifest.json')
    if cm['source_prepared_manifest_sha256'] != a.e01c_manifest_hash:
        raise ValueError('Candidate provenance mismatch')
    if cm['curriculum_order_candidate_sha256'] != a.order_hash:
        raise ValueError('Candidate order mismatch')
    # Original 7B D-04 input was audited; verify original candidate still matches new copy.
    if cm['e01d_v2_provenance']['source_d04_manifest_sha256'] != sha(a.original_candidate / 'manifest.json'):
        raise ValueError('Original D-04 candidate changed')


def d04_command_args(a, mode, run_dir):
    return [mode, '--root', str(a.root), '--run-dir', str(run_dir), '--prepared-dir', str(a.prepared_dir),
      '--candidate-dir', str(a.candidate_dir), '--trainer-audit-dir', str(a.trainer_audit_dir),
      '--audit-steps', str(a.audit_steps), '--model-id', a.model_id, '--revision', a.revision,
      '--initial-adapter', str(a.m1_dir), '--budget-policy', a.budget_policy,
      '--target-per-language', str(a.java_train), '--fixed-source', str(a.source_train),
      '--fixed-total', str(a.train_count), '--source-val', str(a.source_val), '--java-target-val', str(a.java_val),
      '--python-target-val', str(a.python_val), '--seed', str(a.seed), '--max-seq-length', str(a.max_seq_length),
      '--eval-steps', str(a.eval_steps), '--log-steps', str(a.log_steps), '--warmup-steps', str(a.warmup_steps),
      '--per-device-batch-size', str(a.batch_size), '--per-device-eval-batch-size', str(a.eval_batch_size),
      '--gradient-accumulation-steps', str(a.accum), '--expected-world-size', str(a.world_size),
      '--learning-rate', str(a.lr), '--lora-rank', str(a.lora_rank), '--lora-alpha', str(a.lora_alpha),
      '--lora-dropout', str(a.lora_dropout)]


def invoke_original(a, mode, run_dir):
    # Load original D-04 v7 implementation; the only overridden globals are effective
    # Shell-provided dataset/order contract values, never an experimental Python default.
    spec = importlib.util.spec_from_file_location('e01d_original_d04', a.original_d04)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ORDER_SHA256 = a.order_hash
    mod.TRAIN_SHA256 = a.train_hash
    mod.VALID_SHA256 = a.val_hash
    mod.TOTAL_ROWS = a.train_count
    mod.TOTAL_STEPS = a.steps
    mod.WORLD = a.world_size
    mod.ACCUM = a.accum
    # Original main() parses sys.argv; supply every parameter explicitly.
    sys.argv = [str(a.original_d04), *d04_command_args(a, mode, run_dir)]
    mod.main()


def run_distributed(a, mode, run_dir):
    argv = [a.python_bin, '-m', 'torch.distributed.run', '--standalone',
            f'--nproc_per_node={a.world_size}', str(Path(__file__).resolve()),
            mode, *a.raw_cli_args]
    env = os.environ.copy()
    # Keep CUDA visible for the Trainer/Accelerate distributed order audit.
    # Hiding all GPUs makes TrainingArguments infer non-distributed CPU mode,
    # even when torchrun launches 3 ranks, causing false rank-order mismatches.
    # audit-worker only constructs D-04's TinyModel; it never loads the 15B model.
    env['CUDA_VISIBLE_DEVICES'] = a.gpus
    env['E01D_AUDIT_WORKER'] = '1' if mode == 'audit-worker' else '0'
    print('[E-01d v2] DDP audit uses visible CUDA devices (TinyModel only); not 15B training' if mode == 'audit-worker' else '[E-01d v2] GPU training dispatch', flush=True)
    print('E-01d v2 DISPATCH:', ' '.join(argv), flush=True)
    subprocess.run(argv, check=True, env=env)


def verify_training(a):
    audit_inputs(a)
    s = read(a.train_run_dir / 'training_summary.json')
    if s['status'] != 'TRAINING_COMPLETE' or s['step'] != a.steps:
        raise ValueError('Training incomplete')
    if s['model_id'] != a.model_id or s['revision'] != a.revision:
        raise ValueError('Trained wrong backbone')
    if Path(s['initial_adapter']).resolve() != a.m1_dir.resolve() or s['initial_adapter_sha256']['adapter_model.safetensors'] != a.m1_weights_hash:
        raise ValueError('Wrong initial adapter')
    cfg = s['training_config']
    expected = {'target_per_language': a.java_train, 'fixed_source': a.source_train, 'max_steps': a.steps,
       'eval_steps': a.eval_steps, 'seed': a.seed, 'learning_rate': a.lr, 'lora_rank': a.lora_rank,
       'lora_alpha': a.lora_alpha, 'lora_dropout': a.lora_dropout, 'warmup_steps': a.warmup_steps,
       'per_device_batch_size': a.batch_size, 'gradient_accumulation_steps': a.accum}
    for key, val in expected.items():
        if cfg.get(key) != val:
            raise ValueError(f'Training config mismatch {key}: {cfg.get(key)} vs {val}')
    if s['training_plan']['max_steps'] != a.steps:
        raise ValueError('Training plan steps mismatch')
    if s.get('curriculum_order_sha256') != a.order_hash:
        raise ValueError('Curriculum order summary mismatch')
    consumption = read(a.train_run_dir / 'consumption_audit.json')
    if consumption.get('status') != 'CURRICULUM_CONSUMPTION_VERIFIED' or consumption.get('order_sha256') != a.order_hash:
        raise ValueError('Curriculum consumption audit incomplete')
    verify_file(a.train_run_dir / 'consumption_audit.json', s['consumption_audit_sha256'])
    for filename, h in s['adapter_files_sha256'].items():
        verify_file(a.train_run_dir / 'adapter' / filename, h)
    report = {'status': 'E01D_V2_M3_TRAINING_COMPLETE_VERIFIED', 'timestamp_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
              'effective_parameters': effective(a), 'training_summary_sha256': sha(a.train_run_dir/'training_summary.json'),
              'adapter_sha256': s['adapter_files_sha256'], 'prepared_train_sha256': a.train_hash, 'prepared_val_sha256': a.val_hash}
    write(a.train_run_dir/'e01d_m3_manifest.json', report)
    print(json.dumps(report, indent=2), flush=True)


def make_parser():
    p = argparse.ArgumentParser(description='E-01d v2 shell-configured M3 curriculum, original D-04 v7 order')
    p.add_argument('command', choices=['preflight','prepare','audit-trainer','train','verify','audit-worker','training-worker'])
    req = lambda flag, **kw: p.add_argument('--'+flag.replace('_','-'), required=True, **kw)
    for n in ('root','original_d04','m1_dir','e01c_prepared','e01c_summary','original_candidate','prepared_dir','candidate_dir','trainer_audit_dir','train_run_dir'):
        req(n, type=Path)
    for n in ('original_d04_hash','e01c_manifest_hash','e01c_summary_hash','m1_weights_hash','m1_config_hash','train_hash','val_hash','order_hash','candidate_status','model_id','revision','budget_policy','gpus','python_bin'):
        req(n)
    for n in ('train_count','val_count','steps','eval_steps','log_steps','warmup_steps','source_train','java_train','python_train','source_val','java_val','python_val','seed','max_seq_length','batch_size','eval_batch_size','accum','world_size','lora_rank','lora_alpha','audit_steps'):
        req(n,type=int)
    for n in ('lr','lora_dropout'):
        req(n,type=float)
    return p


def main():
    p=make_parser()
    a=p.parse_args()
    a.raw_cli_args=sys.argv[2:].copy()
    for n in ('root','original_d04','m1_dir','e01c_prepared','e01c_summary','original_candidate','prepared_dir','candidate_dir','trainer_audit_dir','train_run_dir'):
        setattr(a,n,getattr(a,n).resolve())
    if a.command == 'preflight':
        print(json.dumps(checks(a), indent=2), flush=True)
    elif a.command == 'prepare':
        prepare(a)
    elif a.command in ('audit-trainer','train'):
        audit_inputs(a)
        if a.command == 'audit-trainer':
            if a.trainer_audit_dir.exists():
                raise FileExistsError('Refusing to overwrite audit dir')
            a.trainer_audit_dir.mkdir(parents=True, exist_ok=False)
            run_distributed(a,'audit-worker',a.trainer_audit_dir)
            report=read(a.trainer_audit_dir/'trainer_order_audit.json')
            if report['status']!='ACTUAL_TRAINER_DATALOADER_ORDER_VERIFIED_CPU_ONLY' or report['audit_steps']!=a.audit_steps:
                raise ValueError('Trainer audit failed')
        else:
            audit=read(a.trainer_audit_dir/'trainer_order_audit.json')
            if audit['curriculum_order_sha256']!=a.order_hash:
                raise ValueError('Trainer order audit mismatch')
            if a.train_run_dir.exists():
                raise FileExistsError('Training run already exists')
            a.train_run_dir.mkdir(parents=True, exist_ok=False)
            run_distributed(a,'training-worker',a.train_run_dir)
            verify_training(a)
    elif a.command == 'audit-worker':
        audit_inputs(a)
        invoke_original(a,'audit-trainer',a.trainer_audit_dir)
    elif a.command == 'training-worker':
        audit_inputs(a)
        invoke_original(a,'train',a.train_run_dir)
    else:
        verify_training(a)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'[E-01d v2] ERROR: {exc}', file=sys.stderr, flush=True)
        raise
