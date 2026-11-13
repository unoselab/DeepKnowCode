#!/usr/bin/env python3
"""G-01d v4: read-only evidence checks and immutable pipeline completion records.

All experiment-specific inputs are mandatory shell-supplied CLI arguments.
Does not train, select datasets, modify prior runs, or load model weights.
"""
import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(piece)
    return digest.hexdigest()


def read_json(path):
    file = Path(path)
    need(file.is_file(), 'missing JSON artifact: ' + str(file))
    return json.loads(file.read_text(encoding='utf-8'))


def write_new(path, obj):
    with Path(path).open('x', encoding='utf-8') as out:
        json.dump(obj, out, sort_keys=True, ensure_ascii=False, indent=2)
        out.write('\n')


def chicago():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def artifact(run, envelope, name):
    expected = envelope.get('artifacts', {}).get(name)
    need(isinstance(expected, str) and len(expected) == 64,
         'missing artifact hash: ' + name)
    file = run / name
    need(file.is_file() and sha256(file) == expected,
         'artifact missing or checksum mismatch: ' + str(file))
    return file


def verify_checkpoint(ckpt, prepared_hash, fingerprint, expected_step):
    meta = read_json(ckpt / 'checkpoint_manifest.json')
    need(meta['status'] == 'G01D_V4_CHECKPOINT_COMPLETE',
         'checkpoint incomplete: ' + str(ckpt))
    need(meta['config_fingerprint'] == fingerprint and
         meta['prepared_manifest_sha256'] == prepared_hash,
         'checkpoint config/dataset drift: ' + str(ckpt))
    need(meta['optimizer_step'] == expected_step, 'checkpoint step drift')
    for name, digest in meta['files_sha256'].items():
        need(name and not Path(name).is_absolute() and '..' not in Path(name).parts,
             'unsafe checkpoint path')
        file = ckpt / name
        need(file.is_file() and sha256(file) == digest,
             'checkpoint artifact hash mismatch: ' + str(file))
    need('optimizer_state.pt' in meta['files_sha256'], 'optimizer state missing')
    return meta


def validate_prepare(a):
    run = a.prepared_run.resolve()
    need(sha256(a.engine) == a.expected_engine_sha256,
         'G-01d Python engine SHA256 drift; must not run unreviewed implementation')
    manifest_file = run / 'prepared' / 'manifest.json'
    need(sha256(manifest_file) == a.expected_manifest_sha256,
         'pinned prepared manifest SHA256 mismatch')
    envelope = read_json(run / 'run_manifest.json')
    need(envelope.get('status') == 'G01C_V1_PREPARE_PASS_NOT_TRAINED',
         'prepared G-01c run did not PASS')
    need(envelope['config_fingerprint'] == a.expected_fingerprint,
         'prepared run config fingerprint drift')
    for name in ('prepared/manifest.json', 'prepared/train.jsonl',
                 'prepared/validation.jsonl', 'prepared/order.jsonl',
                 'prepare_result.json'):
        artifact(run, envelope, name)
    manifest = read_json(manifest_file)
    need(manifest['status'] == envelope['status'], 'prepared status disagreement')
    need(manifest['config_fingerprint'] == a.expected_fingerprint,
         'prepared fingerprint mismatch')
    proof = read_json(run / 'prepare_result.json')
    need(proof['status'] == manifest['status'] and
         proof['config_fingerprint'] == a.expected_fingerprint and
         proof['prepared_manifest_sha256'] == a.expected_manifest_sha256,
         'prepare_result disagreement')
    for key, digest in [('train', a.expected_train_sha256),
                        ('validation', a.expected_val_sha256),
                        ('order', a.expected_order_sha256)]:
        path = run / 'prepared' / (key + '.jsonl')
        need(sha256(path) == digest, 'prepared JSONL content SHA256 mismatch: ' + key)
        need(manifest['files'][key]['sha256'] == digest, 'manifest file identity mismatch')
    budget = manifest['training_budget']
    need(budget['examples'] == a.expected_train_count and
         budget['optimizer_steps'] == a.expected_steps and
         budget['gradient_accumulation'] == a.expected_global_batch and
         budget['epochs'] == a.expected_epochs,
         'training budget changed')
    need(budget['examples'] == a.expected_global_batch * a.expected_steps * a.expected_epochs,
         'training exposure budget inconsistent')
    need(manifest['files']['train']['rows'] == a.expected_train_count and
         manifest['files']['validation']['rows'] == a.expected_val_count and
         manifest['files']['order']['rows'] == a.expected_train_count,
         'prepared row counts differ')
    expected = {'train':a.frozen_train_sha256,
                'validation':a.frozen_val_sha256,
                'manifest':a.frozen_manifest_sha256}
    need(manifest['frozen_source']['frozen_prepared_sha256'] == expected,
         'F-01c frozen dataset hash pin mismatch')
    print(json.dumps({'event':'G01D_PREPARED_PROOF_PASS',
                      'prepared_run':str(run),
                      'prepared_manifest_sha256':a.expected_manifest_sha256,
                      'config_fingerprint':a.expected_fingerprint,
                      'train_pairs':a.expected_train_count,
                      'validation_pairs':a.expected_val_count,
                      'at_chicago':chicago()}), flush=True)


def verify_gpu_memory(rows, a):
    """Require true tensor peak and post-reclaim physical GPU headroom evidence.

    Allocator reserved memory may exceed placement budget without exceeding the
    budget for active tensors; it is intentionally NOT a hard failure check.
    """
    need(isinstance(rows,list) and rows, 'missing GPU memory evidence')
    for entry in rows:
        need(isinstance(entry,dict) and 'gpu' in entry, 'bad GPU memory row')
        fields=('allocated_gib','reserved_gib','peak_allocated_gib',
                'driver_free_gib','driver_total_gib','inactive_cache_gib')
        need(all(key in entry and isinstance(entry[key], (int,float)) and
                 math.isfinite(entry[key]) for key in fields),
             'incomplete or nonfinite CUDA memory accounting')
        need(entry['allocated_gib'] >= 0 and entry['reserved_gib'] >= 0 and
             entry['inactive_cache_gib'] >= 0, 'negative allocator accounting')
        need(entry['peak_allocated_gib'] < a.max_gpu_gib and
             entry['allocated_gib'] < a.max_gpu_gib,
             'GPU actual tensor allocation exceeded hard cap')
        need(entry['driver_free_gib'] >= a.min_headroom_gib,
             'driver free VRAM below minimum headroom')
        need(entry['driver_total_gib'] >= entry['driver_free_gib'] > 0,
             'invalid driver free and total VRAM evidence')
        need(entry['reserved_gib'] + 0.01 >= entry['allocated_gib'],
             'PyTorch allocator accounting inconsistent')


def validate_stage(a):
    run = a.stage_run.resolve()
    env = read_json(run / 'run_manifest.json')
    need(env['status'] == a.expected_status, 'step status did not PASS')
    need(env['config_fingerprint'] == a.expected_fingerprint,
         'stage config fingerprint drift')
    need(env.get('effective_parameters', {}).get('revision') == a.expected_model_revision,
         'stage pinned model revision drift')
    params=env.get('effective_parameters', {})
    need(float(params.get('max_gpu_gib',-1)) == a.max_gpu_gib and
         float(params.get('min_headroom_gib',-1)) == a.min_headroom_gib and
         float(params.get('cache_reclaim_trigger_gib',-1)) == a.cache_reclaim_trigger_gib,
         'GPU memory policy drift from standalone Shell configuration')
    prepared_hash = a.expected_manifest_sha256
    token_audit = read_json(artifact(run, env, 'token_audit.json'))
    frozen_date = read_json(a.prepared_run / 'prepared/manifest.json')['at_chicago'][:10]
    need(token_audit.get('template_date') == frozen_date and
         env.get('effective_parameters', {}).get('template_date') == frozen_date,
         'pinned Harmony date drift from frozen prepared run')
    need(token_audit['status'] == 'G01D_V4_PREPARED_TOKEN_CACHE_VERIFIED' and
         token_audit['config_fingerprint'] == a.expected_fingerprint and
         token_audit['verified_before_model_loading'] is True and
         token_audit['split_counts'] == {'train': a.expected_train_count, 'validation': a.expected_val_count},
         'invalid or incomplete verified Harmony token cache')
    stage = run.name.rsplit('-', 1)[-1]
    if stage in ('stress', 'stability'):
        report = read_json(artifact(run,env,stage+'.json'))
        need(report['status'] == a.expected_status and
             report['prepared_manifest_sha256'] == prepared_hash and
             report['config_fingerprint'] == a.expected_fingerprint,
             'smoke-stage source/binding drift')
        need(report['steps_completed'] == a.expected_steps and
             report['adapter_parameters_changed'] is True,
             'smoke-stage optimizer update missing')
        if stage == 'stability':
            need(report['reloaded_previous_adapter_and_optimizer'] is True,
                 'stability must reload stress optimizer checkpoint')
        for item in report['history']:
            need(math.isfinite(item['loss']) and item['loss'] >= 0 and
                 math.isfinite(item['gradient_norm_before_clip']),
                 'nonfinite stress/stability evidence')
            verify_gpu_memory(item['gpu_memory'],a)
            need(isinstance(item.get('inactive_cache_reclaim_events'),int) and
                 item['inactive_cache_reclaim_events'] >= 0,
                 'missing CUDA cache maintenance evidence')
        ckpt = run / 'checkpoints' / f'step-{a.expected_steps:06d}'
        checkpoint_manifest = artifact(run,env,
                    f'checkpoints/step-{a.expected_steps:06d}/checkpoint_manifest.json')
        need(sha256(checkpoint_manifest) == report['checkpoint_manifest_sha256'],
             'smoke checkpoint proof hash mismatch')
        verify_checkpoint(ckpt, prepared_hash, a.expected_fingerprint, a.expected_steps)
    elif stage == 'train':
        summary = read_json(artifact(run,env,'training_summary.json'))
        history_file = artifact(run,env,'training_history.jsonl')
        identity = read_json(artifact(run,env,'adapter_manifest.json'))
        policy=summary.get('memory_policy',{})
        need(policy.get('max_active_gib') == a.max_gpu_gib and
             policy.get('min_driver_free_gib') == a.min_headroom_gib and
             policy.get('cache_reclaim_trigger_gib') == a.cache_reclaim_trigger_gib and
             policy.get('reserved_gib_is_not_hard_cap') is True,
             'training summary CUDA memory policy mismatch')
        optimizer_events=[]
        for line in history_file.read_text(encoding='utf-8').splitlines():
            need(bool(line.strip()),'blank full training history line')
            item=json.loads(line)
            if item.get('event')=='OPTIMIZER_STEP':
                verify_gpu_memory(item.get('gpu_memory'),a)
                need(isinstance(item.get('inactive_cache_reclaim_events'),int) and
                     item['inactive_cache_reclaim_events'] >= 0,
                     'missing per-step cache reclaim event count')
                optimizer_events.append(item['optimizer_step'])
        need(optimizer_events == list(range(1,a.expected_steps+1)),
             'full optimizer history steps missing/duplicated/out-of-order')
        need(summary['status'] == a.expected_status and
             summary['optimizer_steps'] == a.expected_steps and
             summary['prepared_manifest_sha256'] == prepared_hash and
             summary['config_fingerprint'] == a.expected_fingerprint,
             'training summary config/steps mismatch')
        need(not summary['is_resumed'] and
             summary['newly_executed_steps'] == a.expected_steps,
             'unexpected resume/short training')
        need(identity['status'] == 'G01D_V4_M1_ADAPTER_COMPLETE' and
             identity['optimizer_steps'] == a.expected_steps and
             identity['prepared_manifest_sha256'] == prepared_hash and
             identity['config_fingerprint'] == a.expected_fingerprint,
             'adapter model identity mismatch')
        need(summary['training_history_sha256'] == sha256(history_file) and
             summary['token_audit_sha256'] == sha256(run/'token_audit.json') and
             summary['adapter_manifest_sha256'] == sha256(run / 'adapter_manifest.json'),
             'training history/adapter manifest SHA256 drift')
        adapter = run / 'adapter'
        need(adapter.is_dir() and identity['adapter_files_sha256'], 'final adapter missing')
        for name, digest in identity['adapter_files_sha256'].items():
            need(name and '..' not in Path(name).parts and not Path(name).is_absolute(),
                 'unsafe adapter relative path')
            path = adapter / name
            need(path.is_file() and sha256(path) == digest,
                 'adapter file hash mismatch: ' + str(path))
        need(summary['final_optimizer_log']['optimizer_step'] == a.expected_steps and
             summary['last_validation']['step'] == a.expected_steps,
             'last optimizer/validation step incomplete')
        need(len(summary['checkpoints_written_this_run']) >= 1,
             'training checkpoints absent')
        for entry in summary['checkpoints_written_this_run']:
            path = Path(entry)
            need(path.is_relative_to(run), 'checkpoint outside training run')
            name = path.name
            need(name.startswith('step-') and name[5:].isdigit(),
                 'unknown checkpoint name')
            verify_checkpoint(path, prepared_hash, a.expected_fingerprint,
                              int(name[5:]))
    else:
        raise RuntimeError('unexpected train stage directory name: ' + str(run))
    print(json.dumps({'event':'G01D_STAGE_VERIFIED', 'stage':stage,
                      'steps':a.expected_steps, 'run':str(run),
                      'at_chicago':chicago()}),flush=True)


def finalize(a):
    root = a.pipeline
    need(root.is_dir(), 'missing pipeline directory')
    need(not (root / 'pipeline_result.json').exists(),
         'refusing to overwrite existing pipeline report')
    if a.status == 'PASS':
        need(a.exit_code == 0 and a.train_run and a.stress_run and a.stability_run,
             'PASS pipeline must include all stages')
        env = read_json(a.train_run / 'run_manifest.json')
        summary = read_json(a.train_run / 'training_summary.json')
        need(env['status'] == 'G01D_V4_M1_SOURCE_ONLY_TRAINING_COMPLETE' and
             summary['optimizer_steps'] == a.expected_steps and
             len(summary['validation_events']) == a.expected_val_events and
             summary['prepared_manifest_sha256'] == a.expected_manifest_sha256 and
             summary['config_fingerprint'] == a.fingerprint,
             'full M1 evidence not complete')
        result_status = 'G01D_V4_FULL_M1_TRAINING_COMPLETE'
    else:
        need(a.status == 'FAIL' and a.exit_code != 0,
             'invalid pipeline FAIL report exit code')
        result_status = 'G01D_V4_PIPELINE_FAILED'
    item={'version':'v4','status':result_status,'at_chicago':chicago(),
          'failure_or_final_step':a.step,'exit_code':a.exit_code,
          'elapsed_seconds':a.elapsed_seconds, 'log_file':str(a.log_file),
          'prepared_run':str(a.prepared_run),'config_fingerprint':a.fingerprint,
          'train_run':str(a.train_run) if a.train_run else None,
          'stress_run':str(a.stress_run) if a.stress_run else None,
          'stability_run':str(a.stability_run) if a.stability_run else None}
    write_new(root / 'pipeline_result.json', item)
    print(json.dumps({'event':'G01D_PIPELINE_RESULT','status':result_status,
                      'path':str(root/'pipeline_result.json'),
                      'at_chicago':chicago()}),flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_subparsers(dest='command',required=True)
    prep = actions.add_parser('prepare')
    for name in ('prepared-run','engine','expected-engine-sha256',
                 'expected-manifest-sha256','expected-fingerprint',
                 'expected-train-sha256','expected-val-sha256','expected-order-sha256',
                 'frozen-train-sha256','frozen-val-sha256','frozen-manifest-sha256'):
        prep.add_argument('--'+name,required=True, type=Path if name in ('prepared-run','engine') else str)
    for name in ('expected-train-count','expected-val-count','expected-steps',
                 'expected-global-batch','expected-epochs'):
        prep.add_argument('--'+name,required=True,type=int)
    stage = actions.add_parser('stage')
    for name in ('stage-run','prepared-run','expected-fingerprint',
                 'expected-manifest-sha256','expected-status','expected-model-revision'):
        stage.add_argument('--'+name,required=True,type=Path if name.endswith('-run') else str)
    stage.add_argument('--expected-steps',required=True,type=int)
    stage.add_argument('--max-gpu-gib',required=True,type=float)
    stage.add_argument('--min-headroom-gib',required=True,type=float)
    stage.add_argument('--cache-reclaim-trigger-gib',required=True,type=float)
    stage.add_argument('--expected-train-count',required=True,type=int)
    stage.add_argument('--expected-val-count',required=True,type=int)
    final = actions.add_parser('final-report')
    for name in ('pipeline','status','step','exit-code','log-file','prepared-run',
                 'fingerprint','elapsed-seconds'):
        final.add_argument('--'+name,required=True,
            type=Path if name in ('pipeline','log-file','prepared-run') else
                 int if name in ('exit-code','elapsed-seconds') else str)
    for name in ('train-run','stress-run','stability-run','expected-manifest-sha256'):
        final.add_argument('--'+name,type=Path if name.endswith('-run') else str)
    for name in ('expected-steps','expected-val-events'):
        final.add_argument('--'+name,type=int)
    return p


def main():
    a = parser().parse_args()
    if a.command == 'prepare':
        validate_prepare(a)
    elif a.command == 'stage':
        validate_stage(a)
    else:
        finalize(a)


if __name__ == '__main__':
    main()
