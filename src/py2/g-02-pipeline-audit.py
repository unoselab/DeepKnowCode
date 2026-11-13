#!/usr/bin/env python3
"""G-02 v2 independent artifact and completion auditor. No GPU training here."""
import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def need(ok, msg):
    if not ok: raise RuntimeError(msg)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def read(path):
    p=Path(path);need(p.is_file(),'missing evidence '+str(p))
    return json.loads(p.read_text(encoding='utf-8'))


def write_new(path,obj):
    p=Path(path)
    with p.open('x',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,sort_keys=True,ensure_ascii=False)
        f.write('\n')


def chicago():return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def verify_stage(run,status,steps,prepared_hash,expected_fingerprint,initial_m1_sha,
                 max_gpu_gib,min_headroom_gib,checkpoints=True):
    run=Path(run)
    env=read(run/'run_manifest.json')
    need(env['status']==status,'stage status FAIL or mismatched '+str(run))
    need(env['config_fingerprint']==expected_fingerprint,'stage fingerprint drift')
    for relative,digest in env['artifacts'].items():
        sub=Path(relative)
        need(not sub.is_absolute() and '..' not in sub.parts,'unsafe artifact path')
        file=run/sub
        need(file.is_file() and sha256(file)==digest,'stage artifact differs '+str(file))
    if status=='G02_V2_PREPARE_PASS_NOT_TRAINED':
        manifest=read(run/'prepared/manifest.json')
        need(manifest['config_fingerprint']==expected_fingerprint,'prepared fingerprint mismatch')
        need(manifest['m1_initial_adapter']['adapter_sha256']==initial_m1_sha,
             'prepared M1 initial adapter identity drift')
        for split in ('train','validation','order'):
            item=manifest['files'][split]
            need(sha256(run/'prepared'/(split+'.jsonl'))==item['sha256'],
                 'prepared sample/order SHA256 drift: '+split)
        return {'status':status,'run':str(run),'prepared_sha256':sha256(run/'prepared/manifest.json'),
                'config_fingerprint':expected_fingerprint,
                'train_rows':manifest['training_budget']['examples']}
    token=read(run/'token_audit.json')
    need(token['status']=='G02_V2_PREPARED_TOKEN_CACHE_VERIFIED' and
         token['verified_before_model_loading'] is True and
         token['config_fingerprint']==expected_fingerprint,
         'stage token evidence missing or unapproved')
    if status=='G02_V2_M2_STANDARD_MIXED_TRAINING_COMPLETE':
        summary=read(run/'training_summary.json')
        need(summary['optimizer_steps']==steps and
             summary['initial_m1_adapter_sha256']==initial_m1_sha and
             summary['prepared_manifest_sha256']==prepared_hash,
             'M2 summary missing initial M1 or steps/dataset identity')
        history=run/'training_history.jsonl'
        need(sha256(history)==summary['training_history_sha256'],
             'training history SHA256 differs from summary')
        step_list=[];val_steps=[];checkpoint_steps=[];count=0
        with history.open(encoding='utf-8') as f:
            for line in f:
                need(line.strip(),'blank training history row')
                obj=json.loads(line);event=obj.get('event')
                if event=='OPTIMIZER_STEP':
                    count+=obj['samples'];step_list.append(obj['optimizer_step'])
                    need(obj['samples']==summary['global_effective_batch'],
                         'microbatch exposure differs')
                    need(math.isfinite(obj['loss']) and obj['loss']>=0 and
                         math.isfinite(obj['gradient_norm_before_clip']),
                         'nonfinite training scalar')
                    for m in obj['gpu_memory']:
                        need(m['peak_allocated_gib']<max_gpu_gib and
                             m['driver_free_gib']>=min_headroom_gib,
                             'memory safety budget exceeded')
                elif event=='VALIDATION':val_steps.append(obj['step'])
                elif event=='CHECKPOINT':checkpoint_steps.append(obj['optimizer_step'])
        need(step_list==list(range(1,steps+1)), 'optimizer step history incomplete')
        need(count==summary['train_sample_exposures'], 'M2 exposures incomplete')
        need(len(val_steps)==len(summary['validation_events']) and val_steps[-1]==steps,
             'validation cadence incomplete')
        need(checkpoint_steps[-1]==steps and len(set(checkpoint_steps))==len(checkpoint_steps),
             'checkpoint cadence incomplete')
        adapter=run/'adapter';ident=read(run/'adapter_manifest.json')
        need(ident['status']=='G02_V2_M2_ADAPTER_COMPLETE' and
             ident['prepared_manifest_sha256']==prepared_hash and
             ident['mixed_training_pairs']==count,
             'M2 adapter identity/status drift')
        for relative,digest in ident['adapter_files_sha256'].items():
            name=Path(relative)
            need(not name.is_absolute() and '..' not in name.parts, 'unsafe adapter path')
            need(sha256(adapter/name)==digest,'final M2 adapter binary hash mismatch '+relative)
        for n in checkpoint_steps:
            verify_checkpoint(run/'checkpoints'/f'step-{n:06d}',n,prepared_hash,expected_fingerprint)
        return {'status':status,'run':str(run),'steps':steps,'sample_exposures':count,
                'validation_steps':val_steps,'checkpoint_steps':checkpoint_steps,
                'adapter_weight_sha256':ident['adapter_files_sha256'].get('adapter_model.safetensors')}
    else:
        name='stress' if status.startswith('G02_V2_STRESS') else 'stability'
        proof=read(run/(name+'.json'))
        need(proof['steps_completed']==steps and proof['adapter_parameters_changed'] and
             proof['prepared_manifest_sha256']==prepared_hash,
             'smoke did not update trainable adapter or used wrong dataset')
        if name=='stability':
            need(proof['reloaded_previous_adapter_and_optimizer'] is True,
                 'stability did not load previous checkpoint')
        ck=run/'checkpoints'/f'step-{steps:06d}'
        verify_checkpoint(ck,steps,prepared_hash,expected_fingerprint)
        return {'status':status,'run':str(run),'steps':steps,
                'checkpoint_manifest_sha256':sha256(ck/'checkpoint_manifest.json')}


def verify_checkpoint(ckpt,step,prepared_hash,fingerprint):
    meta=read(ckpt/'checkpoint_manifest.json')
    need(meta['status']=='G02_V2_CHECKPOINT_COMPLETE' and
         meta['optimizer_step']==step and meta['prepared_manifest_sha256']==prepared_hash and
         meta['config_fingerprint']==fingerprint,'checkpoint completion/identity differs')
    need('optimizer_state.pt' in meta['files_sha256'],'checkpoint optimizer missing')
    for relative,digest in meta['files_sha256'].items():
        sub=Path(relative)
        need(not sub.is_absolute() and '..' not in sub.parts,'unsafe checkpoint path')
        f=ckpt/sub
        need(f.is_file() and sha256(f)==digest,'checkpoint binary hash mismatch '+relative)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('stage','finish'))
    p.add_argument('--operation',required=False,choices=('--train','--prepare'))
    for arg in ('pipeline','stage-run','status','prepared-run','expected-fingerprint',
                'm1-adapter-sha256','log-file','step'):
        p.add_argument('--'+arg,required=True)
    for arg in ('steps','exit-code'):
        p.add_argument('--'+arg,required=True,type=int)
    for arg in ('max-gpu-gib','min-headroom-gib'):
        p.add_argument('--'+arg,required=True,type=float)
    a=p.parse_args()
    if a.mode=='stage':
        prepared_hash=(sha256(Path(a.prepared_run)/'prepared/manifest.json')
                      if a.status!='G02_V2_PREPARE_PASS_NOT_TRAINED' else '')
        result=verify_stage(Path(a.stage_run),a.status,a.steps,prepared_hash,
               a.expected_fingerprint,a.m1_adapter_sha256,a.max_gpu_gib,a.min_headroom_gib)
        print(json.dumps({'event':'G02_STAGE_VERIFIED',**result,'at_chicago':chicago()}),flush=True)
    else:
        status=('G02_V2_FULL_M2_TRAINING_COMPLETE' if a.operation=='--train' else
                'G02_V2_PREPARE_ONLY_COMPLETE') if a.exit_code==0 else 'G02_V2_PIPELINE_FAILED'
        output={'version':'v2','status':status,'exit_code':a.exit_code,'step':a.step,
                'at_chicago':chicago(),'pipeline':a.pipeline,
                'prepared_run':a.prepared_run,'m1_adapter_sha256':a.m1_adapter_sha256,
                'log_file':a.log_file,'training_run':a.stage_run}
        write_new(Path(a.pipeline)/'pipeline_result.json',output)
        print(json.dumps({'event':'G02_PIPELINE_RESULT','status':status,
                          'path':str(Path(a.pipeline)/'pipeline_result.json'),
                          'at_chicago':chicago()}),flush=True)

if __name__=='__main__':main()
