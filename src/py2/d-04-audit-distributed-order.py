#!/usr/bin/env python3
"""D-04 v6: read-only curriculum schedule + actual CPU Gloo DDP order audit.

This does NOT integrate with Hugging Face Trainer, train a model, or assert that
Trainer's sampler consumes this order. It defines an explicit rank/microstep
contract to implement in the future M3 custom sampler.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

WORLD=3
ACCUM=8
PER_DEVICE=1
GLOBAL_BATCH=WORLD*ACCUM*PER_DEVICE
STEPS=811
ROWS=GLOBAL_BATCH*STEPS
EXPECTED_PHASES={
    'foundation': (270, {'java_source':5184,'java_mixed':648,'python_mixed':648}),
    'adaptation': (270, {'java_source':4016,'java_mixed':1232,'python_mixed':1232}),
    'specialization': (271, {'java_source':2800,'java_mixed':1852,'python_mixed':1852}),
}
EXPECTED_GROUP={('java_source',0):6000,('java_source',1):6000,
                ('java_mixed',0):1866,('java_mixed',1):1866,
                ('python_mixed',0):1866,('python_mixed',1):1866}
EXPECTED_TRAIN='b7e2db71f36171ed9fe96bef4082b34dc80a56b3c853c852d73684e186351a1e'

def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def pair_id(x):
    return (x['corpus'], str(x['id1']), str(x['id2']), int(x['label']))

def chk(value,expected,reason):
    if value!=expected:raise ValueError(f'{reason}: got {value!r}, expected {expected!r}')

def resolve(root,value):
    p=Path(value)
    return p if p.is_absolute() else root/p

def load_audit(root,prepared,source):
    pm=json.loads((prepared/'manifest.json').read_text())
    vm=json.loads((source/'manifest.json').read_text())
    train_path=prepared/'train.jsonl'; valid_path=prepared/'validation.jsonl'
    ord_path=source/'curriculum_order_candidate.jsonl'
    score_path=source/'difficulty_candidates.jsonl'
    for path, digest in [
       (train_path,pm['sample']['train']['sha256']),
       (valid_path,pm['sample']['validation']['sha256']),
       (ord_path,vm['curriculum_order_candidate_sha256']),
       (score_path,vm['difficulty_candidates_sha256']),
       (prepared/'manifest.json',vm['source_prepared_manifest_sha256'])]:
        chk(hash_file(path),digest,'SHA256 mismatch: '+str(path))
    chk(hash_file(train_path),EXPECTED_TRAIN,'M2/M3 training SHA256')
    chk(hash_file(train_path),vm['train_sha256'],'v5 training SHA256')
    chk(hash_file(valid_path),vm['validation_sha256'],'v5 validation SHA256')
    chk(vm['status'],'AUDIT_ONLY_NO_PRIMARY_SCORE_SELECTED_NO_M3_TRAINING','v5 status')
    train=load_jsonl(train_path);valid=load_jsonl(valid_path)
    order=load_jsonl(ord_path); scores=load_jsonl(score_path)
    chk((len(train),len(valid),len(order),len(scores)),(ROWS,512,ROWS,ROWS),'counts')
    chk(len(set(map(pair_id,train))),ROWS,'training unique pair count')
    if set(map(pair_id,train))&set(map(pair_id,valid)):
        raise ValueError('Pair-level train/validation overlap')
    chk(Counter((r['corpus'],int(r['label'])) for r in train),Counter(EXPECTED_GROUP),'train group budget')
    for i,(o,s) in enumerate(zip(order,scores)):
        # scores are in original training-row order, not curriculum order.
        chk(s['row_index'],i,'score row index')
        chk(pair_id(s),pair_id(train[i]),'score-to-train mapping')
    used=set(); phase_group=Counter();phase_step=defaultdict(set)
    for i,o in enumerate(order):
        step,slot=divmod(i,GLOBAL_BATCH)
        chk(o['sequence_index'],i,'sequence_index')
        chk((o['optimizer_step'],o['batch_slot']),(step+1,slot),'step-slot mapping')
        rid=o['row_index']
        if not isinstance(rid,int) or rid<0 or rid>=ROWS:raise ValueError('Invalid row_index')
        if rid in used:raise ValueError('Duplicate training row')
        used.add(rid)
        chk(pair_id(o),pair_id(train[rid]),'order-to-train mapping')
        chk(o['difficulty_score'],scores[rid]['score_4d'],'v5 score linkage')
        ph=o['phase']
        if ph not in EXPECTED_PHASES:raise ValueError('Unknown phase: '+str(ph))
        expected_ph=('foundation' if step<270 else 'adaptation' if step<540 else 'specialization')
        chk(ph,expected_ph,'phase boundary')
        phase_group[(ph,o['corpus'],int(o['label']))]+=1
        phase_step[ph].add(step+1)
    chk(len(used),ROWS,'unique row coverage')
    for ph,(expected_steps,counts) in EXPECTED_PHASES.items():
        chk(len(phase_step[ph]),expected_steps,'phase steps: '+ph)
        for corpus,count in counts.items():
            for label in (0,1):chk(phase_group[(ph,corpus,label)],count//2,'phase label budget: '+ph+' '+corpus)
    # All steps are 12 positive, 12 negative; no change in dataset.
    for step in range(STEPS):
        batch=order[step*GLOBAL_BATCH:(step+1)*GLOBAL_BATCH]
        chk(Counter(int(x['label']) for x in batch),Counter({0:12,1:12}),'global batch label balance step '+str(step+1))
    # Explicit future sampler contract: per optimizer step, rank r owns
    # 8 consecutive positions [r*8 .. r*8+7] in that 24-example batch.
    # A rank processes its 8 indices in ascending microstep order.
    # This is intentionally NOT claimed to match stock Trainer sampler.
    rank_slots={str(rank):list(range(rank*ACCUM,(rank+1)*ACCUM)) for rank in range(WORLD)}
    info={'status':'ORDER_PLAN_AUDITED_NOT_TRAINER_INTEGRATED_NO_M3_TRAINING',
          'version':'v6','created_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
          'train_pairs':ROWS,'validation_pairs':len(valid),'steps':STEPS,
          'world_size':WORLD,'gradient_accumulation':ACCUM,'per_device_batch_size':PER_DEVICE,
          'global_batch':GLOBAL_BATCH,'global_batch_label_balance':'12_POSITIVE_12_NEGATIVE_ALL_STEPS',
          'per_rank_slots_in_each_optimizer_step':rank_slots,
          'phase_label_counts':{f'{ph}/{corpus}/label={label}':value for (ph,corpus,label),value in sorted(phase_group.items())},
          'source_curriculum_order_sha256':hash_file(ord_path),'source_train_sha256':hash_file(train_path),
          'initial_adapter':pm['initial_adapter'],'initial_adapter_sha256':pm['initial_adapter_sha256'],
          'limitations':['AUDIT ONLY: not actual Trainer dataloader consumption.',
                         'No GPU training, no weights generated, no Trainer sampler patched.',
                         'A future M3 custom sampler must reproduce this exact order under DDP and gradient accumulation.',
                         'Gradient synchronization, checkpointing, and restart semantics require a separate integration test.',
                         'v5 difficulty ordering remains a candidate, not validated model difficulty.']}
    return order,info

def atomic_report(out,report,root,source,ddp=False):
    if out.exists():raise FileExistsError('Refusing to overwrite: '+str(out))
    out.mkdir(parents=True)
    report['ddp_gloo_cross_rank_verified']=ddp
    audit=out/'distributed_order_audit.json'
    audit.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf8')
    manifest={'component':'D-04 Distributed Order Audit','version':'v6',
              'status':report['status'],'distributed_order_audit_sha256':hash_file(audit),
              'source_curriculum_order_sha256':report['source_curriculum_order_sha256'],
              'source_train_sha256':report['source_train_sha256']}
    (out/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':report['status'],'out':str(out),'steps':STEPS,
                      'train_pairs':ROWS,'ddp_gloo_cross_rank_verified':ddp,
                      'source_order_sha256':report['source_curriculum_order_sha256']},sort_keys=True),flush=True)

def audit_ddp(order,report,out):
    import torch
    import torch.distributed as dist
    if not dist.is_available():raise RuntimeError('torch.distributed not available')
    chk(int(os.environ.get('WORLD_SIZE','0')),WORLD,'torchrun WORLD_SIZE')
    dist.init_process_group(backend='gloo')
    try:
        rank=dist.get_rank()
        # Rank-local sampler outputs a (step, microstep, row_index) ledger.
        # No shuffle/padding/drop_last; no duplicated examples.
        ledger=[(step+1,micro,order[step*GLOBAL_BATCH+rank*ACCUM+micro]['row_index'])
                for step in range(STEPS) for micro in range(ACCUM)]
        gathered=[None]*WORLD
        dist.all_gather_object(gathered,ledger)
        if rank==0:
            for r,entries in enumerate(gathered):
                chk(len(entries),STEPS*ACCUM,'rank ledger length')
                for step in range(STEPS):
                    for micro in range(ACCUM):
                        chk(entries[step*ACCUM+micro],
                            (step+1,micro,order[step*GLOBAL_BATCH+r*ACCUM+micro]['row_index']),
                            'actual rank ledger mismatch')
            combined=[(step*GLOBAL_BATCH+r*ACCUM+m,gathered[r][step*ACCUM+m][2])
                      for step in range(STEPS) for r in range(WORLD) for m in range(ACCUM)]
            combined.sort()
            chk([rid for _,rid in combined],[r['row_index'] for r in order],'DDP reconstructed order')
            chk(len(set(rid for _,rid in combined)),ROWS,'DDP unique coverage')
            report['status']='CPU_GLOO_DDP_ORDER_AUDITED_NOT_TRAINER_INTEGRATED_NO_M3_TRAINING'
            atomic_report(out,report,None,None,ddp=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['audit-order','audit-ddp'])
    p.add_argument('--root',default='.')
    p.add_argument('--prepared',default='data/training/prepared/balanced_multilingual')
    p.add_argument('--candidate',default='results/d-04/curriculum-audit-v5')
    p.add_argument('--output',default=None)
    a=p.parse_args()
    root=Path(a.root).resolve();prepared=resolve(root,a.prepared);candidate=resolve(root,a.candidate)
    out=resolve(root,a.output or ('results/d-04/distributed-order-audit-v6-ddp' if a.command=='audit-ddp' else 'results/d-04/distributed-order-audit-v6'))
    order,report=load_audit(root,prepared,candidate)
    if a.command=='audit-order':atomic_report(out,report,root,candidate)
    else:audit_ddp(order,report,out)

if __name__=='__main__':main()
