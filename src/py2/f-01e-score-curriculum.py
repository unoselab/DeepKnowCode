#!/usr/bin/env python3
"""F-01e v1: strict Reduced-4D score ingestion, 3-phase curriculum ordering, audit.

No new difficulty metric is inferred from tokens/identifiers. No GPU training.
All experiment configuration is required CLI arguments from standalone Shell.
"""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v1'
CORPORA = ('java_source', 'java_mixed', 'python_mixed')

def now():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()

def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1048576), b''):
            h.update(buf)
    return h.hexdigest()

def write_new(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.write('\n')

def records(path):
    with path.open('rb') as f:
        for i, raw in enumerate(f):
            if not raw.strip():
                raise ValueError(f'EMPTY_RECORD: {path}:{i+1}')
            yield i, json.loads(raw), sha_bytes(raw.rstrip(b'\r\n'))

def identity(row):
    return (str(row['corpus']), str(row['id1']), str(row['id2']), int(row['label']))

def keystr(k):
    return '|'.join(map(str,k))

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['inspect','prepare','audit'])
    for x in ('root','run-dir','prepared-run','m1-adapter','score-file','log-file'):
        p.add_argument('--'+x, required=True, type=Path)
    for x in ('train-sha256','validation-sha256','manifest-sha256','m1-sha256','model-id','model-revision','model-mode','gpu-ids','precision','score-policy','score-field','score-provenance','score-sha256','log-prefix','log-timestamp-format'):
        p.add_argument('--'+x, required=True)
    for x in ('seed','world-size','per-device-batch','gradient-accumulation','global-batch','max-seq-length','optimizer-steps','train-count','validation-count','source-count','java-target-count','python-target-count'):
        p.add_argument('--'+x, required=True, type=int)
    p.add_argument('--score-component-fields', nargs=4, required=True)
    p.add_argument('--score-weights', nargs=4, type=float, required=True)
    p.add_argument('--score-tolerance', type=float, required=True)
    p.add_argument('--phase-names', nargs=3, required=True)
    for x in ('phase-steps','phase-source','phase-java','phase-python'):
        p.add_argument('--'+x, nargs=3, type=int, required=True)
    return p

def validate(a):
    if a.score_policy != 'reduced-4d-externally-verified-v1':
        raise ValueError('SCORE_POLICY_UNSUPPORTED')
    if a.score_field != 'reduced_4d_score':
        raise ValueError('SCORE_FIELD_UNSUPPORTED')
    if a.score_component_fields!=['d_sim','d_cf','d_len','d_df'] or any(not math.isfinite(w) or w<0 for w in a.score_weights) or abs(sum(a.score_weights)-1)>1e-12:
        raise ValueError('SCORE_4D_COMPONENT_CONFIG_INVALID')
    if not 0<a.score_tolerance<=1e-4:
        raise ValueError('SCORE_TOLERANCE_INVALID')
    if not a.score_provenance or a.score_provenance == 'UNVERIFIED':
        raise ValueError('SCORE_PROVENANCE_REQUIRED')
    if a.world_size * a.per_device_batch * a.gradient_accumulation != a.global_batch:
        raise ValueError('GLOBAL_BATCH_MISMATCH')
    if len(a.gpu_ids.split(',')) != a.world_size:
        raise ValueError('GPU_WORLD_MISMATCH')
    if len(set(a.gpu_ids.split(','))) != a.world_size:
        raise ValueError('DUPLICATE_GPU_ID')
    if a.optimizer_steps * a.global_batch != a.train_count:
        raise ValueError('TOTAL_STEP_BUDGET_MISMATCH')
    if sum(a.phase_steps)!=a.optimizer_steps:
        raise ValueError('PHASE_STEPS_MISMATCH')
    if sum(a.phase_source)!=a.source_count or sum(a.phase_java)!=a.java_target_count or sum(a.phase_python)!=a.python_target_count:
        raise ValueError('PHASE_CORPUS_BUDGET_MISMATCH')
    for i in range(3):
        if a.phase_steps[i]*a.global_batch != a.phase_source[i]+a.phase_java[i]+a.phase_python[i]:
            raise ValueError('PHASE_TOTAL_MISMATCH')
        if any(v%2 for v in (a.phase_source[i],a.phase_java[i],a.phase_python[i])):
            raise ValueError('PHASE_LABEL_BUDGET_MISMATCH')
    for field in ('train_sha256','validation_sha256','manifest_sha256','m1_sha256','score_sha256'):
        v=getattr(a,field)
        if len(v)!=64 or any(c not in '0123456789abcdef' for c in v):
            raise ValueError('INVALID_HASH '+field)
    if a.model_mode != 'codellama-7b' or a.model_id!='meta-llama/CodeLlama-7b-hf':
        raise ValueError('MODEL_ID_MISMATCH')
    if a.max_seq_length<=0 or a.seed<0:
        raise ValueError('PARAMETER_INVALID')

def frozen(a):
    p=a.prepared_run/'prepared'
    expected={'train.jsonl':a.train_sha256,'validation.jsonl':a.validation_sha256,'manifest.json':a.manifest_sha256}
    for fn, h in expected.items():
        if sha(p/fn)!=h:raise ValueError('FROZEN_DATASET_SHA_MISMATCH '+fn)
    if sha(a.m1_adapter)!=a.m1_sha256:raise ValueError('FROZEN_M1_SHA_MISMATCH')
    m=json.loads((p/'manifest.json').read_text())
    if m['sample']['train']['sha256']!=a.train_sha256 or m['sample']['validation']['sha256']!=a.validation_sha256:
        raise ValueError('FROZEN_MANIFEST_DATASET_LINK_MISMATCH')
    counts=Counter(); dataset=[]; seen=set()
    for idx, row, linehash in records(p/'train.jsonl'):
        k=identity(row)
        if k in seen:raise ValueError('FROZEN_PAIR_DUPLICATE '+keystr(k))
        seen.add(k)
        if k[0] not in CORPORA or k[3] not in (0,1):raise ValueError('FROZEN_CORPUS_LABEL_INVALID')
        t=row['tokens']
        if not isinstance(t,int) or not 0<t<=a.max_seq_length:raise ValueError('FROZEN_TOKEN_CAP_INVALID')
        dataset.append({'index':idx,'identity':k,'tokens':t,'line_sha256':linehash})
        counts[k[0],k[3]]+=1
    if len(dataset)!=a.train_count:raise ValueError('FROZEN_TRAIN_COUNT_MISMATCH')
    for corp,n in [('java_source',a.source_count),('java_mixed',a.java_target_count),('python_mixed',a.python_target_count)]:
        for lab in (0,1):
            if counts[corp,lab]!=n//2:raise ValueError('FROZEN_CORPUS_LABEL_MISMATCH '+corp)
    nval=sum(1 for _ in records(p/'validation.jsonl'))
    if nval!=a.validation_count:raise ValueError('FROZEN_VALIDATION_COUNT_MISMATCH')
    return dataset,counts,expected

def load_scores(a,dataset):
    if not a.score_file.is_file():
        raise ValueError('SCORE_FILE_MISSING: supply F01E_SCORE_FILE with verified CodeLlama-effective-code Reduced-4D scores')
    if sha(a.score_file)!=a.score_sha256:
        raise ValueError('SCORE_FILE_SHA_MISMATCH')
    base={r['identity'] for r in dataset}
    scores={}
    for idx,row,_ in records(a.score_file):
        k=identity(row)
        if k not in base:raise ValueError(f'SCORE_PAIR_NOT_IN_FROZEN_DATASET row={idx+1} key={keystr(k)}')
        if k in scores:raise ValueError('SCORE_DUPLICATE_PAIR '+keystr(k))
        score=row.get(a.score_field)
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not (0<=score<=1):
            raise ValueError('SCORE_INVALID [0,1] '+keystr(k))
        components=[]
        for field in a.score_component_fields:
            val=row.get(field)
            if isinstance(val,bool) or not isinstance(val,(float,int)) or not math.isfinite(val) or not 0<=val<=1:
                raise ValueError('SCORE_COMPONENT_INVALID '+field+' '+keystr(k))
            components.append(float(val))
        recomputed=sum(w*c for w,c in zip(a.score_weights,components))
        if abs(recomputed-score)>a.score_tolerance:
            raise ValueError('SCORE_4D_FORMULA_MISMATCH '+keystr(k))
        if row.get('score_policy') not in (None,a.score_policy):raise ValueError('SCORE_POLICY_ROW_MISMATCH')
        if row.get('score_provenance') not in (None,a.score_provenance):raise ValueError('SCORE_PROVENANCE_ROW_MISMATCH')
        scores[k]=float(score)
    missing=base-scores.keys()
    if missing:raise ValueError(f'SCORE_COVERAGE_MISSING {len(missing)} first={keystr(sorted(missing)[0])}')
    return scores

def allocate_phase_rows(a,dataset,scores):
    grouped=defaultdict(list)
    for d in dataset:
        grouped[d['identity'][0],d['identity'][3]].append(d)
    for key in grouped:
        grouped[key].sort(key=lambda d:(scores[d['identity']],d['index']))
    phase_rows=[]
    for i,name in enumerate(a.phase_names):
        p=[]
        for corp,phase_n in [('java_source',a.phase_source[i]),('java_mixed',a.phase_java[i]),('python_mixed',a.phase_python[i])]:
            for label in (0,1):
                take=phase_n//2
                group=grouped[corp,label]
                if len(group)<take:raise ValueError('PHASE_GROUP_UNDERFLOW '+corp)
                p.extend(group[:take]);del group[:take]
        if len(p)!=a.phase_steps[i]*a.global_batch:raise ValueError('PHASE_COUNT_WRONG')
        phase_rows.append((name,p))
    if any(grouped.values()):raise ValueError('CURRICULUM_UNUSED_PAIRS')
    return phase_rows

def ordering(a,dataset,scores):
    phases=allocate_phase_rows(a,dataset,scores)
    output=[];phase_reports=[];step=0
    for pname,rows in phases:
        bylab={lab:sorted((r for r in rows if r['identity'][3]==lab),key=lambda r:(scores[r['identity']],r['index'])) for lab in (0,1)}
        if len(bylab[0])!=len(bylab[1]):raise ValueError('PHASE_LABEL_IMBALANCE')
        nsteps=len(rows)//a.global_batch
        phase_counts=Counter();phase_score=[];group_scores=defaultdict(list)
        for j in range(nsteps):
            step+=1
            batch=[]
            for lab in (0,1):
                batch.extend(bylab[lab][j*(a.global_batch//2):(j+1)*(a.global_batch//2)])
            if len(batch)!=a.global_batch:raise ValueError('BATCH_COUNT_MISMATCH')
            # Stable, label-interleaved batch; no DataLoader ordering claim yet.
            neg=[x for x in batch if x['identity'][3]==0]
            pos=[x for x in batch if x['identity'][3]==1]
            interleaved=[v for pair in zip(neg,pos) for v in pair]
            for b,r in enumerate(interleaved):
                k=r['identity'];s=scores[k]
                phase_counts[f'{k[0]}/{k[3]}']+=1;phase_score.append(s);group_scores[f'{k[0]}/{k[3]}'].append(s)
                output.append({'position':len(output),'global_step':step,'batch_position':b,'phase':pname,
                    'source_index':r['index'],'corpus':k[0],'id1':k[1],'id2':k[2],
                    'label':k[3],'tokens':r['tokens'],'source_row_sha256':r['line_sha256'],
                    'reduced_4d_score':s})
        phase_reports.append({'phase':pname,'steps':nsteps,'pairs':len(rows),
           'corpus_label_counts':dict(sorted(phase_counts.items())),
           'mean_difficulty':sum(phase_score)/len(phase_score),
           'group_mean_difficulty':{k:sum(v)/len(v) for k,v in sorted(group_scores.items())}})
    if len(output)!=a.train_count or step!=a.optimizer_steps:raise ValueError('ORDER_LENGTH_MISMATCH')
    if sorted(r['source_index'] for r in output)!=list(range(a.train_count)):
        raise ValueError('ORDER_PAIR_COVERAGE_FAILED')
    return output,phase_reports

def jsonl_bytes(rows):
    return b''.join((json.dumps(r,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode() for r in rows)

def audit_order(a,dataset,scores,order,phases):
    byidx={d['index']:d for d in dataset}
    stepcounts=Counter();step_labels=Counter()
    for r in order:
        d=byidx[r['source_index']]
        if (r['corpus'],r['id1'],r['id2'],r['label'])!=d['identity'] or r['source_row_sha256']!=d['line_sha256']:
            raise ValueError('ORDER_ROW_IDENTITY_DRIFT')
        if r['reduced_4d_score']!=scores[d['identity']]:raise ValueError('ORDER_SCORE_DRIFT')
        stepcounts[r['global_step']]+=1
        step_labels[r['global_step'],r['label']]+=1
    for step in range(1,a.optimizer_steps+1):
        if stepcounts[step]!=a.global_batch or (step_labels[step,0],step_labels[step,1])!=(a.global_batch//2,a.global_batch//2):
            raise ValueError('BATCH_BALANCE_FAIL '+str(step))
    progress={p['phase']:p['mean_difficulty'] for p in phases}
    for group in (f'{c}/{l}' for c in CORPORA for l in (0,1)):
        means=[p['group_mean_difficulty'][group] for p in phases]
        if means[0]>means[1]+1e-12 or means[1]>means[2]+1e-12:
            raise ValueError('GROUP_CURRICULUM_DIFFICULTY_NOT_MONOTONIC '+group)
    return {'coverage':'PASS','duplicates':0,'missing':0,'global_batches':a.optimizer_steps,
       'batch_label_balance':'12/12' if a.global_batch==24 else f'{a.global_batch//2}/{a.global_batch//2}',
       'phases':phases,'mean_difficulty_by_phase':progress,
       'monotonic_phase_means':all(phases[i]['mean_difficulty']<=phases[i+1]['mean_difficulty'] for i in range(2))}

def main():
    a=parser().parse_args();validate(a)
    dataset,counts,hashes=frozen(a)
    if a.command=='inspect':
        print(json.dumps({'status':'F01E_V1_FROZEN_INPUT_PASS_SCORE_PENDING',
            'training_pairs':len(dataset),'frozen_sha256':hashes,
            'score_file_exists':a.score_file.is_file(),
            'note':'Prepared JSONL has no code bodies/AST; verified external Reduced-4D score file required.'}),flush=True)
        return
    scores=load_scores(a,dataset)
    order,phases=ordering(a,dataset,scores)
    audit=audit_order(a,dataset,scores,order,phases)
    order_bytes=jsonl_bytes(order);order_hash=sha_bytes(order_bytes)
    config={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items() if k!='command'}
    manifest={'status':'F01E_V1_CURRICULUM_ORDER_VERIFIED','version':VERSION,'created_at_chicago':now(),
       'frozen_data_sha256':hashes,'m1_adapter_sha256':a.m1_sha256,
       'difficulty_input_sha256':a.score_sha256,'difficulty_provenance':a.score_provenance,
       'score_policy':a.score_policy,'order_sha256':order_hash,
       'gpu_training_enabled':False,'ddp_dataloader_consumption_verified':False,
       'effective_config':config,'audit':audit}
    orderpath=a.run_dir/'curriculum_order.jsonl'
    if a.command=='prepare':
        a.run_dir.mkdir(parents=True,exist_ok=False)
        with orderpath.open('xb') as f:f.write(order_bytes)
        write_new(a.run_dir/'curriculum_audit.json',audit)
        write_new(a.run_dir/'manifest.json',manifest)
    else:
        if sha(orderpath)!=order_hash:raise ValueError('AUDIT_EXISTING_ORDER_SHA_MISMATCH')
        old=json.loads((a.run_dir/'manifest.json').read_text())
        for k in ('frozen_data_sha256','m1_adapter_sha256','difficulty_input_sha256','difficulty_provenance','score_policy','order_sha256','audit','status','gpu_training_enabled','ddp_dataloader_consumption_verified'):
            if old[k]!=manifest[k]:raise ValueError('AUDIT_MANIFEST_DRIFT '+k)
        if json.loads((a.run_dir/'curriculum_audit.json').read_text())!=audit:
            raise ValueError('AUDIT_FILE_DRIFT')
    print(json.dumps({'status':manifest['status'],'run':str(a.run_dir),'train':len(order),
       'steps':a.optimizer_steps,'order_sha256':order_hash,
       'monotonic_phase_means':audit['monotonic_phase_means'],
       'gpu_training_enabled':False},sort_keys=True),flush=True)

if __name__=='__main__':main()
