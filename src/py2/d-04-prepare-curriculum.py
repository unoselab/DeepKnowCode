#!/usr/bin/env python3
"""D-04 v5: audit-only candidate scores and reproducible curriculum order.

Uses existing D-04 v2 scores (no AST recomputation, no autojunk experiment).
Read-only inputs; output is written to a fresh directory. NO M3 TRAINING.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

CORPORA=('java_source','java_mixed','python_mixed')
EXPECTED={('java_source',0):6000,('java_source',1):6000,('java_mixed',0):1866,
          ('java_mixed',1):1866,('python_mixed',0):1866,('python_mixed',1):1866}
# Each phase is an exact multiple of global batch 24, all group cells remain label-balanced.
PHASES=[('foundation',270,{'java_source':5184,'java_mixed':648,'python_mixed':648}),
        ('adaptation',270,{'java_source':4016,'java_mixed':1232,'python_mixed':1232}),
        ('specialization',271,{'java_source':2800,'java_mixed':1852,'python_mixed':1852})]
W4={'similarity':.45,'cf':.25,'length':.20,'df':.10}
W3={'similarity':.50,'cf':.30,'length':.20}


def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_jsonl(p):
    with Path(p).open(encoding='utf8') as f:
        for line in f:
            if line.strip():yield json.loads(line)


def write_jsonl(p,rows):
    with Path(p).open('w',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')


def write_json(p,d):
    with Path(p).open('w',encoding='utf8') as f:
        json.dump(d,f,indent=2,ensure_ascii=False,sort_keys=True);f.write('\n')


def pkey(row):
    return (row['corpus'],str(row['id1']),str(row['id2']),int(row['label']))


def score(weights,metrics):
    if any(metrics[k] is None for k in weights):return None
    return round(sum(weights[k]*metrics[k] for k in weights),7)


def rank_corr(rows,field_a,field_b):
    # Pearson correlation of average ranks (Spearman with ties), per corpus/label.
    vals=[(r[field_a],r[field_b]) for r in rows if r[field_a] is not None and r[field_b] is not None]
    if len(vals)<3:return None
    def ranks(col):
        v=[t[col] for t in vals]; indices=sorted(range(len(v)),key=lambda i:v[i]); out=[0.0]*len(v); k=0
        while k<len(indices):
            j=k+1
            while j<len(indices) and v[indices[j]]==v[indices[k]]:j+=1
            for idx in indices[k:j]:out[idx]=(k+j-1)/2
            k=j
        return out
    a,b=ranks(0),ranks(1); ma=statistics.mean(a);mb=statistics.mean(b)
    numer=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    da=sum((x-ma)**2 for x in a);db=sum((y-mb)**2 for y in b)
    return round(numer/(da*db)**.5,6) if da and db else None


def run(args):
    root=Path(args.root).resolve()
    def abs_path(x):
        p=Path(x);return p if p.is_absolute() else root/p
    prepared=abs_path(args.prepared); scored=abs_path(args.scored); out=abs_path(args.output)
    for p in (prepared/'manifest.json',prepared/'train.jsonl',prepared/'validation.jsonl',scored/'manifest.json',scored/'difficulty_scores.jsonl'):
        if not p.is_file():raise FileNotFoundError(p)
    if out.exists():raise FileExistsError(f'Refusing to overwrite existing output: {out}')
    pm=json.loads((prepared/'manifest.json').read_text()); sm=json.loads((scored/'manifest.json').read_text())
    tf=prepared/'train.jsonl';vf=prepared/'validation.jsonl';sf=scored/'difficulty_scores.jsonl'
    if sha256(tf)!=pm['sample']['train']['sha256'] or sha256(vf)!=pm['sample']['validation']['sha256']:
        raise ValueError('Prepared input hashes do not match manifest')
    if sha256(sf)!=sm['difficulty_scores_sha256'] or sha256(tf)!=sm['train_sha256'] or sha256(vf)!=sm['validation_sha256']:
        raise ValueError('Scoring input hashes do not match v2 manifest')
    if sha256(prepared/'manifest.json')!=sm['source_manifest_sha256']:
        raise ValueError('Prepared manifest is not the source of v2 scores')
    training=list(read_jsonl(tf)); validation=list(read_jsonl(vf)); raw_scores=list(read_jsonl(sf))
    if len(training)!=19464 or len(validation)!=512 or len(raw_scores)!=19464:raise ValueError('Unexpected row counts')
    if len(set(map(pkey,training)))!=len(training) or set(map(pkey,training))&set(map(pkey,validation)):
        raise ValueError('Duplicate or train-validation overlap')
    if Counter((r['corpus'],int(r['label'])) for r in training)!=EXPECTED:raise ValueError('Dataset group budget mismatch')
    rows=[];groups=defaultdict(list);missing=Counter()
    for i,(tr,sr) in enumerate(zip(training,raw_scores)):
        if sr['row_index']!=i or pkey(sr)!=pkey(tr):raise ValueError(f'Score mapping mismatched at row {i}')
        m={'similarity':sr.get('label_aware_similarity_difficulty'),
           'cf':sr.get('cf_normalized'),'length':sr.get('length_normalized'),
           'df':sr.get('df_proxy_normalized')}
        s4=score(W4,m);s3=score(W3,m)
        if s4 is None or s3 is None:missing[(tr['corpus'],int(tr['label']))]+=1
        r={'row_index':i,'corpus':tr['corpus'],'id1':str(tr['id1']),
           'id2':str(tr['id2']),'label':int(tr['label']),'status_v2':sr['status'],
           'raw_reduced_4d':s4,'raw_reduced_3d':s3,
           'original_5d':None,'detection_5d':None,
           'dataflow_note':'IDENTIFIER_DIVERSITY_PROXY_NOT_DEF_USE',
           'architecture_note':'UNAVAILABLE_NO_VERIFIED_PROVENANCE'}
        rows.append(r);groups[(r['corpus'],r['label'])].append(r)
    # Fallback computed from measured scores in the SAME corpus/label only;
    # record explicit fallback markers, never change the original dataset.
    fallback=[]
    for group,items in groups.items():
        for metric,rawfield,outputfield in [('4d','raw_reduced_4d','score_4d'),('3d','raw_reduced_3d','score_3d')]:
            clean=[r[rawfield] for r in items if r[rawfield] is not None]
            if not clean:raise ValueError(f'No valid scores to impute for {group}/{metric}')
            median=round(statistics.median(clean),7)
            for r in items:
                if r[rawfield] is None:
                    r[outputfield]=median;r[f'{metric}_fallback']='GROUP_MEDIAN_AST_PARSE_FAILED'
                    fallback.append({'row_index':r['row_index'],'corpus':r['corpus'],
                                     'label':r['label'],'metric':metric,'value':median})
                else:r[outputfield]=r[rawfield];r[f'{metric}_fallback']=None
    # Quantile tiers are corpus x label ranks, for audit only; stable tie handling.
    for group,items in groups.items():
        for metric in ('4d','3d'):
            ordered=sorted(items,key=lambda r:(r[f'score_{metric}'],r['row_index']))
            n=len(ordered)
            for i,r in enumerate(ordered):r[f'tier_{metric}']=min(4,4*i//n+1)
    # Candidate order explicitly preliminary: 4D score, ascending separately
    # per corpus+label. Uses 811 x 24 exact samples; no resampling.
    streams={g:sorted(items,key=lambda r:(r['score_4d'],r['row_index'])) for g,items in groups.items()}
    cursors={g:0 for g in groups};ordering=[];phase_counts=[];step_start=1
    for phase,steps,by_corpus in PHASES:
        phase_parts=[];counters={}
        for c in CORPORA:
            n=by_corpus[c]
            if n%2:raise ValueError('Unbalanced label quota')
            for label in (0,1):
                g=(c,label);take=n//2;start=cursors[g];part=streams[g][start:start+take]
                if len(part)!=take:raise ValueError('Insufficient group examples')
                cursors[g]+=take;counters[f'{c}/label={label}']=take
                for j,r in enumerate(part):
                    # Uniformly distribute groups across the phase; within each
                    # group examples move from easier to harder.
                    phase_parts.append(((j+.5)/take,CORPORA.index(c),label,j,r))
        phase_parts.sort(key=lambda x:x[:4])
        if len(phase_parts)!=steps*24:raise ValueError('Phase budget does not match optimizer steps')
        for n,(_,_,_,_,r) in enumerate(phase_parts):
            ordering.append({'sequence_index':len(ordering),'phase':phase,
                             'optimizer_step':step_start+n//24,
                             'batch_slot':n%24,'row_index':r['row_index'],
                             'corpus':r['corpus'],'label':r['label'],
                             'id1':r['id1'],'id2':r['id2'],
                             'difficulty_score':r['score_4d'],
                             'difficulty_tier':r['tier_4d'],
                             'fallback':r['4d_fallback']})
        phase_counts.append({'phase':phase,'steps':steps,'first_optimizer_step':step_start,
                             'last_optimizer_step':step_start+steps-1,'pairs':len(phase_parts),
                             'groups':counters})
        step_start+=steps
    if len(ordering)!=19464 or len(set(r['row_index'] for r in ordering))!=19464 or step_start!=812:
        raise ValueError('Ordering coverage failed')
    if any(cursors[g]!=len(groups[g]) for g in groups):raise ValueError('Missing pairs in phase allocation')
    # Strict coverage and schedule audits; permutation does not alter train bytes.
    out.mkdir(parents=True)
    write_jsonl(out/'difficulty_candidates.jsonl',rows)
    write_jsonl(out/'curriculum_order_candidate.jsonl',ordering)
    correlations={f'{c}/label={y}':rank_corr(items,'score_4d','score_3d') for (c,y),items in sorted(groups.items())}
    profile={f'{c}/label={y}':{'count':len(items),
              'missing_raw_4d':sum(r['raw_reduced_4d'] is None for r in items),
              'mean_4d':round(statistics.mean(r['score_4d'] for r in items),6),
              'mean_3d':round(statistics.mean(r['score_3d'] for r in items),6)}
              for (c,y),items in sorted(groups.items())}
    audit={'status':'AUDIT_ONLY_NO_PRIMARY_SCORE_SELECTED_NO_M3_TRAINING',
           'created_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
           'training_pairs':len(training),'validation_pairs':len(validation),
           'valid_raw_scores':len(training)-sum(missing.values()),'missing_raw_scores':sum(missing.values()),
           'fallback_records':fallback,'phase_counts':phase_counts,
           'corpus_label_profile':profile,'spearman_4d_vs_3d':correlations,
           'original_5d_status':'NOT_COMPUTABLE: architecture and validated semantic taxonomy unavailable',
           'detection_5d_status':'NOT_COMPUTABLE: architecture unavailable',
           'score_candidates':{'reduced_4d':W4,'reduced_3d_without_df_proxy':W3},
           'order_is_candidate':True,
           'limitations':['BCB positive labels originate from benchmark/tool/manual curation, not NiCad.',
             '3D and 4D are heuristic candidates, not empirically validated detection difficulty.',
             'DF is identifier diversity proxy, not true def-use; 5D cannot be claimed complete.',
             'Two invalid AST pairs are retained via corpus/label median and explicitly flagged.',
             'No autojunk settings experiments are performed; existing v2 scores use defaults.',
             'The proposed ordering is a research candidate, not a verified distributed dataloader order.',
             'Phase scheduling changes both ordering and target proportion; effects not isolated.',
             'No model-based difficulty validation or held-out model selection has been performed.']}
    write_json(out/'difficulty_audit_v5.json',audit)
    manifest={'component':'D-04c','version':'v5','status':audit['status'],
              'source_prepared_manifest_sha256':sha256(prepared/'manifest.json'),
              'train_sha256':sha256(tf),'validation_sha256':sha256(vf),
              'source_scoring_manifest_sha256':sha256(scored/'manifest.json'),
              'source_scores_sha256':sha256(sf),
              'difficulty_candidates_sha256':sha256(out/'difficulty_candidates.jsonl'),
              'curriculum_order_candidate_sha256':sha256(out/'curriculum_order_candidate.jsonl'),
              'difficulty_audit_sha256':sha256(out/'difficulty_audit_v5.json')}
    write_json(out/'manifest.json',manifest)
    print(json.dumps({'status':audit['status'],'pairs':len(rows),
        'raw_missing':sum(missing.values()),'order_pairs':len(ordering),
        'out':str(out),'candidate_order_sha256':manifest['curriculum_order_candidate_sha256']},sort_keys=True))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.')
    ap.add_argument('--prepared',default='data/training/prepared/balanced_multilingual')
    ap.add_argument('--scored',default='results/d-04/difficulty-audit-v2')
    ap.add_argument('--output',default='results/d-04/curriculum-audit-v5')
    run(ap.parse_args())

if __name__=='__main__':main()
