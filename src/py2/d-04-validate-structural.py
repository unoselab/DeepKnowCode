#!/usr/bin/env python3
"""D-04 v4: read-only AST canonicalization and similarity sensitivity audit.

Audits frozen D-04 v2 scores without changing labels, pair selection or score files.
BCB source pairs are externally established benchmark pairs, NOT NiCad detections.
"""
import argparse
import collections
import datetime
import difflib
import hashlib
import importlib.util
import json
import math
import random
import statistics
from pathlib import Path
from zoneinfo import ZoneInfo

V2_PATH=Path(__file__).with_name('d-04-score-curriculum.py')
spec=importlib.util.spec_from_file_location('d04_scoring_v2',V2_PATH)
v2=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

def quantiles(a):
    if not a:return {'n':0,'mean':None,'median':None,'p95':None}
    return {'n':len(a),'mean':statistics.mean(a),'median':statistics.median(a),'p95':v2.percentile(a,.95)}

def score_for_pair(a,b,allow_nojunk=True):
    lexical=v2.jaccard(a['leaves'],b['leaves'])
    types1=a['types'];types2=b['types']
    structural=difflib.SequenceMatcher(None,types1,types2,autojunk=True).ratio()
    out={'lexical':lexical,'structural_autojunk_true':structural,
         'composite_autojunk_true':v2.harmonic(lexical,structural),
         'ast_lengths':[len(types1),len(types2)]}
    if allow_nojunk:
        structural2=difflib.SequenceMatcher(None,types1,types2,autojunk=False).ratio()
        out.update({'structural_autojunk_false':structural2,
                    'composite_autojunk_false':v2.harmonic(lexical,structural2)})
    else:
        out.update({'structural_autojunk_false':None,'composite_autojunk_false':None})
    return out

def full_tree_features(parser,code,mode):
    if mode=='wrapped_class':t='class D04SyntheticClass {\n'+code+'\n}'
    elif mode=='wrapped_method':t='class D04SyntheticClass { void d04SyntheticMethod() {\n'+code+'\n} }'
    else:t=code
    root=parser.parse(t.encode('utf8')).root_node
    types=[];leaves=[];todo=[root]
    while todo:
        n=todo.pop()
        if n.type in v2.COMMENTS:continue
        types.append(n.type)
        if not n.children:
            value=n.text.decode('utf8',errors='replace').strip()
            if value:leaves.append(value)
        else:todo.extend(reversed(n.children))
    return {'types':types,'leaves':leaves,'has_error':bool(root.has_error)}

def sample_quantiles(rows,k,seed):
    """Deterministic low/median/high/random mixture; stratification by corpus and label."""
    good=[r for r in rows if r['status']=='OK' and r['composite_similarity'] is not None]
    if not good:return []
    sorted_rows=sorted(good,key=lambda x:(x['composite_similarity'],x['row_index']))
    rng=random.Random(seed)
    quota=min(k,len(good));counts=[quota//4]*4
    for i in range(quota%4):counts[i]+=1
    pools=[sorted_rows[:max(1,len(good)//4)],
           sorted_rows[len(good)//3:2*len(good)//3],
           sorted_rows[-max(1,len(good)//4):],
           sorted_rows]
    selected={}
    for pool,count in zip(pools,counts):
        eligible=[r for r in pool if r['row_index'] not in selected]
        for r in rng.sample(eligible,min(len(eligible),count)):
            selected[r['row_index']]=r
    if len(selected)<quota:
        remaining=[r for r in sorted_rows if r['row_index'] not in selected]
        for r in rng.sample(remaining,min(len(remaining),quota-len(selected))):
            selected[r['row_index']]=r
    return [selected[i] for i in sorted(selected)]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.')
    ap.add_argument('--prepared',default=v2.DEFAULT_PREPARED)
    ap.add_argument('--v2-results',default='results/d-04/difficulty-audit-v2')
    ap.add_argument('--v3-results',default='results/d-04/difficulty-diagnostics-v3')
    ap.add_argument('--output',default='results/d-04/structural-audit-v4')
    ap.add_argument('--samples-per-group',type=int,default=24)
    ap.add_argument('--wrapper-pairs',type=int,default=48)
    ap.add_argument('--max-autojunk-nodes',type=int,default=900)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--metadata-only',action='store_true')
    args=ap.parse_args()
    if args.samples_per_group<1 or args.wrapper_pairs<0 or args.max_autojunk_nodes<1:ap.error('Invalid sample bounds')
    root=Path(args.root).resolve()
    def locate(value):
        p=Path(value)
        return p if p.is_absolute() else root/p
    prepared=locate(args.prepared);inp=locate(args.v2_results);v3inp=locate(args.v3_results);out=locate(args.output)
    if out.exists():raise FileExistsError(f'Never overwrite an existing v4 output: {out}')
    m=json.loads((inp/'manifest.json').read_text(encoding='utf8'))
    scores_file=inp/'difficulty_scores.jsonl'; audit_file=inp/'difficulty_audit.json'
    for p,k in [(scores_file,'difficulty_scores_sha256'),(audit_file,'difficulty_audit_sha256')]:
        if v2.sha256(p)!=m[k]:raise ValueError(f'v2 source integrity mismatch: {p}')
    for p,k in [(prepared/'train.jsonl','train_sha256'),(prepared/'validation.jsonl','validation_sha256'),(prepared/'manifest.json','source_manifest_sha256')]:
        if v2.sha256(p)!=m[k]:raise ValueError(f'Prepared source integrity mismatch: {p}')
    pm=json.loads((prepared/'manifest.json').read_text(encoding='utf8'))
    if pm.get('status')!='PREPARED_NOT_TRAINED':raise ValueError('Not a frozen prepared dataset')
    rows=list(v2.read_jsonl(scores_file));train=list(v2.read_jsonl(prepared/'train.jsonl'))
    if len(rows)!=19464 or len(train)!=19464:raise ValueError('Expected 19464 pairs')
    for i,(r,t) in enumerate(zip(rows,train)):
        if r['row_index']!=i or any(str(r[k])!=str(t[k]) for k in ('corpus','id1','id2','label')):
            raise ValueError(f'Frozen score-to-pair join failed row={i}')
    v3_hash=None
    if v3inp.is_dir():
        vm=json.loads((v3inp/'manifest.json').read_text(encoding='utf8'))
        for p,k in [('diagnostic_report.json','report_sha256'),('diagnostic_examples.jsonl','examples_sha256')]:
            if v2.sha256(v3inp/p)!=vm[k]:raise ValueError('v3 source hash mismatch')
        v3_hash=v2.sha256(v3inp/'manifest.json')
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['corpus'],r['label'])].append(r)
    selected=[];selected_ids=set()
    for j,(key,items) in enumerate(sorted(groups.items())):
        for r in sample_quantiles(items,args.samples_per_group,args.seed+j):
            if r['row_index'] not in selected_ids:selected.append(r);selected_ids.add(r['row_index'])
    wrapped=[r for r in rows if r['status']=='OK' and any(x.startswith('wrapped_') for x in r['parse_modes'])]
    rng=random.Random(args.seed+999)
    for r in rng.sample(wrapped,min(len(wrapped),args.wrapper_pairs)):
        if r['row_index'] not in selected_ids:selected.append(r);selected_ids.add(r['row_index'])
    failing=[r for r in rows if r['status']!='OK']
    for r in failing:
        if r['row_index'] not in selected_ids:selected.append(r);selected_ids.add(r['row_index'])
    selected.sort(key=lambda r:r['row_index'])
    started=datetime.datetime.now(ZoneInfo('America/Chicago')).isoformat()
    results=[]
    if not args.metadata_only:
        codes={}
        for corpus in v2.CORPORA:
            path=v2.resolve_input(pm['data_paths'][corpus]['data.jsonl'],root)
            if v2.sha256(path)!=pm['data_sha256'][corpus]['data.jsonl']:raise ValueError(f'Clean code hash mismatch: {corpus}')
            codes[corpus]=v2.read_codes(path)
        parsers={'java':v2.parser_for('java'),'python':v2.parser_for('python')}
        for n,r in enumerate(selected,1):
            lang='python' if r['corpus']=='python_mixed' else 'java'
            pa=parsers[lang]; features=[];full_features=[];details=[]
            for side,fid in enumerate((r['id1'],r['id2'])):
                code=codes[r['corpus']][str(fid)]
                f=v2.parse_features(pa,code,lang)
                mode=f['parse_mode']
                if mode!=r['parse_modes'][side]:raise ValueError(f'Parser mode drift on row {r["row_index"]}')
                features.append(f)
                ff=full_tree_features(pa,code,mode) if f['error'] is None else None
                full_features.append(ff)
                details.append({'side':side,'code_sha256':hashlib.sha256(code.encode('utf8')).hexdigest(),
                    'code_chars':len(code),'parse_mode':mode,'code_ast_nodes':len(f.get('types',[])),
                    'full_ast_nodes':len(ff['types']) if ff else None,
                    'synthetic_node_overhead':len(ff['types'])-len(f['types']) if ff else None,
                    'parse_error':f['error'],'error_nodes':f.get('error_nodes',[]),
                    'code_leaf_tokens':f.get('token_count'),
                    'truncated_fragment':bool(f['error'])})
            if any(f['error'] for f in features):
                metrics=None;full=None;status='AST_PARSE_FAILED'
            else:
                maxlen=max(len(f['types']) for f in features)
                metrics=score_for_pair(*features,allow_nojunk=maxlen<=args.max_autojunk_nodes)
                full=score_for_pair(*full_features,allow_nojunk=False)
                status='OK'
                if abs(metrics['composite_autojunk_true']-r['composite_similarity'])>1e-4:
                    raise ValueError(f'Frozen v2 similarity drift row={r["row_index"]}')
            results.append({'row_index':r['row_index'],'corpus':r['corpus'],'label':r['label'],
                'id1':r['id1'],'id2':r['id2'],'BCB_label_provenance':'external_benchmark_not_nicad' if r['corpus']=='java_source' else 'clean_dataset_label_unmodified',
                'v2_similarity':r['composite_similarity'],'v2_lexical':r['lexical_similarity'],
                'v2_structural':r['structural_similarity'],'v2_score':r['reduced_4d_score'],
                'metric_status':status,'sides':details,'canonical_metrics':metrics,
                'full_wrapper_ast_metrics':full,
                'wrapper_contamination_delta':full['composite_autojunk_true']-metrics['composite_autojunk_true'] if full and any(x.startswith('wrapped_') for x in r['parse_modes']) else None})
            if n%40==0:print(f'[D-04 v4] DIAGNOSED {n}/{len(selected)}',flush=True)
    else:
        results=[{'row_index':r['row_index'],'corpus':r['corpus'],'label':r['label'],
                 'status':r['status'],'v2_similarity':r['composite_similarity'],
                 'parse_modes':r['parse_modes']} for r in selected]
    buckets=collections.defaultdict(list)
    for r in results:buckets[(r['corpus'],r['label'])].append(r)
    summary={}
    for (corpus,label),items in sorted(buckets.items()):
        valid=[x for x in items if x.get('canonical_metrics') is not None]
        deltas=[x['canonical_metrics']['composite_autojunk_false']-x['canonical_metrics']['composite_autojunk_true'] for x in valid if x['canonical_metrics']['composite_autojunk_false'] is not None]
        contamination=[x['wrapper_contamination_delta'] for x in valid if x['wrapper_contamination_delta'] is not None]
        summary[f'{corpus}/label={label}']={'sampled_pairs':len(items),'valid_sampled_pairs':len(valid),
            'autojunk_delta':quantiles(deltas),'full_wrapper_vs_canonical_delta':quantiles(contamination),
            'mean_v2_similarity':statistics.mean(x['v2_similarity'] for x in items if x['v2_similarity'] is not None) if any(x['v2_similarity'] is not None for x in items) else None}
    report={'version':'v4','status':'STRUCTURAL_AUDIT_ONLY_NO_SCORE_CHANGE',
        'created_chicago':started,'rows_verified':len(rows),'selected_pair_count':len(selected),
        'parsed_sample_count':sum(x.get('metric_status')=='OK' for x in results),
        'unparseable_sample_count':sum(x.get('metric_status')=='AST_PARSE_FAILED' for x in results),
        'sample_groups':summary,'source_frozen_scores_sha256':v2.sha256(scores_file),
        'prepared_train_sha256':m['train_sha256'],'prepared_validation_sha256':m['validation_sha256'],
        'provenance_note':'BigCloneBench positive labels originate from benchmark curation including other detectors and manual inspection, NOT NiCad. No relabeling performed.',
        'method':'Reuses v2 parse_features canonical subtree and recomputes pair similarity on deterministic stratified samples. Compares synthetic full-wrapper AST versus code-only AST, and bounded autojunk=False sensitivity.',
        'limitations':['Diagnostic samples are not the whole population.',
          'v2 canonical subtree extraction already excludes most wrapper nodes; an exact zero-contamination guarantee is not established.',
          'No claim of Type-4 semantic equivalence.',
          'Negative labels may be unverified candidates.',
          'No M1 loss validation, no model training, no curriculum scheduling.',
          'Full-wrapper AST comparison is a deliberately contaminated counterfactual, not a production scoring option.']}
    out.mkdir(parents=True,exist_ok=False)
    v2.dump(out/'structural_audit.json',report)
    with (out/'structural_examples.jsonl').open('w',encoding='utf8') as f:
        for row in results:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    v2.dump(out/'manifest.json',{'version':'v4','status':report['status'],
        'source_v2_scores_sha256':v2.sha256(scores_file),
        'source_v3_manifest_sha256':v3_hash,'prepared_train_sha256':m['train_sha256'],
        'structural_audit_sha256':v2.sha256(out/'structural_audit.json'),
        'structural_examples_sha256':v2.sha256(out/'structural_examples.jsonl'),
        'created_chicago':started})
    print(json.dumps({'status':report['status'],'output':str(out),'verified_pairs':len(rows),
            'diagnosed_pairs':len(results),'metadata_only':args.metadata_only},sort_keys=True),flush=True)

if __name__=='__main__':main()
