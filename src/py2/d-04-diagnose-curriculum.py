#!/usr/bin/env python3
"""D-04 v3: read-only diagnostic of frozen v2 scores, parser fallbacks and similarity.

No re-scoring, training, score changes, curriculum generation, or source mutation.
"""
import argparse
import collections
import datetime
import difflib
import hashlib
import importlib.util
import json
import random
import statistics
from pathlib import Path
from zoneinfo import ZoneInfo

BASE=Path(__file__).with_name('d-04-score-curriculum.py')
spec=importlib.util.spec_from_file_location('d04_v2_source',BASE)
v2=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


def ordered_sample(rows, k, seed):
    rng=random.Random(seed)
    return rng.sample(rows,min(k,len(rows)))


def safe_excerpt(code, max_chars=240):
    return code[:max_chars].replace('\r','\\r').replace('\n','\\n')


def score_stats(rows, field):
    vals=[r[field] for r in rows if r.get(field) is not None]
    return {'count':len(vals),'mean':statistics.mean(vals) if vals else None,
            'median':statistics.median(vals) if vals else None}


def parse_details(parser,code):
    tree=parser.parse(code.encode('utf-8'))
    found=[]; todo=[tree.root_node]
    while todo and len(found)<8:
        node=todo.pop()
        if node.type=='ERROR' or node.is_missing:
            found.append({'type':node.type,'missing':bool(node.is_missing),
                          'row':node.start_point.row,'column':node.start_point.column,
                          'excerpt':safe_excerpt(code.splitlines()[node.start_point.row])
                          if node.start_point.row < len(code.splitlines()) else ''})
        todo.extend(reversed(node.children))
    return {'has_error':bool(tree.root_node.has_error),'root_type':tree.root_node.type,
            'errors':found}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.')
    ap.add_argument('--prepared',default=v2.DEFAULT_PREPARED)
    ap.add_argument('--v2-results',default='results/d-04/difficulty-audit-v2')
    ap.add_argument('--output',default='results/d-04/difficulty-diagnostics-v3')
    ap.add_argument('--summary-only',action='store_true',help='Inspect v2 scores without clean corpus or AST parser')
    ap.add_argument('--wrapper-samples',type=int,default=30)
    ap.add_argument('--source-samples',type=int,default=60)
    ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args()
    if min(args.wrapper_samples,args.source_samples)<0:ap.error('sample counts must be nonnegative')
    root=Path(args.root).resolve(); inp=root/args.v2_results; out=root/args.output
    prepared=root/args.prepared
    if out.exists(): raise FileExistsError(f'Output exists; preserve old diagnostics: {out}')
    if out.resolve()==inp.resolve() or inp.resolve() in out.resolve().parents:raise ValueError('Output cannot be inside v2 results')
    m=json.loads((inp/'manifest.json').read_text()); scoresfile=inp/'difficulty_scores.jsonl'
    if v2.sha256(scoresfile)!=m['difficulty_scores_sha256']:raise ValueError('v2 score SHA256 mismatch')
    if v2.sha256(inp/'difficulty_audit.json')!=m['difficulty_audit_sha256']:raise ValueError('v2 audit SHA256 mismatch')
    scores=list(v2.read_jsonl(scoresfile))
    if len(scores)!=19464 or [r['row_index'] for r in scores]!=list(range(19464)):
        raise ValueError('Expected exactly 19464 consecutive v2 score records')
    if v2.sha256(prepared/'train.jsonl')!=m['train_sha256'] or v2.sha256(prepared/'validation.jsonl')!=m['validation_sha256']:
        raise ValueError('Frozen prepared train/validation SHA256 mismatch')
    meta=json.loads((prepared/'manifest.json').read_text())
    if v2.sha256(prepared/'manifest.json')!=m['source_manifest_sha256']:
        raise ValueError('Prepared manifest SHA256 mismatch')
    groups=collections.defaultdict(list)
    for r in scores:groups[(r['corpus'],r['label'])].append(r)
    group_report={}
    for (c,y),items in sorted(groups.items()):
        group_report[f'{c}/label={y}']={'n':len(items),
            'composite_similarity':score_stats(items,'composite_similarity'),
            'lexical_similarity':score_stats(items,'lexical_similarity'),
            'structural_similarity':score_stats(items,'structural_similarity'),
            'reduced_4d':score_stats(items,'reduced_4d_score'),
            'ast_parse_failures':sum(r['status']!='OK' for r in items),
            'wrapped_pair_count':sum(any(x.startswith('wrapped_') for x in r['parse_modes']) for r in items)}
    missing=[r for r in scores if r['status']!='OK']
    sensitivity=[]
    for r in scores:
        if r['composite_similarity_autojunk_false'] is not None:
            sensitivity.append({ 'corpus':r['corpus'],'label':r['label'],
                'delta':r['composite_similarity_autojunk_false']-r['composite_similarity']})
    slabel={}
    for (c,y),items in sorted(groups.items()):
        ds=[r['composite_similarity_autojunk_false']-r['composite_similarity'] for r in items
            if r['composite_similarity_autojunk_false'] is not None]
        slabel[f'{c}/label={y}']={'n':len(ds),'mean_delta':statistics.mean(ds) if ds else None,
            'delta_gt_0.1':sum(d>.1 for d in ds)}
    report={'status':'DIAGNOSTIC_ONLY_NO_SCORE_CHANGE', 'version':'v3',
        'created_chicago':datetime.datetime.now(ZoneInfo('America/Chicago')).isoformat(),
        'v2_scores_sha256':v2.sha256(scoresfile),'train_sha256':m['train_sha256'],
        'validation_sha256':m['validation_sha256'],'groups':group_report,
        'remaining_failures':[{'row_index':r['row_index'],'corpus':r['corpus'],
            'id1':r['id1'],'id2':r['id2'],'label':r['label'],
            'parse_modes':r['parse_modes'],'parse_error_details':r['parse_error_details']} for r in missing],
        'autojunk_v2_deterministic_sample':slabel,
        'limitations':['No M1 per-example loss validation.','Negative labels are candidate labels.',
            'Wrapper parsing is syntactic only; semantic equivalence not verified.',
            'No original/detection-aware full 5D scoring because architectural provenance is unverified.',
            'No score changes or M3 curriculum/training.']}
    examples=[]
    if not args.summary_only:
        codes={}
        for corpus in v2.CORPORA:
            p=v2.resolve_input(meta['data_paths'][corpus]['data.jsonl'],root)
            if v2.sha256(p)!=meta['data_sha256'][corpus]['data.jsonl']:
                raise ValueError(f'Clean corpus hash mismatch {corpus}')
            codes[corpus]=v2.read_codes(p)
        parsers={'java':v2.parser_for('java'),'python':v2.parser_for('python')}
        # Failure cases: exact function IDs, without modifying snippets.
        for r in missing:
            for side,fid in enumerate([r['id1'],r['id2']]):
                code=codes[r['corpus']][fid]
                lang='python' if r['corpus']=='python_mixed' else 'java'
                pd=parse_details(parsers[lang],code)
                if pd['has_error']:
                    wrapped=[]
                    if lang=='java':
                        for mode,s in [('class','class D04SyntheticClass {\n'+code+'\n}'),
                                       ('method','class D04SyntheticClass { void d04SyntheticMethod() {\n'+code+'\n} }')]:
                            wrapped.append({'mode':mode,**parse_details(parsers[lang],s)})
                    examples.append({'kind':'remaining_failure','row_index':r['row_index'],
                        'corpus':r['corpus'],'side':side,'function_id':fid,
                        'code_sha256':hashlib.sha256(code.encode()).hexdigest(),
                        'code_chars':len(code),'source_head':safe_excerpt(code),
                        'direct':pd,'wrapper_attempts':wrapped})
        wrapped=[r for r in scores if any(x.startswith('wrapped_') for x in r['parse_modes'])]
        # Sample balanced across source/target/label, preserving determinism.
        buckets=collections.defaultdict(list)
        for r in wrapped:buckets[(r['corpus'],r['label'])].append(r)
        n_buckets=max(1,len(buckets));per=max(1,(args.wrapper_samples+n_buckets-1)//n_buckets)
        selected=[]
        for j,key in enumerate(sorted(buckets)):
            selected+=ordered_sample(buckets[key],per,args.seed+j)
        selected=selected[:args.wrapper_samples]
        for r in selected:
            lang='python' if r['corpus']=='python_mixed' else 'java'
            sides=[]
            for k,fid in enumerate([r['id1'],r['id2']]):
                code=codes[r['corpus']][fid];f=v2.parse_features(parsers[lang],code,lang)
                sides.append({'side':k,'function_id':fid,'expected_mode':r['parse_modes'][k],
                     'actual_mode':f['parse_mode'],'matches':f['parse_mode']==r['parse_modes'][k],
                     'body_sha256':hashlib.sha256(code.encode()).hexdigest(),
                     'ast_node_count':len(f.get('types',[])),'leaf_count':len(f.get('leaves',[])),
                     'error':f['error'],'source_head':safe_excerpt(code,140)})
            examples.append({'kind':'wrapper_sample','row_index':r['row_index'],
                'corpus':r['corpus'],'label':r['label'],'sides':sides})
        # Source positives with unusually low similarity: inspect raw lexical vs AST,
        # a mix of very low and seeded random, never infer incorrect clone labels.
        positive=[r for r in scores if r['corpus']=='java_source' and r['label']==1 and r['status']=='OK']
        positive.sort(key=lambda r:r['composite_similarity'])
        low=positive[:args.source_samples//2]
        remain=ordered_sample(positive[args.source_samples//2:],args.source_samples-len(low),args.seed)
        for r in low+remain:
            f1=v2.parse_features(parsers['java'],codes['java_source'][r['id1']],'java')
            f2=v2.parse_features(parsers['java'],codes['java_source'][r['id2']],'java')
            lengths=[len(f1.get('types',[])),len(f2.get('types',[]))]
            # Bound expensive SequenceMatcher(auto junk disabled) test to modest ASTs.
            nj=None
            if max(lengths)<=900 and f1['error'] is None and f2['error'] is None:
                nj=difflib.SequenceMatcher(None,f1['types'],f2['types'],autojunk=False).ratio()
            examples.append({'kind':'source_positive_similarity','row_index':r['row_index'],
                'id1':r['id1'],'id2':r['id2'],'similarity':r['composite_similarity'],
                'lexical':r['lexical_similarity'],'structural_autojunk_true':r['structural_similarity'],
                'structural_autojunk_false':nj,'ast_lengths':lengths,
                'parse_modes':r['parse_modes'],'tokens':r['tokens']})
        report['diagnostic_samples']={'remaining_failure_cases':len([x for x in examples if x['kind']=='remaining_failure']),
             'wrapper_pairs':len(selected),'source_positive_pairs':len(low+remain)}
        report['wrapper_mode_consistency']=all(y['matches'] for x in examples
                                                if x['kind']=='wrapper_sample' for y in x['sides'])
    else:report['diagnostic_samples']='SKIPPED_SUMMARY_ONLY'
    out.mkdir(parents=True,exist_ok=False)
    v2.dump(out/'diagnostic_report.json',report)
    with (out/'diagnostic_examples.jsonl').open('w') as f:
        for x in examples:f.write(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n')
    v2.dump(out/'manifest.json',{'version':'v3','status':'DIAGNOSTIC_ONLY','source_v2_scores_sha256':v2.sha256(scoresfile),
          'report_sha256':v2.sha256(out/'diagnostic_report.json'),
          'examples_sha256':v2.sha256(out/'diagnostic_examples.jsonl'),
          'prepared_train_sha256':m['train_sha256'],
          'created_chicago':report['created_chicago']})
    print(json.dumps({'status':report['status'],'output':str(out),'rows':len(scores),
           'failures':len(missing),'examples':len(examples)},sort_keys=True),flush=True)

if __name__=='__main__':main()
