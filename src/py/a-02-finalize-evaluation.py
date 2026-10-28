#!/usr/bin/env python3
"""A-02 final audit. Read-only inputs; report fixed, outcome-independent full samples.
Historical CodeT5 is preserved in source inputs, excluded only from primary tables.
This does NOT equate Figure10 anonymous numeric IDs to named RQ4 projects.
"""
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
MODELS=('codebert','codegpt','graphcodebert')
LANGS=('java','python')

def rd(path):
    if not path.is_file(): raise FileNotFoundError(f'Required A-02 input missing: {path}')
    with path.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def write(path, rows, cols):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)

def median(values):
    a=sorted(values); n=len(a)
    return (a[(n-1)//2]+a[n//2])/2 if n else float('nan')

def mean(values):return sum(values)/len(values) if values else float('nan')

def wilcoxon(a,b):
    try:
        from scipy.stats import wilcoxon
    except ImportError as e:raise RuntimeError('scipy required in deepknowcode environment') from e
    diffs=[v-u for u,v in zip(a,b)]
    if all(abs(x)<1e-12 for x in diffs):return 0.0,1.0
    t,p=wilcoxon(b,a,alternative='two-sided',zero_method='wilcox')
    return float(t),float(p)

def bh(pairs):
    """Benjamini-Hochberg adjusted p-values within a defined 3-model family."""
    ordered=sorted(pairs,key=lambda x:x[1]); n=len(ordered); result={}; m=1.
    for idx in range(n-1,-1,-1):
        key,p=ordered[idx];m=min(m,p*n/(idx+1));result[key]=m
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--v2-results',type=Path,default=Path('results/a-02'))
    ap.add_argument('--v3-results',type=Path,default=Path('results/a-02-v3'))
    ap.add_argument('--output-dir',type=Path,default=Path('results/a-02-final'))
    args=ap.parse_args()
    src=rd(args.v2_results/'before_after_metrics.csv')
    alignment=rd(args.v2_results/'prediction_alignment.csv')
    fig10=rd(args.v3_results/'figure10_inventory.csv')
    fig11=rd(args.v3_results/'figure11_paired_statistics.csv')
    pairs=defaultdict(dict); bad=[]
    for r in alignment:
        if r['language'] not in LANGS or r['model'] not in MODELS:continue
        if r['status']!='EXACT_IDS':bad.append(r)
    if bad: raise RuntimeError(f'{len(bad)} selected prediction alignments not EXACT_IDS; refuse to finalize')
    for r in src:
        if r['language'] not in LANGS or r['model'] not in MODELS:continue
        key=(r['language'],r['model'],r['project'])
        if r['stage'] in pairs[key]:raise RuntimeError('Repeated project-stage '+str(key))
        pairs[key][r['stage']]=r
    stages=[]; stats=[]; checks=[]; problems=[]
    for lang in LANGS:
        per_model=[]
        for model in MODELS:
            keys=sorted(k for k in pairs if k[:2]==(lang,model))
            if not keys:raise RuntimeError('No rows for '+lang+' '+model)
            if any(set(pairs[k])!={'before','after'} for k in keys):raise RuntimeError('Unpaired project for '+str((lang,model)))
            per_model.append(set(k[2] for k in keys))
        if not all(x==per_model[0] for x in per_model):raise RuntimeError(f'Models have different projects in {lang}; cannot claim fixed sample')
        for model in MODELS:
            keys=sorted(k for k in pairs if k[:2]==(lang,model))
            grouped=[(k[2],pairs[k]['before'],pairs[k]['after']) for k in keys]
            for project,be,af in grouped:
                if int(be['rows'])!=int(af['rows']) or int(be['positive'])!=int(af['positive']):raise RuntimeError('Before/after different sample '+project)
                for stage,r in [('before',be),('after',af)]:
                    # defensively recalculate all headline metrics from confusion matrix and raw totals
                    tp,fp,fn=(int(r[x]) for x in ('tp','fp','fn'))
                    p=tp/(tp+fp) if tp+fp else 0.
                    recall=tp/(tp+fn) if tp+fn else 0.
                    f1=2*p*recall/(p+recall) if p+recall else 0.
                    all_n=int(r['rows']); pos_n=int(r['positive']); penalty=int(r['total_penalty'])
                    per_pair=penalty/all_n if all_n else float('nan')
                    per_positive=penalty/pos_n if pos_n else float('nan')
                    if abs(f1-float(r['f1']))>1e-9 or abs(per_pair-float(r['rcp_per_pair']))>1e-9 or abs(per_positive-float(r['rcp_legacy']))>1e-9:
                        raise RuntimeError('V2 internally inconsistent '+str((lang,model,project,stage)))
                    stages.append(dict(language=lang,model=model,project=project,stage=stage,n_pairs=all_n,n_positive=pos_n,tp=tp,fp=fp,fn=fn,precision=p,recall=recall,f1=f1,total_penalty=penalty,rcp_per_pair=per_pair,rcp_per_positive=per_positive))
            for outcome in ('f1','rcp_per_pair','rcp_per_positive'):
                def val(r):
                    if outcome=='rcp_per_positive':return float(r['rcp_legacy'])
                    return float(r[outcome])
                b=[val(be) for _,be,af in grouped];a=[val(af) for _,be,af in grouped]
                w,p=wilcoxon(b,a)
                stats.append(dict(language=lang,model=model,outcome=outcome,n_projects=len(keys),median_before=median(b),median_after=median(a),median_paired_delta=median([y-x for x,y in zip(b,a)]),mean_paired_delta=mean([y-x for x,y in zip(b,a)]),n_improved=sum(y>x for x,y in zip(b,a)),n_worsened=sum(y<x for x,y in zip(b,a)),n_tied=sum(y==x for x,y in zip(b,a)),w_statistic=w,p_two_sided=p,improvement_direction='higher_is_better',normalization='all_pairs' if outcome=='rcp_per_pair' else ('positive_pairs' if outcome=='rcp_per_positive' else 'not_applicable')))
    for lang in LANGS:
        for outcome in ('f1','rcp_per_pair','rcp_per_positive'):
            group=[r for r in stats if r['language']==lang and r['outcome']==outcome]
            adj=bh([(r['model'],r['p_two_sided']) for r in group])
            for r in group:r['p_bh_within_language_outcome_3_models']=adj[r['model']]
    # Figure10 numeric project IDs are anonymous; compare descriptive aggregates ONLY.
    f10idx={(r['language'],r['model']):r for r in fig10}
    for r in stats:
        if r['outcome']!='f1':continue
        o=f10idx.get((r['language'],r['model']))
        if not o:problems.append('Missing Figure10 entry: '+str((r['language'],r['model'])));continue
        checks.append(dict(language=r['language'],model=r['model'],figure10_n_projects=o['unique_ids'],rq4_n_projects=r['n_projects'],project_identity_mapping_verified=False,figure10_median_f1_before=o['median_before_f1'],rq4_median_f1_before=r['median_before'],figure10_median_f1_after=o['median_after_f1'],rq4_median_f1_after=r['median_after'],figure10_f1_harmonic_mismatch_cells=o['f1_harmonic_mismatch_cells'],comparison_limitation='Aggregate only: Figure10 numeric IDs cannot be matched to RQ4 project names'))
    selected_fig11=[r for r in fig11 if r['model'] in MODELS]
    write(args.output_dir/'final_project_metrics.csv',stages,list(stages[0]))
    write(args.output_dir/'final_paired_statistics.csv',stats,list(stats[0]))
    write(args.output_dir/'figure10_aggregate_comparison.csv',checks,list(checks[0]))
    write(args.output_dir/'historical_figure11_statistics.csv',selected_fig11,list(selected_fig11[0]))
    summary={'status':'AUDIT_COMPLETE_NOT_PAPER_READY','models_in_primary_analysis':list(MODELS),'excluded_from_primary_analysis':['codet5'],'excluded_model_data_preserved':True,'primary_rcp_definition':'sum of existing penalty values / number of evaluated pairs','alternative_rcp_definition':'sum of existing penalty values / number of ground-truth positive pairs','sample_policy':'complete named RQ4 project sets shared across all three included models; no outcome-driven project selection','evaluated_projects_by_language':{lang:len(set(r['project'] for r in stages if r['language']==lang)) for lang in LANGS},'metric_rows':len(stages),'paired_test_rows':len(stats),'figure10_project_mapping':'unverified; do not equate numeric IDs with RQ4 project names','figure10_original_f1_inconsistencies_present':any(int(r['figure10_f1_harmonic_mismatch_cells']) for r in checks),'known_limitations':['Figure10 reported P,R,F1 arithmetic inconsistencies may reflect distinct macro-aggregation; provenance not established','Figure10 Java 50 vs RQ4 Java 48 and Figure10 Python 20 vs RQ4 Python 24; not identical evaluation samples','Original Figure11 Java and Python have different denominators; final tables use the common all-pairs denominator','No raw model inference, training, curriculum or new modern-LLM experiments are claimed','RCP penalty implementation is inherited from RQ4 and has not been independently validated against refactoring outcomes'], 'blocking_errors':problems}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    report=['# A-02 Final Evaluation Audit','',f"Status: **{summary['status']}**",'','Primary models: CodeBERT, GraphCodeBERT, CodeGPT. CodeT5 archived, excluded from primary tables.','',f"Project sets: {summary['evaluated_projects_by_language']}.",'','Primary RCP: sum of penalties / all evaluated pairs. Positive-only RCP is supplementary.','', '## Recomputed paired statistics','', '| Language | Model | Metric | N | Median before | Median after | Median paired delta | p (two-sided) | BH p |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in stats:report.append(f"| {r['language']} | {r['model']} | {r['outcome']} | {r['n_projects']} | {r['median_before']:.4f} | {r['median_after']:.4f} | {r['median_paired_delta']:.4f} | {r['p_two_sided']:.4g} | {r['p_bh_within_language_outcome_3_models']:.4g} |")
    report.extend(['','## Remaining publication-level issues','']+['- '+x for x in summary['known_limitations']])
    (args.output_dir/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('A-02 final audit complete; primary models:',','.join(MODELS))
    print('Projects:',summary['evaluated_projects_by_language'],'model-stage rows:',len(stages))
    for r in stats:
        if r['outcome']=='rcp_per_pair':print('RCP',r['language'],r['model'],'median',round(r['median_before'],4),'->',round(r['median_after'],4),'p=',f"{r['p_two_sided']:.5g}")
    print('Audit status:',summary['status'],'(historical Fig10 lineage unresolved)')
    print('Output:',args.output_dir.resolve())
if __name__=='__main__':main()
