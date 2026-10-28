#!/usr/bin/env python3
"""A-02 v3, read-only Figure 10/11 artifact lineage and reproducibility audit.

No performance-based subject-project selection. Figure 10 numeric IDs are deliberately
NOT mapped to RQ4 named projects without independent mapping evidence.
"""
import argparse
import csv
import json
import math
import re
from collections import defaultdict, Counter
from pathlib import Path

FIG10 = re.compile(r'(?P<id>[1-4])-(?P<lang>Java|Py)-(?P<model>CodeGPT|CodeBERT|GraphCodeBERT|CodeT5)\.csv$')
FIG11 = re.compile(r'rcp_scores_(?P<model>.+?)(?P<py>_py)?\.jsonl$')
EXPECTED = {'java':48, 'python':20}


def output_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def fnum(s):
    try: return float(s)
    except (ValueError, TypeError): return float('nan')


def median(vals):
    v=sorted(x for x in vals if math.isfinite(x))
    if not v: return float('nan')
    n=len(v); return v[n//2] if n%2 else (v[n//2-1]+v[n//2])/2


def safe_avg(vals):
    v=[x for x in vals if math.isfinite(x)]
    return sum(v)/len(v) if v else float('nan')


def pvalue(before, after):
    try:
        from scipy.stats import wilcoxon
        if len(before)<2: return {'w':'','p':'','test_note':'insufficient_pairs'}
        if all(abs(a-b)<1e-12 for a,b in zip(before,after)):
            return {'w':0,'p':1,'test_note':'all_ties'}
        stat,p=wilcoxon(after,before,zero_method='wilcox',alternative='two-sided')
        return {'w':float(stat),'p':float(p),'test_note':'two_sided_paired_wilcoxon'}
    except ImportError:
        return {'w':'','p':'','test_note':'scipy_not_installed'}
    except ValueError as e:
        return {'w':'','p':'','test_note':str(e)}


def figure10(root, out):
    records=[]; details=[]; found=set()
    for path in sorted((root/'input').glob('*.csv')):
        m=FIG10.fullmatch(path.name)
        if not m: continue
        lang='java' if m['lang']=='Java' else 'python'; model=m['model'].lower()
        found.add((lang,model))
        with path.open(encoding='utf-8-sig', newline='') as f: rows=list(csv.DictReader(f))
        ids=[]; bad_ids=0; f1_bad=0; missing_metrics=0
        for row in rows:
            try: ids.append(int(row['Project ID']))
            except (ValueError,KeyError): bad_ids+=1
            detail={'language':lang,'model':model,'project_id':row.get('Project ID',''),'source_file':path.name}
            for stage in ('Before','After'):
                p,r,f1=(fnum(row.get(f'{stage}_{z}')) for z in ('P','R','F1'))
                harmonic=2*p*r/(p+r) if p+r>0 else 0.0
                diff=abs(f1-harmonic)
                detail.update({f'{stage.lower()}_precision':p, f'{stage.lower()}_recall':r,
                               f'{stage.lower()}_f1':f1, f'{stage.lower()}_harmonic_f1':harmonic,
                               f'{stage.lower()}_f1_gap':diff})
                if not all(math.isfinite(x) for x in (p,r,f1)): missing_metrics+=1
                elif diff>0.002: f1_bad+=1
            details.append(detail)
        unique=set(ids)
        subset=[d for d in details if d['source_file']==path.name]
        records.append({'language':lang,'model':model,'file':path.name,'rows':len(rows),
                        'unique_ids':len(unique),'id_min':min(ids) if ids else '',
                        'id_max':max(ids) if ids else '',
                        'duplicate_ids':len(ids)-len(unique),
                        'consecutive_1_to_n':unique==set(range(1,len(rows)+1)),
                        'paper_reported_projects':EXPECTED[lang],
                        'count_matches_paper':len(unique)==EXPECTED[lang],
                        'f1_harmonic_mismatch_cells':f1_bad,
                        'nonfinite_metric_cells':missing_metrics,
                        'median_before_f1':median([d['before_f1'] for d in subset]),
                        'median_after_f1':median([d['after_f1'] for d in subset])})
    output_csv(out/'figure10_inventory.csv',list(records[0]) if records else ['language','file'],records)
    output_csv(out/'figure10_f1_checks.csv',list(details[0]) if details else ['language','file'],details)
    return records


def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError as e: raise ValueError(f'{path}:{i}: {e}')


def model_name(s):
    return s.replace('graph-code-bert','graphcodebert').replace('-','').lower()


def figure11(root,out):
    file_records=[]; per_project=[]; stats=[]; raw_counts={}
    for lang,sub in (('java','ja'),('python','py')):
        for path in sorted((root/'input-rcp'/sub).glob('*.jsonl')):
            m=FIG11.fullmatch(path.name)
            if not m: continue
            model=model_name(m['model'])
            recs=list(read_jsonl(path)); by=defaultdict(list)
            count=Counter(); malformed=0; rounded=0
            for rec in recs:
                stage=str(rec.get('adapt','')).lower()
                subject=str(rec.get('subject',''))
                if stage not in ('before','after') or not subject:
                    malformed+=1; continue
                if lang=='python' and 'score' in rec:
                    project,sep,suffix=subject.rpartition('-')
                    if not sep or not suffix.isdigit():
                        malformed+=1; continue
                    val=fnum(rec['score']); kind='raw_score'
                else:
                    project=subject
                    if 'avg' in rec: val=fnum(rec['avg']); kind='provided_avg'; rounded+=1
                    elif 'score' in rec: val=fnum(rec['score']); kind='raw_score'
                    else: malformed+=1; continue
                if not math.isfinite(val): malformed+=1; continue
                by[(project,stage)].append(val); count[stage]+=1
            projects=set(p for p,s in by)
            paired=sorted(p for p in projects if (p,'before') in by and (p,'after') in by)
            before=[safe_avg(by[(p,'before')]) for p in paired]
            after=[safe_avg(by[(p,'after')]) for p in paired]
            for project in sorted(projects):
                for stage in ('before','after'):
                    vals=by.get((project,stage),[])
                    if vals:
                        per_project.append({'language':lang,'model':model,'project':project,'stage':stage,
                                            'n_observations':len(vals),'mean_score':safe_avg(vals),
                                            'source_file':path.name})
            file_records.append({'language':lang,'model':model,'file':path.name,'records':len(recs),
                                 'before_records':count['before'],'after_records':count['after'],
                                 'projects':len(projects),'paired_projects':len(paired),
                                 'unpaired_projects':';'.join(sorted(projects-set(paired))),
                                 'malformed_records':malformed,'provided_avg_records':rounded,
                                 'median_before':median(before),'median_after':median(after)})
            stat=pvalue(before,after)
            stats.append({'language':lang,'model':model,'paired_projects':len(paired),
                          'median_before':median(before),'median_after':median(after),
                          'median_delta_after_minus_before':median([a-b for a,b in zip(after,before)]),
                          'mean_delta_after_minus_before':safe_avg([a-b for a,b in zip(after,before)]),
                          **stat,'source_file':path.name})
    output_csv(out/'figure11_inventory.csv',list(file_records[0]) if file_records else ['language'],file_records)
    output_csv(out/'figure11_project_scores.csv',list(per_project[0]) if per_project else ['language'],per_project)
    output_csv(out/'figure11_paired_statistics.csv',list(stats[0]) if stats else ['language'],stats)
    return file_records,stats,per_project


def compare_rq4(v2_root,out,fig11):
    path=v2_root/'rcp_metrics.csv'
    if not path.exists(): return []
    with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    rq4={(r['language'],r['model'],r['project'],r['stage']):r for r in rows}
    joined=[]
    for rec in fig11:
        key=tuple(rec[z] for z in ('language','model','project','stage'))
        baseline=rq4.get(key)
        v3=fnum(rec['mean_score'])
        legacy=fnum(baseline['rcp_legacy']) if baseline else float('nan')
        per_pair=fnum(baseline['rcp_per_pair']) if baseline else float('nan')
        joined.append({'language':rec['language'],'model':rec['model'],'project':rec['project'],
                       'stage':rec['stage'],'figure11_score':v3,'rq4_legacy':legacy,
                       'rq4_per_pair':per_pair,'difference_vs_legacy':v3-legacy,
                       'difference_vs_per_pair':v3-per_pair,
                       'named_project_match':bool(baseline),
                       'note':'Figure11 Java may use rounded avg; Python mean raw score may differ in unit from RQ4 CSV'})
    output_csv(out/'figure11_vs_rq4.csv',list(joined[0]) if joined else ['language'],joined)
    return joined


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--journal-graph2',type=Path,default=Path.home()/'research_dream/2026Journal/graph2')
    ap.add_argument('--v2-results',type=Path,default=Path('results/a-02'))
    ap.add_argument('--output-dir',type=Path,default=Path('results/a-02-v3'))
    args=ap.parse_args()
    if not (args.journal_graph2/'input').is_dir() or not (args.journal_graph2/'input-rcp').is_dir():
        ap.error('Missing graph2/input or graph2/input-rcp; check --journal-graph2')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    f10=figure10(args.journal_graph2,args.output_dir)
    f11,stats,per_project=figure11(args.journal_graph2,args.output_dir)
    joined=compare_rq4(args.v2_results,args.output_dir,per_project)
    summary={'figure10_files':len(f10),'figure11_files':len(f11),
             'figure10_project_counts':{r['file']:r['unique_ids'] for r in f10},
             'figure11_paired_counts':{r['file']:r['paired_projects'] for r in f11},
             'figure11_named_rq4_matches':sum(x['named_project_match'] for x in joined),
             'figure11_named_rq4_unmatched':sum(not x['named_project_match'] for x in joined),
             'caveats':['Figure10 Project ID numbers have no validated mapping to RQ4 named projects',
                        'F1 mismatches are arithmetic QC flags, not evidence of fabrication',
                        'Paper subject selection must be independent of model outcomes',
                        'Figure11 JSONL subject unit differs between Java and Python; Python raw observations aggregated by project',
                        'Figure11 Java avg values may be rounded to 2-3 decimals',
                        'Paired Wilcoxon is project-level; post hoc subsets must be reported as exploratory']}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('A-02 v3 complete: '+json.dumps({k:v for k,v in summary.items() if isinstance(v,int)}))
    for r in f10: print('Fig10',r['file'],'n=',r['unique_ids'],'F1 mismatches=',r['f1_harmonic_mismatch_cells'])
    for r in stats: print('Fig11',r['language'],r['model'],'paired=',r['paired_projects'],'before=',round(r['median_before'],4),'after=',round(r['median_after'],4),'p=',r['p'])
    print('Output:',args.output_dir.resolve())

if __name__=='__main__': main()
