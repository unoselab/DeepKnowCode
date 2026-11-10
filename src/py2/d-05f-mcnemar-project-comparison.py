#!/usr/bin/env python3
"""D-05f v1: read-only exact McNemar and project-level M2/M3 diagnostics.

Input: D-05c v1 results directory or ZIP. No GPU or third-party dependencies.
Decision threshold fixed by D-05c (0); no threshold tuning.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MODELS = ('M2', 'M3')
EXPECTED = 7942
EXPECTED_PROJECTS = 24
REQUIRED = ('pair_id','project','true_label','predicted_label','mean_logit_margin','model','decision_threshold')


def require(x, message):
    if not x:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(raw):
    return json.loads(raw.decode('utf-8-sig'))


class Bundle:
    def __init__(self, source):
        self.source = source.resolve()
        self.z = None
        if self.source.is_dir():
            self.prefix = ''
            require((self.source/'manifest.json').is_file(), 'Source manifest missing')
        elif self.source.is_file() and zipfile.is_zipfile(self.source):
            self.z = zipfile.ZipFile(self.source)
            require(self.z.testzip() is None, 'Source ZIP CRC mismatch')
            opts = [n for n in self.z.namelist() if n.endswith('/manifest.json') or n == 'manifest.json']
            require(len(opts) == 1, 'ZIP must contain exactly one manifest.json')
            self.prefix = opts[0][:-len('manifest.json')]
            for member in self.z.infolist():
                require(not member.filename.startswith('/') and '..' not in Path(member.filename).parts, 'Unsafe ZIP member path')
        else:
            raise ValueError('Input should be D-05c directory or ZIP: '+str(source))
        self.manifest_bytes = self.read('manifest.json')
        self.manifest = read_json(self.manifest_bytes)
        require(isinstance(self.manifest.get('files'),dict), 'Missing files map in D-05c manifest')
        for name in ('predictions_M2.csv','predictions_M3.csv','summary.json','preflight.json'):
            require(name in self.manifest['files'], 'D-05c manifest missing '+name)
        # Verify the entire source manifest, not only model predictions.
        for name,meta in self.manifest['files'].items():
            require('/' not in name and '\\' not in name, 'Unsafe manifest entry: '+name)
            raw = self.read(name)
            require(sha(raw) == meta.get('sha256'), 'D-05c SHA256 mismatch: '+name)
            if 'bytes' in meta:
                require(len(raw) == meta['bytes'], 'D-05c file size mismatch: '+name)

    def read(self,name):
        if self.z is not None:
            return self.z.read(self.prefix + name)
        return (self.source/name).read_bytes()

    def close(self):
        if self.z is not None:
            self.z.close()


def load_predictions(bundle, model):
    text=bundle.read('predictions_'+model+'.csv').decode('utf-8-sig')
    reader=csv.DictReader(io.StringIO(text, newline=''))
    require(reader.fieldnames is not None and all(k in reader.fieldnames for k in REQUIRED), 'Prediction schema error '+model)
    items={}
    for line,row in enumerate(reader,2):
        pid=row['pair_id'].strip()
        require(pid and pid not in items, f'Duplicate/empty pair ID {model}:{line}')
        require(row['model']==model, 'Model field mismatch '+model+':'+pid)
        require(row['true_label'] in ('0','1') and row['predicted_label'] in ('0','1'), 'Invalid labels: '+pid)
        require(row['project'] and row['project'] != 'azure-sdk-for-python', 'Invalid project: '+pid)
        score=float(row['mean_logit_margin'])
        require(math.isfinite(score), 'Nonfinite margin: '+pid)
        require(float(row['decision_threshold']) == 0, 'Nonzero decision threshold: '+pid)
        require(int(score>0)==int(row['predicted_label']), 'D-05c decision rule mismatch: '+pid)
        items[pid]={'pair_id':pid,'project':row['project'],'label':int(row['true_label']), 'pred':int(row['predicted_label']), 'score':score}
    require(len(items)==EXPECTED, f'Expected {EXPECTED} {model} records, got {len(items)}')
    return items


def confusion(rows, model):
    tp=fp=tn=fn=0
    for r in rows:
        y=r['label'];p=r[model]['pred']
        if y==1 and p==1:tp+=1
        elif y==1:fn+=1
        elif p==1:fp+=1
        else:tn+=1
    n=tp+fp+tn+fn
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    den=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    mcc=(tp*tn-fp*fn)/math.sqrt(den) if den>0 else None
    return {'n':n,'positive':tp+fn,'negative':tn+fp,'tp':tp,'fp':fp,'tn':tn,'fn':fn,
            'accuracy':(tp+tn)/n if n else None, 'precision':precision,'recall':recall,
            'f1':f1,'mcc':mcc}


def logsumexp(vals):
    if not vals:return float('-inf')
    x=max(vals)
    return x + math.log(sum(math.exp(z-x) for z in vals))


def exact_two_sided_mcnemar(b,c):
    """Exact conditional binomial two-sided p-value, stable for large discordance counts."""
    n=b+c
    if not n:return 1.0,0.0
    k=min(b,c)
    # Binomial symmetric tail. At equality b=c the doubled tail is capped at one.
    logs=[math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-n*math.log(2.0) for i in range(k+1)]
    lp=min(0.0, math.log(2.0)+logsumexp(logs))
    return math.exp(lp),lp/math.log(10.0)


def mcnemar(rows,project):
    both_correct=both_wrong=m2_only=m3_only=0
    for row in rows:
        c2 = row['M2']['pred']==row['label']
        c3 = row['M3']['pred']==row['label']
        if c2 and c3:both_correct+=1
        elif not c2 and not c3:both_wrong+=1
        elif c2:m2_only+=1
        else:m3_only+=1
    p,lp=exact_two_sided_mcnemar(m2_only,m3_only)
    n=len(rows)
    return {'project':project,'n':n,'both_correct':both_correct,'both_incorrect':both_wrong,
            'm2_only_correct':m2_only,'m3_only_correct':m3_only,'discordant':m2_only+m3_only,
            'delta_accuracy_M2_minus_M3':(m2_only-m3_only)/n if n else None,
            'exact_two_sided_p':p,'log10_exact_p':lp}


def holm(pvalues):
    """Holm-Bonferroni adjusted p values, monotone step-down."""
    order=sorted(range(len(pvalues)),key=lambda i:(pvalues[i],i))
    adjusted=[None]*len(pvalues)
    current=0.0
    n=len(pvalues)
    for rank,i in enumerate(order):
        current=max(current,min(1.0,(n-rank)*pvalues[i]))
        adjusted[i]=current
    return adjusted


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(path,rows,fields):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        for row in rows:writer.writerow(row)


def run(args):
    bundle=Bundle(args.input)
    try:
        two={m:load_predictions(bundle,m) for m in MODELS}
        require(set(two['M2'])==set(two['M3']), 'M2/M3 Pair ID sets differ')
        summary=read_json(bundle.read('summary.json'))
        preflight=read_json(bundle.read('preflight.json'))
        require(summary.get('status')=='D05C_V1_EVALUATION_COMPLETE', 'D-05c was not completed')
        require(summary.get('ground_truth_sha256')==preflight.get('gt_sha256') or preflight.get('gt_sha256') is None, 'D-05c ground truth hash mismatch')
        joined=[]
        for pid in sorted(two['M2']):
            a,b=two['M2'][pid],two['M3'][pid]
            require(a['label']==b['label'] and a['project']==b['project'], 'Label/project mismatch: '+pid)
            joined.append({'pair_id':pid,'project':a['project'],'label':a['label'],'M2':a,'M3':b})
        projects=sorted(set(r['project'] for r in joined))
        require(len(projects)==EXPECTED_PROJECTS, 'Project coverage mismatch')
        byproject={p:[r for r in joined if r['project']==p] for p in projects}
        overall={m:confusion(joined,m) for m in MODELS}
        for model in MODELS:
            source_metric=summary['model_metrics'][model]
            for name in ('tp','tn','fp','fn'):
                require(overall[model][name]==source_metric[name], f'D-05c confusion matrix mismatch: {model}.{name}')
        global_mc=mcnemar(joined,'ALL_24_PROJECTS')
        project_mc=[mcnemar(byproject[p],p) for p in projects]
        adj=holm([r['exact_two_sided_p'] for r in project_mc])
        for row,value in zip(project_mc,adj):
            row['holm_adjusted_p_24_projects']=value
            row['holm_reject_at_0_05']=value<0.05
        metrics=[];diffs=[]
        for p in projects:
            a,b=confusion(byproject[p],'M2'),confusion(byproject[p],'M3')
            for model,met in (('M2',a),('M3',b)):
                metrics.append({'project':p,'model':model,**met,'mcc_defined':met['mcc'] is not None})
            diffs.append({'project':p,'n':len(byproject[p]),
                **{'delta_'+k+'_M2_minus_M3':(a[k]-b[k] if a[k] is not None and b[k] is not None else None)
                   for k in ('accuracy','precision','recall','f1','mcc')},
                'm2_only_correct':next(x['m2_only_correct'] for x in project_mc if x['project']==p),
                'm3_only_correct':next(x['m3_only_correct'] for x in project_mc if x['project']==p)})
        macro={}
        for model in MODELS:
            values=[r for r in metrics if r['model']==model]
            macro[model]={}
            for key in ('accuracy','precision','recall','f1','mcc'):
                defined=[r[key] for r in values if r[key] is not None]
                macro[model][key]={'value':sum(defined)/len(defined) if defined else None,'projects_defined':len(defined)}
        cases=[]
        for r in joined:
            correct2=int(r['M2']['pred']==r['label'])
            correct3=int(r['M3']['pred']==r['label'])
            if correct2 != correct3:
                cases.append({'pair_id':r['pair_id'],'project':r['project'],'true_label':r['label'],
                    'M2_predicted_label':r['M2']['pred'],'M3_predicted_label':r['M3']['pred'],
                    'M2_mean_logit_margin':r['M2']['score'],'M3_mean_logit_margin':r['M3']['score'],
                    'M2_correct':correct2,'M3_correct':correct3,
                    'winner':'M2' if correct2 else 'M3'})
        require(len(cases)==global_mc['discordant'], 'Discordance accounting mismatch')
        now=datetime.now(ZoneInfo('America/Chicago'))
        status='D05F_V1_PREFLIGHT_PASS_NOT_ANALYZED' if args.dry_run else 'D05F_V1_MCNEMAR_PROJECT_COMPARISON_COMPLETE'
        output=args.out.resolve() if args.out else args.root.resolve()/'results'/'d-05f'/('python-mcnemar-project-v1-'+now.strftime('%Y%m%d-%H%M%S-%f'))
        report={'status':status,'created_at_chicago':now.isoformat(),'source':str(args.input.resolve()),
            'source_manifest_sha256':sha(bundle.manifest_bytes),
            'source_ground_truth_sha256':summary['ground_truth_sha256'],
            'project_count':len(projects),'pairs':len(joined),'models':list(MODELS),
            'mcnemar_exact_global':global_mc,'project_tests':len(project_mc),
            'project_holm_significant_0_05':sum(r['holm_reject_at_0_05'] for r in project_mc),
            'micro_metrics':overall,'macro_metrics':macro,
            'methodology':{'test':'exact_two_sided_conditional_binomial_McNemar','null':'equal marginal correctness probability',
                'project_adjustment':'Holm-Bonferroni across 24 project-specific McNemar p-values',
                'threshold':'D05C fixed 0, no tuning','macro':'unweighted mean across 24 projects; undefined MCC excluded, count reported'},
            'limitations':['Project-specific tests are exploratory and can have low power.',
                'McNemar tests paired correctness/accuracy, not F1, MCC or AUC.',
                'Pair dependence within projects/functions may violate independence for exact McNemar inference.',
                'This is a balanced, similarity-selected, reviewer-convention-labelled test set.',
                'No additional training leakage scan or reviewer authentication performed.',
                'M0/M1 not evaluated in this script; preserved by original D-05c.']}
        if args.dry_run:
            print(json.dumps({'status':status,'source_manifest_sha256':report['source_manifest_sha256'],
                'pairs':len(joined),'projects':len(projects),'models':list(MODELS),
                'global_discordance':global_mc,'output_if_executed':str(output)},sort_keys=True))
            return
        require(not output.exists(),'Output already exists: '+str(output))
        output.mkdir(parents=True,exist_ok=False)
        mcfields=['project','n','both_correct','both_incorrect','m2_only_correct','m3_only_correct','discordant','delta_accuracy_M2_minus_M3','exact_two_sided_p','log10_exact_p','holm_adjusted_p_24_projects','holm_reject_at_0_05']
        write_csv(output/'mcnemar_results.csv',[global_mc]+project_mc,mcfields)
        metricfields=['project','model','n','positive','negative','tp','fp','tn','fn','accuracy','precision','recall','f1','mcc','mcc_defined']
        write_csv(output/'project_metrics_M2_M3.csv',metrics,metricfields)
        diff_fields=['project','n']+['delta_'+k+'_M2_minus_M3' for k in ('accuracy','precision','recall','f1','mcc')]+['m2_only_correct','m3_only_correct']
        write_csv(output/'project_differences.csv',diffs,diff_fields)
        write_csv(output/'pair_disagreements.csv',cases,['pair_id','project','true_label','M2_predicted_label','M3_predicted_label','M2_mean_logit_margin','M3_mean_logit_margin','M2_correct','M3_correct','winner'])
        dump(output/'analysis_summary.json',report)
        files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in sorted(output.iterdir()) if p.is_file()}
        dump(output/'manifest.json',{'version':'v1','created_at_chicago':now.isoformat(),
            'source_manifest_sha256':report['source_manifest_sha256'], 'files':files})
        print(json.dumps({'status':status,'output':str(output),'pairs':len(joined),'projects':len(projects),
            'm2_only_correct':global_mc['m2_only_correct'],'m3_only_correct':global_mc['m3_only_correct'],
            'global_exact_p':global_mc['exact_two_sided_p'],'global_log10_p':global_mc['log10_exact_p'],
            'project_holm_significant_0_05':report['project_holm_significant_0_05']},sort_keys=True))
    finally:
        bundle.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True,help='D-05c result directory or ZIP')
    ap.add_argument('--root',type=Path,default=Path.cwd(),help='Repository root')
    ap.add_argument('--out',type=Path,help='New output directory (must not exist)')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    try:run(args)
    except (OSError,ValueError,KeyError,TypeError,OverflowError,zipfile.BadZipFile) as exc:
        print('D-05f v1 ERROR:',str(exc),file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':
    sys.exit(main())
