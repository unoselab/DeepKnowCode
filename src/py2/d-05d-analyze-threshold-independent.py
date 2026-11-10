#!/usr/bin/env python3
"""D-05d v1. Read-only D-05c prediction-score evaluation (CPU-only)."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MODELS=('M0','M1','M2','M3')
N=7942
POS=NEG=3971
REQUIRED=('pair_id','project','true_label','predicted_label','mean_logit_margin','model','decision_threshold')

def require(c,m):
    if not c: raise ValueError(m)

def sha(data):return hashlib.sha256(data).hexdigest()

def emit(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n',encoding='utf-8')

def write_csv(path, rows, columns):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)

class Evidence:
    def __init__(self, path):
        self.path=path.resolve()
        if self.path.is_dir():
            self.z=None;self.root=None
        elif self.path.is_file() and zipfile.is_zipfile(self.path):
            self.z=zipfile.ZipFile(self.path)
            require(self.z.testzip() is None,'Source ZIP CRC failure')
            names=[x for x in self.z.namelist() if x.endswith('/manifest.json') or x=='manifest.json']
            require(len(names)==1,'Expected exactly one source manifest.json in ZIP')
            self.root=names[0][:-len('manifest.json')]
            require(all(not n.startswith('/') and '..' not in Path(n).parts for n in self.z.namelist()),'Unsafe ZIP member')
        else:raise ValueError('Input must be D-05c results directory or ZIP: '+str(self.path))
    def read(self,name):
        require('/' not in name and name not in ('','.','..'),'Invalid input name')
        return self.z.read(self.root+name) if self.z else (self.path/name).read_bytes()
    def close(self):
        if self.z:self.z.close()

def load_source(src):
    man_raw=src.read('manifest.json'); man=json.loads(man_raw)
    require(man.get('version')=='v1','Unexpected D-05c version')
    files=man.get('files',{})
    names=['summary.json','preflight.json','aligned_pairs.jsonl']+[f'predictions_{m}.csv' for m in MODELS]+[f'predictions_{m}.json' for m in MODELS]
    for name in names:
        require(name in files,f'Source manifest missing {name}')
    for name, info in files.items():
        require('/' not in name and name not in ('','.','..'),'Unsafe manifest filename')
        raw=src.read(name)
        require(sha(raw)==info['sha256'] and len(raw)==info['bytes'],f'Source manifest hash/size mismatch: {name}')
    summary=json.loads(src.read('summary.json'))
    preflight=json.loads(src.read('preflight.json'))
    require(summary.get('status')=='D05C_V1_EVALUATION_COMPLETE','D-05c incomplete')
    require(preflight.get('pairs')==N,'D-05c preflight pair count differs')
    require(preflight.get('ground_truth',{}).get('class_counts')=={'0':NEG,'1':POS},'D-05c preflight class counts differ')
    require(summary.get('ground_truth_sha256')==preflight['ground_truth']['verified_pairs_sha256'],'D-05c ground truth digest differs')
    require(summary.get('prediction_rule')=='mean(YES logit - NO logit) > 0','Unknown prediction rule')
    reference=None
    bymodel={}
    for model in MODELS:
        raw=src.read(f'predictions_{model}.csv')
        reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig'),newline=''))
        require(set(REQUIRED).issubset(reader.fieldnames or []),f'Missing CSV fields for {model}')
        records={}
        for r in reader:
            pid=r['pair_id'];project=r['project']
            require(pid and pid not in records,f'Duplicate/missing pair ID {model} {pid}')
            require(project and r['model']==model,f'Wrong project/model {model} {pid}')
            require(r['true_label'] in ('0','1') and r['predicted_label'] in ('0','1'),f'Nonbinary label {model} {pid}')
            score=float(r['mean_logit_margin']);threshold=float(r['decision_threshold'])
            require(math.isfinite(score) and threshold==0.0,f'Nonfinite score or unexpected threshold {model} {pid}')
            require(int(score>0)==int(r['predicted_label']),f'Prediction disagrees with margin {model} {pid}')
            records[pid]=(project,int(r['true_label']),score,int(r['predicted_label']))
        require(len(records)==N,f'Expected {N} records for {model}; got {len(records)}')
        require(Counter(y for _,y,_,_ in records.values())=={0:NEG,1:POS},f'Class counts mismatch {model}')
        keyed={pid:(r[0],r[1]) for pid,r in records.items()}
        if reference is None: reference=keyed
        else:require(reference==keyed,f'Model pair/label/project alignment mismatch {model}')
        metadata=json.loads(src.read(f'predictions_{model}.json'))
        require(metadata['model']==model and metadata['records']==N and metadata['csv_sha256']==sha(raw),f'Prediction metadata mismatch {model}')
        cm=Counter((r[1],r[3]) for r in records.values())
        declared=summary['model_metrics'][model]
        for k,v in [('tp',cm[(1,1)]),('tn',cm[(0,0)]),('fp',cm[(0,1)]),('fn',cm[(1,0)])]:
            require(declared[k]==v,f'Source summary {model}.{k} differs from CSV')
        bymodel[model]=records
    return bymodel,{'source_manifest_sha256':sha(man_raw),'ground_truth_sha256':summary['ground_truth_sha256'],'model_registry_sha256s_sha256':preflight['model_registry']['sha256s_sha256'],'source_prediction_sha256':{m:sha(src.read(f'predictions_{m}.csv')) for m in MODELS},'source_created_at_chicago':summary['created_at_chicago']}

def analyze(records):
    from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve,auc
    import numpy as np
    results=[];roc=[];pr=[];dist=[];sweep=[];data={}
    thresholds=[-math.inf,-10,-5,-3,-2,-1,-0.5,-0.25,0,0.25,0.5,1,2,3,5,10,math.inf]
    for m in MODELS:
        ordered=[records[m][k] for k in sorted(records[m])]
        y=np.asarray([x[1] for x in ordered],dtype=np.int8)
        scores=np.asarray([x[2] for x in ordered],dtype=np.float64)
        fpr,tpr,rt=roc_curve(y,scores,drop_intermediate=False)
        precision,recall,pt=precision_recall_curve(y,scores,drop_intermediate=False)
        roc_area=float(roc_auc_score(y,scores)); ap=float(average_precision_score(y,scores)); trap=float(auc(recall,precision))
        require(0<=roc_area<=1 and 0<=ap<=1,'AUC out of range')
        from collections import defaultdict
        score_stats={}
        for label in (0,1):
            v=scores[y==label]
            score_stats[str(label)]={'n':len(v),'min':float(v.min()),'p05':float(np.percentile(v,5)),'median':float(np.median(v)),'mean':float(v.mean()),'p95':float(np.percentile(v,95)),'max':float(v.max()),'positive_margin_rate':float((v>0).mean())}
            dist.append({'model':m,'true_label':label,**score_stats[str(label)]})
        results.append({'model':m,'n':len(y),'positives':int(y.sum()),'negatives':int((1-y).sum()),'roc_auc':roc_area,'average_precision':ap,'pr_auc_trapezoidal':trap,'positive_prevalence':float(y.mean()),'unique_scores':int(np.unique(scores).size),'score_min':float(scores.min()),'score_max':float(scores.max())})
        for i,(x,z,t) in enumerate(zip(fpr,tpr,rt)):
            roc.append({'model':m,'point_index':i,'false_positive_rate':float(x),'true_positive_rate':float(z),'threshold':float(t) if math.isfinite(t) else str(t)})
        for i,(p,r) in enumerate(zip(precision,recall)):
            pr.append({'model':m,'point_index':i,'precision':float(p),'recall':float(r),'threshold':float(pt[i]) if i<len(pt) else ''})
        for t in thresholds:
            pred=(scores>t).astype(np.int8)
            tp=int(((y==1)&(pred==1)).sum());tn=int(((y==0)&(pred==0)).sum())
            fp=int(((y==0)&(pred==1)).sum());fn=int(((y==1)&(pred==0)).sum())
            p=tp/(tp+fp) if tp+fp else 0.0; rec=tp/(tp+fn) if tp+fn else 0.0
            f1=2*p*rec/(p+rec) if p+rec else 0.0
            sweep.append({'model':m,'threshold':str(t) if not math.isfinite(t) else t,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'precision':p,'recall':rec,'f1':f1,'accuracy':(tp+tn)/len(y),'false_positive_rate':fp/(fp+tn)})
        data[m]={'y':y,'scores':scores,'roc':(fpr,tpr),'pr':(recall,precision)}
    return results,roc,pr,dist,sweep,data

def charts(folder,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.6,5.6))
    for m in MODELS:
        x,y=data[m]['roc'];ax.plot(x,y,label=m)
    ax.plot([0,1],[0,1],ls='--',lw=0.8,c='gray');ax.set(xlabel='False positive rate',ylabel='True positive rate',title='D-05d ROC curves (balanced test set)',xlim=(0,1),ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(folder/'roc_curves.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.6,5.6))
    for m in MODELS:
        x,y=data[m]['pr'];ax.plot(x,y,label=m)
    ax.axhline(0.5,ls='--',lw=0.8,c='gray');ax.set(xlabel='Recall',ylabel='Precision',title='D-05d PR curves (50% positive prevalence)',xlim=(0,1),ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(folder/'pr_curves.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5.6))
    for m in MODELS:
        scores=data[m]['scores'];clip=__import__('numpy').clip(scores,-15,15)
        ax.hist(clip,bins=65,histtype='step',density=True,label=m)
    ax.axvline(0,ls='--',lw=0.8,c='gray');ax.set(xlabel='Mean YES–NO logit margin (clipped to [-15,15] for display)',ylabel='Density',title='D-05d model score distribution');ax.legend();fig.tight_layout();fig.savefig(folder/'score_distributions.png',dpi=150);plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True,type=Path,help='D-05c completed results directory OR original ZIP')
    ap.add_argument('--out',type=Path,help='New result directory (must not exist)')
    ap.add_argument('--root',type=Path,default=Path.cwd())
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args(); root=args.root.resolve()
    src=Evidence((root/args.input).resolve())
    try:models,provenance=load_source(src)
    finally:src.close()
    now=datetime.now(ZoneInfo('America/Chicago'))
    stamp=now.strftime('%Y%m%d-%H%M%S-%f')
    out=(root/args.out).resolve() if args.out else root/'results'/'d-05d'/('python-threshold-independent-v1-'+stamp)
    require(not out.exists(),'Output already exists: '+str(out))
    if args.dry_run:
        print(json.dumps({'status':'D05D_V1_PREFLIGHT_PASS_NOT_ANALYZED','records_per_model':N,'positive_per_model':POS,'negative_per_model':NEG,'models':list(MODELS),'source':str(args.input),'source_manifest_sha256':provenance['source_manifest_sha256'],'output_if_executed':str(out)},sort_keys=True))
        return 0
    scores,roc,pr,dist,sweep,data=analyze(models)
    out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'auc_metrics.csv',scores,['model','n','positives','negatives','roc_auc','average_precision','pr_auc_trapezoidal','positive_prevalence','unique_scores','score_min','score_max'])
    write_csv(out/'roc_curve.csv',roc,['model','point_index','false_positive_rate','true_positive_rate','threshold'])
    write_csv(out/'pr_curve.csv',pr,['model','point_index','precision','recall','threshold'])
    write_csv(out/'score_distribution_summary.csv',dist,['model','true_label','n','min','p05','median','mean','p95','max','positive_margin_rate'])
    write_csv(out/'threshold_sweep_exploratory.csv',sweep,['model','threshold','tp','tn','fp','fn','precision','recall','f1','accuracy','false_positive_rate'])
    charts(out,data)
    summary={'version':'v1','status':'D05D_V1_THRESHOLD_INDEPENDENT_COMPLETE','created_at_chicago':now.isoformat(),'source':str(args.input),'provenance':provenance,'dataset':{'pairs':N,'positive':POS,'negative':NEG,'projects':24},'auc':{x['model']:{'roc_auc':x['roc_auc'],'average_precision':x['average_precision'],'pr_auc_trapezoidal':x['pr_auc_trapezoidal']} for x in scores},'definitions':{'score':'D-05c mean YES-minus-NO logit margin; larger implies clone','positive_class':1,'average_precision':'sklearn average_precision_score; step-weighted, not trapezoidal','pr_auc_trapezoidal':'sklearn auc(recall,precision); different estimator than average precision','roc_auc':'sklearn roc_auc_score with ties handled by ranking'},'limitations':['All AUC/AP values refer to 50% positive-prevalence similarity-selected benchmark; AP depends on prevalence.','No threshold optimized on the test set is authorized for final benchmark claims. Threshold sweep is exploratory only.','No resampling-based uncertainty or statistical significance in D-05d; planned D-05e/D-05f.','Ground truth reviewer identity/independence not externally authenticated and leakage audits inherited from D-05bj.','Scores and rankings may be affected by 4bit inference and calibratedness; no calibration claim.']}
    emit(out/'analysis_summary.json',summary)
    files={p.name:{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
    emit(out/'manifest.json',{'version':'v1','created_at_chicago':now.isoformat(),'source_manifest_sha256':provenance['source_manifest_sha256'],'files':files})
    print(json.dumps({'status':summary['status'],'out':str(out),'auc':summary['auc']},sort_keys=True))
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,KeyError,AssertionError,zipfile.BadZipFile) as e:
        print('D-05d v1 ERROR:',e,file=sys.stderr);sys.exit(2)
