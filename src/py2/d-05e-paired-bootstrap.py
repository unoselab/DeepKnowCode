#!/usr/bin/env python3
"""D-05e v1: reproducible paired bootstrap comparison M2 minus M3 on D-05c output.

Read-only original D-05c results, support source directory or release ZIP.
Paired, label-stratified pair sampling and project-cluster bootstrap sensitivity.
Scores use original next-token mean-logit margin; classification threshold is fixed at zero.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

VERSION='v1'
MODELS=('M2','M3')
METRICS=('f1','mcc','roc_auc','average_precision')
REQUIRED={'pair_id','project','true_label','predicted_label','mean_logit_margin','model'}

def require(x,msg):
    if not x: raise ValueError(msg)

def sha(raw):return hashlib.sha256(raw).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def write_csv(path,fields,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

class Bundle:
    def __init__(self,path):
        self.path=path.resolve(); self.z=None
        if self.path.is_file():
            self.z=zipfile.ZipFile(self.path)
            bad=self.z.testzip();require(bad is None,'ZIP CRC mismatch: '+str(bad))
            m=[n for n in self.z.namelist() if n.endswith('/manifest.json')]
            require(len(m)==1,'ZIP must have exactly one nested manifest.json')
            self.prefix=m[0][:-len('manifest.json')]
            require(all(n==self.prefix or n.startswith(self.prefix) for n in self.z.namelist()),'ZIP has unexpected sibling entries')
        else:
            require(self.path.is_dir(),'D-05c input missing: '+str(self.path))
    def read(self,name):
        if self.z is not None:return self.z.read(self.prefix+name)
        return (self.path/name).read_bytes()
    def close(self):
        if self.z:self.z.close()
    def verify(self):
        raw=self.read('manifest.json');man=json.loads(raw)
        require(man.get('version')=='v1','Unexpected D-05c source manifest version')
        refs=man.get('files',{});require(isinstance(refs,dict) and all('predictions_'+m+'.csv' in refs for m in MODELS),'Missing source prediction hashes')
        for name,info in refs.items():
            require(Path(name).name==name and '/' not in name and '\\' not in name,'Unsafe source manifest path: '+name)
            actual=self.read(name)
            require(len(actual)==info['bytes'] and sha(actual)==info['sha256'],'D-05c source manifest mismatch: '+name)
        pf=json.loads(self.read('preflight.json'))
        require(pf['ground_truth']['class_counts']=={'0':3971,'1':3971},'Unexpected D-05c class counts')
        require(pf['ground_truth']['verified_pairs_sha256']==json.loads(self.read('summary.json'))['ground_truth_sha256'],'Ground truth digest disagreement')
        return man,sha(raw),pf

def load_prediction(raw,model):
    s=io.StringIO(raw.decode('utf-8-sig'));r=csv.DictReader(s)
    require(r.fieldnames is not None and REQUIRED.issubset(set(r.fieldnames)),'Missing D-05c CSV fields '+model)
    data={}
    for line,row in enumerate(r,2):
        pid=row['pair_id'].strip();require(pid and pid not in data,model+' duplicate/invalid pair_id at line '+str(line))
        require(row['model']==model,'Model column mismatch '+pid)
        require(row['true_label'] in ('0','1') and row['predicted_label'] in ('0','1'),'Invalid binary label '+pid)
        score=float(row['mean_logit_margin']);require(math.isfinite(score),'Nonfinite logit '+pid)
        pred=int(row['predicted_label']);require(pred==int(score>0),'Predicted label not threshold-zero rule '+pid)
        project=row['project'].strip();require(project,'Missing project '+pid)
        data[pid]=(project,int(row['true_label']),pred,score)
    require(len(data)==7942,model+' expected 7942 rows')
    return data

class Ranked:
    """Weighted ROC-AUC / AP with exact ties; equivalent to sklearn metrics."""
    def __init__(self,scores,y):
        ix=np.argsort(-scores,kind='mergesort')
        sorted_scores=scores[ix]
        self.order=ix
        self.y=y.astype(np.int8)
        self.ends=np.r_[np.flatnonzero(np.diff(sorted_scores)!=0)+1,len(scores)]
        self.starts=np.r_[0,self.ends[:-1]]
    def compute(self,counts,total_pos,total_neg):
        w=counts[self.order]
        pos=np.add.reduceat(w*self.y[self.order],self.starts)
        neg=np.add.reduceat(w*(1-self.y[self.order]),self.starts)
        tp=np.cumsum(pos,dtype=np.float64)
        fp=np.cumsum(neg,dtype=np.float64)
        precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
        # sklearn average_precision_score: stepwise area under precision-recall.
        ap=float(np.dot(pos,precision)/total_pos)
        prev_tp=np.r_[0.,tp[:-1]]
        prev_fp=np.r_[0.,fp[:-1]]
        auc=float(np.sum((tp+prev_tp)*(fp-prev_fp)) /(2*total_pos*total_neg))
        return auc,ap

class Evaluator:
    def __init__(self,rows):
        self.n=len(rows);self.y=np.array([r[1] for r in rows],dtype=np.int8)
        self.preds=np.array([r[2] for r in rows],dtype=np.int8)
        self.scores=np.array([r[3] for r in rows],dtype=np.float64)
        self.ranked=Ranked(self.scores,self.y)
        self.confusion_ids=self.y*2+self.preds
    def eval(self,counts):
        conf=np.bincount(self.confusion_ids,weights=counts,minlength=4)
        tn,fp,fn,tp=conf
        denom=2*tp+fp+fn
        f1=float(2*tp/denom) if denom else 0.
        dd=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
        mcc=float((tp*tn-fp*fn)/math.sqrt(dd)) if dd>0 else 0.
        roc,ap=self.ranked.compute(counts,tp+fn,tn+fp)
        return np.array((f1,mcc,roc,ap),dtype=np.float64)

def bootstrap(y,project_idx,models,mode,reps,seed):
    n=len(y);rng=np.random.default_rng(seed)
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    nproj=int(project_idx.max())+1
    point=np.ones(n,dtype=np.int64)
    base=np.vstack([m.eval(point) for m in models]);diff=base[0]-base[1]
    dist=np.empty((reps,len(METRICS)),dtype=np.float64)
    for b in range(reps):
        if mode=='paired_stratified':
            counts=np.bincount(np.r_[pos[rng.integers(len(pos),size=len(pos))],neg[rng.integers(len(neg),size=len(neg))]],minlength=n)
        else:
            drawn=rng.integers(nproj,size=nproj)
            multiplicity=np.bincount(drawn,minlength=nproj)
            counts=multiplicity[project_idx]
        ps=int(np.dot(counts,y));ns=int(np.sum(counts)-ps)
        if ps==0 or ns==0:raise RuntimeError('Bootstrap replicate missing class: '+str(b))
        dist[b]=models[0].eval(counts)-models[1].eval(counts)
    rows=[]
    for i,name in enumerate(METRICS):
        lo,hi=np.percentile(dist[:,i],[2.5,97.5])
        rows.append({'sampling':mode,'metric':name,'M2':float(base[0,i]),'M3':float(base[1,i]),'delta_M2_minus_M3':float(diff[i]),'ci_lower_95':float(lo),'ci_upper_95':float(hi),'bootstrap_repetitions':reps,'seed':seed,'ci_excludes_zero':bool(lo>0 or hi<0),'ci_method':'percentile'})
    return rows,dist

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=Path.cwd())
    p.add_argument('--input',type=Path,required=True,help='D-05c result directory or ZIP')
    p.add_argument('--out',type=Path,help='New results/d-05e directory; must not exist')
    p.add_argument('--repetitions',type=int,default=10000)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();require(100<=a.repetitions<=100000,'Repetitions must be 100..100000')
    root=a.repo_root.resolve();input_path=a.input if a.input.is_absolute() else root/a.input
    b=Bundle(input_path)
    try:
        manifest,manifest_sha,pf=b.verify()
        aa=load_prediction(b.read('predictions_M2.csv'),'M2')
        bb=load_prediction(b.read('predictions_M3.csv'),'M3')
        require(set(aa)==set(bb),'M2/M3 Pair ID mismatch')
        ids=sorted(aa)
        for pid in ids:
            require(aa[pid][:2]==bb[pid][:2],'M2/M3 project or true label mismatch: '+pid)
        projects=sorted({aa[pid][0] for pid in ids});require(len(projects)==24,'Expected 24 distinct projects')
        pi={name:i for i,name in enumerate(projects)}
        project_codes=np.array([pi[aa[pid][0]] for pid in ids],dtype=np.int32)
        models=[Evaluator([dic[pid] for pid in ids]) for dic in [aa,bb]]
        y=models[0].y
        require(int(y.sum())==3971 and int((1-y).sum())==3971,'Unexpected 3971/3971 class counts')
        require(np.array_equal(y,models[1].y),'Unequal labels')
        point=np.ones(len(ids),dtype=np.int64)
        point_metrics=np.vstack([m.eval(point) for m in models]);summary={'status':'D05E_V1_PREFLIGHT_PASS_NOT_ANALYZED' if a.dry_run else 'D05E_V1_PAIRED_BOOTSTRAP_COMPLETE','created_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),'source':str(input_path.resolve()),'source_manifest_sha256':manifest_sha,'source_ground_truth_sha256':pf['ground_truth']['verified_pairs_sha256'],'n_pairs':len(ids),'n_positive':int(y.sum()),'n_negative':int((1-y).sum()),'n_projects':len(projects),'models':['M2','M3'],'metrics':list(METRICS),'repetitions':a.repetitions,'seed':a.seed,'point_estimates':{'M2':dict(zip(METRICS,map(float,point_metrics[0]))),'M3':dict(zip(METRICS,map(float,point_metrics[1]))),'delta_M2_minus_M3':dict(zip(METRICS,map(float,point_metrics[0]-point_metrics[1])))}}
        if a.dry_run:print(json.dumps(summary,sort_keys=True));return
        timestamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
        output=(a.out if a.out.is_absolute() else root/a.out) if a.out else root/'results/d-05e'/('python-paired-bootstrap-v1-'+timestamp)
        require(not output.exists(),'Output exists; refusing overwrite: '+str(output))
        output.mkdir(parents=True,exist_ok=False)
        rows=[]
        for mode,offset in [('paired_stratified',0),('project_cluster',1)]:
            subset,dist=bootstrap(y,project_codes,models,mode,a.repetitions,a.seed+offset)
            rows.extend(subset)
            write_csv(output/('bootstrap_replicates_'+mode+'.csv'),['replicate',*METRICS],[{'replicate':i+1,**dict(zip(METRICS,map(float,dist[i])))} for i in range(len(dist))])
        write_csv(output/'bootstrap_ci.csv',['sampling','metric','M2','M3','delta_M2_minus_M3','ci_lower_95','ci_upper_95','bootstrap_repetitions','seed','ci_excludes_zero','ci_method'],rows)
        summary['ci_results']=rows
        summary['limitations']=['Paired label-stratified resampling assumes pairs are independent; shared code/functions may induce dependence.', 'Project-cluster bootstrap resamples 24 projects with replacement; its uncertainty targets project-level heterogeneity and may be unstable with 24 clusters.', 'No p-values are claimed; percentile CIs are descriptive intervals under their respective resampling designs.', 'Threshold fixed at 0 for F1 and MCC; ROC-AUC/AP are threshold-independent, calculated on continuous logit margins.', 'No test-set threshold calibration performed.', 'Gold labels use reviewer conventions and source leakage audits inherited from D-05bj/D-05c; not reauthenticated.', 'Balanced similarity-selected evaluation sample does not estimate natural-world prevalence.']
        write_json(output/'bootstrap_summary.json',summary)
        names=sorted(x.name for x in output.iterdir())
        write_json(output/'manifest.json',{'version':VERSION,'created_at_chicago':summary['created_at_chicago'],'source_d05c_manifest_sha256':manifest_sha,'source_ground_truth_sha256':summary['source_ground_truth_sha256'],'analysis':{'repetitions':a.repetitions,'seed':a.seed,'sampling':['paired_stratified','project_cluster'],'metrics':list(METRICS)},'files':{name:{'sha256':file_sha(output/name),'bytes':(output/name).stat().st_size} for name in names}})
        print(json.dumps({'status':summary['status'],'output':str(output),'n_pairs':len(ids),'n_projects':len(projects),'repetitions':a.repetitions,'bootstrap_ci':rows},sort_keys=True))
    finally:b.close()

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,zipfile.BadZipFile,UnicodeError) as exc:
        print('D-05e v1 ERROR:',exc,file=sys.stderr);sys.exit(2)
