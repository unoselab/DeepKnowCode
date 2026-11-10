#!/usr/bin/env python3
"""D-05h v1: read-only aggregation and cross-stage integrity audit (D-05c..D-05g)."""
import argparse,csv,hashlib,json,math,os,sys,zipfile,tempfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from contextlib import ExitStack

VERSION='v1'; STAGES=('d05c','d05d','d05e','d05f','d05g'); MODELS=('M0','M1','M2','M3')
DEFAULTS={
 'd05c':'results/d-05c/20261109-004326-58663',
 'd05d':'results/d-05d/python-threshold-independent-v1-20261110-124418-385790',
 'd05e':'results/d-05e/python-paired-bootstrap-v1-20261110-125627-333861',
 'd05f':'results/d-05f/python-mcnemar-project-v1-20261110-130607-764826',
 'd05g':'results/d-05g/python-error-analysis-v1-20261110-131628-87808'}

def require(x,msg):
 if not x: raise ValueError('D-05h v1 BLOCKED: '+msg)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def jread(p):return json.loads(p.read_text(encoding='utf-8'))
def csvread(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def close(x,y,label,eps=1e-10):
 require(math.isclose(float(x),float(y),rel_tol=eps,abs_tol=eps),f'{label}: {x} != {y}')
def putjson(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
def putcsv(p,rows,cols):
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
def path_for(root,val):
 p=Path(val).expanduser();return (p if p.is_absolute() else root/p).resolve()
def resolve_dir(p,stack):
 if p.is_dir():return p
 require(p.is_file() and p.suffix.lower()=='.zip',f'input missing: {p}')
 tmp=Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='d05h-input-')))
 with zipfile.ZipFile(p) as z:
  bad=z.testzip();require(bad is None,'ZIP CRC failure '+str(p))
  for m in z.infolist():
   dest=(tmp/m.filename).resolve()
   require(dest.is_relative_to(tmp.resolve()),'Unsafe ZIP entry')
   if m.is_dir():dest.mkdir(parents=True,exist_ok=True);continue
   dest.parent.mkdir(parents=True,exist_ok=True)
   with z.open(m) as source,dest.open('wb') as target:
    import shutil;shutil.copyfileobj(source,target)
 tops=[x for x in tmp.iterdir() if x.is_dir()]
 return tops[0] if len(tops)==1 and (tops[0]/'manifest.json').is_file() else tmp

def check_manifest(d,stage):
 m=d/'manifest.json';require(m.is_file(),'missing '+str(m));manifest=jread(m)
 require(manifest.get('version')=='v1',f'{stage}: version unexpected')
 files=manifest.get('files');require(isinstance(files,dict) and files,f'{stage}: missing files index')
 for name,meta in files.items():
  require(Path(name).name==name,f'{stage}: unsafe filename')
  p=d/name;require(p.is_file(),f'{stage}: missing {name}')
  require(p.stat().st_size==int(meta['bytes']),f'{stage}: byte mismatch {name}')
  require(sha(p)==meta['sha256'],f'{stage}: SHA256 mismatch {name}')
 return manifest,sha(m)

def cm(rows,model):
 d={'tp':0,'fp':0,'tn':0,'fn':0};seen=set();by={}
 for r in rows:
  require(r['model']==model,'model name mismatch');pid=r['pair_id'];require(pid not in seen,'duplicate pair ID');seen.add(pid)
  y=int(r['true_label']);p=int(r['predicted_label']);score=float(r['mean_logit_margin']);require(y in (0,1) and p in (0,1) and math.isfinite(score),'bad prediction row')
  require(p==int(score>0),'fixed threshold mismatch')
  d['tp' if y==1 and p==1 else 'fn' if y==1 else 'fp' if p==1 else 'tn']+=1
  by[pid]=(y,p,r['project'])
 return d,by

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
 for stage in STAGES:ap.add_argument('--'+stage,default=DEFAULTS[stage],help='D-05 stage directory or ZIP')
 ap.add_argument('--out',help='Fresh output directory; default under results/d-05h')
 ap.add_argument('--dry-run',action='store_true')
 args=ap.parse_args();root=args.root.resolve();now=datetime.now(ZoneInfo('America/Chicago'))
 out=path_for(root,args.out) if args.out else root/'results/d-05h'/('python-statistical-consolidation-v1-'+now.strftime('%Y%m%d-%H%M%S-%f'))
 require(not out.exists(),f'output exists: {out}')
 with ExitStack() as stack:
  paths={k:resolve_dir(path_for(root,getattr(args,k)),stack) for k in STAGES}
  mm={};shas={}
  for k in STAGES:mm[k],shas[k]=check_manifest(paths[k],k)
  csha=shas['d05c']; gt=jread(paths['d05c']/'summary.json')['ground_truth_sha256'];c=jread(paths['d05c']/'summary.json')
  require(c.get('status')=='D05C_V1_EVALUATION_COMPLETE','D05c status')
  for k in ('d05d','d05f','d05g'):require(mm[k].get('source_manifest_sha256')==csha,f'{k}: wrong D05c manifest')
  require(mm['d05e'].get('source_d05c_manifest_sha256')==csha,'D05e: wrong D05c manifest')
  require(mm['d05e'].get('source_ground_truth_sha256')==gt,'D05e: wrong GT SHA')
  d=jread(paths['d05d']/'analysis_summary.json');e=jread(paths['d05e']/'bootstrap_summary.json');f=jread(paths['d05f']/'analysis_summary.json');g=jread(paths['d05g']/'analysis_summary.json')
  expected={'d05d':'D05D_V1_THRESHOLD_INDEPENDENT_COMPLETE','d05e':'D05E_V1_PAIRED_BOOTSTRAP_COMPLETE','d05f':'D05F_V1_MCNEMAR_PROJECT_COMPARISON_COMPLETE','d05g':'D05G_V1_ERROR_ANALYSIS_COMPLETE'}
  for k,x in [('d05d',d),('d05e',e),('d05f',f),('d05g',g)]:
   require(x.get('status')==expected[k],f'{k}: wrong status')
   require(x.get('source_manifest_sha256',x.get('provenance',{}).get('source_manifest_sha256'))==csha,f'{k}: summary source mismatch')
   require(x.get('source_ground_truth_sha256',x.get('provenance',{}).get('ground_truth_sha256'))==gt,f'{k}: GT mismatch')
  auc={r['model']:r for r in csvread(paths['d05d']/'auc_metrics.csv')}
  err={r['model']:r for r in csvread(paths['d05g']/'model_error_summary.csv')}
  require(set(auc)==set(MODELS) and set(err)==set(MODELS),'model coverage mismatch')
  tables=[];by_model={}; ground=None;pair_projects=set()
  for model in MODELS:
   rows=csvread(paths['d05c']/f'predictions_{model}.csv');conf,pairs=cm(rows,model)
   if ground is None:ground={k:(v[0],v[2]) for k,v in pairs.items()};pair_projects={x[2] for x in pairs.values()}
   else:require({k:(v[0],v[2]) for k,v in pairs.items()}==ground,'pair/label/project alignment mismatch')
   by_model[model]=pairs
   s=c['model_metrics'][model]
   for k in ('tp','fp','tn','fn'):require(conf[k]==s[k]==int(err[model][k]),f'{model} {k} mismatch')
   require(len(rows)==int(s['n'])==int(auc[model]['n'])==7942,'record count mismatch')
   for metric in ('accuracy','precision','recall'):
    close(s[metric],err[model][metric],model+metric)
   for metric in ('roc_auc','average_precision','pr_auc_trapezoidal'):
    close(auc[model][metric],d['auc'][model][metric],model+metric)
   tables.append({'model':model,'n':len(rows),**{k:s[k] for k in ('tp','tn','fp','fn','accuracy','precision','recall','f1','mcc')},'fpr':int(s['fp'])/(int(s['fp'])+int(s['tn'])),'roc_auc':float(auc[model]['roc_auc']),'average_precision':float(auc[model]['average_precision']),'pr_auc_trapezoidal':float(auc[model]['pr_auc_trapezoidal'])})
  require(len(ground)==7942 and len(pair_projects)==24,'unmatched GT size/projects')
  require(sum(y for y,project in ground.values())==3971,'GT positives != 3971')
  ci=csvread(paths['d05e']/'bootstrap_ci.csv');require(len(ci)==8,'expected eight CI rows');cis={(r['sampling'],r['metric']):r for r in ci}
  require(len(cis)==8,'duplicate bootstrap CI');require(set(x[0] for x in cis)=={'paired_stratified','project_cluster'},'bootstrap methods')
  for (samp,metric),row in cis.items():
   require(int(row['bootstrap_repetitions'])==10000,'bootstrap repetition mismatch')
   close(float(row['M2'])-float(row['M3']),row['delta_M2_minus_M3'],'bootstrap delta')
   if metric in ('f1','mcc'):
    for mod in ('M2','M3'):close(row[mod],c['model_metrics'][mod][metric],f'CI {mod} {metric}')
   else:
    for mod in ('M2','M3'):close(row[mod],auc[mod][metric],f'CI {mod} {metric}')
   match=[x for x in e['ci_results'] if x['sampling']==samp and x['metric']==metric]
   require(len(match)==1,'CI not in summary')
   for k in ('ci_lower_95','ci_upper_95','delta_M2_minus_M3'):close(row[k],match[0][k],f'CI {samp} {metric} {k}')
  mc=csvread(paths['d05f']/'mcnemar_results.csv');global_row=next((x for x in mc if x['project']=='ALL_24_PROJECTS'),None);require(global_row is not None,'missing global McNemar')
  m2,m3=by_model['M2'],by_model['M3'];both_correct=both_wrong=m2only=m3only=0
  for pid in ground:
   y=ground[pid][0];a=m2[pid][1]==y;b=m3[pid][1]==y
   if a and b:both_correct+=1
   elif not a and not b:both_wrong+=1
   elif a:m2only+=1
   else:m3only+=1
  counts={'both_correct':both_correct,'both_incorrect':both_wrong,'m2_only_correct':m2only,'m3_only_correct':m3only,'discordant':m2only+m3only}
  for k,v in counts.items():require(v==int(global_row[k])==int(f['mcnemar_exact_global'][k]),f'McNemar {k} mismatch')
  require(counts['discordant']==int(g['counts']['output_rows']['m2_m3_disagreements.csv']),'D05g disagreement mismatch')
  require(counts['m2_only_correct']==int(g['m2_m3_disagreement']['M2']) and counts['m3_only_correct']==int(g['m2_m3_disagreement']['M3']),'D05g winners mismatch')
  close(global_row['delta_accuracy_M2_minus_M3'],float(c['model_metrics']['M2']['accuracy'])-float(c['model_metrics']['M3']['accuracy']),'accuracy delta')
  close(global_row['exact_two_sided_p'],f['mcnemar_exact_global']['exact_two_sided_p'],'global McNemar p')
  identical=csvread(paths['d05g']/'identical_positive_detection.csv');require(len(identical)==4 and all(int(x['n_identical_positive'])==85 for x in identical),'85 identical pairs mismatch')
  require(g['similarity_analysis_status']=='NOT_AVAILABLE_IN_D05C_INPUT','Similarity status unexpectedly changed')
  requires={'d05c':'summary.json','d05d':'analysis_summary.json','d05e':'bootstrap_summary.json','d05f':'analysis_summary.json','d05g':'analysis_summary.json'}
  audit={'version':VERSION,'status':'D05H_V1_PREFLIGHT_PASS_NOT_CONSOLIDATED' if args.dry_run else 'D05H_V1_CONSOLIDATION_COMPLETE','created_at_chicago':now.isoformat(),'n_pairs':len(ground),'n_projects':len(pair_projects),'positive':3971,'negative':3971,'models':list(MODELS),'d05c_ground_truth_sha256':gt,'source_manifests':{k:{'path':str(paths[k]/'manifest.json'),'sha256':shas[k],'file_count':len(mm[k]['files']),'manifest_file_sha256_validated':True} for k in STAGES},'cross_stage_checks':{'all_manifest_files_sha256_verified':True,'same_d05c_source_manifest':True,'same_ground_truth_sha256':True,'prediction_alignment_verified':True,'d05c_d05d_d05g_metrics_agree':True,'d05e_ci_point_estimates_agree':True,'d05e_ci_summary_matches_csv':True,'d05f_pair_correctness_recomputed':True,'d05g_disagreement_matches_d05f':True,'identical_positive_85_verified':True},'m2_m3_discordance':counts,'similarity_analysis_status':g['similarity_analysis_status'],'limitations':list(dict.fromkeys(c.get('limitations',[])+e.get('limitations',[])+f.get('limitations',[])+g.get('limitations',[])))}
  if args.dry_run:
   print(json.dumps({'status':audit['status'],'models':list(MODELS),'n_pairs':len(ground),'n_projects':len(pair_projects),'manifest_checks':len(STAGES),'output_if_executed':str(out)},sort_keys=True));return
  out.parent.mkdir(parents=True,exist_ok=True);out.mkdir(exist_ok=False)
  auc_rows=[{'model':x['model'],**{k:x[k] for k in ('roc_auc','average_precision','pr_auc_trapezoidal')},'positive_prevalence':0.5} for x in tables]
  putcsv(out/'table_model_performance.csv',tables,list(tables[0]))
  putcsv(out/'table_auc_comparison.csv',auc_rows,list(auc_rows[0]))
  putcsv(out/'table_bootstrap_ci.csv',ci,list(ci[0]))
  putcsv(out/'table_mcnemar.csv',mc,list(mc[0]))
  putcsv(out/'table_error_analysis.csv',csvread(paths['d05g']/'model_error_summary.csv'),list(err['M0']))
  putcsv(out/'table_project_comparison.csv',csvread(paths['d05f']/'project_differences.csv'),list(csvread(paths['d05f']/'project_differences.csv')[0]))
  putcsv(out/'table_identical_positive.csv',identical,list(identical[0]))
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  for source,name,xkey,ykey,xlabel,ylabel,title in [('roc_curve.csv','figure_roc_curves.png','false_positive_rate','true_positive_rate','False positive rate','True positive rate','ROC curves'),('pr_curve.csv','figure_pr_curves.png','recall','precision','Recall','Precision','Precision-recall curves')]:
   curve=csvread(paths['d05d']/source);fig,ax=plt.subplots(figsize=(7,5))
   for model in MODELS:
    rr=[x for x in curve if x['model']==model];ax.plot([float(x[xkey]) for x in rr],[float(x[ykey]) for x in rr],label=model,linewidth=1.5)
   ax.set(xlabel=xlabel,ylabel=ylabel,title=title,xlim=(0,1),ylim=(0,1));ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/name,dpi=180);plt.close(fig)
  fig,ax=plt.subplots(figsize=(8,4.5));x=range(4);ax.bar([i-.18 for i in x],[q['fp'] for q in tables],width=.36,label='False positive');ax.bar([i+.18 for i in x],[q['fn'] for q in tables],width=.36,label='False negative');ax.set_xticks(list(x),MODELS);ax.set(title='D-05c error counts (threshold = 0)',ylabel='Pairs');ax.legend();fig.tight_layout();fig.savefig(out/'figure_confusion_errors.png',dpi=180);plt.close(fig)
  fig,ax=plt.subplots(figsize=(8,4.8));metrics=('f1','mcc','roc_auc','average_precision');y=range(4)
  for j,samp in enumerate(('paired_stratified','project_cluster')):
   shift=(-.11 if j==0 else .11)
   for i,metric in enumerate(metrics):
    z=cis[samp,metric];mid=float(z['delta_M2_minus_M3']);lo=float(z['ci_lower_95']);hi=float(z['ci_upper_95'])
    ax.errorbar(mid,i+shift,xerr=[[mid-lo],[hi-mid]],fmt=('o' if j==0 else 's'),capsize=2,label=samp if i==0 else None)
  ax.axvline(0,color='black',linewidth=.8,linestyle='--');ax.set_yticks(list(y),metrics);ax.invert_yaxis();ax.set(xlabel='M2 minus M3 (95% percentile CI)',title='Paired vs project-cluster bootstrap');ax.legend();fig.tight_layout();fig.savefig(out/'figure_bootstrap_ci.png',dpi=180);plt.close(fig)
  summary={'status':audit['status'],'created_at_chicago':now.isoformat(),'dataset':{'pairs':7942,'positive':3971,'negative':3971,'projects':24},'source_manifest_sha256s':{k:shas[k] for k in STAGES},'model_performance':tables,'m2_m3_bootstrap_ci':ci,'global_mcnemar':global_row,'macro_metrics':f['macro_metrics'],'project_holm_significant_0_05':f['project_holm_significant_0_05'],'identical_positive_detection':identical,'similarity_analysis_status':g['similarity_analysis_status'],'interpretation':'M2 leads F1/MCC/ROC-AUC/AP at the frozen decision rule/data; M3 has higher recall but greater false positives. Pair and cluster bootstrap CIs exclude zero for four M2-minus-M3 metrics. No causal or population claim.','limitations':audit['limitations']}
  putjson(out/'statistical_summary.json',summary);putjson(out/'reproducibility_manifest.json',audit)
  files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file()}
  putjson(out/'manifest.json',{'version':VERSION,'status':audit['status'],'created_at_chicago':now.isoformat(),'files':files,'source_manifest_sha256s':{k:shas[k] for k in STAGES},'ground_truth_sha256':gt})
  print(json.dumps({'status':audit['status'],'output':str(out),'models':list(MODELS),'pairs':len(ground),'projects':len(pair_projects),'source_manifests_verified':len(STAGES),'output_files_verified':len(files),'m2_only_correct':m2only,'m3_only_correct':m3only},sort_keys=True))

if __name__=='__main__':
 try:main()
 except Exception as exc:
  print('D-05h v1 ERROR:',str(exc),file=sys.stderr);sys.exit(2)
