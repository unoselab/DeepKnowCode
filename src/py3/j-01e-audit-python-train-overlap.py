#!/usr/bin/env python3
"""J-01e: fail-closed, version-specific evaluation vs training overlap audit."""
import argparse,csv,hashlib,json,re,sys
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ID=re.compile(r'^py-sha256:[a-f0-9]{64}$')
FIELDS={'id','function_id','code_id','func_id','pair_left_func','pair_right_func','left_func','right_func','left_id','right_id'}
CODE_FIELDS={'code','source_code','code_text','function_code','code1','code2','left_code','right_code'}
PAIR_FIELDS=[('pair_left_func','pair_right_func'),('left_id','right_id'),('left_func','right_func'),('code1','code2'),('left_code','right_code')]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def func_ids(obj):
 ids=set();c=Counter()
 def visit(x):
  if isinstance(x,dict):
   for k,v in x.items():
    if isinstance(v,str):
     if k in FIELDS and ID.fullmatch(v):ids.add(v);c['explicit_ids']+=1
     elif k in CODE_FIELDS and v.strip() and not ID.fullmatch(v):ids.add('py-sha256:'+hashlib.sha256(v.encode()).hexdigest());c['hashed_code_fields']+=1
    elif isinstance(v,(dict,list)):visit(v)
  elif isinstance(x,list):
   for y in x:visit(y)
 visit(obj);return ids,c
def token(v):
 if not isinstance(v,str):return None
 if ID.fullmatch(v):return v
 if not v.strip():return None
 return 'py-sha256:'+hashlib.sha256(v.encode()).hexdigest()
def explicit_pairs(obj):
 pairs=set()
 def walk(x):
  if isinstance(x,dict):
   for l,r in PAIR_FIELDS:
    if l in x and r in x:
     a,b=token(x[l]),token(x[r]);
     if a and b:pairs.add(tuple(sorted((a,b))))
   for y in x.values():
    if isinstance(y,(dict,list)):walk(y)
  elif isinstance(x,list):
   for y in x:walk(y)
 walk(obj);return pairs
def eval_data(path):
 ids=set(); pairs=set(); labels=Counter(); count=0;per={}
 files=sorted(path.glob('*.csv'))
 if not files:raise ValueError(f'{path}: no CSV files')
 for f in files:
  c=0
  with f.open(encoding='utf-8-sig',newline='') as stream:
   reader=csv.DictReader(stream)
   if reader.fieldnames!=['pair_left_func','pair_right_func','clone_label']:raise ValueError(f'{f}: invalid columns')
   for row in reader:
    a,b,label=row['pair_left_func'],row['pair_right_func'],row['clone_label']
    if not ID.fullmatch(a) or not ID.fullmatch(b) or label not in ('0','1'):raise ValueError(f'{f}: invalid row')
    ids.update((a,b));pairs.add(tuple(sorted((a,b))));labels[label]+=1;count+=1;c+=1
  per[f.name]=c
 return {'ids':ids,'pairs':pairs,'labels':labels,'count':count,'per':per,'files':files}
def main():
 p=argparse.ArgumentParser()
 for x in ('v1-dir','v2-dir','train-file','run-dir','report-name','experiment','version','output-policy','expected-versions-json'):p.add_argument('--'+x,required=True)
 a=p.parse_args();run=Path(a.run_dir).resolve();train=Path(a.train_file).resolve()
 if not run.is_dir() or not train.is_file():raise FileNotFoundError('training file or run directory missing')
 if a.output_policy!='report-only':raise ValueError('only report-only policy supported')
 versions=json.loads(a.expected_versions_json)
 if versions!=['v1','v2']:raise ValueError('expected v1,v2')
 ev={v:eval_data(Path(getattr(a,v+'_dir')).resolve()) for v in versions}
 train_ids=set();train_pairs=set();fields=Counter();records=0;without=0;bad=0;sample=[]
 with train.open(encoding='utf-8') as f:
  for n,line in enumerate(f,1):
   if not line.strip():continue
   obj=json.loads(line);records+=1
   if not isinstance(obj,dict):raise ValueError(f'{train}:{n}: training row not object')
   ids,cts=func_ids(obj);pairs=explicit_pairs(obj);train_ids.update(ids);train_pairs.update(pairs);fields.update(cts)
   if not ids:without+=1
   if len(sample)<3:sample.append(sorted(obj.keys()))
 report={'experiment':a.experiment,'version':a.version,'status':'PASS','timestamp_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),'train_file':str(train),'train_sha256':sha(train),'train_records':records,'train_distinct_function_ids_recognized':len(train_ids),'train_explicit_unordered_pairs':len(train_pairs),'train_rows_without_recognized_function_ids':without,'extraction_counts':dict(fields),'sample_top_level_keys':sample,'training_schema_audit_complete':without==0,'limitations':['Exact match is only detectable for explicit py-sha256 IDs or code-only fields hashed with the same UTF-8 convention.','Text prompts, tokenized sequences, renamed or transformed code, and non-standard code fields may escape detection.','Unordered pair overlap is measurable only for explicitly represented training pairs.','Identical content hashes can refer to distinct source locations.'],'datasets':{}}
 for v,d in ev.items():
  matched=sorted(d['ids']&train_ids);paired=sorted(d['pairs']&train_pairs)
  affected=sum(any(q in train_ids for q in pair) for pair in d['pairs'])
  report['datasets'][v]={'csv_directory':str(Path(getattr(a,v+'_dir')).resolve()),'csv_files':len(d['files']),'csv_sha256':{f.name:sha(f) for f in d['files']},'pairs_rows':d['count'],'unique_unordered_pairs':len(d['pairs']),'unique_function_ids':len(d['ids']),'positive':d['labels']['1'],'negative':d['labels']['0'],'overlapping_function_ids':len(matched),'overlapping_explicit_unordered_pairs':len(paired),'unordered_eval_pairs_with_at_least_one_overlapping_function':affected,'overlap_function_id_examples':matched[:20],'overlap_pair_examples':paired[:20],'interpretation':'INCONCLUSIVE: train records lack recognizable function/code fields' if without else ('OVERLAP_DETECTED' if matched or paired else 'NO_EXACT_OVERLAP_IN_RECOGNIZED_FIELDS')}
 with (run/a.report_name).open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
 print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except Exception as e:print(f'J01E_FAIL:{type(e).__name__}:{e}',file=sys.stderr);sys.exit(1)
