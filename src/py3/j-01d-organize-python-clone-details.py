#!/usr/bin/env python3
"""J-01d: preserve old refactorability details and build provenance-grounded clone_details per version."""
import argparse,csv,hashlib,json,os,shutil,sys,tempfile
from collections import Counter,defaultdict,deque
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

COLS=['pair_left_func','pair_right_func','pair_id','project','gold_label','label_source','source_metadata_json']
OFF=['pair_left_func','pair_right_func','clone_label']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p,cols):
 with p.open(encoding='utf-8-sig',newline='') as f:
  r=csv.DictReader(f)
  if r.fieldnames!=cols:raise ValueError(f'{p}: invalid columns {r.fieldnames}')
  for i,x in enumerate(r,2):
   if None in x or any(v is None for v in x.values()):raise ValueError(f'{p}:{i}: malformed')
   yield x
def read_source_pairs(path):
 """Build canonical provenance records directly from verified source_code and code_sha256."""
 index=defaultdict(lambda:defaultdict(deque)); count=0; ids=set(); counts=Counter()
 with path.open(encoding='utf-8') as f:
  for n,line in enumerate(f,1):
   obj=json.loads(line)
   for required in ('pair_id','project','gold_label','label_source','left','right'):
    if required not in obj:raise ValueError(f'{path}:{n}: missing {required}')
   ident=str(obj['pair_id']); project=str(obj['project']); label=obj['gold_label']
   if ident in ids:raise ValueError(f'{path}:{n}: duplicate pair_id {ident}')
   ids.add(ident)
   if type(label)!=int or label not in (0,1):raise ValueError(f'{path}:{n}: invalid gold_label')
   ids_and_codes=[]
   for side in ('left','right'):
    fun=obj[side]
    if not isinstance(fun,dict):raise ValueError(f'{path}:{n}: invalid {side} object')
    code=fun.get('source_code'); code_hash=fun.get('code_sha256')
    if not isinstance(code,str) or not isinstance(code_hash,str):
     raise ValueError(f'{path}:{n}: {side}.source_code/code_sha256 missing')
    computed=hashlib.sha256(code.encode('utf-8')).hexdigest()
    if computed!=code_hash:
     raise ValueError(f'{path}:{n}: {side} source_code SHA256 mismatch: expected={code_hash} computed={computed}')
    ids_and_codes.append('py-sha256:'+code_hash)
   if 'pair_code_hashes' in obj:
    if sorted(obj['pair_code_hashes'])!=sorted([obj['left']['code_sha256'],obj['right']['code_sha256']]):
     raise ValueError(f'{path}:{n}: pair_code_hashes mismatch')
   metadata={k:v for k,v in obj.items() if k not in ('left','right')}
   metadata['left']={k:v for k,v in obj['left'].items() if k!='source_code'}
   metadata['right']={k:v for k,v in obj['right'].items() if k!='source_code'}
   record={'pair_left_func':ids_and_codes[0],'pair_right_func':ids_and_codes[1],
           'pair_id':ident,'project':project,'gold_label':str(label),
           'label_source':str(obj['label_source']),
           'source_metadata_json':json.dumps(metadata,ensure_ascii=False,sort_keys=True,separators=(',',':'))}
   index[project][(record['pair_left_func'],record['pair_right_func'],record['gold_label'])].append(record)
   count+=1;counts[str(label)]+=1
 return index,count,counts

def main():
 p=argparse.ArgumentParser()
 for key in ('root','source','run-dir','mode','experiment','version','report-name','old-dir','renamed-dir','details-dir','versions-json','owner-map-json','expected-pairs-json','expected-detail-cols-json'):p.add_argument('--'+key,required=True)
 a=p.parse_args(); root=Path(a.root).resolve(); src=Path(a.source).resolve(); run=Path(a.run_dir).resolve(); old=root/a.old_dir; renamed=root/a.renamed_dir; dest=root/a.details_dir
 versions=json.loads(a.versions_json); owners=json.loads(a.owner_map_json); expected=json.loads(a.expected_pairs_json)
 if a.mode not in ('inspect','build'):raise ValueError('mode must be inspect/build')
 if json.loads(a.expected_detail_cols_json)!=COLS or not versions or any(v not in ('v1','v2') for v in versions):raise ValueError('invalid config')
 if not isinstance(owners,dict):raise ValueError('owner map must be a dictionary')
 if not run.is_dir() or not src.is_file():raise ValueError('missing run directory or original verified source')
 index,source_total,source_labels=read_source_pairs(src)
 if old.exists() and renamed.exists():raise FileExistsError('details and refactorability both exist; manual review needed')
 if old.exists() and not old.is_dir():raise ValueError('old details is not directory')
 if renamed.exists() and not renamed.is_dir():raise ValueError('refactorability is not directory')
 # Historical refactorability data are not a source for clone metadata.
 # Rename only after entire V1/V2 dataset has been validated.
 legacy=old if old.exists() else renamed if renamed.exists() else None
 legacy_csv=list(legacy.glob('*.csv')) if legacy else []
 planned={}; report={'experiment':a.experiment,'version':a.version,'status':'VALIDATED','timestamp_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),'mode':a.mode,'source_sha256':sha(src),'source_pairs_verified':source_total,'metadata_mapping':'verified_pairs_source_code_sha256','historical_directory':str(legacy) if legacy else None,'historical_csv_sha256':{p.name:sha(p) for p in legacy_csv},'source_label_counts':dict(source_labels),'versions':{}}
 for ver in versions:
  folder=root/ver
  if not folder.is_dir():raise FileNotFoundError(folder)
  expected_names={f'{owner.lower()}__{project.lower()}.csv':project for project,owner in owners.items()}
  legacy_names={f'{owner.lower()}_{project.lower()}.csv':project for project,owner in owners.items()}
  files={p.name:p for p in folder.glob('*.csv')}
  if set(files)==set(expected_names):m=expected_names
  elif set(files)==set(legacy_names):m=legacy_names
  else:raise ValueError(f'{ver}: unexpected CSV filenames {sorted(set(files)^set(expected_names))}')
  counts=Counter(); output={}
  for filename,proj in m.items():
   csvpath=files[filename]; available={key:deque(e.copy() for e in entries) for key,entries in index[proj].items()}
   found=[]
   for r in rows(csvpath,OFF):
    k=(r['pair_left_func'],r['pair_right_func'],r['clone_label'])
    if not available.get(k):raise ValueError(f'{csvpath}: pair missing from verified source or multiplicity exceeded')
    found.append(available[k].popleft());counts[r['clone_label']]+=1
   output[f'{owners[proj].lower()}__{proj.lower()}.csv']=found
  expected_count=expected[ver]
  if len(output)!=len(owners) or sum(len(x) for x in output.values())!=expected_count:raise ValueError(f'{ver}: expected {expected_count} rows')
  if ver=='v1' and sum(len(x) for x in output.values())!=source_total:raise ValueError('v1 not full source')
  if ver=='v2' and counts['0']!=counts['1']:raise ValueError('v2 imbalance')
  if ver=='v2' and any(sum(1 for r in rr if r['gold_label']=='0') != sum(1 for r in rr if r['gold_label']=='1') for rr in output.values()):raise ValueError('v2 project-level imbalance')
  planned[ver]=output
  report['versions'][ver]={'pairs':sum(len(x) for x in output.values()),'positive':counts['1'],'negative':counts['0'],'files':len(output),'official_csv_sha256':{name:sha(path) for name,path in files.items()}}
 v1ids={r['pair_id'] for rr in planned['v1'].values() for r in rr}
 v2ids={r['pair_id'] for rr in planned['v2'].values() for r in rr}
 if len(v1ids)!=source_total or len(v2ids)!=expected['v2'] or not v2ids.issubset(v1ids):
  raise ValueError('V1/V2 provenance identity mismatch')
 if dest.exists():raise FileExistsError(f'{dest}: destination exists; no overwrite')
 if a.mode=='build':
  # Stage from original verified_pairs.jsonl; legacy details CSV are not read.
  tmp=Path(tempfile.mkdtemp(prefix='.j01d-',dir=str(root)))
  moved=False
  try:
   for ver,output in planned.items():
    sub=tmp/ver;sub.mkdir()
    for name,rr in output.items():
     with (sub/name).open('x',encoding='utf-8',newline='') as f:
      w=csv.DictWriter(f,fieldnames=COLS,lineterminator='\n');w.writeheader();w.writerows(rr)
   # Preserve original refactorability CSV files byte-for-byte when renamed.
   if old.exists():
    if renamed.exists():raise FileExistsError('refactorability destination exists')
    old.rename(renamed);moved=True
   try:tmp.rename(dest)
   except Exception:
    if moved:renamed.rename(old)
    raise
   report['status']='PASS';report['clone_detail_dir']=str(dest)
  finally:
   if tmp.exists():shutil.rmtree(tmp)
 else:report['status']='PASS';report['planned_clone_detail_dir']=str(dest)
 with (run/a.report_name).open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
 print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except Exception as e:print(f'J01D_FAIL:{type(e).__name__}:{e}',file=sys.stderr);sys.exit(1)
