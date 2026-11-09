#!/usr/bin/env python3
"""D-05ba v6: active-training schema-aware Python source/line and leakage audit.

Uses only existing D-05b v4 candidates, active repos, and training inputs.
Never runs clone detection; never assigns gold labels; never overwrites results.
"""
import argparse
import ast
import collections
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION='v6'
STATUS='D05BA_V6_ACTIVE_TRAINING_HASH_AUDIT_GOLD_NOT_VERIFIED'
TRAIN_KEYS=(('code1','code2'),('code_a','code_b'),('left_code','right_code'),('source1','source2'),('source_code1','source_code2'),('snippet1','snippet2'),('func1','func2'))

def digest(b):return hashlib.sha256(b).hexdigest()
def normalize(s):return ''.join(s.split())
def hcode(s):return digest(s.encode('utf-8')),digest(normalize(s).encode('utf-8'))
def checksum(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def relative(p,root):return str(p.relative_to(root))
def jlines(path):
 with path.open(encoding='utf-8-sig') as f:
  for line_no,line in enumerate(f,1):
   if line.strip():
    obj=json.loads(line)
    if not isinstance(obj,dict):raise ValueError(f'Expected object {path}:{line_no}')
    yield obj

def safe_file(root,project,name):
 base=(root/'data/repositories'/project).resolve()
 if not name or '\\' in name or '\x00' in name or re.match(r'^[A-Za-z]:',name) or os.path.isabs(name):return None
 rel=Path(name)
 if '..' in rel.parts:return None
 opts=[base/rel]
 if rel.parts and rel.parts[0]==project:opts.append(base.joinpath(*rel.parts[1:]))
 hits=set()
 for path in opts:
  resolved=path.resolve()
  if resolved.is_relative_to(base) and resolved.is_file() and resolved.suffix=='.py':hits.add(resolved)
 return next(iter(hits)) if len(hits)==1 else None

def function_nodes(tree):
 output=[]
 stack=[(tree,[])]
 while stack:
  node,parents=stack.pop()
  current=parents
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
   start=node.lineno
   decorated=min([start]+[getattr(d,'lineno',start) for d in node.decorator_list])
   output.append({'name':node.name,'qualname':'.'.join(parents+[node.name]),'start':start,
                  'decorated':decorated,'end':node.end_lineno})
   current=parents+[node.name]
  elif isinstance(node,ast.ClassDef):current=parents+[node.name]
  stack.extend((child,current) for child in ast.iter_child_nodes(node))
 return output

def make_source_resolver(root,expected_commits):
 cache={};stats=collections.Counter(); filehash={}
 def resolve(project,loc):
  if not isinstance(loc,dict):return {'status':'BAD_LOCATOR'}
  path=safe_file(root,project,loc.get('file'))
  if path is None:return {'status':'FILE_UNRESOLVED'}
  key=str(path)
  if key not in cache:
   try:
    raw=path.read_bytes();txt=raw.decode('utf-8')
    with warnings.catch_warnings():
     warnings.simplefilter('ignore',SyntaxWarning)
     tree=ast.parse(txt,filename=key)
    cache[key]=(txt.splitlines(keepends=True),function_nodes(tree),None)
    filehash[key]=digest(raw)
   except (UnicodeError,SyntaxError,RecursionError,ValueError,OSError) as exc:
    cache[key]=(None,None,type(exc).__name__)
  lines,nodes,error=cache[key]
  if error:return {'status':'AST_PARSE_ERROR','error':error}
  try:start=int(loc.get('start'));end=int(loc.get('end'))
  except (TypeError,ValueError):return {'status':'BAD_LINE_RANGE'}
  if start<=0 or end<start or end>len(lines):return {'status':'BAD_LINE_RANGE'}
  rawname=str(loc.get('function') or '').strip()
  simple=rawname.rsplit('.',1)[-1]
  if not simple:return {'status':'NAME_MISSING'}
  # Strict: expected function name, then range boundaries. No fuzzy nearest-function guesses.
  exact=[n for n in nodes if n['name']==simple and n['end']==end and (n['start']==start or n['decorated']==start)]
  if len(exact)!=1:
   return {'status':'AMBIGUOUS_AST' if len(exact)>1 else 'AST_SPAN_OR_NAME_MISMATCH',
           'candidate_name_matches':sum(n['name']==simple for n in nodes)}
  n=exact[0]
  if '.' in rawname and n['qualname']!=rawname:
   return {'status':'QUALIFIED_NAME_MISMATCH','ast_qualname':n['qualname']}
  code=''.join(lines[n['decorated']-1:n['end']]);raw,normal=hcode(code)
  return {'status':'EXACT_AST_FUNCTION','ast_qualname':n['qualname'],'start_line':n['decorated'],
          'end_line':n['end'],'code_sha256':raw,'normalized_code_sha256':normal,
          'repository_file':relative(path,root),'repository_file_sha256':filehash[key],
          'code_length':len(code),'code':code}
 return resolve,stats

def active_training_hashes(root, code_index):
 """Pair ID -> active corpus idx/func. No legacy directories or pseudo-labels."""
 base=root/'data/training'; prepared=base/'prepared/balanced_multilingual'
 corpus_names=('java_source','java_mixed','python_mixed')
 corpus_data={}; corpus_audit={}; input_hashes={}
 for corpus in corpus_names:
  path=base/'clean'/corpus/'data.jsonl'
  if not path.is_file():raise ValueError('Required ACTIVE function corpus missing: '+str(path))
  input_hashes[relative(path,root)]=checksum(path)
  lut={}; duplicates=0
  for idx,obj in enumerate(jlines(path),1):
   ident,code=obj.get('idx'),obj.get('func')
   if not isinstance(ident,str) or not ident or not isinstance(code,str) or not code.strip():
    raise ValueError(f'Invalid active function record {path}:{idx} (expected idx and func)')
   if ident in lut:
    duplicates+=1;raise ValueError(f'Duplicate active function id {corpus}:{ident}')
   lut[ident]=code
  corpus_data[corpus]=lut
  corpus_audit[corpus]={'path':relative(path,root),'sha256':input_hashes[relative(path,root)],'functions':len(lut),'duplicate_ids':duplicates}
 exact={'train':set(),'validation':set()};normal={'train':set(),'validation':set()}
 pairs={'train':set(),'validation':set()}; code_pairs={'train':set(),'validation':set()}; seenids={'train':set(),'validation':set()}
 report={};errors=collections.Counter();python_pair_ids={'train':set(),'validation':set()}
 for split in ('train','validation'):
  path=prepared/(split+'.jsonl')
  if not path.is_file():raise ValueError('Required active split missing: '+str(path))
  input_hashes[relative(path,root)]=checksum(path)
  counts=collections.Counter();unresolved=[];identity_collisions=[]
  for rowno,obj in enumerate(jlines(path),1):
   corpus=obj.get('corpus');id1=obj.get('id1');id2=obj.get('id2')
   if corpus not in corpus_data or not isinstance(id1,str) or not isinstance(id2,str) or not id1 or not id2:
    raise ValueError(f'Unexpected active training row schema at {path}:{rowno}')
   counts['rows_total']+=1;counts['rows_'+corpus]+=1
   key=(corpus,*sorted((id1,id2)))
   if key in pairs[split]:counts['duplicate_pair_rows']+=1
   pairs[split].add(key)
   if corpus=='python_mixed':python_pair_ids[split].add(key)
   look=corpus_data[corpus]
   for ident in (id1,id2):
    counts['function_occurrences']+=1
    seenids[split].add((corpus,ident))
    if ident not in look:
     counts['unmapped_function_occurrences']+=1
     if len(unresolved)<30:unresolved.append({'row':rowno,'corpus':corpus,'id':ident})
     continue
    counts['mapped_function_occurrences']+=1
    if corpus=='python_mixed':
     h,n=hcode(look[ident]);exact[split].add(h);normal[split].add(n)
     counts['python_function_occurrences_mapped']+=1
   if corpus=='python_mixed':
    if id1 in look and id2 in look:
     counts['python_pairs_fully_resolved']+=1
     pairhashes=tuple(sorted((hcode(look[id1])[0],hcode(look[id2])[0])))
     code_pairs[split].add(pairhashes)
    else:counts['python_pairs_unresolved']+=1
  report[split]={'path':relative(path,root),'sha256':input_hashes[relative(path,root)],
                 'counts':dict(counts),'unmapped_examples_max30':unresolved,
                 'unique_python_exact_hashes':len(exact[split]),'unique_python_normalized_hashes':len(normal[split])}
  if counts['unmapped_function_occurrences']:errors['unmapped_'+split]=counts['unmapped_function_occurrences']
 # The held-out evaluation uses Python only. Java is indexed for audit completeness,
 # not falsely treated as Python-language code. Cross-language semantic leakage is not tested.
 joined=exact['train']|exact['validation'];norm_joined=normal['train']|normal['validation']
 shared=exact['train']&exact['validation']
 return {'train':exact['train'],'validation':exact['validation']}, {'train':normal['train'],'validation':normal['validation']}, code_pairs, {
    'active_corpses':corpus_audit,'split_report':report,'input_sha256':input_hashes,
    'python_ids':{k:len([x for x in v if x[0]=='python_mixed']) for k,v in seenids.items()},
    'unique_python_pair_keys':{k:len(v) for k,v in python_pair_ids.items()},
    'train_validation_shared_exact_code_hashes':len(shared),
    'training_reference_complete':not bool(errors),
    'training_mapping_errors':dict(errors),
    'unique_python_train_and_validation_exact_code_hashes':len(joined),
    'unique_python_train_and_validation_normalized_code_hashes':len(norm_joined),
    'M1_training_exposure_scope':'NOT_ESTABLISHED_FROM_BALANCED_MULTILINGUAL_SPLITS',
    'near_duplicate_training_exposure_scope':'NOT_CHECKED',
    'active_only_legacy_training_excluded':True,
    'legacy_csv_only_as_v4_candidate_provenance':True
 }

def run(args):
 root=args.root.resolve();source=args.source.resolve();dest=args.output.resolve()
 if not source.is_dir():raise ValueError('D-05b v4 source directory absent: '+str(source))
 if dest.exists():raise ValueError('v6 destination already exists (no overwrites): '+str(dest))
 if source==dest or dest in source.parents or source in dest.parents:raise ValueError('Output/source overlap')
 candfile=source/'candidates.jsonl';auditfile=source/'provenance_leakage_audit.json';manifestfile=source/'manifest.json'
 for p in [candfile,auditfile,manifestfile]:
  if not p.is_file():raise ValueError('Missing D-05b v4 input: '+str(p))
 audit=json.loads(auditfile.read_text());manifest=json.loads(manifestfile.read_text())
 for name in ('candidates.jsonl','provenance_leakage_audit.json'):
  expected=(manifest.get('artifacts_sha256') or {}).get(name)
  if not expected or checksum(source/name)!=expected:raise ValueError('D-05b v4 manifest SHA256 mismatch: '+name)
 print('Read-only D-05b v4 input:',source,flush=True)
 if audit.get('counts',{}).get('c01_projects_with_results')!=24:raise ValueError('v4 C-01 project coverage not 24/24')
 active=set(audit.get('active_repositories',[]));commits=audit.get('active_repository_commits',{})
 if len(active)!=24 or 'scikit-learn' not in active or 'azure-sdk-for-python' in active:raise ValueError('Wrong active subject roster')
 for project in sorted(active):
  git=subprocess.run(['git','-C',str(root/'data/repositories'/project),'rev-parse','HEAD'],capture_output=True,text=True,check=False)
  if git.returncode or git.stdout.strip()!=commits.get(project):raise ValueError('Active repository HEAD changed: '+project)
 codefile=root/'data/evaluation/code/python.jsonl'
 code_index={};code_values={}
 if not codefile.is_file():raise ValueError('Missing python.jsonl')
 for o in jlines(codefile):
  ident=o.get('id');code=o.get('code')
  if not isinstance(ident,str) or not isinstance(code,str) or ident in code_index:raise ValueError('Invalid/duplicate python code corpus')
  code_index[ident]=hcode(code);code_values[ident]=code
 train_exact,train_normalized,training_code_pairs,training_audit=active_training_hashes(root,code_index)
 train_raw=train_exact["train"]|train_exact["validation"]
 train_norm=train_normalized["train"]|train_normalized["validation"]
 resolve,_=make_source_resolver(root,commits)
 records=list(jlines(candfile));stats=collections.Counter();projectstats=collections.defaultdict(collections.Counter)
 if len(records)!=audit.get('counts',{}).get('total_candidate_records'):raise ValueError('Candidate counts do not match D-05b v4 audit')
 # Collect all exact-resolved code hashes first, so duplicate reports cover all origins.
 exact_pair_records=collections.defaultdict(list)
 for o in records:
  if o.get('gold_label') is not None or o.get('evaluation_eligible') is not False:raise ValueError('Unexpected verified/evaluable v4 candidate')
  project=o['project'];origin=o['origin'];flags=set(o.get('review_flags',[]))
  o['v6_source_resolution']={}
  for side in ('left','right'):
   if origin=='c01_hybrid_detector':
    r=resolve(project,o.get(side+'_locator'))
    # code string is not duplicated into audit output; annotations can use source spans separately
    if 'code' in r:r.pop('code')
   elif origin=='legacy_prediction_csv':
    ident=o.get(side+'_id')
    if ident in code_index:
     rh,nh=code_index[ident]
     r={'status':'EXACT_EVALUATION_CORPUS_ID','code_sha256':rh,'normalized_code_sha256':nh,'code_length':len(code_values[ident])}
    else:r={'status':'ID_UNRESOLVED'}
   else:raise ValueError('Unknown origin '+str(origin))
   o['v6_source_resolution'][side]=r
   stats['side_'+r['status']]+=1;projectstats[project]['side_'+r['status']]+=1
  sides=o['v6_source_resolution'];both=all(sides[s].get('code_sha256') for s in ('left','right'))
  if both:
   pair=tuple(sorted([sides['left']['code_sha256'],sides['right']['code_sha256']]))
   exact_pair_records[(project,pair)].append(o['pair_id'])
   stats['pairs_both_code_resolved']+=1
  else:stats['pairs_not_fully_resolved']+=1;flags.add('SOURCE_IDENTITY_NOT_FULLY_RESOLVED')
  o['v6_code_pair_sha256']=digest(('\0'.join(pair)).encode()) if both else None
  o['v6_within_pair_exact_code']=bool(both and sides['left']['code_sha256']==sides['right']['code_sha256'])
  o['v6_within_pair_normalized_code']=bool(both and sides['left']['normalized_code_sha256']==sides['right']['normalized_code_sha256'])
  split_matches={}
  if both:
   flags.discard('C01_FUNCTION_CODE_LINK_UNVERIFIED')
  if training_audit['training_reference_complete']:
   flags.discard('TRAIN_CODE_HASH_LEAKAGE_NOT_ESTABLISHED')
  for split in ('train','validation'):
   exact_sides=[s for s in ('left','right') if sides[s].get('code_sha256') in train_exact[split]]
   norm_sides=[s for s in ('left','right') if sides[s].get('normalized_code_sha256') in train_normalized[split]]
   pair_exact_match=bool(both and pair in training_code_pairs[split])
   split_matches[split]={'exact_sides':exact_sides,'normalized_sides':norm_sides,'exact_code_pair_overlap':pair_exact_match}
   if pair_exact_match:stats[split+'_exact_pair_overlap']+=1
   if exact_sides:stats[split+'_exact_overlap_pairs']+=1
   if norm_sides:stats[split+'_normalized_overlap_pairs']+=1
  matching_sides=sorted(set(sum((v['exact_sides']+v['normalized_sides'] for v in split_matches.values()),[])))
  has_overlap=bool(matching_sides)
  if has_overlap:
   status='EXACT_OR_NORMALIZED_ACTIVE_TRAINING_CODE_HASH_OVERLAP'
   flags.add('ACTIVE_TRAIN_OR_VALIDATION_HASH_OVERLAP')
  elif not both:status='NOT_ESTABLISHED_UNRESOLVED_SOURCE'
  elif not training_audit['training_reference_complete']:
   status='NOT_ESTABLISHED_ACTIVE_TRAIN_ID_MAPPING_INCOMPLETE'
  elif not train_raw and not train_norm:status='NOT_ESTABLISHED_TRAIN_CODE_UNAVAILABLE'
  else:status='NO_MATCH_AGAINST_M2_M3_ACTIVE_PYTHON_TRAIN_VALID_HASHES'
  o['code_hash_train_overlap_status']=status
  o['v6_code_leakage_matches_by_split']=split_matches
  o['v6_code_leakage_matched_sides']=matching_sides
  # Never claim near-duplicate detection without an actual algorithm and thresholds.
  o['near_duplicate_leakage_status']='NOT_CHECKED'
  o['gold_label']=None;o['evaluation_eligible']=False;o['annotation_status']='pending'
  o['review_flags']=sorted(flags)
  stats['leakage_'+status]+=1;projectstats[project]['pairs']+=1
 dups={key:ids for key,ids in exact_pair_records.items() if len(ids)>1}
 for o in records:
  sides=o['v6_source_resolution'];both=bool(o['v6_code_pair_sha256'])
  if both:
   pair=tuple(sorted([sides['left']['code_sha256'],sides['right']['code_sha256']]))
   o['v6_exact_code_pair_duplicate_count']=len(exact_pair_records[(o['project'],pair)])
   if o['v6_exact_code_pair_duplicate_count']>1:o['review_flags'].append('EXACT_CODE_PAIR_DUPLICATE')
  else:o['v6_exact_code_pair_duplicate_count']=None
 auditout={'version':VERSION,'status':STATUS,'created_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
           'input_v4':{'candidate_path':relative(candfile,root),'candidate_sha256':checksum(candfile),
                      'audit_path':relative(auditfile,root),'audit_sha256':checksum(auditfile),
                      'manifest_path':relative(manifestfile,root),'manifest_sha256':checksum(manifestfile)},
           'active_repositories':sorted(active),'active_commits':commits,
           'active_training_audit':training_audit,
           'training_unique_available_exact_code_hashes':len(train_raw),
           'training_unique_available_normalized_code_hashes':len(train_norm),
           'training_reference_complete':training_audit['training_reference_complete'],
           'evaluation_code_sha256':checksum(codefile),'counts':dict(stats),
           'project_counts':{k:dict(v) for k,v in sorted(projectstats.items())},
           'exact_code_pair_duplicate_groups':len(dups),
           'exact_code_pair_duplicate_records':sum(len(v) for v in dups.values()),
           'limitations':['Strict AST name+start/end matching can leave valid locators unresolved; no fuzzy matching.',
                          'Source code not copied into output; see original repository file/line locators.',
                          'Training hashes derived from active balanced_multilingual split IDs joined to active clean corpus idx/func only; historical M1 training exposures may require separate evidence.',
                          'Training-vs-test near-duplicate checking not performed.',
                          'No independently verified positive/negative labels; detector output and legacy prediction are not gold ground truth.']}
 if args.dry_run:
  print(json.dumps({'status':'DRY_RUN','candidate_records':len(records),'counts':dict(stats),'duplicate_groups':len(dups),'training_hashes_available':len(train_raw),'training_reference_complete':training_audit['training_reference_complete']},sort_keys=True));return
 dest.parent.mkdir(parents=True,exist_ok=True);dest.mkdir(exist_ok=False)
 # This destination is an output snapshot, not ground_truth/python replacement.
 output=dest/'candidates_source_audited.jsonl'
 with output.open('x',encoding='utf-8') as f:
  for o in records:f.write(json.dumps(o,ensure_ascii=False,sort_keys=True)+'\n')
 with (dest/'annotation_queue.csv').open('x',newline='',encoding='utf-8') as f:
  keys=['pair_id','project','origin','left_source_status','right_source_status','left_repository_file','right_repository_file','code_leakage_status','exact_pair_duplicate_count','gold_label','annotation_status','review_flags']
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for o in records:
   r=o['v6_source_resolution']
   w.writerow({'pair_id':o['pair_id'],'project':o['project'],'origin':o['origin'],
     'left_source_status':r['left']['status'],'right_source_status':r['right']['status'],
     'left_repository_file':r['left'].get('repository_file',''),'right_repository_file':r['right'].get('repository_file',''),
     'code_leakage_status':o['code_hash_train_overlap_status'],
     'exact_pair_duplicate_count':o['v6_exact_code_pair_duplicate_count'],
     'gold_label':'','annotation_status':'pending','review_flags':';'.join(o['review_flags'])})
 with (dest/'source_leakage_audit.json').open('x',encoding='utf-8') as f:json.dump(auditout,f,indent=2,sort_keys=True);f.write('\n')
 files={p.name:{'sha256':checksum(p),'bytes':p.stat().st_size} for p in dest.iterdir() if p.is_file()}
 with (dest/'manifest.json').open('x',encoding='utf-8') as f:json.dump({'version':VERSION,'status':STATUS,'files':files,'created_at_chicago':auditout['created_at_chicago']},f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({'status':STATUS,'candidate_records':len(records),'counts':dict(stats),
                   'duplicate_groups':len(dups),'training_hashes_available':len(train_raw),'training_reference_complete':training_audit['training_reference_complete'],'out':str(dest)},sort_keys=True))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',type=Path,default=Path.cwd())
 p.add_argument('--source',type=Path,default=None,help='D-05b v4 ground truth candidate directory, default data/evaluation/ground_truth/python')
 p.add_argument('--output',type=Path,default=None,help='Exclusive v6 result directory')
 p.add_argument('--dry-run',action='store_true')
 a=p.parse_args();a.root=a.root.resolve()
 a.source=a.source or a.root/'data/evaluation/ground_truth/python'
 a.output=a.output or a.root/'results/d-05b'/('python-source-leakage-audit-v6-'+datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f'))
 try:run(a)
 except (ValueError,OSError,KeyError,TypeError,json.JSONDecodeError) as e:
  print('D-05ba v6 ERROR:',str(e),file=sys.stderr);return 2
 return 0
if __name__=='__main__':sys.exit(main())
