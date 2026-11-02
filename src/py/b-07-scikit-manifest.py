#!/usr/bin/env python3
"""Create a versioned 24-subject manifest without modifying the B-06 baseline."""
import argparse,csv,subprocess,sys,hashlib,os
from pathlib import Path

SCIKIT_SHA='4dc45d9a0afa48d741fe5924f167a236d8605210'
SCIKIT_REMOTE='https://github.com/scikit-learn/scikit-learn.git'
DEFAULT_BASE='results/b-06/20261101-161810-903689/repository_manifest.csv'
DEFAULT_OUT='results/b-07/subject-manifest-v9.csv'

def normalized(s):return s.rstrip('/').removesuffix('.git').lower()
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
 ap.add_argument('--baseline',default=DEFAULT_BASE)
 ap.add_argument('--output',default=DEFAULT_OUT)
 a=ap.parse_args(); root=a.root.resolve()
 baseline=root/a.baseline; output=root/a.output
 if not baseline.is_file():ap.error('Baseline manifest missing: '+str(baseline))
 if baseline.resolve()==output.resolve():ap.error('Refusing to overwrite baseline manifest')
 with baseline.open(newline='',encoding='utf-8') as f:
  reader=csv.DictReader(f);fields=list(reader.fieldnames or []);rows=list(reader)
 required={'project','github_slug','url','mapping_status','status','commit','branch','remote','error'}
 if not required.issubset(fields):ap.error('Unexpected manifest columns: '+str(fields))
 if len(rows)!=24 or len(set(r['project'] for r in rows))!=24:ap.error('Expected 24 distinct original subjects')
 matching=[r for r in rows if r['project']=='azure-sdk-for-python']
 if len(matching)!=1 or any(r['project']=='scikit-learn' for r in rows):ap.error('Unexpected original subject records')
 repo=root/'data/repositories/scikit-learn'
 if not (repo/'.git').exists():ap.error('scikit-learn missing: '+str(repo))
 def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
 head=git('rev-parse','HEAD');remote=git('remote','get-url','origin')
 if head!=SCIKIT_SHA:ap.error('scikit-learn SHA mismatch: '+head)
 if normalized(remote)!=normalized(SCIKIT_REMOTE):ap.error('scikit-learn remote mismatch: '+remote)
 if git('status','--porcelain'):ap.error('Dirty scikit-learn checkout')
 if (root/'data/repositories/azure-sdk-for-python').exists():ap.error('Azure SDK still present in active repositories; verify your exclusion layout')
 replacement={'project':'scikit-learn','github_slug':'scikit-learn/scikit-learn','url':SCIKIT_REMOTE,'mapping_status':'replacement','status':'CLONED','commit':SCIKIT_SHA,'branch':git('branch','--show-current'),'remote':remote,'error':''}
 updated=[replacement if r['project']=='azure-sdk-for-python' else r for r in rows]
 if len(updated)!=24 or len({r['project'] for r in updated})!=24:ap.error('Final manifest cardinality mismatch')
 lines=[]
 import io
 buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(updated)
 content=buf.getvalue()
 output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists():
  if output.read_text(encoding='utf-8')!=content:ap.error('Versioned manifest already exists with different content; refusing overwrite')
  print('REUSED',output)
 else:
  with output.open('x',encoding='utf-8',newline='') as f:f.write(content)
  print('CREATED',output)
 print('ACTIVE_SUBJECTS',len(updated),'SCIKIT_SHA',head,'MANIFEST_SHA256',hashlib.sha256(content.encode()).hexdigest())
 print('EXCLUDED',str(root/'data/repositories-excluded/azure-sdk-for-python'))
if __name__=='__main__':main()
