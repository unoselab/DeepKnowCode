#!/usr/bin/env python3
"""C-01 v7 sequential independent detector orchestration (no B-07 input)."""
import argparse,csv,json,re,subprocess,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

EXCLUDED={'azure-sdk-for-python'}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',type=Path,required=True)
 p.add_argument('--projects',nargs='*',help='Explicit project names. Default: all active Git repositories except scikit-learn.')
 p.add_argument('--skip-completed',action='store_true',default=True)
 p.add_argument('--include-scikit-learn',action='store_true')
 p.add_argument('--dry-run',action='store_true')
 a=p.parse_args(); root=a.root.resolve(); repos=root/'data/repositories'
 if not repos.is_dir():p.error('Missing repositories: '+str(repos))
 available={x.name:x for x in repos.iterdir() if x.is_dir() and (x/'.git').exists()}
 if not available:p.error('No Git repositories')
 selected=a.projects if a.projects else sorted(set(available)-EXCLUDED-({} if a.include_scikit_learn else {'scikit-learn'}))
 if len(selected)!=len(set(selected)):p.error('Duplicate project names')
 for s in selected:
  if s not in available or s in EXCLUDED:p.error('Missing or excluded project: '+s)
 timestamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
 batch=root/'results/c-01-batch'/timestamp
 history={}
 for file in sorted((root/'results/c-01').glob('*/project_run.json')):
  try:
   obj=json.loads(file.read_text());k=obj.get('project');
   if obj.get('status')=='COMPLETED' and k:history.setdefault(k,[]).append((obj.get('commit'),str(file.parent)))
  except (ValueError,OSError):pass
 plan=[]
 for name in selected:
  sha=subprocess.check_output(['git','-C',str(available[name]),'rev-parse','HEAD'],text=True).strip()
  if not re.fullmatch(r'[0-9a-f]{40}',sha):p.error('Bad SHA for '+name)
  completed=[run for commit,run in history.get(name,[]) if commit==sha]
  plan.append({'project':name,'commit':sha,'action':'SKIP_COMPLETED' if completed else 'RUN','existing':completed[-1] if completed else None})
 print(json.dumps({'dry_run':a.dry_run,'selected':len(selected),'plan':plan},indent=2),flush=True)
 if a.dry_run:return 0
 batch.mkdir(parents=True)
 with (batch/'plan.json').open('w') as f:json.dump(plan,f,indent=2)
 results=[]
 for item in plan:
  name=item['project'];out=batch/name
  if item['action']=='SKIP_COMPLETED':results.append(dict(item,status='SKIPPED_COMPLETED'));continue
  cmd=[sys.executable,'-u',str(root/'src/py/c-01-clone-detection.py'),'--root',str(root),'--project',name,'--expected-commit',item['commit']]
  log=batch/(name+'.log')
  print('START',name,flush=True)
  with log.open('w') as f:proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=root)
  print('END',name,'rc',proc.returncode,'log',log,flush=True)
  results.append(dict(item,status='COMPLETED_COMMAND' if proc.returncode==0 else 'FAILED',exit_code=proc.returncode,log=str(log)))
  (batch/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
 (batch/'summary.json').write_text(json.dumps({'version':'v7','results':results},indent=2)+'\n')
 return 1 if any(r['status']=='FAILED' for r in results) else 0
if __name__=='__main__':sys.exit(main())
