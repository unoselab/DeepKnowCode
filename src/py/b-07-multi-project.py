#!/usr/bin/env python3
"""B-07 v4 project orchestrator: isolate, audit and resume completed projects."""
import argparse,csv,json,subprocess,sys,hashlib,os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
SETTINGS={'nicad':'functions py default','jscpd':'5.4.2; python; similarity 0.70; min-lines 10; min-tokens 50; json','probe':'per-file NiCad functions py default'}
def stamp():return datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
def load_manifest(path):
    with path.open(newline='',encoding='utf-8') as h:rows=list(csv.DictReader(h))
    if not rows or any(not r.get('project') for r in rows):raise ValueError('Manifest has missing project')
    if len(set(r['project'] for r in rows))!=len(rows):raise ValueError('Duplicate project in manifest')
    return {r['project']:r for r in rows}
def existing_success(root,project,record):
    basedir=root/'results/b-07'
    if not basedir.is_dir():return None
    for p in sorted(basedir.glob('*/projects/*/project_run.json'),reverse=True):
        if p.parent.name!=project:continue
        try:
            d=json.loads(p.read_text())
            if all([d.get('status')=='COMPLETED',d.get('project')==project,d.get('commit')==record['commit'],d.get('remote','').rstrip('/').removesuffix('.git').lower()==record['remote'].rstrip('/').removesuffix('.git').lower(),d.get('version') in ('v4','v8','v9'),d.get('settings')==SETTINGS,(p.parent/'hybrid/hybrid_function_clone_pairs.csv').is_file(),(p.parent/'hybrid/summary.json').is_file(),(p.parent/'jscpd/jscpd-report.json').is_file(),(p.parent/'file_parse_audit.csv').is_file()]):return str(p.parent)
        except (OSError,ValueError,KeyError):pass
    return None
def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--project',help='One project')
    g.add_argument('--include',help='Comma-separated projects')
    g.add_argument('--all',action='store_true',help='All 24 manifest subjects')
    p.add_argument('--root',type=Path,default=HERE.parents[1])
    p.add_argument('--manifest',type=Path,default=Path('results/b-06/20261101-161810-903689/repository_manifest.csv'))
    p.add_argument('--skip-completed',action='store_true')
    p.add_argument('--dry-run',action='store_true',help='Validate manifest selection; do not run detectors or create results')
    p.add_argument('--per-file-timeout',type=int,default=100)
    a=p.parse_args();root=a.root.resolve();manifest=a.manifest if a.manifest.is_absolute() else root/a.manifest
    if not manifest.is_file():p.error('Missing B-06 manifest: '+str(manifest))
    try:records=load_manifest(manifest)
    except ValueError as e:p.error(str(e))
    names=list(records) if a.all else ([a.project] if a.project else [v.strip() for v in a.include.split(',') if v.strip()])
    if not names or len(set(names))!=len(names):p.error('No projects or duplicated project selection')
    for name in names:
        if name not in records:p.error('Unknown project: '+name)
        m=records[name]
        if not m.get('commit') or not m.get('remote'):p.error('Project '+name+' has missing commit or remote')
        if m.get('status') not in ('CLONED','SKIPPED_EXISTING','UPDATED_EXISTING'):
            p.error(f'Project {name} has unusable manifest status {m.get("status")}')
    # Salt reference is checked by the worker; other projects use their own manifest commits.
    planned=[]
    for name in names:
        previous=existing_success(root,name,records[name]) if a.skip_completed else None
        planned.append({'project':name,'commit':records[name]['commit'],'action':'SKIP_COMPLETED' if previous else 'RUN','previous_result':previous})
    print(json.dumps({'manifest':str(manifest),'dry_run':a.dry_run,'projects':planned},indent=2),flush=True)
    if a.dry_run:return
    out=root/'results/b-07'/stamp()
    if out.exists():p.error('Run output already exists: '+str(out))
    (out/'projects').mkdir(parents=True)
    (out/'plan.json').write_text(json.dumps({'manifest':str(manifest),'settings':SETTINGS,'projects':planned},indent=2)+'\n')
    results=[]
    try:
        for item in planned:
            name=item['project'];dest=out/'projects'/name
            if item['action']=='SKIP_COMPLETED':
                results.append({**item,'status':'SKIPPED_COMPLETED'});continue
            cmd=[sys.executable,'-u',str(HERE/'b-07-hybrid-python-clone-detection.py'),'--root',str(root),'--manifest',str(manifest),'--project',name,'--out',str(dest),'--per-file-timeout',str(a.per_file_timeout)]
            log=out/(name+'.log')
            print('PROJECT START:',name,flush=True)
            with log.open('w',encoding='utf-8') as h:
                proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1)
                for line in proc.stdout:
                    h.write(line);print(line,end='',flush=True)
                rc=proc.wait()
            if rc==0 and (dest/'project_run.json').is_file():
                d=json.loads((dest/'project_run.json').read_text());results.append({**item,'status':'COMPLETED','metrics':d['metrics'],'result':str(dest)})
            else:
                results.append({**item,'status':'FAILED','exit_code':rc,'log':str(log)})
            (out/'run_summary.json').write_text(json.dumps({'results':results},indent=2)+'\n')
    finally:
        (out/'run_summary.json').write_text(json.dumps({'manifest':str(manifest),'results':results},indent=2)+'\n')
        fields=['project','status','commit','nicad_primary_unique_pairs','jscpd_strict_fallback_unique_pairs','hybrid_unique_pairs','result','previous_result','log']
        with (out/'project_summary.csv').open('w',newline='',encoding='utf-8') as h:
            w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
            for r in results:
                row={k:r.get(k,'') for k in fields};row.update({k:r.get('metrics',{}).get(k,'') for k in fields if k in ('nicad_primary_unique_pairs','jscpd_strict_fallback_unique_pairs','hybrid_unique_pairs')});w.writerow(row)
        print('B-07 V9 SUMMARY:',out,flush=True)
    if any(r['status']=='FAILED' for r in results):sys.exit(1)
if __name__=='__main__':main()
