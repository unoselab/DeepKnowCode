#!/usr/bin/env python3
"""B-06: acquire Python RQ3 subject repositories; never modify prior evaluation datasets."""
import argparse
import csv
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys

# Candidates identified by repository name, not historically validated provenance.
PROJECTS = [
    ('azure-sdk-for-python', 'Azure/azure-sdk-for-python', 'candidate'),
    ('salt', 'saltstack/salt', 'known'),
    ('keystone', 'openstack/keystone', 'candidate'),
    ('sympy', 'sympy/sympy', 'candidate'),
    ('core', 'home-assistant/core', 'legacy_structure_matched'),
    ('werkzeug', 'pallets/werkzeug', 'candidate'),
    ('web2py', 'web2py/web2py', 'candidate'),
    ('horizon', 'openstack/horizon', 'candidate'),
    ('ansible', 'ansible/ansible', 'candidate'),
    ('libcloud', 'apache/libcloud', 'candidate'),
    ('Python', 'TheAlgorithms/Python', 'legacy_structure_matched'),
    ('neutron', 'openstack/neutron', 'candidate'),
    ('ironic', 'openstack/ironic', 'candidate'),
    ('keras', 'keras-team/keras', 'candidate'),
    ('paramiko', 'paramiko/paramiko', 'candidate'),
    ('sunnypilot', 'sunnypilot/sunnypilot', 'candidate'),
    ('gensim', 'piskvorky/gensim', 'candidate'),
    ('nova', 'openstack/nova', 'candidate'),
    ('h2ogpt', 'h2oai/h2ogpt', 'candidate'),
    ('matplotlib', 'matplotlib/matplotlib', 'candidate'),
    ('pandas', 'pandas-dev/pandas', 'candidate'),
    ('boto3', 'boto/boto3', 'candidate'),
    ('django', 'django/django', 'candidate'),
    ('swift', 'openstack/swift', 'candidate'),
]

def run(argv, timeout=180):
    return subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)

def git_output(repo, *args):
    x = run(['git','-C',str(repo),*args], timeout=30)
    return x.stdout.strip() if x.returncode == 0 else ''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2])
    p.add_argument('--include', action='append', default=[], help='project names; repeatable')
    p.add_argument('--execute', action='store_true', help='actually clone; default dry-run')
    p.add_argument('--update-existing', action='store_true', help='fetch existing repos (never reset/checkout)')
    p.add_argument('--depth', type=int, default=1)
    p.add_argument('--override', action='append', default=[], metavar='NAME=OWNER/REPO')
    a=p.parse_args()
    if a.depth < 1: p.error('--depth must be positive')
    overrides={}
    for entry in a.override:
        if '=' not in entry: p.error('override must be NAME=OWNER/REPO')
        key,val=entry.split('=',1)
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',val): p.error('override must use OWNER/REPO')
        overrides[key]=val
    root=a.root.resolve(); repos=root/'data'/'repositories'
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    out=root/'results'/'b-06'/stamp
    out.mkdir(parents=True, exist_ok=False)
    chosen=set(a.include)
    known={n for n,_,_ in PROJECTS}
    if chosen-known: p.error('unknown projects: '+','.join(sorted(chosen-known)))
    rows=[]
    print('B-06 Python Subject Repository Acquisition | '+('EXECUTE' if a.execute else 'DRY RUN'))
    print('Repository root:',repos)
    for name, slug, verification in PROJECTS:
        if chosen and name not in chosen: continue
        slug=overrides.get(name,slug)
        if name in overrides: verification='user_override'
        target=repos/name
        row={'project':name,'github_slug':slug,'url':f'https://github.com/{slug}.git' if slug else '',
             'mapping_status':verification,'status':'','commit':'','branch':'','remote':'','error':''}
        if not slug:
            row['status']='UNRESOLVED'
        elif target.exists():
            if not (target/'.git').exists():
                row['status']='BLOCKED_NON_GIT_DIRECTORY'
            else:
                row['remote']=git_output(target,'remote','get-url','origin')
                row['commit']=git_output(target,'rev-parse','HEAD')
                row['branch']=git_output(target,'symbolic-ref','--short','HEAD')
                # Do not fetch/modify when existing origin mismatches the candidate mapping.
                if row['remote'].rstrip('/').removesuffix('.git').lower() != row['url'].rstrip('/').removesuffix('.git').lower():
                    row['status']='REVIEW_EXISTING_REMOTE'
                elif a.execute and a.update_existing:
                    x=run(['git','-C',str(target),'fetch','--depth',str(a.depth),'origin'],timeout=600)
                    row['status']='FETCHED_ONLY' if x.returncode==0 else 'FETCH_FAILED'
                    row['error']=x.stderr[-400:] if x.returncode else ''
                else:
                    row['status']='SKIPPED_EXISTING'
        elif not a.execute:
            row['status']='WOULD_CLONE'
        else:
            repos.mkdir(parents=True,exist_ok=True)
            try:
                x=run(['git','clone','--depth',str(a.depth),'--',row['url'],str(target)],timeout=1800)
                if x.returncode:
                    row['status']='CLONE_FAILED'; row['error']=x.stderr[-600:]
                else:
                    row['status']='CLONED'; row['commit']=git_output(target,'rev-parse','HEAD'); row['branch']=git_output(target,'symbolic-ref','--short','HEAD'); row['remote']=git_output(target,'remote','get-url','origin')
            except (subprocess.TimeoutExpired,OSError) as e:
                row['status']='CLONE_FAILED'; row['error']=str(e)
        rows.append(row)
        print(f"{name:24} {row['status']:26} {row['commit'][:12]:12} {slug}")
    fields=['project','github_slug','url','mapping_status','status','commit','branch','remote','error']
    with (out/'repository_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps({'experiment':'b-06','mode':'execute' if a.execute else 'dry-run','results':str(out),'count':len(rows),'statuses':{s:sum(x['status']==s for x in rows) for s in sorted({r['status'] for r in rows})},'repos':rows},indent=2)+'\n',encoding='utf-8')
    print('OUTPUT_DIR='+str(out))
    print('Note: candidate GitHub slugs are not verified historical RQ3 origin identities.')
    return 1 if any(r['status'] in ('CLONE_FAILED','FETCH_FAILED','BLOCKED_NON_GIT_DIRECTORY') for r in rows) else 0

if __name__=='__main__': sys.exit(main())
