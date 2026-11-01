#!/usr/bin/env python3
"""Read-only legacy-source identity audit for B-06 subject repositories."""
import argparse
import csv
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys
from urllib.parse import urlparse

# Deliberately independent of acquisition script. No code execution of legacy files.
PROJECTS = [
 ('azure-sdk-for-python','Azure/azure-sdk-for-python'), ('salt','saltstack/salt'),
 ('keystone','openstack/keystone'), ('sympy','sympy/sympy'),
 ('core','home-assistant/core'), ('werkzeug','pallets/werkzeug'),
 ('web2py','web2py/web2py'), ('horizon','openstack/horizon'),
 ('ansible','ansible/ansible'), ('libcloud','apache/libcloud'),
 ('Python','TheAlgorithms/Python'), ('neutron','openstack/neutron'),
 ('ironic','openstack/ironic'), ('keras','keras-team/keras'),
 ('paramiko','paramiko/paramiko'), ('sunnypilot','sunnypilot/sunnypilot'),
 ('gensim','piskvorky/gensim'), ('nova','openstack/nova'),
 ('h2ogpt','h2oai/h2ogpt'), ('matplotlib','matplotlib/matplotlib'),
 ('pandas','pandas-dev/pandas'), ('boto3','boto/boto3'),
 ('django','django/django'), ('swift','openstack/swift'),
]
META_NAMES = {'pyproject.toml','setup.py','setup.cfg','PKG-INFO','README.md','README.rst','README.txt',
              '.readthedocs.yaml','tox.ini','CITATION.cff','package.json'}
GITHUB_RE = re.compile(r'(?:https?://|git@)github\.com[:/]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)',re.I)

def metadata_urls(folder):
    found=[]
    candidates=[]
    for p in folder.iterdir():
        if p.is_file() and (p.name in META_NAMES or p.name.lower().startswith('readme')):
            candidates.append(p)
    for sub in ('homeassistant','src'):
        d=folder/sub
        if d.is_dir():
            for name in ('setup.py','pyproject.toml','setup.cfg','PKG-INFO'):
                p=d/name
                if p.is_file(): candidates.append(p)
    for p in candidates:
        try:
            data=p.read_text(encoding='utf-8',errors='replace')[:180000]
        except OSError: continue
        for m in GITHUB_RE.finditer(data):
            slug=m.group(1)+'/'+m.group(2).removesuffix('.git')
            if slug.lower().endswith(('.png','.jpg','.svg')): continue
            found.append((slug,str(p.relative_to(folder))))
    unique={}
    for slug,path in found: unique.setdefault(slug,path)
    return list(unique.items())[:30]

def repo_dirs(folder):
    dirs=sorted(p.name for p in folder.iterdir() if p.is_dir() and not p.is_symlink())
    return dirs[:40]

def remote_check(slug, timeout=20):
    try:
        r=subprocess.run(['git','ls-remote','--exit-code','--heads',f'https://github.com/{slug}.git'],
                         text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
        if r.returncode==0:return 'EXISTS'
        return 'UNREACHABLE_OR_INVALID'
    except (FileNotFoundError,subprocess.TimeoutExpired):return 'CHECK_UNAVAILABLE'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[2])
    p.add_argument('--legacy-systems',type=pathlib.Path,default=pathlib.Path('/home/user1-system11/research_dream/2026JournalData/RQ3/python/code/Dataset-Preparation/detect_clones/NiCad-py-block/systems'))
    p.add_argument('--check-remotes',action='store_true',help='git ls-remote; network check, not identity proof')
    p.add_argument('--show-dirs',type=int,default=12)
    a=p.parse_args()
    if a.show_dirs < 0: p.error('--show-dirs must be >= 0')
    root=a.root.resolve(); legacy=a.legacy_systems.resolve()
    if not legacy.is_dir():p.error('Legacy systems not found: '+str(legacy))
    out=root/'results'/'b-06'/('identity-audit-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S-%f'))
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    print('B-06 v3 | Legacy Source Repository Identity Audit | READ ONLY',flush=True)
    print('LEGACY:',legacy,'\n')
    for name,slug in PROJECTS:
        path=legacy/name
        row={'project':name,'github_slug':slug,'legacy_exists':path.is_dir(),'legacy_dirs':'',
             'metadata_github_urls':'','metadata_match':'','remote_status':'NOT_CHECKED','review_status':''}
        if not path.is_dir():
            row['review_status']='MISSING_LEGACY'
        else:
            dirs=repo_dirs(path)
            urls=metadata_urls(path)
            row['legacy_dirs']='; '.join(dirs)
            row['metadata_github_urls']='; '.join(f'{s} [{source}]' for s,source in urls)
            direct=any(s.casefold()==slug.casefold() for s,_ in urls)
            row['metadata_match']='YES' if direct else ('OTHER_URLS' if urls else 'NO_URLS')
            # Structural fingerprint is evidence of likely identity, not proof of exact historical commit.
            if direct:row['review_status']='METADATA_MATCH_REVIEW'
            elif name=='core' and {'homeassistant','pylint','script'}.issubset(dirs):row['review_status']='STRUCTURE_MATCH_REVIEW'
            elif name=='Python' and {'backtracking','bit_manipulation','data_structures'}.issubset(dirs):row['review_status']='STRUCTURE_MATCH_REVIEW'
            else:row['review_status']='NEEDS_MANUAL_REVIEW'
        if a.check_remotes: row['remote_status']=remote_check(slug)
        rows.append(row)
        print(f'{name:24} {row["review_status"]:25}  {slug}')
        if row['legacy_dirs']:print('  folders:',', '.join(row['legacy_dirs'].split('; ')[:a.show_dirs]))
        if row['metadata_github_urls']:print('  github refs:',row['metadata_github_urls'][:380])
        print('')
    fields=list(rows[0]);csv_path=out/'repository_identity_audit.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps({'experiment':'b-06','version':'v3','mode':'identity-audit',
        'legacy_systems':str(legacy),'total':len(rows),
        'status_counts':{s:sum(r['review_status']==s for r in rows) for s in sorted({r['review_status'] for r in rows})},
        'results':str(out)},indent=2)+'\n')
    print('OUTPUT_DIR='+str(out))
    print('CAUTION: metadata and directory matches do not establish historical commit equivalence.')
    print('CAUTION: even a valid GitHub URL is not identity proof if it is merely a dependency/reference.')
    return 0
if __name__=='__main__':sys.exit(main())
