#!/usr/bin/env python3
"""C-01 v5: non-destructive per-project NiCad + jscpd hybrid detection.

NiCad and per-file probes are serial and isolated via unique C-01 cache basename.
"""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent

def module(name, filename):
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

policy=module('b07policy','c-01-source-policy.py')
probe=module('b07probe','c-01-parse-probe.py')
classify=module('b07classify','c-01-parse-classification.py')

EXPECTED={'salt':'0fa7bec68485f8d89b7de2bd6b8743d040fb956b'}
REFERENCE={'staged_files':1006,'ni_parse_failed_files':282,'nicad_primary_unique_pairs':595,'jscpd_strict_fallback_unique_pairs':535,'hybrid_unique_pairs':1130,'jscpd_raw_duplicates':3250,'ni_failed_files_in_strict_fallback':144}
FIELDS=classify.FIELDS

def run(cmd, log_path, cwd=None, env=None, timeout=None):
    print('RUN:', ' '.join(map(str,cmd)),flush=True)
    with log_path.open('w',encoding='utf-8') as h:
        p=subprocess.Popen(list(map(str,cmd)),cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1)
        try:
            for line in p.stdout:
                h.write(line)
                if re.search(r'failed|error|extracted|clone|warning|total|report',line,re.I):print(line.rstrip()[:350],flush=True)
            rc=p.wait(timeout=timeout)
        finally:
            if p.poll() is None:p.kill();p.wait()
    if rc:raise RuntimeError(f'Exit {rc} from {cmd[0]}; see {log_path}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',required=True,help='Directory name within data/repositories')
    p.add_argument('--root',type=Path,default=HERE.parents[1])
    p.add_argument('--expected-commit',default='',help='Optional fixed 40-hex SHA')
    p.add_argument('--compare',type=Path,help='Completed B-07 project output (comparison only after detection)')
    p.add_argument('--out',type=Path)
    p.add_argument('--per-file-timeout',type=int,default=100)
    p.add_argument('--max-files',type=int,default=0,help='pilot only; not comparable')
    a=p.parse_args();root=a.root.resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.project):p.error('Invalid project')
    repo=root/'data/repositories'/a.project
    if not (repo/'.git').exists():p.error('Git repository missing: '+str(repo))
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if a.expected_commit and head!=a.expected_commit:p.error('Unexpected Git commit: '+head)
    remote=subprocess.check_output(['git','-C',str(repo),'remote','get-url','origin'],text=True).strip()
    status=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    if status:p.error('Dirty Git repository; refuses to run')
    stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
    out=(a.out if a.out and a.out.is_absolute() else root/(a.out or Path('results/c-01')/stamp)).resolve()
    if out.exists():p.error('Output exists: '+str(out))
    out.mkdir(parents=True,exist_ok=False)
    stage=out/'stage'/('c01'+stamp.replace('-',''))
    stage.mkdir(parents=True)
    stats=Counter();audit=[]
    for src in sorted(repo.rglob('*.py')):
        rel=src.relative_to(repo);stats['files_seen']+=1
        if policy.test_file(rel):stats['files_excluded_by_path']+=1;continue
        try:
            source=src.read_text(encoding='utf-8');tree=ast.parse(source)
        except (SyntaxError,UnicodeError,ValueError) as exc:
            audit.append({'file':rel.as_posix(),'action':'skip_parse_error','detail':str(exc)[:250]});stats['files_excluded_parse_error']+=1;continue
        lines=source.splitlines(keepends=True);spans=policy.masking_spans(tree,lines)
        for start,end in spans:
            if start>=len(lines):continue
            indent=re.match(r'\s*',lines[start]).group().replace('\n','')
            lines[start]=indent+'pass\n'
            for i in range(start+1,min(end,len(lines))):lines[i]='\n'
        output=''.join(lines)
        try:ast.parse(output)
        except SyntaxError as exc:
            audit.append({'file':rel.as_posix(),'action':'skip_after_mask','detail':str(exc)[:250]});stats['files_excluded_after_mask_error']+=1;continue
        dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(output,encoding='utf-8')
        stats['files_staged']+=1;stats['test_declarations_masked']+=len(spans)
        if a.max_files and stats['files_staged']>=a.max_files:break
    with (out/'filter_audit.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=['file','action','detail']);w.writeheader();w.writerows(audit)
    (out/'filter_summary.json').write_text(json.dumps({'counts':dict(stats),'commit':head,'stage':str(stage)},indent=2)+'\n')
    print('Staged files:',stats['files_staged'],flush=True)
    nicad=root/'tools/nicad-7.0.1-linux-x86_64';txl=root/'tools/opentxl-11.3.7-linux-x64'
    if not (nicad/'bin/nicad').is_file() or not (txl/'bin/txl').is_file():p.error('NiCad/OpenTxl installation missing')
    env=os.environ.copy();env['PATH']=str(txl/'bin')+os.pathsep+env.get('PATH','')
    cache=nicad/'nicadclones'/stage.name
    if cache.exists() or cache.is_symlink():p.error('NiCad unique cache unexpectedly already exists: '+str(cache))
    # NiCad is not safe to run concurrently against same installation. Shell wrapper takes flock.
    run([nicad/'bin/nicad','functions','py',stage,'default'],out/'nicad.log',cwd=nicad,env=env)
    xmls=sorted(cache.rglob('*functions-blind-clones-0.30.xml'))
    if len(xmls)!=1:raise RuntimeError('Expected exactly one pair-level NiCad XML under '+str(cache)+', got '+str(xmls))
    xml_dir=out/'nicad_xml';xml_dir.mkdir();xml=xml_dir/xmls[0].name;shutil.copy2(xmls[0],xml)
    # Per-file probe uses the exact B-02 parsing logic, but isolates its cache and input.
    probe_root=out/'probe';probe_root.mkdir();tmp=probe_root/'c01_probe';pcache=nicad/'nicadclones'/'c01_probe'
    if pcache.exists() or pcache.is_symlink():raise RuntimeError('c01_probe cache already exists; another C-01 run may be active')
    if pcache.exists() or tmp.exists():raise RuntimeError('C-01 probe cache already exists; refusing to remove preexisting data')
    data=[];counts=Counter()
    try:
        pyfiles=sorted(stage.rglob('*.py'))
        for i,src in enumerate(pyfiles,1):
            rel=src.relative_to(stage).as_posix();raw=src.read_bytes()
            py_status='pass';py_error=''
            try:ast.parse(raw,filename=rel)
            except (SyntaxError,ValueError,UnicodeError) as exc:py_status='fail';py_error=str(exc)[:300]
            ns,nfail,nextract,nrc,msg,secs=probe.single_probe(src,tmp,pcache,nicad,env,a.per_file_timeout)
            row={'file':rel,'python_ast':py_status,'python_error':py_error,'nicad_status':ns,'nicad_failed_count':nfail,'nicad_extracted_functions':nextract,'nicad_exit':nrc,'nicad_error_excerpt':msg,'size_bytes':len(raw)}
            row['nicad_category']=classify.classify(row);counts[row['nicad_category']]+=1;data.append(row)
            if i%25==0 or i==len(pyfiles):print(f'PROBE {i}/{len(pyfiles)} failures={counts["NICAD_PARSE_FAILURE"]}',flush=True)
    finally:
        if pcache.is_dir() and not pcache.is_symlink():shutil.rmtree(pcache)
        if tmp.is_dir() and not tmp.is_symlink():shutil.rmtree(tmp)
    audit_csv=out/'file_parse_audit.csv'
    with audit_csv.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=FIELDS,extrasaction='ignore');w.writeheader();w.writerows(data)
    (out/'parse_categories.json').write_text(json.dumps(dict(counts),indent=2)+'\n')
    jsout=out/'jscpd';jsout.mkdir()
    run(['npx','--yes','jscpd@5.4.2',stage,'--format','python','--similarity','0.70','--min-lines','10','--min-tokens','50','--reporters','json','--output',jsout],out/'jscpd.log',cwd=root)
    jsfile=jsout/'jscpd-report.json'
    if not jsfile.is_file():raise RuntimeError('Missing jscpd JSON report')
    merged=out/'hybrid'
    run([sys.executable,HERE/'c-01-hybrid-merge.py','--project-root',root,'--stage',stage,'--ni-file',xml,'--audit-file',audit_csv,'--js-file',jsfile,'--output',merged],out/'merge.log',cwd=root)
    summary=json.loads((merged/'summary.json').read_text())
    rawjs=len(json.loads(jsfile.read_text()).get('duplicates',[]))
    metrics={k:summary.get(k) for k in ('staged_files','functions_indexed','python_ast_errors',
        'ni_parse_failed_files','nicad_raw_reports','nicad_primary_unique_pairs',
        'jscpd_ast_similar_raw_reports','jscpd_strict_fallback_unique_pairs','hybrid_unique_pairs',
        'ni_failed_files_in_strict_fallback')}
    metrics['jscpd_raw_duplicates']=rawjs
    done={'status':'COMPLETED','experiment':'C-01','version':'v5','project':a.project,
          'commit':head,'remote':remote,'repository_path':str(repo),
          'source':str(repo)+' (raw git checkout, staged independently)',
          'settings':{'nicad':'functions py default; blind renaming UPI 0.30',
                      'jscpd':'5.4.2; python; similarity 0.70; min-lines 10; min-tokens 50; json',
                      'probe':'per-file NiCad functions py default'},'metrics':metrics,
          'completed_at':datetime.now(ZoneInfo('America/Chicago')).isoformat()}
    (out/'project_run.json').write_text(json.dumps(done,indent=2)+'\n')
    if a.compare:
        compare_path=a.compare if a.compare.is_absolute() else root/a.compare
        run([sys.executable,HERE/'c-01-compare.py','--c01',out,'--b07',compare_path],out/'comparison.log',cwd=root)
    print('C-01 DONE:',out,'HYBRID_PAIRS=',metrics['hybrid_unique_pairs'])

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('C-01 FAILED:',repr(exc),file=sys.stderr)
        raise
