#!/usr/bin/env python3
"""B-02 read-only audit of NiCad parsing against B-01 staged Python sources."""
import argparse, ast, csv, json, os, re, shutil, subprocess, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

COMMIT='0fa7bec68485f8d89b7de2bd6b8743d040fb956b'
FIELDS=['file','python_ast','python_error','nicad_status','nicad_failed_count','nicad_extracted_functions','nicad_exit','nicad_error_excerpt','size_bytes']

def safe_delete(path, expected_parent):
    path=Path(path); parent=Path(expected_parent).resolve()
    if path.is_symlink() or path.parent.resolve()!=parent or path.name!='b02_probe':
        raise RuntimeError('Unsafe cache delete refused: '+str(path))
    if path.exists():shutil.rmtree(path)

def get_count(s, regex):
    matches=re.findall(regex,s,flags=re.IGNORECASE)
    return int(matches[-1]) if matches else None

def single_probe(source, tmp_in, cache, nicad_home, env, timeout):
    safe_delete(cache,nicad_home/'nicadclones')
    if tmp_in.exists():shutil.rmtree(tmp_in)
    tmp_in.mkdir(parents=True)
    shutil.copyfile(source,tmp_in/'target.py')
    cmd=[str(nicad_home/'bin/nicad'),'functions','py',str(tmp_in),'default']
    start=time.monotonic()
    try:
        p=subprocess.run(cmd,cwd=str(nicad_home),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout,errors='replace')
        output=p.stdout;rc=p.returncode
        failed=get_count(output,r'Warning:\s*(\d+)\s+source files failed to parse') or 0
        extracted=get_count(output,r'Extracted\s+(\d+)\s+functions')
        status='failed' if failed else 'passed'
        if rc!=0: status='error'
        if extracted is None and rc==0:status='unknown'
        message=' | '.join(line.strip() for line in output.splitlines() if re.search(r'warn|error|fail|parse|extract',line,re.I))[-900:]
        return status,failed,extracted,rc,message,round(time.monotonic()-start,3)
    except subprocess.TimeoutExpired:
        return 'timeout',None,None,124,'NiCad per-file probe timed out',round(time.monotonic()-start,3)
    finally:
        safe_delete(cache,nicad_home/'nicadclones')
        if tmp_in.exists():shutil.rmtree(tmp_in)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--mode',choices=['inventory','probe'],default='probe')
    p.add_argument('--max-files',type=int,default=0,help='0=all staged files')
    p.add_argument('--per-file-timeout',type=int,default=100)
    p.add_argument('--b01-result',help='B-01 result containing filter_summary.json')
    p.add_argument('--out')
    p.add_argument('--python-bin',default=sys.executable)
    args=p.parse_args()
    root=Path(args.root).resolve();repo=root/'data/repositories/salt'
    stage=root/'results/b-01/_staging/salt';nicad=root/'tools/nicad-7.0.1-linux-x86_64';txl=root/'tools/opentxl-11.3.7-linux-x64'
    if not stage.is_dir():raise SystemExit('ERROR: missing B-01 staged input '+str(stage))
    if not (nicad/'bin/nicad').is_file() or not (nicad/'lib/nicad/scripts/nicadpair.sh').exists():raise SystemExit('ERROR: NiCad installation missing')
    if not (txl/'bin/txl').exists():raise SystemExit('ERROR: OpenTxl missing')
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if commit!=COMMIT:raise SystemExit('ERROR: unexpected salt commit '+commit)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f');out=Path(args.out).resolve() if args.out else root/'results/b-02'/stamp
    out.mkdir(parents=True,exist_ok=False)
    b01=None
    if args.b01_result:
        b01=Path(args.b01_result).resolve()/'filter_summary.json'
    else:
        choices=sorted((root/'results/b-01').glob('*/filter_summary.json'))
        choices=[c for c in choices if c.parent.name!='_staging']
        b01=choices[-1] if choices else None
    baseline={}
    if b01 and b01.exists():baseline=json.loads(b01.read_text())
    files=sorted(stage.rglob('*.py'))
    if args.max_files>0:files=files[:args.max_files]
    print('B-02 | NiCad parse failure audit | Python '+sys.version.split()[0],flush=True)
    print('Repository commit:',commit,flush=True)
    print('Mode:',args.mode,'| staged files:',len(files),'| B-01 reported failures: 282',flush=True)
    env=os.environ.copy();env['PATH']=str(txl/'bin')+os.pathsep+env.get('PATH','')
    probe_base=root/'results/b-02/_probe';probe_base.mkdir(parents=True,exist_ok=True)
    tmp_in=probe_base/'b02_probe';cache=nicad/'nicadclones/b02_probe'
    if args.mode=='probe' and cache.exists():raise SystemExit('ERROR: existing B-02 probe cache; ensure no concurrent B-02 process before cleaning')
    counts=Counter();rows=[];times=[]
    try:
        for i,f in enumerate(files,1):
            rel=f.relative_to(stage).as_posix();data=f.read_bytes();status='pass';error=''
            try:ast.parse(data,filename=rel)
            except (SyntaxError,ValueError,UnicodeError) as e:status='fail';error=str(e)[:300]
            counts['python_ast_'+status]+=1
            if args.mode=='probe':
                ns,nfail,nextract,nrc,msg,sec=single_probe(f,tmp_in,cache,nicad,env,args.per_file_timeout)
                times.append(sec);counts['nicad_'+ns]+=1
            else:ns,nfail,nextract,nrc,msg='not_probed',None,None,None,''
            rows.append(dict(file=rel,python_ast=status,python_error=error,nicad_status=ns,nicad_failed_count=nfail,nicad_extracted_functions=nextract,nicad_exit=nrc,nicad_error_excerpt=msg,size_bytes=len(data)))
            if i%25==0 or i==len(files):print(f'[B-02] {i}/{len(files)} Python AST fail={counts["python_ast_fail"]} NiCad failed={counts["nicad_failed"]} NiCad errors={counts["nicad_error"]}',flush=True)
    finally:
        if args.mode=='probe':
            safe_delete(cache,nicad/'nicadclones')
            if tmp_in.exists():shutil.rmtree(tmp_in)
    with (out/'file_parse_audit.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    failures=[r for r in rows if r['nicad_status']=='failed']
    with (out/'failed_files.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=FIELDS);w.writeheader();w.writerows(failures)
    categories=Counter()
    for row in rows:
        if row['nicad_status']=='failed':
            categories['NiCad_failed_Python_AST_'+row['python_ast']]+=1
        elif row['nicad_status'] in ('timeout','error','unknown'):categories['NiCad_'+row['nicad_status']]+=1
    with (out/'parse_error_categories.csv').open('w',newline='') as h:
        w=csv.writer(h);w.writerow(['category','count']);w.writerows(sorted(categories.items()))
    summary={'status':'COMPLETE','experiment':'B-02','version':'v1','mode':args.mode,'repo_commit':commit,'python_version':sys.version.split()[0],'staged_root':str(stage),'files_checked':len(rows),'counts':dict(counts),'categories':dict(categories),'b01_filter_summary':str(b01) if b01 else None,'b01_counts':baseline.get('counts'), 'b01_nicad_reported_parse_failures':282,'per_file_nicad_total_seconds':round(sum(times),2),'limitations':['B-01 NiCad reported 282 parse failures, but the exact list is not present in the uploaded XML; per-file probes reconstruct status independently.','Per-file NiCad extraction can behave differently from whole-project extraction; compare totals before claiming exact reproduction.','Python ast.parse acceptance does not guarantee compatibility with NiCad Python grammar.','Single-file probes can be time-consuming; a full probe is needed for complete coverage.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=['# B-02 NiCad Python Parse Failure Audit','',f'- Commit: `{commit}`',f'- Python: `{sys.version.split()[0]}`',f'- Mode: `{args.mode}`',f'- Files examined: **{len(rows)}**',f'- B-01 reported parsing failures: **282**','', '## Counts']
    report += [f'- {k}: {v}' for k,v in sorted(counts.items())]
    report+=['','## Interpretation','- Compare `nicad_failed` with B-01 reported 282 only after full probe.','- `NiCad_failed_Python_AST_pass` suggests parser/version mismatch, not proof of the root cause.','- Review extracted error lines and problematic files before deciding whether to change detectors.']
    (out/'report.md').write_text('\n'.join(report)+'\n')
    print('[B-02] COMPLETE OUTPUT_DIR='+str(out),flush=True)
if __name__=='__main__':main()
