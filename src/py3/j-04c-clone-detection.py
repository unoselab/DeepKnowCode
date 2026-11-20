#!/usr/bin/env python3
"""J-04c v2: subject-isolated C-01-compatible clone detection of J-04b v8 verified inputs.

Configuration is owned by src/sh3/j-04c-clone-detection.sh and required here via CLI.
Never re-mask/re-filter source code, and never write inside the J-04b v8 input.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

HEX = re.compile(r'^[0-9a-f]{64}$')
SUBJECT = re.compile(r'^[a-z0-9][a-z0-9_.-]*__[a-z0-9][a-z0-9_.-]*$')
PROBE_FIELDS = ['file','python_ast','python_error','nicad_status','nicad_category',
                'nicad_failed_count','nicad_extracted_functions','nicad_exit',
                'nicad_error_excerpt','size_bytes']
PAIR_FIELDS = ['subject','pair_id','file_a','start_a','end_a','function_a',
               'file_b','start_b','end_b','function_b','source',
               'ni_parse_failed_a','ni_parse_failed_b']


def chicago():
    return datetime.now(ZoneInfo(os.environ['J04C_LOG_TIMEZONE'])).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def safe_rel(raw: str) -> Path:
    p = PurePosixPath(raw)
    require(bool(raw) and not p.is_absolute() and p.suffix == '.py' and
            str(p) == raw and all(part not in ('', '.', '..') for part in raw.split('/')) and
            not any('\\' in part or '\x00' in part for part in p.parts),
            'Unsafe Python relative path: '+repr(raw))
    return Path(*p.parts)


def no_symlink_chain(file: Path, stop: Path):
    probe = file
    while True:
        require(not probe.is_symlink(), 'Symlink rejected: '+str(probe))
        if probe == stop:
            break
        require(stop in probe.parents, 'Path escaped verified root: '+str(file))
        probe = probe.parent


def read_json(path: Path):
    require(path.is_file() and not path.is_symlink(), 'Missing/unsafe JSON: '+str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def write_json_new(path: Path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write('\n')


def write_csv_new(path: Path, rows, fields):
    with path.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def load_csv(path: Path, fields):
    require(path.is_file() and not path.is_symlink(), 'Missing/unsafe CSV: '+str(path))
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == fields, 'CSV header mismatch: '+str(path))
        return list(reader)


def inventory(args, *, hash_check: bool, subjects_subset=None):
    """Fail closed on J-04b v8 manifest/CSV/tree mismatch and unlisted files."""
    parent = args.dataset_root
    root = args.input_root
    require(parent.is_dir() and not parent.is_symlink(), 'Missing/unsafe dataset root')
    require(root == parent / 'repositories' and root.is_dir() and not root.is_symlink(),
            'Expected /repositories child as input root')
    manifest_file = parent / 'clone-input-manifest.json'
    accepted_file = parent / 'clone-accepted.csv'
    excluded_file = parent / 'clone-excluded-parsing-failures.csv'
    m = read_json(manifest_file)
    require(m.get('schema') == 'deepknowcode-j04b-clone-input-v8' and
            m.get('source_data_version') == args.source_data_version and
            m.get('policy') == args.input_policy, 'Unexpected J-04b manifest identity/policy')
    require(Path(m.get('clone_input_root', '')).resolve() == root.resolve(),
            'J-04b manifest clone_input_root differs from configured source')
    count = m.get('counts', {})
    require(count.get('subjects') == args.expected_subjects and
            count.get('upstream_included') == args.expected_upstream_included and
            count.get('clone_included') == args.expected_files and
            count.get('unparsed_excluded') == args.expected_unparsed,
            'J-04b expected counts mismatch')
    require(m.get('accepted_csv_sha256') == digest(accepted_file) and
            m.get('excluded_csv_sha256') == digest(excluded_file),
            'J-04b input audit CSV SHA256 mismatch')
    acc = load_csv(accepted_file, ['subject', 'file', 'sha256'])
    exc = load_csv(excluded_file, ['subject', 'file', 'sha256', 'reason'])
    require(len(acc) == args.expected_files and len(exc) == args.expected_unparsed,
            'Unexpected accepted/excluded CSV counts')
    subjects = m.get('subjects', [])
    require(len(subjects) == args.expected_subjects, 'Manifest subject count mismatch')
    names = [s['subject'] for s in subjects]
    require(len(set(names)) == len(names) and all(SUBJECT.fullmatch(n) for n in names),
            'Manifest subject names invalid or duplicated')
    require(set(names) == {p.name for p in root.iterdir() if p.is_dir() and not p.is_symlink()},
            'Unexpected subject directory inventory')
    require(all(p.is_dir() and not p.is_symlink() for p in root.iterdir()),
            'Unsafe non-directory/symlink under subject root')
    by_subject = {n: [] for n in names}
    seen = set()
    for rec in acc:
        n = rec['subject']
        require(n in by_subject and HEX.fullmatch(rec['sha256']) is not None,
                'Invalid accepted CSV subject/hash: '+str(rec))
        rel = safe_rel(rec['file'])
        key = (n, rel.as_posix())
        require(key not in seen, 'Duplicate accepted input: '+str(key))
        seen.add(key)
        by_subject[n].append(rec)
    for rec in exc:
        require(rec['subject'] in by_subject and HEX.fullmatch(rec['sha256']) and rec['reason'],
                'Invalid excluded CSV record: '+str(rec))
        key = (rec['subject'], safe_rel(rec['file']).as_posix())
        require(key not in seen, 'Accepted/excluded path collision: '+str(key))
        seen.add(key)
    subject_meta = {s['subject']: s for s in subjects}
    for n in names:
        require(subject_meta[n].get('clone_accepted') == len(by_subject[n]),
                'Subject manifest accepted count mismatch: '+n)
    only = set(subjects_subset) if subjects_subset is not None else set(names)
    require(only <= set(names), 'Unrecognized subject selection: '+repr(only-set(names)))
    for n in sorted(only):
        directory = root / n
        expected_files = {r['file'] for r in by_subject[n]}
        found = set()
        for base, dirs, files in os.walk(directory, followlinks=False):
            basepath = Path(base)
            for item in dirs:
                require(not (basepath / item).is_symlink(), 'Unsafe source directory symlink')
            for item in files:
                f = basepath / item
                no_symlink_chain(f, root)
                require(f.is_file(), 'Non-regular source file: '+str(f))
                found.add(f.relative_to(directory).as_posix())
        require(found == expected_files,
                f'Subject file-set differs from manifest: {n} actual={len(found)} expected={len(expected_files)}')
        if hash_check:
            for r in by_subject[n]:
                f = directory / safe_rel(r['file'])
                require(digest(f) == r['sha256'], 'J-04b filtered source SHA256 mismatch: '+str(f))
        print(f'PASS INPUT subject={n} files={len(by_subject[n])} sha256_checked={hash_check}', flush=True)
    return m, by_subject, subject_meta


def check_tools(a):
    require(sys.version.split()[0] == a.expected_python_version,
            'Python version mismatch: '+sys.version.split()[0]+' vs '+a.expected_python_version)
    require(a.nicad_bin.is_file() and os.access(a.nicad_bin, os.X_OK), 'Missing NiCad executable')
    require(a.txl_bin.is_file() and os.access(a.txl_bin, os.X_OK), 'Missing TXL executable')
    require(shutil.which(a.jscpd_launcher) is not None, 'Missing npx/jscpd launcher')
    require(a.merge_script.is_file() and not a.merge_script.is_symlink(), 'Missing C-01 merge implementation')
    result = subprocess.run(['git','hash-object',str(a.merge_script)],capture_output=True,text=True,check=True)
    require(result.stdout.strip() == a.merge_git_blob_sha, 'C-01 merge source Git blob differs from pinned version')
    require((a.nicad_home / 'nicadclones').is_dir(), 'Missing NiCad cache root')
    print(f'PASS TOOLS python={sys.version.split()[0]} nicad={a.nicad_bin} txl={a.txl_bin} '
          f'c01_merge_blob={a.merge_git_blob_sha}',flush=True)


def run_cmd(cmd, log_path: Path, *, cwd: Path, env, timeout: int):
    print('COMMAND', ' '.join(map(str,cmd)), flush=True)
    with log_path.open('x',encoding='utf-8') as f:
        start = time.monotonic()
        proc = subprocess.Popen([str(v) for v in cmd], stdout=f, stderr=subprocess.STDOUT,
                                cwd=cwd, env=env)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError(f'Command timed out after {timeout}s; see {log_path}')
    print(f'COMMAND_DONE rc={rc} elapsed_sec={time.monotonic()-start:.1f} log={log_path}',flush=True)
    if rc:
        raise RuntimeError(f'Command failed exit={rc}: {cmd[0]}; see {log_path}')
    return log_path.read_text(encoding='utf-8',errors='replace')


def classify(row):
    """Exact C-01 parse-classification categories (v5 detector uses these)."""
    msg = row.get('nicad_error_excerpt') or ''
    def integer(v):
        try: return int(v)
        except (TypeError, ValueError): return None
    fail = integer(row.get('nicad_failed_count')) or 0
    extracted = integer(row.get('nicad_extracted_functions'))
    code = integer(row.get('nicad_exit'))
    status = (row.get('nicad_status') or '').strip()
    if fail or re.search(r'source files? failed to parse', msg, re.I): return 'NICAD_PARSE_FAILURE'
    if status == 'timeout' or code == 124: return 'TIMEOUT'
    if extracted == 0: return 'NO_FUNCTIONS_EXTRACTED'
    if code not in (None, 0) and re.search(r'renaming failed',msg,re.I): return 'RENAMING_FAILURE'
    if code not in (None, 0): return 'OTHER_EXECUTION_ERROR'
    if status == 'not_probed': return 'NOT_PROBED'
    if extracted is None: return 'UNKNOWN'
    return 'SUCCESS'


def extract_count(s, regex):
    matches = re.findall(regex, s, flags=re.I)
    return int(matches[-1]) if matches else None


def single_probe(file, tmp, cache, a, env):
    """Isolated C-01-style per-file NiCad probe; never touch C-01's c01_probe cache."""
    require(not tmp.exists() and not tmp.is_symlink() and not cache.exists() and not cache.is_symlink(),
            'Probe staging/cache occupied; refusing overwrite')
    tmp.mkdir(parents=True)
    shutil.copyfile(file,tmp/'target.py')
    t0 = time.monotonic()
    try:
        try:
            p=subprocess.run([str(a.nicad_bin),a.nicad_granularity,a.nicad_language,str(tmp),a.nicad_config],
                 cwd=a.nicad_home,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                 text=True,errors='replace',timeout=a.probe_timeout_sec)
            output=p.stdout or ''
            failed=extract_count(output,r'Warning:\s*(\d+)\s+source files failed to parse') or 0
            extracted=extract_count(output,r'Extracted\s+(\d+)\s+functions')
            state='failed' if failed else 'passed'
            if p.returncode: state='error'
            if extracted is None and not p.returncode: state='unknown'
            msg=' | '.join(line.strip() for line in output.splitlines()
                 if re.search(r'warn|error|fail|parse|extract',line,re.I))[-900:]
            return state,failed,extracted,p.returncode,msg,round(time.monotonic()-t0,3)
        except subprocess.TimeoutExpired:
            return 'timeout',None,None,124,'NiCad per-file probe timed out',round(time.monotonic()-t0,3)
    finally:
        # These are this invocation's unique paths, and only these two can be removed.
        for path in (cache,tmp):
            if path.is_symlink():
                raise RuntimeError('Unsafe probe symlink during cleanup: '+str(path))
            if path.is_dir(): shutil.rmtree(path)
            elif path.exists(): raise RuntimeError('Unexpected non-directory probe path: '+str(path))


def probe_all(a, stage, out, env, nicad_bulk_log):
    bulk_failed = extract_count(nicad_bulk_log,r'Warning:\s*(\d+)\s+source files failed to parse') or 0
    should_probe = a.probe_mode == 'all' or bulk_failed > 0
    files = sorted(stage.rglob('*.py'))
    dest = out/'probe'
    dest.mkdir(exist_ok=False)
    temp = dest / (stage.name + '_probe')
    cache = a.nicad_home / 'nicadclones' / temp.name
    require(not temp.exists() and not cache.exists() and not cache.is_symlink(),
            'Pre-existing unique probe cache')
    rows=[]; counts=Counter();times=0.0
    for i, file in enumerate(files,1):
        rel = file.relative_to(stage).as_posix()
        raw = file.read_bytes()
        try:
            ast.parse(raw,filename=rel)
            p_status,py_error='pass',''
        except (SyntaxError,ValueError,UnicodeError,RecursionError) as exc:
            p_status,py_error='fail',str(exc)[:300]
        if should_probe:
            ns,nf,ne,nrc,msg,secs=single_probe(file,temp,cache,a,env)
            times+=secs
        else:
            ns,nf,ne,nrc,msg='not_probed',None,None,None,''
        row={'file':rel,'python_ast':p_status,'python_error':py_error,'nicad_status':ns,
             'nicad_failed_count':nf,'nicad_extracted_functions':ne,'nicad_exit':nrc,
             'nicad_error_excerpt':msg,'size_bytes':len(raw)}
        row['nicad_category']=classify(row)
        counts[row['nicad_category']]+=1
        rows.append(row)
        if i % a.probe_progress_every == 0 or i == len(files):
            print(f'PROBE_PROGRESS {i}/{len(files)} nicad_parse_failures={counts["NICAD_PARSE_FAILURE"]} '
                  f'elapsed_sec={times:.1f}',flush=True)
    write_csv_new(out/'file_parse_audit.csv',rows,PROBE_FIELDS)
    write_json_new(out/'parse_categories.json',dict(counts))
    return {'bulk_nicad_parse_failures':bulk_failed,'probed_files':len(files) if should_probe else 0,
            'probe_elapsed_sec':round(times,3),'categories':dict(counts)}


def locate_xml(cache, glob_pattern):
    require(cache.is_dir() and not cache.is_symlink(),'NiCad cache not created: '+str(cache))
    matches=list(cache.rglob(glob_pattern))
    require(len(matches)==1 and matches[0].is_file() and not matches[0].is_symlink(),
            f'Expected one NiCad clone-pairs XML, found {len(matches)} in {cache}')
    return matches[0]


def run_subject(a, manifest, groups, metadata):
    n=a.subject
    require(n in groups, 'Unknown subject: '+n)
    check_tools(a)
    # Always compare every source byte against J-04b audit before a detector starts.
    inventory(a,hash_check=True,subjects_subset=[n])
    out=a.output_root/n
    require(not out.exists() and not out.is_symlink(), 'Run output exists; refusing overwrite: '+str(out))
    require(not out.parent.is_symlink(),'Unsafe run root symlink')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.mkdir(exist_ok=False)
    start=time.monotonic()
    stage_name='j04c'+''.join(ch for ch in a.run_id.lower() if ch.isalnum())[:55]+hashlib.sha256(n.encode()).hexdigest()[:8]
    stage=out/'stage'/stage_name
    stage.mkdir(parents=True,exist_ok=False)
    cache=a.nicad_home/'nicadclones'/stage.name
    try:
        require(not cache.exists() and not cache.is_symlink(),
                'NiCad project cache occupied; refusing overwrite: '+str(cache))
        rows=[]
        for i,item in enumerate(groups[n],1):
            rel=safe_rel(item['file'])
            src=a.input_root/n/rel
            dst=stage/rel
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst)
            require(digest(dst)==item['sha256'],'Stage-copy SHA256 mismatch: '+str(dst))
            rows.append({'file':item['file'],'bytes':src.stat().st_size,'sha256':item['sha256']})
            if i % a.copy_progress_every == 0 or i == len(groups[n]):
                print(f'COPY_PROGRESS subject={n} copied={i}/{len(groups[n])}',flush=True)
        write_csv_new(out/'input_inventory.csv',rows,['file','bytes','sha256'])
        env=os.environ.copy()
        env['PATH']=str(a.txl_bin.parent)+os.pathsep+env.get('PATH','')
        env['TZ']='America/Chicago'
        print(f'[{chicago()}] START DETECTOR=NiCad subject={n}',flush=True)
        ni_log=run_cmd([a.nicad_bin,a.nicad_granularity,a.nicad_language,stage,a.nicad_config],
                        out/'nicad.log',cwd=a.nicad_home,env=env,timeout=a.nicad_timeout_sec)
        ni_xml=locate_xml(cache,a.nicad_xml_pattern)
        xml_out=out/'nicad_xml';xml_out.mkdir(exist_ok=False)
        xml_dest=xml_out/ni_xml.name
        shutil.copy2(ni_xml,xml_dest)
        require(digest(xml_dest)==digest(ni_xml),'NiCad XML copy mismatch')
        print(f'[{chicago()}] PASS DETECTOR=NiCad subject={n} xml={xml_dest}',flush=True)
        print(f'[{chicago()}] START DETECTOR=NiCadProbe subject={n} mode={a.probe_mode}',flush=True)
        probe_summary=probe_all(a,stage,out,env,ni_log)
        print(f'[{chicago()}] PASS DETECTOR=NiCadProbe subject={n} '
              f'parse_failed={probe_summary["categories"].get("NICAD_PARSE_FAILURE",0)}',flush=True)
        jsout=out/'jscpd';jsout.mkdir(exist_ok=False)
        print(f'[{chicago()}] START DETECTOR=jscpd subject={n}',flush=True)
        run_cmd([a.jscpd_launcher,'--yes',a.jscpd_package,str(stage),'--format',a.jscpd_format,
                 '--similarity',str(a.jscpd_similarity),'--min-lines',str(a.jscpd_min_lines),
                 '--min-tokens',str(a.jscpd_min_tokens),'--reporters',a.jscpd_reporter,
                 '--output',str(jsout)],out/'jscpd.log',cwd=a.root,env=env,timeout=a.jscpd_timeout_sec)
        jsfile=jsout/'jscpd-report.json'
        require(jsfile.is_file() and not jsfile.is_symlink(),'Missing jscpd report JSON')
        print(f'[{chicago()}] PASS DETECTOR=jscpd subject={n}',flush=True)
        merged=out/'hybrid'
        print(f'[{chicago()}] START DETECTOR=HybridMerge subject={n}',flush=True)
        run_cmd([sys.executable,a.merge_script,'--project-root',a.root,'--stage',stage,
                 '--ni-file',xml_dest,'--audit-file',out/'file_parse_audit.csv',
                 '--js-file',jsfile,'--output',merged,
                 '--ni-threshold',str(a.ni_mapping_threshold),
                 '--strict-fraction',str(a.strict_function_fraction),
                 '--boundary-tolerance',str(a.boundary_tolerance_lines)],
                out/'merge.log',cwd=a.root,env=env,timeout=a.merge_timeout_sec)
        hybrid=read_json(merged/'summary.json')
        require(hybrid['staged_files']==len(rows),'Hybrid staged_files mismatch')
        require(hybrid['python_ast_errors']==0,
                'Hybrid AST errors: expected zero with native 3.14 verified input')
        # Preserve unmodified C-01 core merge reports for direct methodology traceability,
        # and add a J-04c-specific clone-pairs CSV for downstream indexing.
        pairs=[]
        old=load_csv(merged/'hybrid_function_clone_pairs.csv',
            ['pair_id','file_a','start_a','end_a','function_a','file_b','start_b','end_b','function_b',
             'source','ni_parse_failed_a','ni_parse_failed_b'])
        fileset={r['file'] for r in rows}
        for i,item in enumerate(old,1):
            require(item['file_a'] in fileset and item['file_b'] in fileset,
                    'Cross-subject or absent source endpoint detected')
            pairs.append({'subject':n,**item,'pair_id':f'J04C-{i:08d}'})
        require(len(pairs)==hybrid['hybrid_unique_pairs'],'Hybrid pair count mismatch')
        write_csv_new(out/'j04c_function_clone_pairs.csv',pairs,PAIR_FIELDS)
        print(f'[{chicago()}] PASS DETECTOR=HybridMerge subject={n} '
              f'ni_pairs={hybrid["nicad_primary_unique_pairs"]} '
              f'jscpd_fallback={hybrid["jscpd_strict_fallback_unique_pairs"]} '
              f'hybrid_pairs={hybrid["hybrid_unique_pairs"]}',flush=True)
        keep=['input_inventory.csv','nicad.log','file_parse_audit.csv','parse_categories.json',
              'jscpd.log','merge.log','j04c_function_clone_pairs.csv',
              'hybrid/summary.json','hybrid/input_manifest.csv','hybrid/python_ast_errors.csv',
              'hybrid/nicad_primary_pairs.csv','hybrid/jscpd_fallback_pairs.csv',
              'hybrid/hybrid_function_clone_pairs.csv','jscpd/jscpd-report.json',
              'nicad_xml/'+xml_dest.name]
        for name in keep: require((out/name).is_file(),'Expected detector artifact missing: '+name)
        done={'schema':f'deepknowcode-j04c-{a.version}','experiment':a.experiment,'version':a.version,
              'status':'COMPLETED','timestamp_chicago':chicago(),
              'subject':n,'run_id':a.run_id,'input_data_version':a.source_data_version,
              'input_policy':a.input_policy,'input_root':str(a.input_root),
              'input_manifest_sha256':digest(a.dataset_root/'clone-input-manifest.json'),
              'accepted_csv_sha256':digest(a.dataset_root/'clone-accepted.csv'),
              'subject_git_head':metadata[n]['git_head'],
              'source_files':len(rows),'stage':str(stage),'nicad_cache':str(cache),
              'settings':{'python':sys.version.split()[0],
                          'nicad':{'bin':str(a.nicad_bin),'granularity':a.nicad_granularity,
                                   'language':a.nicad_language,'config':a.nicad_config,
                                   'clone_xml_pattern':a.nicad_xml_pattern},
                          'jscpd':{'launcher':a.jscpd_launcher,'package':a.jscpd_package,
                                   'format':a.jscpd_format,'similarity':a.jscpd_similarity,
                                   'min_lines':a.jscpd_min_lines,'min_tokens':a.jscpd_min_tokens,
                                   'reporter':a.jscpd_reporter},
                          'hybrid':{'merge_script':str(a.merge_script),'merge_git_blob_sha':a.merge_git_blob_sha,
                                    'ni_mapping_threshold':a.ni_mapping_threshold,
                                    'strict_function_fraction':a.strict_function_fraction,
                                    'boundary_tolerance_lines':a.boundary_tolerance_lines},
                          'probe':{'mode':a.probe_mode,'timeout_sec':a.probe_timeout_sec},
                          'timeouts':{'nicad':a.nicad_timeout_sec,'jscpd':a.jscpd_timeout_sec,
                                      'merge':a.merge_timeout_sec}},
              'probe_summary':probe_summary,'metrics':hybrid,
              'elapsed_sec':round(time.monotonic()-start,3),
              'artifacts_sha256':{name:digest(out/name) for name in keep}}
        write_json_new(out/'j04c-run.json',done)
        print(f'PASS J04C_SUBJECT subject={n} files={len(rows)} '
              f'clone_pairs={len(pairs)} run={out}',flush=True)
    except BaseException as error:
        # Preserve all partial run data, including failures. Never touch input or existing results.
        fail={'schema':f'deepknowcode-j04c-{a.version}','experiment':a.experiment,'version':a.version,
              'status':'FAILED','subject':n,'run_id':a.run_id,'timestamp_chicago':chicago(),
              'error':str(error),'run':str(out)}
        write_json_new(out/'j04c-failed.json',fail)
        raise
    return out


def verify_subject(a, n, group):
    run = a.output_root/n
    require(run.is_dir() and not run.is_symlink(), 'No J-04c subject result directory: '+str(run))
    m=read_json(run/'j04c-run.json')
    require(m.get('schema')==f'deepknowcode-j04c-{a.version}' and m.get('status')=='COMPLETED' and
            m.get('version')==a.version and m.get('subject')==n and
            m.get('input_data_version')==a.source_data_version and
            m.get('source_files')==len(group),'J04c run manifest mismatch: '+str(run))
    require(m.get('input_manifest_sha256')==digest(a.dataset_root/'clone-input-manifest.json') and
            m.get('accepted_csv_sha256')==digest(a.dataset_root/'clone-accepted.csv'),
            'Source dataset changed since this clone detection run')
    stage=Path(m['stage'])
    require(stage.is_dir() and not stage.is_symlink() and run in stage.parents,
            'Missing/unsafe original detector stage')
    input_records=load_csv(run/'input_inventory.csv',['file','bytes','sha256'])
    require(len(input_records)==len(group),'Run stage inventory count mismatch')
    expected={r['file']:r['sha256'] for r in group}
    for item in input_records:
        require(item['file'] in expected and item['sha256']==expected[item['file']],
                'Stage inventory differs from source audit')
        file=stage/safe_rel(item['file'])
        require(file.is_file() and not file.is_symlink() and
                digest(file)==item['sha256'] and file.stat().st_size==int(item['bytes']),
                'Detector stage changed: '+str(file))
    require({p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()}==set(expected),
            'Detector stage contains missing/extra Python files')
    for name,sha in m.get('artifacts_sha256',{}).items():
        file=run/name
        require(file.is_file() and not file.is_symlink() and digest(file)==sha,
                'Detector output artifact corrupted: '+str(file))
    pairs=load_csv(run/'j04c_function_clone_pairs.csv',PAIR_FIELDS)
    require(len(pairs)==m['metrics']['hybrid_unique_pairs'], 'Stored hybrid pair count mismatch')
    for row in pairs:
        require(row['subject']==n and row['file_a'] in expected and row['file_b'] in expected,
                'Invalid clone endpoint in J04c pairs')
    print(f'PASS J04C_VERIFY subject={n} files={len(group)} '
          f'pairs={len(pairs)} run={run}',flush=True)
    return {'subject':n,'source_files':len(group),'clone_pairs':len(pairs),'run':str(run),
            'metrics':m['metrics']}


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','version','source-data-version','input-policy','expected-python-version',
                 'nicad-granularity','nicad-language','nicad-config','nicad-xml-pattern',
                 'jscpd-launcher','jscpd-package','jscpd-format','jscpd-reporter','probe-mode',
                 'merge-git-blob-sha','run-id','log-timezone','batch-mode'):
        p.add_argument('--'+name,required=True)
    for name in ('root','dataset-root','input-root','output-root','result-root','publish-root','nicad-home','nicad-bin','txl-bin',
                 'merge-script','report-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('expected-subjects','expected-files','expected-upstream-included',
                 'expected-unparsed','jscpd-min-lines','jscpd-min-tokens',
                 'boundary-tolerance-lines','probe-timeout-sec','probe-progress-every',
                 'copy-progress-every','nicad-timeout-sec','jscpd-timeout-sec','merge-timeout-sec'):
        p.add_argument('--'+name,type=int,required=True)
    for name in ('jscpd-similarity','ni-mapping-threshold','strict-function-fraction'):
        p.add_argument('--'+name,type=float,required=True)
    p.add_argument('--hash-check',required=True,choices=['true','false'])
    p.add_argument('--mode',choices=['inspect','dry-run','list-subjects','run-subject',
                                     'verify','summary','copy-result'],required=True)
    p.add_argument('--subject',required=True)
    a=p.parse_args(argv)
    require(a.log_timezone == 'America/Chicago', 'This experiment requires Chicago timestamps')
    os.environ['J04C_LOG_TIMEZONE'] = a.log_timezone
    for name in ('root','dataset_root','input_root','output_root','result_root','publish_root','nicad_home','nicad_bin','txl_bin',
                 'merge_script','report_dir'):
        setattr(a,name,getattr(a,name).absolute())
    require(a.experiment=='j-04c' and a.version.startswith('v') and a.version[1:].isdigit(),
            'Unexpected experiment/version')
    require(a.probe_mode in ('all','on_bulk_failures'),'Invalid --probe-mode')
    require(a.batch_mode in ('inspect','dry-run','smoke','subject','all','verify','copy-result'),
            'Invalid --batch-mode')
    require(a.expected_subjects>0 and a.expected_files>0 and
            a.expected_upstream_included>=a.expected_files and a.expected_unparsed>=0,
            'Invalid dataset expectations')
    require(a.expected_upstream_included==a.expected_files+a.expected_unparsed,
            'Expected inventory conservation failed')
    require(a.jscpd_min_lines>0 and a.jscpd_min_tokens>0 and
            0<a.jscpd_similarity<=1 and 0<a.ni_mapping_threshold<=1 and
            0<a.strict_function_fraction<=1 and a.boundary_tolerance_lines>=0,
            'Invalid clone parameters')
    require(all(getattr(a,n)>0 for n in ('probe_timeout_sec','probe_progress_every','copy_progress_every',
                'nicad_timeout_sec','jscpd_timeout_sec','merge_timeout_sec')),
            'Non-positive logging/probe/timeouts')
    require(a.output_root == a.report_dir/'subjects' and a.report_dir.parent == a.result_root,
            'Result/output hierarchy mismatch')
    require(a.dataset_root not in (a.result_root,a.publish_root) and
            a.dataset_root not in a.result_root.parents and
            a.result_root not in a.dataset_root.parents and
            a.dataset_root not in a.publish_root.parents and
            a.publish_root not in a.dataset_root.parents and
            a.result_root not in a.publish_root.parents and
            a.publish_root not in a.result_root.parents,
            'Input/result/publish roots overlap')
    require(bool(re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',a.run_id)), 'Invalid run ID')
    if a.mode in ('run-subject',):
        require(SUBJECT.fullmatch(a.subject) is not None, 'Invalid selected subject')
    return a



def report_candidates(result_root, mode):
    require(result_root.is_dir() and not result_root.is_symlink(),
            'Result root missing/unsafe: '+str(result_root))
    return sorted((d for d in result_root.iterdir()
                   if d.is_dir() and not d.is_symlink() and f'-{mode}-' in d.name),
                  key=lambda d:d.name, reverse=True)


def latest_all_result(a):
    candidates = report_candidates(a.result_root, 'all')
    require(bool(candidates), 'No J-04c --all result found; --copy-result requires completed --all')
    # Never fall back to an older completed run if the newest --all attempt failed.
    chosen = candidates[0]
    summary = read_json(chosen/'batch-summary.json')
    require(summary.get('experiment')==a.experiment and summary.get('version')==a.version and
            summary.get('batch_mode')=='all' and summary.get('status')=='COMPLETED' and
            summary.get('completed_subjects')==a.expected_subjects and
            summary.get('expected_subjects')==a.expected_subjects and
            summary.get('source_data_version')==a.source_data_version and
            summary.get('source_input')==str(a.input_root),
            'Latest --all result is not a complete matching J-04c batch: '+str(chosen))
    return chosen,summary


def verify_batch(a, root, groups, selected):
    require(root.is_dir() and not root.is_symlink(), 'Missing/unsafe result directory: '+str(root))
    a.output_root=root/'subjects'
    results=[]
    for n in selected:
        require(n in groups, 'Unknown subject: '+n)
        results.append(verify_subject(a,n,groups[n]))
    return results


def collect_source_files(source_root):
    """Walk a completed subject result without following symlinks or special files."""
    require(source_root.is_dir() and not source_root.is_symlink(),
            'Missing/unsafe result source: '+str(source_root))
    paths={}
    for base,dirs,files in os.walk(source_root,followlinks=False):
        here=Path(base)
        for name in dirs:
            require(not (here/name).is_symlink(), 'Symlinked source directory: '+str(here/name))
        for name in files:
            src=here/name
            require(src.is_file() and not src.is_symlink(), 'Unsafe source file: '+str(src))
            rel=src.relative_to(source_root).as_posix()
            require(rel not in paths, 'Duplicate file in result tree: '+rel)
            paths[rel]=src
    require(bool(paths), 'Empty subject result: '+str(source_root))
    return paths


def copy_result(a, groups):
    latest, summary=latest_all_result(a)
    names=sorted(groups)
    recorded=summary.get('subjects',[])
    require(len(recorded)==len(names) and
            {r['subject'] for r in recorded}==set(names) and
            all(r.get('status')=='COMPLETED' for r in recorded),
            'Incomplete or inconsistent subjects in --all summary')
    print(f'[{chicago()}] COPY_SOURCE full_all={latest} subjects={len(names)}',flush=True)
    verified=verify_batch(a,latest,groups,names)
    require(len(verified)==a.expected_subjects and
            sum(v['source_files'] for v in verified)==a.expected_files and
            sum(v['clone_pairs'] for v in verified)==summary.get('clone_pairs_total'),
            'Source result aggregate mismatch')
    for r in recorded:
        v=next(x for x in verified if x['subject']==r['subject'])
        require(r['source_files']==v['source_files'] and r['clone_pairs']==v['clone_pairs'] and
                r['run']==v['run'], 'Subject summary/source identity mismatch: '+r['subject'])
    dest=a.publish_root
    require(not dest.is_symlink(), 'Publish root is a symlink: '+str(dest))
    exists=dest.exists()
    if exists:
        require(dest.is_dir() and not dest.is_symlink() and not any(dest.iterdir()),
                'Publish destination already exists and is not empty; refusing overwrite: '+str(dest))
    require(not dest.parent.is_symlink(), 'Publish parent is a symlink')
    # Stage alongside destination. Destination itself appears only after all copies verify.
    dest.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.'+dest.name+'.j04c-pending-',dir=dest.parent))
    entries=[]
    try:
        for n in names:
            source=latest/'subjects'/n
            files=collect_source_files(source)
            target=temporary/n
            target.mkdir(parents=True,exist_ok=False)
            hashes={}
            for index,(rel,src) in enumerate(sorted(files.items()),1):
                dst=target/Path(rel)
                dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(src,dst,follow_symlinks=False)
                sha=digest(src)
                require(digest(dst)==sha and dst.stat().st_size==src.stat().st_size,
                        'Result copy SHA256 mismatch: '+str(dst))
                hashes[rel]=sha
                if index % a.copy_progress_every==0:
                    print(f'[{chicago()}] COPY_PROGRESS subject={n} copied={index}/{len(files)}',flush=True)
            src_run=read_json(source/'j04c-run.json')
            entries.append({'subject':n,'source_result':str(source),
                            'files_copied':len(files),
                            'source_files':src_run['source_files'],
                            'clone_pairs':src_run['metrics']['hybrid_unique_pairs'],
                            'j04c_run_sha256':hashes['j04c-run.json'],
                            'copy_file_sha256':hashes})
            print(f'[{chicago()}] PASS COPY_SUBJECT subject={n} files={len(files)} pairs={entries[-1]["clone_pairs"]}',flush=True)
        shutil.copy2(latest/'batch-summary.json',temporary/'batch-summary.json')
        require(digest(temporary/'batch-summary.json') == digest(latest/'batch-summary.json'),
                'Copied batch-summary.json hash differs from full run')
        receipt={'schema':f'deepknowcode-j04c-copy-{a.version}',
                 'experiment':a.experiment,'version':a.version,
                 'status':'COMPLETED','timestamp_chicago':chicago(),
                 'source_data_version':a.source_data_version,
                 'source_dataset_root':str(a.input_root),
                 'source_all_result':str(latest),
                 'source_batch_summary_sha256':digest(latest/'batch-summary.json'),
                 'published_root':str(dest),'published_subjects':len(entries),
                 'source_files':sum(r['source_files'] for r in entries),
                 'clone_pairs_total':sum(r['clone_pairs'] for r in entries),
                 'subjects':entries}
        write_json_new(temporary/'j04c-copy-manifest.json',receipt)
        require(not dest.is_symlink(), 'Publish destination symlink appeared during copy')
        if exists:
            require(dest.is_dir() and not any(dest.iterdir()),
                    'Publish destination became nonempty during copy: '+str(dest))
        else:
            require(not dest.exists(), 'Publish destination appeared during copy: '+str(dest))
        # POSIX rename atomically publishes a complete tree; an existing empty
        # destination directory may be replaced, never a nonempty one.
        temporary.rename(dest)
        print(f'PASS J04C_COPY_RESULT subjects={len(entries)} input_files={a.expected_files} '
              f'clone_pairs={receipt["clone_pairs_total"]} destination={dest}',flush=True)
        return receipt
    except BaseException:
        # Only pending output created by this invocation may be removed; never source/older results.
        if temporary.is_dir() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise


def main(argv=None):
    a=arguments(argv)
    if a.mode=='list-subjects':
        _,groups,_=inventory(a,hash_check=False,subjects_subset=[])
        for n in sorted(groups): print(n)
        return 0
    if a.mode in ('inspect','dry-run'):
        m,groups,_=inventory(a,hash_check=a.hash_check=='true')
        if a.mode=='dry-run': check_tools(a)
        print(f'PASS {a.mode.upper()} subjects={len(groups)} files={sum(len(v) for v in groups.values())} '
              f'unparsed_excluded={m["counts"]["unparsed_excluded"]} '
              f'result_root={a.result_root} publish_root={a.publish_root}',flush=True)
        return 0
    if a.mode=='run-subject':
        m,groups,metadata=inventory(a,hash_check=False,subjects_subset=[])
        run_subject(a,m,groups,metadata)
        return 0
    if a.mode=='verify':
        _,groups,_=inventory(a,hash_check=a.hash_check=='true',subjects_subset=[])
        if a.subject=='ALL':
            latest,_=latest_all_result(a)
            selected=sorted(groups)
        else:
            # For subject checks use the newest output containing this subject, including failed.
            candidates=[d for d in a.result_root.iterdir()
                        if d.is_dir() and not d.is_symlink() and
                        any(f'-{mode}-' in d.name for mode in ('all','smoke','subject')) and
                        (d/'subjects'/a.subject).exists()]
            require(bool(candidates),'No subject result found for '+a.subject)
            latest=sorted(candidates,key=lambda d:d.name)[-1]
            selected=[a.subject]
        results=verify_batch(a,latest,groups,selected)
        a.report_dir.mkdir(parents=True,exist_ok=True)
        filename='verification-all.json' if a.subject=='ALL' else f'verification-{a.subject}.json'
        write_json_new(a.report_dir/filename,{'status':'PASS','timestamp_chicago':chicago(),
                       'source_result_root':str(latest),'verified_subjects':len(results),
                       'results':results})
        print(f'PASS J04C_VERIFY_ALL verified_subjects={len(results)} '
              f'expected_subjects={a.expected_subjects} source={latest}',flush=True)
        return 0
    if a.mode=='summary':
        _,groups,_=inventory(a,hash_check=False,subjects_subset=[])
        require(a.batch_mode in ('all','smoke','subject'), 'Unexpected summary batch mode')
        names=sorted(groups) if a.batch_mode=='all' else [a.subject]
        require(all(n in groups for n in names), 'Invalid summary subject')
        summary=[]
        for n in names:
            path=a.output_root/n
            require(path.is_dir() and not path.is_symlink(), 'Missing batch subject result: '+n)
            d=read_json(path/'j04c-run.json')
            require(d.get('status')=='COMPLETED' and d.get('version')==a.version and
                    d.get('subject')==n and d.get('run_id')==a.run_id,
                    'Batch subject is not completed by this invocation: '+n)
            summary.append({'subject':n,'status':'COMPLETED',
                            'source_files':d['source_files'],
                            'clone_pairs':d['metrics']['hybrid_unique_pairs'],
                            'run':str(path)})
        total=sum(x['source_files'] for x in summary)
        if a.batch_mode=='all':
            require(len(summary)==a.expected_subjects and total==a.expected_files,
                    'Full --all batch has incorrect subject/file counts')
        obj={'schema':f'deepknowcode-j04c-batch-{a.version}',
             'experiment':a.experiment,'version':a.version,'status':'COMPLETED',
             'batch_mode':a.batch_mode,'run_id':a.run_id,
             'timestamp_chicago':chicago(),
             'source_data_version':a.source_data_version,'source_input':str(a.input_root),
             'result_root':str(a.report_dir),'publish_root':str(a.publish_root),
             'expected_subjects':a.expected_subjects,
             'completed_subjects':len(summary),'input_files_total':total,
             'clone_pairs_total':sum(x['clone_pairs'] for x in summary),
             'subjects':summary}
        write_json_new(a.report_dir/'batch-summary.json',obj)
        print(f'PASS J04C_SUMMARY batch_mode={a.batch_mode} '
              f'subjects_completed={len(summary)}/{a.expected_subjects} '
              f'pairs={obj["clone_pairs_total"]} report={a.report_dir / "batch-summary.json"}',flush=True)
        return 0
    if a.mode=='copy-result':
        _,groups,_=inventory(a,hash_check=a.hash_check=='true')
        copy_result(a,groups)
        return 0
    raise AssertionError('Unreachable mode')


if __name__=='__main__':
    try:
        sys.exit(main())
    except (OSError,RuntimeError,ValueError,KeyError,subprocess.SubprocessError) as exc:
        print(f'FAIL J04C {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        sys.exit(2)
