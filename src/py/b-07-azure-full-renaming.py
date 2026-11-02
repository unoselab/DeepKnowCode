#!/usr/bin/env python3
"""B-07 v8: lossless, atomic chunked NiCad blind renaming of an existing function cache.
Does not change source XML; global clone matching is performed separately by NiCad.
"""
import argparse, hashlib, json, os, subprocess, tempfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import re

SOURCE_RE = re.compile(rb'(?ms)^<source\s[^>]*>\r?\n.*?^</source>\r?\n?')
META_RE = re.compile(rb'^<source\s[^>]*>', re.M)

def blocks(data):
    matches=list(SOURCE_RE.finditer(data))
    if not matches or data[:matches[0].start()].strip() or data[matches[-1].end():].strip():
        raise RuntimeError('Invalid source-block stream')
    for a,b in zip(matches,matches[1:]):
        if data[a.end():b.start()].strip(): raise RuntimeError('Non-source content between blocks')
    return [m.group() for m in matches]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage-name',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--chunk-size',type=int,default=250)
    p.add_argument('--timeout',type=int,default=300)
    p.add_argument('--expected-count',type=int,default=24284)
    p.add_argument('--expected-sha256',help='Optional pinned source XML checksum')
    a=p.parse_args()
    if a.chunk_size<1 or a.timeout<1:p.error('chunk-size and timeout must be >0')
    if not re.fullmatch(r'b07[0-9]{20,}',a.stage_name):p.error('Unexpected stage name')
    root=a.root.resolve();nicad=root/'tools/nicad-7.0.1-linux-x86_64'
    cache=nicad/'nicadclones'/a.stage_name
    source=cache/(a.stage_name+'_functions.xml')
    dest=cache/(a.stage_name+'_functions-blind.xml')
    stage=a.stage_name
    if cache.is_symlink() or not cache.is_dir() or not source.is_file():p.error('Existing safe NiCad cache not found')
    if dest.is_symlink():p.error('Unsafe symlink at destination: '+str(dest))
    if dest.exists() and not dest.is_file():p.error('Destination is not a regular file: '+str(dest))
    # Preserve preexisting partial output from the failed NiCad run.
    # Publish only after all chunks validate, with an archival hardlink first.
    preexisting=dest.exists()
    if preexisting and dest.stat().st_nlink > 1:
        p.error('Existing renamed XML has multiple hardlinks; refusing unexpected replacement')
    stream=nicad/'lib/nicad/tools/streamprocess.x'
    txl=nicad/'lib/nicad/txl/py-rename-blind-functions.x'
    if not stream.is_file() or not txl.is_file():p.error('NiCad binary not found')
    out=a.output_dir.resolve()
    if out.exists():p.error('Output dir exists; refusing overwrite: '+str(out))
    out.mkdir(parents=True)
    report={'version':'v8','stage_name':stage,'status':'STARTED','chunk_size':a.chunk_size,'source_xml':str(source),'destination_xml':str(dest),'started_at':datetime.now(ZoneInfo('America/Chicago')).isoformat()}
    tmp=None
    try:
        data=source.read_bytes();original=blocks(data);sha=hashlib.sha256(data).hexdigest()
        report.update(source_sha256=sha,source_blocks=len(original),source_bytes=len(data))
        if a.expected_sha256 and sha.lower()!=a.expected_sha256.lower():raise RuntimeError('Original XML SHA256 mismatch')
        if a.expected_count and len(original)!=a.expected_count:raise RuntimeError('Original source count mismatch')
        fd,name=tempfile.mkstemp(prefix='.b07-v8-rename-',suffix='.tmp',dir=cache)
        tmp=Path(name);completed=0;chunks=0;digest=hashlib.sha256()
        # Adaptive divide-and-conquer: a TXL parse time limit may affect only one
        # large or pathological source. Never omit or silently rewrite that source.
        attempts=0;split_count=0;min_successful_chunk=None;failed_attempts=[]
        def rename_group(group,first_index):
            nonlocal attempts,split_count,min_successful_chunk
            attempts+=1
            input_data=b''.join(group)
            try:
                proc=subprocess.run([str(stream),str(txl)+' stdin'],input=input_data,
                                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=nicad,timeout=a.timeout)
                error=proc.stderr[-1500:].decode(errors='replace')
                if proc.returncode:
                    raise RuntimeError(f'TXL exit={proc.returncode}; '+error)
                transformed=blocks(proc.stdout)
                if len(transformed)!=len(group):
                    raise RuntimeError(f'Source count changed: {len(group)} -> {len(transformed)}')
                for local,(before,after) in enumerate(zip(group,transformed)):
                    if META_RE.search(before).group()!=META_RE.search(after).group():
                        raise RuntimeError(f'Source metadata changed at function {first_index+local+1}')
                min_successful_chunk=len(group) if min_successful_chunk is None else min(min_successful_chunk,len(group))
                return proc.stdout
            except (RuntimeError,subprocess.TimeoutExpired) as exc:
                err=str(exc)
                if len(group)==1:
                    header=META_RE.search(group[0]).group().decode('utf-8','replace')
                    failure={'function_index_1based':first_index+1,'source_header':header,
                             'source_bytes':len(group[0]),'failure':err[-1500:]}
                    report['isolated_failure']=failure
                    raise RuntimeError('Single-function NiCad TXL failure; no output published: '+json.dumps(failure,ensure_ascii=False)) from exc
                split_count+=1
                midpoint=len(group)//2
                if len(failed_attempts)<50:
                    failed_attempts.append({'first_function_1based':first_index+1,'group_size':len(group),'error':err[-450:]})
                print(f'SPLIT function={first_index+1} size={len(group)} into {midpoint}+{len(group)-midpoint}',flush=True)
                left=rename_group(group[:midpoint],first_index)
                right=rename_group(group[midpoint:],first_index+midpoint)
                return left+right
        with os.fdopen(fd,'wb') as h:
            for idx in range(0,len(original),a.chunk_size):
                group=original[idx:idx+a.chunk_size]
                renamed=rename_group(group,idx)
                h.write(renamed);digest.update(renamed)
                completed+=len(group);chunks+=1
                print(f'RENAME {completed}/{len(original)} chunks={chunks} splits={split_count} attempts={attempts}',flush=True)
            h.flush();os.fsync(h.fileno())
        report.update(attempts=attempts,adaptive_splits=split_count,
                      min_successful_chunk=min_successful_chunk,
                      failed_attempts_sample=failed_attempts)
        output=tmp.read_bytes();output_blocks=blocks(output)
        if len(output_blocks)!=len(original):raise RuntimeError('Final output count mismatch')
        if [META_RE.search(x).group() for x in output_blocks]!=[META_RE.search(x).group() for x in original]:raise RuntimeError('Final metadata/order mismatch')
        if preexisting:
            if dest.is_symlink() or not dest.is_file():
                raise RuntimeError('Destination changed type during renaming')
            existing=dest.read_bytes()
            report.update(existing_bytes=len(existing),existing_sha256=hashlib.sha256(existing).hexdigest())
            if existing == output:
                report['publication']='IDENTICAL_PREEXISTING_REUSED'
            else:
                # Historical broken output is never deleted: retain an archival hardlink
                # before atomic replacement. The archive name must not already exist.
                backup=cache/(dest.name+'.pre-v8-'+report['existing_sha256'][:16]+'.bak')
                if backup.exists() or backup.is_symlink():
                    raise RuntimeError('Backup path already exists; manual review required: '+str(backup))
                os.link(dest,backup,follow_symlinks=False)
                report['previous_output_archive']=str(backup)
                # Recheck that concurrent modifications have not happened.
                if hashlib.sha256(dest.read_bytes()).hexdigest()!=report['existing_sha256']:
                    raise RuntimeError('Preexisting destination changed during renaming; refusing publication')
                os.replace(tmp,dest)
                report['publication']='ARCHIVED_PREEXISTING_AND_ATOMICALLY_REPLACED'
        else:
            os.link(tmp,dest)
            report['publication']='CREATED_NO_OVERWRITE'
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=digest.hexdigest():
            raise RuntimeError('Published file hash mismatch')
        report.update(status='PASS',output_blocks=len(output_blocks),chunks=chunks,output_sha256=digest.hexdigest(),output_bytes=len(output),completed_at=datetime.now(ZoneInfo('America/Chicago')).isoformat())
    except Exception as exc:
        report.update(status='ERROR',error=str(exc),attempts=locals().get('attempts',0),adaptive_splits=locals().get('split_count',0),failed_attempts_sample=locals().get('failed_attempts',[]))
        raise
    finally:
        if tmp and tmp.exists():tmp.unlink()
        (out/'renaming_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
