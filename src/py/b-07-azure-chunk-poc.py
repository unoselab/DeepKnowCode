#!/usr/bin/env python3
"""B-07 v5: read-only proof of concept for NiCad per-source chunked blind renaming."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

SOURCE_RE = re.compile(rb'(?ms)^<source\s[^>]*>\r?\n.*?^</source>\r?\n?')
META_RE = re.compile(rb'^<source\s[^>]*>', re.M)

def blocks(data):
    matches = list(SOURCE_RE.finditer(data))
    if not matches or data[:matches[0].start()].strip() or data[matches[-1].end():].strip():
        raise ValueError('Input has non-source content or no source blocks')
    for a,b in zip(matches, matches[1:]):
        if data[a.end():b.start()].strip():
            raise ValueError('Unexpected content between source blocks')
    return [m.group() for m in matches]

def process(chunks, stream, txl, nicad, timeout):
    outs=[]
    cmd=[str(stream),str(txl)+' stdin']
    for i,part in enumerate(chunks):
        proc=subprocess.run(cmd,input=part,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=nicad,timeout=timeout)
        if proc.returncode:
            raise RuntimeError(f'TXL chunk {i}: exit={proc.returncode}; stderr={proc.stderr[-1800:].decode(errors="replace")}')
        if len(blocks(proc.stdout)) != len(blocks(part)):
            raise RuntimeError(f'Chunk {i}: source block count mismatch')
        outs.append(proc.stdout)
    return b''.join(outs)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--source-xml',type=Path,required=True)
    ap.add_argument('--sample-functions',type=int,default=40)
    ap.add_argument('--chunk-size',type=int,default=10)
    ap.add_argument('--timeout',type=int,default=120)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.sample_functions<2 or a.chunk_size<1 or a.chunk_size>=a.sample_functions:
        ap.error('Require sample-functions >=2 and 1<=chunk-size<sample-functions')
    root=a.root.resolve();nicad=root/'tools/nicad-7.0.1-linux-x86_64'
    stream=nicad/'lib/nicad/tools/streamprocess.x'
    txl=nicad/'lib/nicad/txl/py-rename-blind-functions.x'
    for p in (a.source_xml,stream,txl):
        if not p.is_file():ap.error('Missing '+str(p))
    if a.out.exists():ap.error('Output directory already exists: '+str(a.out))
    a.out.mkdir(parents=True)
    report={'version':'v5-poc','source_xml':str(a.source_xml.resolve()),'sample_functions':a.sample_functions,'chunk_size':a.chunk_size,'status':'STARTED'}
    target=a.out/'comparison.json'
    try:
        # Read original but never modify it. Preserve the original metadata and source order.
        data=a.source_xml.read_bytes();all_blocks=blocks(data)
        sample=all_blocks[:a.sample_functions]
        if len(sample)!=a.sample_functions:raise ValueError('Insufficient source blocks')
        direct=process([b''.join(sample)],stream,txl,nicad,a.timeout)
        parts=[b''.join(sample[i:i+a.chunk_size]) for i in range(0,len(sample),a.chunk_size)]
        chunked=process(parts,stream,txl,nicad,a.timeout)
        direct_blocks=blocks(direct);chunk_blocks=blocks(chunked)
        same=direct==chunked
        # Compare each source block and metadata independently; stream separators may differ.
        blockwise=direct_blocks==chunk_blocks
        metadata=(list(map(lambda x:META_RE.search(x).group(),direct_blocks)) == list(map(lambda x:META_RE.search(x).group(),chunk_blocks)))
        report.update(status='PASS' if blockwise and metadata else 'FAIL',byte_identical=same,block_identical=blockwise,metadata_identical=metadata,source_blocks_total=len(all_blocks),direct_sha256=hashlib.sha256(direct).hexdigest(),chunk_sha256=hashlib.sha256(chunked).hexdigest(),direct_bytes=len(direct),chunk_bytes=len(chunked))
        if not blockwise: (a.out/'direct-sample.xml').write_bytes(direct);(a.out/'chunked-sample.xml').write_bytes(chunked)
    except Exception as exc:
        report.update(status='ERROR',error=str(exc))
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if report['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
