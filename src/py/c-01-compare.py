#!/usr/bin/env python3
"""Read-only comparison of independent C-01 results to an explicitly supplied B-07 run."""
import argparse,csv,json
from pathlib import Path

def rows(file):
    with file.open(newline='',encoding='utf-8-sig') as h:return list(csv.DictReader(h))

def pair_set(path):
    seen=set()
    for r in rows(path):
        a=(r['file_a'],int(r['start_a']),int(r['end_a']))
        b=(r['file_b'],int(r['start_b']),int(r['end_b']))
        if a!=b:seen.add(tuple(sorted((a,b))))
    return seen

def main():
    p=argparse.ArgumentParser();p.add_argument('--c01',type=Path,required=True);p.add_argument('--b07',type=Path,required=True)
    a=p.parse_args();c=a.c01.resolve();b=a.b07.resolve()
    if not (c/'project_run.json').exists():p.error('C-01 run incomplete')
    result={'experiment':'C-01 vs B-07','c01':str(c),'b07':str(b),'metrics':{},'pair_identity':{},'limitations':[]}
    cs=json.loads((c/'hybrid/summary.json').read_text())
    ref=b/'hybrid/summary.json'
    if not ref.exists():p.error('Missing B-07 hybrid/summary.json at '+str(ref))
    bs=json.loads(ref.read_text())
    for key in sorted(set(cs)&set(bs)):
        if key in {'staged_files','functions_indexed','ni_parse_failed_files','nicad_primary_unique_pairs',
                   'jscpd_strict_fallback_unique_pairs','hybrid_unique_pairs'}:
            result['metrics'][key]={'c01':cs[key],'b07':bs[key],'equal':cs[key]==bs[key]}
    cman=c/'hybrid/input_manifest.csv';bman=b/'hybrid/input_manifest.csv'
    if cman.exists() and bman.exists():
        cm={r['file']:r['sha256'] for r in rows(cman)};bm={r['file']:r['sha256'] for r in rows(bman)}
        result['input_byte_identity']={'equal':cm==bm,'c01_files':len(cm),'b07_files':len(bm),
                                      'only_c01':len(cm.keys()-bm.keys()),'only_b07':len(bm.keys()-cm.keys()),
                                      'changed_same_path':sum(cm[k]!=bm[k] for k in cm.keys()&bm.keys())}
    else:result['input_byte_identity']={'equal':None,'reason':'Input manifests unavailable'}
    for file in ('nicad_primary_pairs.csv','jscpd_fallback_pairs.csv','hybrid_function_clone_pairs.csv'):
        cf=c/'hybrid'/file;bf=b/'hybrid'/file
        if cf.exists() and bf.exists():
            left=pair_set(cf);right=pair_set(bf)
            result['pair_identity'][file]={'c01':len(left),'b07':len(right),'intersection':len(left&right),
                'only_c01':len(left-right),'only_b07':len(right-left),'exact_equal':left==right}
        else:result['pair_identity'][file]={'available':False,'reason':'One or both pair-level CSVs missing'}
    result['limitations'].append('Count agreement does not imply pair identity; inspect pair_identity and input_byte_identity.')
    result['limitations'].append('The comparison reads B-07 only after C-01 detection is completed; B-07 is never a detection input.')
    dest=c/'comparison_to_b07.json';dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
