#!/usr/bin/env python3
"""D-03c v4: conservative detector-based clone-type evidence extraction.

Reads v3 reconciled Python pairs and original C-01/B-07 hybrid CSVs. Never
converts detection method or similarity into an unjustified exact T1/T2/T3/T4.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

APPROVED = {
    'data.jsonl':'da8fdd6362df692e225a5bf5b72278fd6211cef75e48e6a539dda637b76a1843',
    'train.txt':'e8c8998f2e73a3587190904fb258ac1ccda306f0be62dec499622f4258603693',
    'valid.txt':'33c5a0506fdbbd8ff5f9604a9be1d4fba1b6b6d52943ff11ff53928861ac48a7',
}
FIELDS = ['id_a','id_b','split','label','canonical_project','project_status',
          'clone_type','clone_type_status','detector_category','detector_modes',
          'possible_types','evidence_rows','report_files','evidence_details']
RAW_FIELDS = ['id_a','id_b','split','canonical_project','raw_project','source_pair_id',
              'source','detector_source','report_kind','reported_clone_type',
              'reported_kind','mode_interpretation','matching_status']
TYPES = {'type-1':'TYPE_1','type-2':'TYPE_2','type-3':'TYPE_3','type-4':'TYPE_4',
         't1':'TYPE_1','t2':'TYPE_2','t3':'TYPE_3','t4':'TYPE_4',
         'type1':'TYPE_1','type2':'TYPE_2','type3':'TYPE_3','type4':'TYPE_4'}
KINDS = {'exact':'TYPE_1', 'renamed':'TYPE_2', 'similar':'TYPE_3'}


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        yield from csv.DictReader(f)


def write_csv(path,fields,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def dataset_check(root):
    base=root/'data/training/clean/python_mixed'
    for filename,expected in APPROVED.items():
        path=base/filename
        if not path.is_file() or sha256(path)!=expected:
            raise RuntimeError(f'C-10 approved file differs: {path}')
    pairs={}
    for split,filename in [('train','train.txt'),('valid','valid.txt')]:
        for line in (base/filename).read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            a,b,label=line.split()
            k=tuple(sorted([a,b]))
            if k in pairs:raise RuntimeError('Duplicate pair in C-10')
            pairs[k]=(split,int(label))
    return pairs


def find_v3(root,explicit):
    if explicit:return explicit.resolve()
    runroot=root/'results/d-03c'
    candidates=sorted((p for p in runroot.glob('*-alias-resolution') if (p/'alias_resolution_summary.json').is_file()),key=lambda p:p.name)
    if not candidates:raise FileNotFoundError('No D-03c v3 alias-resolution run; pass --input-run')
    return candidates[-1]


def validate_v3(run):
    summary=json.loads((run/'alias_resolution_summary.json').read_text(encoding='utf-8'))
    required=['python_pair_evidence_v3.csv','pair_provenance_decisions.csv']
    for name in required:
        p=run/name
        if not p.is_file():raise FileNotFoundError(p)
        claimed=summary.get('output_sha256',{}).get(name)
        if not claimed or sha256(p)!=claimed:raise RuntimeError(f'Input integrity mismatch: {p}')
    recorded=summary.get('approved_c10_sha256',summary.get('python_sha256'))
    if recorded!=APPROVED:
        raise RuntimeError('v3 recorded a different Python C-10 corpus')
    return summary


def parse_type(row):
    # Only explicit per-pair type labels are eligible for classification.
    for col in ('clone_type','cloneType','clone_type_label','type'):
        raw=(row.get(col) or '').strip().lower().replace('_','-').replace(' ','')
        if raw in TYPES:return TYPES[raw],f'explicit:{col}={row[col]}'
    # Strong semantic claims (T4) are NOT inferred from an unvalidated `kind`.
    for col in ('clone_kind','kind','match_kind'):
        raw=(row.get(col) or '').strip().lower()
        if raw in KINDS:return KINDS[raw],f'explicit_detector_kind:{col}={row[col]}'
    return None,''


def load_hybrid(root,report_path,cache):
    if report_path in cache:return cache[report_path]
    relative=Path(report_path)
    if relative.is_absolute():
        raise RuntimeError(f'Absolute evidence path not permitted: {report_path}')
    if '..' in relative.parts:raise RuntimeError(f'Path traversal in evidence path: {report_path}')
    p=(root/relative).resolve()
    if root not in p.parents:raise RuntimeError(f'Evidence path outside project: {report_path}')
    if not p.is_file():
        cache[report_path]=None
        return None
    lookup={}
    for r in csv_rows(p):
        k=r.get('pair_id','')
        if k:
            if k in lookup:raise RuntimeError(f'Duplicate detector pair id: {p} {k}')
            lookup[k]=r
    cache[report_path]=lookup
    return lookup


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--input-run',type=Path,help='completed D-03c v3 alias-resolution run; newest auto-selected')
    ap.add_argument('--run-dir',type=Path,required=True)
    args=ap.parse_args()
    root=args.root.resolve();inp=find_v3(root,args.input_run);out=args.run_dir.resolve()
    if out==inp or out in inp.parents or inp in out.parents:raise RuntimeError('Input/output overlap')
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Nonempty output: {out}')
    pairs=dataset_check(root);previous=validate_v3(inp)
    decisions={}
    for row in csv_rows(inp/'pair_provenance_decisions.csv'):
        k=tuple(sorted([row['id_a'],row['id_b']]))
        if k not in pairs or pairs[k]!=(row['split'],int(row['label'])):
            raise RuntimeError(f'v3 decision not in C-10: {k}')
        if k in decisions:raise RuntimeError(f'Duplicate v3 decision {k}')
        decisions[k]=row
    grouped=defaultdict(list);cache={};raw_rows=[];counts=Counter()
    for ev in csv_rows(inp/'python_pair_evidence_v3.csv'):
        k=tuple(sorted([ev['id_a'],ev['id_b']]))
        if k not in decisions:raise RuntimeError(f'Evidence for unrecognized pair: {k}')
        path=ev['source'];lookup=load_hybrid(root,path,cache)
        matched=None
        if lookup is not None:
            matched=lookup.get(ev['source_pair_id'])
        if lookup is None: status='ORIGINAL_REPORT_MISSING'
        elif matched is None:status='REPORT_PAIR_ID_NOT_FOUND'
        elif matched.get('source','')!=ev['detector_source']:status='REPORT_DETECTOR_SOURCE_MISMATCH'
        else:status='DETECTOR_SOURCE_MATCHED'
        # Even source matched is NOT a type label.
        inferred,note=parse_type(matched) if matched is not None and status=='DETECTOR_SOURCE_MATCHED' else (None,'')
        mode=(ev['detector_source'] or 'UNKNOWN').strip()
        if mode=='nicad_primary':interpret='NICAD_BLIND_0.30_SUPPORTS_T1_T2_T3_NOT_DISAMBIGUATED'
        elif mode=='jscpd_ast_fallback':interpret='JSCPD_FALLBACK_DETECTED_TYPE_NOT_RECORDED'
        else:interpret='UNKNOWN_DETECTION_MODE'
        raw_rows.append({'id_a':k[0],'id_b':k[1],'split':ev['split'],
                         'canonical_project':ev['canonical_project'],'raw_project':ev['project'],
                         'source_pair_id':ev['source_pair_id'],'source':path,
                         'detector_source':mode,'report_kind':note,'reported_clone_type':inferred or '',
                         'reported_kind':(matched or {}).get('kind','') if matched else '',
                         'mode_interpretation':interpret,'matching_status':status})
        grouped[k].append((ev,mode,status,inferred,note))
        counts[status]+=1
    if not counts.get('DETECTOR_SOURCE_MATCHED',0):
        raise RuntimeError('No original detector pair reports matched; verify results/c-01-batch and results/b-07 on the server')
    output=[];type_counts=Counter();status_counts=Counter();covered=Counter()
    for key,(split,label) in pairs.items():
        if label!=1:continue
        d=decisions.get(key,{})
        entries=grouped.get(key,[])
        modes=sorted({e[1] for e in entries})
        # Strict per-pair types only. Mixed or non-explicit detector evidence remains unresolved.
        explicit={e[3] for e in entries if e[3] and e[2]=='DETECTOR_SOURCE_MATCHED'}
        has_invalid=any(e[2]!='DETECTOR_SOURCE_MATCHED' for e in entries)
        if len(explicit)==1 and not has_invalid:
            clone_type=next(iter(explicit));state='EXPLICIT_DETECTOR_TYPE_NEEDS_METHOD_REVIEW'
        elif len(explicit)>1:clone_type='UNRESOLVED';state='CONFLICTING_EXPLICIT_TYPES'
        else:clone_type='UNRESOLVED';state='NO_DISAMBIGUATING_TYPE_IN_DETECTOR_OUTPUT'
        if not entries:state='NO_ORIGINAL_PAIR_EVIDENCE'
        # NiCad blind+30% admits T1,T2,T3; jscpd fallback has no defensible exact subtype.
        possible='TYPE_1|TYPE_2|TYPE_3' if 'nicad_primary' in modes else 'UNRESOLVED'
        project=d.get('canonical_project','UNKNOWN')
        if d.get('project_status')=='CROSS_PROJECT_CONFLICT_REQUIRES_REVIEW':project='CONFLICT'
        output.append({'id_a':key[0],'id_b':key[1],'split':split,'label':label,
                       'canonical_project':project,'project_status':d.get('project_status','NO_PROJECT_EVIDENCE'),
                       'clone_type':clone_type,'clone_type_status':state,
                       'detector_category':'|'.join(modes) if modes else 'NONE',
                       'detector_modes':'|'.join(modes) if modes else '',
                       'possible_types':possible,'evidence_rows':len(entries),
                       'report_files':len({e[0]['source'] for e in entries}),
                       'evidence_details':'|'.join(sorted({e[4] for e in entries if e[4]}))})
        type_counts[clone_type]+=1;status_counts[state]+=1
        if project not in ('UNKNOWN','CONFLICT'):covered[project]+=1
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'python_detector_clone_types.csv',FIELDS,output)
    write_csv(out/'original_detector_evidence.csv',RAW_FIELDS,raw_rows)
    summary={'task':'D-03c v4 detector-based clone-type extraction',
             'status':'EVIDENCE_AUDIT_COMPLETE_NO_TRAINING',
             'timestamp_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
             'input_run':str(inp),'input_v3_summary_sha256':sha256(inp/'alias_resolution_summary.json'),
             'input_c10_sha256':APPROVED,'positive_pairs':len(output),'all_c10_pairs':len(pairs),
             'clone_type_counts':dict(type_counts),'type_status_counts':dict(status_counts),
             'report_match_status_counts':dict(counts),'project_count_with_evidence':len(covered),
             'source_hybrid_files':len(cache),
             'limitations':['Original C-01 hybrid CSV source is a detector category, not a definitive T1/T2/T3/T4 label.',
                            'NiCad blind 0.30 can cover T1/T2/T3 without disambiguating individual pairs.',
                            'jscpd_ast_fallback is a routing category; it does not prove a type.',
                            'No semantic equivalence Type-4 inferred from NiCad/jscpd detection.',
                            'Original report missing means source evidence cannot be reconfirmed.',
                            'Project provenance and negative-label uncertainty are not resolved by type extraction.',
                            'This is not training or approval for type-stratified sampling.'],
             'outputs_sha256':{f:sha256(out/f) for f in ['python_detector_clone_types.csv','original_detector_evidence.csv']}}
    (out/'detector_type_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'[D-03c v4] C-10 VERIFIED positive_pairs={len(output)} original_hybrid_files={len(cache)}')
    print(f'[D-03c v4] clone_type_counts={dict(type_counts)} evidence={dict(counts)}')
    print(f'[D-03c v4] output={out} (NO TRAINING)',flush=True)


if __name__=='__main__':main()
