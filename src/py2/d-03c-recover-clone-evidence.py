#!/usr/bin/env python3
"""D-03c v6: restore C-10 pair similarities from C-01 NiCad / jscpd CSV.

Use pair-level source functions, not XML fragment line positions. C-01 results
are preferred, B-07 results are cross-check / fallback. No detector or training.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

APPROVED = {
    'data.jsonl': 'da8fdd6362df692e225a5bf5b72278fd6211cef75e48e6a539dda637b76a1843',
    'train.txt': 'e8c8998f2e73a3587190904fb258ac1ccda306f0be62dec499622f4258603693',
    'valid.txt': '33c5a0506fdbbd8ff5f9604a9be1d4fba1b6d6d52943ff11ff53928861ac48a7',
}
# C-10 valid.txt checksum correction enforced below to preserve approved baseline.
APPROVED['valid.txt'] = '33c5a0506fdbbd8ff5f9604a9be1d4fba1b6b6d52943ff11ff53928861ac48a7'

AUDIT = ['id_a','id_b','split','canonical_project','provenance','source_pair_id','detector',
         'hybrid_report','detector_csv','similarity_raw','similarity_percent','match_status',
         'match_rule','candidate_count','source_pair_file_a','source_pair_start_a','source_pair_end_a',
         'source_pair_file_b','source_pair_start_b','source_pair_end_b']
OUTPUT = ['id_a','id_b','split','label','canonical_project','project_status','detector',
          'similarity_percent','clone_type_category','category_status','selected_provenance',
          'matched_evidence_rows','matched_c01_rows','matched_b07_rows',
          'similarity_values_percent','matching_rules','source_csv_files']
PROJECT = ['canonical_project','positive_pairs','typed_pairs','type_1_or_2','type_3_candidate',
           'unresolved','nicad_typed','jscpd_typed']


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with path.open('r',newline='',encoding='utf-8-sig') as f:
        yield from csv.DictReader(f)


def write_csv(path,fields,entries):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(entries)


def safe_path(root,relative):
    p=Path(str(relative))
    if p.is_absolute() or '..' in p.parts:raise RuntimeError('Unsafe project-relative input path: '+str(p))
    abs_path=(root/p).resolve()
    if abs_path!=root and root not in abs_path.parents:raise RuntimeError('Path escapes project root: '+str(p))
    return abs_path


def latest(root):
    options=sorted(p for p in (root/'results/d-03c').glob('*-alias-resolution')
                   if (p/'alias_resolution_summary.json').is_file())
    if not options:raise FileNotFoundError('D-03c v3 alias-resolution run missing')
    return options[-1]


def validate_v3(inp):
    info=json.loads((inp/'alias_resolution_summary.json').read_text(encoding='utf-8'))
    if info.get('approved_c10_sha256')!=APPROVED:raise RuntimeError('D-03c v3 C-10 hashes differ')
    for name in ('pair_provenance_decisions.csv','python_pair_evidence_v3.csv'):
        expect=info.get('output_sha256',{}).get(name)
        if not expect or sha256(inp/name)!=expect:raise RuntimeError('D-03c v3 evidence checksum mismatch: '+name)
    return info


def validate_c10(root):
    base=root/'data/training/clean/python_mixed'
    for name,expect in APPROVED.items():
        if not (base/name).is_file() or sha256(base/name)!=expect:
            raise RuntimeError('Approved C-10 checksum mismatch: '+str(base/name))
    pairs={}
    for split in ('train','valid'):
        for line in (base/(split+'.txt')).read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            a,b,label=line.split();key=tuple(sorted((a,b)))
            if key in pairs:raise RuntimeError('Duplicated C-10 pair: '+str(key))
            pairs[key]=(split,int(label))
    if len(pairs)!=27180 or sum(label==1 for split,label in pairs.values())!=13596:
        raise RuntimeError('Unexpected C-10 approved pair totals')
    return pairs


def normal_file(value):
    s=str(value or '').replace('\\','/')
    parts=[]
    for x in s.split('/'):
        if x in ('','.'):continue
        if x=='..':raise ValueError('Unsafe file name in report')
        parts.append(x)
    return '/'.join(parts)


def function_key(row,side):
    return (normal_file(row['file_'+side]),int(row['start_'+side]),int(row['end_'+side]))


def pair_key(row):
    return tuple(sorted((function_key(row,'a'),function_key(row,'b'))))


def candidate_keys(row):
    """Exact boundary first; one-line boundary tolerance only as an audited fallback.

    Original C-01 merge creates identical function keys, so tolerance is normally
    unnecessary. Endline offset never substitutes for an ambiguous exact match.
    """
    a=function_key(row,'a');b=function_key(row,'b')
    exact=tuple(sorted((a,b)))
    offsets=set()
    for d1 in (-1,0,1):
        for d2 in (-1,0,1):
            if d1==d2==0:continue
            x=(a[0],a[1],a[2]+d1);y=(b[0],b[1],b[2]+d2)
            if x[2]<x[1] or y[2]<y[1]:continue
            offsets.add(tuple(sorted((x,y))))
    return exact,offsets


def as_percent(raw,detector):
    try:x=Decimal(str(raw).strip())
    except (InvalidOperation,ValueError):raise ValueError('Invalid similarity: '+str(raw))
    if not x.is_finite():raise ValueError('Nonfinite similarity')
    if detector=='nicad_primary':p=x
    elif detector=='jscpd_ast_fallback':p=x*100
    else:raise ValueError('Unexpected detector: '+detector)
    if not Decimal('70')<=p<=Decimal('100'):
        raise ValueError('Similarity outside configured 70-100 range: '+str(raw))
    return p


def fmt(value):
    return format(value.normalize(),'f')


def provenance(report):
    return 'C01' if report.startswith('results/c-01-batch/') else ('B07' if report.startswith('results/b-07/') else 'UNKNOWN')


def csv_for_hybrid(root,hybrid,mode):
    filename={'nicad_primary':'nicad_primary_pairs.csv','jscpd_ast_fallback':'jscpd_fallback_pairs.csv'}.get(mode)
    if not filename:return None,None
    raw=str((Path(hybrid).parent/filename).as_posix())
    return raw,safe_path(root,raw)


def load_hybrid(root,path,cache):
    if path not in cache:
        loc=safe_path(root,path)
        if not loc.is_file():cache[path]=None
        else:
            entries={}
            for row in read_csv(loc):
                k=row.get('pair_id','')
                if not k or k in entries:raise RuntimeError('Invalid / duplicate hybrid pair id in '+path)
                entries[k]=row
            cache[path]=entries
    return cache[path]


def load_detector(root,raw,cache):
    if raw not in cache:
        path=safe_path(root,raw)
        if not path.is_file():cache[raw]=None
        else:
            direct=defaultdict(list)
            for r in read_csv(path):
                direct[pair_key(r)].append(r)
            cache[raw]=direct
    return cache[raw]


def recover_one(root,ev,hybrid_cache,detector_cache):
    raw=ev.get('source','');mode=ev.get('detector_source','');pid=ev.get('source_pair_id','')
    p=provenance(raw)
    if p=='UNKNOWN':return None,'UNKNOWN_REPORT_SOURCE','','',0
    reports=load_hybrid(root,raw,hybrid_cache)
    if reports is None:return None,'HYBRID_REPORT_MISSING','','',0
    pair=reports.get(pid)
    if pair is None:return None,'HYBRID_PAIR_MISSING','','',0
    if pair.get('source')!=mode:return None,'DETECTOR_SOURCE_MISMATCH','','',0
    csv_rel,csv_path=csv_for_hybrid(root,raw,mode)
    if csv_path is None:return None,'UNKNOWN_DETECTOR','','',0
    lookup=load_detector(root,csv_rel,detector_cache)
    if lookup is None:return None,'DETECTOR_CSV_MISSING',csv_rel,'',0
    exact,offset=candidate_keys(pair)
    matches=lookup.get(exact,[])
    rule='EXACT_FUNCTION_PAIR'
    if not matches:
        matches=[r for k in offset for r in lookup.get(k,[])]
        rule='ENDLINE_OFFSET_AT_MOST_ONE'
    if not matches:return None,'DETECTOR_PAIR_NOT_FOUND',csv_rel,rule,0
    # Multiple CSV records for a single function pair are possible; accept
    # duplicates only when they have identical score and hence no ambiguity.
    scores=[]
    for r in matches:
        try:scores.append(as_percent(r.get('similarity',''),mode))
        except ValueError:return None,'INVALID_SIMILARITY',csv_rel,rule,len(matches)
    if len(set(scores))!=1:return None,'CONFLICTING_DETECTOR_SCORES',csv_rel,rule,len(matches)
    return scores[0],'MATCHED',csv_rel,rule,len(matches)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--input-run',type=Path,help='Explicit D-03c v3 alias-resolution run')
    ap.add_argument('--run-dir',type=Path,required=True)
    args=ap.parse_args()
    root=args.root.resolve();inp=args.input_run.resolve() if args.input_run else latest(root)
    out=args.run_dir.resolve()
    if out==inp or inp in out.parents or out in inp.parents:raise RuntimeError('Input/output overlap')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Refusing to overwrite a nonempty run directory')
    v3=validate_v3(inp);approved_pairs=validate_c10(root)
    decisions={}
    for row in read_csv(inp/'pair_provenance_decisions.csv'):
        k=tuple(sorted((row['id_a'],row['id_b'])))
        if approved_pairs.get(k)!=(row['split'],int(row['label'])):raise RuntimeError('v3 decision differs from approved C-10')
        if k in decisions:raise RuntimeError('Duplicate v3 pair decision')
        decisions[k]=row
    evidence=defaultdict(list)
    for row in read_csv(inp/'python_pair_evidence_v3.csv'):
        k=tuple(sorted((row['id_a'],row['id_b'])))
        if k not in decisions:raise RuntimeError('v3 evidence without corresponding decision')
        evidence[k].append(row)
    hybrid_cache={};detector_cache={};rows=[];audits=[];project_counts=defaultdict(Counter)
    status=Counter();cats=Counter();detectors=Counter()
    for key,(split,label) in sorted(approved_pairs.items()):
        if label!=1:continue
        decision=decisions.get(key,{})
        canonical=decision.get('canonical_project','UNKNOWN') or 'UNKNOWN'
        pstatus=decision.get('project_status','NO_PROJECT_EVIDENCE')
        if pstatus=='CROSS_PROJECT_CONFLICT_REQUIRES_REVIEW':canonical='CONFLICT'
        found=[];rules=set();source_files=set()
        for ev in evidence.get(key,[]):
            raw=ev.get('source','');mode=ev.get('detector_source','');pid=ev.get('source_pair_id','')
            score,match,csv_rel,rule,n=recover_one(root,ev,hybrid_cache,detector_cache)
            audit={'id_a':key[0],'id_b':key[1],'split':split,'canonical_project':canonical,
                   'provenance':provenance(raw),'source_pair_id':pid,'detector':mode,
                   'hybrid_report':raw,'detector_csv':csv_rel,'similarity_raw':'',
                   'similarity_percent':fmt(score) if score is not None else '',
                   'match_status':match,'match_rule':rule,'candidate_count':n,
                   'source_pair_file_a':'','source_pair_start_a':'','source_pair_end_a':'',
                   'source_pair_file_b':'','source_pair_start_b':'','source_pair_end_b':''}
            if raw in hybrid_cache and hybrid_cache[raw] and pid in hybrid_cache[raw]:
                h=hybrid_cache[raw][pid]
                for f in ('file_a','start_a','end_a','file_b','start_b','end_b'):
                    audit['source_pair_'+f]=h.get(f,'')
            audits.append(audit);status[match]+=1
            if score is not None:
                found.append((provenance(raw),mode,score,csv_rel));rules.add(rule);source_files.add(csv_rel)
        selected=[];state='';category='UNRESOLVED';score_value='';detector='NONE';origin='NONE'
        # Explicitly conflicting project assignments must never be used as
        # a training-ready typed record, even if they have a similarity.
        if pstatus=='CROSS_PROJECT_CONFLICT_REQUIRES_REVIEW':state='CROSS_PROJECT_CONFLICT'
        elif not found:state='NO_MATCHED_SIMILARITY'
        else:
            c01=[x for x in found if x[0]=='C01'];b07=[x for x in found if x[0]=='B07']
            selected=c01 if c01 else b07
            # If independent reports disagree, abstain rather than silently
            # choose one detector/score. Duplicate provenance is deduplicated.
            scores={x[2] for x in found};modes={x[1] for x in found}
            if len(scores)!=1:state='CROSS_REPORT_SCORE_CONFLICT'
            elif len(modes)!=1:state='CROSS_REPORT_DETECTOR_CONFLICT'
            else:
                score=next(iter(scores));mode=next(iter(modes));score_value=fmt(score)
                origin='C01' if c01 else 'B07';detector=mode
                category='TYPE_1_OR_2' if score==100 else 'TYPE_3_CANDIDATE'
                state='MATCHED_'+origin
        record={'id_a':key[0],'id_b':key[1],'split':split,'label':1,
                'canonical_project':canonical,'project_status':pstatus,'detector':detector,
                'similarity_percent':score_value,'clone_type_category':category,
                'category_status':state,'selected_provenance':origin,
                'matched_evidence_rows':len(found),
                'matched_c01_rows':sum(x[0]=='C01' for x in found),
                'matched_b07_rows':sum(x[0]=='B07' for x in found),
                'similarity_values_percent':'|'.join(sorted({fmt(x[2]) for x in found})),
                'matching_rules':'|'.join(sorted(rules)),
                'source_csv_files':'|'.join(sorted(source_files))}
        rows.append(record);cats[category]+=1;detectors[detector]+=1
        pc=project_counts[canonical];pc['positive_pairs']+=1
        if category!='UNRESOLVED':
            pc['typed_pairs']+=1;pc[category]+=1;pc[detector+'_typed']+=1
        else:pc['unresolved']+=1
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'python_detector_clone_types_v6.csv',OUTPUT,rows)
    write_csv(out/'detector_similarity_match_audit.csv',AUDIT,audits)
    projects=[]
    for name,counts in sorted(project_counts.items()):
        projects.append({'canonical_project':name,'positive_pairs':counts['positive_pairs'],
            'typed_pairs':counts['typed_pairs'],'type_1_or_2':counts['TYPE_1_OR_2'],
            'type_3_candidate':counts['TYPE_3_CANDIDATE'],'unresolved':counts['unresolved'],
            'nicad_typed':counts['nicad_primary_typed'],'jscpd_typed':counts['jscpd_ast_fallback_typed']})
    write_csv(out/'project_similarity_coverage.csv',PROJECT,projects)
    summary={'task':'D-03c v6 C-01/B-07 unified detector similarity recovery',
      'status':'DETECTOR_EVIDENCE_AUDIT_NO_TRAINING',
      'timestamp_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
      'approved_c10_sha256':APPROVED,'input_v3_run':str(inp),
      'input_v3_summary_sha256':sha256(inp/'alias_resolution_summary.json'),
      'approved_pairs':len(approved_pairs),'positive_pairs':len(rows),
      'project_count_with_positive_evidence':len(project_counts),
      'category_counts':dict(cats),'typed_detector_counts':dict(detectors),
      'audit_match_status_counts':dict(status),
      'hybrid_reports_examined':len(hybrid_cache),
      'detector_csvs_examined':len(detector_cache),
      'methodology':{'nicad_similarity_scale':'70..100',
         'jscpd_similarity_scale':'0.70..1.00, multiplied by 100',
         'operational_assumption':'Detector similarity values are treated as a common threshold scale; internal algorithms need not match.',
         'similarity_100':'TYPE_1_OR_2 candidate (not formally separated)',
         'similarity_70_to_below_100':'TYPE_3_CANDIDATE (not verified formal Type-3)',
         'type_4':'Not inferred without independent semantic evidence',
         'source_priority':'C-01 primary, B-07 cross-check or fallback',
         'join':'Same-run hybrid pair id -> same-run detector CSV, exact function key (file,start,end), both sides order-independent; audited +/-1 endline fallback only',
         'conflict':'Cross-run score / detector disagreement and cross-project conflicts remain UNRESOLVED'},
      'limitations':['All categories are detector-based operational categories, not verified clone ground truth.',
          'C-10 negative candidate labels and train/held-out leakage are not validated by this script.',
          'Java Target provenance and classifier-based Type-4 detection are outside scope.',
          'No model training, pair sampling, or prior result replacement.'],
      'output_sha256':{name:sha256(out/name) for name in ('python_detector_clone_types_v6.csv','detector_similarity_match_audit.csv','project_similarity_coverage.csv')}}
    (out/'unified_similarity_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'[D-03c v6] C-10 VERIFIED positives={len(rows)} detector_csvs={len(detector_cache)} hybrid_reports={len(hybrid_cache)}')
    print(f'[D-03c v6] categories={dict(cats)} audit_matches={dict(status)}')
    print(f'[D-03c v6] output={out} (NO TRAINING)',flush=True)

if __name__=='__main__':main()
