#!/usr/bin/env python3
"""C-06 v1: deterministic risk-stratified review queue; never approves negative labels."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

C05_DEFAULT='results/c-05/20261103-135539-677672'
C04_DEFAULT='results/c-04/20261103-193519-475848'
COLUMNS=['split','id_a','id_b','project','screening','flags','token_jaccard','sequence_similarity','length_ratio','ast_shape_equal','exact_code_equal','normalized_code_equal','same_positive_component','known_positive_pair']
HIGH_FLAGS={'HIGH_TOKEN_JACCARD','HIGH_TOKEN_SEQUENCE_SIMILARITY','SAME_NORMALIZED_AST_SHAPE','AST_PARSE_UNAVAILABLE'}
REVIEW_FIELDS=['reviewer','adjudication','confidence','review_notes','reviewed_at_chicago']

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load_csv(path):
    with path.open(newline='',encoding='utf-8-sig') as f:
        rdr=csv.DictReader(f)
        if not set(COLUMNS).issubset(set(rdr.fieldnames or [])):
            raise ValueError('C-05 CSV missing required columns')
        return list(rdr)

def group_of(row):
    flags={x for x in row['flags'].split(';') if x}  # C-05 delimiter checked below
    if row['screening']=='SCREENED_UNVERIFIED':
        if flags:raise ValueError('SCREENED_UNVERIFIED row has flags')
        return 'SCREENED_BASELINE'
    if row['screening']!='REVIEW_RISK':raise ValueError('Unsupported screening: '+row['screening'])
    if not flags:raise ValueError('REVIEW_RISK row missing flags')
    if flags & HIGH_FLAGS:return 'HIGH_PRIORITY_SIMILARITY_AST'
    if flags == {'LENGTH_IMBALANCE'}:return 'LENGTH_ONLY_REVIEW'
    return 'OTHER_REVIEW_FLAGS'

def rank(row):
    # Stable across servers without depending on input row ordering.
    key='|'.join(row[k] for k in ('project','split','id_a','id_b'))
    return hashlib.sha256(key.encode()).hexdigest()

def select(rows,per_group,high_cap):
    grouped=defaultdict(list)
    for row in rows:grouped[(row['project'],group_of(row))].append(row)
    selected=[]
    for (project,group),rr in sorted(grouped.items()):
        rr.sort(key=rank)
        if group=='HIGH_PRIORITY_SIMILARITY_AST':
            take=rr if high_cap==0 else rr[:high_cap]
        elif group=='OTHER_REVIEW_FLAGS':
            take=rr if high_cap==0 else rr[:high_cap]
        else:
            # Split-aware deterministic quotas when both splits exist.
            bysplit=defaultdict(list)
            for row in rr:bysplit[row['split']].append(row)
            splits=sorted(bysplit)
            if len(splits)==2 and per_group>=2:
                each={splits[0]:per_group//2,splits[1]:per_group//2}
                if per_group%2:each[splits[0]]+=1
                take=[]
                for split in splits:take.extend(bysplit[split][:each[split]])
                missing=per_group-len(take)
                if missing:
                    chosen={id(x) for x in take}
                    leftovers=sorted((x for x in rr if id(x) not in chosen),key=rank)
                    take.extend(leftovers[:missing])
            else:take=rr[:per_group]
        selected.extend(take)
    selected.sort(key=lambda x:(x['project'],group_of(x),x['split'],rank(x)))
    return selected

def parse_args(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--c05',type=Path,default=Path(C05_DEFAULT))
    ap.add_argument('--c04',type=Path,default=Path(C04_DEFAULT))
    ap.add_argument('--out',type=Path,default=None)
    ap.add_argument('--per-project-group',type=int,default=5)
    ap.add_argument('--high-priority-cap',type=int,default=0,help='0 = include all high-priority candidates')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args(argv)
    if args.per_project_group<1 or args.high_priority_cap<0:ap.error('invalid sample sizes')
    return args

def resolve(root,p):return (root/p).resolve() if not p.is_absolute() else p.resolve()

def run(argv=None):
    args=parse_args(argv);root=args.root.resolve()
    c05=resolve(root,args.c05);c04=resolve(root,args.c04)
    for p in (c05/'negative_verification.csv',c05/'audit.json',c05/'manifest.json',c04/'data.jsonl'):
        if not p.is_file():raise ValueError('Missing required input: '+str(p))
    a=json.loads((c05/'audit.json').read_text());m=json.loads((c05/'manifest.json').read_text())
    if a.get('status')!='SCREENING_COMPLETE_REVIEW_REQUIRED' or m.get('training_approval') is not False or m.get('negative_labels_verified') is not False:
        raise ValueError('C-05 status/approval not eligible for C-06 audit')
    expected=m.get('input_sha256',{}).get('data.jsonl')
    if not expected or digest(c04/'data.jsonl')!=expected:
        raise ValueError('C-04 data.jsonl SHA256 mismatch vs C-05 manifest')
    rr=load_csv(c05/'negative_verification.csv')
    if len(rr)!=a['counts']['total_negative_candidates']:raise ValueError('C-05 candidate count mismatch')
    screening=Counter(r['screening'] for r in rr)
    if dict(screening)!=a['screening']:raise ValueError('C-05 screening totals mismatch')
    seen=set()
    flag_counts=Counter()
    for r in rr:
        key=tuple(r[k] for k in ('split','id_a','id_b'))
        if key in seen:raise ValueError('Duplicate candidate pair')
        seen.add(key)
        flags=[x for x in r['flags'].split(';') if x]
        flag_counts.update(flags)
        group_of(r)
    if dict(flag_counts)!=a['flag_counts']:raise ValueError('C-05 flag counts mismatch; inspect CSV delimiter')
    queue=select(rr,args.per_project_group,args.high_priority_cap)
    tiers=Counter(group_of(r) for r in rr)
    chosen=Counter(group_of(r) for r in queue)
    summary={'version':'v1','status':'REVIEW_QUEUE_PREPARED_NOT_ADJUDICATED','training_approval':False,
             'negative_labels_verified':False,'source_c05':str(c05),'source_c04':str(c04),
             'total_candidates':len(rr),'screening':dict(screening),'candidate_groups':dict(sorted(tiers.items())),
             'queue_groups':dict(sorted(chosen.items())),'queue_total':len(queue),
             'projects':len({r['project'] for r in rr}),'per_project_group':args.per_project_group,
             'high_priority_cap':args.high_priority_cap,
             'limitations':['Queue membership does not determine non-clone ground truth.',
               'Length imbalance is not evidence that a pair is a clone.',
               'Screened candidates are still unverified.',
               'Human review or independent benchmark is required for training approval.']}
    print('C-06 '+json.dumps({k:summary[k] for k in ('status','total_candidates','candidate_groups','queue_groups','queue_total','projects')},sort_keys=True))
    if args.dry_run:
        print('DRY RUN: no output created')
        return
    if args.out is None:
        stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
        out=root/'results'/'c-06'/stamp
    else:out=resolve(root,args.out)
    if out.exists():raise FileExistsError('Output already exists (preservation policy): '+str(out))
    # Load only source code selected for review; C-04 is authoritative by SHA.
    selected_ids={r[k] for r in queue for k in ('id_a','id_b')}
    corpus={}
    with (c04/'data.jsonl').open(encoding='utf-8-sig') as f:
        for n,line in enumerate(f,1):
            if not line.strip():continue
            o=json.loads(line);idx=o['idx'];code=o['func']
            if idx in selected_ids:
                if idx in corpus or not isinstance(code,str):raise ValueError('Duplicate or invalid C-04 function: '+str(idx))
                corpus[idx]=code
    if set(corpus)!=selected_ids:raise ValueError('Missing selected function IDs in C-04 data.jsonl')
    out.mkdir(parents=True,exist_ok=False)
    fields=COLUMNS+['review_group','review_id']
    with (out/'review_queue.csv').open('w',newline='',encoding='utf-8') as fcsv, (out/'adjudication_template.csv').open('w',newline='',encoding='utf-8') as ftemplate, (out/'review_queue_code.jsonl').open('w',encoding='utf-8') as fcode:
        writer=csv.DictWriter(fcsv,fieldnames=fields);writer.writeheader()
        twriter=csv.DictWriter(ftemplate,fieldnames=['review_id','project','split','id_a','id_b','review_group']+REVIEW_FIELDS);twriter.writeheader()
        for r in queue:
            group=group_of(r);review_id=hashlib.sha256('|'.join(r[k] for k in ('project','split','id_a','id_b')).encode()).hexdigest()[:20]
            writer.writerow({**r,'review_group':group,'review_id':review_id})
            twriter.writerow({'review_id':review_id,'project':r['project'],'split':r['split'],'id_a':r['id_a'],'id_b':r['id_b'],'review_group':group,**{x:'' for x in REVIEW_FIELDS}})
            fcode.write(json.dumps({'review_id':review_id,'project':r['project'],'split':r['split'],'id_a':r['id_a'],'id_b':r['id_b'],'review_group':group,'code_a':corpus[r['id_a']],'code_b':corpus[r['id_b']]},ensure_ascii=False)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    manifest={'version':'v1','timestamp_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
              'timezone':'America/Chicago','status':summary['status'],'training_approval':False,
              'c05_manifest_sha256':digest(c05/'manifest.json'),'c05_audit_sha256':digest(c05/'audit.json'),
              'c05_negative_verification_sha256':digest(c05/'negative_verification.csv'),
              'c04_data_sha256':digest(c04/'data.jsonl'),
              'output_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('OUTPUT:',out)
    print('NOTE: review_queue_code.jsonl contains repository source code. Do not publish automatically.')

if __name__=='__main__':
    try:run()
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
        print('C-06 ERROR:',exc,file=sys.stderr);sys.exit(2)
