#!/usr/bin/env python3
"""C-11a v2: immutable final quality-gate audit of C-10 candidate, not an approval."""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v2'
SPLITS = ('train', 'valid')
ROOT = Path(__file__).resolve().parents[2]
C10_DEFAULT = 'results/c-10/20261106-220451-402850'
C09_DEFAULT = 'results/c-09/20261106-213122-749426'
C04_DEFAULT = 'results/c-04/20261103-193519-475848'
OUTPUT_FIELDS = ['audit.json', 'label_integrity.csv', 'split_leakage.csv', 'heldout_leakage.csv',
                 'project_distribution.csv', 'negative_reliability.json', 'training_decision.json']

class AuditError(ValueError):
    pass

def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def file_required(path):
    if not path.is_file(): raise AuditError(f'Missing input: {path}')
    return path

def canonical(a,b):
    if not a or not b or a==b: raise AuditError('Missing/self-pair')
    return tuple(sorted((a,b)))

def load_csv(path, required):
    file_required(path)
    with path.open(newline='',encoding='utf-8-sig') as f:
        rd=csv.DictReader(f)
        if not rd.fieldnames or not set(required).issubset(rd.fieldnames):
            raise AuditError(f'Invalid columns {path}: {sorted(set(required)-set(rd.fieldnames or []))}')
        return list(rd)

def load_corpus(path):
    out={}
    with file_required(path).open(encoding='utf-8-sig') as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            try:
                obj=json.loads(line); idx=obj['idx']; code=obj['func']
            except (ValueError,KeyError,TypeError) as e: raise AuditError(f'Invalid corpus JSONL line {n}: {e}') from e
            if not isinstance(idx,str) or not idx or not isinstance(code,str) or not code.strip() or idx in out:
                raise AuditError(f'Invalid/duplicate function record {path}:{n}')
            out[idx]=code
    return out

def load_pairs(path,corpus):
    out={}
    with file_required(path).open(encoding='utf-8-sig') as f:
        for n,line in enumerate(f,1):
            if not line.strip():continue
            row=line.split()
            if len(row)!=3 or row[2] not in ('0','1'):
                raise AuditError(f'Invalid 3-column pair {path}:{n}')
            a,b,label=row; p=canonical(a,b)
            if a not in corpus or b not in corpus or p in out:
                raise AuditError(f'Dangling function or duplicate pair {path}:{n}')
            out[p]=int(label)
    return out

def key(row):
    if not row.get('project') or row.get('split') not in SPLITS:
        raise AuditError(f'Invalid project/split: {row}')
    return row['project'],row['split'],*canonical(row['id_a'],row['id_b'])

def read_index(rows,name):
    out={}
    for row in rows:
        k=key(row)
        if k in out: raise AuditError(f'Duplicate {name} pair {k}')
        out[k]=row
    return out

def code_hash(code):
    return hashlib.sha256(code.encode('utf-8')).hexdigest()

def normalized_hash(code):
    return code_hash(''.join(code.split()))

def read_heldout(path):
    """Read the A-08/C-05 held-out schema (id, code).

    Also accept the training corpus schema (idx, func) for backward
    compatibility. Do not infer other keys: an unknown schema must fail
    explicitly rather than producing a false clean leakage audit.
    """
    out = []
    seen_ids = set()
    schema = None
    with file_required(path).open(encoding='utf-8-sig') as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise AuditError(f'Malformed heldout JSONL {path}:{n}') from e
            if not isinstance(obj, dict):
                raise AuditError(f'Unrecognized heldout schema {path}:{n}; expected JSON object')
            if isinstance(obj.get('code'), str) and isinstance(obj.get('id'), (str, int)):
                this_schema = 'id/code'
                ident, code = str(obj['id']), obj['code']
            elif isinstance(obj.get('func'), str) and isinstance(obj.get('idx'), (str, int)):
                this_schema = 'idx/func'
                ident, code = str(obj['idx']), obj['func']
            else:
                raise AuditError(
                    f'Unrecognized heldout schema {path}:{n}; expected '
                    'id + code (A-08/C-05) or idx + func. '
                    f'Available keys: {sorted(obj.keys())}'
                )
            if schema is not None and this_schema != schema:
                raise AuditError(f'Mixed heldout schemas in {path}:{n}: {schema} vs {this_schema}')
            schema = this_schema
            if not ident.strip() or not code.strip() or ident in seen_ids:
                raise AuditError(f'Empty/duplicate heldout ID or empty code {path}:{n}')
            seen_ids.add(ident)
            out.append((ident, code))
    if not out:
        raise AuditError(f'Empty heldout corpus: {path}')
    return out

def csv_save(path, cols, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');wr.writeheader();wr.writerows(rows)

def checks(c10,c09,c04,heldouts):
    data=c10/'python_mixed_screened_candidate'
    corpus=load_corpus(data/'data.jsonl')
    pairs={s:load_pairs(data/f'{s}.txt',corpus) for s in SPLITS}
    all_pairs={(s,p):y for s, ps in pairs.items() for p,y in ps.items()}
    counts={s:Counter(ps.values()) for s,ps in pairs.items()}
    if set(pairs['train']) & set(pairs['valid']):
        raise AuditError('Identical pair reused across splits')
    c10_manifest=json.loads(file_required(c10/'manifest.json').read_text())
    c10_audit=json.loads(file_required(c10/'audit.json').read_text())
    if c10_manifest.get('training_approval') is not False or c10_audit.get('training_approval') is not False:
        raise AuditError('Unexpected upstream training approval state')
    source_hashes=c10_manifest.get('output_sha256',{})
    for rel,h in source_hashes.items():
        actual=sha_file(file_required(c10/rel))
        if actual!=h: raise AuditError(f'C10 manifest SHA256 mismatch: {rel}')
    for s in SPLITS:
        expected=c10_audit.get('corrected_counts',{}).get(s,{})
        if (counts[s][1],counts[s][0])!=(expected.get('positive'),expected.get('negative')):
            raise AuditError(f'C10 audit count mismatch for {s}')
    c10_neg=read_index(load_csv(data/'negative_candidates.csv', ['project','split','id_a','id_b','label','verification','c09_status','training_approved']),'C10 negatives')
    extra_neg=read_index(load_csv(c10/'negative_label_provenance.csv',['project','split','id_a','id_b','label','verification','c09_status','training_approved']),'C10 provenance')
    if c10_neg!=extra_neg: raise AuditError('C10 negative provenance mismatch')
    c09_idx=read_index(load_csv(c09/'negative_candidate_correction_ledger.csv',['project','split','id_a','id_b','c09_status']),'C09 ledger')
    c04_corpus=load_corpus(c04/'data.jsonl')
    original={s:load_pairs(c04/f'{s}.txt',c04_corpus) for s in SPLITS}
    orig_neg=read_index(load_csv(c04/'negative_candidates.csv',['project','split','id_a','id_b','label']),'C04 negatives')
    if set(c09_idx)!=set(orig_neg): raise AuditError('C09 and C04 negative population mismatch')
    if corpus!=c04_corpus: raise AuditError('Function corpus differs from C04 source')
    for s in SPLITS:
        source_pos={p for p,y in original[s].items() if y==1}
        actual_pos={p for p,y in pairs[s].items() if y==1}
        if source_pos != actual_pos: raise AuditError(f'Positive pair set changed in {s}')
    # C04 positive provenance is independently checked when present. Its absence is
    # a non-silent audit limitation; never synthesize source identity or commits.
    positive_provenance_status='NOT_VERIFIED'
    provenance_path=c04/'positive_provenance.csv'
    if provenance_path.is_file():
        provenance=load_csv(provenance_path,['project','id_a','id_b','c04_split'])
        supported={(s,p) for s in SPLITS for p,y in original[s].items() if y==1}
        observed=set()
        for row in provenance:
            s=row['c04_split'].strip(); p=canonical(row['id_a'],row['id_b'])
            if not row['project'].strip() or (s,p) not in supported:
                raise AuditError(f'Invalid C04 positive provenance entry: {s}/{p}')
            observed.add((s,p))
        if observed != supported:
            raise AuditError(f'C04 positive provenance coverage: missing {len(supported-observed)}')
        positive_provenance_status='PASS'
    expected_neg=set()
    clone_excluded=set()
    uncertain_excluded=set()
    statuses=Counter()
    for k,r in c09_idx.items():
        status=r['c09_status'].strip();statuses[status]+=1
        _,s,a,b=k
        if (a,b) not in original[s] or original[s][(a,b)]!=0: raise AuditError(f'C09 pair absent from C04 negatives {k}')
        if status=='EXCLUDE_CONFIRMED_CLONE': clone_excluded.add(k)
        elif status=='UNCERTAIN_REVIEW': uncertain_excluded.add(k)
        elif status in ('REVIEWED_NONCLONE','UNREVIEWED'):expected_neg.add(k)
        else:raise AuditError(f'Unrecognized C09 status {status}')
    if set(c10_neg)!=expected_neg: raise AuditError('C10 candidate set differs from C09 retained set')
    for k,r in c10_neg.items():
        _,s,a,b=k
        if all_pairs.get((s,(a,b)))!=0:raise AuditError(f'C10 negative missing/wrong label {k}')
        expect='HUMAN_REVIEWED_NONCLONE' if c09_idx[k]['c09_status']=='REVIEWED_NONCLONE' else 'SCREENING_ONLY_UNVERIFIED'
        if r['label']!='0' or r['verification']!=expect or r['c09_status']!=c09_idx[k]['c09_status'] or r['training_approved'].lower()!='false':
            raise AuditError(f'C10 provenance status/label mismatch {k}')
    for k in clone_excluded|uncertain_excluded:
        _,s,a,b=k
        if (a,b) in pairs[s]: raise AuditError(f'Excluded conflict still in C10 pair file {k}')
    if len(c10_neg)!=sum(counts[s][0] for s in SPLITS):raise AuditError('Negative pair count mismatch')
    if sum(counts[s][1] for s in SPLITS)!=13596:raise AuditError('Unexpected total positive pair count')
    ids={s:{x for pair in pairs[s] for x in pair} for s in SPLITS}
    # Cross-split, recheck independent of C10 audit.
    exact={s:{code_hash(corpus[i]) for i in ids[s]} for s in SPLITS}
    norm={s:{normalized_hash(corpus[i]) for i in ids[s]} for s in SPLITS}
    overlaps={'function_id':ids['train']&ids['valid'],
              'exact_code_hash':exact['train']&exact['valid'],
              'normalized_code_hash':norm['train']&norm['valid']}
    if any(overlaps.values()): raise AuditError('Train/validation leakage detected: '+str({k:len(v) for k,v in overlaps.items()}))
    split_rows=[{'check':k,'overlap_count':len(v),'status':'PASS','note':'Independent full-pair audit'} for k,v in overlaps.items()]
    split_rows.append({'check':'project_disjointness','overlap_count':'NOT_ASSESSED','status':'NOT_ESTABLISHED',
                       'note':'Project-level generalization is a different split design; project memberships require evaluation provenance'})
    label_rows=[]
    for s in SPLITS:
        for y in (1,0):
            label_rows.append({'split':s,'label':y,'pairs':counts[s][y], 'status':'PASS',
                               'note':'positive matches C04' if y else 'negative reconciled with C09'})
    label_rows += [{'split':'all','label':'excluded_confirmed_clone','pairs':len(clone_excluded),'status':'PASS','note':'C09 manual conflict exclusions'},
                   {'split':'all','label':'excluded_uncertain','pairs':len(uncertain_excluded),'status':'PASS','note':'C09 uncertain exclusions'}]
    proj=defaultdict(Counter)
    for k,r in c10_neg.items():
        proj[(k[0],k[1])][r['verification']]+=1
    for k in clone_excluded:proj[(k[0],k[1])]['excluded_clones']+=1
    project_rows=[]
    for (project,s),c in sorted(proj.items()):
        project_rows.append({'project':project,'split':s,'reviewed_nonclone':c['HUMAN_REVIEWED_NONCLONE'],
            'screening_unverified':c['SCREENING_ONLY_UNVERIFIED'],'excluded_clones':c['excluded_clones']})
    heldout_rows=[];heldout_detail=[]; heldout_status='NOT_VERIFIED'
    all_hash={v for sub in exact.values() for v in sub}; all_norm={v for sub in norm.values() for v in sub}
    provided={name:path for name,path in heldouts.items() if path is not None}
    for name in ('java','python'):
        p=heldouts.get(name)
        if p is None or not p.is_file():
            heldout_rows.append({'heldout':name,'records':'','exact_overlap':'','normalized_overlap':'',
                                 'status':'NOT_VERIFIED','path':str(p) if p else '',
                                 'note':'Corpus absent; no silent PASS'})
            continue
        rows=read_heldout(p)
        ex=[i for i,code in rows if code_hash(code) in all_hash]
        nm=[i for i,code in rows if normalized_hash(code) in all_norm]
        heldout_rows.append({'heldout':name,'records':len(rows),'exact_overlap':len(ex),
                             'normalized_overlap':len(nm),'status':'FAIL' if ex or nm else 'PASS',
                             'path':str(p),'note':'Exact/whitespace-normalized hashing, NOT near-duplicate'})
    if all(r['status']!='NOT_VERIFIED' for r in heldout_rows):
        heldout_status='FAIL' if any(r['status']=='FAIL' for r in heldout_rows) else 'PASS'
    negative_total=len(c10_neg)
    reliability={'candidate_total':negative_total,'human_reviewed_nonclone':statuses['REVIEWED_NONCLONE'],
        'screening_only_unverified':statuses['UNREVIEWED'], 'confirmed_clone_excluded':len(clone_excluded),
        'uncertain_excluded':len(uncertain_excluded),
        'unverified_fraction':round(statuses['UNREVIEWED']/negative_total,8) if negative_total else 0,
        'sampling':'All high-priority screened, plus project-stratified nonprobability review sample',
        'manual_review_caveat':'Blank C07 adjudications treated as nonclone by user-declared policy; reviewer metadata incomplete',
        'limitations':['Screened candidates are not confirmed nonclones','12/274 is not the population error rate',
                       'Near-duplicate leakage and project-level heldout independence remain unverified'],
        'training_approval':False}
    blockers=[]
    if heldout_status!='PASS':blockers.append('HELDOUT_'+heldout_status)
    if positive_provenance_status!='PASS':blockers.append('POSITIVE_PROVENANCE_NOT_VERIFIED')
    blockers.extend(['NEAR_DUPLICATE_LEAKAGE_NOT_ESTABLISHED','PROJECT_LEVEL_TEST_INDEPENDENCE_NOT_ESTABLISHED',
                     'UNVERIFIED_NEGATIVE_LABELS_REMAIN'])
    decision={'status':'AUDIT_PASSED_REVIEW_REQUIRED' if heldout_status!='FAIL' else 'AUDIT_FAILED_HELDOUT_OVERLAP',
              'training_approval':False,'controlled_training_approval':False,
              'c11a_only':True,'approval_policy':'C11b is a separate, explicit decision',
              'blockers':blockers}
    audit={'version':VERSION,'status':decision['status'],
           'total_pairs':sum(sum(v.values()) for v in counts.values()),
           'label_counts':{s:{'positive':counts[s][1],'negative':counts[s][0]} for s in SPLITS},
           'positive_preservation':'PASS','positive_provenance':positive_provenance_status,'negative_pair_reconciliation':'PASS',
           'upstream_manifest_sha256':'PASS','excluded_confirmed_clones':len(clone_excluded),
           'split_leakage':{k:len(v) for k,v in overlaps.items()},'heldout_leakage':heldout_rows,
           'near_duplicate_test_leakage':'NOT_ASSESSED','project_level_test_leakage':'NOT_ESTABLISHED',
           'training_approval':False,'warnings':reliability['limitations']}
    return audit,label_rows,split_rows,heldout_rows,project_rows,reliability,decision

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--c10',default=C10_DEFAULT)
    ap.add_argument('--c09',default=C09_DEFAULT)
    ap.add_argument('--c04',default=C04_DEFAULT)
    ap.add_argument('--heldout-java',default='data/evaluation/code/java.jsonl')
    ap.add_argument('--heldout-python',default='data/evaluation/code/python.jsonl')
    ap.add_argument('--output',default=None)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args(argv)
    root=args.root.resolve()
    resolve=lambda x: (root/Path(x)).resolve() if not Path(x).is_absolute() else Path(x).resolve()
    c10,c09,c04=map(resolve,(args.c10,args.c09,args.c04))
    heldouts={'java':resolve(args.heldout_java) if args.heldout_java else None,
              'python':resolve(args.heldout_python) if args.heldout_python else None}
    audit,labels,splits,heldout,projects,reliability,decision=checks(c10,c09,c04,heldouts)
    print('C-11a',json.dumps({'status':audit['status'],'total_pairs':audit['total_pairs'],
       'label_counts':audit['label_counts'],'excluded_clones':audit['excluded_confirmed_clones'],
       'heldout_statuses':{r['heldout']:r['status'] for r in heldout},
       'training_approval':False},sort_keys=True))
    if args.dry_run:
        print('DRY RUN: no output created');return 0
    out=resolve(args.output) if args.output else root/'results/c-11a'/datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
    if out.exists():raise AuditError(f'Refusing to overwrite output: {out}')
    out.mkdir(parents=True)
    json_out={'audit.json':audit,'negative_reliability.json':reliability,'training_decision.json':decision}
    for name,value in json_out.items():
        (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    csv_save(out/'label_integrity.csv',['split','label','pairs','status','note'],labels)
    csv_save(out/'split_leakage.csv',['check','overlap_count','status','note'],splits)
    csv_save(out/'heldout_leakage.csv',['heldout','records','exact_overlap','normalized_overlap','status','path','note'],heldout)
    csv_save(out/'project_distribution.csv',['project','split','reviewed_nonclone','screening_unverified','excluded_clones'],projects)
    inputs=[c10/'audit.json',c10/'manifest.json',c10/'negative_label_provenance.csv',
        c10/'python_mixed_screened_candidate'/'data.jsonl',c10/'python_mixed_screened_candidate'/'train.txt',
        c10/'python_mixed_screened_candidate'/'valid.txt',
        c10/'python_mixed_screened_candidate'/'negative_candidates.csv',
        c09/'negative_candidate_correction_ledger.csv', c04/'data.jsonl',c04/'train.txt',c04/'valid.txt',c04/'negative_candidates.csv']
    if (c04/'positive_provenance.csv').is_file(): inputs.append(c04/'positive_provenance.csv')
    inputs.extend(p for p in heldouts.values() if p and p.is_file())
    manifest={'version':VERSION,'created_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
        'inputs_sha256':{str(p):sha_file(p) for p in inputs},
        'output_sha256':{name:sha_file(out/name) for name in OUTPUT_FIELDS},
        'training_approval':False,'immutable_inputs':True,'approval_decision_deferred_to':'C-11b'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('OUTPUT:',out)
    return 0

if __name__=='__main__':
    try: sys.exit(main())
    except (AuditError,OSError,ValueError) as e:
        print('C-11a ERROR:',e,file=sys.stderr);sys.exit(2)
