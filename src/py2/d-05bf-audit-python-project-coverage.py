#!/usr/bin/env python3
"""D-05bf v4: exhaustive small-project coverage audit; REPORT ONLY, no gold labels."""
import argparse, collections, csv, hashlib, itertools, json, re, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v4'
TOKENS = re.compile(r'[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\s]', re.UNICODE)

def sha_bytes(value): return hashlib.sha256(value).hexdigest()
def sha_file(path): return sha_bytes(path.read_bytes())
def objrows(path):
    with path.open(encoding='utf-8') as inp:
        for line in inp:
            if line.strip(): yield json.loads(line)
def newest(root, glob):
    paths=sorted(p for p in (root/'results/d-05b').glob(glob) if p.is_dir())
    if not paths: raise ValueError('Missing input directory: '+glob)
    return paths[-1]
def file_from_manifest(directory, filename):
    manifest_path=directory/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    entry=(manifest.get('files') or {}).get(filename)
    expected=entry.get('sha256') if isinstance(entry,dict) else entry
    if not expected: expected=(manifest.get('artifacts_sha256') or {}).get(filename)
    path=directory/filename
    if not path.is_file() or not expected or sha_file(path)!=expected:
        raise ValueError('Missing or SHA256 mismatch: '+str(path))
    return path,sha_file(manifest_path)
def grams(code):
    toks=TOKENS.findall(code)
    if not toks:return frozenset()
    n=min(5,len(toks))
    return frozenset(hashlib.blake2b('\0'.join(toks[i:i+n]).encode(),digest_size=8).digest() for i in range(len(toks)-n+1))
def score(a,b):
    return len(a&b)/len(a|b) if a and b else 0.0
def pairkey(a,b):return tuple(sorted((a,b)))
def bin_name(value):
    if not .10<=value<.50:return None
    idx=min(7,int((value-.10+1e-12)/.05))
    return f'{.10+idx*.05:.2f}-{.15+idx*.05:.2f}'
def code_from_repository(root,project,side,cache):
    rp=side['repository_file'];base=(root/'data/repositories').resolve();path=(root/rp).resolve()
    if not path.is_relative_to(base) or path.relative_to(base).parts[0]!=project or not path.is_file():
        raise ValueError('Invalid or missing source file '+rp)
    if path not in cache:
        data=path.read_bytes();cache[path]=(sha_bytes(data),data.decode('utf-8').splitlines(keepends=True))
    digest,lines=cache[path]
    if digest!=side['repository_file_sha256']:raise ValueError('Source file changed '+rp)
    i,j=int(side['start_line']),int(side['end_line'])
    if not (1<=i<=j<=len(lines)):raise ValueError('Invalid source range '+rp)
    code=''.join(lines[i-1:j])
    if sha_bytes(code.encode())!=side['code_sha256']:raise ValueError('Code SHA mismatch '+rp)
    return code

def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--heldout',type=Path)
    p.add_argument('--full-c01-audit',type=Path);p.add_argument('--annotation-source',type=Path,
        help='Optional D-05bd v1 directory with source-verified code; otherwise read frozen repositories')
    p.add_argument('--negative-v3',type=Path,help='Optional D-05bf v3 results directory for project coverage cross-check')
    p.add_argument('--project',default='paramiko');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--max-functions',type=int,default=500,help='Safety ceiling for exhaustive combinations')
    args=p.parse_args();root=args.root.resolve(); project=args.project
    held=(args.heldout or newest(root,'python-heldout-c01-only-v9-*')).resolve()
    full=(args.full_c01_audit or newest(root,'python-source-leakage-audit-v6-*')).resolve()
    held_file,held_mf=file_from_manifest(held,'heldout_candidates.jsonl')
    full_file,full_mf=file_from_manifest(full,'candidates_source_audited.jsonl')
    nodes={};held_pairs=0
    for row in objrows(held_file):
        if row.get('project')!=project:continue
        held_pairs+=1
        if row.get('origin')!='c01_hybrid_detector' or row.get('gold_label') is not None:
            raise ValueError('C-01 only / unlabeled invariant violated')
        if row.get('code_hash_train_overlap_status')!='NO_MATCH_AGAINST_M2_M3_ACTIVE_PYTHON_TRAIN_VALID_HASHES':
            raise ValueError('Missing training exclusion verification')
        for key in ('left','right'):
            s=row['v6_source_resolution'][key]
            if s.get('status')=='EXACT_AST_FUNCTION':
                h=s['code_sha256']
                if h in nodes and nodes[h]['normalized_code_sha256']!=s['normalized_code_sha256']:
                    raise ValueError('Inconsistent source hash metadata')
                nodes.setdefault(h,s)
    if not held_pairs:raise ValueError('No heldout C-01 positives for project '+project)
    if len(nodes)>args.max_functions:raise ValueError('Exhaustive pool size exceeds --max-functions')
    full_c01=set()
    for row in objrows(full_file):
        if row.get('origin')!='c01_hybrid_detector':continue
        s=row.get('v6_source_resolution') or {};a=s.get('left') or {};b=s.get('right') or {}
        if not a.get('code_sha256') or not b.get('code_sha256'):raise ValueError('Unresolved full C-01 pair')
        full_c01.add(pairkey(a['code_sha256'],b['code_sha256']))
    annotation_mf=None;source_codes={}
    if args.annotation_source:
        path,annotation_mf=file_from_manifest(args.annotation_source.resolve(),'annotation_packets.jsonl')
        for row in objrows(path):
            if row.get('project')!=project:continue
            for side in ('left','right'):
                obj=row[side];h=obj['code_sha256'];code=obj['code']
                if sha_bytes(code.encode('utf-8'))!=h:raise ValueError('Annotation source SHA mismatch')
                if h in source_codes and source_codes[h]!=code:raise ValueError('Conflicting code for hash')
                source_codes[h]=code
    source_cache={};fingerprints={}
    for h,s in nodes.items():
        c=source_codes.get(h) if args.annotation_source else code_from_repository(root,project,s,source_cache)
        if c is None:raise ValueError('Missing confirmed source code for '+h)
        if sha_bytes(c.encode())!=h:raise ValueError('Source SHA mismatch '+h)
        fingerprints[h]=grams(c)
    rows=[];counts=collections.Counter();bins=collections.Counter();max_unblocked=0.0
    for ha,hb in itertools.combinations(sorted(nodes),2):
        a,b=nodes[ha],nodes[hb];sim=score(fingerprints[ha],fingerprints[hb]);bi=bin_name(sim)
        if pairkey(ha,hb) in full_c01:status='BLOCKED_C01_DETECTED_PAIR'
        elif a['normalized_code_sha256']==b['normalized_code_sha256']:status='BLOCKED_IDENTICAL_NORMALIZED_FUNCTION'
        elif bi is None:status='OUTSIDE_SIMILARITY_RANGE'
        else:status='ELIGIBLE_NEGATIVE_ANNOTATION_CANDIDATE';bins[bi]+=1
        counts[status]+=1
        if status!='BLOCKED_C01_DETECTED_PAIR':max_unblocked=max(max_unblocked,sim)
        rows.append({'project':project,'left_code_sha256':ha,'right_code_sha256':hb,
            'left_file':a['repository_file'],'left_start_line':a['start_line'],
            'right_file':b['repository_file'],'right_start_line':b['start_line'],
            'similarity':round(sim,10),'similarity_bin':bi or '',
            'status':status,'gold_label':'','annotation_status':'pending'})
    v3_project_count=None;v3_mf=None
    if args.negative_v3:
        f,v3_mf=file_from_manifest(args.negative_v3.resolve(),'negative_annotation_packets.jsonl')
        v3_project_count=sum(1 for x in objrows(f) if x.get('project')==project)
    summary={'version':VERSION,'project':project,'status':'DRY_RUN' if args.dry_run else 'D05BF_V4_EXHAUSTIVE_PROJECT_COVERAGE_AUDIT_ONLY',
             'source_policy':'C01_ONLY_HELDOUT_FUNCTION_POOL','similarity_metric':'token_shingle_jaccard',
             'similarity_shingle_n':5,'similarity_range':'[0.10,0.50)',
             'heldout_positive_pairs':held_pairs,'eligible_unique_function_hashes':len(nodes),
             'possible_pair_combinations':len(rows),'candidate_counts_by_status':dict(sorted(counts.items())),
             'eligible_new_negative_candidates':counts['ELIGIBLE_NEGATIVE_ANNOTATION_CANDIDATE'],
             'eligible_similarity_bin_counts':dict(sorted(bins.items())),
             'max_similarity_among_unblocked_pairs':round(max_unblocked,10),
             'negative_v3_project_count':v3_project_count,'gold_labels_created':0,
             'input_manifest_sha256':{'heldout_v9':held_mf,'full_c01_v6':full_mf,'annotation_source_v1':annotation_mf,'negative_v3':v3_mf},
             'limitations':['Existing C01-only heldout function pool only; NOT a complete repository-wide non-clone audit.',
                 'A non-detected C-01 pair is not automatically a negative gold label.',
                 'Expanding the function pool requires active train/validation identity leakage filtering.',
                 'All candidate labels remain pending; no existing data files are modified.']}
    if args.dry_run:print(json.dumps(summary,sort_keys=True));return
    stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
    out=root/'results/d-05b'/('python-project-coverage-audit-v4-'+project+'-'+stamp)
    out.mkdir(parents=True,exist_ok=False)
    path=out/'exhaustive_pair_similarity.csv'
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['project','status']);w.writeheader();w.writerows(rows)
    path2=out/'project_coverage_audit.json'
    summary['created_at_chicago']=datetime.now(ZoneInfo('America/Chicago')).isoformat()
    path2.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({'version':VERSION,'created_at_chicago':summary['created_at_chicago'],
        'files':{name:{'bytes':(out/name).stat().st_size,'sha256':sha_file(out/name)} for name in (path.name,path2.name)},
        'input_manifest_sha256':summary['input_manifest_sha256']},indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({**summary,'out':str(out)},sort_keys=True))

if __name__=='__main__':
    try:run()
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print('D-05bf v4 ERROR: '+str(exc),file=sys.stderr);sys.exit(2)
