#!/usr/bin/env python3
"""D-05bf v3: adaptive reviewer-ready NEGATIVE CANDIDATES, never automatic gold labels.
Sources: active C-01-only heldout v9, active C-01 full v6 and confirmed positives.
No Clone Detection executed; legacy data prohibited.
"""
import argparse
import collections
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v3'
TOKEN_PATTERN = re.compile(r'[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\s]', re.UNICODE)
BIN_EDGES = [(round(0.10+i*0.05, 2), round(0.15+i*0.05, 2)) for i in range(8)]
SEP = '\u0000'

def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def hashtext(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def shingles(code, n=5):
    tokens=TOKEN_PATTERN.findall(code)
    if not tokens:return frozenset()
    k=min(n,len(tokens))
    return frozenset(hashlib.blake2b('\0'.join(tokens[i:i+k]).encode('utf-8'),digest_size=8).digest()
                     for i in range(len(tokens)-k+1))

def similarity(sa,sb):
    if not sa or not sb:return 0.0
    if min(len(sa),len(sb))*2 < max(len(sa),len(sb))*0.2:
        return 0.0  # only used to reject pairs guaranteed below 0.10
    return len(sa & sb)/len(sa | sb)

def bin_index(score):
    if not (0.10<=score<0.50):return None
    return min(7,int((score-0.10+1e-12)/0.05))

def bin_key(i):
    a,b=BIN_EDGES[i]
    return f'{a:.2f}-{b:.2f}'

def apportioned(total, weights):
    """Largest remainder allocation; tie order is ascending bin index."""
    exact=[total*w/sum(weights) for w in weights]
    base=[int(x) for x in exact]
    rem=total-sum(base)
    for i in sorted(range(len(weights)),key=lambda i:(-(exact[i]-base[i]),i))[:rem]:base[i]+=1
    return base

def key(a, b):
    return tuple(sorted((a, b)))

def rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def newest(root, pattern):
    matches = sorted((root/'results/d-05b').glob(pattern))
    matches = [p for p in matches if p.is_dir()]
    if not matches: raise ValueError('Missing input: '+pattern)
    return matches[-1]

def checked_input(directory, data_name):
    mf = directory/'manifest.json'
    if not mf.is_file(): raise ValueError('Missing manifest: '+str(mf))
    manifest = json.loads(mf.read_text(encoding='utf-8'))
    data = directory/data_name
    if not data.is_file(): raise ValueError('Missing data: '+str(data))
    files = manifest.get('files') or {}
    expected = files.get(data_name)
    if isinstance(expected, dict): expected = expected.get('sha256')
    if not expected:
        expected = (manifest.get('artifacts_sha256') or {}).get(data_name)
    if not expected or expected != checksum(data):
        raise ValueError('Input manifest SHA256 mismatch: '+str(data))
    return data,checksum(mf)

def relative_path(root, val):
    p=(root/val).resolve()
    base=(root/'data/repositories').resolve()
    if not p.is_relative_to(base) or not p.is_file() or p.suffix != '.py':
        raise ValueError('Invalid candidate source path: '+str(val))
    return p

def read_code(root, side, project, cache):
    rp=side['repository_file']
    path=relative_path(root,rp)
    if path.relative_to((root/'data/repositories').resolve()).parts[0] != project:
        raise ValueError('Project/path mismatch '+project+' '+str(path))
    if path not in cache:
        b=path.read_bytes()
        cache[path]=(hashlib.sha256(b).hexdigest(),b.decode('utf-8').splitlines(keepends=True))
    file_hash, lines=cache[path]
    if file_hash != side['repository_file_sha256']:
        raise ValueError('Source file has changed: '+rp)
    a=int(side['start_line']);b=int(side['end_line'])
    if not (1<=a<=b<=len(lines)):raise ValueError('Source line boundaries changed: '+rp)
    code=''.join(lines[a-1:b])
    if hashtext(code)!=side['code_sha256']:
        raise ValueError('Function source SHA mismatch: '+rp)
    return code

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--heldout',type=Path,help='D-05bb v9 output directory')
    p.add_argument('--full-c01-audit',type=Path,help='D-05ba v6 output directory')
    p.add_argument('--positives',type=Path,help='D-05be v2 output directory')
    p.add_argument('--ratio',type=int,default=1,help='Negative annotation candidates per confirmed positive (default 1)')
    p.add_argument('--seed',type=int,default=20261108)
    p.add_argument('--bin-weights',default='12.5,12.5,12.5,12.5,12.5,12.5,12.5,12.5',
                   help='Eight percentages for bins 0.10-0.15 ... 0.45-0.50; must sum to 100')
    p.add_argument('--max-attempts-per-project',type=int,default=250000,
                   help='Maximum candidate pair comparisons per project (default: 250000)')
    p.add_argument('--allow-shortfall',action='store_true',
                   help='Export partial candidates when bin targets cannot be filled; full target is default strict')
    p.add_argument('--pool-factor',type=int,default=4,help='Maximum collected proposals per project relative to project quota (default 4)')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.ratio<1 or args.ratio>3:p.error('--ratio must be between 1 and 3')
    try:weights=[float(x) for x in args.bin_weights.split(',')]
    except ValueError:p.error('Invalid --bin-weights: expected eight numeric percentages')
    if len(weights)!=8 or any(x<0 for x in weights) or abs(sum(weights)-100)>1e-7:
        p.error('--bin-weights requires eight nonnegative percentages summing to 100')
    if args.max_attempts_per_project<1:p.error('--max-attempts-per-project must be positive')
    if not 1<=args.pool_factor<=20:p.error('--pool-factor must be in [1,20]')
    root=args.root.resolve()
    if not (root/'data/repositories').is_dir():p.error('Missing active repositories')
    held=(args.heldout or newest(root,'python-heldout-c01-only-v9-*')).resolve()
    full=(args.full_c01_audit or newest(root,'python-source-leakage-audit-v6-*')).resolve()
    pos=(args.positives or newest(root,'python-confirmed-positive-v2-*')).resolve()
    held_data,held_mf=checked_input(held,'heldout_candidates.jsonl')
    full_data,full_mf=checked_input(full,'candidates_source_audited.jsonl')
    pos_data,pos_mf=checked_input(pos,'reviewer_confirmed_positive_pairs.jsonl')
    # Full C-01 detected pair blocking including training/validation-overlapping detections.
    c01_all=set();all_origin=collections.Counter()
    for row in rows(full_data):
        all_origin[row.get('origin')]+=1
        if row.get('origin') != 'c01_hybrid_detector':continue
        sides=row.get('v6_source_resolution') or {}
        a=sides.get('left') or {};b=sides.get('right') or {}
        if not a.get('code_sha256') or not b.get('code_sha256'):
            raise ValueError('Unresolved C-01 source in v6 full audit')
        c01_all.add(key(a['code_sha256'],b['code_sha256']))
    if len(c01_all)<1:raise ValueError('No verified C-01 pair keys')
    # V9 is already train/validation-filtered. Build source pool *only* from v9 C-01 sides.
    anchors=collections.defaultdict(dict);positive_ids=collections.defaultdict(set)
    positives_by_project=collections.Counter();held_ids=set();held_pairhashes=set()
    for row in rows(held_data):
        if row.get('origin')!='c01_hybrid_detector' or row.get('gold_label') is not None:
            raise ValueError('V9 input violates C01_ONLY/unlabeled invariant')
        if row.get('code_hash_train_overlap_status') != 'NO_MATCH_AGAINST_M2_M3_ACTIVE_PYTHON_TRAIN_VALID_HASHES':
            raise ValueError('Unverified train/validation hash filtering')
        project=row['project'];pairid=row['pair_id']
        if pairid in held_ids:raise ValueError('Duplicate heldout pair ID')
        held_ids.add(pairid);held_pairhashes.add(row['v6_code_pair_sha256'])
        sides=row['v6_source_resolution']
        for s in ('left','right'):
            obj=sides[s]
            if obj['status'] not in ('EXACT_AST_FUNCTION','EXACT_EVALUATION_CORPUS_ID'):
                raise ValueError('Unverified source identity')
            if obj['status']=='EXACT_AST_FUNCTION':
                anchors[project].setdefault(obj['code_sha256'],obj)
            else:
                # Corpus-ID-only sources may lack a stable repository file: not eligible as negative source.
                continue
    poskeys=set();pos_ids=set()
    for row in rows(pos_data):
        if row.get('gold_label')!=1 or row.get('label_source')!='C01_DETECTED_REVIEWER_A_B_CONFIRMED':
            raise ValueError('Positive confirmation provenance/label invalid')
        if row.get('pair_id') not in held_ids:raise ValueError('Confirmed positive ID not found in heldout v9')
        if row['pair_id'] in pos_ids:raise ValueError('Duplicate confirmed positive ID')
        pos_ids.add(row['pair_id']);positives_by_project[row['project']]+=1
        poskeys.add(key(row['left_code_sha256'],row['right_code_sha256']))
    if pos_ids!=held_ids:raise ValueError('Confirmed positives do not cover entire v9 heldout set')
    if len(positives_by_project)!=24:raise ValueError('Expected 24 active projects in confirmed positives')
    active={x.name for x in (root/'data/repositories').iterdir() if x.is_dir() and (x/'.git').exists()}
    if 'azure-sdk-for-python' in active or active!=set(positives_by_project):
        raise ValueError('Active repositories differ from confirmed 24-project population')
    if not poskeys.issubset(c01_all):
        raise ValueError('Positive pair missing from full C-01 detection blocklist')
    # D-05bf v3: build candidate pools FIRST, then adaptively allocate globally.
    # Pair similarity is between the proposed two functions (NOT train leakage).
    full_target=sum(positives_by_project.values())*args.ratio
    global_bin_target=apportioned(full_target,weights)
    pools={};sample_stats={};source_cache={};rejected=collections.Counter()
    for project in sorted(positives_by_project):
        nodes=sorted(anchors[project].values(),key=lambda r:(r['code_sha256'],r['repository_file'],r['start_line']))
        quota=positives_by_project[project]*args.ratio
        if len(nodes)<2:raise ValueError('Insufficient code functions in '+project)
        rng=random.Random(args.seed ^ int(hashtext(project)[:16],16))
        grams=[];postings=collections.defaultdict(list)
        for i,node in enumerate(nodes):
            ss=shingles(read_code(root,node,project,source_cache))
            grams.append(ss)
            for gram in ss:postings[gram].append(i)
        gram_choices=[]
        for i,ss in enumerate(grams):
            # Cap large ubiquitous postings and give less-common shingles some chance.
            eligible=sorted((g for g in ss if len(postings[g])>1),key=lambda g:(len(postings[g]),g))
            gram_choices.append(eligible)
        collected=[];seen=set();bin_found=collections.Counter();trials=0
        pool_limit=max(quota*args.pool_factor,quota+128)
        # Lower bound attempts permits sampling across bins, not merely first bin-filled hits.
        attempts_floor=min(args.max_attempts_per_project,max(3000,quota*12))
        while trials<args.max_attempts_per_project:
            if len(collected)>=pool_limit and trials>=attempts_floor:break
            trials+=1
            idx=rng.randrange(len(nodes));opts=gram_choices[idx]
            if not opts:
                rejected['no_shared_shingle']+=1;continue
            # Mix rare with common shingles: rare is informative; common broadens coverage.
            if rng.random()<0.7:
                g=opts[rng.randrange(min(len(opts),max(1,len(opts)//3)))]
            else:
                g=opts[rng.randrange(len(opts))]
            nbr=postings[g]
            if len(nbr)<2:continue
            j=nbr[rng.randrange(len(nbr))]
            if j==idx:continue
            a,b=nodes[idx],nodes[j]
            ha,hb=a['code_sha256'],b['code_sha256']
            if ha==hb or a['normalized_code_sha256']==b['normalized_code_sha256']:
                rejected['same_identity']+=1;continue
            pair=key(ha,hb)
            if pair in seen:continue
            seen.add(pair)
            if pair in c01_all or pair in poskeys:
                rejected['blocked_c01']+=1;continue
            score=similarity(grams[idx],grams[j]);bi=bin_index(score)
            if bi is None:
                rejected['outside_range']+=1;continue
            row={'pair_id':'negcand_'+hashtext('D05BFv3'+SEP+project+SEP+pair[0]+SEP+pair[1])[:24],
                'project':project,'language':'python','candidate_source':'C01_HELDOUT_FUNCTION_POOL_RECOMBINATION',
                'selection_method':'adaptive_seeded_shingle_neighbor_pool_and_global_quota_reallocation',
                'similarity_metric':'token_shingle_jaccard','similarity_shingle_n':5,
                'pair_similarity':round(score,8),'similarity_bin':bin_key(bi),
                'c01_detected_pair':False,'c01_full_detected_pair_blocklist_checked':True,
                'active_training_validation_exact_normalized_overlap':False,
                'left':a,'right':b,'pair_code_hashes':list(pair),
                'gold_label':None,'annotation_status':'pending','evaluation_eligible':False,
                'label_source':None,'negative_label_assumed':False}
            collected.append(row);bin_found[bin_key(bi)]+=1
        pools[project]=sorted(collected,key=lambda r:(r['similarity_bin'],r['pair_id']))
        sample_stats[project]={'confirmed_positives':positives_by_project[project],
            'negative_candidate_target':quota,'eligible_unique_function_hashes':len(nodes),
            'possible_combinations':len(nodes)*(len(nodes)-1)//2,
            'trials':trials,'unique_pairs_considered':len(seen),
            'candidate_pool_size':len(collected),'pool_limit':pool_limit,
            'pool_bin_counts':{bin_key(i):bin_found[bin_key(i)] for i in range(8)}}

    # Allocate using global weights as soft goals; first honor 1:1 project quota.
    selected=[];used_pairs=set();bin_used=collections.Counter();project_used=collections.Counter()
    for project in sorted(pools):
        quota=positives_by_project[project]*args.ratio
        per_bin=apportioned(quota,weights)
        groups={bin_key(i):[] for i in range(8)}
        for row in pools[project]:groups[row['similarity_bin']].append(row)
        # Stage A: the feasible part of each project's target-bin quota.
        for i in range(8):
            for row in groups[bin_key(i)][:per_bin[i]]:
                pair=tuple(row['pair_code_hashes'])
                if pair in used_pairs:continue
                selected.append(row);used_pairs.add(pair)
                project_used[project]+=1;bin_used[row['similarity_bin']]+=1
        # Stage B: fill PROJECT quota from any eligible similarity bin.
        left=[r for r in pools[project] if tuple(r['pair_code_hashes']) not in used_pairs]
        while project_used[project]<quota and left:
            # Weighted global bin deficit; never reject in-range candidate to force balance.
            next_bin=max((b for b in groups if any(r['similarity_bin']==b for r in left)),
                key=lambda b:(global_bin_target[int(round((float(b[:4])-0.10)/0.05))]-bin_used[b],-bin_used[b],b))
            row=next(r for r in left if r['similarity_bin']==next_bin)
            left.remove(row);pair=tuple(row['pair_code_hashes'])
            if pair in used_pairs:continue
            selected.append(row);used_pairs.add(pair)
            project_used[project]+=1;bin_used[row['similarity_bin']]+=1

    # Stage C: redistribute any shortfall to projects with additional valid candidates.
    redistributed=0
    while len(selected)<full_target:
        candidates=[]
        for project in sorted(pools):
            # Prioritize balanced project representation when surplus capacity exists.
            surplus=[r for r in pools[project] if tuple(r['pair_code_hashes']) not in used_pairs]
            if surplus:
                candidates.append((project,surplus))
        if not candidates:break
        project,available=min(candidates,key=lambda t:(
            project_used[t[0]]/max(1,positives_by_project[t[0]]*args.ratio),t[0]))
        row=max(available,key=lambda r:(
            global_bin_target[int(round((float(r['similarity_bin'][:4])-0.10)/0.05))]-bin_used[r['similarity_bin']],
            -bin_used[r['similarity_bin']],r['pair_id']))
        pair=tuple(row['pair_code_hashes'])
        selected.append(row);used_pairs.add(pair);project_used[project]+=1
        bin_used[row['similarity_bin']]+=1;redistributed+=1
    selected.sort(key=lambda r:(r['project'],r['pair_id']))
    overall=len(selected);shortfall=full_target-overall
    for project in sorted(sample_stats):
        stats=sample_stats[project]
        stats['negative_candidate_created']=project_used[project]
        stats['deviation_from_original_project_quota']=project_used[project]-stats['negative_candidate_target']
        stats['bin_achieved']={bin_key(i):sum(1 for r in selected if r['project']==project and r['similarity_bin']==bin_key(i)) for i in range(8)}
    achieved_percent={bin_key(i):round(bin_used[bin_key(i)]*100/overall,4) if overall else 0 for i in range(8)}
    summary={'status':'DRY_RUN' if args.dry_run else 'D05BF_V3_ADAPTIVE_NEGATIVE_CANDIDATES_NOT_GOLD',
             'source_policy':'C01_ONLY_FUNCTION_POOL_NO_LEGACY',
             'active_projects':24,'positive_reference_pairs':len(pos_ids),
             'negative_candidates':overall,'negative_candidate_target':full_target,
             'verified_negative_labels':0,'negative_labels_automatically_assigned':0,
             'full_c01_detected_pair_keys_blocked':len(c01_all),
             'similarity_metric':'token_shingle_jaccard','similarity_shingle_n':5,
             'similarity_range':'[0.10,0.50)',
             'bin_target_percentages':dict(zip(map(bin_key,range(8)),weights)),
             'bin_target_counts':{bin_key(i):global_bin_target[i] for i in range(8)},
             'bin_counts':{bin_key(i):bin_used[bin_key(i)] for i in range(8)},
             'bin_achieved_percentages':achieved_percent,
             'sampling_shortfall':shortfall,'allow_shortfall':args.allow_shortfall,
             'original_project_target_shortfall':sum(max(0,stats['negative_candidate_target']-stats['negative_candidate_created']) for stats in sample_stats.values()),
             'cross_project_quota_redistributed':redistributed,
             'total_candidate_pool':sum(len(v) for v in pools.values()),
             'sampling_ratio':args.ratio,'seed':args.seed,
             'max_attempts_per_project':args.max_attempts_per_project,
             'pool_factor':args.pool_factor,
             'annotation_rule':'BLANK_MEANS_PENDING_EXPLICIT_ZERO_REQUIRED',
             'input_sha256':{'heldout_v9_manifest':held_mf,'full_c01_v6_manifest':full_mf,'confirmed_positive_v2_manifest':pos_mf}}
    if shortfall and not args.allow_shortfall:
        raise ValueError(f'Adaptive sampling shortfall {shortfall}: {overall}/{full_target}; '
            'run --dry-run --allow-shortfall to inspect project/bin availability; '
            'increase --max-attempts-per-project or --pool-factor for broader exploration. '
            'No outputs were written and no gold labels were changed.')
    if args.dry_run:
        print(json.dumps(summary,sort_keys=True));return 0
    stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
    out=root/'results/d-05b'/('python-negative-candidates-v3-'+stamp)
    out.mkdir(parents=True,exist_ok=False)
    input_files={}
    for tag,dat in (('heldout_v9',held_data),('full_c01_v6',full_data),('positive_v2',pos_data)):
        input_files[tag]={'path':str(dat),'sha256':checksum(dat)}
    packets=out/'negative_annotation_packets.jsonl'
    with packets.open('w',encoding='utf-8') as f:
        for row in selected:
            packet=dict(row)
            for side in ('left','right'):
                item=packet[side]
                code=read_code(root,item,row['project'],source_cache)
                packet[side]={'path':item['repository_file'],'start_line':item['start_line'],
                    'end_line':item['end_line'],'qualname':item.get('ast_qualname'),
                    'code_sha256':item['code_sha256'],'normalized_code_sha256':item['normalized_code_sha256'],
                    'source_code':code}
            f.write(json.dumps(packet,ensure_ascii=False,sort_keys=True)+'\n')
    template_names=('reviewer_a_negative_template.csv','reviewer_b_negative_template.csv')
    for name in template_names:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['pair_id','project','clone_label','confidence','notes'])
            writer.writeheader()
            for row in selected:writer.writerow({'pair_id':row['pair_id'],'project':row['project'],'clone_label':'','confidence':'','notes':''})
    review=out/'negative_review_manifest.csv'
    with review.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['pair_id','project','similarity','similarity_bin','left_path','left_start','right_path','right_start','annotation_status'])
        writer.writeheader()
        for row in selected:
            a,b=row['left'],row['right']
            writer.writerow({'pair_id':row['pair_id'],'project':row['project'],'similarity':row['pair_similarity'],'similarity_bin':row['similarity_bin'],'left_path':a['repository_file'],
                'left_start':a['start_line'],'right_path':b['repository_file'],'right_start':b['start_line'],'annotation_status':'pending'})
    audit=out/'negative_candidate_audit.json'
    summary.update({'created_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
                    'input_artifacts':input_files,'project_sampling':sample_stats,'sampling_rejection_counts':dict(rejected),
                    'limitations':['A pair not detected by C-01 is not necessarily a non-clone.',
                      'A source-function pool drawn from detector hits is a selected population, not a random universe of all functions.',
                      'Two reviewer independent NEGATIVE decisions are needed before gold_label=0.',
                      'Blank cells in negative review templates mean PENDING, never confirmation.',
                      'Positive confirmations were agreement with detector output, not independent blind discovery.',
                      'Similarity is calculated between the two candidate functions, not candidate-to-training leakage.',
                      'Sampling uses bounded randomized shingle-overlap proposals; results may not represent the entire eligible pair population.',
                      'Adaptive cross-project reallocation changes the original project-level 1:1 sampling mix; audit each project for selection bias.',
                      'No C-01 rerun, no legacy CSV, no automatic Gold Label assignment.']})
    audit.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    artifact_names=[packets.name,*template_names,review.name,audit.name]
    manifest={'version':VERSION,'created_at_chicago':summary['created_at_chicago'],
              'source_policy':'C01_ONLY_FUNCTION_POOL_NO_LEGACY',
              'files':{name:{'bytes':(out/name).stat().st_size,'sha256':checksum(out/name)} for name in artifact_names},
              'input_sha256':summary['input_sha256']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({**{k:summary[k] for k in ('status','active_projects','positive_reference_pairs','negative_candidates','negative_candidate_target','sampling_shortfall','bin_counts','bin_achieved_percentages','verified_negative_labels','full_c01_detected_pair_keys_blocked','annotation_rule')},'out':str(out)},sort_keys=True))
    return 0
if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,KeyError,OSError,TypeError) as exc:
        print('D-05bf v3 ERROR:',str(exc),file=sys.stderr);sys.exit(2)
