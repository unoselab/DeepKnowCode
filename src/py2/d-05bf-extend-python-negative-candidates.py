#!/usr/bin/env python3
"""D-05bf v5: append-only paramiko supplemental negative *candidates*, no gold labels."""
import argparse,csv,hashlib,json,sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION='v5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def newest(root,glob):
    pp=sorted((root/'results/d-05b').glob(glob))
    pp=[p for p in pp if p.is_dir()]
    if not pp:raise ValueError('No result directory for '+glob)
    return pp[-1]
def verified(d,name):
    mp=d/'manifest.json';mf=json.loads(mp.read_text());entry=(mf.get('files') or {}).get(name)
    expected=entry.get('sha256') if isinstance(entry,dict) else entry
    if not expected:expected=(mf.get('artifacts_sha256') or {}).get(name)
    f=d/name
    if not f.is_file() or not expected or sha(f)!=expected:raise ValueError('Missing or incorrect manifest SHA256: '+str(f))
    return f,sha(mp)
def pairkey(a,b):return tuple(sorted([a,b]))
def run():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--coverage-audit',type=Path)
    ap.add_argument('--annotation-source',type=Path)
    ap.add_argument('--negative-v3',type=Path)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args();root=args.root.resolve()
    coverage=(args.coverage_audit or newest(root,'python-project-coverage-audit-v4-paramiko-*')).resolve()
    annotation=(args.annotation_source or newest(root,'python-gold-annotation-preparation-v1-*')).resolve()
    negative=(args.negative_v3 or newest(root,'python-negative-candidates-v3-*')).resolve()
    cf,cm=verified(coverage,'exhaustive_pair_similarity.csv')
    af,am=verified(annotation,'annotation_packets.jsonl')
    nf,nm=verified(negative,'negative_annotation_packets.jsonl')
    src={};npackets=0
    for r in rows(af):
        if r['project']!='paramiko':continue
        npackets+=1
        for side in ('left','right'):
            x=r[side]; h=x['code_sha256'];code=x['code']
            if hashlib.sha256(code.encode()).hexdigest()!=h:raise ValueError('Source code SHA256 failure')
            if h in src and src[h]!=x:raise ValueError('Conflicting source record for '+h)
            src[h]=x
    if not src:raise ValueError('No paramiko source packet records')
    existing=set();n_existing=0;projects=set();negative_hashes={}
    for r in rows(nf):
        n_existing+=1;projects.add(r['project']); a,b=r['pair_code_hashes']; existing.add((r['project'],pairkey(a,b)))
        for side in ('left','right'):
            x=r[side];h=x['code_sha256']
            if hashlib.sha256(x['source_code'].encode()).hexdigest()!=h:raise ValueError('v3 source SHA mismatch')
            negative_hashes[h]=x
    if n_existing!=3971:raise ValueError('Expected 3971 immutable v3 candidate records')
    outrows=[];statuses=Counter();total=0
    with cf.open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            total+=1
            if r['project']!='paramiko':raise ValueError('Unexpected project in coverage audit')
            sim=float(r['similarity']);key=pairkey(r['left_code_sha256'],r['right_code_sha256'])
            if r['status']=='BLOCKED_C01_DETECTED_PAIR':statuses['blocked_c01']+=1;continue
            if not (.05<=sim<.10):continue
            if r['status']!='OUTSIDE_SIMILARITY_RANGE':raise ValueError('Unexpected v4 status')
            if key[0] not in src or key[1] not in src:raise ValueError('Missing source for candidate')
            if ('paramiko',key) in existing:raise ValueError('Already included in v3')
            h=hashlib.sha256(('paramiko:'+':'.join(key)).encode()).hexdigest()[:24]
            sides=[]
            for ch in key:
                x=src[ch]
                sides.append({'code_sha256':ch,'path':x['path'],'start_line':x['start_line'],
                    'end_line':x['end_line'],'qualname':x['function'],'source_code':x['code']})
            item={'pair_id':'negcand_paramiko_extra_'+h,'project':'paramiko','language':'python',
                'left':sides[0],'right':sides[1],'pair_code_hashes':list(key),
                'pair_similarity':sim,'similarity_bin':'0.05-0.10',
                'similarity_metric':'token_shingle_jaccard','similarity_shingle_n':5,
                'selection_method':'v4_exhaustive_paramiko_coverage_0.05_0.10',
                'candidate_source':'C01_HELDOUT_FUNCTION_POOL_RECOMBINATION',
                'c01_detected_pair':False,'c01_full_detected_pair_blocklist_checked':True,
                'active_training_validation_exact_normalized_overlap':False,
                'negative_label_assumed':False,'annotation_status':'pending',
                'gold_label':None,'label_source':None,'evaluation_eligible':False,
                'coverage_extension_policy':'PARAMIKO_ONLY_0.05_TO_0.10',
                'source_evidence':'D05BD_SOURCE_VERIFIED_AND_D05BF_V4_EXHAUSTIVE_AUDIT'}
            if sides[0]['code_sha256']==sides[1]['code_sha256']:raise ValueError('Same function')
            outrows.append(item)
    if total!=210:raise ValueError('Expected 210 exhaustive paramiko pair combinations')
    if not outrows:raise ValueError('No valid extra candidates')
    if len({x['pair_id'] for x in outrows})!=len(outrows):raise ValueError('Duplicate extra candidates')
    if len(set(tuple(x['pair_code_hashes']) for x in outrows))!=len(outrows):raise ValueError('Duplicate hashes')
    summary={'version':VERSION,'status':'DRY_RUN' if args.dry_run else 'D05BF_V5_PARAMIKO_SUPPLEMENTAL_CANDIDATES_NOT_GOLD',
        'project':'paramiko','source_policy':'C01_ONLY_HELDOUT_FUNCTION_POOL','similarity_range':'[0.05,0.10)',
        'metric':'token_shingle_jaccard','exhaustive_pairs_audited':total,'v3_original_candidates_preserved':n_existing,
        'supplemental_negative_annotation_candidates':len(outrows),
        'total_candidate_count_if_combined':n_existing+len(outrows),
        'projects_covered_if_combined':len(projects|{'paramiko'}),
        'gold_labels_created':0,'annotation_rule':'BLANK_MEANS_PENDING_EXPLICIT_ZERO_REQUIRED',
        'input_manifest_sha256':{'v4_coverage_audit':cm,'d05bd_annotation_source':am,'negative_v3':nm}}
    if args.dry_run:print(json.dumps(summary,sort_keys=True));return
    stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
    out=root/'results/d-05b'/('python-paramiko-supplemental-negative-v5-'+stamp)
    out.mkdir(parents=True,exist_ok=False)
    output=out/'supplemental_negative_annotation_packets.jsonl'
    with output.open('w',encoding='utf-8') as f:
        for x in outrows:f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n')
    names=[]
    for reviewer in ('a','b'):
        name=f'reviewer_{reviewer}_supplemental_template.csv';names.append(name)
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['pair_id','project','clone_label','confidence','notes'])
            for x in outrows:w.writerow([x['pair_id'],x['project'],'','',''])
    names+=['supplemental_review_manifest.csv']
    with (out/'supplemental_review_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['pair_id','project','similarity','similarity_bin','left_path','left_start_line','right_path','right_start_line','annotation_status'])
        for x in outrows:w.writerow([x['pair_id'],x['project'],x['pair_similarity'],x['similarity_bin'],x['left']['path'],x['left']['start_line'],x['right']['path'],x['right']['start_line'],'pending'])
    summary['created_at_chicago']=datetime.now(ZoneInfo('America/Chicago')).isoformat()
    (out/'supplemental_candidate_audit.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    filenames=['supplemental_negative_annotation_packets.jsonl',*names,'supplemental_candidate_audit.json']
    (out/'manifest.json').write_text(json.dumps({'version':VERSION,'created_at_chicago':summary['created_at_chicago'],
        'files':{f:{'sha256':sha(out/f),'bytes':(out/f).stat().st_size} for f in filenames},
        'input_manifest_sha256':summary['input_manifest_sha256']},indent=2,sort_keys=True)+'\n')
    print(json.dumps({**summary,'out':str(out)},sort_keys=True))
if __name__=='__main__':
    try:run()
    except (ValueError,KeyError,OSError,TypeError) as e:
        print('D-05bf v5 ERROR: '+str(e),file=sys.stderr);sys.exit(2)
