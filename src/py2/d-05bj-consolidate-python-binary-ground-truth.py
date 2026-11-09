#!/usr/bin/env python3
"""D-05bj v2: consolidate independently submitted reviewer conventions into binary GT.

Consumes D-05bd v1 source, D-05be v2 confirmed positives, D-05bh v1
unified negatives and two submitted NEGATIVE review CSVs. Does not alter
upstream artifacts, archived v4 candidate dataset, training or detection.
No reviewer identities are authenticated. Blank-negative policy is a
USER-AUTHORIZED approval convention and is not independent semantic proof.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v2'
POS_CONVENTION = 'BLANK_MEANS_REVIEWER_CONFIRMS_C01_CLONE_POSITIVE'
NEG_CONVENTION = 'BLANK_MEANS_REVIEWER_CONFIRMS_NONCLONE_NEGATIVE'
POS_PREPARED = 'annotation_packets.jsonl'
POS_CONFIRMED = 'reviewer_confirmed_positive_pairs.jsonl'
NEG_PACKETS = 'unified_negative_annotation_packets.jsonl'
EXPECTED_POS = 3971
EXPECTED_NEG_CAND = 3975
EXPECTED_NEG_GOLD = 3971
EXPECTED_REJECTED_CLONES = 4
EXPECTED_PROJECTS = 24
CSV_COLS = ('pair_id', 'project', 'clone_label', 'confidence', 'notes')


def fail(message):
    raise ValueError(message)


def require(condition, message):
    if not condition:
        fail(message)


def hashbytes(data):
    return hashlib.sha256(data).hexdigest()


def hashfile(path):
    return hashbytes(path.read_bytes())


def json_bytes(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8')


def read_bundle(path, packet_name):
    """Verify ALL upstream manifest-listed files, including packet SHA256."""
    require(path.exists(), 'Missing input bundle: ' + str(path))
    if path.is_dir():
        manifest_raw = (path/'manifest.json').read_bytes()
        manifest = json.loads(manifest_raw)
        def read(name):
            require('/' not in name and name not in ('.', '..'), 'Unsafe manifest filename')
            return (path/name).read_bytes()
    else:
        require(path.suffix.lower() == '.zip', 'Expected directory or ZIP: '+str(path))
        with zipfile.ZipFile(path) as z:
            require(z.testzip() is None, 'Corrupt ZIP: '+str(path))
            names = [n for n in z.namelist() if n.endswith('/'+packet_name)]
            require(len(names) == 1, 'ZIP must contain one packet file '+packet_name)
            base = names[0].rsplit('/', 1)[0]+'/'
            manifest_raw = z.read(base+'manifest.json')
            manifest = json.loads(manifest_raw)
            records = manifest.get('files') or {k:{'sha256':v} for k,v in manifest.get('artifacts_sha256',{}).items()}
            require(records and packet_name in records, 'Missing checksum of '+packet_name)
            for name, meta in records.items():
                require('/' not in name and name not in ('.','..'), 'Unsafe manifest filename')
                data = z.read(base+name)
                exp = meta['sha256'] if isinstance(meta, dict) else meta
                require(hashbytes(data) == exp, 'ZIP manifest SHA mismatch '+name)
                if isinstance(meta,dict) and 'bytes' in meta:
                    require(len(data)==meta['bytes'], 'ZIP size mismatch '+name)
            return parse_jsonl(z.read(base+packet_name),packet_name),hashbytes(manifest_raw)
    records = manifest.get('files') or {k:{'sha256':v} for k,v in manifest.get('artifacts_sha256',{}).items()}
    require(records and packet_name in records, 'Missing manifest checksum of '+packet_name)
    for name, meta in records.items():
        data=read(name)
        exp=meta['sha256'] if isinstance(meta,dict) else meta
        require(hashbytes(data)==exp, 'Manifest SHA mismatch: '+str(path/name))
        if isinstance(meta,dict) and 'bytes' in meta:
            require(len(data)==meta['bytes'], 'Manifest size mismatch: '+str(path/name))
    return parse_jsonl(read(packet_name),packet_name),hashbytes(manifest_raw)


def parse_jsonl(raw,name):
    out=[]
    for number,line in enumerate(raw.decode('utf-8-sig').splitlines(),1):
        require(bool(line.strip()), f'Blank line in {name}:{number}')
        obj=json.loads(line)
        require(isinstance(obj,dict), f'Nonobject in {name}:{number}')
        out.append(obj)
    return out


def resolve(root,explicit,pattern):
    if explicit:
        selected=Path(explicit).expanduser()
        return (selected if selected.is_absolute() else root/selected).resolve()
    candidates=sorted([p for p in (root/'results/d-05b').glob(pattern) if p.is_dir()])
    require(len(candidates)>0, 'Missing default input: '+pattern+'; provide explicit path')
    return candidates[-1].resolve()


def index(rows,label):
    result={}
    for row in rows:
        pid=row.get('pair_id')
        require(isinstance(pid,str) and pid and pid not in result, 'Duplicate/invalid '+label+' pair_id '+str(pid))
        require(row.get('project') not in ('',None,'azure-sdk-for-python'), 'Invalid project '+str(pid))
        result[pid]=row
    return result


def source_side(side,side_name,pid):
    code=side.get('source_code',side.get('code'))
    require(isinstance(code,str), f'Missing source code in {pid}/{side_name}')
    h=hashbytes(code.encode('utf-8'))
    require(h==side.get('code_sha256'), f'Source SHA256 mismatch: {pid}/{side_name}')
    return {
        'source_code':code, 'code_sha256':h,
        'path':side.get('path',side.get('repository_file')),
        'function':side.get('function',side.get('qualname')),
        'start_line':side.get('start_line'),'end_line':side.get('end_line'),
    }


def pairkey(row):
    return tuple(sorted([row['left']['code_sha256'],row['right']['code_sha256']]))


def review_csv(path,packets,who):
    require(path.is_file(), 'Missing reviewer CSV: '+str(path))
    result={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        require(reader.fieldnames and set(CSV_COLS).issubset(reader.fieldnames), 'Bad '+who+' CSV columns')
        for n,r in enumerate(reader,2):
            pid=(r.get('pair_id') or '').strip()
            require(pid in packets and pid not in result, f'Unexpected/duplicate {who} pair_id {pid} on line {n}')
            require((r.get('project') or '').strip()==packets[pid]['project'], f'{who} project mismatch {pid}')
            label=(r.get('clone_label') or '').strip().lower()
            require(label in ('','0','1','uncertain'),f'Invalid label {who} {pid}: {label}')
            result[pid]={'raw_label':label,'mapped_label':('0' if label=='' else label),
                         'confidence':(r.get('confidence') or '').strip(), 'notes':(r.get('notes') or '').strip()}
    require(set(result)==set(packets),f'{who} CSV does not cover all {len(packets)} rows')
    return result


def validate_sources(positive_prepared,positive_confirmed,neg_candidates):
    pre=index(positive_prepared,'prepared positive')
    con=index(positive_confirmed,'confirmed positive')
    neg=index(neg_candidates,'negative candidate')
    require(len(pre)==EXPECTED_POS and len(con)==EXPECTED_POS,'Expected 3,971 prepared/confirmed positives')
    require(len(neg)==EXPECTED_NEG_CAND,'Expected 3,975 negatives for v1')
    require(set(pre)==set(con),'Positive pair_id coverage mismatch')
    positive=[];positive_keys=set();negative_keys=set()
    for pid,p in sorted(pre.items()):
        c=con[pid]
        require(p['project']==c['project'],'Positive project mismatch '+pid)
        require(c.get('gold_label')==1 and c.get('reviewer_a_confirmed') is True and c.get('reviewer_b_confirmed') is True,'Positive reviewer confirmation missing '+pid)
        require(c.get('label_source')=='C01_DETECTED_REVIEWER_A_B_CONFIRMED','Positive provenance invalid '+pid)
        require(c.get('independent_blind_annotation') is False,'Positive independent blind claim inconsistent '+pid)
        l,r=source_side(p['left'],'left',pid),source_side(p['right'],'right',pid)
        require(c.get('left_code_sha256')==l['code_sha256'] and c.get('right_code_sha256')==r['code_sha256'],'Positive source hash mismatch '+pid)
        for side_name,side in (('left',l),('right',r)):
            locator=c.get(side_name+'_locator',{})
            for attr,key in (('path','path'),('start_line','start_line'),('end_line','end_line'),('function','function')):
                require(locator.get(attr)==side[key],f'Positive locator mismatch {pid}/{side_name}/{attr}')
        out={'pair_id':pid,'project':p['project'],'language':'python','gold_label':1,
             'left':l,'right':r,'pair_code_hashes':sorted([l['code_sha256'],r['code_sha256']]),
             'label_source':'C01_DETECTED_REVIEWER_A_B_CONFIRMED',
             'annotation_convention':POS_CONVENTION,'reviewer_a_confirmed':True,
             'reviewer_b_confirmed':True,'independent_blind_annotation':False,
             'evaluation_eligible':True}
        k=pairkey(out)
        # Type-1 clone positives may legitimately contain byte-identical functions.
        # Reject repeated PAIR keys across records, not identical sides within one pair.
        require(k not in positive_keys,'Duplicate positive code pair across records '+pid)
        positive_keys.add(k);positive.append(out)
    for pid,n in neg.items():
        l,r=source_side(n['left'],'left',pid),source_side(n['right'],'right',pid)
        k=tuple(sorted([l['code_sha256'],r['code_sha256']]))
        require(k[0]!=k[1] and k not in negative_keys,'Duplicate/identical negative code pair '+pid)
        negative_keys.add(k)
        require(n.get('gold_label') is None,'Negative candidate has preassigned gold label '+pid)
        # D-05bh unified review packets are a *projection* of D-05bf v3/v5
        # and intentionally omit original provenance flags. Never treat
        # missing fields as False; validate mandatory upstream review audit.
        for name in ('c01_detected_pair','active_training_validation_exact_normalized_overlap','negative_label_assumed'):
            if name in n:
                require(n[name] is False, 'Excluded/auto-labeled candidate '+pid+'/'+name)
        sim=float(n['pair_similarity'])
        require((0.05<=sim<0.10) if n['project']=='paramiko' else (0.10<=sim<0.50), 'Invalid similarity range '+pid)
    require(not (positive_keys & negative_keys),'Positive/negative code-hash pair overlap')
    projects={p['project'] for p in positive}
    require(len(projects)==EXPECTED_PROJECTS and 'scikit-learn' in projects,'Positive project coverage not 24')
    require({n['project'] for n in neg.values()}==projects,'Negative candidate project coverage != positives')
    return positive,neg,positive_keys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--positive-prepared',help='D-05bd v1 folder or ZIP')
    ap.add_argument('--positive-confirmed',help='D-05be v2 folder or ZIP')
    ap.add_argument('--negative-review',help='D-05bh v1 unified review folder or ZIP')
    ap.add_argument('--reviewer-a',required=True,help='D-05bh Reviewer A submitted CSV')
    ap.add_argument('--reviewer-b',required=True,help='D-05bh Reviewer B submitted CSV')
    ap.add_argument('--out',help='New canonical output directory, default data/evaluation/ground_truth/python')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    try:
        root=args.root.resolve()
        psrc=resolve(root,args.positive_prepared,'python-gold-annotation-preparation-v1-*')
        csrc=resolve(root,args.positive_confirmed,'python-confirmed-positive-v2-*')
        nsrc=resolve(root,args.negative_review,'python-unified-negative-review-v1-*')
        ra=(root/args.reviewer_a).resolve();rb=(root/args.reviewer_b).resolve()
        out=(root/(args.out or 'data/evaluation/ground_truth/python')).resolve()
        require(ra!=rb,'Reviewer CSV A and B must be different files')
        require(not out.exists(),'Output already exists; will not overwrite: '+str(out))
        pre,psha=read_bundle(psrc,POS_PREPARED)
        con,csha=read_bundle(csrc,POS_CONFIRMED)
        neg,nsha=read_bundle(nsrc,NEG_PACKETS)
        require(nsrc.is_dir(), 'For v1, D-05bh review must be a directory with verifiable audit JSON')
        review_audit=json.loads((nsrc/'negative_review_audit.json').read_text(encoding='utf-8'))
        require(review_audit.get('status')=='D05BH_V1_NEGATIVE_REVIEW_READY_NOT_GOLD',
                'Unexpected D-05bh audit status')
        require(review_audit.get('candidate_count')==EXPECTED_NEG_CAND and
                review_audit.get('v3_candidates')==3971 and
                review_audit.get('paramiko_supplemental_candidates')==4 and
                review_audit.get('project_coverage')==EXPECTED_PROJECTS and
                review_audit.get('source_policy')=='C01_ONLY_NO_LEGACY',
                'Upstream D-05bh audit does not establish expected provenance')
        require(isinstance(review_audit.get('input_manifest_sha256'),dict) and
                all(review_audit['input_manifest_sha256'].get(k) for k in ('negative_v3','paramiko_supplemental_v5')),
                'Missing D-05bf v3/v5 provenance manifest hashes')
        pos,nmap,_=validate_sources(pre,con,neg)
        a=review_csv(ra,nmap,'Reviewer A');b=review_csv(rb,nmap,'Reviewer B')
        negative=[];reclassified=[];review_counts=collections.Counter()
        for pid,n in sorted(nmap.items()):
            x,y=a[pid],b[pid]
            require(x['mapped_label']==y['mapped_label'] and x['mapped_label']!='uncertain', 'Unresolved reviewer disagreement/uncertain '+pid)
            label=x['mapped_label'];review_counts[label]+=1
            row={'pair_id':pid,'project':n['project'],'language':'python',
                 'left':source_side(n['left'],'left',pid), 'right':source_side(n['right'],'right',pid),
                 'pair_similarity':float(n['pair_similarity']),'similarity_bin':n['similarity_bin'],
                 'pair_code_hashes':sorted([n['left']['code_sha256'],n['right']['code_sha256']]),
                 'reviewer_a_raw_label':x['raw_label'],'reviewer_b_raw_label':y['raw_label'],
                 'reviewer_a_interpreted_label':int(label),'reviewer_b_interpreted_label':int(label),
                 'annotation_convention':NEG_CONVENTION,
                 'reviewer_a_confirmed':True,'reviewer_b_confirmed':True,
                 'label_source':'NEGATIVE_REVIEWER_A_B_CONFIRMED_BY_USER_CONVENTION',
                 'independent_blind_annotation':False,
                 'candidate_source':n.get('candidate_source'),
                 'coverage_extension_policy':n.get('coverage_extension_policy')}
            if label=='0':
                row.update({'gold_label':0,'evaluation_eligible':True});negative.append(row)
            else:
                row.update({'gold_label':1,'evaluation_eligible':False,
                            'disposition':'REVIEWER_IDENTIFIED_CLONE_EXCLUDED_FROM_NEGATIVE_GOLD'});
                reclassified.append(row)
        require(len(negative)==EXPECTED_NEG_GOLD and len(reclassified)==EXPECTED_REJECTED_CLONES,
                'Unexpected negative/nonclone count: '+str((len(negative),len(reclassified))))
        positive_projects={r['project'] for r in pos}
        require({r['project'] for r in negative}==positive_projects,'Negative Gold lacks project coverage')
        allrows=sorted(pos+negative,key=lambda r:(r['project'],r['pair_id'],r['gold_label']))
        all_ids=[r['pair_id'] for r in allrows]
        require(len(all_ids)==len(set(all_ids)),'Cross-source pair_id duplication')
        all_hashes=[pairkey(r) for r in allrows]
        require(len(all_hashes)==len(set(all_hashes)),'Cross-source code pair duplication')
        require(len(allrows)==7942,'Binary Gold count mismatch')
        now=datetime.now(ZoneInfo('America/Chicago'))
        summary={'status':'DRY_RUN' if args.dry_run else 'D05BJ_V2_BINARY_GROUND_TRUTH_CONSOLIDATED',
                 'created_at_chicago':now.isoformat(), 'positive_gold':len(pos),
                 'negative_gold':len(negative),'excluded_reviewer_detected_clones':len(reclassified),
                 'verified_binary_pairs':len(allrows),'project_coverage':len(positive_projects),
                 'reviewer_neg_label_counts':dict(review_counts),
                 'identical_code_positive_pairs':sum(1 for r in pos if r['left']['code_sha256']==r['right']['code_sha256']),
                 'negative_blank_means':'CONFIRMED_NONCLONE_BY_EXPLICIT_USER_POLICY',
                 'reviewer_identity_authenticated':False,
                 'positive_independent_blind_review':False,
                 'negative_independent_blind_review_authenticated':False,
                 'archived_v4_candidate_files_reused':False,
                 'output':str(out)}
        if args.dry_run:
            print(json.dumps(summary,sort_keys=True));return 0
        # Output is never allowed to replace an existing directory.
        out.mkdir(parents=True,exist_ok=False)
        def write_lines(name,rows):
            with (out/name).open('w',encoding='utf-8') as f:
                for row in rows:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
        write_lines('positive_verified_pairs.jsonl',pos)
        write_lines('negative_verified_pairs.jsonl',negative)
        write_lines('verified_pairs.jsonl',allrows)
        write_lines('excluded_negative_candidate_clones.jsonl',reclassified)
        audit=dict(summary)
        audit.update({'input_paths':{'positive_prepared':str(psrc),'positive_confirmed':str(csrc),
                 'negative_review':str(nsrc),'reviewer_a':str(ra),'reviewer_b':str(rb)},
             'input_sha256':{'positive_prepared_manifest':psha,'positive_confirmed_manifest':csha,
                 'negative_review_manifest':nsha,'reviewer_a_csv':hashfile(ra),'reviewer_b_csv':hashfile(rb)},
             'annotation_conventions':{'positive':POS_CONVENTION,'negative':NEG_CONVENTION},
             'project_counts':dict(sorted(collections.Counter(r['project'] for r in allrows).items())),
             'leakage_validation_scope':'Verified upstream exact/normalized overlap flags and upstream manifests; no independent training corpus rescan',
             'c01_detection_policy':'Positive provenance C01 reviewer confirmed; negative C01 exclusion inherited from D-05bh audit of D-05bf v3/v5, not reverified from full C01 pair blocklist',
             'limitations':['A blank Negative reviewer cell is mapped to explicit Non-clone agreement under user-supplied convention.',
                 'CSV cannot establish external reviewer identity, actual independent visual review, or pre-agreement of blank convention.',
                 'The positive reference was C-01 detector-aware confirmation, not independent blind discovery.',
                 'Sampling was constructed from C-01-detected functions and is not representative of naturally occurring pairs.',
                 'Exact/normalized training exclusion inherited from upstream audited packet flags and manifests; near-duplicates are report-only.',
                 'The four negative-candidate clone discoveries are excluded from binary frozen set and retained as a separate audit file.',
                 'Type-1 exact-identical function pairs are permitted for positive gold, while duplicate records of the same code-hash pair remain forbidden.']})
        (out/'annotation_audit.json').write_bytes(json_bytes(audit))
        leakage={'status':'UPSTREAM_EVIDENCE_VERIFIED_NOT_RECOMPUTED','active_training_validation_identity_overlap_candidates':0,
                 'c01_detected_negative_candidates':'NOT_RECOMPUTED_IN_D05BJ','source_code_sha256_verified':True,
                 'upstream_manifest_sha256_verified':True,'cross_label_duplicate_code_hash_pairs':0,
                 'near_duplicate_policy':'REPORT_ONLY_NO_AUTO_EXCLUSION',
                 'limitations':'No independent source-repository, training-set, or C-01 pair blocklist re-audit; uses upstream D-05bd/D-05be/D-05bh checksum-protected evidence.'}
        (out/'leakage_audit.json').write_bytes(json_bytes(leakage))
        names=['positive_verified_pairs.jsonl','negative_verified_pairs.jsonl','verified_pairs.jsonl',
               'excluded_negative_candidate_clones.jsonl','annotation_audit.json','leakage_audit.json']
        manifest={'version':VERSION,'status':summary['status'],'created_at_chicago':now.isoformat(),
                  'record_counts':{'positive':len(pos),'negative':len(negative),'binary':len(allrows),'excluded_clone_candidates':len(reclassified)},
                  'reviewer_conventions':{'positive':POS_CONVENTION,'negative':NEG_CONVENTION},
                  'input_sha256':audit['input_sha256'],
                  'files':{name:{'sha256':hashfile(out/name),'bytes':(out/name).stat().st_size} for name in names}}
        (out/'manifest.json').write_bytes(json_bytes(manifest))
        print(json.dumps(summary,sort_keys=True))
        return 0
    except (OSError,ValueError,KeyError,TypeError,IndexError,zipfile.BadZipFile,json.JSONDecodeError) as e:
        print('D-05bj v2 ERROR:',e,file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())
