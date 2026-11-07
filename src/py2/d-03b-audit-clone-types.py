#!/usr/bin/env python3
"""Non-mutating clone-type/provenance audit for D-03 M2.

Not a clone-type classifier: unannotated type labels stay UNKNOWN.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

PYTHON_SHA = {
    'data.jsonl': 'da8fdd6362df692e225a5bf5b72278fd6211cef75e48e6a539dda637b76a1843',
    'train.txt': 'e8c8998f2e73a3587190904fb258ac1ccda306f0be62dec499622f4258603693',
    'valid.txt': '33c5a0506fdbbd8ff5f9604a9be1d4fba1b6b6d52943ff11ff53928861ac48a7',
}
DOMAINS = ('java_mixed', 'python_mixed')
TYPES = {'T1','T2','T3','T4','UNKNOWN'}
TOKEN_RE = re.compile(r'[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s]')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_codes(path):
    out = {}
    with path.open('r', encoding='utf-8-sig') as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            row=json.loads(line)
            key,code=str(row['idx']),row['func']
            if key in out or not isinstance(code,str) or not code.strip():
                raise ValueError(f'invalid or duplicate function {path}:{ln}')
            out[key]=code
    return out


def canonical_pair(a,b):
    return tuple(sorted((str(a),str(b))))


def read_annotations(path):
    if not path:
        return {}, {}
    if not path.is_file():
        raise FileNotFoundError(path)
    annotations={}
    evidence=Counter()
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        required={'domain','id1','id2','clone_type','project_id','evidence'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('annotation CSV must have '+','.join(sorted(required)))
        for line_num,row in enumerate(reader,2):
            domain=(row['domain'] or '').strip()
            if domain not in DOMAINS: raise ValueError(f'invalid domain at line {line_num}')
            key=(domain, *canonical_pair(row['id1'],row['id2']))
            if key in annotations: raise ValueError(f'duplicate annotation {key}')
            clone_type=(row['clone_type'] or 'UNKNOWN').strip().upper()
            if clone_type not in TYPES: raise ValueError(f'invalid clone type {clone_type} at {line_num}')
            project=(row['project_id'] or '').strip() or 'UNKNOWN'
            source=(row['evidence'] or '').strip()
            # A non-UNKNOWN type must have a specific cited review/detector ground truth.
            if clone_type!='UNKNOWN' and (not source or source.upper() in {'UNKNOWN','HEURISTIC','SIMILARITY'}):
                raise ValueError(f'unsupported clone type annotation: {key}; evidence required')
            annotations[key]={'clone_type':clone_type,'project_id':project,'evidence':source}
            evidence[domain]+=1
    return annotations, dict(evidence)


def read_pairs(path,known):
    seen=set()
    with path.open('r',encoding='utf-8-sig') as f:
        for ln,line in enumerate(f,1):
            if not line.strip():continue
            parts=line.split()
            if len(parts)!=3 or parts[2] not in ('0','1'):
                raise ValueError(f'invalid pair {path}:{ln}')
            a,b,label=parts
            if a not in known or b not in known: raise ValueError(f'unknown function ID {path}:{ln}')
            if a==b: raise ValueError(f'self pair {path}:{ln}')
            key=canonical_pair(a,b)
            if key in seen: raise ValueError(f'duplicate pair {path}:{ln}')
            seen.add(key)
            yield a,b,int(label)


def jaccard(a,b):
    return len(a & b)/max(1,len(a | b))


def bucket(score):
    if score < .25:return '0.00-0.25'
    if score < .50:return '0.25-0.50'
    if score < .75:return '0.50-0.75'
    if score < .90:return '0.75-0.90'
    return '0.90-1.00'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--annotations',type=Path,help='Optional audited CSV with domain,id1,id2,clone_type,project_id,evidence')
    args=p.parse_args()
    root=args.root.resolve()
    out=args.run_dir.resolve()
    if out.exists():
        if any(out.iterdir()): raise FileExistsError('Run directory is not empty: '+str(out))
    else:out.mkdir(parents=True)
    annotations, annotation_counts=read_annotations(args.annotations)
    python_path=root/'data/training/clean/python_mixed'
    actual_hashes={k:sha256(python_path/k) for k in PYTHON_SHA}
    bad={k:{'expected':v,'actual':actual_hashes[k]} for k,v in PYTHON_SHA.items() if actual_hashes[k]!=v}
    if bad:raise RuntimeError('Approved C-10 dataset SHA256 mismatch: '+json.dumps(bad,sort_keys=True))
    now=datetime.now(ZoneInfo('America/Chicago')).isoformat()
    summary={'task':'D-03 M2 clone type and provenance exploratory audit',
             'timestamp_chicago':now,'status':'AUDIT_COMPLETE_LABELS_UNVERIFIED',
             'python_c10_sha256':actual_hashes,
             'annotations_file':str(args.annotations.resolve()) if args.annotations else None,
             'annotations_sha256':sha256(args.annotations) if args.annotations else None,
             'annotation_rows':annotation_counts,
             'limitations':[
                 'Clone-type labels are UNKNOWN unless provided with external evidence in the annotation CSV.',
                 'Token Jaccard bands are NOT verified Type-1/2/3/4 categories.',
                 'Project provenance is UNKNOWN without mapped project identifiers.',
                 'C-10 negative candidates are not all manually adjudicated.',
                 'This audit does not establish near-duplicate independence or project-wise separation.',
                 'No training or model adaptation is performed.'],
             'domains':{}}
    annotation_matched=set()
    pair_csv=out/'pair_audit.csv'
    with pair_csv.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['domain','split','id1','id2','label','clone_type','project_id','evidence','token_set_jaccard','similarity_band'])
        writer.writeheader()
        for domain in DOMAINS:
            source=root/'data/training/clean'/domain
            codes=read_codes(source/'data.jsonl')
            tokens={k:set(TOKEN_RE.findall(v)) for k,v in codes.items()}
            stats={'functions':len(codes),'splits':{},'confirmed_clone_type_counts':Counter(),
                   'projects':Counter(),'unknown_positive_type':0,'positive_type_evidence':0}
            seen_by_split={}
            for split,filename in [('train','train.txt'),('valid','valid.txt')]:
                labels=Counter(); bands=defaultdict(Counter); pairs=set(); unknown=0
                for a,b,y in read_pairs(source/filename,codes):
                    labels[str(y)]+=1
                    pairkey=canonical_pair(a,b)
                    pairs.add(pairkey)
                    key=(domain,*pairkey)
                    ann=annotations.get(key, {})
                    if ann:annotation_matched.add(key)
                    clone_type=ann.get('clone_type','UNKNOWN') if y else 'NONCLONE'
                    if y and clone_type=='UNKNOWN':unknown+=1
                    if not y and ann.get('clone_type','UNKNOWN')!='UNKNOWN':
                        raise ValueError(f'nonclone annotated as a clone type: {key}')
                    project=ann.get('project_id','UNKNOWN')
                    evidence=ann.get('evidence','')
                    similarity=jaccard(tokens[a],tokens[b])
                    bkt=bucket(similarity)
                    bands[str(y)][bkt]+=1
                    if split=='train':
                        stats['projects'][project]+=1
                        if y:
                            stats['confirmed_clone_type_counts'][clone_type]+=1
                            stats['unknown_positive_type']+=int(clone_type=='UNKNOWN')
                            stats['positive_type_evidence']+=int(clone_type!='UNKNOWN')
                    writer.writerow({'domain':domain,'split':split,'id1':a,'id2':b,'label':y,
                                     'clone_type':clone_type,'project_id':project,'evidence':evidence,
                                     'token_set_jaccard':f'{similarity:.6f}','similarity_band':bkt})
                seen_by_split[split]=pairs
                stats['splits'][split]={'pair_count':sum(labels.values()),'labels':dict(labels),
                                        'positive_unknown_type':unknown,
                                        'token_similarity_bands_by_label':{k:dict(v) for k,v in bands.items()}}
            intersect=seen_by_split['train']&seen_by_split['valid']
            if intersect:raise ValueError(f'train/valid duplicate pair in {domain}: {len(intersect)}')
            summary['domains'][domain]={**stats,
                'confirmed_clone_type_counts':dict(stats['confirmed_clone_type_counts']),
                'projects':dict(stats['projects'])}
    unmatched=set(annotations)-annotation_matched
    if unmatched:raise ValueError(f'annotations not found in dataset pairs: {len(unmatched)}; sample={list(sorted(unmatched))[:3]}')
    summary['pair_audit_sha256']=sha256(pair_csv)
    summary_path=out/'clone_type_audit.json'
    summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
    for domain,entry in summary['domains'].items():
        print(f"[D-03 TYPE AUDIT] {domain}: train={entry['splits']['train']['pair_count']} "
              f"positive_unknown_type={entry['unknown_positive_type']} "
              f"project_unknown={entry['projects'].get('UNKNOWN',0)}")
    print('[D-03 TYPE AUDIT] output='+str(summary_path))
    print('[D-03 TYPE AUDIT] No clone type inferred from similarity scores; no training performed')

if __name__=='__main__':
    main()
