#!/usr/bin/env python3
"""D-05bc v1: report-only token-shingle near-duplicate audit for C-01 Python held-out candidates.

No clone detection; no source modifications; no exclusion/labeling based on similarity.
"""
import argparse
import ast
import collections
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v1'
POLICY = 'C01_ONLY_NEAR_DUPLICATE_REPORT_ONLY_NO_EXCLUSION'
TOKEN_PATTERN = re.compile(r'[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\s]', re.UNICODE)


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_jsonl(path):
    with path.open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f'Not JSON object: {path}:{number}')
                yield obj


def tokenize_shingles(code, shingle_n):
    tokens = TOKEN_PATTERN.findall(code)
    if not tokens:
        return frozenset()
    n = min(shingle_n, len(tokens))
    return frozenset(hashlib.blake2b('\x00'.join(tokens[i:i+n]).encode('utf-8'), digest_size=8).digest()
                     for i in range(len(tokens)-n+1))


def training_codes(root):
    base = root/'data/training'
    corpus = base/'clean/python_mixed/data.jsonl'
    split_files = {split: base/'prepared/balanced_multilingual'/(split+'.jsonl') for split in ('train', 'validation')}
    for file in [corpus, *split_files.values()]:
        if not file.is_file():
            raise ValueError('Missing ACTIVE training input: '+str(file))
    lookup = {}
    for row in read_jsonl(corpus):
        ident, code = row.get('idx'), row.get('func')
        if not isinstance(ident, str) or not ident or not isinstance(code, str) or not code.strip() or ident in lookup:
            raise ValueError('Invalid / repeated active python_mixed idx')
        lookup[ident] = code
    used = collections.defaultdict(set)
    pair_counts = {}
    for split, file in split_files.items():
        counts = collections.Counter()
        for row in read_jsonl(file):
            if row.get('corpus') != 'python_mixed':
                continue
            a, b = row.get('id1'), row.get('id2')
            if not isinstance(a,str) or not isinstance(b,str) or a not in lookup or b not in lookup:
                raise ValueError(f'Unresolved ACTIVE python_mixed training IDs in {split}: {a}, {b}')
            used[a].add(split); used[b].add(split)
            counts['python_pair_rows'] += 1
        pair_counts[split] = counts['python_pair_rows']
    codes = {}
    for ident, splits in used.items():
        digest = sha_text(lookup[ident])
        if digest in codes:
            codes[digest]['splits'].update(splits)
        else:
            codes[digest] = {'code': lookup[ident], 'splits': set(splits), 'id': ident}
    inputs = {str(path.relative_to(root)): sha_file(path) for path in [corpus,*split_files.values()]}
    return codes, pair_counts, inputs


def function_nodes(tree):
    out = []
    stack = [(tree, [])]
    while stack:
        node, parents = stack.pop()
        current = parents
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min([node.lineno] + [getattr(x,'lineno',node.lineno) for x in node.decorator_list])
            out.append((node.name, '.'.join(parents+[node.name]), start, node.end_lineno))
            current = parents+[node.name]
        elif isinstance(node, ast.ClassDef):
            current = parents+[node.name]
        stack.extend((child,current) for child in ast.iter_child_nodes(node))
    return out


class SourceResolver:
    def __init__(self, root):
        self.root = root
        self.cache = {}

    def get(self, record, side):
        info = record.get('v6_source_resolution',{}).get(side,{})
        loc = record.get(side+'_locator') or {}
        if info.get('status') != 'EXACT_AST_FUNCTION':
            raise ValueError(f'Expected verified AST source for pair {record.get("pair_id")} {side}')
        relative = info.get('repository_file')
        if not isinstance(relative,str) or not relative.startswith('data/repositories/') or '\\' in relative:
            raise ValueError('Unsafe source reference')
        project = record['project']
        path = (self.root/relative).resolve()
        repository = (self.root/'data/repositories'/project).resolve()
        if not path.is_relative_to(repository) or path.suffix!='.py' or not path.is_file():
            raise ValueError('Repository path escaped or missing: '+relative)
        key = str(path)
        if key not in self.cache:
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest()!=info.get('repository_file_sha256'):
                raise ValueError('Source file SHA changed: '+relative)
            text = content.decode('utf-8')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', SyntaxWarning)
                tree = ast.parse(text, filename=key)
            self.cache[key] = (text.splitlines(keepends=True),function_nodes(tree))
        lines, nodes = self.cache[key]
        start,end = int(info['start_line']),int(info['end_line'])
        name = loc.get('function')
        matches = [node for node in nodes if node[1]==name and node[2]==start and node[3]==end]
        if len(matches)!=1 or start<=0 or end>len(lines):
            raise ValueError('AST Function span/identity changed: '+relative)
        code = ''.join(lines[start-1:end])
        if sha_text(code)!=info.get('code_sha256'):
            raise ValueError('AST Function exact SHA differs from v6: '+relative)
        return code,info['code_sha256']


def compare_shingles(shingles, refs, postings, sizes, threshold):
    if not shingles:
        return {'max_score':0.0,'match':None,'comparisons':0,'flag':False}
    hits = collections.Counter()
    # Exact postings (no approximations or subsampling). Empty shared shingle means Jaccard = 0.
    for shingle in shingles:
        hits.update(postings.get(shingle, ()))
    bestscore = 0.0
    best = None
    n = len(shingles)
    comparisons = 0
    for idx, shared in hits.items():
        other_n = sizes[idx]
        # Prune only when even a complete match of the smaller set cannot beat current score.
        if min(n, other_n)/max(n, other_n) < bestscore:
            continue
        score = shared/(n+other_n-shared)
        comparisons += 1
        if score > bestscore or (score == bestscore and best is not None and refs[idx]['id']<refs[best]['id']):
            bestscore=score
            best=idx
    return {'max_score':round(bestscore,6),'match':None if best is None else {
        'training_function_id':refs[best]['id'],
        'training_code_sha256':refs[best]['sha256'],
        'splits':refs[best]['splits']
    },'comparisons':comparisons,'flag':bestscore >= threshold}


def run(args):
    root=args.root.resolve(); source=args.source.resolve(); output=args.output.resolve()
    if not source.is_dir() or not (source/'heldout_candidates.jsonl').is_file():
        raise ValueError('Missing D-05bb v9 heldout candidates')
    if output.exists():
        raise ValueError('Destination exists (refuse overwrite): '+str(output))
    if output in source.parents or source in output.parents or output==source:
        raise ValueError('Input/output directory overlap')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    # Manifest schema from v9; fail closed on missing/mismatched digest.
    expected={name: entry.get('sha256') for name, entry in manifest.get('files',{}).items()}
    if not expected:
        raise ValueError('Expected v9 artifacts_sha256 in manifest')
    for name in ('heldout_candidates.jsonl','filter_audit.json'):
        if sha_file(source/name)!=expected.get(name):
            raise ValueError('v9 input SHA256 mismatch: '+name)
    prior=json.loads((source/'filter_audit.json').read_text())
    if not str(prior.get('candidate_source_policy','')).startswith('C01_ONLY') or not prior.get('training_reference_complete'):
        raise ValueError('Requires C01_ONLY with complete training reference')
    if not (0 < args.threshold <= 1):
        raise ValueError('threshold must be in (0,1]')
    codes, pair_counts, training_input_sha = training_codes(root)
    refs=[];postings=collections.defaultdict(list);sizes=[]
    for sha, entry in sorted(codes.items()):
        sh=tokenize_shingles(entry['code'],args.shingle_n)
        if not sh:
            raise ValueError('Active Training Python Function has no tokens')
        idx=len(refs)
        refs.append({'id':entry['id'],'sha256':sha,'splits':sorted(entry['splits'])})
        sizes.append(len(sh))
        for token in sh: postings[token].append(idx)
    candidates=list(read_jsonl(source/'heldout_candidates.jsonl'))
    if len(candidates)!=prior.get('counts',{}).get('heldout_candidates_after_dedup'):
        raise ValueError('Candidate count not equal v9 audit')
    projects=set();cache={};audit_counts=collections.Counter(); project_counts=collections.defaultdict(collections.Counter)
    resolver=SourceResolver(root)
    result=[]; rows=[]
    for rec in candidates:
        if rec.get('origin')!='c01_hybrid_detector' or rec.get('gold_label') is not None or rec.get('evaluation_eligible') is not False:
            raise ValueError('Non-C01 or labeled/evaluable candidate encountered')
        project=rec['project']; projects.add(project)
        similarities={}
        for side in ('left','right'):
            code,code_sha=resolver.get(rec,side)
            if code_sha not in cache:
                sh=tokenize_shingles(code,args.shingle_n)
                cache[code_sha]=compare_shingles(sh,refs,postings,sizes,args.threshold)
            similarities[side]=cache[code_sha]
            audit_counts['sides_checked']+=1
            if similarities[side]['flag']:
                audit_counts['sides_flagged']+=1
        maximum=max(similarities['left']['max_score'],similarities['right']['max_score'])
        flagged=any(similarities[s]['flag'] for s in ('left','right'))
        rec['near_duplicate_leakage_status']='AUDITED_REPORT_ONLY_HIGH_SIMILARITY' if flagged else 'AUDITED_REPORT_ONLY_BELOW_THRESHOLD'
        rec['d05bc_similarity']={'metric':'token_shingle_jaccard','n':args.shingle_n,
                                  'threshold':args.threshold,'highest_side_score':round(maximum,6),
                                  'flagged_report_only':flagged, 'left':similarities['left'],'right':similarities['right']}
        if rec['gold_label'] is not None or rec['evaluation_eligible'] is not False:
            raise ValueError('Label/evaluation status modified unexpectedly')
        audit_counts['candidates_preserved']+=1
        audit_counts['candidate_near_duplicate_flagged' if flagged else 'candidate_below_threshold']+=1
        project_counts[project]['records']+=1
        if flagged: project_counts[project]['flagged']+=1
        rows.append({'pair_id':rec.get('pair_id'),'project':project,'left_similarity':similarities['left']['max_score'],
                     'right_similarity':similarities['right']['max_score'],'maximum_similarity':round(maximum,6),
                     'near_duplicate_flag_report_only':flagged})
        result.append(rec)
    if len(projects)!=24 or 'scikit-learn' not in projects or 'azure-sdk-for-python' in projects:
        raise ValueError('24 ACTIVE project coverage failed')
    if len(result)!=len(candidates) or len(result)!=3971:
        raise ValueError('All 3971 v9 candidates must be preserved')
    audit={'status':'D05BC_V1_NEAR_DUPLICATE_AUDIT_REPORT_ONLY_GOLD_NOT_VERIFIED',
           'version':VERSION,'created_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
           'policy':POLICY,'metric':{'name':'token_shingle_jaccard','token_pattern':TOKEN_PATTERN.pattern,
                        'shingle_n':args.shingle_n,'threshold_for_flag_only':args.threshold,
                        'candidate_comparison':'exhaustive_shared_shingle_inverted_index_no_random_sampling',
                        'interpretation':'lexical similarity only; not semantic equivalence or proof of leakage'},
           'source_dir':str(source),'source_v9_sha256':sha_file(source/'heldout_candidates.jsonl'),
           'training_input_sha256':training_input_sha,'training_python_pair_rows':pair_counts,
           'active_training_unique_exact_code_hashes':len(refs),
           'active_projects':len(projects),'counts':dict(audit_counts),
           'project_counts':{k:dict(v) for k,v in sorted(project_counts.items())},
           'verified_gold_labels':0,'candidates_excluded_due_to_near_duplicate':0,
           'limitations':['All original D-05bb v9 held-out candidates are retained.',
                          'Training/validation exact/normalized identity filtering was performed upstream in v9.',
                          'Near duplicate similarity does not determine a clone/nonclone gold label.',
                          'No new negative candidate generation, annotation, model evaluation, or C-01 rerun.',
                          'This audit uses active M2/M3 python_mixed train/validation only; M1 exposure and other sources are separate.']}
    print(json.dumps({'status':'DRY_RUN' if args.dry_run else audit['status'],
                      'candidates_preserved':len(result),'candidate_near_duplicate_flagged':audit_counts['candidate_near_duplicate_flagged'],
                      'excluded_due_to_similarity':0,'active_projects':len(projects),
                      'training_function_hashes':len(refs),'threshold':args.threshold},sort_keys=True),flush=True)
    if args.dry_run:return
    output.mkdir(parents=True,exist_ok=False)
    files={}
    path=output/'candidates_near_duplicate_audited.jsonl'
    with path.open('w',encoding='utf-8') as f:
        for obj in result:f.write(json.dumps(obj,ensure_ascii=False,sort_keys=True)+'\n')
    files[path.name]=sha_file(path)
    path=output/'near_duplicate_review.csv'
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['pair_id','project','left_similarity','right_similarity','maximum_similarity','near_duplicate_flag_report_only'])
        writer.writeheader();writer.writerows(rows)
    files[path.name]=sha_file(path)
    path=output/'near_duplicate_audit.json';path.write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n',encoding='utf-8');files[path.name]=sha_file(path)
    manifest={'version':VERSION,'created_at_chicago':audit['created_at_chicago'],'policy':POLICY,
              'input_sha256':audit['source_v9_sha256'],'artifacts_sha256':files}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('OUT:',output,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--source',type=Path,help='Exact D-05bb v9 results directory; default selects latest')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--threshold',type=float,default=0.80)
    parser.add_argument('--shingle-n',type=int,default=5)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    args.root=args.root.resolve()
    if args.shingle_n<1:parser.error('--shingle-n must be >=1')
    if args.source is None:
        paths=sorted((args.root/'results/d-05b').glob('python-heldout-c01-only-v9-*/heldout_candidates.jsonl'))
        if not paths:parser.error('No D-05bb v9 result found; provide --source')
        args.source=paths[-1].parent
    elif not args.source.is_absolute():args.source=args.root/args.source
    if args.output is None:
        stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
        args.output=args.root/'results/d-05b'/('python-near-duplicate-audit-v1-'+stamp)
    elif not args.output.is_absolute():args.output=args.root/args.output
    try:run(args)
    except (OSError,ValueError,KeyError,json.JSONDecodeError,SyntaxError,UnicodeError) as error:
        print('D-05bc v1 ERROR:',error,file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':sys.exit(main())
