#!/usr/bin/env python3
"""B-01 read-only, bounded Python function clone-candidate discovery pilot."""
import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone

EXPECTED = '0fa7bec68485f8d89b7de2bd6b8743d040fb956b'

def git(repo, *args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()

def sha(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()

def atomic_csv(path, fields, records):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)

def table(title, fields, rows):
    print('\n'+title)
    data=[[str(v) for v in row] for row in rows]
    sizes=[max(len(str(fields[i])),*(len(row[i]) for row in data)) for i in range(len(fields))]
    line='+'+'+'.join('-'*(w+2) for w in sizes)+'+'
    fmt=lambda vals:'| '+' | '.join(str(x).ljust(sizes[i]) for i,x in enumerate(vals))+' |'
    print(line);print(fmt(fields));print(line)
    for r in data:print(fmt(r))
    print(line)

def norm(node, identifiers=False):
    n=ast.fix_missing_locations(ast.parse(ast.unparse(node))).body[0]
    class Transformer(ast.NodeTransformer):
        def visit_Name(self,x):
            if identifiers:x.id='ID'
            return x
        def visit_arg(self,x):
            if identifiers:x.arg='ARG'
            return x
        def visit_Constant(self,x):
            if identifiers:x.value='STR' if isinstance(x.value,str) else 0 if isinstance(x.value,(int,float,complex)) and not isinstance(x.value,bool) else x.value
            return x
        def visit_FunctionDef(self,x):
            if identifiers:x.name='FUNC'
            return self.generic_visit(x)
        def visit_AsyncFunctionDef(self,x):
            if identifiers:x.name='FUNC'
            return self.generic_visit(x)
        def visit_Attribute(self,x):
            if identifiers:x.attr='ATTR'
            return self.generic_visit(x)
    n=Transformer().visit(n)
    return ast.dump(n,annotate_fields=False,include_attributes=False)

def shingles(code,k=5):
    # structural token sequence; normalized names/literals, fixed k-grams
    words=re.findall(r'[A-Za-z_]+|\d+',code)
    if len(words)<k:return set(words)
    return {sha(' '.join(words[i:i+k]))[:16] for i in range(len(words)-k+1)}

def bands(sh):
    # deterministic minhash: 12 bands * 3 minima; candidates only, not exhaustive
    nums=[int(x,16) for x in sh]
    if not nums:return []
    results=[]
    for b in range(12):
        vals=sorted(((v ^ ((b+1)*0x9e3779b97f4a7c15)) & ((1<<64)-1)) for v in nums)[:3]
        results.append((b,tuple(vals)))
    return results

def is_test_path(rel):
    parts=[x.lower() for x in rel.parts]
    name=parts[-1]
    excluded={'tests','test','testing','testdata','test_data','fixtures','__pycache__','.tox','.venv','venv'}
    return any(x in excluded for x in parts[:-1]) or name.startswith(('test_','pytest_')) or name.endswith(('_test.py','_tests.py')) or name=='conftest.py'

def decorator_name(n):
    if isinstance(n,ast.Name):return n.id
    if isinstance(n,ast.Attribute):return decorator_name(n.value)+'.'+n.attr
    if isinstance(n,ast.Call):return decorator_name(n.func)
    return ''

def excluded_node(n,parents):
    if any(isinstance(q,ast.ClassDef) and (q.name.startswith('Test') or any('testcase' in decorator_name(b).lower() for b in q.bases)) for q in parents):return True
    if n.name.lower().startswith('test_') or n.name.lower().endswith('_test'):return True
    return any(decorator_name(d).lower().startswith(('pytest.','pytest_','unittest.')) or decorator_name(d).lower() in {'fixture','parametrize'} for d in n.decorator_list)

def eligible_functions(tree):
    result=[]
    def visit(node,parents):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            if excluded_node(node,parents):return
            result.append(node)
        if isinstance(node,ast.ClassDef) and (node.name.startswith('Test') or any('testcase' in decorator_name(b).lower() for b in node.bases)):return
        for child in ast.iter_child_nodes(node):visit(child,parents+[node])
    visit(tree,[])
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--repo',default=None)
    p.add_argument('--out',default=None)
    p.add_argument('--expected-commit',default=EXPECTED)
    p.add_argument('--min-lines',type=int,default=6)
    p.add_argument('--min-tokens',type=int,default=30)
    p.add_argument('--type3-threshold',type=float,default=.82)
    p.add_argument('--max-bucket',type=int,default=80)
    p.add_argument('--max-candidates',type=int,default=250000)
    p.add_argument('--max-python-files',type=int,default=0)
    a=p.parse_args()
    root=Path(a.root).expanduser().resolve()
    repo=Path(a.repo).resolve() if a.repo else root/'data/repositories/salt'
    if not (repo/'.git').exists(): raise SystemExit('FAIL: no git directory: '+str(repo))
    commit=git(repo,'rev-parse','HEAD')
    if a.expected_commit and commit!=a.expected_commit: raise SystemExit('FAIL: expected commit '+a.expected_commit+' but found '+commit)
    if git(repo,'status','--porcelain'): raise SystemExit('FAIL: repository has local changes; clean snapshot required')
    ts=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    out=Path(a.out).resolve() if a.out else root/'results/b-01'/ts
    if out.exists():raise SystemExit('FAIL: output directory already exists: '+str(out))
    out.mkdir(parents=True)
    print('='*90+'\nB-01 SALT PYTHON CLONE DISCOVERY | DISCOVERY ONLY | NOT GROUND TRUTH\n'+'='*90)
    print('Repository:',repo,'\nCommit:',commit,'\nRemote:',git(repo,'remote','get-url','origin'))
    files=sorted(x for x in repo.rglob('*.py') if '.git' not in x.parts and not is_test_path(x.relative_to(repo)))
    if a.max_python_files:files=files[:a.max_python_files]
    counts=Counter();functions=[];errors=[]
    for i,path in enumerate(files,1):
        try:source=path.read_text(encoding='utf-8',errors='replace');tree=ast.parse(source,filename=str(path))
        except (SyntaxError,ValueError,UnicodeError) as e:
            errors.append({'path':str(path.relative_to(repo)),'reason':str(e)[:300]});continue
        for node in eligible_functions(tree):
            counts['functions_found']+=1
            lines=node.end_lineno-node.lineno+1
            raw=ast.get_source_segment(source,node) or ''
            nt=len(re.findall(r'\w+|[^\s\w]',raw))
            if lines<a.min_lines or nt<a.min_tokens:
                counts['functions_too_short']+=1;continue
            try:
                normalized=norm(node,True)
                exact=norm(node,False)
            except (SyntaxError,ValueError,TypeError):counts['normalization_errors']+=1;continue
            functions.append({'id':len(functions),'file':str(path.relative_to(repo)),'name':node.name,'start':node.lineno,'end':node.end_lineno,'loc':lines,'tokens':nt,'exact_hash':sha(exact),'type2_hash':sha(normalized),'fingerprints':shingles(normalized)})
        if i%2000==0:print(f'[B-01] parsed {i:,}/{len(files):,} files, retained {len(functions):,} functions',flush=True)
    print(f'[B-01] parsing complete: {len(files):,} files, {len(functions):,} retained functions',flush=True)
    # deterministic exact and type2 grouping; bound pair explosion per group by representatives
    exactgroups=defaultdict(list);t2groups=defaultdict(list)
    for f in functions:
        exactgroups[f['exact_hash']].append(f['id']);t2groups[f['type2_hash']].append(f['id'])
    pairs={}; emitted=Counter(); group_sizes=Counter()
    def record(x,y,typ,sim):
        if x==y:return
        key=(min(x,y),max(x,y))
        prev=pairs.get(key)
        if prev is None or (typ<prev[0]):pairs[key]=(typ,sim)
    for typ,groups in [('type1',exactgroups),('type2',t2groups)]:
        for ids in groups.values():
            if len(ids)<2:continue
            group_sizes[typ]+=1
            limit=min(len(ids),a.max_bucket)
            if len(ids)>limit:counts['group_pair_cap_triggered']+=1
            for i in range(limit):
                for j in range(i+1,limit):record(ids[i],ids[j],typ,1.0)
    # candidate retrieval via deterministic shingle-index prefix keys + banded signatures
    buckets=defaultdict(list)
    for f in functions:
        for key in bands(f['fingerprints']):buckets[key].append(f['id'])
    candidate_pairs=set()
    for ids in buckets.values():
        if len(ids)>a.max_bucket:
            counts['oversized_lsh_buckets_skipped']+=1;continue
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                x,y=ids[i],ids[j]
                if x>y:x,y=y,x
                if (x,y) not in pairs:candidate_pairs.add((x,y))
                if len(candidate_pairs)>=a.max_candidates:break
            if len(candidate_pairs)>=a.max_candidates:break
        if len(candidate_pairs)>=a.max_candidates:break
    counts['type3_retrieved_candidates']=len(candidate_pairs)
    for x,y in sorted(candidate_pairs):
        a1,b1=functions[x]['fingerprints'],functions[y]['fingerprints']
        sim=len(a1&b1)/len(a1|b1) if a1 or b1 else 0
        if sim>=a.type3_threshold:record(x,y,'type3_candidate',round(sim,6))
    output=[]
    for (x,y),(kind,score) in sorted(pairs.items()):
        l,r=functions[x],functions[y]
        output.append({'type':kind,'similarity':score,'file_a':l['file'],'name_a':l['name'],'line_a':l['start'],'file_b':r['file'],'name_b':r['name'],'line_b':r['start'],'same_file':int(l['file']==r['file'])})
        emitted[kind]+=1
    atomic_csv(out/'clone_candidates.csv',list(output[0]) if output else ['type','similarity','file_a','name_a','line_a','file_b','name_b','line_b','same_file'],output)
    inventory=[{k:v for k,v in f.items() if k!='fingerprints'} for f in functions]
    atomic_csv(out/'function_inventory.csv', ['id','file','name','start','end','loc','tokens','exact_hash','type2_hash'],inventory)
    atomic_csv(out/'parse_errors.csv',['path','reason'],errors)
    summary={'status':'COMPLETE','role':'DISCOVERY_ONLY_NOT_GROUND_TRUTH','repository':str(repo),'remote':git(repo,'remote','get-url','origin'),'commit':commit,'files_scanned':len(files),'parse_errors':len(errors),'functions_found':counts['functions_found'],'functions_retained':len(functions),'functions_too_short':counts['functions_too_short'],'exact_clone_groups':sum(len(v)>1 for v in exactgroups.values()),'type2_clone_groups':sum(len(v)>1 for v in t2groups.values()),'pair_counts':dict(emitted),'total_candidate_pairs':len(output),'type3_retrieved_candidates':counts['type3_retrieved_candidates'],'oversized_lsh_buckets_skipped':counts['oversized_lsh_buckets_skipped'],'group_pair_cap_triggered':counts['group_pair_cap_triggered'],'settings':{'min_lines':a.min_lines,'min_tokens':a.min_tokens,'type3_threshold':a.type3_threshold,'max_bucket':a.max_bucket,'max_candidates':a.max_candidates,'max_python_files':a.max_python_files},'limitations':['Type-3 retrieval is approximate and non-exhaustive; no recall guarantee','Type-1/2 pair outputs capped to max_bucket per class; raw class counts retained','Candidate pairs are not independently verified ground truth','Latest snapshot differs from historic RQ3 dataset; cannot calculate historic detector recall','AST-based matching may be affected by Python parser compatibility']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    table('[1] REPOSITORY INVENTORY',['Python files','Parsed functions','Retained','Parse errors'],[[f'{len(files):,}',f'{counts["functions_found"]:,}',f'{len(functions):,}',len(errors)]])
    table('[2] CLONE CANDIDATES',['Type-1','Type-2 (non-Type-1)','Type-3 Candidate','All'],[[f'{emitted["type1"]:,}',f'{emitted["type2"]:,}',f'{emitted["type3_candidate"]:,}',f'{len(output):,}']])
    table('[3] AUDIT & LIMITS',['Exact groups','Normalized groups','LSH candidates','Skipped large buckets'],[[summary['exact_clone_groups'],summary['type2_clone_groups'],f'{counts["type3_retrieved_candidates"]:,}',counts['oversized_lsh_buckets_skipped']]])
    print('OUTPUT:',out,'\nWARNING: candidates are not ground truth; old 70 salt positive pairs not compared because historical function-ID mapping is unavailable.')
    print('[B-01] COMPLETE')
if __name__=='__main__':main()
