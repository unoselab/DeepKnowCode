#!/usr/bin/env python3
"""Stage a line-preserving, policy-filtered Salt Python source snapshot for NiCad.
Never modifies the original Git checkout. Test helpers are heuristically screened.
"""
import argparse, ast, csv, hashlib, json, re, shutil, subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPECTED='0fa7bec68485f8d89b7de2bd6b8743d040fb956b'
SKIP_DIR={'tests','test','testing','testdata','test_data','fixtures','__pycache__','.tox','.venv','venv','build','dist','site-packages'}
SKIP_FILE={'conftest.py','setup.py','noxfile.py','toxfile.py'}

def test_file(rel):
    ps=[p.lower() for p in rel.parts]
    name=ps[-1]
    return any(p in SKIP_DIR for p in ps[:-1]) or name in SKIP_FILE or name.startswith('test_') or name.endswith('_test.py') or name.startswith('pytest_') or name.endswith('_tests.py')

def dotted(x):
    if isinstance(x,ast.Name):return x.id
    if isinstance(x,ast.Attribute):return dotted(x.value)+'.'+x.attr
    if isinstance(x,ast.Call):return dotted(x.func)
    return ''

def is_test_decor(n):
    for d in n.decorator_list:
        name=dotted(d).lower()
        if name.startswith(('pytest.','pytest_','unittest.')) or name in {'fixture','parametrize','mark.parametrize','mark.fixture'} or '.pytest.' in name:
            return True
    return False

def is_test_func(n):
    return n.name.lower().startswith('test_') or n.name.lower().endswith('_test') or is_test_decor(n)

def is_test_class(n):
    return n.name.startswith('Test') or any('testcase' in dotted(b).lower() for b in n.bases) or is_test_decor(n)

def masking_spans(tree, lines):
    # Remove entire test class/function definitions, while preserving original line count.
    nodes=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.ClassDef) and is_test_class(n):nodes.append(n)
        elif isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and is_test_func(n):nodes.append(n)
    spans=[]
    for n in nodes:
        first=min([n.lineno]+[d.lineno for d in n.decorator_list])
        start=first-1;end=n.end_lineno
        if any(s<=start and end<=e for s,e in spans):continue
        spans=[(s,e) for s,e in spans if not(start<=s and e<=end)]
        spans.append((start,end))
    spans.sort()
    return spans

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--repo');p.add_argument('--out');p.add_argument('--expected-commit',default=EXPECTED)
    p.add_argument('--max-python-files',type=int,default=0)
    a=p.parse_args();root=Path(a.root).resolve();repo=Path(a.repo).resolve() if a.repo else root/'data/repositories/salt'
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if a.expected_commit and commit!=a.expected_commit:raise SystemExit('ERROR: unexpected Git commit '+commit)
    status=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    if status:raise SystemExit('ERROR: Git checkout not clean')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    out=Path(a.out).resolve() if a.out else root/'results/b-01'/stamp
    if out.exists():raise SystemExit('ERROR: output exists '+str(out))
    stage=out/'nicad_input'/'salt';stage.mkdir(parents=True)
    stats=Counter();audit=[]
    files=sorted(repo.rglob('*.py'))
    if a.max_python_files:files=files[:a.max_python_files]
    for f in files:
        rel=f.relative_to(repo);stats['files_seen']+=1
        if test_file(rel):stats['files_excluded_by_path']+=1;continue
        try:
            src=f.read_text(encoding='utf-8');tree=ast.parse(src)
        except (SyntaxError,ValueError,UnicodeError) as e:
            stats['files_excluded_parse_error']+=1;audit.append({'file':str(rel),'action':'skip_parse_error','detail':str(e)[:180]});continue
        lines=src.splitlines(keepends=True)
        spans=masking_spans(tree,lines)
        for start,end in spans:
            if start>=len(lines):continue
            # Entire definition becomes an indented 'pass' and blank lines, preserving subsequent line locations.
            indent=re.match(r'\s*',lines[start]).group().replace('\n','')
            lines[start]=indent+'pass\n'
            for idx in range(start+1,min(end,len(lines))):lines[idx]='\n'
        output=''.join(lines)
        try:ast.parse(output)
        except SyntaxError as e:
            stats['files_excluded_after_mask_error']+=1;audit.append({'file':str(rel),'action':'skip_after_mask','detail':str(e)[:180]});continue
        dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(output,encoding='utf-8')
        stats['files_staged']+=1;stats['test_declarations_masked']+=len(spans)
    (out/'filter_summary.json').write_text(json.dumps({'status':'PREPARED','repository':str(repo),'commit':commit,'staged_root':str(stage),'settings':{'excluded_dirs':sorted(SKIP_DIR),'excluded_files':sorted(SKIP_FILE),'exclude_test_named_functions':True,'exclude_pytest_decorators':True},'counts':dict(stats),'limitations':['Heuristic exclusion may retain nonstandard tests and exclude production code with test-like names','Staged line numbers preserved but masked source is not identical to original','NiCad may reject some valid modern Python syntax']},indent=2)+'\n')
    with (out/'filter_audit.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=['file','action','detail']);w.writeheader();w.writerows(audit)
    print('B-01 v2 NiCad source staged:',stage)
    for k,v in sorted(stats.items()):print('  ',k,':',v)
    print('OUTPUT_DIR='+str(out));print('STAGE_DIR='+str(stage))
if __name__=='__main__':main()
