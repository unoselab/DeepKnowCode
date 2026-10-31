#!/usr/bin/env python3
"""B-04: read-only, descriptive audit of NiCad Python parse failures from B-03."""
import argparse, ast, collections, csv, datetime as dt, hashlib, json, pathlib, re, sys, tokenize, io

FEATURES = ('match_case','walrus','fstring','async_await','positional_only','type_parameter_syntax','union_type_hint','except_star','decorators','annotations','non_ascii','multiline_string','nested_functions')

def inspect_source(src):
    found=set(); samples={}; counts=collections.Counter()
    try: tree=ast.parse(src)
    except (SyntaxError,ValueError) as e: return set(),{}, {},str(e)
    for n in ast.walk(tree):
        names=[]
        if isinstance(n,ast.Match): names.append('match_case')
        if isinstance(n,ast.NamedExpr): names.append('walrus')
        if isinstance(n,ast.JoinedStr): names.append('fstring')
        if isinstance(n,(ast.AsyncFunctionDef,ast.Await,ast.AsyncFor,ast.AsyncWith)): names.append('async_await')
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
            if n.args.posonlyargs: names.append('positional_only')
            if getattr(n,'type_params',[]): names.append('type_parameter_syntax')
            if n.decorator_list: names.append('decorators')
            if n.returns or any(a.annotation for a in n.args.posonlyargs+n.args.args+n.args.kwonlyargs): names.append('annotations')
        if isinstance(n,ast.AnnAssign): names.append('annotations')
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.BitOr) and isinstance(getattr(n,'parent',None),ast.expr): pass
        if isinstance(n,ast.TryStar): names.append('except_star')
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
            if any(isinstance(c,(ast.FunctionDef,ast.AsyncFunctionDef)) for c in ast.walk(n) if c is not n): names.append('nested_functions')
        for name in names:
            counts[name]+=1;found.add(name);samples.setdefault(name,getattr(n,'lineno',1))
    # Union annotation: only check the annotation AST, not arbitrary bitwise OR
    for n in ast.walk(tree):
        annotations=[]
        if isinstance(n,ast.AnnAssign): annotations=[n.annotation]
        if isinstance(n,ast.arg) and n.annotation: annotations=[n.annotation]
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.returns: annotations=[n.returns]
        if any(isinstance(t,ast.BinOp) and isinstance(t.op,ast.BitOr) for a in annotations for t in ast.walk(a)):
            found.add('union_type_hint');samples.setdefault('union_type_hint',getattr(n,'lineno',1));counts['union_type_hint']+=1
    if any(ord(c)>127 for c in src): found.add('non_ascii'); samples['non_ascii']=next((i for i,l in enumerate(src.splitlines(),1) if not l.isascii()),1);counts['non_ascii']=1
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type==tokenize.STRING and ('\n' in tok.string or tok.string.startswith(('"""',"'''"))):
                found.add('multiline_string');samples.setdefault('multiline_string',tok.start[0]);counts['multiline_string']+=1
    except (tokenize.TokenError,IndentationError): pass
    return found,samples,dict(counts),None

def read_csv(path):
    with path.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def category(row):
    return (row.get('nicad_category') or '').strip().upper()

def excerpt(src,line,context=3,max_lines=9,max_chars=170):
    lines=src.splitlines();start=max(1,line-context);stop=min(len(lines),line+context,max(start+max_lines-1,line+context))
    return '\n'.join(f'{i:>5} {lines[i-1][:max_chars]}' for i in range(start,stop+1))

def write_csv(path, rows, fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=pathlib.Path,default=None)
    ap.add_argument('--input-result',default='results/b-03/20261031-212736-582683')
    ap.add_argument('--source-mode',choices=['auto','staged','original'],default='auto')
    ap.add_argument('--examples',type=int,default=12)
    ap.add_argument('--max-snippet-lines',type=int,default=9)
    a=ap.parse_args(); root=(a.root or pathlib.Path(__file__).resolve().parents[2]).resolve()
    inp=pathlib.Path(a.input_result);inp=inp if inp.is_absolute() else root/inp
    input_csv=inp/'file_parse_audit.csv'
    if not input_csv.is_file(): ap.error('Missing B-03 CSV: '+str(input_csv))
    rows=read_csv(input_csv)
    if not rows or 'nicad_category' not in rows[0]: ap.error('Input is not B-03 classified file_parse_audit.csv')
    repository=root/'data/repositories/salt'; staged=root/'results/b-01/_staging/salt'
    if not repository.is_dir(): ap.error('Missing Salt repository: '+str(repository))
    if a.source_mode=='staged' and not staged.is_dir(): ap.error('Staged source missing: '+str(staged))
    timestamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S-%f')+'-'+str(__import__('os').getpid())
    out=root/'results/b-04'/timestamp
    out.mkdir(parents=True,exist_ok=False)
    audit=[];record_by_category=collections.Counter();feature_n=collections.defaultdict(collections.Counter);missing=[];examples=[]
    for r in rows:
        rel=pathlib.PurePosixPath((r.get('file') or '').replace('\\','/'))
        if rel.is_absolute() or '..' in rel.parts or not rel.parts: raise RuntimeError('Unsafe file path in input CSV: '+str(rel))
        cat=category(r);record_by_category[cat]+=1
        srcfile=(staged/rel if a.source_mode!='original' and (staged/rel).is_file() else repository/rel)
        # When auto falls back to original, explicitly record so cross-source conclusions are not made silently.
        if a.source_mode=='staged' and not (staged/rel).is_file():srcfile=staged/rel
        source_kind='staged' if srcfile.is_relative_to(staged) else 'original'
        item={'file':str(rel),'nicad_category':cat,'source_kind':source_kind,'source_exists':int(srcfile.is_file()),'python_ast_now':'','python_ast_error':'','lines':0,'functions':0,'features':'','feature_counts_json':'{}','sha256':''}
        if not srcfile.is_file():missing.append(str(rel));audit.append(item);continue
        try: src=srcfile.read_text(encoding='utf-8')
        except (UnicodeError,OSError) as e:item['python_ast_now']='READ_ERROR';item['python_ast_error']=str(e);audit.append(item);continue
        item['sha256']=hashlib.sha256(src.encode()).hexdigest();item['lines']=len(src.splitlines())
        fs,lines,counts,err=inspect_source(src)
        item['python_ast_now']='FAIL' if err else 'PASS';item['python_ast_error']=err or ''
        item['features']=';'.join(sorted(fs));item['feature_counts_json']=json.dumps(counts,sort_keys=True)
        if not err:
            tree=ast.parse(src);item['functions']=sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for n in ast.walk(tree))
        for name in fs:feature_n[cat][name]+=1
        if cat=='NICAD_PARSE_FAILURE':
            # Example selection is deterministic input-file order, and line is a feature occurrence, NOT proven failure position.
            chosen=next(((f,lines[f]) for f in FEATURES if f in lines),('unspecified',max(1,min(item['lines'],1))))
            examples.append({'file':str(rel),'feature':chosen[0],'line':chosen[1],'source_kind':source_kind,'excerpt':excerpt(src,chosen[1],max_lines=a.max_snippet_lines),'nicad_error':r.get('nicad_error_excerpt','')})
        audit.append(item)
    fields=['file','nicad_category','source_kind','source_exists','python_ast_now','python_ast_error','lines','functions','features','feature_counts_json','sha256']
    write_csv(out/'syntax_feature_audit.csv',audit,fields)
    write_csv(out/'parse_failure_features.csv',[x for x in audit if x['nicad_category']=='NICAD_PARSE_FAILURE'],fields)
    comp=[]
    for f in FEATURES:
        row={'feature':f}
        for c in ['NICAD_PARSE_FAILURE','NO_FUNCTIONS_EXTRACTED','SUCCESS']:
            total=sum(x['nicad_category']==c and x['source_exists'] for x in audit)
            count=feature_n[c][f]
            row[c+'_files']=count;row[c+'_total']=total;row[c+'_pct']=round(100*count/total,2) if total else ''
        comp.append(row)
    compare_fields=['feature']+[f'{c}_{suffix}' for c in ['NICAD_PARSE_FAILURE','NO_FUNCTIONS_EXTRACTED','SUCCESS'] for suffix in ('files','total','pct')]
    write_csv(out/'syntax_feature_comparison.csv',comp,compare_fields)
    n=max(0,a.examples)
    # Round-robin features to avoid showing one feature only
    chosen=[];used=set()
    for f in FEATURES:
        for e in examples:
            if e['feature']==f and e['file'] not in used:
                chosen.append(e);used.add(e['file']);break
            if len(chosen)>=n:break
    for e in examples:
        if len(chosen)>=n:break
        if e['file'] not in used:chosen.append(e);used.add(e['file'])
    chosen=chosen[:n]
    write_csv(out/'manual_review_examples.csv',[{k:v for k,v in e.items() if k!='excerpt'} for e in chosen],['file','feature','line','source_kind','nicad_error'])
    summary={'status':'COMPLETE','experiment':'B-04','version':'v1','input_result':str(inp),'source_mode':a.source_mode,'repo_commit_expected':'0fa7bec68485f8d89b7de2bd6b8743d040fb956b','rows':len(rows),'categories':dict(record_by_category),'missing_source_files':len(missing),'python_ast_current_failures':sum(x['python_ast_now']=='FAIL' for x in audit),'sample_count':len(chosen),'notes':['Syntax feature presence is association, not proof of NiCad parse-failure cause','Snippet line is illustrative and not an NiCad error location','B-03 file classification inherited without rerunning NiCad','Staged source can be modified by test masking; original source may not match NiCad input']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    report=['# B-04 Python Grammar Compatibility Audit','',f'- Input: `{inp}`',f'- Audited: {len(rows)}',f'- Categories: {dict(record_by_category)}',f'- Missing sources: {len(missing)}','', '## Syntax feature frequency','', '| Feature | NiCad parse fail | Success |','|---|---:|---:|']
    for r in comp: report.append(f"| {r['feature']} | {r['NICAD_PARSE_FAILURE_files']}/{r['NICAD_PARSE_FAILURE_total']} ({r['NICAD_PARSE_FAILURE_pct']}%) | {r['SUCCESS_files']}/{r['SUCCESS_total']} ({r['SUCCESS_pct']}%) |")
    report+=['','## Manual review examples','']
    for e in chosen:report.extend([f"### {e['file']}",f"- Candidate feature: `{e['feature']}` at line {e['line']} ({e['source_kind']})",f"- NiCad message: `{e['nicad_error'][:250]}`",'```python',e['excerpt'],'```',''])
    report.extend(['## Limitations','- Feature frequency does not demonstrate causality.','- NiCad warning text generally lacks precise error locations; snippets are illustrative.','- Exact B-01 extraction failures require original NiCad diagnostics for individual root causes.',''])
    (out/'report.md').write_text('\n'.join(report),encoding='utf-8')
    print('='*88);print('B-04 v1 | NiCad Python Grammar Compatibility Audit | READ ONLY');print('='*88)
    print(f'INPUT: {inp}\nFILES: {len(rows)} | missing: {len(missing)} | Python AST current fails: {summary["python_ast_current_failures"]}')
    for c in ['NICAD_PARSE_FAILURE','NO_FUNCTIONS_EXTRACTED','SUCCESS']: print(f'  {c:<30} {record_by_category[c]:>5}')
    print('\n[FEATURE FREQUENCY] Parse Failure vs Success (feature presence, not causality)')
    print(f'{"Feature":<28} {"Failure":>12} {"Success":>12}')
    for r in comp:print(f"{r['feature']:<28} {r['NICAD_PARSE_FAILURE_files']:>5}/{r['NICAD_PARSE_FAILURE_total']:<6} {r['SUCCESS_files']:>5}/{r['SUCCESS_total']:<6}")
    print(f'\n[MANUAL REVIEW EXAMPLES] {len(chosen)} of {record_by_category["NICAD_PARSE_FAILURE"]} parse failures')
    for i,e in enumerate(chosen,1):
        print(f'\n--- Example {i}/{len(chosen)}: {e["file"]} ---')
        print(f'Feature hint: {e["feature"]} | line: {e["line"]} | source: {e["source_kind"]}')
        print(f'NiCad: {e["nicad_error"][:240]}')
        print(e['excerpt'])
    print('\nNOTICE: Examples are NOT confirmed root-cause locations.');print(f'[B-04 v1] COMPLETE OUTPUT_DIR={out}')

if __name__=='__main__':main()
