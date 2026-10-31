#!/usr/bin/env python3
"""A-17 v3: separately audit A-08 population, M1/M2/M3 selected splits and curriculum.
No inference; read-only; NEVER combine adaptation with held-out subject projects.
"""
import argparse, collections, csv, hashlib, json
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def table(headers, rows, right=()):
    text=[[str(v) for v in row] for row in rows]
    widths=[max([len(str(h))]+[len(r[i]) for r in text]) for i,h in enumerate(headers)]
    bar='+-'+'-+-'.join('-'*x for x in widths)+'-+'
    fmt=lambda row:'| '+' | '.join(v.rjust(widths[i]) if i in right else v.ljust(widths[i]) for i,v in enumerate(row))+' |'
    return '\n'.join([bar,fmt(headers),bar]+[fmt(row) for row in text]+[bar])

def latest_prepared(root, experiment, explicit):
    if explicit:
        path=Path(explicit)
        if path.name!='prepared': path=path/'prepared'
        return path
    candidates=sorted((root/'results'/experiment).glob('*/prepared'))
    if not candidates: raise FileNotFoundError(f'Missing {experiment} prepared directory; set A17_{experiment.upper().replace("-", "")}_PREPARED')
    return candidates[-1]

def count_population(root):
    # Counts original A-08 cleaned split files; no project assignment is available here.
    rows=[]
    for corpus in ('java_source','java_mixed','python_mixed'):
        for split,filename in (('train','train.txt'),('validation','valid.txt')):
            path=root/'data/training/clean'/corpus/filename
            if not path.is_file():raise FileNotFoundError(path)
            n=0
            with path.open('r',encoding='utf-8') as f:
                for line in f:
                    if line.strip(): n+=1
            rows.append({'scope':'A08_cleaned','model':'population','split':split,'corpus':corpus,
                         'pairs':n,'positive_pairs':None,'negative_pairs':None,'sha256':digest(path)})
    return rows

def read_selected(prepared, model):
    manifest_path=prepared/'manifest.json'
    meta=json.loads(manifest_path.read_text(encoding='utf-8'))
    rows=[]; keys={}; split_meta={}
    for split in ('train','validation'):
        path=prepared/f'{split}.jsonl'
        hist=collections.Counter(); examples=collections.Counter(); seen=[]
        with path.open('r',encoding='utf-8') as f:
            for line in f:
                if not line.strip():continue
                x=json.loads(line)
                corpus=x.get('corpus', 'java_source' if model=='M1' else None)
                if corpus is None:raise ValueError(f'Missing corpus {path}')
                label=int(x['label'])
                if label not in (0,1): raise ValueError(f'Invalid label {label}')
                hist[(corpus,label)]+=1
                key=(corpus,str(x['id1']),str(x['id2']),label)
                examples[key]+=1
                seen.append(key)
        expected=meta.get('sample',{}).get(split,{})
        if len(seen)!=expected.get('selected'):raise ValueError(f'Manifest selected count mismatch {path}')
        check=expected.get('sample_sha256') or expected.get('sha256')
        if check and digest(path)!=check:raise ValueError(f'Manifest SHA mismatch {path}')
        keys[split]=examples
        split_meta[split]={'pairs':len(seen),'sha256':digest(path)}
        for corpus in sorted({c for c,y in hist}):
            n0,n1=hist[(corpus,0)],hist[(corpus,1)]
            rows.append({'scope':'selected','model':model,'split':split,'corpus':corpus,
                         'pairs':n0+n1,'positive_pairs':n1,'negative_pairs':n0,'sha256':digest(path)})
    return rows,keys,meta,split_meta

def write_csv(path,rows):
    fields=['scope','model','split','corpus','pairs','positive_pairs','negative_pairs','sha256']
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def run(args):
    root=args.root.resolve(); output=args.output_dir
    if not output.is_dir():raise FileNotFoundError('A-17 evaluation output directory missing: '+str(output))
    if (output/'adaptation_summary.json').exists():raise FileExistsError('Refusing to overwrite adaptation audit files')
    prepared={m:latest_prepared(root,exp,getattr(args,m.lower()+'_prepared')) for m,exp in [('M1','a-11'),('M2','a-13'),('M3','a-15')]}
    pop=count_population(root)
    selected=[]; keys={}; manifests={}; split_meta={}
    for m in ('M1','M2','M3'):
        rows,keys[m],manifests[m],split_meta[m]=read_selected(prepared[m],m)
        selected+=rows
    match={split:keys['M2'][split]==keys['M3'][split] for split in ('train','validation')}
    if not all(match.values()): raise ValueError('M2 and M3 selected pair multisets DO NOT MATCH')
    # A-13 and A-15 must use the same M1 initial adapter.
    if manifests['M2'].get('initial_adapter_sha256') != manifests['M3'].get('initial_adapter_sha256'):
        raise ValueError('M2/M3 initial M1 adapter hash mismatch')
    if split_meta['M2']['validation']['sha256'] != split_meta['M3']['validation']['sha256']:
        raise ValueError('M2/M3 validation ordering/hash mismatch')
    schedule=manifests['M3'].get('curriculum_schedule',[])
    if not schedule: raise ValueError('Missing M3 curriculum schedule')
    phase_rows=[]
    for phase in schedule:
        phase_rows.append({'phase':phase['phase'],'steps':phase['steps'], 'source':phase['source'],
                           'java_target':phase['java_target'],'python_target':phase['python_target'],
                           'target_fraction':phase['target_fraction']})
    totals={corpus:sum(x[corpus] for x in phase_rows) for corpus in ('source','java_target','python_target')}
    if totals != {'source':11000,'java_target':800,'python_target':200}:
        raise ValueError('Unexpected curriculum totals '+repr(totals))
    # Validate exact per-corpus M3 counts against phase totals.
    m3={r['corpus']:r['pairs'] for r in selected if r['model']=='M3' and r['split']=='train'}
    if (m3.get('java_source'),m3.get('java_mixed'),m3.get('python_mixed')) != (11000,800,200):
        raise ValueError('M3 sample does not match phase totals')
    write_csv(output/'adaptation_distribution.csv',pop+selected)
    with (output/'adaptation_curriculum.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(phase_rows[0]));w.writeheader();w.writerows(phase_rows)
    payload={'status':'COMPLETE','scope':'A-08 cleaned split populations and A-11/A-13/A-15 sampled adaptation splits',
             'population':pop,'selected':selected,'m2_m3_multiset_identical':match,
             'm2_m3_validation_file_identical':True, 'prepared_directories':{m:str(path) for m,path in prepared.items()},
             'prepared_manifest_sha256':{m:digest(path/'manifest.json') for m,path in prepared.items()},
             'curriculum_schedule':phase_rows,
             'limitations':['The A-08 corpora do not identify 72 held-out subject project memberships.',
             'java_mixed/python_mixed are corpus tags, not guaranteed exact language/subject-project labels.',
             'Original cleaned mixed corpora contain source-overlapping examples; selected target rows are filtered by training scripts.',
             'Function-level train-validation overlap may remain; do not infer leak-free splits from this distribution audit.',
             'Held-out ground truth is read only by the separate A-17 project audit.']}
    (output/'adaptation_summary.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    lines=['# A-17 Adaptation training/validation distribution','',
           'These counts are distinct from the 39,002-pair held-out evaluation set.','',
           '## A-08 cleaned corpus split populations','',
           '| Corpus | Split | Pair rows |','|---|---|---:|']
    for r in pop:lines.append(f"| {r['corpus']} | {r['split']} | {r['pairs']:,} |")
    lines+=['','## Selected training/validation samples','','| Model | Split | Corpus | Pairs | Clone | Non-clone |','|---|---|---|---:|---:|---:|']
    for r in selected: lines.append(f"| {r['model']} | {r['split']} | {r['corpus']} | {r['pairs']:,} | {r['positive_pairs']:,} | {r['negative_pairs']:,} |")
    lines+=['','## M3 Curriculum phases','','| Phase | Steps | Source | Java target | Python target | Target % |','|---|---|---:|---:|---:|---:|']
    for r in phase_rows: lines.append(f"| {r['phase']} | {r['steps'][0]}-{r['steps'][1]} | {r['source']:,} | {r['java_target']:,} | {r['python_target']:,} | {100*r['target_fraction']:.2f}% |")
    lines+=['','M2 and M3 selected training/validation pair multisets verified identical (order-independent).',
            'Source/Target is a corpus/sampling designation, not one of the held-out subject projects.','']
    (output/'adaptation_paper_table.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'+'='*90+'\nA-17  ADAPTATION TRAINING / VALIDATION DATASET (SEPARATE FROM HELD-OUT TEST)\n'+'='*90)
    print('\n[6] A-08 CLEANED CORPUS POPULATION (BEFORE EXPERIMENT SAMPLING)')
    print(table(['Corpus','Split','Pairs'],[[r['corpus'],r['split'],f"{r['pairs']:,}"] for r in pop],(2,)))
    print('\n[7] ACTUALLY SELECTED TRAIN / VALIDATION PAIRS')
    print(table(['Model','Split','Corpus','Pairs','Clone (+)','Non-clone (-)'],
                [[r['model'],r['split'],r['corpus'],f"{r['pairs']:,}",f"{r['positive_pairs']:,}",f"{r['negative_pairs']:,}"] for r in selected],(3,4,5)))
    print('\n[8] M3 CURRICULUM PHASES')
    print(table(['Phase','Steps','Source','Java target','Python target','Target %'],
                [[r['phase'],f"{r['steps'][0]}-{r['steps'][1]}",f"{r['source']:,}",r['java_target'],r['python_target'],f"{r['target_fraction']*100:.2f}%"] for r in phase_rows],(0,2,3,4,5)))
    print('\n[9] CROSS-EXPERIMENT INVARIANTS')
    print('M2/M3 selected train pair multisets identical: PASS')
    print('M2/M3 selected validation pair multisets identical: PASS')
    print('M2/M3 validation JSONL SHA-256 identical: PASS')
    print('M2/M3 initial M1 adapter hash identical: PASS')
    print('\n[10] OUTPUT FILES')
    for x in ('adaptation_summary.json','adaptation_distribution.csv','adaptation_curriculum.csv','adaptation_paper_table.md'):
        print('  '+str(output/x))
    print('\nNOTE: adaptation source/target corpora are NOT the 72 held-out subject projects.')
    print('NOTE: population and sampled counts are different denominators.\n[A-17] ADAPTATION AUDIT COMPLETE',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    for m in ('m1','m2','m3'):p.add_argument('--'+m+'-prepared')
    run(p.parse_args())
if __name__=='__main__':main()
