#!/usr/bin/env python3
"""Read-only D-05a evaluation evidence inventory. Does not infer labels or authorize inference."""
import argparse, csv, hashlib, json, os, sys
from pathlib import Path
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

CODE = ('java', 'python')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def inspect_jsonl(p):
    count=0; keys=Counter(); ids=Counter(); duplicates=[]; examples=[]; missing_code=0; code_hashes=set(); missing_id=0
    with p.open(encoding='utf-8-sig') as f:
        for ln,line in enumerate(f,1):
            if not line.strip(): continue
            row=json.loads(line)
            if not isinstance(row,dict): raise ValueError(f'{p}:{ln}: expected object')
            count+=1; keys.update(row.keys())
            if len(examples)<2: examples.append({k: (str(v)[:120] if k!='code' else f'<code length={len(str(v))}>') for k,v in row.items()})
            if not isinstance(row.get('code'),str):missing_code+=1
            else: code_hashes.add(hashlib.sha256(''.join(row['code'].split()).encode()).hexdigest())
            if 'id' not in row:missing_id+=1
            else:
                k=str(row['id']); ids[k]+=1
                if ids[k]==2 and len(duplicates)<10: duplicates.append(k)
    return {'path':str(p),'count':count,'sha256':sha(p),'fields':dict(keys),'missing_code':missing_code,'missing_id':missing_id,'duplicate_ids_first_10':duplicates,'unique_code_whitespace_hashes':len(code_hashes),'sample_structure_redacted':examples},set(ids),code_hashes

def inspect_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:
        head=f.read(8192); f.seek(0)
        try:dialect=csv.Sniffer().sniff(head,delimiters=',;\t')
        except csv.Error:dialect=csv.excel
        reader=csv.DictReader(f,dialect=dialect)
        columns=reader.fieldnames or []
        n=0; rows=[]; nulls=Counter(); uniq={c:set() for c in columns if any(x in c.lower() for x in ('label','clone','type','refactor'))}
        for row in reader:
            n+=1
            if len(rows)<2:rows.append({k:str(v)[:100] for k,v in row.items()})
            for c in columns:
                v=row.get(c)
                if v is None or not str(v).strip():nulls[c]+=1
                if c in uniq and len(uniq[c])<12: uniq[c].add(str(v))
    return {'name':p.name,'sha256':sha(p),'rows':n,'columns':columns,'missing_by_field':dict(nulls),'possible_label_values_sample':{k:sorted(v) for k,v in uniq.items()},'sample_rows':rows}

def pair_audit(p, split):
    pairs=set(); bycorpus=Counter(); func=set(); dup=0; n=0
    with p.open(encoding='utf-8') as f:
        for ln,line in enumerate(f,1):
            if not line.strip():continue
            r=json.loads(line); corpus=str(r['corpus']); a=str(r['id1']); b=str(r['id2'])
            key=(corpus,*sorted((a,b))); dup+=key in pairs; pairs.add(key)
            func.add((corpus,a));func.add((corpus,b));bycorpus[corpus]+=1;n+=1
    return {'split':split,'sha256':sha(p),'pairs':n,'duplicate_unordered_pairs':dup,'by_corpus':dict(bycorpus)}, pairs,func

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path.cwd());ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();root=args.root.resolve();out=args.out.resolve()
    if out.exists():sys.exit(f'BLOCKED: output exists: {out}')
    code_dir=root/'data/evaluation/code';gt=root/'data/evaluation/ground_truth';prep=root/'data/training/prepared/balanced_multilingual';reg=root/'models/registry/benchmark-20261108-v1'
    required=[code_dir/f'{x}.jsonl' for x in CODE]+[prep/f'{s}.jsonl' for s in ('train','validation')]+[reg/'SHA256SUMS']
    for path in required:
        if not path.is_file():sys.exit(f'BLOCKED: missing {path}')
    code={}; all_ids={}; all_hashes={}
    for lang in CODE:code[lang],all_ids[lang],all_hashes[lang]=inspect_jsonl(code_dir/f'{lang}.jsonl')
    summaries={}; pairs={}; funcs={}
    for name in ('train','validation'):summaries[name],pairs[name],funcs[name]=pair_audit(prep/f'{name}.jsonl',name)
    gt_files={}
    for lang in CODE:
        files=sorted((gt/lang).glob('*.csv')) if (gt/lang).is_dir() else []
        gt_files[lang]=[inspect_csv(p) for p in files]
    registry={}
    for mid,dirname in [('M0','m0-base'),('M1','m1-source-lora'),('M2','m2-standard-mixed-lora'),('M3','m3-curriculum-lora')]:
        dr=reg/dirname; weight=dr/'adapter_model.safetensors'; conf=dr/'adapter_config.json'; model=dr/'model.json'
        registry[mid]={'exists':dr.is_dir(),'adapter_sha256':sha(weight) if weight.is_file() else None,'adapter_config_sha256':sha(conf) if conf.is_file() else None,'model_json_sha256':sha(model) if model.is_file() else None}
    unexpected=[]
    for k in ('M1','M2','M3'):
        if registry[k]['adapter_sha256'] is None:unexpected.append(k+' missing adapter')
    common_pairs=pairs['train']&pairs['validation']; common_funcs=funcs['train']&funcs['validation']
    checklist={
      'code_corpus_present':all(code[x]['count']>0 and not code[x]['missing_code'] for x in CODE),
      'ground_truth_csv_present':all(len(gt_files[x])>0 for x in CODE),
      'model_registry_present':not unexpected,
      'train_validation_pair_disjoint':not common_pairs,
      'train_validation_function_id_disjoint':not common_funcs,
      'ground_truth_pair_schema_validated':False,
      'heldout_pair_label_join_validated':False,
      'test_train_near_duplicate_leakage_validated':False,
      'm0_m3_identical_prediction_protocol_validated':False,
    }
    report={'created_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),'status':'D05A_INVENTORY_COMPLETE_EVALUATION_NOT_YET_AUTHORIZED','root':str(root),'registry':registry,'evaluation_code':code,'ground_truth':{'by_language':gt_files,'file_counts':{x:len(gt_files[x]) for x in CODE},'row_counts':{x:sum(d['rows'] for d in gt_files[x]) for x in CODE}},'training_splits':summaries,'train_validation_exact_unordered_pair_overlap':len(common_pairs),'train_validation_function_id_overlap':len(common_funcs),'train_validation_overlap_examples':{'pairs':[list(x) for x in sorted(common_pairs)[:10]],'function_ids':[list(x) for x in sorted(common_funcs)[:10]]},'checklist':checklist,'limitations':['CSV column inspection is descriptive only; no join to code IDs is asserted.','Code corpora are not assumed to contain pair-level binary clone labels.','Train/validation same-ID checks are performed within corpus namespaces; cross-dataset code leakage requires evidence of code mapping.','No predictions, labels, or full model evaluations are produced by this script.']}
    out.mkdir(parents=True,exist_ok=False)
    target=out/'evaluation_inventory.json'; target.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    manifest={'created_at_chicago':report['created_at_chicago'],'status':report['status'],'evaluation_inventory_sha256':sha(target)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'code_rows':{x:code[x]['count'] for x in CODE},'ground_truth_files':report['ground_truth']['file_counts'],'train_validation_pair_overlap':len(common_pairs),'out':str(out)},ensure_ascii=False))

if __name__=='__main__':main()
