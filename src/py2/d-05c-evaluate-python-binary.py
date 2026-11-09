#!/usr/bin/env python3
"""D-05c v1: evaluate frozen D-05bj v2 binary Python clone ground truth on M0-M3.

Strictly read-only on ground truth, model registry, and training inputs.
Uses a pinned StarCoder2-7B base, optional PEFT LoRA, one-token YES/NO next-token
logits and lossless Cartesian windowing. Evaluation is conditional on the
reviewer label convention; this script does NOT independently prove labels.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v1'
BASE_MODEL = 'bigcode/starcoder2-7b'
BASE_REVISION = 'bb9afde76d7945da5745592525db122d4d729eb1'  # D-02 pinned revision
N_TOTAL = 7942
N_PER_CLASS = 3971
REGISTRY = {'M0': 'm0-base', 'M1': 'm1-source-lora',
            'M2': 'm2-standard-mixed-lora', 'M3': 'm3-curriculum-lora'}
PRED_FIELDS = ['pair_id', 'project', 'true_label', 'predicted_label',
               'mean_logit_margin', 'n_windows', 'original_input_tokens',
               'windowed', 'model', 'decision_threshold']

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def digest_code(code):
    return hashlib.sha256(code.encode('utf-8')).hexdigest()

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def atomic_json(path, obj):
    tmp = path.with_name(path.name + '.partial')
    require(not tmp.exists() and not path.exists(), 'Refusing to overwrite '+str(path))
    write_json(tmp, obj)
    tmp.replace(path)

def now():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()

def verify_manifest(gt):
    manifest_path = gt / 'manifest.json'
    require(manifest_path.is_file(), 'Missing D-05bj manifest.json')
    man = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(isinstance(man.get('files'), dict), 'Manifest files map missing')
    needed = {'positive_verified_pairs.jsonl', 'negative_verified_pairs.jsonl',
              'verified_pairs.jsonl', 'excluded_negative_candidate_clones.jsonl',
              'annotation_audit.json', 'leakage_audit.json'}
    require(needed <= set(man['files']), 'Missing D-05bj artifact(s) in manifest')
    for name, metadata in man['files'].items():
        require('/' not in name and name not in ('.', '..'), 'Unsafe manifest filename: '+name)
        file = gt / name
        require(file.is_file() and sha(file) == metadata['sha256'],
                'Ground truth SHA256 mismatch '+name)
        if 'bytes' in metadata:
            require(file.stat().st_size == metadata['bytes'], 'Ground truth byte mismatch '+name)
    return sha(manifest_path)

def iter_jsonl(path):
    with path.open('r', encoding='utf-8-sig') as f:
        for n, line in enumerate(f, 1):
            require(bool(line.strip()), f'Blank line in {path}:{n}')
            value = json.loads(line)
            require(isinstance(value, dict), f'Not an object in {path}:{n}')
            yield value

def load_gt(gt):
    manifest_sha = verify_manifest(gt)
    merged_file = gt / 'verified_pairs.jsonl'
    pos_file = gt / 'positive_verified_pairs.jsonl'
    neg_file = gt / 'negative_verified_pairs.jsonl'
    counts = collections.Counter()
    project_counts = collections.Counter()
    seen_ids, seen_pair_keys = set(), set()
    out = []
    for item in iter_jsonl(merged_file):
        pid, project, label = item.get('pair_id'), item.get('project'), item.get('gold_label')
        require(isinstance(pid, str) and pid and pid not in seen_ids,
                'Duplicate/invalid pair_id: '+str(pid))
        require(isinstance(project, str) and project and project != 'azure-sdk-for-python',
                'Invalid project: '+str(pid))
        require(type(label) is int and label in (0, 1), 'Invalid gold label: '+pid)
        require(item.get('evaluation_eligible') is True, 'Pair not evaluation eligible: '+pid)
        codes = []
        hashes = []
        for s in ('left', 'right'):
            side = item.get(s)
            require(isinstance(side, dict), 'Missing side '+pid+'/'+s)
            code = side.get('source_code')
            require(isinstance(code, str) and code.strip(), 'Missing code '+pid+'/'+s)
            code_sha = side.get('code_sha256')
            require(digest_code(code) == code_sha, 'Source code SHA256 mismatch: '+pid+'/'+s)
            codes.append(code)
            hashes.append(code_sha)
        pair_key = tuple(sorted(hashes))
        require(pair_key not in seen_pair_keys, 'Repeated code pair across records: '+pid)
        seen_pair_keys.add(pair_key)
        seen_ids.add(pid)
        out.append({'pair_id':pid, 'project':project, 'label':label,
                    'language':'python', 'code_a':codes[0], 'code_b':codes[1]})
        counts[label] += 1
        project_counts[project] += 1
    require(len(out) == N_TOTAL and counts == {0:N_PER_CLASS, 1:N_PER_CLASS},
            'Expected balanced 3,971 + 3,971 heldout pairs')
    require(len(project_counts) == 24, 'Expected 24 projects')
    # Verify merged content consists of exactly the two validated source files.
    def canonical(row):
        return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    base = collections.Counter(canonical(row) for file in (pos_file, neg_file)
                               for row in iter_jsonl(file))
    combined = collections.Counter(canonical(row) for row in iter_jsonl(merged_file))
    require(base == combined, 'Merged rows differ from verified positive/negative inputs')
    return out, {'manifest_sha256':manifest_sha,
                 'verified_pairs_sha256':sha(merged_file),
                 'project_counts':dict(sorted(project_counts.items())),
                 'class_counts':{'0':counts[0], '1':counts[1]}}

def validate_registry(root):
    reg = root / 'models/registry/benchmark-20261108-v1'
    sumfile = reg / 'SHA256SUMS'
    require(sumfile.is_file(), 'Missing registry SHA256SUMS '+str(sumfile))
    expected = {}
    for line in sumfile.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        pieces = line.split(maxsplit=1)
        require(len(pieces)==2 and len(pieces[0])==64, 'Malformed registry SHA256SUMS')
        rel = pieces[1].strip().lstrip('*')
        require(rel.startswith('./') and '..' not in Path(rel).parts,
                'Unsafe registry checksum path: '+rel)
        expected[rel[2:]] = pieces[0].lower()
    result = {}
    for model, folder in REGISTRY.items():
        d = reg / folder
        names = ('model.json',) if model == 'M0' else ('adapter_config.json', 'adapter_model.safetensors')
        model_files = {}
        for name in names:
            path = d / name
            rel = folder+'/'+name
            require(path.is_file() and rel in expected, 'Missing registry artifact/checksum '+rel)
            found = sha(path)
            require(found == expected[rel], 'Registry SHA256 mismatch '+rel)
            model_files[name] = {'sha256':found, 'bytes':path.stat().st_size}
        if model == 'M0':
            cfg=json.loads((d/'model.json').read_text(encoding='utf-8'))
            require(cfg.get('base_model') == BASE_MODEL and cfg.get('model_type') == 'base',
                    'M0 base model metadata differs')
        else:
            cfg=json.loads((d/'adapter_config.json').read_text(encoding='utf-8'))
            require(cfg.get('base_model_name_or_path') == BASE_MODEL and
                    str(cfg.get('peft_type','')).upper() == 'LORA',
                    'Adapter base/PEFT mismatch '+model)
            require((d/'adapter_model.safetensors').stat().st_size > 1024,
                    'Suspiciously small adapter '+model)
        result[model] = {'registry_path':str(d), 'files':model_files}
    return {'registry':str(reg), 'sha256s_sha256':sha(sumfile), 'models':result}

def prompt(pair):
    return ('Determine whether the following two code fragments are code clones. '
            'Answer YES if they are clones and NO otherwise.\n'
            'Language: python\nCode A:\n'+pair['code_a']+'\n'
            'Code B:\n'+pair['code_b']+'\nAnswer:')

def window_ranges(n, cap, overlap):
    require(cap >= 1 and 0 <= overlap < 1, 'Invalid window range parameters')
    if n <= cap: return [(0,n)]
    stride=max(1,int(cap*(1-overlap)))
    last=n-cap
    starts=list(range(0,last+1,stride))
    if not starts or starts[-1]!=last: starts.append(last)
    return [(s,s+cap) for s in starts]

def source_windows(source, offsets, cap, overlap):
    chunks=[]
    for a,b in window_ranges(len(offsets),cap,overlap):
        first=0 if a==0 else offsets[a][0]
        last=len(source) if b==len(offsets) else offsets[b][0]
        require(0<=first<=last<=len(source), 'Invalid tokenizer offsets')
        chunks.append({'start':a,'end':b,'text':source[first:last]})
    return chunks

def allocate_code_budget(la,lb,budget):
    mini=128
    require(budget>=mini*2, 'Not enough window budget')
    if la<=lb and la<=budget-mini: return max(1,la),budget-la
    if lb<la and lb<=budget-mini: return budget-lb,max(1,lb)
    share=la/max(1,la+lb)
    a=max(mini,min(budget-mini,round(budget*share)))
    return a,budget-a

def tokenize_windows(pair,tok,limit,overlap,max_windows):
    """D-02 / A-16 inspired: lossless Cartesian code windows; never truncate."""
    require(limit >= 512, 'max-input-tokens must be >=512')
    full=tok.encode(prompt(pair), add_special_tokens=False)
    if len(full)<=limit:
        yield (full, len(full), 1, False)
        return
    require(tok.is_fast, 'Fast tokenizer required for lossless windows')
    oa=tok(pair['code_a'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    ob=tok(pair['code_b'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    base=dict(pair,code_a='',code_b='')
    budget=limit-len(tok.encode(prompt(base),add_special_tokens=False))-128
    require(budget>=256, 'Prompt itself exceeds window budget')
    for attempt in range(48):
        ca,cb=allocate_code_budget(len(oa),len(ob),budget)
        wa=source_windows(pair['code_a'],oa,ca,overlap)
        wb=source_windows(pair['code_b'],ob,cb,overlap)
        count=len(wa)*len(wb)
        require(count<=max_windows, f'Too many Cartesian windows ({count}>{max_windows}); increase --max-windows')
        overflow=0
        for x in wa:
            for y in wb:
                obj=dict(pair,code_a=x['text'],code_b=y['text'])
                size=len(tok.encode(prompt(obj),add_special_tokens=False))
                overflow=max(overflow,size-limit)
                if overflow:break
            if overflow:break
        if overflow==0:
            for x in wa:
                for y in wb:
                    obj=dict(pair,code_a=x['text'],code_b=y['text'])
                    ids=tok.encode(prompt(obj),add_special_tokens=False)
                    yield (ids,len(full),count,True)
            return
        budget-=max(32,overflow+32)
        require(budget>=256, 'Insufficient budget for lossless windows')
    raise RuntimeError('Lossless window planner did not converge')

def preflight(args):
    root=args.root.resolve()
    gt=(root/args.ground_truth).resolve()
    require(gt.is_dir(), 'Missing ground truth directory '+str(gt))
    require(args.max_input_tokens>=512 and 0 <= args.window_overlap < 1 and args.max_windows>=1,
            'Invalid scoring parameters')
    rows, ground_truth = load_gt(gt)
    registry=validate_registry(root)
    require(not args.run_dir.exists(), 'Output run directory exists: '+str(args.run_dir))
    summary={'status':'D05C_V1_PREFLIGHT_PASS_NOT_EVALUATED',
             'version':VERSION,'created_at_chicago':now(),
             'model_id':BASE_MODEL,'revision':BASE_REVISION,
             'reference_scoring':'D-02 / A-16 one-token YES/NO logit margin',
             'prediction_rule':'mean(YES logit - NO logit) > 0',
             'max_input_tokens':args.max_input_tokens,
             'window_overlap':args.window_overlap,
             'max_windows':args.max_windows,
             'ground_truth':ground_truth,'model_registry':registry,
             'pairs':len(rows), 'model_ids':list(REGISTRY),
             'limitations':[
               'Reviewer identities and independent blind review are not authenticated.',
               'Negative blank label = confirmed nonclone under explicit user annotation convention.',
               'GT leakage audit is inherited; this does not repeat a full training-data overlap scan.',
               'Balanced similarity-selected sampling does not represent natural-world class prevalence.',
               'M0 registry metadata has placeholder revision; pinned D-02 revision explicitly selected here.']}
    args.run_dir.mkdir(parents=True,exist_ok=False)
    # Append-only freeze: copy exactly the inspected canonical rows as aligned source input.
    with (args.run_dir/'aligned_pairs.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    summary['aligned_pairs_sha256']=sha(args.run_dir/'aligned_pairs.jsonl')
    write_json(args.run_dir/'preflight.json',summary)
    print(json.dumps({'status':summary['status'],'pairs':len(rows),'models':list(REGISTRY),
                      'run':str(args.run_dir),'gt_sha256':ground_truth['verified_pairs_sha256']},sort_keys=True))

def checked_preflight(args):
    file=args.run_dir/'preflight.json'
    require(file.is_file(), 'Run preflight first: '+str(file))
    pf=json.loads(file.read_text(encoding='utf-8'))
    require(pf.get('status')=='D05C_V1_PREFLIGHT_PASS_NOT_EVALUATED', 'Wrong preflight status')
    require(pf['model_id']==BASE_MODEL and pf['revision']==BASE_REVISION, 'Pinned base mismatch')
    require(sha(args.run_dir/'aligned_pairs.jsonl')==pf['aligned_pairs_sha256'], 'Frozen aligned pairs changed')
    gt=(args.root.resolve()/args.ground_truth).resolve()
    require(verify_manifest(gt)==pf['ground_truth']['manifest_sha256'], 'Ground truth manifest changed')
    require(sha(gt/'verified_pairs.jsonl')==pf['ground_truth']['verified_pairs_sha256'], 'Ground truth pairs changed')
    require(validate_registry(args.root.resolve())==pf['model_registry'], 'Model registry changed since preflight')
    return pf

def inference(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from transformers import BitsAndBytesConfig
    require(torch.cuda.is_available(), 'CUDA required for inference')
    pf=checked_preflight(args)
    model_id=args.model.upper()
    require(model_id in REGISTRY, 'Unknown model '+model_id)
    final=args.run_dir/f'predictions_{model_id}.csv'
    tmp=args.run_dir/f'predictions_{model_id}.csv.partial'
    require(not final.exists() and not tmp.exists(), 'Model output already exists (no overwrite): '+str(final))
    torch.manual_seed(2026)
    model_options={'revision':BASE_REVISION,'local_files_only':True,'trust_remote_code':False}
    tok=AutoTokenizer.from_pretrained(BASE_MODEL,**model_options)
    require(tok.is_fast,'Fast tokenizer required for lossless windows')
    answer={x:tok.encode(' '+x,add_special_tokens=False) for x in ('YES','NO')}
    require(all(len(x)==1 for x in answer.values()), 'YES/NO not single tokens: '+str(answer))
    dtype=torch.float16
    if args.load_mode=='4bit':
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                                 bnb_4bit_compute_dtype=dtype,bnb_4bit_use_double_quant=True)
        model=AutoModelForCausalLM.from_pretrained(BASE_MODEL,quantization_config=quant,
                                                    device_map={'':0},**model_options)
    else:
        model=AutoModelForCausalLM.from_pretrained(BASE_MODEL,torch_dtype=dtype,
                                                    low_cpu_mem_usage=True,device_map={'':0},**model_options)
    if model_id!='M0':
        adapter=pf['model_registry']['models'][model_id]['registry_path']
        model=PeftModel.from_pretrained(model,adapter,is_trainable=False,local_files_only=True)
    model.eval()
    print('D-05c MODEL_LOADED',model_id,'mode',args.load_mode,flush=True)
    n=0
    with tmp.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=PRED_FIELDS)
        writer.writeheader()
        for pair in iter_jsonl(args.run_dir/'aligned_pairs.jsonl'):
            total=0.0
            wcount=0
            original_tokens=0
            is_windowed=False
            for ids,original,nwindows,windowed in tokenize_windows(pair,tok,pf['max_input_tokens'],
                                                                      pf['window_overlap'],pf['max_windows']):
                x=torch.tensor([ids],dtype=torch.long,device='cuda:0')
                with torch.inference_mode():
                    logits=model(input_ids=x).logits[0,-1,:]
                    margin=(logits[answer['YES'][0]]-logits[answer['NO'][0]]).float().item()
                require(math.isfinite(margin), 'Nonfinite margin '+pair['pair_id'])
                total+=margin
                wcount+=1
                original_tokens=original
                is_windowed=windowed
                del x,logits
            require(wcount>0 and wcount==nwindows, 'Window count mismatch: '+pair['pair_id'])
            mean=total/wcount
            writer.writerow({'pair_id':pair['pair_id'], 'project':pair['project'],
                             'true_label':pair['label'],'predicted_label':int(mean>0),
                             'mean_logit_margin':f'{mean:.10f}', 'n_windows':wcount,
                             'original_input_tokens':original_tokens,
                             'windowed':str(is_windowed).lower(), 'model':model_id,
                             'decision_threshold':'0'})
            n+=1
            if n%args.progress_every==0:
                print('D-05c PROGRESS',model_id,n,'/',pf['pairs'],flush=True)
        f.flush()
        os.fsync(f.fileno())
    require(n==N_TOTAL, 'Incomplete inference count; partial output retained for diagnosis')
    tmp.replace(final)
    atomic_json(args.run_dir/f'predictions_{model_id}.json',
                {'model':model_id,'completed_at_chicago':now(), 'records':n,
                 'csv_sha256':sha(final),'mode':args.load_mode,
                 'predictions_file':final.name,'registry_sha256s_sha256':pf['model_registry']['sha256s_sha256'],
                 'ground_truth_sha256':pf['ground_truth']['verified_pairs_sha256']})
    print('D-05c INFERENCE_COMPLETE',model_id,n,'sha256',sha(final),flush=True)

def metrics(rows):
    tp=sum(r['truth']==1 and r['pred']==1 for r in rows)
    tn=sum(r['truth']==0 and r['pred']==0 for r in rows)
    fp=sum(r['truth']==0 and r['pred']==1 for r in rows)
    fn=sum(r['truth']==1 and r['pred']==0 for r in rows)
    n=tp+tn+fp+fn
    safe=lambda num,den:float(num)/den if den else 0.0
    accuracy=safe(tp+tn,n)
    precision=safe(tp,tp+fp)
    recall=safe(tp,tp+fn)
    f1=safe(2*tp,2*tp+fp+fn)
    mcc_den=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    mcc=(tp*tn-fp*fn)/math.sqrt(mcc_den) if mcc_den else 0.0
    return {'n':n,'tp':tp,'fp':fp,'tn':tn,'fn':fn,
            'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1,'mcc':mcc}

def summarize(args):
    pf=checked_preflight(args)
    require(not (args.run_dir/'summary.json').exists(), 'Summary already exists, refusing overwrite')
    expected={r['pair_id']:(r['project'],r['label']) for r in iter_jsonl(args.run_dir/'aligned_pairs.jsonl')}
    require(len(expected)==N_TOTAL, 'Frozen input pair count mismatch')
    report={'version':VERSION,'status':'D05C_V1_EVALUATION_COMPLETE',
            'created_at_chicago':now(), 'ground_truth_sha256':pf['ground_truth']['verified_pairs_sha256'],
            'base_model':BASE_MODEL,'revision':BASE_REVISION,
            'scoring_protocol':pf['reference_scoring'],
            'prediction_rule':pf['prediction_rule'],
            'model_metrics':{},'prediction_files':{},'limitations':pf['limitations']}
    long_rows=[]
    bymodel={}
    for mid in REGISTRY:
        path=args.run_dir/f'predictions_{mid}.csv'
        record=args.run_dir/f'predictions_{mid}.json'
        require(path.is_file() and record.is_file(), 'Missing completed model '+mid)
        meta=json.loads(record.read_text(encoding='utf-8'))
        require(meta['csv_sha256']==sha(path) and meta['records']==N_TOTAL,
                'Prediction hash/count mismatch '+mid)
        require(meta['ground_truth_sha256']==report['ground_truth_sha256'],
                'Model prediction ground truth mismatch '+mid)
        seen=set(); data=[]
        with path.open('r',encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                pid=row['pair_id']
                require(pid in expected and pid not in seen,'Unexpected/duplicate '+mid+'/'+pid)
                seen.add(pid)
                project,label=expected[pid]
                require(row['project']==project and int(row['true_label'])==label,
                        'Mismatched model prediction truth '+mid+'/'+pid)
                p=int(row['predicted_label'])
                require(p in (0,1) and row['model']==mid,'Invalid prediction '+mid+'/'+pid)
                data.append({'project':project,'truth':label,'pred':p})
        require(seen==set(expected), 'Missing model predictions '+mid)
        report['model_metrics'][mid]=metrics(data)
        report['prediction_files'][mid]={'sha256':sha(path),'bytes':path.stat().st_size}
        for proj in sorted(set(x['project'] for x in data)):
            m=metrics([x for x in data if x['project']==proj]);long_rows.append({'model':mid,'project':proj,**m})
        bymodel[mid]=data
    project_metrics=args.run_dir/'project_metrics.csv'
    with project_metrics.open('w',encoding='utf-8',newline='') as f:
        fields=['model','project','n','tp','fp','tn','fn','accuracy','precision','recall','f1','mcc']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(long_rows)
    report['project_metrics_sha256']=sha(project_metrics)
    atomic_json(args.run_dir/'summary.json',report)
    artifact_files=['aligned_pairs.jsonl','preflight.json','project_metrics.csv','summary.json']
    artifact_files += [f'predictions_{m}.csv' for m in REGISTRY]
    artifact_files += [f'predictions_{m}.json' for m in REGISTRY]
    atomic_json(args.run_dir/'manifest.json',{'version':VERSION,'created_at_chicago':now(),
        'files':{f:{'sha256':sha(args.run_dir/f),'bytes':(args.run_dir/f).stat().st_size}
                 for f in artifact_files}})
    print(json.dumps({'status':report['status'],'run':str(args.run_dir),
                      'metrics':report['model_metrics']},sort_keys=True))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('preflight','infer','summarize'))
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--ground-truth',type=Path,default=Path('data/evaluation/ground_truth/python'))
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--model',choices=tuple(REGISTRY))
    ap.add_argument('--load-mode',choices=('4bit','fp16'),default='4bit')
    ap.add_argument('--max-input-tokens',type=int,default=4096)
    ap.add_argument('--window-overlap',type=float,default=0.20)
    ap.add_argument('--max-windows',type=int,default=256)
    ap.add_argument('--progress-every',type=int,default=100)
    args=ap.parse_args()
    try:
        if args.command=='preflight':preflight(args)
        elif args.command=='infer':
            require(args.model is not None,'--model required for infer')
            inference(args)
        else:summarize(args)
    except (ValueError,RuntimeError,KeyError,OSError,AssertionError,json.JSONDecodeError) as exc:
        print('D-05c v1 ERROR:',exc,file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':
    sys.exit(main())
