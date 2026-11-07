#!/usr/bin/env python3
"""D-02: independent evaluation of D-01 LoRA on A-series held-out panel.

Scoring protocol mirrors A-16: one-token YES/NO next-token logits,
mean margin across lossless Cartesian source-code windows, threshold > 0.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MODEL = 'bigcode/starcoder2-7b'
REV = 'bb9afde76d7945da5745592525db122d4d729eb1'
PRED_COLUMNS = ['language','project','row','left_id','right_id','true_label','predicted_label','yes_logit','no_logit','worker','input_tokens','original_input_tokens','truncated','windowed','window_count','total_window_tokens','mean_logit_margin','max_logit_margin']
WINDOW_COLUMNS = ['language','project','row','window_index','window_count','a_token_start','a_token_end','b_token_start','b_token_end','input_tokens','yes_logit','no_logit','logit_margin']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def dump(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')


def csv_write(path, rows, names):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=names, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def gt(id_a, id_b):
    aa = id_a.split('_'); bb = id_b.split('_')
    if len(aa) < 2 or len(bb) < 2:
        raise ValueError(f'Unparseable ground-truth IDs: {id_a} {id_b}')
    return int(aa[1] == bb[1])  # Same established A-06/A-16 reference-label rule


def validate_adapter(args):
    config = args.adapter_dir / 'adapter_config.json'
    weights = args.adapter_dir / 'adapter_model.safetensors'
    if not config.is_file() or not weights.is_file() or weights.stat().st_size < 1024:
        raise FileNotFoundError(f'Incomplete adapter: {args.adapter_dir}')
    cfg = json.loads(config.read_text(encoding='utf-8'))
    if str(cfg.get('peft_type','')).upper() != 'LORA':
        raise ValueError('Adapter is not PEFT LoRA')
    if not isinstance(cfg.get('r'), int) or cfg['r'] < 1:
        raise ValueError('Invalid LoRA rank')
    # Safetensors header validation without optional safetensors dependency.
    with weights.open('rb') as f:
        header_len = int.from_bytes(f.read(8), 'little')
        if not 0 < header_len < 100_000_000 or header_len > weights.stat().st_size - 8:
            raise ValueError('Malformed safetensors header')
        header = json.loads(f.read(header_len))
        tensors = {k:v for k,v in header.items() if k != '__metadata__'}
        if not tensors: raise ValueError('No tensors in adapter')
        data_size = weights.stat().st_size - 8 - header_len
        for name, meta in tensors.items():
            a,b = meta['data_offsets']
            if a < 0 or b <= a or b > data_size:
                raise ValueError(f'Invalid tensor offsets: {name}')
    summary_path = args.adapter_dir.parent / 'training_summary.json'
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary.get('status') != 'TRAINING_COMPLETE':
            raise ValueError('D-01 summary does not mark training complete')
    return {'adapter_dir':str(args.adapter_dir.resolve()),'adapter_config_sha256':sha256(config),'adapter_weights_sha256':sha256(weights),'adapter_bytes':weights.stat().st_size,'tensor_count':len(tensors),'lora_rank':cfg['r'],'training_summary_sha256':sha256(summary_path) if summary_path.is_file() else None}


def preflight(args):
    # The standalone shell wrapper reserves this new run directory before
    # calling Python. Accept that empty directory, but never reuse one with
    # existing artifacts (including a previous failed/partial preflight).
    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            raise NotADirectoryError(args.output_dir)
        if any(args.output_dir.iterdir()):
            raise FileExistsError(f'D-02 output directory is not empty: {args.output_dir}')
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    adapter = validate_adapter(args)
    tasks = []
    report = {}
    for lang in args.languages:
        code_path = args.data_root / 'code' / f'{lang}.jsonl'
        gt_dir = args.data_root / 'ground_truth' / lang
        files = sorted(gt_dir.glob('pred_*_with_refactorability.csv'))
        if not code_path.is_file() or not files:
            raise FileNotFoundError(f'Missing staged held-out panel {code_path} or {gt_dir}')
        codes = {}
        with code_path.open(encoding='utf-8-sig') as f:
            for line in f:
                if not line.strip(): continue
                o = json.loads(line)
                idx, code = o['id'], o['code']
                if not isinstance(idx,str) or not isinstance(code,str) or not code.strip() or idx in codes:
                    raise ValueError(f'Invalid/duplicate code ID: {code_path} {idx}')
                codes[idx] = code
        count = 0
        projects = 0
        for fp in files:
            project = fp.name.removeprefix('pred_').removesuffix('_with_refactorability.csv')
            with fp.open(encoding='utf-8-sig',newline='') as f:
                rows = list(csv.DictReader(f))
            if not rows: continue
            projects += 1
            for ix, row in enumerate(rows):
                a = row['pair_left_func'].strip(); b = row['pair_right_func'].strip()
                if a not in codes or b not in codes:
                    raise ValueError(f'Unresolved test function in {fp}:{ix+2}: {a} / {b}')
                tasks.append({'language':lang,'project':project,'row':ix,'left_id':a,'right_id':b,'label':gt(a,b),'code_a':codes[a],'code_b':codes[b]})
                count += 1
        report[lang] = {'projects':projects,'pairs':count,'codes':len(codes), 'code_sha256':sha256(code_path), 'ground_truth_sha256':{p.name:sha256(p) for p in files}}
    if not tasks: raise RuntimeError('No held-out test pairs')
    with (args.output_dir/'aligned_pairs.jsonl').open('w',encoding='utf-8') as f:
        for task in tasks: f.write(json.dumps(task,ensure_ascii=False)+'\n')
    payload={'status':'PREFLIGHT_COMPLETE_NOT_EVALUATED','created_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),'n_pairs':len(tasks),'languages':report,'adapter':adapter,'aligned_pairs_sha256':sha256(args.output_dir/'aligned_pairs.jsonl'), 'protocol':{'model_id':args.model_id,'revision':args.revision,'max_input_tokens':args.max_input_tokens,'window_overlap':args.window_overlap,'aggregation':'mean-margin','prediction_rule':'mean(YES logit - NO logit) > 0','reference':'A-16'}}
    dump(args.output_dir/'preflight.json',payload)
    print(f'D-02 PREFLIGHT_PASS pairs={len(tasks)} adapter_tensors={adapter["tensor_count"]}',flush=True)


def prompt(pair):
    return ('Determine whether the following two code fragments are code clones. '
            'Answer YES if they are clones and NO otherwise.\n'
            f'Language: {pair["language"]}\nCode A:\n{pair["code_a"]}\n'
            f'Code B:\n{pair["code_b"]}\nAnswer:')


def window_ranges(n, cap, overlap):
    if cap < 1 or not 0 <= overlap < 1: raise ValueError('Invalid window capacity/overlap')
    if n <= cap: return [(0,n)]
    stride = max(1,int(cap*(1-overlap)))
    last = n - cap
    starts = list(range(0,last+1,stride))
    if starts[-1] != last: starts.append(last)
    return [(s,s+cap) for s in starts]


def source_windows(source, offsets, cap, overlap):
    chunks=[]
    for start,end in window_ranges(len(offsets),cap,overlap):
        first = 0 if start == 0 else offsets[start][0]
        last = len(source) if end == len(offsets) else offsets[end][0]
        if not 0 <= first <= last <= len(source): raise RuntimeError('Bad offsets')
        chunks.append({'start':start,'end':end,'text':source[first:last]})
    return chunks


def allocate_code_budget(la,lb,budget):
    mini=128
    if budget < mini*2: raise RuntimeError('Window budget insufficient')
    if la <= lb and la <= budget-mini: return max(1,la), budget-la
    if lb < la and lb <= budget-mini: return budget-lb,max(1,lb)
    share=la/max(1,la+lb)
    a=max(mini,min(budget-mini,round(budget*share)))
    return a,budget-a


def tokenize_windows(pair,tok,limit,overlap):
    """Equivalent planning rules to A-16, streaming windows to avoid large allocations."""
    if limit < 512: raise ValueError('max-input-tokens must be >= 512')
    full = tok.encode(prompt(pair),add_special_tokens=False)
    if len(full) <= limit:
        yield (0,full,None,None,1,len(full),False)
        return
    if not tok.is_fast: raise RuntimeError('Fast tokenizer required for lossless windows')
    offsets_a = tok(pair['code_a'], add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    offsets_b = tok(pair['code_b'], add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    empty=dict(pair,code_a='',code_b='')
    budget=limit-len(tok.encode(prompt(empty),add_special_tokens=False))-128
    if budget < 256: raise RuntimeError('Insufficient prompt window budget')
    for _ in range(48):
        ca,cb=allocate_code_budget(len(offsets_a),len(offsets_b),budget)
        aa=source_windows(pair['code_a'],offsets_a,ca,overlap)
        bb=source_windows(pair['code_b'],offsets_b,cb,overlap)
        overflow=0
        for a in aa:
            for b in bb:
                edited=dict(pair,code_a=a['text'],code_b=b['text'])
                length=len(tok.encode(prompt(edited),add_special_tokens=False))
                overflow=max(overflow,length-limit)
                if overflow: break
            if overflow: break
        if not overflow:
            for k,(a,b) in enumerate((a,b) for a in aa for b in bb):
                edited=dict(pair,code_a=a['text'],code_b=b['text'])
                ids=tok.encode(prompt(edited),add_special_tokens=False)
                yield (k,ids,a,b,len(aa)*len(bb),len(full),True)
            return
        budget -= max(32,overflow+32)
        if budget < 256: raise RuntimeError('Window planner budget exhausted')
    raise RuntimeError('Window planner did not converge')


def infer(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from transformers import BitsAndBytesConfig
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    torch.manual_seed(2026)
    opts={'revision':args.revision,'local_files_only':True,'trust_remote_code':False}
    tok=AutoTokenizer.from_pretrained(args.model_id,**opts)
    if not tok.is_fast: raise RuntimeError('Fast tokenizer required for sliding windows')
    answer={key:tok.encode(' '+key,add_special_tokens=False) for key in ('YES','NO')}
    if any(len(x)!=1 for x in answer.values()): raise RuntimeError(f'YES/NO must be single tokens: {answer}')
    torch_dtype=torch.float16
    if args.load_mode == '4bit':
        q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch_dtype,bnb_4bit_use_double_quant=True)
        model=AutoModelForCausalLM.from_pretrained(args.model_id,quantization_config=q,device_map={'':0},**opts)
    else:
        model=AutoModelForCausalLM.from_pretrained(args.model_id,dtype=torch_dtype,low_cpu_mem_usage=True,device_map={'':0},**opts)
    model=PeftModel.from_pretrained(model,str(args.adapter_dir),is_trainable=False,local_files_only=True).eval()
    print(f'D-02 ADAPTER_LOADED worker={args.worker_index} mode={args.load_mode}',flush=True)
    pred_path=args.output_dir/f'predictions-worker-{args.worker_index}.csv'
    window_path=args.output_dir/f'window-predictions-worker-{args.worker_index}.csv'
    with (args.output_dir/'aligned_pairs.jsonl').open(encoding='utf-8') as inf, pred_path.open('w',encoding='utf-8',newline='') as pf,window_path.open('w',encoding='utf-8',newline='') as wf:
        pred=csv.DictWriter(pf,fieldnames=PRED_COLUMNS);pred.writeheader()
        win=csv.DictWriter(wf,fieldnames=WINDOW_COLUMNS);win.writeheader()
        processed=0
        for pos,line in enumerate(inf):
            if pos%args.num_workers!=args.worker_index:continue
            pair=json.loads(line)
            yes_sum=no_sum=0.0
            max_margin=float('-inf');max_tokens=total_tokens=windows=0
            was_windowed=False;original_tokens=0
            for index,ids,a,b,n_windows,orig,windowed in tokenize_windows(pair,tok,args.max_input_tokens,args.window_overlap):
                with torch.inference_mode():
                    logits=model(input_ids=torch.tensor([ids],device='cuda:0'),use_cache=False).logits[0,-1]
                    yes=float(logits[answer['YES'][0]].item());no=float(logits[answer['NO'][0]].item())
                del logits
                yes_sum+=yes;no_sum+=no
                max_margin=max(max_margin,yes-no)
                max_tokens=max(max_tokens,len(ids));total_tokens+=len(ids)
                windows+=1;was_windowed=windowed;original_tokens=orig
                if windowed:
                    win.writerow({'language':pair['language'],'project':pair['project'],'row':pair['row'],'window_index':index,'window_count':n_windows,'a_token_start':a['start'],'a_token_end':a['end'],'b_token_start':b['start'],'b_token_end':b['end'],'input_tokens':len(ids),'yes_logit':yes,'no_logit':no,'logit_margin':yes-no})
            if windows<1:raise RuntimeError('No prediction windows')
            avg_yes=yes_sum/windows;avg_no=no_sum/windows;margin=avg_yes-avg_no
            pred.writerow({'language':pair['language'],'project':pair['project'],'row':pair['row'],'left_id':pair['left_id'],'right_id':pair['right_id'],'true_label':pair['label'],'predicted_label':int(margin>0),'yes_logit':avg_yes,'no_logit':avg_no,'worker':args.worker_index,'input_tokens':max_tokens,'original_input_tokens':original_tokens,'truncated':'false','windowed':str(was_windowed).lower(),'window_count':windows,'total_window_tokens':total_tokens,'mean_logit_margin':margin,'max_logit_margin':max_margin})
            processed+=1
            if processed%args.progress_every==0:
                pf.flush();wf.flush();print(f'D-02 PROGRESS worker={args.worker_index} pairs={processed}',flush=True)
    print(f'D-02 WORKER_COMPLETE worker={args.worker_index} pairs={processed}',flush=True)


def cm(rows):
    tp=sum(r['true_label']=='1' and r['predicted_label']=='1' for r in rows)
    tn=sum(r['true_label']=='0' and r['predicted_label']=='0' for r in rows)
    fp=sum(r['true_label']=='0' and r['predicted_label']=='1' for r in rows)
    fn=sum(r['true_label']=='1' and r['predicted_label']=='0' for r in rows)
    n=len(rows)
    return {'n':n,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'accuracy':(tp+tn)/n if n else 0.0,'precision':tp/(tp+fp) if tp+fp else 0.0,'recall':tp/(tp+fn) if tp+fn else 0.0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}


def summarize(args):
    pf=json.loads((args.output_dir/'preflight.json').read_text(encoding='utf-8'))
    expected={}
    with (args.output_dir/'aligned_pairs.jsonl').open(encoding='utf-8') as f:
        for line in f:
            d=json.loads(line)
            key=(d['language'],d['project'],str(d['row']))
            if key in expected:raise ValueError(f'Duplicate aligned pair key {key}')
            expected[key]=(d['left_id'],d['right_id'],str(d['label']))
    if len(expected)!=pf['n_pairs'] or sha256(args.output_dir/'aligned_pairs.jsonl')!=pf['aligned_pairs_sha256']:
        raise ValueError('Aligned held-out test panel mutated since preflight')
    rows=[];seen=set()
    for worker in range(args.num_workers):
        with (args.output_dir/f'predictions-worker-{worker}.csv').open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                key=(row['language'],row['project'],row['row'])
                if key in seen or key not in expected:raise ValueError(f'Unknown/duplicate prediction key {key}')
                if (row['left_id'],row['right_id'],row['true_label'])!=expected[key]:
                    raise ValueError(f'Prediction identity or label mismatch {key}')
                if row['predicted_label'] not in ('0','1'):raise ValueError(f'Invalid prediction {key}')
                if row['truncated']!='false' or int(row['window_count'])<1:raise ValueError('Truncation/missing windows')
                seen.add(key);rows.append(row)
    if len(rows)!=len(expected):raise ValueError(f'Missing predictions {len(rows)}/{len(expected)}')
    grouped=defaultdict(list)
    for r in rows:grouped[(r['language'],r['project'])].append(r)
    metrics=[];unwindowed=[]
    for (lang,project),group in sorted(grouped.items()):
        metrics.append({'language':lang,'project':project,'n_expected':len(group),'n_scored':len(group),**cm(group),'complete':True})
        un=[r for r in group if r['windowed']=='false']
        unwindowed.append({'language':lang,'project':project,'n_original':len(group),'n_excluded_windowed':len(group)-len(un),'n_scored':len(un),**cm(un)})
    cols=['language','project','n_expected','n_scored','tp','fp','fn','tn','accuracy','precision','recall','f1','complete']
    csv_write(args.output_dir/'project_metrics.csv',metrics,cols)
    csv_write(args.output_dir/'project_metrics_unwindowed.csv',unwindowed,['language','project','n_original','n_excluded_windowed','n_scored','tp','fp','fn','tn','accuracy','precision','recall','f1'])
    sorted_rows=sorted(rows,key=lambda r:(r['language'],r['project'],int(r['row'])))
    csv_write(args.output_dir/'predictions_all.csv',sorted_rows,PRED_COLUMNS)
    info={'status':'COMPLETE','created_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),'expected_pairs':len(expected),'scored_pairs':len(rows),'projects':len(grouped),'metrics_micro':cm(rows),'metrics_by_language':{language:cm([r for r in rows if r['language']==language]) for language in args.languages},'model_role':'D01_CORRECTED_PYTHON_QLORA','prediction_rule':'mean(YES logit - NO logit) > 0','window_limit':args.max_input_tokens,'window_overlap':args.window_overlap,'aggregation':'mean-margin','windowed_pairs':sum(r['windowed']=='true' for r in rows),'truncated_pairs':0,'adapter':pf['adapter'],'holdout_data':pf['languages'],'protocol_comparability':'A-16 scoring and windowing rules matched; quantized inference may differ from fp16 baseline','output_sha256':{p.name:sha256(p) for p in sorted(args.output_dir.glob('*.csv'))},'evaluation_limitations':['C-11 scope-limited approval: near-duplicate and project independence not established','Training negative-label uncertainty remains','D-01 base-model initialization and Python-only training differ from M1/M2/M3']}
    dump(args.output_dir/'summary.json',info)
    print('D-02 COMPLETE '+json.dumps({'pairs':len(rows),'f1':info['metrics_micro']['f1'],'by_language':info['metrics_by_language']}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['preflight','infer','summarize'])
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--adapter-dir',type=Path,required=True)
    p.add_argument('--model-id',default=MODEL)
    p.add_argument('--revision',default=REV)
    p.add_argument('--languages',nargs='+',choices=['java','python'],default=['java','python'])
    p.add_argument('--worker-index',type=int,default=0)
    p.add_argument('--num-workers',type=int,default=3)
    p.add_argument('--max-input-tokens',type=int,default=12000)
    p.add_argument('--window-overlap',type=float,default=.20)
    p.add_argument('--load-mode',choices=['4bit','fp16'],default='fp16')
    p.add_argument('--progress-every',type=int,default=100)
    args=p.parse_args()
    if not (0<=args.window_overlap<1):p.error('Invalid --window-overlap')
    if args.command=='preflight':preflight(args)
    elif args.command=='infer':infer(args)
    else:summarize(args)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'D-02 ERROR: {e}',file=sys.stderr,flush=True)
        raise
