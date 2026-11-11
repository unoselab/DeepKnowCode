#!/usr/bin/env python3
"""E-01e v2: independent 7B/15B D-05c-compatible held-out binary clone evaluation.

The D-05c prompting, pair validation, lossless Cartesian windowing and binary
logit decision rule are preserved. Model size affects only pinned base/adapter
selection. No training, adapter mutation, or test-label-based threshold fitting.
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

PRED_FIELDS = ['pair_id','project','true_label','predicted_label','mean_logit_margin',
               'n_windows','original_input_tokens','windowed','model','decision_threshold']
MODELS = ('M0','M1','M2','M3')
STATUS = 'E01E_V2_PREFLIGHT_PASS_NOT_EVALUATED'

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1048576),b''): h.update(c)
    return h.hexdigest()

def canonical_digest(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()

def now():return datetime.now(ZoneInfo('America/Chicago')).isoformat()

def write_json(p,obj):
    require(not p.exists(),f'Refusing overwrite: {p}')
    temp=p.with_name(p.name+'.partial')
    require(not temp.exists(),f'Partial output exists: {temp}')
    temp.write_text(json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    temp.replace(p)

def records(p):
    with p.open(encoding='utf-8-sig') as f:
        for lineno,line in enumerate(f,1):
            require(bool(line.strip()),f'Empty JSONL line {p}:{lineno}')
            d=json.loads(line)
            require(isinstance(d,dict),f'Nonobject JSONL {p}:{lineno}')
            yield d

def validate_gt(gt,a):
    manp=gt/'manifest.json'
    require(manp.is_file(),'Missing Ground Truth manifest.json')
    man=json.loads(manp.read_text())
    files=man.get('files')
    require(isinstance(files,dict),'GT manifest files missing')
    required={'positive_verified_pairs.jsonl','negative_verified_pairs.jsonl',
              'verified_pairs.jsonl','excluded_negative_candidate_clones.jsonl',
              'annotation_audit.json','leakage_audit.json'}
    require(required <= files.keys(),'D-05bj GT evidence incomplete')
    for name,meta in files.items():
        require(name not in ('.','..') and '/' not in name and '\\' not in name,'Unsafe GT path')
        p=gt/name
        require(p.is_file() and sha(p)==meta['sha256'],f'GT checksum mismatch: {name}')
        if 'bytes' in meta:require(p.stat().st_size==meta['bytes'],f'GT byte mismatch: {name}')
    pairp=gt/'verified_pairs.jsonl'
    actual_sha=sha(pairp)
    if a.expected_gt_sha256 != 'record-only':
        require(actual_sha==a.expected_gt_sha256,'GT SHA256 differs from pinned shell parameter')
    seen=set();codepairs=set();counts=collections.Counter();projects=collections.Counter();rows=[]
    for item in records(pairp):
        pid=item.get('pair_id');project=item.get('project');label=item.get('gold_label')
        require(isinstance(pid,str) and pid and pid not in seen,'Duplicate/bad pair_id')
        require(isinstance(project,str) and project and project!='azure-sdk-for-python','Invalid project')
        require(type(label) is int and label in (0,1),'Invalid label')
        require(item.get('evaluation_eligible') is True,'Ineligible evaluation pair')
        codes=[];hashes=[]
        for side in ('left','right'):
            x=item.get(side)
            require(isinstance(x,dict),'Missing code side')
            code=x.get('source_code')
            require(isinstance(code,str) and code.strip(),'Empty source code')
            digest=canonical_digest(code)
            require(x.get('code_sha256')==digest,'Source code digest mismatch')
            hashes.append(digest);codes.append(code)
        key=tuple(sorted(hashes))
        require(key not in codepairs,'Duplicate code pair')
        codepairs.add(key);seen.add(pid)
        rows.append(dict(pair_id=pid,project=project,label=label,language='python',code_a=codes[0],code_b=codes[1]))
        counts[label]+=1;projects[project]+=1
    require(len(rows)==a.expected_pairs and counts=={0:a.expected_per_class,1:a.expected_per_class},'GT balance/count mismatch')
    require(len(projects)==a.expected_projects,'Project count mismatch')
    def can(r):return json.dumps(r,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    merged=collections.Counter(can(v) for v in records(pairp))
    originals=collections.Counter(can(v) for p in ('positive_verified_pairs.jsonl','negative_verified_pairs.jsonl') for v in records(gt/p))
    require(merged==originals,'Merged pairs differ from verified positive/negative records')
    return rows,dict(verified_pairs_sha256=actual_sha,manifest_sha256=sha(manp),
                     class_counts={'0':counts[0],'1':counts[1]},project_counts=dict(sorted(projects.items())))

def file_info(p,expected):
    require(p.is_file(),f'Missing artifact: {p}')
    digest=sha(p)
    require(expected=='record-only' or digest==expected,f'Artifact hash mismatch: {p} actual={digest}')
    return {'path':str(p.resolve()),'sha256':digest,'bytes':p.stat().st_size}

def validate_models(a):
    result={}
    if a.model_mode=='starcoder2-7b':
        reg=a.registry_7b
        sums=reg/'SHA256SUMS';require(sums.is_file(),'Missing 7B SHA256SUMS')
        expected={}
        for line in sums.read_text().splitlines():
            if not line.strip():continue
            chunks=line.split(maxsplit=1)
            require(len(chunks)==2,'Invalid SHA256SUMS line')
            rel=chunks[1].strip().lstrip('*')
            require(rel.startswith('./') and '..' not in Path(rel).parts,'Bad registry relative path')
            expected[rel[2:]]=chunks[0].lower()
        for model,folder in zip(MODELS,('m0-base','m1-source-lora','m2-standard-mixed-lora','m3-curriculum-lora')):
            d=reg/folder
            names=('model.json',) if model=='M0' else ('adapter_config.json','adapter_model.safetensors')
            evidence={}
            for n in names:
                rel=folder+'/'+n
                require(rel in expected,'Missing checksum '+rel)
                evidence[n]=file_info(d/n,expected[rel])
            if model=='M0':
                cfg=json.loads((d/'model.json').read_text())
                require(cfg.get('base_model')==a.model_id and cfg.get('model_type')=='base','7B M0 metadata mismatch')
            else:
                cfg=json.loads((d/'adapter_config.json').read_text())
                require(cfg.get('base_model_name_or_path')==a.model_id and str(cfg.get('peft_type','')).upper()=='LORA','7B LoRA base mismatch')
                require((d/'adapter_model.safetensors').stat().st_size>1024,'Adapter unexpectedly small')
            result[model]={'adapter_dir':None if model=='M0' else str(d.resolve()),'files':evidence}
        return {'registry_path':str(reg.resolve()),'registry_sha256s_sha256':sha(sums),'models':result}
    require(a.model_mode=='starcoder2-15b','Unsupported model mode')
    result['M0']={'adapter_dir':None,'files':{}}
    for model,folder,weights,config in (
        ('M1',a.adapter_m1,a.adapter_m1_sha256,a.adapter_m1_config_sha256),
        ('M2',a.adapter_m2,a.adapter_m2_sha256,a.adapter_m2_config_sha256),
        ('M3',a.adapter_m3,a.adapter_m3_sha256,a.adapter_m3_config_sha256)):
        cfg=json.loads((folder/'adapter_config.json').read_text())
        require(cfg.get('base_model_name_or_path')==a.model_id,'15B adapter base mismatch '+model)
        require(str(cfg.get('peft_type','')).upper()=='LORA','15B adapter not LoRA '+model)
        result[model]={'adapter_dir':str(folder.resolve()),'files':{
            'adapter_config.json':file_info(folder/'adapter_config.json',config),
            'adapter_model.safetensors':file_info(folder/'adapter_model.safetensors',weights)}}
    return {'registry_path':None,'registry_sha256s_sha256':None,'models':result}

def prompt(pair):
    return ('Determine whether the following two code fragments are code clones. '
            'Answer YES if they are clones and NO otherwise.\n'
            'Language: python\nCode A:\n'+pair['code_a']+'\n'
            'Code B:\n'+pair['code_b']+'\nAnswer:')

def window_ranges(n,cap,overlap):
    require(cap>=1 and 0<=overlap<1,'Invalid window parameters')
    if n<=cap:return [(0,n)]
    stride=max(1,int(cap*(1-overlap)));last=n-cap
    starts=list(range(0,last+1,stride))
    if not starts or starts[-1]!=last:starts.append(last)
    return [(s,s+cap) for s in starts]

def source_windows(source,offsets,cap,overlap):
    out=[]
    for a,b in window_ranges(len(offsets),cap,overlap):
        first=0 if a==0 else offsets[a][0]
        last=len(source) if b==len(offsets) else offsets[b][0]
        require(0<=first<=last<=len(source),'Invalid offsets')
        out.append(dict(start=a,end=b,text=source[first:last]))
    return out

def allocate_code_budget(la,lb,budget):
    mini=128
    require(budget>=mini*2,'Insufficient code window budget')
    if la<=lb and la<=budget-mini:return max(1,la),budget-la
    if lb<la and lb<=budget-mini:return budget-lb,max(1,lb)
    share=la/max(1,la+lb)
    a=max(mini,min(budget-mini,round(budget*share)))
    return a,budget-a

def tokenize_windows(pair,tok,limit,overlap,max_windows):
    require(limit>=512,'Input token cap must be >=512')
    full=tok.encode(prompt(pair),add_special_tokens=False)
    if len(full)<=limit:
        yield full,len(full),1,False
        return
    require(tok.is_fast,'Fast tokenizer required')
    oa=tok(pair['code_a'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    ob=tok(pair['code_b'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    base=dict(pair,code_a='',code_b='')
    budget=limit-len(tok.encode(prompt(base),add_special_tokens=False))-128
    require(budget>=256,'Prompt exceeds window budget')
    for _ in range(48):
        ca,cb=allocate_code_budget(len(oa),len(ob),budget)
        wa=source_windows(pair['code_a'],oa,ca,overlap);wb=source_windows(pair['code_b'],ob,cb,overlap)
        count=len(wa)*len(wb)
        require(count<=max_windows,f'Too many Cartesian windows: {count} > {max_windows}')
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
                    yield ids,len(full),count,True
            return
        budget-=max(32,overflow+32)
        require(budget>=256,'Budget exhausted planning windows')
    raise RuntimeError('Window planner failed to converge')

def common_config(a):
    return {'experiment':'E-01e','version':'v2','model_mode':a.model_mode,'model_id':a.model_id,
            'revision':a.revision,'ground_truth_dir':str(a.ground_truth.resolve()),
            'registry_7b':str(a.registry_7b.resolve()),
            'adapter_dirs':{'M1':str(a.adapter_m1.resolve()),'M2':str(a.adapter_m2.resolve()),'M3':str(a.adapter_m3.resolve())},
            'adapter_expected_sha256':{'M1':a.adapter_m1_sha256,'M2':a.adapter_m2_sha256,'M3':a.adapter_m3_sha256},
            'adapter_config_expected_sha256':{'M1':a.adapter_m1_config_sha256,'M2':a.adapter_m2_config_sha256,'M3':a.adapter_m3_config_sha256},
            'expected_gt_sha256':a.expected_gt_sha256,'expected_pairs':a.expected_pairs,
            'expected_per_class':a.expected_per_class,'expected_projects':a.expected_projects,
            'load_mode':a.load_mode,'max_input_tokens':a.max_input_tokens,
            'window_overlap':a.window_overlap,'max_windows':a.max_windows,
            'progress_every':a.progress_every,'seed':a.seed,'gpu_id':a.gpu_id,
            'local_files_only':a.local_files_only,'trust_remote_code':a.trust_remote_code,
            'answer_yes':a.answer_yes,'answer_no':a.answer_no,
            'decision_threshold':a.decision_threshold}

def preflight(a):
    require(a.model_mode in ('starcoder2-7b','starcoder2-15b'),'Invalid model mode')
    require(a.model_id==('bigcode/starcoder2-7b' if a.model_mode=='starcoder2-7b' else 'bigcode/starcoder2-15b'),'Base model/mode conflict')
    require(len(a.revision)==40 and all(c in '0123456789abcdef' for c in a.revision),'Revision must be 40-char commit SHA')
    require(a.max_input_tokens>=512 and a.max_windows>=1 and 0<=a.window_overlap<1,'Invalid window config')
    require(a.expected_pairs==a.expected_per_class*2 and a.expected_per_class>0 and a.expected_projects>0,'Invalid counts')
    require(a.decision_threshold==0,'D-05c protocol requires zero threshold')
    require(a.answer_yes==' YES' and a.answer_no==' NO','D-05c answer token strings must be exact')
    require(a.local_files_only and not a.trust_remote_code,'Unsafe or non-reproducible model loading flags')
    require(a.gpu_id>=0 and a.seed>=0 and a.progress_every>0,'Invalid seed/device/progress')
    require(a.load_mode in ('4bit','fp16'),'Invalid load mode')
    require(not a.run_dir.exists(),'Output run directory already exists')
    rows,gt=validate_gt(a.ground_truth,a)
    reg=validate_models(a)
    a.run_dir.mkdir(parents=True,exist_ok=False)
    aligned=a.run_dir/'aligned_pairs.jsonl'
    with aligned.open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n')
    pf={'status':STATUS,'created_at_chicago':now(),'effective_parameters':common_config(a),
        'ground_truth':gt,'model_registry':reg,'pairs':len(rows),'aligned_pairs_sha256':sha(aligned),
        'reference_scoring':'D-05c one-token YES/NO next-token margin, lossless Cartesian windowing',
        'prediction_rule':'mean(YES logit - NO logit) > 0',
        'limitations':['Reviewer identity and independent blind review not externally authenticated',
         'Balanced similarity-selected dataset does not represent natural class prevalence',
         'Preflight validates D-05bj evidence but does not recalculate all Training/Test code leakage',
         '4bit predictions are not assumed numerically identical to fp16']}
    write_json(a.run_dir/'preflight.json',pf)
    print(json.dumps({'status':STATUS,'model_mode':a.model_mode,'pairs':len(rows),'run':str(a.run_dir),
                      'gt_sha256':gt['verified_pairs_sha256']}),flush=True)

def checked(a):
    pfpath=a.run_dir/'preflight.json'
    require(pfpath.is_file(),'Missing preflight.json')
    pf=json.loads(pfpath.read_text())
    require(pf.get('status')==STATUS,'Unexpected preflight status')
    require(pf['effective_parameters']==common_config(a),'Effective shell parameters differ from preflight')
    require(sha(a.run_dir/'aligned_pairs.jsonl')==pf['aligned_pairs_sha256'],'Aligned pair contents changed')
    require(sha(a.ground_truth/'manifest.json')==pf['ground_truth']['manifest_sha256'],'Ground truth manifest changed')
    require(sha(a.ground_truth/'verified_pairs.jsonl')==pf['ground_truth']['verified_pairs_sha256'],'Ground truth pairs changed')
    require(validate_models(a)==pf['model_registry'],'Model artifact changed since preflight')
    return pf

def infer(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    require(torch.cuda.is_available(),'CUDA required for inference')
    pf=checked(a)
    mid=a.model
    require(mid in MODELS,'Invalid inference model')
    out=a.run_dir/f'predictions_{mid}.csv';tmp=out.with_name(out.name+'.partial')
    require(not out.exists() and not tmp.exists(),'Inference output exists; no overwrite')
    torch.manual_seed(a.seed)
    opts={'revision':a.revision,'local_files_only':a.local_files_only,'trust_remote_code':a.trust_remote_code}
    tokenizer=AutoTokenizer.from_pretrained(a.model_id,**opts)
    require(tokenizer.is_fast,'Fast tokenizer required')
    tokens={x:tokenizer.encode(text,add_special_tokens=False) for x,text in [('YES',a.answer_yes),('NO',a.answer_no)]}
    require(all(len(v)==1 for v in tokens.values()),'YES/NO is not single token')
    dtype=torch.float16
    if a.load_mode=='4bit':
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=dtype,bnb_4bit_use_double_quant=True)
        model=AutoModelForCausalLM.from_pretrained(a.model_id,quantization_config=quant,device_map={'':0},**opts)
    else:
        model=AutoModelForCausalLM.from_pretrained(a.model_id,torch_dtype=dtype,low_cpu_mem_usage=True,device_map={'':0},**opts)
    if mid!='M0':
        model=PeftModel.from_pretrained(model,pf['model_registry']['models'][mid]['adapter_dir'],
                                         is_trainable=False,local_files_only=True)
    model.eval()
    n=0
    with tmp.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=PRED_FIELDS);writer.writeheader()
        for pair in records(a.run_dir/'aligned_pairs.jsonl'):
            total=0.;wcount=0;ntokens=0;iswindow=False;nwindows=0
            for ids,ntokens,nwindows,iswindow in tokenize_windows(pair,tokenizer,a.max_input_tokens,a.window_overlap,a.max_windows):
                x=torch.tensor([ids],dtype=torch.long,device='cuda:0')
                with torch.inference_mode():
                    logits=model(input_ids=x).logits[0,-1,:]
                    margin=(logits[tokens['YES'][0]]-logits[tokens['NO'][0]]).float().item()
                require(math.isfinite(margin),'Nonfinite margin '+pair['pair_id'])
                total+=margin;wcount+=1
                del x,logits
            require(wcount>0 and wcount==nwindows,'Window count mismatch')
            mean=total/wcount
            writer.writerow({'pair_id':pair['pair_id'],'project':pair['project'],'true_label':pair['label'],
              'predicted_label':int(mean>a.decision_threshold),'mean_logit_margin':f'{mean:.10f}',
              'n_windows':wcount,'original_input_tokens':ntokens,'windowed':str(iswindow).lower(),
              'model':mid,'decision_threshold':str(a.decision_threshold)})
            n+=1
            if n%a.progress_every==0:print('E-01e INFERENCE_PROGRESS',a.model_mode,mid,n,'/',a.expected_pairs,flush=True)
        f.flush();os.fsync(f.fileno())
    require(n==a.expected_pairs,'Incomplete inference; .partial retained')
    tmp.replace(out)
    write_json(a.run_dir/f'predictions_{mid}.json',{'model':mid,'model_mode':a.model_mode,
        'records':n,'completed_at_chicago':now(),'csv_sha256':sha(out),
        'load_mode':a.load_mode,'ground_truth_sha256':pf['ground_truth']['verified_pairs_sha256'],
        'adapter_sha256':None if mid=='M0' else pf['model_registry']['models'][mid]['files']['adapter_model.safetensors']['sha256']})
    print('E-01e INFERENCE_COMPLETE',mid,n,sha(out),flush=True)

def metrics(rows):
    tp=sum(r['truth']==1 and r['pred']==1 for r in rows)
    tn=sum(r['truth']==0 and r['pred']==0 for r in rows)
    fp=sum(r['truth']==0 and r['pred']==1 for r in rows)
    fn=sum(r['truth']==1 and r['pred']==0 for r in rows)
    n=tp+tn+fp+fn
    safe=lambda x,y:x/y if y else 0.
    den=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return {'n':n,'tp':tp,'fp':fp,'tn':tn,'fn':fn,'accuracy':safe(tp+tn,n),
            'precision':safe(tp,tp+fp),'recall':safe(tp,tp+fn),
            'f1':safe(2*tp,2*tp+fp+fn),'mcc':(tp*tn-fp*fn)/math.sqrt(den) if den else 0.}

def summarize(a):
    from sklearn.metrics import roc_auc_score,average_precision_score
    pf=checked(a)
    require(not (a.run_dir/'summary.json').exists(),'Summary exists')
    expected={r['pair_id']:(r['project'],r['label']) for r in records(a.run_dir/'aligned_pairs.jsonl')}
    require(len(expected)==a.expected_pairs,'Aligned pair count mismatch')
    report={'status':'E01E_V2_EVALUATION_COMPLETE','created_at_chicago':now(),
        'version':'v2','model_mode':a.model_mode,'base_model':a.model_id,'revision':a.revision,
        'ground_truth_sha256':pf['ground_truth']['verified_pairs_sha256'],
        'scoring_protocol':pf['reference_scoring'],'prediction_rule':pf['prediction_rule'],
        'effective_parameters':pf['effective_parameters'],'model_metrics':{},'prediction_files':{},
        'limitations':pf['limitations']}
    allprojects=[]
    for mid in MODELS:
        p=a.run_dir/f'predictions_{mid}.csv';mp=a.run_dir/f'predictions_{mid}.json'
        require(p.is_file() and mp.is_file(),'Missing complete predictions: '+mid)
        meta=json.loads(mp.read_text())
        require(meta['csv_sha256']==sha(p) and meta['records']==a.expected_pairs,'Prediction artifact mismatch')
        require(meta['ground_truth_sha256']==report['ground_truth_sha256'],'Prediction GT mismatch')
        seen=set();rows=[];y=[];scores=[]
        with p.open(encoding='utf-8',newline='') as f:
            reader=csv.DictReader(f)
            require(reader.fieldnames==PRED_FIELDS,'Prediction CSV fields changed')
            for item in reader:
                pid=item['pair_id'];require(pid in expected and pid not in seen,'Unexpected/duplicate pair')
                seen.add(pid);proj,label=expected[pid]
                require(item['project']==proj and int(item['true_label'])==label,'Ground truth mismatch')
                pred=int(item['predicted_label']);score=float(item['mean_logit_margin'])
                require(pred in (0,1) and math.isfinite(score) and pred==int(score>a.decision_threshold), 'Prediction threshold mismatch')
                require(item['model']==mid and float(item['decision_threshold'])==a.decision_threshold,'Prediction protocol mismatch')
                rows.append(dict(project=proj,truth=label,pred=pred));y.append(label);scores.append(score)
        require(len(rows)==a.expected_pairs and seen==set(expected),'Missing prediction rows')
        m=metrics(rows);m['roc_auc']=float(roc_auc_score(y,scores));m['average_precision']=float(average_precision_score(y,scores))
        report['model_metrics'][mid]=m
        report['prediction_files'][mid]={'sha256':sha(p),'bytes':p.stat().st_size}
        for proj in sorted({r['project'] for r in rows}):
            allprojects.append({'model':mid,'project':proj,**metrics([r for r in rows if r['project']==proj])})
    projfile=a.run_dir/'project_metrics.csv'
    with projfile.open('x',encoding='utf-8',newline='') as f:
        fields=['model','project','n','tp','fp','tn','fn','accuracy','precision','recall','f1','mcc']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(allprojects)
    report['project_metrics_sha256']=sha(projfile)
    write_json(a.run_dir/'summary.json',report)
    files=['aligned_pairs.jsonl','preflight.json','project_metrics.csv','summary.json']
    files += [f'predictions_{m}.{ext}' for m in MODELS for ext in ('csv','json')]
    write_json(a.run_dir/'manifest.json',{'version':'v2','created_at_chicago':now(),
               'files':{name:{'sha256':sha(a.run_dir/name),'bytes':(a.run_dir/name).stat().st_size} for name in files}})
    print(json.dumps({'status':report['status'],'model_mode':a.model_mode,'metrics':report['model_metrics']},sort_keys=True),flush=True)

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('preflight','infer','summarize'))
    p.add_argument('--model-mode',required=True,choices=('starcoder2-7b','starcoder2-15b'))
    p.add_argument('--model-id',required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--ground-truth',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--registry-7b',type=Path,required=True)
    for m in ('m1','m2','m3'):
        p.add_argument(f'--adapter-{m}',type=Path,required=True)
        p.add_argument(f'--adapter-{m}-sha256',required=True)
        p.add_argument(f'--adapter-{m}-config-sha256',required=True)
    p.add_argument('--expected-gt-sha256',required=True)
    p.add_argument('--expected-pairs',type=int,required=True)
    p.add_argument('--expected-per-class',type=int,required=True)
    p.add_argument('--expected-projects',type=int,required=True)
    p.add_argument('--load-mode',required=True,choices=('4bit','fp16'))
    p.add_argument('--max-input-tokens',type=int,required=True)
    p.add_argument('--window-overlap',type=float,required=True)
    p.add_argument('--max-windows',type=int,required=True)
    p.add_argument('--progress-every',type=int,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--gpu-id',type=int,required=True)
    p.add_argument('--local-files-only',type=int,choices=(0,1),required=True)
    p.add_argument('--trust-remote-code',type=int,choices=(0,1),required=True)
    p.add_argument('--answer-yes',required=True)
    p.add_argument('--answer-no',required=True)
    p.add_argument('--decision-threshold',type=float,required=True)
    p.add_argument('--model',choices=MODELS)
    a=p.parse_args(argv)
    a.local_files_only=bool(a.local_files_only);a.trust_remote_code=bool(a.trust_remote_code)
    a.root=a.root.resolve();a.ground_truth=a.ground_truth.resolve();a.run_dir=a.run_dir.resolve()
    a.registry_7b=a.registry_7b.resolve()
    for m in ('m1','m2','m3'):setattr(a,f'adapter_{m}',getattr(a,f'adapter_{m}').resolve())
    return a

def main(argv=None):
    a=parse_args(argv)
    try:
        if a.command=='preflight':preflight(a)
        elif a.command=='infer':
            require(a.model is not None,'--model is required with infer')
            infer(a)
        else:summarize(a)
    except (ValueError,RuntimeError,KeyError,OSError,AssertionError,json.JSONDecodeError) as e:
        print('E-01e v2 ERROR:',e,file=sys.stderr,flush=True)
        return 2
    return 0

if __name__=='__main__':sys.exit(main())
