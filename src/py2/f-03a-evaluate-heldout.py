#!/usr/bin/env python3
"""F-03a v1. Read-only M0/M1/M2/M3 CodeLlama held-out evaluation.
D-05c / E-01e protocol: 7,942 verified pairs, 4096-token lossless
Cartesian windows, zero-threshold next-token YES-NO logit margin.
All experiment-specific values are mandatory Shell -> argparse arguments.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MODELS = ('M0','M1','M2','M3')
PRED_FIELDS = ['pair_id','project','true_label','predicted_label','mean_logit_margin','n_windows','original_input_tokens','windowed','model','decision_threshold']

def check(condition, message):
    if not condition: raise ValueError(message)

def now():return datetime.now(ZoneInfo('America/Chicago')).isoformat()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def code_sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        for n,line in enumerate(f,1):
            check(bool(line.strip()),f'Blank JSONL line {path}:{n}')
            x=json.loads(line)
            check(isinstance(x,dict),f'Nonobject JSONL at {path}:{n}')
            yield x

def write_json(path,obj):
    path=Path(path)
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,sort_keys=True)
        f.write('\n')

def gt_check(a):
    gt=a.ground_truth
    m=read_json(gt/'manifest.json')
    names=('positive_verified_pairs.jsonl','negative_verified_pairs.jsonl','verified_pairs.jsonl',
           'excluded_negative_candidate_clones.jsonl','annotation_audit.json','leakage_audit.json')
    check(isinstance(m.get('files'),dict) and set(names)<=set(m['files']), 'Ground truth evidence manifest incomplete')
    for name,meta in m['files'].items():
        check(name not in ('.','..') and '/' not in name and '\\' not in name,'Unsafe manifest name')
        path=gt/name
        check(path.is_file() and sha(path)==meta['sha256'],f'Ground truth evidence hash mismatch: {name}')
        if 'bytes' in meta:check(path.stat().st_size==meta['bytes'],f'Byte mismatch {name}')
    datafile=gt/'verified_pairs.jsonl'
    check(sha(datafile)==a.expected_gt_sha256,'Frozen ground truth SHA256 mismatch')
    parsed=[]; seen=set();seen_codes=set();cnt=collections.Counter();projects=collections.Counter()
    for r in rows(datafile):
        pid=r.get('pair_id'); project=r.get('project');label=r.get('gold_label')
        check(isinstance(pid,str) and pid and pid not in seen,'Missing/duplicate pair_id')
        check(isinstance(project,str) and project and project!=a.excluded_project,'Unapproved project')
        check(type(label)==int and label in (0,1) and r.get('evaluation_eligible') is True,'Unapproved gold_label')
        codes=[]; hashes=[]
        for key in ('left','right'):
            side=r.get(key)
            check(isinstance(side,dict),'Missing verified code side')
            code=side.get('source_code')
            check(isinstance(code,str) and code.strip(),'Empty code')
            digest=code_sha(code)
            check(side.get('code_sha256')==digest,'Effective source code SHA mismatch')
            codes.append(code);hashes.append(digest)
        fingerprint=tuple(sorted(hashes))
        check(fingerprint not in seen_codes,'Duplicate held-out code pair')
        seen_codes.add(fingerprint);seen.add(pid)
        parsed.append({'pair_id':pid,'project':project,'label':label,'language':'python','code_a':codes[0],'code_b':codes[1]})
        cnt[label]+=1;projects[project]+=1
    check(len(parsed)==a.expected_pairs and cnt[0]==a.expected_per_class and cnt[1]==a.expected_per_class,'GT count/class imbalance')
    check(len(projects)==a.expected_projects,'GT project count mismatch')
    def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    original=collections.Counter(canonical(x) for name in names[:2] for x in rows(gt/name))
    merged=collections.Counter(canonical(x) for x in rows(datafile))
    check(original==merged,'GT merged pair set differs from signed positive/negative data')
    return parsed,{'verified_pairs_sha256':sha(datafile),'manifest_sha256':sha(gt/'manifest.json'),
                   'project_counts':dict(sorted(projects.items())),'class_counts':dict(sorted(cnt.items()))}

def artifact_check(path,expected):
    path=Path(path)
    check(path.is_file() and path.stat().st_size>1024, f'Missing/empty adapter: {path}')
    digest=sha(path)
    check(expected==digest, f'Adapter SHA256 mismatch: {path} actual={digest}, expected={expected}')
    return {'path':str(path.resolve()),'sha256':digest,'bytes':path.stat().st_size}

def adapters_check(a):
    out={}
    pinned_snapshot=(a.hf_home/'hub'/('models--'+a.model_id.replace('/','--'))/'snapshots'/a.revision).resolve()
    allowed_bases={a.model_id,str(pinned_snapshot)}
    for mid in ('M1','M2','M3'):
        folder=getattr(a,'adapter_'+mid.lower())
        c=folder/'adapter_config.json'
        check(c.is_file(),'Missing LoRA config: '+str(c))
        config=read_json(c)
        check(config.get('base_model_name_or_path') in allowed_bases,'Adapter/model snapshot mismatch '+mid)
        check(str(config.get('peft_type','')).upper()=='LORA','Not a LoRA adapter '+mid)
        expected=getattr(a,'adapter_'+mid.lower()+'_sha256')
        data=artifact_check(folder/'adapter_model.safetensors',expected)
        out[mid]={'adapter_dir':str(folder.resolve()),'weight':data,'config_sha256':sha(c)}
    check(a.m1_training_summary.is_file() and a.m2_training_summary.is_file() and a.m3_training_summary.is_file(), 'Training summaries missing')
    check(a.m2_approval_manifest.is_file() and a.m3_training_evidence.is_file(), 'Approved M2/M3 evidence missing')
    m1,m2,m3=[read_json(p) for p in (a.m1_training_summary,a.m2_training_summary,a.m3_training_summary)]
    m2_approval=read_json(a.m2_approval_manifest)
    check(m2_approval.get('status')=='F01C_V24_M2_TRAINING_COMPLETE','M2 final approval status invalid')
    evidence=read_json(a.m3_training_evidence)
    check(sha(a.m3_training_evidence)==a.m3_training_evidence_sha256,'M3 training evidence SHA mismatch')
    check(evidence.get('output_hashes',{}).get('adapter/adapter_model.safetensors')==a.adapter_m3_sha256,'M3 trained weight evidence mismatch')
    check(evidence.get('optimizer_steps')==a.expected_train_steps,'M3 evidence step count mismatch')
    check(m3.get('training_evidence_sha256')==a.m3_training_evidence_sha256,'M3 summary/evidence linkage mismatch')
    check(m3.get('status')=='F02A_V3_FULL_M3_TRAINING_COMPLETE','Unapproved M3 training status')
    check(m3.get('optimizer_steps')==a.expected_train_steps and m3.get('full_pair_consumption_verified') is True,'M3 full order incomplete')
    check(m3.get('initial_m1_sha256')==a.adapter_m1_sha256,'M3 was not initialized from pinned M1')
    check(m3.get('curriculum_order_sha256')==a.curriculum_order_sha256,'M3 curriculum hash mismatch')
    check(m3.get('frozen_train_sha256')==a.frozen_train_sha256 and m3.get('frozen_validation_sha256')==a.frozen_validation_sha256,'M3 frozen data hash mismatch')
    check(m2.get('initial_m1_sha256')==a.adapter_m1_sha256,'M2 initial adapter mismatch')
    check(m2.get('global_step')==a.expected_train_steps,'M2 steps mismatch')
    check(m2.get('adapter_sha256',{}).get('adapter_model.safetensors')==a.adapter_m2_sha256,'M2 training summary/weight mismatch')
    # M1 summary varies by prior F-01b implementation; only verify a matching weight signature where present.
    if 'adapter_sha256' in m1:
        recorded=m1['adapter_sha256']
        if isinstance(recorded,dict) and 'adapter_model.safetensors' in recorded:
            check(recorded['adapter_model.safetensors']==a.adapter_m1_sha256,'M1 summary/weight mismatch')
    for p in (a.frozen_train,a.frozen_validation,a.curriculum_order):
        check(p.is_file(),'Missing frozen experiment input '+str(p))
    check(sha(a.frozen_train)==a.frozen_train_sha256 and sha(a.frozen_validation)==a.frozen_validation_sha256, 'Frozen dataset bytes drift')
    check(sha(a.curriculum_order)==a.curriculum_order_sha256,'Curriculum order bytes drift')
    return out,{'M1':sha(a.m1_training_summary),'M2':sha(a.m2_training_summary),'M3':sha(a.m3_training_summary),
                'M2_final_manifest':sha(a.m2_approval_manifest),'M3_evidence':sha(a.m3_training_evidence)}

def parameters(a):
    return {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in ('command','model')}

def preflight(a):
    check(not a.run_dir.exists(),'Run directory already exists; refusing overwrite')
    check(len(a.revision)==40 and all(c in '0123456789abcdef' for c in a.revision),'Model revision must be pinned 40-hex')
    check(a.answer_yes==' YES' and a.answer_no==' NO' and a.threshold==0.,'D-05c answer/threshold changed')
    check(a.max_input_tokens>=512 and a.max_windows>0 and 0<=a.overlap<1,'Invalid Cartesian windowing settings')
    check(a.expected_pairs==2*a.expected_per_class,'Unbalanced expected count')
    check(a.load_mode in ('4bit','bf16') and a.local_files_only and not a.trust_remote_code,'Unsupported inference loading policy')
    snapshot=a.hf_home/'hub'/('models--'+a.model_id.replace('/','--'))/'snapshots'/a.revision
    check(snapshot.is_dir(),'Pinned base model snapshot missing')
    cfg=read_json(snapshot/'config.json')
    check(cfg.get('model_type')==a.expected_model_type,'Base model architecture mismatch')
    parsed,gt=gt_check(a)
    adapters,summaries=adapters_check(a)
    a.run_dir.mkdir(parents=True,exist_ok=False)
    aligned=a.run_dir/'aligned_pairs.jsonl'
    with aligned.open('x',encoding='utf-8') as f:
        for row in parsed:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    proof={'version':'v1','status':'F03A_V1_PREFLIGHT_PASS_NOT_EVALUATED','at_chicago':now(),
           'parameters':parameters(a),'ground_truth':gt,'adapters':adapters,
           'training_summary_sha256':summaries,'aligned_pairs_sha256':sha(aligned),
           'protocol':'D-05c/E-01e lossless Cartesian windows and one-token next-token margin; context-verified CodeLlama YES/NO IDs',
           'limitations':['No test-set threshold fitting','4bit inference differs from BF16 training',
                          'M2 Hugging Face Trainer versus M3 Custom DDP training-loop confounder']}
    write_json(a.run_dir/'preflight.json',proof)
    print(json.dumps({'status':proof['status'],'pairs':len(parsed),'run':str(a.run_dir),'at_chicago':now()}),flush=True)

def checked(a):
    proof=read_json(a.run_dir/'preflight.json')
    check(proof['status']=='F03A_V1_PREFLIGHT_PASS_NOT_EVALUATED','Preflight status invalid')
    check(proof['parameters']==parameters(a),'Shell configuration drift since preflight')
    check(sha(a.run_dir/'aligned_pairs.jsonl')==proof['aligned_pairs_sha256'],'Aligned pairs drift')
    check(sha(a.ground_truth/'verified_pairs.jsonl')==proof['ground_truth']['verified_pairs_sha256'],'GT drift')
    check(sha(a.ground_truth/'manifest.json')==proof['ground_truth']['manifest_sha256'],'GT manifest drift')
    ad,sm=adapters_check(a)
    check(ad==proof['adapters'] and sm==proof['training_summary_sha256'],'Adapter/training lineage drift')
    return proof

def prompt(pair):
    return ('Determine whether the following two code fragments are code clones. '
            'Answer YES if they are clones and NO otherwise.\n'
            'Language: python\nCode A:\n'+pair['code_a']+'\n'
            'Code B:\n'+pair['code_b']+'\nAnswer:')

def window_ranges(n,cap,overlap):
    check(cap>=1 and 0<=overlap<1,'Invalid window ranges')
    if n<=cap:return [(0,n)]
    stride=max(1,int(cap*(1-overlap)));last=n-cap
    starts=list(range(0,last+1,stride))
    if not starts or starts[-1]!=last:starts.append(last)
    return [(s,s+cap) for s in starts]

def source_windows(source,offsets,cap,overlap):
    out=[]
    for left,right in window_ranges(len(offsets),cap,overlap):
        first=0 if left==0 else offsets[left][0]
        last=len(source) if right==len(offsets) else offsets[right][0]
        check(0<=first<=last<=len(source),'Bad character offsets')
        out.append({'start':left,'end':right,'text':source[first:last]})
    return out

def allocate_code_budget(la,lb,budget):
    mini=128
    check(budget>=mini*2,'Insufficient code budget')
    if la<=lb and la<=budget-mini:return max(1,la),budget-la
    if lb<la and lb<=budget-mini:return budget-lb,max(1,lb)
    share=la/max(1,la+lb)
    left=max(mini,min(budget-mini,round(budget*share)))
    return left,budget-left

def tokenize_windows(pair,tok,limit,overlap,max_windows):
    full=tok.encode(prompt(pair),add_special_tokens=False)
    if len(full)<=limit:
        yield full,len(full),1,False
        return
    check(tok.is_fast,'Fast tokenizer is required')
    oa=tok(pair['code_a'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    ob=tok(pair['code_b'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    empty=dict(pair,code_a='',code_b='')
    budget=limit-len(tok.encode(prompt(empty),add_special_tokens=False))-128
    check(budget>=256,'Prompt too large')
    for _ in range(48):
        ca,cb=allocate_code_budget(len(oa),len(ob),budget)
        wa=source_windows(pair['code_a'],oa,ca,overlap)
        wb=source_windows(pair['code_b'],ob,cb,overlap)
        count=len(wa)*len(wb)
        check(count<=max_windows,f'Too many Cartesian windows: {count}>{max_windows}')
        overflow=0
        for x in wa:
            for y in wb:
                size=len(tok.encode(prompt(dict(pair,code_a=x['text'],code_b=y['text'])),add_special_tokens=False))
                overflow=max(overflow,size-limit)
                if overflow:break
            if overflow:break
        if overflow==0:
            for x in wa:
                for y in wb:
                    ids=tok.encode(prompt(dict(pair,code_a=x['text'],code_b=y['text'])),add_special_tokens=False)
                    yield ids,len(full),count,True
            return
        budget-=max(32,overflow+32)
        check(budget>=256,'Window budget exhausted')
    raise RuntimeError('Window planner failed to converge')

def answer_ids(tok,a):
    probe={'code_a':a.probe_code_a,'code_b':a.probe_code_b}
    query=prompt(probe)
    prefix=tok.encode(query,add_special_tokens=False)
    result={}
    for key,answer,expected in (('YES',a.answer_yes,a.yes_token_id),('NO',a.answer_no,a.no_token_id)):
        combined=tok.encode(query+answer,add_special_tokens=False)
        check(combined[:len(prefix)]==prefix and len(combined)==len(prefix)+1,'Contextual answer tokenization drift')
        check(combined[-1]==expected,'Pinned CodeLlama contextual answer token mismatch')
        result[key]=expected
    check(result['YES']!=result['NO'],'YES/NO tokens identical')
    return result

def infer(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    check(torch.cuda.is_available(),'No CUDA GPU')
    check(a.model in MODELS,'Unknown model')
    proof=checked(a)
    target=a.run_dir/f'predictions_{a.model}.csv'
    partial=target.with_suffix('.csv.partial')
    check(not target.exists() and not partial.exists(),'Prediction output already exists')
    torch.manual_seed(a.seed);random.seed(a.seed)
    snapshot=a.hf_home/'hub'/('models--'+a.model_id.replace('/','--'))/'snapshots'/a.revision
    check(snapshot.is_dir(),'Pinned local model snapshot not found: '+str(snapshot))
    opts=dict(local_files_only=True,trust_remote_code=False,use_fast=True)
    tok=AutoTokenizer.from_pretrained(str(snapshot),**opts)
    check(tok.is_fast,'Requires fast tokenizer')
    answer=answer_ids(tok,a)
    if a.load_mode=='4bit':
        from transformers import BitsAndBytesConfig
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
        model=AutoModelForCausalLM.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,quantization_config=quant,device_map={'':0})
    else:
        model=AutoModelForCausalLM.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={'':0})
    if a.model!='M0':
        model=PeftModel.from_pretrained(model,proof['adapters'][a.model]['adapter_dir'],is_trainable=False,local_files_only=True)
    model.eval()
    n=0
    with partial.open('x',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=PRED_FIELDS)
        wr.writeheader()
        for pair in rows(a.run_dir/'aligned_pairs.jsonl'):
            total=0.; count=0;original_tokens=0; n_windows=0; windowed=False
            for ids,original_tokens,n_windows,windowed in tokenize_windows(pair,tok,a.max_input_tokens,a.overlap,a.max_windows):
                # The same next-token YES/NO logit margin as D-05c; no threshold tuning.
                x=torch.tensor([ids],dtype=torch.long,device='cuda:0')
                with torch.inference_mode():
                    logits=model(input_ids=x,use_cache=False).logits[0,-1,:]
                    margin=(logits[answer['YES']]-logits[answer['NO']]).float().item()
                check(math.isfinite(margin),'Nonfinite margin pair '+pair['pair_id'])
                count+=1;total+=margin
                del x,logits
            check(count>0 and count==n_windows,'Window count mismatch')
            mean=total/count
            wr.writerow({'pair_id':pair['pair_id'],'project':pair['project'],'true_label':pair['label'],
              'predicted_label':int(mean>a.threshold),'mean_logit_margin':f'{mean:.10f}',
              'n_windows':count,'original_input_tokens':original_tokens,'windowed':str(windowed).lower(),
              'model':a.model,'decision_threshold':str(a.threshold)})
            n+=1
            if n%a.progress_every==0:
                print(json.dumps({'event':'INFERENCE_PROGRESS','model':a.model,'processed':n,'total':a.expected_pairs,'at_chicago':now()}),flush=True)
        f.flush();os.fsync(f.fileno())
    check(n==a.expected_pairs,'Inference count mismatch; .partial preserved')
    partial.rename(target)
    write_json(a.run_dir/f'predictions_{a.model}.json',{'model':a.model,'rows':n,'csv_sha256':sha(target),
               'completed_at_chicago':now(),'ground_truth_sha256':proof['ground_truth']['verified_pairs_sha256'],
               'adapter_sha256':None if a.model=='M0' else proof['adapters'][a.model]['weight']['sha256'],
               'effective_parameters':proof['parameters'],'answer_token_ids':answer})
    print(json.dumps({'event':'INFERENCE_COMPLETE','model':a.model,'rows':n,'csv_sha256':sha(target),'at_chicago':now()}),flush=True)

def metrics(records):
    tp=sum(x['truth']==1 and x['pred']==1 for x in records)
    tn=sum(x['truth']==0 and x['pred']==0 for x in records)
    fp=sum(x['truth']==0 and x['pred']==1 for x in records)
    fn=sum(x['truth']==1 and x['pred']==0 for x in records)
    n=tp+tn+fp+fn
    den=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    safe=lambda x,y:x/y if y else 0.
    return {'n':n,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'accuracy':safe(tp+tn,n),
            'precision':safe(tp,tp+fp),'recall':safe(tp,tp+fn),
            'f1':safe(2*tp,2*tp+fp+fn),'mcc':(tp*tn-fp*fn)/math.sqrt(den) if den else 0.}

def summarize(a):
    from sklearn.metrics import roc_auc_score,average_precision_score
    proof=checked(a)
    check(not (a.run_dir/'summary.json').exists(),'Existing evaluation summary')
    expected={x['pair_id']:(x['project'],x['label']) for x in rows(a.run_dir/'aligned_pairs.jsonl')}
    check(len(expected)==a.expected_pairs,'Aligned dataset incomplete')
    out={'status':'F03A_V1_M0_M3_HELDOUT_EVALUATION_COMPLETE','version':'v1','at_chicago':now(),
         'ground_truth_sha256':proof['ground_truth']['verified_pairs_sha256'],
         'effective_parameters':proof['parameters'],'protocol':proof['protocol'],
         'model_metrics':{},'prediction_files':{},'limitations':proof['limitations']}
    projects=[]
    for mid in MODELS:
        csvpath=a.run_dir/f'predictions_{mid}.csv'
        metadata=read_json(a.run_dir/f'predictions_{mid}.json')
        check(metadata['csv_sha256']==sha(csvpath) and metadata['rows']==a.expected_pairs,'Prediction SHA/count mismatch '+mid)
        check(metadata['ground_truth_sha256']==out['ground_truth_sha256'],'Prediction GT mismatch '+mid)
        check(metadata['effective_parameters']==proof['parameters'],'Prediction parameters drift '+mid)
        seen=set();samples=[];y=[];scores=[]
        with csvpath.open(encoding='utf-8',newline='') as f:
            rd=csv.DictReader(f)
            check(rd.fieldnames==PRED_FIELDS,'Prediction schema changed')
            for x in rd:
                pid=x['pair_id']
                check(pid in expected and pid not in seen,'Unexpected/duplicate pair')
                seen.add(pid);proj,label=expected[pid]
                check(x['project']==proj and int(x['true_label'])==label,'Label/project mismatch')
                pred=int(x['predicted_label']);margin=float(x['mean_logit_margin'])
                check(pred in (0,1) and math.isfinite(margin) and pred==int(margin>a.threshold),'Decision rule mismatch')
                check(x['model']==mid and float(x['decision_threshold'])==a.threshold,'Model/threshold drift')
                check(int(x['n_windows'])>0,'No inference window')
                samples.append({'project':proj,'truth':label,'pred':pred});y.append(label);scores.append(margin)
        check(seen==set(expected),'Missing heldout prediction pairs')
        met=metrics(samples)
        met['roc_auc']=float(roc_auc_score(y,scores))
        met['average_precision']=float(average_precision_score(y,scores))
        out['model_metrics'][mid]=met
        out['prediction_files'][mid]={'sha256':sha(csvpath),'bytes':csvpath.stat().st_size}
        for proj in sorted({x['project'] for x in samples}):
            projects.append({'model':mid,'project':proj,**metrics([x for x in samples if x['project']==proj])})
    outfile=a.run_dir/'project_metrics.csv'
    with outfile.open('x',encoding='utf-8',newline='') as f:
        cols=['model','project','n','tp','tn','fp','fn','accuracy','precision','recall','f1','mcc']
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(projects)
    write_json(a.run_dir/'summary.json',out)
    names=['aligned_pairs.jsonl','preflight.json','project_metrics.csv','summary.json']
    names += [f'predictions_{mid}.{ext}' for mid in MODELS for ext in ('csv','json')]
    write_json(a.run_dir/'manifest.json',{'version':'v1','at_chicago':now(),
               'files':{name:{'sha256':sha(a.run_dir/name),'bytes':(a.run_dir/name).stat().st_size} for name in names}})
    print(json.dumps({'status':out['status'],'metrics':out['model_metrics'],'at_chicago':now()}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('preflight','infer','summarize'))
    p.add_argument('--model',choices=MODELS)
    for k in ('root','run-dir','ground-truth','hf-home','adapter-m1','adapter-m2','adapter-m3',
              'm1-training-summary','m2-training-summary','m3-training-summary',
              'm2-approval-manifest','m3-training-evidence',
              'frozen-train','frozen-validation','curriculum-order'):
        p.add_argument('--'+k,type=Path,required=True)
    for k in ('model-id','revision','expected-model-type','expected-gt-sha256','excluded-project',
              'adapter-m1-sha256','adapter-m2-sha256','adapter-m3-sha256',
              'curriculum-order-sha256','frozen-train-sha256','frozen-validation-sha256',
              'm3-training-evidence-sha256',
              'load-mode','answer-yes','answer-no','probe-code-a','probe-code-b'):
        p.add_argument('--'+k,required=True)
    for k in ('expected-pairs','expected-per-class','expected-projects','max-input-tokens','max-windows',
              'progress-every','seed','gpu-id','yes-token-id','no-token-id','expected-train-steps'):
        p.add_argument('--'+k,type=int,required=True)
    for k in ('overlap','threshold'):
        p.add_argument('--'+k,type=float,required=True)
    for k in ('local-files-only','trust-remote-code'):
        p.add_argument('--'+k,choices=('true','false'),required=True)
    a=p.parse_args()
    a.local_files_only=a.local_files_only=='true';a.trust_remote_code=a.trust_remote_code=='true'
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.expanduser().resolve())
    if a.command=='preflight':preflight(a)
    elif a.command=='infer':
        check(a.model is not None,'Inference model is required')
        infer(a)
    else:summarize(a)

if __name__=='__main__':
    main()
