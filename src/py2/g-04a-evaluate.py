#!/usr/bin/env python3
"""G-04a v1: GPT-OSS-120B M0/M1/M2/M3 on same frozen Python 7,942 heldout pairs.

Read-only evaluation: contextual Harmony final-turn YES/NO logit margins, fixed
threshold zero, tokenizer offset-based lossless Cartesian windows. All experiment
values are Shell-required CLI arguments. Runs M0-M3 sequentially on 3 sharded GPUs.
Historical protocol reference: F-03a v1; GPT-OSS prompt/label boundary: G-02 v2.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MODELS=('M0','M1','M2','M3')
PRED_FIELDS=['pair_id','project','true_label','predicted_label','mean_logit_margin',
             'n_windows','original_input_tokens','windowed','model','decision_threshold']


def need(ok, message):
    if not ok:raise RuntimeError(message)


def now():return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def digest_code(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        for i,s in enumerate(f,1):
            need(bool(s.strip()),f'EMPTY_JSONL:{path}:{i}')
            r=json.loads(s)
            need(isinstance(r,dict),f'INVALID_JSONL:{path}:{i}')
            yield r


def new_json(path,x):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(x,f,sort_keys=True,ensure_ascii=False,indent=2);f.write('\n')


def gt_check(a):
    root=a.ground_truth
    m=read(root/'manifest.json')
    names=('positive_verified_pairs.jsonl','negative_verified_pairs.jsonl','verified_pairs.jsonl',
           'excluded_negative_candidate_clones.jsonl','annotation_audit.json','leakage_audit.json')
    need(isinstance(m.get('files'),dict) and set(names)<=set(m['files']),
         'FROZEN_GROUND_TRUTH_EVIDENCE_MANIFEST_INCOMPLETE')
    for name,metadata in m['files'].items():
        need(name not in ('.','..') and '/' not in name and '\\' not in name,
             'UNSAFE_GT_EVIDENCE_FILE_NAME')
        path=root/name
        need(path.is_file() and sha(path)==metadata['sha256'],
             'GT_EVIDENCE_SHA256_DRIFT:'+name)
        if 'bytes' in metadata:
            need(path.stat().st_size==metadata['bytes'], 'GT_EVIDENCE_SIZE_DRIFT:'+name)
    path=root/'verified_pairs.jsonl'
    need(sha(root/'manifest.json')==a.expected_gt_manifest_sha256,
         'FROZEN_GT_MANIFEST_PIN_MISMATCH')
    need(sha(path)==a.expected_gt_sha256,'FROZEN_GT_PIN_MISMATCH')
    pairs=[]; ids=set();code_pairs=set();counter=collections.Counter();projects=collections.Counter()
    for p in lines(path):
        pid=p.get('pair_id');project=p.get('project');label=p.get('gold_label')
        need(isinstance(pid,str) and bool(pid) and pid not in ids,
             'HELDOUT_PAIR_ID_DUPLICATE_OR_MISSING')
        need(isinstance(project,str) and bool(project) and project!=a.excluded_project,
             'UNAPPROVED_HELDOUT_PROJECT')
        need(type(label)==int and label in (0,1) and p.get('evaluation_eligible') is True,
             'UNAPPROVED_GOLD_LABEL')
        codes=[];hashes=[]
        for side in ('left','right'):
            part=p.get(side)
            need(isinstance(part,dict),'HELDOUT_MISSING_CODE')
            code=part.get('source_code')
            need(isinstance(code,str) and bool(code.strip()),'HELDOUT_EMPTY_CODE')
            h=digest_code(code)
            need(part.get('code_sha256')==h,'HELDOUT_CODE_SHA256_DRIFT')
            codes.append(code);hashes.append(h)
        key=tuple(sorted(hashes))
        need(key not in code_pairs,'HELDOUT_DUPLICATE_CODE_PAIR')
        code_pairs.add(key)
        pairs.append({'pair_id':pid,'project':project,'label':label,
                      'language':'python','code_a':codes[0],'code_b':codes[1]})
        ids.add(pid);counter[label]+=1;projects[project]+=1
    need(len(pairs)==a.expected_pairs and
         counter[0]==a.expected_per_class and counter[1]==a.expected_per_class and
         len(projects)==a.expected_projects,'HELDOUT_COUNT_OR_BALANCE_MISMATCH')
    def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    src=collections.Counter(canonical(p) for name in names[:2] for p in lines(root/name))
    combined=collections.Counter(canonical(p) for p in lines(path))
    need(src==combined,'HELDOUT_POS_NEG_MERGE_MISMATCH')
    return pairs,{'verified_pairs_sha256':sha(path),'manifest_sha256':sha(root/'manifest.json'),
                  'projects':len(projects),'labels':dict(counter)}


def model_sources(a):
    root=a.g03d_run
    reportpath=root/'verification_report.json';manifestpath=root/'verification_manifest.json'
    need(reportpath.is_file() and manifestpath.is_file(),'G03D_PASS_REPORT_NOT_FOUND')
    rm=read(manifestpath)
    need(rm['status']==a.g03d_manifest_status and
         rm['verification_report_sha256']==sha(reportpath),
         'G03D_VERIFIED_REPORT_HASH_DRIFT')
    r=read(reportpath)
    need(r['status']==a.g03d_pass_status and
         r['consumed_pairs_verified']==a.train_pairs and
         r['optimizer_steps_verified']==a.train_steps and
         r['heldout_test_evaluated'] is False and
         r['model_id']==a.model_id and r['revision']==a.revision and
         r['g03b_order_sha256']==a.g03b_order_sha256 and
         r['g02_train_sha256']==a.g02_train_sha256 and
         r['g02_validation_sha256']==a.g02_validation_sha256 and
         r['m2_m3_initialization']==a.expected_init_policy and
         r['actual_adapter_weight_bytes_verified'] is True,
         'G03D_M3_LINEAGE_PROOF_INVALID')
    adapters={}
    snapshot=(a.hf_home/'hub'/('models--'+a.model_id.replace('/','--'))/'snapshots'/a.revision).resolve()
    allowed={a.model_id,str(snapshot)}
    for model,pin in [('M1',a.m1_sha256),('M2',a.m2_sha256),('M3',a.m3_sha256)]:
        val=r['adapter_weights'][model]
        folder=Path(val['adapter_dir']).resolve()
        need(folder.is_dir() and val['weight_sha256']==pin and
             sha(folder/'adapter_model.safetensors')==pin,
             'ADAPTER_WEIGHT_DRIFT:'+model)
        conf=read(folder/'adapter_config.json')
        need(conf.get('peft_type')=='LORA' and
             conf.get('base_model_name_or_path') in allowed,
             'ADAPTER_BASE_MODEL_OR_PEFT_DRIFT:'+model)
        adapters[model]={'adapter_dir':str(folder),'adapter_sha256':pin,
                         'adapter_config_sha256':sha(folder/'adapter_config.json')}
    return {'g03d_report_sha256':sha(reportpath),'g03d_manifest_sha256':sha(manifestpath),
            'g03d_run':str(root.resolve()),'adapters':adapters}


def prompt_messages(a,code_a,code_b):
    need('Language: java\n' in a.user_prefix,'SHELL_PROMPT_LANGUAGE_ANCHOR_ABSENT')
    prefix=a.user_prefix.replace('Language: java\n','Language: python\n')
    return [{'role':'system','content':a.system_text},
            {'role':'user','content':prefix+code_a+a.user_separator+code_b+a.user_suffix}]


def boundary(tok,a,code_a,code_b):
    # Exact G-02/G-03c training-time Harmony final-turn construction.
    from datetime import datetime as _date
    stamp=_date.fromisoformat(a.template_date)
    def date_func(fmt):return stamp.strftime(fmt)
    messages=prompt_messages(a,code_a,code_b)
    answers={}
    for name,text in (('YES',a.answer_yes),('NO',a.answer_no)):
        full=tok.apply_chat_template(messages+[{'role':'assistant','channel':'final','content':text}],
                   tokenize=False,add_generation_prompt=False,strftime_now=date_func)
        need(isinstance(full,str),'GPTOSS_CHAT_TEMPLATE_RENDER_FAILED')
        point=full.rfind(text)
        need(point>0 and bool(full[point+len(text):]),'HARMONY_FINAL_TURN_BOUNDARY_INVALID')
        prefix=full[:point];suffix=full[point+len(text):]
        pids=tok.encode(prefix,add_special_tokens=False)
        joined=tok.encode(prefix+text,add_special_tokens=False)
        need(joined[:len(pids)]==pids and len(joined)==len(pids)+1,
             'HARMONY_ANSWER_NOT_ONE_CONTEXTUAL_TOKEN')
        need(tok.apply_chat_template(messages+[{'role':'assistant','channel':'final','content':text}],
                  tokenize=True,add_generation_prompt=False,strftime_now=date_func)==
             tok.encode(full,add_special_tokens=False),
             'HARMONY_TEMPLATE_TOKEN_IDS_CHANGED')
        answers[name]={'prefix':prefix,'prefix_ids':pids,'id':joined[-1],'suffix':suffix}
    need(answers['YES']['prefix']==answers['NO']['prefix'] and
         answers['YES']['suffix']==answers['NO']['suffix'] and
         answers['YES']['id']!=answers['NO']['id'],
         'HARMONY_LABEL_CONTEXT_MISMATCH')
    return answers


def windows(n,cap,overlap):
    need(cap>=1 and 0<=overlap<1,'WINDOW_POLICY_INVALID')
    if n<=cap:return [(0,n)]
    stride=max(1,int(cap*(1-overlap)));last=n-cap
    starts=list(range(0,last+1,stride))
    if starts[-1]!=last:starts.append(last)
    return [(start,start+cap) for start in starts]


def source_windows(source,offsets,cap,overlap):
    out=[]
    for left,right in windows(len(offsets),cap,overlap):
        first=0 if left==0 else offsets[left][0]
        last=len(source) if right==len(offsets) else offsets[right][0]
        need(0<=first<=last<=len(source),'WINDOW_OFFSETS_INVALID')
        out.append(source[first:last])
    return out


def code_budget(la,lb,budget,min_side):
    need(budget>=2*min_side,'WINDOW_CODE_BUDGET_TOO_SMALL')
    if la<=lb and la<=budget-min_side:return max(1,la),budget-la
    if lb<la and lb<=budget-min_side:return budget-lb,max(1,lb)
    left=max(min_side,min(budget-min_side,round(budget*la/max(1,la+lb))))
    return left,budget-left


def pair_windows(pair,tok,a):
    full=boundary(tok,a,pair['code_a'],pair['code_b'])
    source_tokens=len(full['YES']['prefix_ids'])
    if source_tokens<=a.max_input_tokens:
        yield full['YES']['prefix_ids'],source_tokens,1,False,full['YES']['id'],full['NO']['id']
        return
    need(getattr(tok,'is_fast',False),'GPTOSS_FAST_TOKENIZER_REQUIRED_FOR_WINDOWS')
    ca=tok(pair['code_a'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    cb=tok(pair['code_b'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    empty=boundary(tok,a,'','')
    budget=a.max_input_tokens-len(empty['YES']['prefix_ids'])-a.window_safety_tokens
    need(budget>=a.min_code_tokens*2,'WINDOW_PROMPT_OVERHEAD_EXCEEDS_CONTEXT')
    for iteration in range(a.window_max_iterations):
        left,right=code_budget(len(ca),len(cb),budget,a.min_code_tokens)
        xa=source_windows(pair['code_a'],ca,left,a.window_overlap)
        xb=source_windows(pair['code_b'],cb,right,a.window_overlap)
        n=len(xa)*len(xb)
        need(n<=a.max_windows,f'PAIR_EXCEEDS_WINDOW_CAP:{pair["pair_id"]}:{n}')
        compiled=[];overflow=0
        for aa in xa:
            for bb in xb:
                obj=boundary(tok,a,aa,bb)
                ids=obj['YES']['prefix_ids']
                need(obj['YES']['id']==full['YES']['id'] and
                     obj['NO']['id']==full['NO']['id'],
                     'HARMONY_WINDOW_CONTEXTUAL_LABEL_ID_DRIFT')
                if len(ids)>a.max_input_tokens:
                    overflow=max(overflow,len(ids)-a.max_input_tokens)
                    break
                compiled.append(ids)
            if overflow:break
        if overflow==0:
            for ids in compiled:
                yield ids,source_tokens,n,True,full['YES']['id'],full['NO']['id']
            return
        budget-=max(a.window_backoff_tokens,overflow+a.window_backoff_tokens)
        need(budget>=a.min_code_tokens*2,'WINDOW_BUDGET_EXHAUSTED')
    raise RuntimeError('WINDOW_PLANNER_CONVERGENCE_FAILURE:'+pair['pair_id'])


def params(a):
    # Parameters must be identical between preflight, inference, summary and verify.
    omitted={'command','model','run_dir','log_file'}
    return {k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items() if k not in omitted}


def snapshot_path(a):
    return a.hf_home/'hub'/('models--'+a.model_id.replace('/','--'))/'snapshots'/a.revision


def preflight(a):
    need(not a.run_dir.exists(),'REFUSE_OVERWRITE_G04A_RUN')
    need(a.train_steps>0 and a.train_pairs==a.train_steps*a.train_batch,
         'M2_M3_TRAIN_BUDGET_INVALID')
    need(a.threshold==0.0 and a.answer_yes==' YES' and a.answer_no==' NO',
         'GPTOSS_LABEL_DECISION_PROTOCOL_CHANGED')
    need(a.max_input_tokens>=512 and a.max_windows>0 and a.window_max_iterations>0 and
         0<=a.window_overlap<1 and a.expected_pairs==2*a.expected_per_class,
         'EVALUATION_WINDOW_OR_LABEL_POLICY_INVALID')
    need(a.model_order==','.join(MODELS) and a.local_files_only=='true' and a.trust_remote_code=='false' and
         a.device_map=='balanced' and a.precision=='bf16' and a.quant_method=='mxfp4' and
         a.attention_backend=='eager' and a.expected_gpus==3,
         'GPTOSS_INFERENCE_MODEL_POLICY_CHANGED')
    snap=snapshot_path(a)
    need(snap.is_dir(),'PINNED_GPTOSS_SNAPSHOT_MISSING:'+str(snap))
    cfg=read(snap/'config.json')
    need(cfg.get('model_type')==a.model_type,'PINNED_GPTOSS_MODEL_TYPE_MISMATCH')
    pairs,gt=gt_check(a)
    models=model_sources(a)
    # CPU-only tokenizer preflight; no GPU/model weight allocation.
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(snap),local_files_only=True,
                                      trust_remote_code=False,use_fast=True)
    need(tok.is_fast,'GPTOSS_FAST_TOKENIZER_REQUIRED')
    probe=boundary(tok,a,a.probe_code_a,a.probe_code_b)
    context_ids={'YES':probe['YES']['id'],'NO':probe['NO']['id']}
    need(context_ids['YES']!=context_ids['NO'],'SAME_LABEL_TOKEN_IDS')
    a.run_dir.mkdir(parents=True,exist_ok=False)
    aligned=a.run_dir/'aligned_pairs.jsonl'
    with aligned.open('x',encoding='utf-8') as f:
        for p in pairs:f.write(json.dumps(p,sort_keys=True,ensure_ascii=False)+'\n')
    result={'status':'G04A_V1_PREFLIGHT_PASS_NOT_EVALUATED','version':'v1',
            'at_chicago':now(),'parameters':params(a),'ground_truth':gt,'lineage':models,
            'aligned_pairs_sha256':sha(aligned),'contextual_answer_token_ids':context_ids,
            'protocol':{'harmony_final_turn':True,'template_date':a.template_date,
                        'logit_margin':'logits(YES)-logits(NO)',
                        'aggregation':'unweighted mean across lossless Cartesian windows',
                        'decision_threshold':a.threshold,'max_input_tokens':a.max_input_tokens,
                        'window_overlap':a.window_overlap,'max_windows':a.max_windows,
                        'train_validation_independent_heldout':True,
                        'label_threshold_tuned_on_heldout':False},
            'warnings':['Python heldout only; not an independently verified Java heldout',
                        'M3 training label review remains pending in original immutable manifest',
                        'The heldout test must not be used for checkpoint selection or threshold tuning']}
    new_json(a.run_dir/'preflight.json',result)
    print(json.dumps({'status':result['status'],'pairs':len(pairs),
                      'label_token_ids':context_ids,'run':str(a.run_dir),
                      'at_chicago':now()}),flush=True)


def checked(a):
    proof=read(a.run_dir/'preflight.json')
    need(proof['status']=='G04A_V1_PREFLIGHT_PASS_NOT_EVALUATED','G04A_PREFLIGHT_NOT_PASS')
    need(proof['parameters']==params(a),'G04A_SHELL_PARAMETERS_DRIFT_AFTER_PREFLIGHT')
    need(sha(a.run_dir/'aligned_pairs.jsonl')==proof['aligned_pairs_sha256'],
         'ALIGNED_FROZEN_PAIRS_DRIFT')
    need(sha(a.ground_truth/'verified_pairs.jsonl')==proof['ground_truth']['verified_pairs_sha256'] and
         sha(a.ground_truth/'manifest.json')==proof['ground_truth']['manifest_sha256'],
         'GT_CHANGED_AFTER_PREFLIGHT')
    need(model_sources(a)==proof['lineage'],'G03D_ADAPTER_LINEAGE_DRIFT_AFTER_PREFLIGHT')
    return proof


def quantized_map(model,a):
    placement=getattr(model,'hf_device_map',None)
    need(isinstance(placement,dict) and bool(placement),'GPTOSS_NO_SHARDED_DEVICE_MAP')
    values={str(x) for x in placement.values()}
    need(not any(x in ('cpu','disk','meta') for x in values),'GPTOSS_OFFLOAD_PROHIBITED')
    need(all(str(i) in values or 'cuda:'+str(i) in values for i in range(a.expected_gpus)),
         'GPTOSS_NOT_SHARDED_ACROSS_THREE_GPUS')
    quant=getattr(model,'hf_quantizer',None)
    need(quant is not None,'GPTOSS_NATIVE_MXFP4_QUANTIZER_ABSENT')
    qc=quant.quantization_config
    method=qc.get('quant_method') if isinstance(qc,dict) else getattr(qc,'quant_method',None)
    dequant=qc.get('dequantize') if isinstance(qc,dict) else getattr(qc,'dequantize',None)
    need(method==a.quant_method and dequant is not True,'GPTOSS_MXFP4_QUANTIZATION_DRIFT')
    return {k:str(v) for k,v in placement.items()}


def infer(a):
    need(a.model in MODELS,'EVALUATION_MODEL_UNSET')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer
    need(torch.cuda.is_available() and torch.cuda.device_count()==a.expected_gpus,
         'EXPECTED_THREE_CUDA_GPUS_NOT_VISIBLE')
    for i in range(a.expected_gpus):
        need(torch.cuda.get_device_capability(i)[0]>=a.min_compute_major,
             'GPU_COMPUTE_CAPABILITY_BELOW_PIN')
    proof=checked(a)
    target=a.run_dir/('predictions_'+a.model+'.csv')
    temp=a.run_dir/('predictions_'+a.model+'.csv.partial')
    need(not target.exists() and not temp.exists(),'REFUSE_OVERWRITE_EVALUATION_PREDICTIONS')
    torch.manual_seed(a.seed);random.seed(a.seed)
    tok=AutoTokenizer.from_pretrained(str(snapshot_path(a)),local_files_only=True,
                                       trust_remote_code=False,use_fast=True)
    probe=boundary(tok,a,a.probe_code_a,a.probe_code_b)
    need({'YES':probe['YES']['id'],'NO':probe['NO']['id']}==proof['contextual_answer_token_ids'],
         'CONTEXTUAL_LABEL_IDS_CHANGED')
    opts={'local_files_only':True,'trust_remote_code':False,'device_map':a.device_map,
          'dtype':torch.bfloat16,'attn_implementation':a.attention_backend,
          'max_memory':{**{i:str(a.max_gpu_gib)+'GiB' for i in range(a.expected_gpus)},
                        'cpu':str(a.max_cpu_gib)+'GiB'}}
    model=AutoModelForCausalLM.from_pretrained(str(snapshot_path(a)),**opts)
    placement=quantized_map(model,a)
    if a.model!='M0':
        model=PeftModel.from_pretrained(model,proof['lineage']['adapters'][a.model]['adapter_dir'],
                                       is_trainable=False,local_files_only=True)
    model.eval()
    input_device=model.get_input_embeddings().weight.device
    need(input_device.type=='cuda','EMBEDDING_TENSOR_NOT_ON_CUDA')
    count=0;seen_windows=0;windowed_pairs=0;total_tokens=0
    with temp.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=PRED_FIELDS)
        writer.writeheader()
        for pair in lines(a.run_dir/'aligned_pairs.jsonl'):
            total=0.;n=0;original=0;total_windows=0;was_windowed=False
            for ids,original,total_windows,was_windowed,yes,no in pair_windows(pair,tok,a):
                need(len(ids)<=a.max_input_tokens and len(ids)>0,
                     'MODEL_CONTEXT_TOKENS_OUT_OF_BOUNDS')
                x=torch.tensor([ids],dtype=torch.long,device=input_device)
                with torch.inference_mode():
                    raw=model(input_ids=x,use_cache=False).logits[0,-1,:]
                    margin=(raw[yes]-raw[no]).float().item()
                need(math.isfinite(margin),'NONFINITE_HELDOUT_LOGIT_MARGIN:'+pair['pair_id'])
                total+=margin;n+=1;del x,raw
            need(n==total_windows and n>0,'WINDOW_COUNT_INCONSISTENT')
            mean=total/n
            writer.writerow({'pair_id':pair['pair_id'],'project':pair['project'],
                'true_label':pair['label'],'predicted_label':int(mean>a.threshold),
                'mean_logit_margin':format(mean,'.17g'),'n_windows':n,
                'original_input_tokens':original,'windowed':str(was_windowed).lower(),
                'model':a.model,'decision_threshold':format(a.threshold,'.1f')})
            count+=1;seen_windows+=n;total_tokens+=original
            if was_windowed:windowed_pairs+=1
            if count%a.progress_every==0:
                print(json.dumps({'event':'G04A_V1_HELDOUT_INFERENCE_PROGRESS',
                    'model':a.model,'pairs':count,'windows':seen_windows,
                    'total_pairs':a.expected_pairs,'at_chicago':now()}),flush=True)
            if count%a.cache_clear_every==0:
                torch.cuda.empty_cache()
        f.flush();os.fsync(f.fileno())
    need(count==a.expected_pairs,'HELDOUT_INFERENCE_PAIR_COUNT_INCOMPLETE')
    temp.rename(target)
    metadata={'status':'G04A_V1_MODEL_HELDOUT_INFERENCE_PASS','model':a.model,
        'rows':count,'windows':seen_windows,'windowed_pairs':windowed_pairs,
        'csv_sha256':sha(target),'ground_truth_sha256':proof['ground_truth']['verified_pairs_sha256'],
        'adapter_sha256':None if a.model=='M0' else proof['lineage']['adapters'][a.model]['adapter_sha256'],
        'answer_token_ids':proof['contextual_answer_token_ids'],
        'parameters':params(a),'hf_device_map':placement,'at_chicago':now()}
    new_json(a.run_dir/('predictions_'+a.model+'.json'),metadata)
    print(json.dumps({'status':metadata['status'],'model':a.model,'rows':count,
                      'windows':seen_windows,'csv_sha256':sha(target),
                      'at_chicago':now()}),flush=True)


def metrics(records):
    tp=sum(x['truth']==1 and x['pred']==1 for x in records)
    tn=sum(x['truth']==0 and x['pred']==0 for x in records)
    fp=sum(x['truth']==0 and x['pred']==1 for x in records)
    fn=sum(x['truth']==1 and x['pred']==0 for x in records)
    n=tp+tn+fp+fn
    ratio=lambda a,b:a/b if b else 0.
    denom=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return {'n':n,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'accuracy':ratio(tp+tn,n),
            'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),
            'f1':ratio(2*tp,2*tp+fp+fn),
            'mcc':(tp*tn-fp*fn)/math.sqrt(denom) if denom else 0.}


def predictions(a,mid,proof):
    path=a.run_dir/('predictions_'+mid+'.csv')
    metadata=read(a.run_dir/('predictions_'+mid+'.json'))
    need(metadata['status']=='G04A_V1_MODEL_HELDOUT_INFERENCE_PASS' and
         metadata['model']==mid and metadata['rows']==a.expected_pairs and
         metadata['csv_sha256']==sha(path) and
         metadata['ground_truth_sha256']==proof['ground_truth']['verified_pairs_sha256'] and
         metadata['parameters']==params(a) and
         metadata['answer_token_ids']==proof['contextual_answer_token_ids'],
         'HELDOUT_MODEL_EVIDENCE_DRIFT:'+mid)
    expected={p['pair_id']:(p['project'],p['label']) for p in lines(a.run_dir/'aligned_pairs.jsonl')}
    need(len(expected)==a.expected_pairs,'HELDOUT_ALIGNED_ID_DUPLICATE')
    samples=[];seen=set()
    with path.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        need(reader.fieldnames==PRED_FIELDS,'PREDICTION_CSV_SCHEMA_DRIFT:'+mid)
        for row in reader:
            pid=row['pair_id']
            need(pid in expected and pid not in seen,'INVALID_HELDOUT_PREDICTION_ID')
            seen.add(pid);project,label=expected[pid]
            y=int(row['true_label']);pred=int(row['predicted_label']);score=float(row['mean_logit_margin'])
            need(row['project']==project and y==label and pred in (0,1) and
                 math.isfinite(score) and pred==int(score>a.threshold) and
                 row['model']==mid and float(row['decision_threshold'])==a.threshold and
                 int(row['n_windows'])>0,
                 'HELDOUT_PREDICTION_OR_LABEL_DRIFT:'+mid+':'+pid)
            samples.append({'pair_id':pid,'project':project,'truth':y,'pred':pred,'score':score})
    need(seen==set(expected),'MISSING_HELDOUT_PREDICTIONS:'+mid)
    need(metadata['adapter_sha256']==(None if mid=='M0' else
         proof['lineage']['adapters'][mid]['adapter_sha256']),
         'EVALUATED_ADAPTER_DRIFT:'+mid)
    return samples


def evaluation_summary(a,write):
    from sklearn.metrics import roc_auc_score,average_precision_score
    from scipy.stats import binomtest
    proof=checked(a)
    all_models={};by_model={};projects=[]
    for model in MODELS:
        samples=predictions(a,model,proof)
        by_model[model]={x['pair_id']:x for x in samples}
        met=metrics(samples)
        met['roc_auc']=float(roc_auc_score([x['truth'] for x in samples],
                                            [x['score'] for x in samples]))
        met['average_precision']=float(average_precision_score([x['truth'] for x in samples],
                                                       [x['score'] for x in samples]))
        all_models[model]=met
        for project in sorted({x['project'] for x in samples}):
            projects.append({'model':model,'project':project,**metrics([x for x in samples if x['project']==project])})
    pairwise={}
    for a_model,b_model in [('M0','M1'),('M1','M2'),('M1','M3'),('M2','M3')]:
        aa=by_model[a_model];bb=by_model[b_model]
        a_only=sum(x['pred']==x['truth'] and bb[pid]['pred']!=x['truth'] for pid,x in aa.items())
        b_only=sum(x['pred']!=x['truth'] and bb[pid]['pred']==x['truth'] for pid,x in aa.items())
        total=a_only+b_only
        pairwise[a_model+'_vs_'+b_model]={'a_correct_b_wrong':a_only,
            'a_wrong_b_correct':b_only,'discordant':total,
            'mcnemar_exact_two_sided_p':float(binomtest(min(a_only,b_only),total,0.5).pvalue)
                                     if total else 1.0,
            'accuracy_difference_b_minus_a':all_models[b_model]['accuracy']-all_models[a_model]['accuracy']}
    output={'status':'G04A_V1_M0_M3_PYTHON_HELDOUT_EVALUATION_PASS','version':'v1',
        'at_chicago':now(),'ground_truth_sha256':proof['ground_truth']['verified_pairs_sha256'],
        'g03d_report_sha256':proof['lineage']['g03d_report_sha256'],
        'model_metrics':all_models,'pairwise_comparisons':pairwise,
        'protocol':proof['protocol'],'training_sample_overlap_not_inferred_from_model_accuracy':True,
        'performance_validation_not_used_for_heldout_threshold':True,
        'limitations':proof['warnings']}
    if write:
        need(not (a.run_dir/'summary.json').exists(),'REFUSE_OVERWRITE_EVAL_SUMMARY')
        with (a.run_dir/'project_metrics.csv').open('x',encoding='utf-8',newline='') as f:
            fields=['model','project','n','tp','tn','fp','fn','accuracy','precision','recall','f1','mcc']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(projects)
        new_json(a.run_dir/'summary.json',output)
        names=['aligned_pairs.jsonl','preflight.json','project_metrics.csv','summary.json']
        names += [f'predictions_{m}.{ext}' for m in MODELS for ext in ('csv','json')]
        new_json(a.run_dir/'manifest.json',{'status':'G04A_V1_EVALUATION_MANIFEST_PASS',
                 'version':'v1','at_chicago':now(),
                 'files':{n:{'sha256':sha(a.run_dir/n),
                             'bytes':(a.run_dir/n).stat().st_size} for n in names}})
    return output


def verify_existing(a):
    need(a.run_dir.is_dir(),'EVALUATION_RUN_MISSING')
    original=read(a.run_dir/'summary.json')
    computed=evaluation_summary(a,False)
    for key in ('status','version','ground_truth_sha256','g03d_report_sha256',
                'model_metrics','pairwise_comparisons','protocol','limitations'):
        need(original[key]==computed[key],'EVALUATION_SUMMARY_RECOMPUTE_DRIFT:'+key)
    mf=read(a.run_dir/'manifest.json')
    need(mf['status']=='G04A_V1_EVALUATION_MANIFEST_PASS','EVAL_FILE_MANIFEST_STATUS')
    for name,meta in mf['files'].items():
        need(Path(name).name==name and (a.run_dir/name).is_file() and
             sha(a.run_dir/name)==meta['sha256'] and
             (a.run_dir/name).stat().st_size==meta['bytes'],
             'EVALUATION_OUTPUT_HASH_DRIFT:'+name)
    need(len(mf['files'])==12,'EVAL_MANIFEST_FILE_COUNT_DRIFT')
    print(json.dumps({'status':'G04A_V1_INDEPENDENT_EVALUATION_VERIFY_PASS',
                      'models':list(original['model_metrics']),
                      'heldout_pairs_per_model':a.expected_pairs,
                      'run':str(a.run_dir),'at_chicago':now()}),flush=True)


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('preflight','infer','summarize','verify'))
    p.add_argument('--model',choices=MODELS)
    for k in ('run-dir','g03d-run','ground-truth','hf-home','log-file'):
        p.add_argument('--'+k,type=Path,required=True)
    for k in ('model-id','revision','model-type','precision','quant-method','attention-backend',
              'device-map','device-ids','model-order','local-files-only','trust-remote-code','expected-gt-sha256','expected-gt-manifest-sha256','excluded-project',
              'm1-sha256','m2-sha256','m3-sha256','g03b-order-sha256',
              'g02-train-sha256','g02-validation-sha256','g03d-pass-status',
              'g03d-manifest-status','expected-init-policy',
              'template-date','system-text','user-prefix','user-separator','user-suffix',
              'answer-yes','answer-no','probe-code-a','probe-code-b'):
        p.add_argument('--'+k,required=True)
    for k in ('train-pairs','train-steps','train-batch','expected-pairs','expected-per-class',
              'expected-projects','max-input-tokens','max-windows','min-code-tokens',
              'window-safety-tokens','window-max-iterations','window-backoff-tokens',
              'progress-every','cache-clear-every','seed','expected-gpus','min-compute-major',
              'max-gpu-gib','max-cpu-gib'):
        p.add_argument('--'+k,type=int,required=True)
    for k in ('window-overlap','threshold'):
        p.add_argument('--'+k,type=float,required=True)
    a=p.parse_args()
    a.run_dir=a.run_dir.expanduser().resolve()
    a.g03d_run=a.g03d_run.expanduser().resolve()
    a.ground_truth=a.ground_truth.expanduser().resolve()
    a.hf_home=a.hf_home.expanduser().resolve()
    a.log_file=a.log_file.expanduser().resolve()
    return a

if __name__=='__main__':
    try:
        a=parse()
        if a.command=='preflight':preflight(a)
        elif a.command=='infer':infer(a)
        elif a.command=='summarize':
            out=evaluation_summary(a,True)
            print(json.dumps({'status':out['status'],'model_metrics':out['model_metrics'],
                              'run':str(a.run_dir),'at_chicago':now()}),flush=True)
        else:verify_existing(a)
    except Exception as e:
        print(json.dumps({'event':'G04A_FAIL','error_type':type(e).__name__,
                          'error':str(e),'at_chicago':now()}),file=sys.stderr,flush=True)
        raise
