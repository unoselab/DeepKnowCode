#!/usr/bin/env python3
"""E-01c v1: fail-closed StarCoder2-15B M2 orchestrator for original D-03a Python logic."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MODEL = 'bigcode/starcoder2-15b'
REV = '46d44742909c03ac8cee08eb03fdebce02e193ec'
M1_RUN = '20261110-144838-92489-train'
M1_PREP_HASH = 'c30c94db158210e5c2e4854d7a7e605c50720d250f37fcb96c03ed4e9792f8da'
M1_CONFIG_HASH = 'df4d929e53e062633ecb0ba8962f0193a2d3495b1ba4dee6a89d9c984e90adbd'
M1_WEIGHT_HASH = 'fd2c4b713a8bbc8ae8d9e0b11f3f1eb3923fbe361433628c9b691e7017b3d9ae'
M1_SOURCE_SHA = '688135d21eb6b3ec132a2ca4c35f00a0bfeba389d2c63c5265856d2d810fb2bc'
GROUND_TRUTH_SHA = '309cd69ef64e4c840557fd62d3a732894cc322054f5b80d39956c1411db3cb13'
PYTHON_MIXED = {'data.jsonl':'da8fdd6362df692e225a5bf5b72278fd6211cef75e48e6a539dda637b76a1843',
                'train.txt':'e8c8998f2e73a3587190904fb258ac1ccda306f0be62dec499622f4258603693',
                'valid.txt':'33c5a0506fdbbd8ff5f9604a9be1d4fba1b6b6d52943ff11ff53928861ac48a7'}
SOURCE_GIT_BLOB = 'src/py2/d-03a-train-balanced-mixed-lora.py (DeepKnowCode D-03a standard-mixed implementation)'

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def stamp():return datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')

def now():return datetime.now(ZoneInfo('America/Chicago')).isoformat()

def write_new(p,obj):
    p=Path(p)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')

def require_hash(p,expected):
    if not p.is_file():raise RuntimeError(f'Missing required file: {p}')
    actual=digest(p)
    if actual!=expected:raise RuntimeError(f'SHA256 mismatch for {p}: got {actual}, expected {expected}')

def audit(root,heavy=True):
    source=root/'src/py2/d-03a-train-balanced-mixed-lora.py'
    if not source.is_file():raise RuntimeError(f'Original D-03a M2 Python source missing: {source}. Do not substitute another training algorithm.')
    content=source.read_text(encoding='utf-8')
    for term in ('PeftModel.from_pretrained','standard shuffled mixing','C10_HASHES','budget_policy','global_batch','training_plan'):
        if term not in content:raise RuntimeError(f'D-03a source missing expected implementation marker: {term}')
    m1=root/'results/e-01b'/M1_RUN
    adapter=m1/'adapter'
    config=adapter/'adapter_config.json'
    weights=adapter/'adapter_model.safetensors'
    require_hash(m1/'prepared/manifest.json',M1_PREP_HASH)
    require_hash(m1/'src/py/a-11-train-source-lora.py',M1_SOURCE_SHA) if (m1/'src/py/a-11-train-source-lora.py').exists() else None
    require_hash(config,M1_CONFIG_HASH)
    if heavy:require_hash(weights,M1_WEIGHT_HASH)
    elif not weights.is_file():raise RuntimeError(f'Missing M1 weights: {weights}')
    m=load(m1/'e01b_m1_manifest.json');t=load(m1/'training_summary.json');p=load(m1/'prepared/manifest.json')
    if m.get('status')!='E01B_V1_M1_TRAINING_COMPLETE' or t.get('status')!='TRAINING_COMPLETE' or t.get('step')!=500:
        raise RuntimeError('M1 full-run provenance invalid; smoke adapter not permitted')
    for obj in (m,t):
        if obj.get('model_id')!=MODEL or obj.get('revision')!=REV:raise RuntimeError('M1 model/revision mismatch')
    if t.get('prepared_manifest_sha256')!=M1_PREP_HASH or p.get('model')!={'id':MODEL,'revision':REV}:
        raise RuntimeError('M1 prepared manifest mismatch')
    if t.get('adapter_files_sha256',{}).get('adapter_model.safetensors')!=M1_WEIGHT_HASH or m.get('adapter_sha256',{}).get('adapter_model.safetensors')!=M1_WEIGHT_HASH:
        raise RuntimeError('M1 adapter weight manifest mismatch')
    if load(config).get('base_model_name_or_path')!=MODEL:raise RuntimeError('M1 adapter config base mismatch')
    if digest(root/'src/py/a-11-train-source-lora.py')!=M1_SOURCE_SHA:raise RuntimeError('M1 training source changed after reference run')
    for filename,expected in PYTHON_MIXED.items():
        require_hash(root/'data/training/clean/python_mixed'/filename,expected)
    gt=root/'data/evaluation/ground_truth/python/manifest.json'
    require_hash(gt,GROUND_TRUTH_SHA)
    for c in ('java_source','java_mixed'):
        for fname in ('data.jsonl','train.txt','valid.txt'):
            if not (root/'data/training/clean'/c/fname).is_file():raise RuntimeError(f'Missing training corpus {c}/{fname}')
    for lang in ('java','python'):
        if not (root/'data/evaluation/code'/f'{lang}.jsonl').is_file():raise RuntimeError(f'Missing heldout code data/evaluation/code/{lang}.jsonl')
    return {'status':'E01C_V1_PREFLIGHT_PASS_NOT_TRAINED','created_at_chicago':now(),'model_id':MODEL,'revision':REV,
            'm1_run':str(m1),'initial_adapter':str(adapter),'initial_adapter_sha256':{'adapter_config.json':M1_CONFIG_HASH,'adapter_model.safetensors':M1_WEIGHT_HASH},
            'm1_prepared_manifest_sha256':M1_PREP_HASH,'original_m1_source_sha256':M1_SOURCE_SHA,'original_m2_source':str(source),
            'original_m2_source_sha256':digest(source),'ground_truth_manifest_sha256':GROUND_TRUTH_SHA,
            'protocol':{'budget_policy':'fixed-source','source_train':12000,'target_per_language':3000,
                        'target_train_total':6000,'total_train':18000,'global_batch':24,'max_steps_expected':750,
                        'source_val':412,'java_target_val':60,'python_target_val':40,'seed':2026,
                        'max_seq_length':2048,'bf16':True,'gpus':3,'gradient_accumulation_steps':8,
                        'per_device_batch_size':1,'learning_rate':0.0002,'lora_rank':16,'lora_alpha':32,'lora_dropout':0.05},
            'validation_scope':'Full source files and full M1 weights SHA256' if heavy else 'M1 weight presence only (no tensor checksum)'}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['preflight','prepare','train','verify'])
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--run-dir',type=Path)
    ap.add_argument('--gpus',default='0,1,2')
    a=ap.parse_args();root=a.root.resolve()
    ids=a.gpus.split(',')
    if len(ids)!=3 or len(set(ids))!=3 or any(not re.fullmatch('[0-9]+',x) for x in ids):raise RuntimeError('Exactly three distinct GPU IDs required')
    if a.command=='preflight':print(json.dumps(audit(root),sort_keys=True));return
    if a.command=='verify':
        if a.run_dir is None:raise RuntimeError('--run-dir required for verify')
        run=a.run_dir.resolve();report=audit(root)
        summary=load(run/'training_summary.json')
        if summary.get('status')!='TRAINING_COMPLETE' or summary.get('step')!=750:raise RuntimeError('Incomplete E-01c M2 training')
        if summary.get('model_id')!=MODEL or summary.get('revision')!=REV:raise RuntimeError('M2 model/revision mismatch')
        if summary.get('initial_adapter_sha256',{}).get('adapter_model.safetensors')!=M1_WEIGHT_HASH:raise RuntimeError('M2 initial M1 adapter mismatch')
        if summary.get('prepared_manifest_sha256')!=digest(run/'prepared/manifest.json'):raise RuntimeError('M2 manifest SHA mismatch')
        if summary.get('training_plan',{}).get('max_steps')!=750:raise RuntimeError('M2 training plan mismatch')
        for name,expected in summary.get('adapter_files_sha256',{}).items():require_hash(run/'adapter'/name,expected)
        print(json.dumps({'status':'E01C_V1_M2_VERIFIED','run':str(run),'best_checkpoint':summary.get('best_checkpoint'), 'step':750,'m1_adapter_sha256':M1_WEIGHT_HASH},sort_keys=True));return
    report=audit(root)
    if a.run_dir is None:
        output=root/'results/e-01c'/f'{stamp()}-{a.command}'
    else:output=a.run_dir.resolve()
    if output.exists():raise RuntimeError('Output directory already exists; never overwrite: '+str(output))
    output.parent.mkdir(parents=True,exist_ok=True);output.mkdir()
    write_new(output/'e01c_preflight.json',report)
    script=root/'src/py2/d-03a-train-balanced-mixed-lora.py'
    args=['--root',str(root),'--run-dir',str(output),'--model-id',MODEL,'--revision',REV,
          '--initial-adapter',report['initial_adapter'],'--budget-policy','fixed-source',
          '--target-per-language','3000','--fixed-source','12000', '--source-val','412',
          '--java-target-val','60','--python-target-val','40','--seed','2026',
          '--max-seq-length','2048','--eval-steps','100','--log-steps','10','--warmup-steps','20',
          '--per-device-batch-size','1','--per-device-eval-batch-size','1',
          '--gradient-accumulation-steps','8','--expected-world-size','3','--learning-rate','0.0002',
          '--lora-rank','16','--lora-alpha','32','--lora-dropout','0.05']
    base=[sys.executable,str(script)]
    print('E-01c PREPARE',str(output),flush=True)
    subprocess.run(base+['prepare']+args,check=True,cwd=root)
    prep=load(output/'prepared/manifest.json')
    if prep.get('model')!={'id':MODEL,'revision':REV} or prep.get('training_plan',{}).get('max_steps')!=750:raise RuntimeError('Prepared training budget mismatch')
    if prep.get('initial_adapter_sha256',{}).get('adapter_model.safetensors')!=M1_WEIGHT_HASH:raise RuntimeError('M1 hash not in prepared manifest')
    write_new(output/'e01c_prepared_receipt.json',{'status':'E01C_V1_PREPARED_NOT_TRAINED','prepared_manifest_sha256':digest(output/'prepared/manifest.json'), 'm1_adapter_sha256':M1_WEIGHT_HASH,'original_m2_source_sha256':report['original_m2_source_sha256'],'created_at_chicago':now(),'run':str(output)})
    if a.command=='prepare':print(json.dumps({'status':'E01C_V1_PREPARED_NOT_TRAINED','run':str(output),'steps':750},sort_keys=True));return
    print('E-01c TRAIN 3-GPU BF16 DDP',flush=True)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=a.gpus;env['TOKENIZERS_PARALLELISM']='false';env.setdefault('OMP_NUM_THREADS','4')
    subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=3']+base+['train']+args,check=True,cwd=root,env=env)
    summary=load(output/'training_summary.json')
    if summary.get('status')!='TRAINING_COMPLETE' or summary.get('step')!=750:raise RuntimeError('Training did not complete 750 steps')
    if summary.get('initial_adapter_sha256',{}).get('adapter_model.safetensors')!=M1_WEIGHT_HASH:raise RuntimeError('Trained M2 M1 source mismatch')
    receipt={'status':'E01C_V1_M2_TRAINING_COMPLETE','created_at_chicago':now(),'run':str(output),'model_id':MODEL,'revision':REV,
             'original_m2_source_sha256':report['original_m2_source_sha256'],'m1_adapter_sha256':M1_WEIGHT_HASH,
             'prepared_manifest_sha256':digest(output/'prepared/manifest.json'),'training_summary_sha256':digest(output/'training_summary.json'),
             'adapter_sha256':summary.get('adapter_files_sha256',{}),'best_checkpoint':summary.get('best_checkpoint'),'step':750}
    write_new(output/'e01c_m2_manifest.json',receipt)
    print(json.dumps(receipt,sort_keys=True))

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('E01C_V1_ERROR:',e,file=sys.stderr,flush=True)
        raise
