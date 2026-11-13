#!/usr/bin/env python3
"""G-01b v1: GPT-OSS-120B frozen source + Harmony + single-process sharded-LoRA feasibility.

No full training, no dataset mutation, no automatic quantization fallback.
Every experiment setting is a required CLI argument supplied by the shell.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v1'


def chicago():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write('\n')


def records(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        for lineno, line in enumerate(handle, 1):
            require(bool(line.strip()), f'empty JSONL record {path}:{lineno}')
            item = json.loads(line)
            require(isinstance(item, dict), f'invalid JSONL object {path}:{lineno}')
            yield item


def canonical_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def boundary(tok, messages, yes, no):
    """Derive a shared assistant-turn prefix from the ACTUAL pinned chat template.

    Also verify prefix-token stability; do not infer GPT-OSS semantics from standalone
    `YES`/`NO` tokenization. No test labels enter the token boundary decision.
    """
    output = {}
    for label, answer in ((1, yes), (0, no)):
        turns = messages + [{'role': 'assistant', 'channel': 'final', 'content': answer}]
        full = tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=False)
        require(isinstance(full, str), 'assistant chat template did not render text')
        index = full.rfind(answer)
        require(index > 0, f'assistant response {answer!r} absent from template')
        prefix, suffix = full[:index], full[index + len(answer):]
        require(bool(suffix), 'template lacks explicit assistant message terminator')
        joined = tok.encode(prefix + answer, add_special_tokens=False)
        pids = tok.encode(prefix, add_special_tokens=False)
        require(joined[:len(pids)] == pids and len(joined) == len(pids) + 1,
                f'assistant continuation is not one stable token: {answer!r}')
        full_ids = tok.apply_chat_template(turns, tokenize=True, add_generation_prompt=False)
        require(full_ids == tok.encode(full, add_special_tokens=False),
                'rendered assistant template does not round-trip to token IDs')
        output[label] = {'prefix': prefix, 'suffix': suffix,
                         'prefix_ids': pids, 'answer_id': joined[-1],
                         'rendered_sha256': canonical_hash(full)}
    require(output[0]['prefix'] == output[1]['prefix'], 'YES/NO assistant prefixes differ')
    require(output[0]['suffix'] == output[1]['suffix'], 'YES/NO terminators differ')
    require(output[0]['answer_id'] != output[1]['answer_id'], 'YES and NO tokens identical')
    return output


def messages_for(a, code_a, code_b):
    user = a.user_prefix + code_a + a.user_separator + code_b + a.user_suffix
    return [{'role': 'system', 'content': a.system_text},
            {'role': 'user', 'content': user}]


def configured(a):
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(a).items()}


def validate_cli(a):
    require(re.fullmatch('[0-9a-f]{40}', a.revision) is not None, 'invalid immutable model revision')
    require('/' in a.model_id and a.model_type == 'gpt_oss', 'invalid model ID/type')
    require(a.quant_method == 'mxfp4', 'quantization must remain MXFP4')
    ids = a.device_ids.split(',')
    require(len(ids) == a.expected_gpus and len(set(ids)) == len(ids) and
            all(x.isdigit() for x in ids), 'GPU ID list and world size mismatch')
    require(a.max_gpu_gib > 0 and a.max_cpu_gib > 0 and a.min_headroom_gib > 0, 'invalid memory budget')
    require(a.dataset_train_count > 0 and a.dataset_val_count > 0, 'invalid source sample counts')
    require(a.frozen_train_count >= a.dataset_train_count and a.frozen_val_count >= a.dataset_val_count,
            'frozen counts invalid')
    require(a.max_seq_length >= a.smoke_seq_cap >= 32, 'invalid sequence budgets')
    require(a.smoke_steps in (1, 2), 'only 1 or 2 smoke optimizer steps allowed')
    require(a.lora_rank > 0 and a.lora_alpha > 0 and 0 <= a.lora_dropout < 1,
            'invalid LoRA controls')
    require(a.learning_rate > 0 and a.max_grad_norm > 0, 'invalid optimization controls')
    require(a.optimizer == 'adamw_torch', 'only audited AdamW optimizer permitted')
    require(a.attention_backend in ('eager', 'sdpa'), 'attention implementation not explicitly audited')
    require(a.device_map == 'balanced', 'only audited balanced sharding supported')
    require(a.precision == 'bf16', 'only BF16 LoRA smoke supported')
    require(a.expected_shards > 0 and a.min_shard_bytes > 0, 'invalid weight shard controls')
    require(a.min_target_modules >= 4, 'too few LoRA target modules')
    require(len(set(a.target_modules.split(','))) == len(a.target_modules.split(',')), 'duplicate LoRA targets')
    require(a.allow_network in ('true', 'false'), 'allow-network must be explicit bool')
    require(a.run_dir.is_absolute() and a.snapshot.is_absolute(), 'absolute paths required')
    require(a.snapshot.name == a.revision, 'model snapshot does not match pinned revision')
    for val in (a.frozen_train_sha256, a.frozen_val_sha256, a.frozen_manifest_sha256):
        require(re.fullmatch('[0-9a-f]{64}', val) is not None, 'frozen SHA pin malformed')
    require(a.mode in ('preflight', 'lora-smoke', 'train-smoke'), 'unknown mode')


def verify_frozen(a):
    paths = {'train': a.frozen_prepared / 'train.jsonl',
             'validation': a.frozen_prepared / 'validation.jsonl',
             'manifest': a.frozen_prepared / 'manifest.json'}
    expected = {'train': a.frozen_train_sha256, 'validation': a.frozen_val_sha256,
                'manifest': a.frozen_manifest_sha256}
    for name, p in paths.items():
        require(p.is_file(), f'missing immutable prepared {name}: {p}')
        require(sha256(p) == expected[name], f'prepared SHA256 differs: {name}')
    manifest = json.loads(paths['manifest'].read_text(encoding='utf-8'))
    src_sha = manifest['dataset']['sha256']['java_source']['data.jsonl']
    require(re.fullmatch('[0-9a-f]{64}', src_sha) is not None,
            'no pinned source code hash in frozen F-01c manifest')
    require(a.source_codes.is_file() and sha256(a.source_codes) == src_sha,
            'source code differs from F-01c frozen manifest')
    def subset(path, expected_total, expected_source):
        all_keys, pairs, histogram = set(), [], collections.Counter()
        total = 0
        for item in records(path):
            total += 1
            corpus = item['corpus']
            require(corpus in ('java_source', 'java_mixed', 'python_mixed'), 'unexpected corpus')
            require(type(item['label']) is int and item['label'] in (0, 1), 'invalid label')
            key = (corpus, str(item['id1']), str(item['id2']), item['label'])
            require(key not in all_keys, 'duplicate frozen pair identity')
            all_keys.add(key)
            if corpus == 'java_source':
                require(not item.get('code_a_override') and not item.get('code_b_override'),
                        'source code unexpectedly overridden')
                pairs.append({'id1': str(item['id1']), 'id2': str(item['id2']), 'label': item['label']})
                histogram[item['label']] += 1
        require(total == expected_total and len(pairs) == expected_source,
                f'frozen row count/source count mismatch: {total}/{len(pairs)}')
        require(histogram[0] == histogram[1], 'source labels not balanced')
        return pairs, dict(histogram)
    train, train_labels = subset(paths['train'], a.frozen_train_count, a.dataset_train_count)
    val, val_labels = subset(paths['validation'], a.frozen_val_count, a.dataset_val_count)
    selected = {item[k] for item in train + val for k in ('id1','id2')}
    codes = {}
    for item in records(a.source_codes):
        key = str(item['idx'])
        if key not in selected:
            continue
        require(key not in codes, f'duplicate code idx {key}')
        code = item.get('func')
        require(isinstance(code, str) and bool(code.strip()), f'invalid code {key}')
        codes[key] = code
    require(set(codes) == selected, f'missing {len(selected) - len(codes)} referenced source code IDs')
    return train, val, codes, {'frozen_prepared_sha256': expected,
            'source_codes_sha256': src_sha,
            'selected_code_count': len(selected),
            'training_pairs': len(train), 'validation_pairs': len(val),
            'training_labels': train_labels, 'validation_labels': val_labels,
            'source_identity_policy': 'F01C_FROZEN_JAVA_SOURCE_SUBSET_NOT_PROVEN_IDENTICAL_TO_F01B_M1'}


def snapshot_check(a):
    from importlib.metadata import version
    import torch, transformers, accelerate, peft
    from transformers import AutoConfig, AutoTokenizer
    require(torch.cuda.is_available() and torch.cuda.device_count() == a.expected_gpus,
            'pinned CUDA GPU count unavailable')
    require(torch.version.cuda is not None and torch.cuda.is_bf16_supported(),
            'BF16 CUDA not supported')
    for name, actual in (('torch', torch.__version__), ('transformers', transformers.__version__),
                         ('peft', peft.__version__), ('accelerate', accelerate.__version__),
                         ('kernels', version('kernels')), ('triton', version('triton')),
                         ('huggingface-hub', version('huggingface-hub'))):
        require(actual.startswith(getattr(a, 'expected_' + name.replace('-', '_'))),
                f'package dependency drift {name}: {actual}')
    for idx in range(a.expected_gpus):
        prop = torch.cuda.get_device_properties(idx)
        require(prop.total_memory / 2**30 >= a.max_gpu_gib + a.min_headroom_gib,
                f'GPU{idx} total VRAM leaves insufficient safety budget')
        require(torch.cuda.get_device_capability(idx)[0] >= a.min_compute_major,
                f'GPU{idx} compute capability not approved')
    require(a.snapshot.is_dir(), 'missing pinned model snapshot')
    cfgp, idxp = a.snapshot / 'config.json', a.snapshot / 'model.safetensors.index.json'
    require(cfgp.is_file() and idxp.is_file(), 'missing model snapshot config/index')
    index = json.loads(idxp.read_text())
    shards = sorted(set(index['weight_map'].values()))
    require(len(shards) == a.expected_shards, 'model shard count differs')
    for shard in shards:
        require(re.fullmatch('model-[0-9]{5}-of-[0-9]{5}\\.safetensors', shard), 'bad shard name')
        require((a.snapshot / shard).is_file() and (a.snapshot / shard).stat().st_size >= a.min_shard_bytes,
                f'missing/incomplete shard {shard}')
    cfg = AutoConfig.from_pretrained(str(a.snapshot), local_files_only=True, trust_remote_code=False)
    require(cfg.model_type == a.model_type and
            cfg.quantization_config.get('quant_method') == a.quant_method, 'model quantization mismatch')
    tok = AutoTokenizer.from_pretrained(str(a.snapshot), local_files_only=True, trust_remote_code=False)
    require(tok.is_fast, 'fast tokenizer required')
    return tok, {'revision': a.revision, 'model_type': cfg.model_type,
                 'quantization_method': a.quant_method,
                 'shards_count': len(shards), 'config_sha256': sha256(cfgp),
                 'index_sha256': sha256(idxp),
                 'packages': {p: getattr(a, 'expected_' + p.replace('-', '_'))
                              for p in ('torch','transformers','peft','accelerate','kernels','triton','huggingface-hub')}}


def tokenize_data(a, tok, train, val, codes):
    counts = collections.Counter()
    over_cap = collections.Counter()
    lengths = {'train': [], 'validation': []}
    candidates = {}
    for split, pairs in [('train', train), ('validation', val)]:
        for index, row in enumerate(pairs):
            bm = boundary(tok, messages_for(a, codes[row['id1']], codes[row['id2']]),
                          a.answer_yes, a.answer_no)
            # Both candidate answers have the same pinned completion prefix.
            n = len(bm[row['label']]['prefix_ids']) + 1
            lengths[split].append(n)
            if n > a.max_seq_length:
                over_cap[split] += 1
            if split == 'train' and n <= a.smoke_seq_cap and row['label'] not in candidates:
                candidates[row['label']] = {
                    'row_index': index, 'row': row, 'prefix_ids': bm[row['label']]['prefix_ids'],
                    'answer_id': bm[row['label']]['answer_id'], 'n_tokens': n,
                    'prefix_hash': canonical_hash(bm[row['label']]['prefix'])}
            if (index + 1) % a.audit_progress_every == 0:
                print(json.dumps({'event': 'TOKEN_AUDIT_PROGRESS', 'split': split,
                     'processed': index + 1, 'total': len(pairs), 'at_chicago': chicago()}), flush=True)
        counts[split] = len(pairs)
    require(len(candidates) == 2, 'no balanced pair of original training rows meets smoke token cap')
    def stats(values):
        v = sorted(values)
        return {'min': v[0], 'median': v[len(v)//2], 'p95': v[min(len(v)-1, math.ceil(0.95*len(v))-1)],
                'max': v[-1], 'mean': sum(v)/len(v)}
    evidence = {'version': VERSION, 'status': 'G01B_V1_CPU_DATASET_BOUNDARY_AUDIT_PASS',
                'at_chicago': chicago(), 'total': dict(counts),
                'max_seq_length': a.max_seq_length,
                'over_max_seq_length': dict(over_cap),
                'length_stats': {s: stats(v) for s, v in lengths.items()},
                'smoke_candidates': {str(k): {'row_index': c['row_index'], 'pair_id1': c['row']['id1'],
                                    'pair_id2': c['row']['id2'], 'tokens': c['n_tokens'],
                                    'prefix_sha256': c['prefix_hash']} for k, c in candidates.items()},
                'sft_target_policy': 'ONE_CONTEXTUAL_ASSISTANT_FINAL_ANSWER_TOKEN_ONLY',
                'no_input_truncation': True,
                'full_training_approved': False}
    return evidence, candidates


def check_quantization(model, a):
    placement = getattr(model, 'hf_device_map', None)
    require(isinstance(placement, dict) and bool(placement), 'missing placement map')
    destinations = {str(x) for x in placement.values()}
    require(not any(x in ('cpu','disk','meta') for x in destinations), 'CPU/disk/meta offloading detected')
    for gpu in range(a.expected_gpus):
        require(str(gpu) in destinations or f'cuda:{gpu}' in destinations,
                f'GPU{gpu} unused by model placement')
    q = getattr(model, 'hf_quantizer', None)
    require(q is not None, 'MXFP4 quantizer not detected; reject dequantized base')
    qc = getattr(q, 'quantization_config', None)
    method = qc.get('quant_method') if isinstance(qc, dict) else getattr(qc, 'quant_method', None)
    dequant = qc.get('dequantize') if isinstance(qc, dict) else getattr(qc, 'dequantize', None)
    require(method == a.quant_method and dequant is not True,
            'base is not native MXFP4 or was dequantized')
    return placement, {'method': method, 'dequantize': dequant}


def memory_snapshot(a):
    import torch
    return [{'gpu': i, 'allocated_gib': torch.cuda.memory_allocated(i)/2**30,
             'reserved_gib': torch.cuda.memory_reserved(i)/2**30,
             'peak_allocated_gib': torch.cuda.max_memory_allocated(i)/2**30}
            for i in range(a.expected_gpus)]


def enforce_memory(a, mem):
    for item in mem:
        require(item['reserved_gib'] <= a.max_gpu_gib + a.min_headroom_gib,
                f'GPU{item["gpu"]} reserved exceeded memory safety budget')
        require(item['peak_allocated_gib'] <= a.max_gpu_gib + a.min_headroom_gib,
                f'GPU{item["gpu"]} peak allocation exceeded memory safety budget')


def lora_smoke(a, tok, candidates, baseline):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.cuda.empty_cache()
    kwargs = {'local_files_only': True, 'trust_remote_code': False,
              'device_map': a.device_map, 'dtype': torch.bfloat16,
              'attn_implementation': a.attention_backend,
              'max_memory': {**{i: f'{a.max_gpu_gib}GiB' for i in range(a.expected_gpus)},
                             'cpu': f'{a.max_cpu_gib}GiB'}}
    model = AutoModelForCausalLM.from_pretrained(str(a.snapshot), **kwargs)
    placement, quant = check_quantization(model, a)
    targets = a.target_modules.split(',')
    target_hits = {x: [] for x in targets}
    for name, module in model.named_modules():
        suffix = name.rsplit('.', 1)[-1]
        if suffix in target_hits:
            require(isinstance(module, torch.nn.Linear),
                    f'LoRA target {name} is not nn.Linear: {type(module).__name__}')
            target_hits[suffix].append(name)
    require(all(len(target_hits[x]) > 0 for x in targets), 'missing one or more requested target module families')
    require(sum(len(v) for v in target_hits.values()) >= a.min_target_modules,
            'too few actual LoRA targets')
    cfg = LoraConfig(r=a.lora_rank, lora_alpha=a.lora_alpha,
                     lora_dropout=a.lora_dropout, bias=a.lora_bias,
                     target_modules=targets, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, cfg)
    all_params = list(model.named_parameters())
    base_trainable = [(n,p) for n,p in all_params if p.requires_grad and 'lora_' not in n]
    require(not base_trainable, 'non-LoRA base parameters unexpectedly trainable')
    trainable = [(n,p) for n,p in all_params if p.requires_grad]
    require(trainable, 'no trainable LoRA parameters')
    require(any('lora_B' in n for n,p in trainable), 'no LoRA B parameter found')
    model.config.use_cache = False
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            'use_reentrant': a.gradient_checkpoint_use_reentrant})
    audit = {'version': VERSION, 'status': 'G01B_V1_MXFP4_LORA_ATTACH_PASS_BACKWARD_NOT_TESTED',
             'at_chicago': chicago(), 'base_model': a.model_id,
             'sharded_placement': placement, 'quantization': quant,
             'target_modules': {key: len(v) for key,v in target_hits.items()},
             'lora_trainable_parameters': sum(p.numel() for _,p in trainable),
             'total_parameters': sum(p.numel() for _,p in all_params),
             'non_lora_trainable_parameter_count': len(base_trainable),
             'memory': memory_snapshot(a), 'gpu_backward_performed': False,
             'full_training_approved': False}
    enforce_memory(a, audit['memory'])
    write_new(a.run_dir / 'lora_attach.json', audit)
    print(json.dumps({'event': 'LORA_ATTACH_PASS', 'trainable_parameters': audit['lora_trainable_parameters'],
                      'memory': audit['memory'], 'at_chicago': chicago()}), flush=True)
    if a.mode == 'lora-smoke':
        return
    require(a.mode == 'train-smoke', 'LoRA smoke mode mismatch')
    # No automatic dequantization, no training a large dataset, no saved adapter.
    model.train()
    opt = torch.optim.AdamW([p for _,p in trainable], lr=a.learning_rate,
                            weight_decay=a.weight_decay)
    before_name, before_param = next((n,p) for n,p in trainable if 'lora_B' in n)
    before = before_param.detach().float().cpu().clone()
    evidence = []
    for step in range(1, a.smoke_steps + 1):
        selected = candidates[step % 2]  # 1 -> Positive, 2 -> Negative
        ids = selected['prefix_ids'] + [selected['answer_id']]
        require(len(ids) <= a.smoke_seq_cap, 'candidate changed beyond smoke cap')
        labels = [-100] * (len(ids)-1) + [selected['answer_id']]
        emb = model.get_input_embeddings().weight
        require(emb.device.type == 'cuda', 'sharded model embedding not on CUDA')
        tensors = {
            'input_ids': torch.tensor([ids], dtype=torch.long, device=emb.device),
            'labels': torch.tensor([labels], dtype=torch.long, device=emb.device),
        }
        opt.zero_grad(set_to_none=True)
        out = model(**tensors, use_cache=False)
        loss = out.loss
        require(loss is not None and torch.isfinite(loss.detach()).item(), 'nonfinite loss')
        loss.backward()
        grads = [(n,p.grad) for n,p in trainable if p.grad is not None]
        require(grads and any(torch.any(g.detach() != 0).item() for _,g in grads),
                'no nonzero LoRA gradients from backward')
        require(all(torch.isfinite(g.detach()).all().item() for _,g in grads),
                'nonfinite LoRA gradient')
        clip = torch.nn.utils.clip_grad_norm_([p for _,p in trainable], a.max_grad_norm)
        require(torch.isfinite(clip).item(), 'nonfinite clipped gradient norm')
        opt.step()
        mem = memory_snapshot(a)
        enforce_memory(a, mem)
        row = {'optimizer_step': step, 'label': selected['row']['label'],
               'source_row_index': selected['row_index'], 'source_code_ids': [selected['row']['id1'], selected['row']['id2']],
               'tokens': len(ids), 'loss': float(loss.detach().float().item()),
               'gradient_norm_before_clip': float(clip.float().item()),
               'learning_rate': a.learning_rate, 'gpu_memory': mem,
               'at_chicago': chicago()}
        evidence.append(row)
        print(json.dumps({'event': 'REAL_MXFP4_SHARDED_LORA_OPTIMIZER_STEP_PASS', **row}), flush=True)
        del out, loss, tensors
    changed = bool(torch.any(before_param.detach().float().cpu() != before).item())
    require(changed, f'LoRA optimizer failed to change parameter {before_name}')
    result = {'version': VERSION, 'status': 'G01B_V1_2STEP_OR_1STEP_LORA_FEASIBILITY_PASS_NOT_M1_TRAINING',
              'at_chicago': chicago(), 'optimizer_steps': len(evidence),
              'training_history': evidence, 'adapter_parameter_update_verified': True,
              'checked_adapter_parameter': before_name, 'full_m1_training_performed': False,
              'full_training_approved': False,
              'limitation': 'Single-process sharded autograd smoke; NOT DDP/FSDP, 2048-token feasibility, or 12000-pair training.',
              'effective_parameters': configured(a),
              'frozen_train_sha256': a.frozen_train_sha256,
              'frozen_validation_sha256': a.frozen_val_sha256}
    write_new(a.run_dir / 'training_smoke.json', result)
    print(json.dumps({'status': result['status'], 'optimizer_steps': len(evidence),
                      'at_chicago': chicago()}), flush=True)


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('preflight','lora-smoke','train-smoke'))
    for n in ('model-id','revision','model-type','quant-method','device-ids','attention-backend',
              'device-map','precision','answer-yes','answer-no','system-text','user-prefix',
              'user-separator','user-suffix','target-modules','lora-bias','optimizer',
              'allow-network','expected-torch','expected-transformers','expected-peft',
              'expected-accelerate','expected-kernels','expected-triton','expected-huggingface-hub'):
        p.add_argument('--'+n, required=True)
    for n in ('root','run-dir','snapshot','frozen-prepared','source-codes'):
        p.add_argument('--'+n, required=True, type=Path)
    for n in ('frozen-train-sha256','frozen-val-sha256','frozen-manifest-sha256'):
        p.add_argument('--'+n, required=True)
    for n in ('expected-shards','min-shard-bytes','expected-gpus','min-compute-major',
              'dataset-train-count','dataset-val-count','frozen-train-count','frozen-val-count',
              'max-seq-length','smoke-seq-cap','smoke-steps','seed','lora-rank','lora-alpha',
              'min-target-modules','audit-progress-every'):
        p.add_argument('--'+n, required=True, type=int)
    for n in ('max-gpu-gib','max-cpu-gib','min-headroom-gib','lora-dropout','learning-rate',
              'weight-decay','max-grad-norm'):
        p.add_argument('--'+n, required=True, type=float)
    for n in ('gradient-checkpointing','gradient-checkpoint-use-reentrant'):
        p.add_argument('--'+n, required=True, choices=('true','false'))
    a = p.parse_args()
    for path in ('root','run_dir','snapshot','frozen_prepared','source_codes'):
        setattr(a, path, getattr(a,path).expanduser().resolve())
    a.gradient_checkpointing = a.gradient_checkpointing == 'true'
    a.gradient_checkpoint_use_reentrant = a.gradient_checkpoint_use_reentrant == 'true'
    return a


def main():
    a = parse()
    validate_cli(a)
    os.environ['HF_HUB_OFFLINE'] = '1' if a.allow_network == 'false' else '0'
    require(not a.run_dir.exists(), 'refusing to overwrite result directory')
    tok, model_audit = snapshot_check(a)
    train, val, codes, frozen_audit = verify_frozen(a)
    print(json.dumps({'event': 'SOURCE_DATA_SHA256_PASS', 'training_rows': len(train),
                      'validation_rows': len(val), 'at_chicago': chicago()}), flush=True)
    token_audit, candidates = tokenize_data(a, tok, train, val, codes)
    baseline = {'version': VERSION, 'status': 'G01B_V1_PREFLIGHT_PASS_NOT_TRAINED',
                'at_chicago': chicago(), 'effective_parameters': configured(a),
                'model': model_audit, 'frozen_dataset': frozen_audit,
                'tokenizer_dataset_audit': token_audit,
                'limitations': ['F-01c frozen Java-source subset need not equal original F-01b M1 sample IDs',
                                'Token feasibility does not prove GPU 2048-token memory feasibility',
                                'Held-out leakage is inherited from F-01c audit, not independently recalculated',
                                'A one-token Harmony boundary does not establish calibrated held-out scoring']}
    a.run_dir.mkdir(parents=True, exist_ok=False)
    write_new(a.run_dir / 'preflight.json', baseline)
    print(json.dumps({'status': baseline['status'],
                      'source_train': len(train), 'source_val': len(val),
                      'over_cap': token_audit['over_max_seq_length'],
                      'at_chicago': chicago()}), flush=True)
    if a.mode != 'preflight':
        lora_smoke(a, tok, candidates, baseline)
    files = sorted(p for p in a.run_dir.iterdir() if p.is_file() and p.suffix == '.json')
    statuses = {'preflight': 'G01B_V1_PREFLIGHT_PASS_NOT_TRAINED',
                'lora-smoke': 'G01B_V1_LORA_ATTACH_PASS_NOT_TRAINED',
                'train-smoke': 'G01B_V1_DIAGNOSTIC_OPTIMIZER_SMOKE_PASS_NOT_M1'}
    write_new(a.run_dir / 'run_manifest.json', {
        'version': VERSION, 'at_chicago': chicago(), 'status': statuses[a.mode],
        'effective_parameters': configured(a),
        'files': {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in files},
        'full_training_approved': False})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[G-01b {VERSION}] FAIL {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise
