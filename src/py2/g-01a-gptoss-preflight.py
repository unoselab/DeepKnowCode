#!/usr/bin/env python3
"""G-01a v2 GPT-OSS snapshot/tokenizer/MXFP4 3-GPU inference-only smoke.
All experiment values are required CLI arguments from the standalone shell.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v2'

def now():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()

def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(run, name, data):
    run.mkdir(parents=True, exist_ok=True)
    dst = run / name
    with dst.open('x', encoding='utf-8') as out:
        json.dump(data, out, ensure_ascii=False, indent=2, sort_keys=True)
        out.write('\n')
    return dst

def validate_args(a):
    require(re.fullmatch('[0-9a-f]{40}', a.revision) is not None, 'revision must be immutable 40-hex SHA')
    require(a.model_id == 'openai/gpt-oss-120b', 'wrong model ID')
    require(a.model_type == 'gpt_oss', 'wrong expected model type')
    require(a.quant_method == 'mxfp4', 'only the pinned MXFP4 checkpoint is permitted')
    require(a.device_ids == '0,1,2' and a.expected_gpus == 3, 'only three pinned GPUs accepted in v1')
    require(a.max_gpu_gib >= 1 and a.max_cpu_gib >= 1, 'memory budgets invalid')
    require(a.min_compute_major >= 1 and a.max_input_tokens >= 16, 'invalid hardware/token budget')
    require(a.attention_backend in ('eager','sdpa'), 'unsupported attention backend')
    require(a.allow_network in ('true', 'false'), 'allow-network must be true/false')
    require(a.inference_text.strip() and a.system_text.strip(), 'empty inference messages')
    require(a.max_new_tokens >= 1 and a.max_new_tokens <= 16, 'inference limited to 1..16 tokens')
    require(a.run_dir.is_absolute() and a.snapshot.is_absolute(), 'paths must be absolute')
    require(a.snapshot.name == a.revision, 'snapshot revision mismatch')

def snapshot_audit(a):
    require(a.snapshot.is_dir(), 'snapshot not found: ' + str(a.snapshot))
    cfgpath = a.snapshot / 'config.json'
    ixpath = a.snapshot / 'model.safetensors.index.json'
    require(cfgpath.is_file() and ixpath.is_file(), 'missing config/index')
    cfg = json.loads(cfgpath.read_text())
    index = json.loads(ixpath.read_text())
    require(cfg.get('model_type') == a.model_type, 'model_type mismatch')
    require(cfg.get('architectures') == ['GptOssForCausalLM'], 'unexpected architecture')
    require(cfg.get('quantization_config', {}).get('quant_method') == a.quant_method,
            'MXFP4 quantization config mismatch')
    mapping = index.get('weight_map')
    require(isinstance(mapping, dict) and bool(mapping), 'empty weight_map')
    shards = sorted(set(mapping.values()))
    require(len(shards) == a.expected_shards, 'weight shard count mismatch')
    evidence = []
    for shard in shards:
        require(re.fullmatch(r'model-[0-9]{5}-of-[0-9]{5}\.safetensors', shard), 'unsafe shard name')
        path = a.snapshot / shard
        require(path.is_file() and path.stat().st_size >= a.min_shard_bytes, 'missing/incomplete shard: ' + shard)
        evidence.append({'name': shard, 'bytes': path.stat().st_size})
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        require((a.snapshot / name).is_file(), 'missing tokenizer asset ' + name)
    return {'snapshot': str(a.snapshot), 'revision': a.revision, 'config_sha256': digest(cfgpath),
            'index_sha256': digest(ixpath), 'shards': evidence,
            'shard_total_bytes': sum(x['bytes'] for x in evidence),
            'quantization_config': cfg['quantization_config']}

def inspect_chat_template(tok, messages, ids, rendered):
    """Validate the tokenizer's effective template, not an assumed delimiter spelling.

    GPT-OSS tokenizer assets are pinned by the snapshot. A literal rendered-string
    delimiter assertion can reject legitimate template renderings. Re-encoding the
    rendered text and verifying the messages are present provides a direct check.
    Delimiter presence is recorded as evidence, never silently fabricated.
    """
    require(isinstance(rendered, str) and bool(rendered.strip()),
            'chat template returned no rendered text')
    require(isinstance(ids, list) and len(ids) > 0 and
            all(type(token_id) is int and token_id >= 0 for token_id in ids),
            'chat template returned invalid token IDs')
    require(isinstance(messages, list) and bool(messages), 'empty chat messages')
    positions = []
    for message in messages:
        content = message.get('content')
        require(isinstance(content, str) and bool(content.strip()),
                'invalid chat message content')
        position = rendered.find(content)
        require(position >= 0, 'chat template omitted supplied message content')
        positions.append(position)
    require(positions == sorted(positions) and len(set(positions)) == len(positions),
            'chat template changed message order')
    reconstructed = tok.encode(rendered, add_special_tokens=False)
    require(ids == reconstructed,
            'chat template tokenize=True differs from rendered re-tokenization')
    return {
        'rendered_sha256': hashlib.sha256(rendered.encode('utf-8')).hexdigest(),
        'rendered_prefix_repr': repr(rendered[:160]),
        'rendered_length': len(rendered),
        'token_roundtrip_verified': True,
        'message_contents_preserved': True,
        'message_order_verified': True,
        'rendered_has_im_start': '<|im_start|>' in rendered,
        'rendered_has_im_sep': '<|im_sep|>' in rendered,
        'rendered_has_assistant': 'assistant' in rendered,
        'delimiter_strings_diagnostic_only': True,
    }


def tokenizer_audit(a):
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(str(a.snapshot), local_files_only=True, trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(str(a.snapshot), local_files_only=True, trust_remote_code=False)
    require(cfg.model_type == a.model_type, 'AutoConfig mismatch')
    require(tok.is_fast, 'fast tokenizer required')
    messages = [{'role': 'system', 'content': a.system_text},
                {'role': 'user', 'content': a.inference_text}]
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    require(isinstance(ids, list) and 0 < len(ids) <= a.max_input_tokens, 'chat template token count invalid')
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    template_evidence = inspect_chat_template(tok, messages, ids, rendered)
    yes = tok.encode(a.answer_yes, add_special_tokens=False)
    no = tok.encode(a.answer_no, add_special_tokens=False)
    info = {'model_type': cfg.model_type, 'architectures': cfg.architectures,
            'tokenizer': type(tok).__name__, 'tokenizer_is_fast': tok.is_fast,
            'vocab_size': cfg.vocab_size, 'chat_tokens': len(ids), 'chat_last_20_ids': ids[-20:],
            'template_evidence': template_evidence,
            'rendered_prefix_sha256': template_evidence['rendered_sha256'],
            'yes_token_ids': yes, 'no_token_ids': no,
            'single_token_yes_no': len(yes) == 1 and len(no) == 1 and yes != no,
            'token_boundary_requires_separate_validation': True,
            'harmonized_scoring_approved': False}
    return info, tok, ids

def hardware_audit(a):
    import torch
    from importlib.metadata import version
    import transformers, accelerate
    require(torch.cuda.is_available(), 'CUDA unavailable')
    require(torch.cuda.device_count() == a.expected_gpus,
            'visible GPU count differs from shell configuration')
    properties = []
    for idx in range(a.expected_gpus):
        p = torch.cuda.get_device_properties(idx)
        cc = torch.cuda.get_device_capability(idx)
        require(cc[0] >= a.min_compute_major, 'GPU capability not approved')
        require(p.total_memory / 2**30 >= a.max_gpu_gib + a.min_gpu_headroom_gib,
                'per-GPU memory budget leaves insufficient headroom')
        properties.append({'gpu': idx, 'name': p.name, 'compute': list(cc),
                           'total_gib': p.total_memory / 2**30})
    for name, installed in [('torch', torch.__version__), ('transformers', transformers.__version__),
                            ('accelerate', accelerate.__version__)]:
        require(installed.startswith(getattr(a, 'expected_' + name)), 'dependency drift ' + name)
    for name in ('kernels','triton','huggingface-hub'):
        require(version(name) == getattr(a, 'expected_' + name.replace('-', '_')), 'dependency drift ' + name)
    return {'gpus': properties, 'torch': torch.__version__,
            'transformers': transformers.__version__, 'accelerate': accelerate.__version__,
            'kernels': version('kernels'), 'triton': version('triton'),
            'huggingface_hub': version('huggingface-hub')}

def baseline(a):
    validate_args(a)
    data = {'version': VERSION, 'at_chicago': now(), 'mode': a.mode,
            'effective_parameters': {k: str(v) if isinstance(v, Path) else v
                                     for k, v in vars(a).items()},
            'snapshot': snapshot_audit(a)}
    tokens, tok, ids = tokenizer_audit(a)
    data['tokenizer'] = tokens
    data['hardware'] = hardware_audit(a)
    data['status'] = 'G01A_V2_TOKENIZER_PREFLIGHT_PASS_NOT_LOADED'
    save(a.run_dir, 'tokenizer_preflight.json', data)
    print(json.dumps({'status': data['status'], 'tokens': tokens['chat_tokens'],
                      'yes_no_single_token': tokens['single_token_yes_no'], 'at_chicago': now()}), flush=True)
    return data, tok, ids

def load_smoke(a, tok, ids, preflight):
    import torch
    from transformers import AutoModelForCausalLM
    torch.cuda.empty_cache()
    kwargs = {'local_files_only': True, 'trust_remote_code': False,
              'device_map': 'balanced', 'torch_dtype': 'auto',
              'attn_implementation': a.attention_backend,
              'max_memory': {**{i: f'{a.max_gpu_gib}GiB' for i in range(a.expected_gpus)},
                             'cpu': f'{a.max_cpu_gib}GiB'}}
    model = AutoModelForCausalLM.from_pretrained(str(a.snapshot), **kwargs)
    model.eval()
    placement = getattr(model, 'hf_device_map', None)
    require(isinstance(placement, dict) and placement, 'device_map missing; cannot audit sharding')
    destinations = {str(v) for v in placement.values()}
    require(not {'cpu', 'disk', 'meta'}.intersection(destinations),
            'CPU/disk/meta offload detected: not approved for G-01a')
    require({str(i) for i in range(a.expected_gpus)} <= destinations or
            {f'cuda:{i}' for i in range(a.expected_gpus)} <= destinations,
            'all three GPUs not proven to be used')
    quantizer = getattr(model, 'hf_quantizer', None)
    require(quantizer is not None, 'no MXFP4 quantizer attached; reject potential dequantized load')
    q = getattr(quantizer, 'quantization_config', None)
    method = q.get('quant_method') if isinstance(q, dict) else getattr(q, 'quant_method', None)
    dequant = q.get('dequantize') if isinstance(q, dict) else getattr(q, 'dequantize', None)
    require(method == a.quant_method and dequant is not True,
            'model is dequantized or MXFP4 runtime disabled')
    device_memory = []
    for i in range(a.expected_gpus):
        allocated = torch.cuda.memory_allocated(i) / 2**30
        reserved = torch.cuda.memory_reserved(i) / 2**30
        require(reserved <= a.max_gpu_gib + a.min_gpu_headroom_gib,
                'GPU allocation exceeds protective budget')
        device_memory.append({'gpu': i, 'allocated_gib': allocated, 'reserved_gib': reserved})
    result = {'version': VERSION, 'status': 'G01A_V2_SHARDED_MXFP4_LOAD_PASS_INFERENCE_PENDING',
              'at_chicago': now(), 'model_type': model.config.model_type,
              'device_map': placement, 'quant_method': method, 'dequantize_flag': dequant,
              'gpu_memory': device_memory,
              'limitation': 'Weights loaded across GPUs; this does not establish trainability or numerical fidelity.'}
    save(a.run_dir, 'load_smoke.json', result)
    print(json.dumps({'status': result['status'], 'memory': device_memory, 'at_chicago': now()}), flush=True)
    if a.mode == 'load-smoke':
        return
    # Short forward: language-model logits; no weight mutation, no generation sampling.
    embeddings = model.get_input_embeddings()
    first_device = embeddings.weight.device
    require(first_device.type == 'cuda', 'embedding not on CUDA')
    tensor = torch.tensor([ids], dtype=torch.long, device=first_device)
    with torch.inference_mode():
        out = model(input_ids=tensor, use_cache=False)
        logits = out.logits[0, -1, :].float()
    require(torch.isfinite(logits).all().item(), 'nonfinite forward logits')
    # YES/NO is diagnostic only; contextual scoring still requires separate validation.
    yes_ids = preflight['tokenizer']['yes_token_ids']
    no_ids = preflight['tokenizer']['no_token_ids']
    margin = None
    if len(yes_ids) == 1 and len(no_ids) == 1:
        margin = float(logits[yes_ids[0]].item() - logits[no_ids[0]].item())
        require(math.isfinite(margin), 'nonfinite diagnostic margin')
    evidence = {'version': VERSION, 'status': 'G01A_V2_MXFP4_FORWARD_PASS_INFERENCE_ONLY',
                'at_chicago': now(), 'tokens': len(ids), 'logit_shape': list(out.logits.shape),
                'diagnostic_standalone_yes_no_margin': margin,
                'scoring_protocol_approved': False,
                'training_or_backward_performed': False,
                'gpu_peak_allocated_gib': [torch.cuda.max_memory_allocated(i)/2**30 for i in range(a.expected_gpus)]}
    save(a.run_dir, 'inference_smoke.json', evidence)
    print(json.dumps({'status': evidence['status'], 'at_chicago': now()}), flush=True)

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('tokenizer', 'load-smoke', 'inference-smoke'))
    for x in ('model-id','revision','model-type','quant-method','device-ids',
              'expected-torch','expected-transformers','expected-accelerate',
              'expected-kernels','expected-triton','expected-huggingface-hub',
              'attention-backend','system-text','inference-text','answer-yes','answer-no','allow-network'):
        p.add_argument('--'+x, required=True)
    for x in ('snapshot','run-dir'):
        p.add_argument('--'+x, type=Path, required=True)
    for x in ('expected-shards','min-shard-bytes','expected-gpus','min-compute-major',
              'max-input-tokens','max-new-tokens'):
        p.add_argument('--'+x, type=int, required=True)
    for x in ('max-gpu-gib','max-cpu-gib','min-gpu-headroom-gib'):
        p.add_argument('--'+x, type=float, required=True)
    return p

def main():
    a = parser().parse_args()
    a.snapshot = a.snapshot.resolve()
    a.run_dir = a.run_dir.resolve()
    os.environ['HF_HUB_OFFLINE'] = '1' if a.allow_network == 'false' else '0'
    baseline_data, tok, ids = baseline(a)
    if a.mode != 'tokenizer':
        load_smoke(a, tok, ids, baseline_data)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[G-01a {VERSION}] FAIL {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise
