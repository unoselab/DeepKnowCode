#!/usr/bin/env python3
"""F-01a v1: pinned CodeLlama base-model readiness, integrity and protocol audit.

No training. --preflight is read-only w.r.t. source inputs and HF cache.
--download explicitly permits cache download; --smoke explicitly loads a 4bit model.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def chicago():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def snapshot_path(hf_home, model_id, revision):
    slug = 'models--' + model_id.replace('/', '--')
    return hf_home / 'hub' / slug / 'snapshots' / revision


def file_info(path):
    return {'bytes': path.stat().st_size, 'sha256': sha256(path)}


def examine_cache(path, expected_type, expected_layers, expected_hidden, expected_heads, expected_context,
                  expected_target_modules):
    out = {'snapshot': str(path), 'exists': path.is_dir(), 'metadata': {}, 'shards': {}, 'errors': [],
           'notes': ['Shard existence/size is checked, but full weight contents are NOT hashed by default.']}
    if not path.is_dir():
        out['errors'].append('Pinned snapshot not cached; use explicit --download after checking access/license.')
        return out
    for name in ('config.json', 'tokenizer_config.json', 'model.safetensors.index.json'):
        item = path / name
        if not item.is_file() or item.stat().st_size == 0:
            out['errors'].append('Missing/empty ' + name)
        else:
            out['metadata'][name] = file_info(item)
    require_tokenizer = [path / 'tokenizer.json', path / 'tokenizer.model']
    if not any(x.is_file() and x.stat().st_size > 0 for x in require_tokenizer):
        out['errors'].append('Missing tokenizer.json/tokenizer.model')
    for name in ('tokenizer.json', 'tokenizer.model', 'special_tokens_map.json'):
        f = path / name
        if f.is_file() and f.stat().st_size:
            out['metadata'][name] = file_info(f)
    if out['errors']:
        return out
    try:
        cfg = json.loads((path / 'config.json').read_text(encoding='utf-8'))
        values = {'model_type': cfg.get('model_type'), 'num_hidden_layers': cfg.get('num_hidden_layers'),
                  'hidden_size': cfg.get('hidden_size'), 'num_attention_heads': cfg.get('num_attention_heads'),
                  'max_position_embeddings': cfg.get('max_position_embeddings'),
                  'architectures': cfg.get('architectures')}
        out['architecture'] = values
        for k, expected in [('model_type', expected_type), ('num_hidden_layers', expected_layers),
                            ('hidden_size', expected_hidden), ('num_attention_heads', expected_heads),
                            ('max_position_embeddings', expected_context)]:
            if cfg.get(k) != expected:
                out['errors'].append(f'Architecture {k} mismatch: {cfg.get(k)} != {expected}')
        index = json.loads((path / 'model.safetensors.index.json').read_text(encoding='utf-8'))
        names = sorted(set(index['weight_map'].values()))
        require(names and all(re.fullmatch(r'[^/\\]+\.safetensors', n) for n in names), 'Invalid shard filenames')
        total = 0
        for name in names:
            f = path / name
            if not f.is_file() or f.stat().st_size < 4096:
                out['errors'].append('Missing/incomplete shard ' + name)
            else:
                out['shards'][name] = {'bytes': f.stat().st_size}
                total += f.stat().st_size
        out['weight_shard_count'] = len(names)
        out['cached_weight_bytes'] = total
        out['index_total_size'] = index.get('metadata', {}).get('total_size')
        out['target_module_expectation'] = expected_target_modules
        weight_map = index['weight_map']
        # Check presence of Llama attention projection module names without loading weights.
        keys = list(weight_map)
        for module in expected_target_modules:
            if not any(('.' + module + '.weight') in k for k in keys):
                out['errors'].append('LoRA target module absent in safetensors index: ' + module)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        out['errors'].append('Invalid metadata: ' + str(exc))
    return out


def audit_tokenizer(model_id, revision, expected_answers, prompt_text, allow_remote, hf_home):
    from transformers import AutoTokenizer
    kwargs = {'revision': revision, 'local_files_only': not allow_remote,
              'trust_remote_code': False, 'cache_dir': str(hf_home / 'hub')}
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, **kwargs)
    result = {'class': type(tok).__name__, 'is_fast': bool(tok.is_fast), 'vocab_size': len(tok),
              'answer_tokens': {}, 'prompt_tokens': len(tok.encode(prompt_text, add_special_tokens=False)),
              'is_compatible': bool(tok.is_fast)}
    for answer in expected_answers:
        ids = tok.encode(answer, add_special_tokens=False)
        result['answer_tokens'][answer] = ids
        if len(ids) != 1:
            result['is_compatible'] = False
    return result


def gpu_audit(expected_gpus, min_vram_gib, require_bf16):
    import torch
    result = {'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
              'bf16_required': require_bf16, 'gpus': [], 'eligible': False}
    if not result['cuda_available']:
        return result
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        result['gpus'].append({'index': i, 'name': prop.name, 'vram_gib': round(prop.total_memory / 2**30, 3)})
    result['bf16_supported'] = bool(torch.cuda.is_bf16_supported())
    result['eligible'] = (len(result['gpus']) >= expected_gpus
                          and all(x['vram_gib'] >= min_vram_gib for x in result['gpus'][:expected_gpus])
                          and (not require_bf16 or result['bf16_supported']))
    return result


def smoke_test(model_id, revision, prompt_text, load_mode, cuda_index, hf_home):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    require(torch.cuda.is_available(), 'CUDA unavailable for smoke')
    require(0 <= cuda_index < torch.cuda.device_count(), 'CUDA device index out of range')
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True,
                                        trust_remote_code=False, cache_dir=str(hf_home / 'hub'), use_fast=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    options = {'revision': revision, 'local_files_only': True, 'trust_remote_code': False,
               'cache_dir': str(hf_home / 'hub'), 'device_map': {'': cuda_index}}
    if load_mode == '4bit':
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                   bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, **options)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **options)
    model.eval()
    ids = tok.encode(prompt_text, add_special_tokens=False)
    tensor = torch.tensor([ids], dtype=torch.long, device=f'cuda:{cuda_index}')
    with torch.inference_mode():
        logits = model(input_ids=tensor).logits[0, -1]
    require(bool(torch.isfinite(logits).all()), 'Nonfinite logits')
    return {'model_class': type(model).__name__, 'token_count': len(ids),
            'vocab_logits': int(logits.numel()), 'dtype': str(dtype), 'load_mode': load_mode,
            'finite_logits': True}


def inventory(paths, root):
    out = {}
    for path in paths:
        p = (root / path).resolve()
        require(p.is_relative_to(root), 'Unsafe dataset path ' + path)
        out[path] = {'exists': p.is_file(), 'bytes': p.stat().st_size if p.is_file() else None,
                     'sha256': sha256(p) if p.is_file() else None}
    return out


def run(args):
    require(re.fullmatch(r'[a-f0-9]{40}', args.revision) is not None, 'Full 40-hex revision required')
    require(args.model_id and '/' in args.model_id, 'Valid Hugging Face model-id required')
    require(args.expected_gpus >= 1 and args.min_vram_gib > 0, 'Invalid GPU requirements')
    require(args.expected_layers > 0 and args.expected_hidden > 0 and args.expected_context > 0, 'Invalid architecture')
    require(args.answer_yes != args.answer_no, 'YES and NO string cannot match')
    require(args.result_file.parent.is_dir(), 'Result directory must exist')
    require(not args.result_file.exists(), 'Refusing to overwrite ' + str(args.result_file))
    root = args.root.resolve()
    hf_home = args.hf_home.expanduser().resolve()
    snap = snapshot_path(hf_home, args.model_id, args.revision)
    report = {'experiment': args.experiment, 'version': args.version, 'model_mode': args.model_mode,
              'mode': args.command, 'created_at_chicago': chicago(), 'model_id': args.model_id,
              'revision': args.revision, 'hf_home': str(hf_home), 'model_snapshot': str(snap),
              'parameters': {'expected_model_type': args.expected_model_type, 'expected_layers': args.expected_layers,
                             'expected_hidden': args.expected_hidden, 'expected_heads': args.expected_heads,
                             'expected_context': args.expected_context, 'target_modules': args.target_modules,
                             'answer_yes': args.answer_yes, 'answer_no': args.answer_no,
                             'expected_gpus': args.expected_gpus, 'min_vram_gib': args.min_vram_gib,
                             'require_bf16': args.require_bf16, 'load_mode': args.load_mode,
                             'cuda_index': args.cuda_index, 'dataset_paths': args.dataset_paths,
                             'expected_ground_truth_sha256': args.expected_ground_truth_sha256},
              'checks': {}, 'errors': [], 'warnings': [], 'limitations': [
                  'No training, no model quality evaluation, no leakage scan.',
                  'Shard completeness and index structure do not verify shard content hashes.',
                  '4bit inference compatibility is confirmed only when --smoke succeeds.',
                  'Tokenizer single-token YES/NO compatibility is mandatory for unchanged D-05c scoring.']}
    try:
        if args.command == 'download':
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=args.model_id, revision=args.revision,
                              allow_patterns=['*.safetensors', '*.json', '*.model', '*.txt'],
                              local_files_only=False, cache_dir=str(hf_home / 'hub'))
        report['checks']['cache'] = examine_cache(snap, args.expected_model_type, args.expected_layers,
                                                    args.expected_hidden, args.expected_heads,
                                                    args.expected_context, args.target_modules)
        if report['checks']['cache']['errors']:
            report['errors'] += report['checks']['cache']['errors']
        if args.dataset_paths:
            report['checks']['datasets'] = inventory(args.dataset_paths, root)
            for path, detail in report['checks']['datasets'].items():
                if not detail['exists']:
                    report['errors'].append('Missing dataset ' + path)
            gt = report['checks']['datasets'].get(args.ground_truth_relative)
            if gt and gt['exists'] and gt['sha256'] != args.expected_ground_truth_sha256:
                report['errors'].append('Ground truth SHA256 mismatch')
        if not args.skip_gpu:
            report['checks']['gpu'] = gpu_audit(args.expected_gpus, args.min_vram_gib, args.require_bf16)
            if not report['checks']['gpu']['eligible']:
                report['errors'].append('GPU requirements not met')
        else:
            report['warnings'].append('GPU inventory skipped by explicit configuration')
        if not report['checks']['cache']['errors']:
            try:
                report['checks']['tokenizer'] = audit_tokenizer(args.model_id, args.revision,
                         (args.answer_yes, args.answer_no), args.prompt_text,
                         False, hf_home)
                if not report['checks']['tokenizer']['is_compatible']:
                    report['errors'].append('Tokenizer is not D-05c one-token YES/NO compatible')
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                report['errors'].append('Tokenizer audit failed: ' + str(exc))
        if args.command == 'smoke' and not report['errors']:
            report['checks']['smoke'] = smoke_test(args.model_id, args.revision, args.prompt_text,
                                                    args.load_mode, args.cuda_index, hf_home)
        elif args.command == 'smoke':
            report['errors'].append('Smoke denied: preflight prerequisites failed')
    except Exception as exc:
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
    report['status'] = ('F01A_V1_'+args.command.upper()+'_PASS' if not report['errors']
                        else 'F01A_V1_'+args.command.upper()+'_FAIL')
    # Never overwrite. Atomic reserve creates a permanent record, including failures.
    with args.result_file.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write('\n')
    print(json.dumps({'status': report['status'], 'model_mode': args.model_mode,
                      'result_file': str(args.result_file), 'errors': report['errors'],
                      'warnings': report['warnings']}, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if not report['errors'] else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('preflight', 'download', 'smoke'))
    for arg in ('experiment', 'version', 'model-mode', 'model-id', 'revision',
                'expected-model-type', 'answer-yes', 'answer-no', 'prompt-text',
                'expected-ground-truth-sha256', 'ground-truth-relative'):
        p.add_argument('--'+arg, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--hf-home', type=Path, required=True)
    p.add_argument('--result-file', type=Path, required=True)
    p.add_argument('--expected-layers', type=int, required=True)
    p.add_argument('--expected-hidden', type=int, required=True)
    p.add_argument('--expected-heads', type=int, required=True)
    p.add_argument('--expected-context', type=int, required=True)
    p.add_argument('--target-modules', nargs='+', required=True)
    p.add_argument('--dataset-paths', nargs='+', required=True)
    p.add_argument('--expected-gpus', type=int, required=True)
    p.add_argument('--min-vram-gib', type=float, required=True)
    p.add_argument('--cuda-index', type=int, required=True)
    p.add_argument('--load-mode', choices=('4bit', 'bf16'), required=True)
    p.add_argument('--require-bf16', choices=('true', 'false'), required=True)
    p.add_argument('--skip-gpu', choices=('true', 'false'), required=True)
    args = p.parse_args()
    args.require_bf16 = args.require_bf16 == 'true'
    args.skip_gpu = args.skip_gpu == 'true'
    return run(args)

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print('F-01a v1 ERROR: '+str(e), file=sys.stderr)
        sys.exit(2)
