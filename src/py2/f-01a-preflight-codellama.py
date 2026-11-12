#!/usr/bin/env python3
"""F-01a v4: CodeLlama pinned-model and contextual answer-scoring audit.

No training. --preflight is read-only w.r.t. source inputs and HF cache.
--download explicitly permits cache download; --smoke explicitly loads a 4bit model.
"""
import argparse
import collections
import hashlib
import json
import math
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


def make_prompt(code_a, code_b, prompt_prefix, prompt_b_separator, prompt_suffix):
    """Exact D-05c prompt, with every text component supplied by Shell."""
    return prompt_prefix + code_a + prompt_b_separator + code_b + prompt_suffix


def continuation_ids(tok, prompt_text, answer):
    """Derive the *context-conditioned* answer tokens, not standalone IDs."""
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    require(prompt_ids, 'Empty scoring prompt')
    combined = tok.encode(prompt_text + answer, add_special_tokens=False)
    stable = combined[:len(prompt_ids)] == prompt_ids
    return {'prefix_stable': stable, 'continuation_ids': combined[len(prompt_ids):] if stable else None,
            'prompt_token_count': len(prompt_ids)}


def validate_contextual_prompt(tok, prompt_text, answer_yes, answer_no, yes_token_id, no_token_id):
    """Validate exact one-token continuations at the specified prompt boundary."""
    require(yes_token_id != no_token_id, 'YES/NO token IDs must differ')
    y = continuation_ids(tok, prompt_text, answer_yes)
    n = continuation_ids(tok, prompt_text, answer_no)
    require(y['prefix_stable'] and n['prefix_stable'], 'Prompt prefix changed upon answer append')
    require(y['continuation_ids'] == [yes_token_id], 'YES continuation token ID mismatch')
    require(n['continuation_ids'] == [no_token_id], 'NO continuation token ID mismatch')
    require(y['prompt_token_count'] == n['prompt_token_count'], 'Answer candidates disagree on prompt')
    return y['prompt_token_count']


def contextual_token_audit(tok, prompt_text, answer_yes, answer_no, audit_variants,
                           scoring_policy, expected_yes_token_id, expected_no_token_id):
    """Keep D-05c exact result separate from an explicitly selected contextual policy."""
    require(scoring_policy in ('d05c-exact', 'contextual-one-token'), 'Unknown scoring policy')
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    require(prompt_ids, 'Empty scoring prompt')
    records = {}
    for label, answer in [('YES', answer_yes), ('NO', answer_no)]:
        standalone = tok.encode(answer, add_special_tokens=False)
        combined = tok.encode(prompt_text + answer, add_special_tokens=False)
        stable = combined[:len(prompt_ids)] == prompt_ids
        continuation = combined[len(prompt_ids):] if stable else None
        records[label] = {'answer': answer, 'standalone_ids': standalone,
                          'prompt_prefix_stable': stable, 'continuation_ids': continuation,
                          'continuation_single_token': bool(stable and len(continuation) == 1),
                          'standalone_single_token': len(standalone) == 1}
    yes, no = records['YES'], records['NO']
    d05c_exact = bool(yes['standalone_single_token'] and no['standalone_single_token']
                      and yes['prompt_prefix_stable'] and no['prompt_prefix_stable']
                      and yes['continuation_ids'] == yes['standalone_ids']
                      and no['continuation_ids'] == no['standalone_ids']
                      and yes['standalone_ids'] != no['standalone_ids'])
    candidate = bool(yes['continuation_single_token'] and no['continuation_single_token']
                     and yes['continuation_ids'] != no['continuation_ids'])
    expected = bool(yes['continuation_ids'] == [expected_yes_token_id]
                    and no['continuation_ids'] == [expected_no_token_id]
                    and expected_yes_token_id != expected_no_token_id)
    approved = bool(tok.is_fast and expected and (d05c_exact if scoring_policy=='d05c-exact' else candidate))
    variant_records = []
    for value in audit_variants:
        standalone = tok.encode(value, add_special_tokens=False)
        combined = tok.encode(prompt_text + value, add_special_tokens=False)
        stable = combined[:len(prompt_ids)] == prompt_ids
        suffix = combined[len(prompt_ids):] if stable else None
        variant_records.append({'text': value, 'standalone_ids': standalone,
                                'prompt_prefix_stable': stable, 'continuation_ids': suffix,
                                'continuation_single_token': bool(stable and len(suffix) == 1)})
    return {'class': type(tok).__name__, 'is_fast': bool(tok.is_fast),
            'vocab_size': len(tok), 'prompt_tokens': len(prompt_ids),
            'd05c_exact_compatible': d05c_exact,
            'contextual_single_token_candidate': candidate,
            'expected_contextual_ids_match': expected,
            'selected_scoring_policy': scoring_policy,
            'approved_for_selected_policy': approved,
            'approved_for_evaluation': False,
            'approval_policy': 'Model-specific contextual one-token logit margin; per-window guard required by future evaluator',
            'expected_yes_token_id': expected_yes_token_id, 'expected_no_token_id': expected_no_token_id,
            'answers': records, 'additional_variants': variant_records,
            'note': 'Validation of scoring tokenization is not proof of model accuracy or cross-model calibration.'}


def audit_tokenizer(model_id, revision, answer_yes, answer_no, prompt_text, audit_variants, hf_home,
                    scoring_policy, expected_yes_token_id, expected_no_token_id):
    from transformers import AutoTokenizer
    kwargs = {'revision': revision, 'local_files_only': True,
              'trust_remote_code': False, 'cache_dir': str(hf_home / 'hub')}
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, **kwargs)
    return contextual_token_audit(tok, prompt_text, answer_yes, answer_no, audit_variants,
                                  scoring_policy, expected_yes_token_id, expected_no_token_id)


def audit_frozen_pairs(gt_file, tok, params):
    """Stream-verify full D-05c evaluation prompts at the actual scoring boundary.

    This checks all *unwindowed* full prompts. It does not claim to check every
    dynamically chosen Cartesian window; future evaluators must guard each window.
    """
    counted = collections.Counter()
    pairs = 0
    errors = []
    observed = set()
    digest = hashlib.sha256()
    proxy_count = 0
    failed_count = 0
    with gt_file.open('r', encoding='utf-8-sig') as source:
        for line in source:
            require(line.strip(), 'Blank ground-truth record')
            obj = json.loads(line)
            pid = obj.get('pair_id')
            require(isinstance(pid, str) and pid and pid not in observed, 'Missing/duplicate pair ID')
            observed.add(pid)
            label = obj.get('gold_label')
            require(type(label) is int and label in (0, 1), 'Invalid ground truth label')
            require(obj.get('evaluation_eligible') is True, 'Ineligible evaluation pair')
            a, b = obj.get('left'), obj.get('right')
            require(isinstance(a, dict) and isinstance(b, dict), 'Missing code pair sides')
            ca, cb = a.get('source_code'), b.get('source_code')
            require(isinstance(ca, str) and ca.strip() and isinstance(cb, str) and cb.strip(), 'Empty code pair')
            prompt = make_prompt(ca, cb, params.prompt_prefix, params.prompt_b_separator, params.prompt_suffix)
            try:
                tokens = validate_contextual_prompt(tok, prompt, params.answer_yes, params.answer_no,
                                                    params.expected_yes_token_id, params.expected_no_token_id)
                if params.window_proxy_stride > 0 and pairs % params.window_proxy_stride == 0:
                    # Fixed-prefix/suffix stress probes only: NOT D-05c Cartesian window replay.
                    for xa, xb in ((ca[:params.window_proxy_chars], cb[:params.window_proxy_chars]),
                                   (ca[-params.window_proxy_chars:], cb[-params.window_proxy_chars:])):
                        stress = make_prompt(xa, xb, params.prompt_prefix,
                                             params.prompt_b_separator, params.prompt_suffix)
                        validate_contextual_prompt(tok, stress, params.answer_yes, params.answer_no,
                                                   params.expected_yes_token_id, params.expected_no_token_id)
                        proxy_count += 1
                digest.update(pid.encode('utf-8'))
                digest.update(b'\x00')
                digest.update(hashlib.sha256(prompt.encode('utf-8')).digest())
                digest.update(b'\x00')
            except (ValueError, RuntimeError) as exc:
                failed_count += 1
                if len(errors) < params.max_audit_failures:
                    errors.append({'pair_id': pid, 'reason': str(exc)})
            counted[label] += 1
            pairs += 1
            if params.audit_progress_every > 0 and pairs % params.audit_progress_every == 0:
                print('F-01a CONTEXTUAL_AUDIT_PROGRESS', pairs, '/', params.expected_pairs, flush=True)
    passed = bool(pairs == params.expected_pairs and counted[1] == params.expected_positive
                  and counted[0] == params.expected_negative and failed_count == 0)
    return {'status': 'PASS' if passed else 'FAIL', 'pairs_audited': pairs,
            'expected_pairs': params.expected_pairs,
            'label_counts': {'0': counted[0], '1': counted[1]},
            'window_proxy_prompts_audited': proxy_count,
            'scope': 'All full unwindowed D-05c prompts; sparse deterministic prefix/suffix window proxies',
            'limitation': 'Not a replay of every dynamic Cartesian window; future evaluation must validate each emitted window',
            'pair_prompt_chain_sha256': digest.hexdigest(),
            'failed_pair_count': failed_count, 'first_failures': errors}

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


def smoke_test(model_id, revision, prompt_text, load_mode, cuda_index, hf_home, target_modules,
               prompt_prefix, prompt_b_separator, prompt_suffix, smoke_code_a,
               smoke_clone_code_b, smoke_nonclone_code_b, answer_yes, answer_no,
               expected_yes_token_id, expected_no_token_id):
    """Independently load model, verify projections, and exercise contextual logits."""
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
    names = [name for name, _ in model.named_modules()]
    missing = [target for target in target_modules
               if not any(name.endswith('.' + target) for name in names)]
    require(not missing, 'LoRA target modules missing in loaded model: ' + ','.join(missing))
    samples = []
    scoring_errors = []
    for sample_type, code_b in (('identical_code_probe', smoke_clone_code_b),
                                ('different_code_probe', smoke_nonclone_code_b)):
        prompt = make_prompt(smoke_code_a, code_b, prompt_prefix, prompt_b_separator, prompt_suffix)
        ids = tok.encode(prompt, add_special_tokens=False)
        tensor = torch.tensor([ids], dtype=torch.long, device=f'cuda:{cuda_index}')
        with torch.inference_mode():
            logits = model(input_ids=tensor).logits[0, -1]
        require(bool(torch.isfinite(logits).all()), 'Nonfinite GPU logits')
        entry = {'probe': sample_type, 'prompt_tokens': len(ids),
                 'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                 'finite_logits': True}
        try:
            validate_contextual_prompt(tok, prompt, answer_yes, answer_no,
                                       expected_yes_token_id, expected_no_token_id)
            y = float(logits[expected_yes_token_id].float().item())
            n = float(logits[expected_no_token_id].float().item())
            margin = y - n
            require(all(math.isfinite(z) for z in (y, n, margin)), 'Nonfinite answer logits')
            entry.update({'contextual_scoring': 'PASS', 'yes_logit': y, 'no_logit': n,
                          'yes_minus_no_margin': margin,
                          'prediction_at_zero_threshold': int(margin > 0)})
        except (ValueError, RuntimeError, IndexError) as exc:
            scoring_errors.append(str(exc))
            entry.update({'contextual_scoring': 'FAIL', 'scoring_error': str(exc)})
        samples.append(entry)
        del tensor, logits
    return {'model_class': type(model).__name__, 'dtype': str(dtype), 'load_mode': load_mode,
            'target_modules_verified': target_modules, 'yes_token_id': expected_yes_token_id,
            'no_token_id': expected_no_token_id, 'finite_logits': True,
            'contextual_gpu_score_status': 'PASS' if not scoring_errors else 'FAIL',
            'contextual_gpu_score_errors': scoring_errors,
            'scoring_formula': 'mean(YES_contextual_logit - NO_contextual_logit) > 0',
            'probes': samples,
            'warning': 'Two probes test execution, not prediction accuracy or full held-out performance.'}
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
    require(args.expected_layers > 0 and args.expected_hidden > 0 and args.expected_context > 0,
            'Invalid architecture')
    require(args.answer_yes != args.answer_no, 'YES and NO strings cannot match')
    require(args.expected_yes_token_id >= 0 and args.expected_no_token_id >= 0
            and args.expected_yes_token_id != args.expected_no_token_id, 'Invalid expected token IDs')
    require(args.expected_pairs > 0 and args.expected_positive + args.expected_negative == args.expected_pairs,
            'Invalid expected dataset counts')
    require(args.window_proxy_stride >= 0 and args.window_proxy_chars > 0
            and args.max_audit_failures > 0 and args.audit_progress_every >= 0, 'Invalid audit settings')
    require(args.scoring_policy in ('contextual-one-token', 'd05c-exact'), 'Invalid scoring policy')
    require(args.prompt_text == make_prompt(args.smoke_code_a, args.smoke_clone_code_b,
                                             args.prompt_prefix, args.prompt_b_separator, args.prompt_suffix),
            'Smoke prompt differs from configured D-05c prompt assembly')
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
                             'expected_yes_token_id': args.expected_yes_token_id,
                             'expected_no_token_id': args.expected_no_token_id,
                             'scoring_policy': args.scoring_policy,
                             'prompt_prefix': args.prompt_prefix, 'prompt_b_separator': args.prompt_b_separator,
                             'prompt_suffix': args.prompt_suffix, 'prompt_text': args.prompt_text,
                             'smoke_code_a': args.smoke_code_a,
                             'smoke_clone_code_b': args.smoke_clone_code_b,
                             'smoke_nonclone_code_b': args.smoke_nonclone_code_b,
                             'expected_pairs': args.expected_pairs, 'expected_positive': args.expected_positive,
                             'expected_negative': args.expected_negative,
                             'window_proxy_stride': args.window_proxy_stride,
                             'window_proxy_chars': args.window_proxy_chars,
                             'max_audit_failures': args.max_audit_failures,
                             'audit_progress_every': args.audit_progress_every,
                             'expected_gpus': args.expected_gpus, 'min_vram_gib': args.min_vram_gib,
                             'require_bf16': args.require_bf16, 'load_mode': args.load_mode,
                             'cuda_index': args.cuda_index, 'dataset_paths': args.dataset_paths,
                             'expected_ground_truth_sha256': args.expected_ground_truth_sha256,
                             'audit_variants': args.audit_variants},
              'checks': {}, 'errors': [], 'warnings': [], 'limitations': [
                  'No training, no label validation, no model quality evaluation, no leakage scan.',
                  'Shard completeness and index structure do not verify shard content hashes.',
                  'Contextual IDs cannot be substituted into the unchanged D-05c evaluator.',
                  'All full-pair prompts and only sampled window proxies are audited; runtime per-window guard required.',
                  'A successful GPU smoke is not proof of full evaluation compatibility or performance.']}
    report['scoring_blockers'] = []
    report['scoring_status'] = 'NOT_AUDITED'
    report['gpu_smoke_status'] = 'NOT_RUN'
    report['evaluation_ready'] = False  # Requires future evaluator implementation and per-window guards.
    report['evaluation_engine_integrated'] = False
    report['training_model_ready'] = False
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
            report['errors'].extend(report['checks']['cache']['errors'])
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
                report['checks']['tokenizer'] = audit_tokenizer(
                    args.model_id, args.revision, args.answer_yes, args.answer_no,
                    args.prompt_text, args.audit_variants, hf_home,
                    args.scoring_policy, args.expected_yes_token_id, args.expected_no_token_id)
                if report['checks']['tokenizer']['approved_for_selected_policy']:
                    report['scoring_status'] = ('D05C_EXACT_VERIFIED' if args.scoring_policy == 'd05c-exact'
                                                else 'CONTEXTUAL_ONE_TOKEN_CANDIDATE')
                else:
                    report['scoring_status'] = 'SELECTED_SCORING_BLOCKED'
                    report['scoring_blockers'].append('Configured answer-token policy not satisfied')
                if args.command == 'preflight' and not report['scoring_blockers'] and not report['errors']:
                    from transformers import AutoTokenizer
                    tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision,
                                    local_files_only=True, trust_remote_code=False,
                                    cache_dir=str(hf_home / 'hub'), use_fast=True)
                    gt_path = (root / args.ground_truth_relative).resolve()
                    report['checks']['pair_prompt_audit'] = audit_frozen_pairs(gt_path, tok, args)
                    if report['checks']['pair_prompt_audit']['status'] != 'PASS':
                        report['scoring_blockers'].append('Full held-out prompt audit failed')
                    else:
                        report['scoring_status'] = ('D05C_EXACT_VERIFIED' if args.scoring_policy == 'd05c-exact'
                                                    else 'CONTEXTUAL_ONE_TOKEN_VERIFIED_FULL_PROMPTS')
            except (ImportError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                report['scoring_status'] = 'TOKENIZER_AUDIT_FAIL'
                report['scoring_blockers'].append('Tokenizer audit failed: ' + str(exc))
        else:
            report['scoring_blockers'].append('Tokenizer audit cannot run without cached model')
        if args.command == 'smoke':
            # Independent GPU smoke still runs when scoring policy is blocked.
            if not report['errors']:
                try:
                    report['checks']['smoke'] = smoke_test(
                        args.model_id, args.revision, args.prompt_text, args.load_mode,
                        args.cuda_index, hf_home, args.target_modules,
                        args.prompt_prefix, args.prompt_b_separator, args.prompt_suffix,
                        args.smoke_code_a, args.smoke_clone_code_b, args.smoke_nonclone_code_b,
                        args.answer_yes, args.answer_no,
                        args.expected_yes_token_id, args.expected_no_token_id)
                    report['gpu_smoke_status'] = 'PASS'
                    report['training_model_ready'] = True
                    if report['checks']['smoke']['contextual_gpu_score_status'] != 'PASS':
                        report['scoring_status'] = 'GPU_CONTEXTUAL_PROBE_FAIL'
                        report['scoring_blockers'].append('Contextual GPU answer-logit probe failed')
                except Exception as exc:
                    report['gpu_smoke_status'] = 'FAIL'
                    report['errors'].append('GPU smoke failed: ' + type(exc).__name__ + ': ' + str(exc))
            else:
                report['gpu_smoke_status'] = 'BLOCKED_BY_INFRASTRUCTURE'
        if args.command != 'smoke':
            report['errors'].extend(report['scoring_blockers'])
    except Exception as exc:
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
    if args.command == 'smoke' and report['scoring_blockers']:
        report['warnings'].extend(report['scoring_blockers'])
    report['scoring_protocol_validated'] = (report['scoring_status'] in
                                           ('D05C_EXACT_VERIFIED', 'CONTEXTUAL_ONE_TOKEN_VERIFIED_FULL_PROMPTS'))
    report['status'] = ('F01A_V4_'+args.command.upper()+'_PASS' if not report['errors']
                        else 'F01A_V4_'+args.command.upper()+'_FAIL')
    with args.result_file.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write('\n')
    print(json.dumps({'status': report['status'], 'model_mode': args.model_mode,
                      'result_file': str(args.result_file), 'errors': report['errors'],
                      'warnings': report['warnings'], 'scoring_status': report['scoring_status'],
                      'scoring_protocol_validated': report['scoring_protocol_validated'],
                      'gpu_smoke_status': report['gpu_smoke_status'],
                      'contextual_gpu_score_status': report['checks'].get('smoke', {}).get('contextual_gpu_score_status', 'NOT_RUN'),
                      'training_model_ready': report['training_model_ready'],
                      'evaluation_ready': report['evaluation_ready']},
                     ensure_ascii=False, sort_keys=True), flush=True)
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
    p.add_argument('--audit-variants', nargs='+', required=True)
    for name in ('prompt-prefix', 'prompt-b-separator', 'prompt-suffix',
                 'smoke-code-a', 'smoke-clone-code-b', 'smoke-nonclone-code-b'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--scoring-policy', choices=('d05c-exact','contextual-one-token'), required=True)
    for name in ('expected-yes-token-id','expected-no-token-id','expected-pairs',
                 'expected-positive','expected-negative', 'window-proxy-stride',
                 'window-proxy-chars','max-audit-failures','audit-progress-every'):
        p.add_argument('--'+name, type=int, required=True)
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
        print('F-01a v4 ERROR: '+str(e), file=sys.stderr)
        sys.exit(2)
