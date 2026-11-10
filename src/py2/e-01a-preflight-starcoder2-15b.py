#!/usr/bin/env python3
"""E-01a v1: read-only StarCoder2-15B base-model cache and protocol preflight.

Optional explicit --download downloads the *base* model into Hugging Face cache,
then evaluates it. No GPU model loading or training occurs in this script.
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

VERSION = 'v1'
MODEL = 'bigcode/starcoder2-15b'
CACHE_SLUG = 'models--bigcode--starcoder2-15b'
REQUIRED_META = ('config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json')
OLD_MODEL = 'bigcode/starcoder2-7b'
OLD_REVISION = 'bb9afde76d7945da5745592525db122d4d729eb1'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def iso_now():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def fail(msg):
    raise RuntimeError(msg)


def cache_path():
    if os.environ.get('HF_HUB_CACHE'):
        return Path(os.environ['HF_HUB_CACHE']).expanduser().resolve()
    if os.environ.get('HF_HOME'):
        return (Path(os.environ['HF_HOME']).expanduser() / 'hub').resolve()
    if os.environ.get('XDG_CACHE_HOME'):
        return (Path(os.environ['XDG_CACHE_HOME']).expanduser() / 'huggingface' / 'hub').resolve()
    return Path('~/.cache/huggingface/hub').expanduser().resolve()


def cached_snapshots(cache, pinned_revision):
    model_dir = cache / CACHE_SLUG
    snapshots = model_dir / 'snapshots'
    if not snapshots.is_dir():
        return []
    entries = sorted((p for p in snapshots.iterdir() if p.is_dir()), key=lambda p: p.name)
    if pinned_revision:
        entries = [p for p in entries if p.name == pinned_revision]
    return entries


def examine_snapshot(path):
    info = {'revision': path.name, 'snapshot': str(path), 'files': {}, 'weight_shards': 0,
            'weight_bytes': 0, 'expected_weight_bytes': None, 'errors': []}
    if not re.fullmatch('[0-9a-f]{40}', path.name):
        info['errors'].append('snapshot revision is not a 40-hex commit SHA')
    for name in REQUIRED_META:
        file = path / name
        if not file.is_file() or file.stat().st_size < 2:
            info['errors'].append('missing or empty ' + name)
        else:
            info['files'][name] = {'bytes': file.stat().st_size, 'sha256': digest(file)}
    if info['errors']:
        return info
    try:
        cfg = json.loads((path / 'config.json').read_text(encoding='utf-8'))
        index = json.loads((path / 'model.safetensors.index.json').read_text(encoding='utf-8'))
        assert cfg.get('model_type') == 'starcoder2', 'model_type mismatch'
        assert cfg.get('num_hidden_layers') == 40 and cfg.get('hidden_size') == 6144, 'Unexpected 15B architecture'
        weight_map = index['weight_map']
        assert isinstance(weight_map, dict) and weight_map, 'empty weight map'
        names = sorted(set(weight_map.values()))
        assert all(re.fullmatch(r'.+\.safetensors', name) and '/' not in name for name in names), 'bad shard names'
        info['expected_weight_bytes'] = index.get('metadata', {}).get('total_size')
        for name in names:
            shard = path / name
            if not shard.is_file() or shard.stat().st_size < 4096:
                info['errors'].append('missing/incomplete shard ' + name)
                continue
            info['weight_shards'] += 1
            info['weight_bytes'] += shard.stat().st_size
        info['shard_names'] = names
        if len(names) != info['weight_shards']:
            info['errors'].append('Weight shard completeness check failed')
        # An index can be internally consistent yet have corrupted weights. Do not claim full integrity.
    except (ValueError, KeyError, AssertionError, TypeError) as exc:
        info['errors'].append('Metadata validation failed: ' + str(exc))
    return info


def gpu_inventory():
    report = {'torch_available': False, 'cuda_available': False, 'gpu_count': 0, 'gpus': [],
              'bf16_supported': None, 'torch_version': None}
    try:
        import torch
        report['torch_available'] = True
        report['torch_version'] = torch.__version__
        report['cuda_available'] = torch.cuda.is_available()
        if report['cuda_available']:
            report['gpu_count'] = torch.cuda.device_count()
            report['bf16_supported'] = bool(torch.cuda.is_bf16_supported())
            for i in range(report['gpu_count']):
                props = torch.cuda.get_device_properties(i)
                report['gpus'].append({'index': i, 'name': props.name,
                                       'vram_gib': round(props.total_memory / 2**30, 3)})
    except ImportError:
        report['note'] = 'torch not installed: hardware inventory unavailable'
    return report


def data_inventory(root):
    paths = [root/'data/training/clean/java_source/train.txt',
             root/'data/training/clean/java_mixed/train.txt',
             root/'data/training/clean/python_mixed/train.txt',
             root/'data/evaluation/ground_truth/python/manifest.json',
             root/'data/evaluation/ground_truth/python/verified_pairs.jsonl']
    return {str(x.relative_to(root)): {'exists': x.is_file(), 'bytes': x.stat().st_size if x.is_file() else None}
            for x in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--mode', choices=('check', 'download'), default='check')
    parser.add_argument('--revision', help='Exact 40-hex commit SHA; required for download, optional for check')
    parser.add_argument('--output', type=Path, help='New JSON report file (must not exist); omitted means stdout only')
    parser.add_argument('--skip-hardware', action='store_true', help='Skip optional CUDA/torch inspection')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if args.revision and not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        parser.error('--revision must be a full 40-character commit SHA')
    if args.output and args.output.expanduser().absolute().exists():
        fail('Refusing to overwrite existing report: ' + str(args.output))
    cache = cache_path()
    if args.mode == 'download':
        if not args.revision:
            parser.error('--download requires --revision (40-hex immutable commit SHA); resolve revision first')
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL, revision=args.revision, local_files_only=False,
                          allow_patterns=['*.safetensors', '*.json', '*.txt', '*.model'])
    available = [examine_snapshot(p) for p in cached_snapshots(cache, args.revision)]
    good = [x for x in available if not x['errors']]
    disk = shutil.disk_usage(cache if cache.exists() else Path.home())
    report = {
        'version': VERSION, 'created_at_chicago': iso_now(), 'model_id': MODEL,
        'model_role': 'BASE_PRETRAINED_NOT_INSTRUCT', 'old_7b_model_id': OLD_MODEL,
        'old_7b_revision': OLD_REVISION,
        'cache_dir': str(cache), 'requested_revision': args.revision,
        'snapshots_found': available, 'selected_snapshot': good[-1] if good else None,
        'disk_free_gib': round(disk.free / 2**30, 3),
        'gpu': {'skipped': True} if args.skip_hardware else gpu_inventory(),
        'training_data_inventory': data_inventory(root),
        'protocol_controls': {
            'M0': 'Same 15B base; inference only',
            'M1': 'Train 15B source LoRA from new 15B base',
            'M2': 'Initialize from same frozen 15B M1 adapter; standard mixed order',
            'M3': 'Initialize from same frozen 15B M1 adapter; curriculum order',
            'training_schema': 'Freeze datasets, seeds, prompts, global batch, budget, LoRA settings and evaluation protocol wherever feasible',
            'do_not_reuse': '7B adapters cannot initialize 15B M1/M2/M3',
            'must_audit': ['tokenizer version and YES/NO tokenization', 'LoRA parameter shapes',
                           'GPU BF16/DDP feasibility', 'actual 15B revision', 'holdout leakage',
                           '15B vs 7B training budget parity']},
        'validation_scope': 'Offline filesystem/index/architecture completeness only; does not load model or verify each weight tensor',
        'status': 'E01A_V1_CACHE_PREFLIGHT_PASS_NOT_TRAINED' if good else 'E01A_V1_BASE_MODEL_NOT_CACHED_OR_INCOMPLETE'
    }
    if args.output:
        dest = args.output.expanduser().absolute()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('x', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write('\n')
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not good:
        sys.exit(2)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('E-01a v1 ERROR: ' + str(exc), file=sys.stderr)
        sys.exit(1)
