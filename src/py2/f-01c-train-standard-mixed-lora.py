#!/usr/bin/env python3
"""F-01c v1 multi-LLM M2 standard mixed LoRA entrypoint.

All model-specific and experiment-specific settings are required CLI arguments
passed explicitly from the standalone F-01c Shell launcher. No silent defaults.
"""
import argparse
import sys
from pathlib import Path

import f01c_core as core


def strict_boolean(raw):
    if raw == 'true':
        return True
    if raw == 'false':
        return False
    raise argparse.ArgumentTypeError('Boolean arguments must be exactly true or false')


def parser_factory():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('preflight', 'prepare', 'worker', 'finalize', 'verify'))
    for name in (
        'm1-source-identity-policy', 'model-mode', 'model-id', 'model-revision', 'model-family', 'expected-model-type',
        'token-policy', 'expected-yes-token', 'expected-no-token', 'target-modules',
        'source-corpus', 'expected-ground-truth-sha256', 'verify-holdout-overlap',
        'a10-audit-file', 'a10-discovery', 'sample-policy', 'answer-yes', 'answer-no',
        'prompt-prefix', 'prompt-b-separator', 'prompt-suffix', 'probe-code-a', 'probe-code-b',
        'precision', 'lora-bias', 'lr-scheduler', 'optimizer', 'report-to', 'save-strategy',
        'eval-strategy', 'metric-for-best-model', 'train-order', 'purpose',
        'gpu-ids', 'log-file', 'log-dir', 'log-prefix', 'log-timestamp', 'log-timestamp-format',
    ):
        p.add_argument('--' + name, required=True)
    for name in ('initial-adapter-sha256', 'prepared-policy', 'dataset-c10-expected-sha256', 'source-pair-ids-sha256', 'm1-manifest-file', 'initial-adapter'):
        p.add_argument('--' + name, required=True)
    for name in (
        'root', 'run-dir', 'source-dir', 'staging-manifest', 'ground-truth',
        'heldout-java-codes', 'heldout-python-codes', 'hf-home',
        'prepared-dir', 'adapter-dir', 'checkpoint-dir', 'java-target-dir', 'python-target-dir',
    ):
        p.add_argument('--' + name, required=True, type=Path)
    for name in (
        'max-overlap-examples', 'expected-a10-test-pairs', 'java-target-seed-offset', 'python-target-seed-offset', 'seed', 'max-train-examples', 'java-target-train', 'python-target-train',
        'max-val-examples', 'source-val', 'java-target-val', 'python-target-val', 'max-seq-length', 'sample-rng-label-stride',
        'sample-oversample-min', 'sample-oversample-divisor', 'sample-val-seed-offset',
        'sample-order-seed-offset', 'expected-world-size', 'lora-rank', 'lora-alpha',
        'per-device-batch-size', 'per-device-eval-batch-size', 'gradient-accumulation-steps',
        'max-steps', 'eval-steps', 'log-steps', 'warmup-steps', 'dataloader-workers',
        'save-total-limit', 'eval-accumulation-steps',
    ):
        p.add_argument('--' + name, type=int, required=True)
    for name in ('min-vram-gib', 'learning-rate', 'lora-dropout', 'weight-decay',
                 'max-grad-norm', 'label-smoothing-factor'):
        p.add_argument('--' + name, type=float, required=True)
    for name in (
        'gradient-checkpointing', 'gradient-checkpoint-use-reentrant',
        'ddp-find-unused-parameters', 'logging-first-step', 'load-best-model-at-end',
        'greater-is-better', 'remove-unused-columns', 'save-safetensors',
        'disable-tqdm', 'use-cache', 'dataloader-drop-last',
        'full-determinism', 'bf16-full-eval',
    ):
        p.add_argument('--' + name, type=strict_boolean, required=True)
    return p


def main():
    p = parser_factory()
    args = p.parse_args()
    args.root = args.root.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.source_dir = args.source_dir.expanduser().resolve()
    args.java_target_dir = args.java_target_dir.expanduser().resolve()
    args.python_target_dir = args.python_target_dir.expanduser().resolve()
    args.initial_adapter = str(Path(args.initial_adapter).expanduser().resolve())
    args.m1_manifest_file = str(Path(args.m1_manifest_file).expanduser().resolve())
    args.staging_manifest = args.staging_manifest.expanduser().resolve()
    args.ground_truth = args.ground_truth.expanduser().resolve()
    args.heldout_java_codes = args.heldout_java_codes.expanduser().resolve()
    args.heldout_python_codes = args.heldout_python_codes.expanduser().resolve()
    args.hf_home = args.hf_home.expanduser().resolve()
    args.prepared_dir = args.prepared_dir.expanduser().resolve()
    args.adapter_dir = args.adapter_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    core.validate_args(args)
    if args.command == 'preflight':
        core.run_preflight(args)
    elif args.command == 'prepare':
        core.run_prepare(args)
    elif args.command == 'worker':
        core.run_worker(args)
    elif args.command == 'finalize':
        core.run_finalize(args)
    elif args.command == 'verify':
        core.run_verify(args)
    else:
        p.error('Unrecognized command')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[F-01c v1] ERROR {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise
