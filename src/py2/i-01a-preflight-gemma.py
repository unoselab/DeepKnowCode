#!/usr/bin/env python3
"""I-01a v3 Gemma 4 31B IT read-only preflight engine (no optimizer/train/save).

Configuration belongs to i-01a-preflight-gemma.sh. Every experiment parameter is
required on the CLI. Modes: audit, tokenizer, weights, lora-probe.
In v3 the Shell pins the Gemma revision and exact text-only Jinja chat template.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo


class PreflightError(RuntimeError):
    pass


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def chicago() -> str:
    return datetime.datetime.now(ZoneInfo("America/Chicago")).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_read(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    ensure(isinstance(result, dict), f"JSON_OBJECT_REQUIRED:{path}")
    return result


def json_new(path: Path, record: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")


def bool_arg(raw: str) -> bool:
    if raw == "true":
        return True
    if raw == "false":
        return False
    raise argparse.ArgumentTypeError("boolean argument must be exactly true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    fields = {
        "version": str, "experiment": str, "mode": str, "stage": str,
        "run_dir": Path, "log_file": Path, "root": Path, "model_id": str,
        "model_mode": str, "model_type": str, "model_architecture": str,
        "hf_home": Path, "cache_root": Path, "snapshot": Path,
        "model_revision": str, "revision_policy": str, "chat_template_sha256": str,
        "tokenizer_policy": str, "tokenizer_use_fast": bool_arg, "base_precision": str,
        "quantization": str, "require_shard_index": bool_arg,
        "min_weight_gib": float, "device_ids": str, "expected_gpus": int,
        "min_compute_major": int, "gpu_query_bin": str,
        "min_gpu_free_gib": float, "min_transformers": str,
        "min_peft": str, "min_torch": str, "min_accelerate": str,
        "min_safetensors": str, "min_pillow": str, "lora_rank": int, "lora_alpha": int,
        "lora_dropout": float, "lora_bias": str,
        "lora_task_type": str, "low_cpu_mem_usage": bool_arg,
        "lora_target_regex": str,
        "min_target_modules": int, "device_map": str, "seed": int,
        "learning_rate": float, "m1_train_pairs": int, "mixed_train_pairs": int,
        "mixed_validation_pairs": int, "heldout_pairs": int,
        "expected_gt_sha256": str, "max_seq_length": int,
        "system_text": str, "user_prefix": str, "user_separator": str,
        "user_suffix": str, "probe_code_a": str, "probe_code_b": str,
        "answer_yes": str, "answer_no": str, "enable_thinking": bool_arg,
        "tokenizer_local_only": bool_arg, "trust_remote_code": bool_arg,
        "log_dir": Path, "log_prefix": str, "log_timestamp_format": str,
        "log_tee": bool_arg, "log_stderr": bool_arg,
    }
    for name, caster in fields.items():
        parser.add_argument("--" + name.replace("_", "-"), type=caster, required=True)
    parser.add_argument("--run-status", choices=("PASS", "FAIL"))
    parser.add_argument("--exit-code", type=int)
    a = parser.parse_args()
    ensure(a.mode in ("--audit", "--tokenizer", "--weights", "--lora-probe"), "INVALID_MODE")
    ensure(a.stage in ("cache", "gpu", "environment", "tokenizer", "weights", "lora_probe", "finalize"), "INVALID_STAGE")
    if a.stage == "finalize":
        ensure(a.run_status is not None and a.exit_code is not None, "MISSING_FINAL_STATUS")
    else:
        ensure(a.run_status is None and a.exit_code is None, "UNEXPECTED_FINAL_STATUS")
    return a


def validate_common(a: argparse.Namespace) -> None:
    ensure(a.experiment == "i-01a" and re.fullmatch(r"v[0-9]+", a.version) is not None,
           "INVALID_EXPERIMENT_VERSION")
    ensure(re.fullmatch(r"[0-9a-fA-F]{40}", a.model_revision) is not None, "INVALID_REVISION")
    ensure(a.revision_policy == "shell-pinned", "I01A_V3_REQUIRES_SHELL_PINNED_REVISION")
    ensure(a.tokenizer_policy == "pinned-jinja-autotokenizer-text-only" and a.tokenizer_use_fast,
           "INVALID_TEXT_TOKENIZER_POLICY")
    ensure(re.fullmatch(r"[0-9a-f]{64}", a.chat_template_sha256) is not None,
           "CHAT_TEMPLATE_SHA256_MISSING_OR_INVALID")
    ensure(a.base_precision == "bf16" and a.quantization == "none", "REQUIRES_PLAIN_BF16_LORA_NO_QLORA")
    ensure(a.tokenizer_local_only and not a.trust_remote_code and not a.enable_thinking,
           "REQUIRES_LOCAL_TOKENIZER_NO_REMOTE_CODE_THINKING_DISABLED")
    ensure(a.lora_rank > 0 and a.lora_alpha > 0 and 0 <= a.lora_dropout < 1 and
           a.lora_bias == "none" and a.lora_task_type == "CAUSAL_LM" and
           a.low_cpu_mem_usage and a.min_target_modules > 0, "INVALID_LORA_CONFIG")
    ensure(a.expected_gpus > 0 and a.min_compute_major > 0 and a.min_weight_gib > 0,
           "INVALID_GPU_OR_WEIGHT_REQUIREMENT")
    ensure(a.min_gpu_free_gib >= 0 and a.max_seq_length > 0 and a.learning_rate > 0,
           "INVALID_MEMORY_OR_TRAINING_CONFIG")
    ensure(a.m1_train_pairs > 0 and a.mixed_train_pairs > a.m1_train_pairs and
           a.mixed_validation_pairs > 0 and a.heldout_pairs > 0, "INVALID_DATASET_BUDGET")
    ensure(re.fullmatch(r"[0-9a-f]{64}", a.expected_gt_sha256) is not None,
           "INVALID_GROUND_TRUTH_SHA256")
    ensure(bool(a.model_type.strip()) and bool(a.model_architecture.strip()) and
           re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", a.model_id) is not None,
           "INVALID_MODEL_ID_ARCHITECTURE_PARAMETERS")
    ensure(a.snapshot.is_dir() and a.cache_root.is_dir() and a.run_dir.is_dir(), "REQUIRED_PATH_MISSING")
    ensure(a.snapshot.resolve().is_relative_to(a.cache_root.resolve()), "SNAPSHOT_ESCAPES_CACHE_ROOT")
    ensure(a.snapshot.name == a.model_revision, "REVISION_SNAPSHOT_MISMATCH")
    ensure(len(a.device_ids.split(",")) == a.expected_gpus and
           all(part.isdigit() for part in a.device_ids.split(",")) and
           len(set(a.device_ids.split(","))) == a.expected_gpus, "GPU_IDS_INVALID")
    ensure("language_model" in a.lora_target_regex and "q_proj" in a.lora_target_regex,
           "LORA_REGEX_MUST_SCOPE_LANGUAGE_MODEL")
    re.compile(a.lora_target_regex)
    if a.stage == "lora_probe":
        ensure(a.revision_policy == "shell-pinned", "LORA_PROBE_REQUIRES_SHELL_PINNED_REVISION")


def file_in_cache(path: Path, root: Path, label: str) -> None:
    ensure(path.is_file(), f"MISSING_{label}:{path}")
    ensure(path.resolve().is_relative_to(root.resolve()), f"UNSAFE_{label}_SYMLINK:{path}")
    ensure(path.stat().st_size > 0, f"EMPTY_{label}:{path}")


def cache_stage(a: argparse.Namespace) -> dict:
    ensure(a.snapshot.parent.name == "snapshots" and a.snapshot.parent.parent.resolve() == a.cache_root.resolve(),
           "CACHE_SNAPSHOT_DIRECTORY_LAYOUT_INVALID")
    ensure(a.cache_root.name == "models--" + a.model_id.replace("/", "--"),
           "MODEL_CACHE_ID_MISMATCH")
    config_path = a.snapshot / "config.json"
    file_in_cache(config_path, a.cache_root, "CONFIG")
    config = json_read(config_path)
    ensure(config.get("model_type") == a.model_type, "MODEL_TYPE_MISMATCH")
    ensure(a.model_architecture in config.get("architectures", []), "MODEL_ARCHITECTURE_MISMATCH")
    ensure(isinstance(config.get("text_config"), dict) and isinstance(config.get("vision_config"), dict),
           "EXPECTED_GEMMA4_MULTIMODAL_CONFIG_MISSING")
    ensure("quantization_config" not in config, "QUANTIZED_CONFIG_NOT_ALLOWED")
    ensure(config.get("dtype", config.get("torch_dtype")) in (None, "bfloat16"),
           "MODEL_CONFIG_DTYPE_NOT_BF16")
    index_path = a.snapshot / "model.safetensors.index.json"
    if a.require_shard_index:
        file_in_cache(index_path, a.cache_root, "SHARD_INDEX")
        index = json_read(index_path)
        mapping = index.get("weight_map")
        ensure(isinstance(mapping, dict) and bool(mapping), "EMPTY_WEIGHT_MAP")
        shard_names = sorted(set(mapping.values()))
    else:
        ensure((a.snapshot / "model.safetensors").is_file(), "SINGLE_WEIGHT_FILE_MISSING")
        shard_names = ["model.safetensors"]
    ensure(all(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", name)
               for name in shard_names), "INVALID_SHARD_NAME")
    total_bytes = 0
    for name in shard_names:
        path = a.snapshot / name
        file_in_cache(path, a.cache_root, "WEIGHT_SHARD")
        total_bytes += path.stat().st_size
    min_bytes = int(a.min_weight_gib * (1024 ** 3))
    ensure(total_bytes >= min_bytes,
           f"WEIGHTS_TOTAL_BELOW_MINIMUM:{total_bytes}<{min_bytes}")
    # Processor metadata stays part of the pinned cache audit. However, text-only
    # tokenizer checks never instantiate multimodal image/video processors.
    file_in_cache(a.snapshot / "processor_config.json", a.cache_root, "PROCESSOR_CONFIG")
    template_path = a.snapshot / "chat_template.jinja"
    file_in_cache(template_path, a.cache_root, "CHAT_TEMPLATE")
    ensure(sha256(template_path) == a.chat_template_sha256,
           "PINNED_CHAT_TEMPLATE_SHA256_MISMATCH")
    specs = {}
    for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json", "processor_config.json", "chat_template.jinja"):
        path = a.snapshot / name
        if path.is_file():
            file_in_cache(path, a.cache_root, "SPEC")
            specs[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return {
        "model_type": config.get("model_type"), "architecture": a.model_architecture,
        "text_model_type": config["text_config"].get("model_type"),
        "vision_model_type": config["vision_config"].get("model_type"),
        "revision": a.model_revision, "revision_policy": a.revision_policy,
        "tokenizer_policy": a.tokenizer_policy, "tokenizer_use_fast": a.tokenizer_use_fast,
        "chat_template_sha256": a.chat_template_sha256,
        "revision_is_frozen_in_shell": a.revision_policy == "shell-pinned",
        "weight_shards": len(shard_names), "weight_bytes": total_bytes,
        "weight_gib": round(total_bytes / 1024**3, 3),
        "weight_shard_names": shard_names, "spec_file_hashes": specs,
        "full_weight_sha256_verified": False,
        "disclaimer": "Shard presence and size are checked, but full shard digests require --weights."
    }


def gpu_stage(a: argparse.Namespace) -> dict:
    try:
        proc = subprocess.run([
            a.gpu_query_bin,
            "--query-gpu=index,name,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=60)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise PreflightError(f"NVIDIA_SMI_FAILED:{error}") from error
    devices = {}
    for line in proc.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        ensure(len(fields) == 5, "NVIDIA_SMI_UNEXPECTED_FIELDS")
        index, name, total, free, capability = fields
        devices[index] = {"name": name, "memory_total_mib": int(total),
                          "memory_free_mib": int(free), "compute_capability": capability}
    selected = []
    for index in a.device_ids.split(","):
        ensure(index in devices, f"GPU_NOT_FOUND:{index}")
        item = {"physical_index": index, **devices[index]}
        ensure(item["memory_free_mib"] >= a.min_gpu_free_gib * 1024,
               f"GPU_FREE_MEMORY_LOW:{index}")
        ensure(int(item["compute_capability"].split(".")[0]) >= a.min_compute_major,
               f"BF16_GPU_COMPUTE_CAPABILITY_UNSUPPORTED:{index}")
        selected.append(item)
    return {"selected_gpus": selected, "selected_total_vram_gib": round(sum(
        dev["memory_total_mib"] for dev in selected) / 1024, 3),
        "selected_free_vram_gib": round(sum(
            dev["memory_free_mib"] for dev in selected) / 1024, 3),
        "disclaimer": "GPU presence and BF16-capable compute capability do not prove a backward pass fits memory."}


def environment_stage(a: argparse.Namespace) -> dict:
    from packaging.version import Version

    versions = {}
    packages = {
        "torch": a.min_torch,
        "transformers": a.min_transformers,
        "peft": a.min_peft,
        "accelerate": a.min_accelerate,
        "safetensors": a.min_safetensors,
        "Pillow": a.min_pillow,
    }
    for package, required in packages.items():
        try:
            detected = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise PreflightError(f"REQUIRED_PACKAGE_MISSING:{package}") from error
        versions[package] = detected
        ensure(Version(detected) >= Version(required),
               f"PACKAGE_TOO_OLD:{package}:{detected}<{required}")
    try:
        from PIL import Image
    except ImportError as error:
        raise PreflightError("PILLOW_IMPORT_FAILED:install Pillow in deepknowcode-gemma") from error
    import torch
    import transformers
    import peft
    ensure(hasattr(transformers, "AutoTokenizer") and
           hasattr(transformers, "AutoModelForMultimodalLM") and
           hasattr(transformers, "AutoConfig"), "GEMMA4_MULTIMODAL_MODEL_API_MISSING")
    ensure(hasattr(peft, "get_peft_model") and hasattr(peft, "LoraConfig"),
           "PEFT_LORA_API_MISSING")
    ensure(torch.cuda.is_available(), "TORCH_CUDA_UNAVAILABLE")
    ensure(torch.cuda.device_count() == a.expected_gpus,
           f"CUDA_VISIBLE_GPU_COUNT_MISMATCH:{torch.cuda.device_count()}!={a.expected_gpus}")
    for index in range(a.expected_gpus):
        major, _ = torch.cuda.get_device_capability(index)
        ensure(major >= a.min_compute_major,
               f"TORCH_GPU_NOT_BF16_CAPABLE:{index}")
    ensure(torch.cuda.is_bf16_supported(), "CUDA_BF16_NOT_SUPPORTED")
    config = transformers.AutoConfig.from_pretrained(
        str(a.snapshot), local_files_only=True, trust_remote_code=False)
    ensure(config.model_type == a.model_type,
           "TRANSFORMERS_CANNOT_LOAD_GEMMA4_MODEL_TYPE")
    return {"versions": versions, "torch_cuda_runtime": torch.version.cuda,
            "visible_gpus": torch.cuda.device_count(),
            "gemma_config_class": type(config).__name__,
            "pillow_image_module": Image.__name__,
            "supports_multimodal_model_autoclass": True,
            "text_only_tokenizer_policy": a.tokenizer_policy,
            "disclaimer": "Package and config compatibility do not demonstrate LoRA training compatibility."}


def sample_prompt(a: argparse.Namespace) -> str:
    return (a.user_prefix + a.probe_code_a + a.user_separator +
            a.probe_code_b + a.user_suffix)


def text_only_tokenizer(a: argparse.Namespace):
    """Load only text assets. Do not construct Gemma4[Video|Image]Processor.

    The canonical Jinja template is supplied explicitly because the pinned
    tokenizer_config.json may not embed its chat template. This is fail-closed:
    no implicit fallback to another template or token-scoring policy.
    """
    ensure(a.tokenizer_policy == "pinned-jinja-autotokenizer-text-only",
           "TEXT_TOKENIZER_POLICY_MISMATCH")
    template_path = a.snapshot / "chat_template.jinja"
    file_in_cache(template_path, a.cache_root, "CHAT_TEMPLATE")
    digest = sha256(template_path)
    ensure(digest == a.chat_template_sha256,
           f"CHAT_TEMPLATE_SHA256_MISMATCH:{digest}")
    template = template_path.read_text(encoding="utf-8")
    ensure(bool(template.strip()), "EMPTY_PINNED_CHAT_TEMPLATE")
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(a.snapshot), local_files_only=a.tokenizer_local_only,
            trust_remote_code=a.trust_remote_code, use_fast=a.tokenizer_use_fast)
    except (ImportError, OSError, ValueError) as error:
        raise PreflightError(f"GEMMA4_TEXT_TOKENIZER_LOAD_FAILED:{error}") from error
    ensure(callable(getattr(tokenizer, "apply_chat_template", None)),
           "TOKENIZER_CHAT_TEMPLATE_API_UNAVAILABLE")
    return tokenizer, template, digest


def render_text_prompt(a: argparse.Namespace, tokenizer, template: str) -> str:
    messages = [
        {"role": "system", "content": a.system_text},
        {"role": "user", "content": sample_prompt(a)},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages, chat_template=template, tokenize=False,
            add_generation_prompt=True, enable_thinking=a.enable_thinking)
    except Exception as error:
        raise PreflightError(f"GEMMA4_TEXT_CHAT_TEMPLATE_INCOMPATIBLE:{error}") from error
    ensure(isinstance(rendered, str) and len(rendered) > 5,
           "INVALID_RENDERED_CHAT_PROMPT")
    return rendered


def tokenizer_stage(a: argparse.Namespace) -> dict:
    tokenizer, template, digest = text_only_tokenizer(a)
    rendered = render_text_prompt(a, tokenizer, template)
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    ensure(isinstance(prompt_ids, list) and len(prompt_ids) > 0,
           "TOKENIZER_PROMPT_ENCODING_EMPTY")
    ensure(len(prompt_ids) <= a.max_seq_length,
           f"TOKENIZER_PROBE_EXCEEDS_MAX_SEQUENCE_LENGTH:{len(prompt_ids)}")
    labels = {}
    for key, answer in (("YES", a.answer_yes), ("NO", a.answer_no)):
        sequence_ids = tokenizer.encode(rendered + answer, add_special_tokens=False)
        ensure(sequence_ids[:len(prompt_ids)] == prompt_ids and
               len(sequence_ids) == len(prompt_ids) + 1,
               f"CONTEXTUAL_ANSWER_NOT_SINGLE_TOKEN:{key}")
        labels[key] = {"string": answer, "token_id": sequence_ids[-1]}
    ensure(labels["YES"]["token_id"] != labels["NO"]["token_id"],
           "YES_NO_TOKEN_IDS_IDENTICAL")
    return {"processor_class": "NOT_INSTANTIATED_TEXT_ONLY",
            "tokenizer_policy": a.tokenizer_policy,
            "tokenizer_class": type(tokenizer).__name__,
            "pinned_chat_template_sha256": digest,
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "prompt_tokens": len(prompt_ids), "label_token_ids": labels,
            "template_thinking_enabled": a.enable_thinking,
            "probe_only_not_full_training_prompt_freeze": True,
            "disclaimer": "Single-token audit for pinned text-only Jinja/probe; no multimodal processor loaded."}


def weights_stage(a: argparse.Namespace) -> dict:
    cache = json_read(a.run_dir / "cache.json")
    names = cache["details"]["weight_shard_names"]
    verified = []
    for idx, name in enumerate(names, start=1):
        path = a.snapshot / name
        file_in_cache(path, a.cache_root, "WEIGHT_SHARD")
        target = path.resolve()
        blob_name = target.name.lower()
        ensure(re.fullmatch(r"[0-9a-f]{64}", blob_name) is not None,
               f"SHARD_BLOB_SHA256_NOT_AVAILABLE:{name}")
        observed = sha256(path)
        ensure(observed == blob_name, f"WEIGHT_SHA256_MISMATCH:{name}")
        verified.append({"name": name, "sha256": observed, "bytes": path.stat().st_size})
        print(f"I01A_WEIGHT_VERIFIED={idx}/{len(names)} NAME={name} SHA256={observed}", flush=True)
    return {"verified_shards": verified, "full_weight_sha256_verified": True,
            "weight_count": len(verified)}


def lora_probe_stage(a: argparse.Namespace) -> dict:
    ensure((a.run_dir / "tokenizer.json").is_file(), "TOKENIZER_AUDIT_MISSING")
    ensure((a.run_dir / "weights.json").is_file(), "FULL_WEIGHT_AUDIT_MISSING")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForMultimodalLM

    ensure(torch.cuda.is_available() and torch.cuda.device_count() == a.expected_gpus,
           "CUDA_NOT_READY_FOR_LORA_PROBE")
    torch.manual_seed(a.seed)
    torch.cuda.empty_cache()
    print("I01A_LORA_PROBE_BEGIN BF16_BASE=true QLORA=false; no optimizer step", flush=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        str(a.snapshot), dtype=torch.bfloat16, device_map=a.device_map,
        local_files_only=True, trust_remote_code=False,
        low_cpu_mem_usage=a.low_cpu_mem_usage)
    ensure(not getattr(model, "is_loaded_in_4bit", False) and
           not getattr(model, "is_loaded_in_8bit", False), "QUANTIZED_MODEL_LOADED")
    placement = getattr(model, "hf_device_map", None)
    ensure(isinstance(placement, dict) and bool(placement), "MODEL_DEVICE_MAP_NOT_AVAILABLE")
    ensure(all(str(device) != "cpu" and str(device) != "disk" for device in placement.values()),
           "BF16_MODEL_OFFLOADED_TO_CPU_OR_DISK")
    target = re.compile(a.lora_target_regex)
    matched = [name for name, _ in model.named_modules() if target.fullmatch(name)]
    ensure(len(matched) >= a.min_target_modules,
           f"GEMMA_TEXT_LORA_TARGET_MODULES_TOO_FEW:{len(matched)}")
    ensure(all("language_model" in name and "vision_tower" not in name and
               "vision_model" not in name for name in matched),
           "LORA_TARGET_ESCAPES_LANGUAGE_MODEL")
    config = LoraConfig(
        task_type=a.lora_task_type, inference_mode=False, r=a.lora_rank,
        lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        bias=a.lora_bias, target_modules=a.lora_target_regex)
    model = get_peft_model(model, config)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    ensure(bool(trainable), "LORA_NO_TRAINABLE_PARAMETERS")
    ensure(all("lora_" in name for name, _ in trainable),
           "NON_LORA_BASE_PARAMETERS_ARE_TRAINABLE")
    trainable_count = sum(p.numel() for _, p in trainable)
    tokenizer, template, _ = text_only_tokenizer(a)
    prompt = render_text_prompt(a, tokenizer, template)
    ids = tokenizer.encode(prompt + a.answer_yes, add_special_tokens=False)
    ensure(len(ids) > 2 and len(ids) <= a.max_seq_length,
           "LORA_PROBE_INPUT_LENGTH_INVALID")
    embeddings = model.get_input_embeddings()
    ensure(embeddings is not None, "INPUT_EMBEDDING_NOT_AVAILABLE")
    device = embeddings.weight.device
    ensure(device.type == "cuda", "INPUT_EMBEDDING_IS_NOT_ON_CUDA")
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    attention = torch.ones_like(inputs)
    labels = torch.full_like(inputs, -100)
    labels[0, -1] = inputs[0, -1]
    model.train()
    model.zero_grad(set_to_none=True)
    outputs = model(input_ids=inputs, attention_mask=attention,
                    labels=labels, use_cache=False)
    ensure(outputs.loss is not None and torch.isfinite(outputs.loss).item(),
           "LORA_FORWARD_LOSS_INVALID")
    loss_value = float(outputs.loss.detach().cpu())
    outputs.loss.backward()
    grad_params = sum(p.grad is not None and torch.isfinite(p.grad).all().item()
                      for _, p in trainable)
    ensure(grad_params > 0, "LORA_BACKWARD_NO_FINITE_GRADIENT")
    return {"bf16_base_loaded": True, "quantization": "none",
            "actual_target_modules": len(matched),
            "example_target_names": matched[:12],
            "trainable_lora_parameters": trainable_count,
            "trainable_parameters_with_finite_grad": grad_params,
            "loss": loss_value, "input_tokens": len(ids),
            "model_device_map": {str(k): str(v) for k, v in placement.items()},
            "optimizer_steps": 0, "checkpoint_saved": False,
            "disclaimer": "One small LoRA backward probe is not a 2048-token stress/stability test."}


def finalize_stage(a: argparse.Namespace) -> dict:
    required = ["cache", "gpu", "environment"]
    if a.mode in ("--tokenizer", "--lora-probe"):
        required.append("tokenizer")
    if a.mode in ("--weights", "--lora-probe"):
        required.append("weights")
    if a.mode == "--lora-probe":
        required.append("lora_probe")
    items = {}
    for name in required:
        file = a.run_dir / f"{name}.json"
        if file.exists():
            items[name] = json_read(file)
        else:
            items[name] = {"status": "NOT_RUN"}
    achieved = all(items[name].get("status") == "PASS" for name in required)
    ensure((a.run_status == "PASS") == achieved,
           "FINAL_STATUS_MISMATCH")
    label = "FAIL"
    if achieved:
        label = {
            "--audit": "AUDIT_PASS_SHELL_PINNED_NOT_TRAINING",
            "--tokenizer": "TOKENIZER_PASS_SHELL_PINNED_NOT_TRAINING",
            "--weights": "WEIGHT_HASH_PASS_SHELL_PINNED_NOT_TRAINING",
            "--lora-probe": "LORA_BACKWARD_PROBE_PASS_NOT_TRAINING",
        }[a.mode]
    details = {
        "status": label, "shell_status": a.run_status,
        "exit_code": a.exit_code,
        "version": a.version, "experiment": a.experiment,
        "mode": a.mode, "at_chicago": chicago(),
        "revision_is_shell_pinned": a.revision_policy == "shell-pinned",
        "no_model_training_or_optimizer_step": True,
        "not_authorized_to_begin_m1_training": True,
        "next_action": "Verify tokenizer and weight SHA256, then run a separately authorized BF16 LoRA memory probe before i-01b.",
        "stages": {name: items[name].get("status") for name in required},
        "stage_files": {name: str(a.run_dir / f"{name}.json") for name in required},
        "effective_parameters": {key: str(value) if isinstance(value, Path) else value
                                 for key, value in vars(a).items()
                                 if key not in ("run_status", "exit_code")},
    }
    json_new(a.run_dir / "preflight.json", details)
    return {"final_status": label, "preflight_json": str(a.run_dir / "preflight.json")}


def main() -> int:
    a = parse_args()
    start = time.monotonic()
    try:
        validate_common(a)
        stages = {
            "cache": cache_stage,
            "gpu": gpu_stage,
            "environment": environment_stage,
            "tokenizer": tokenizer_stage,
            "weights": weights_stage,
            "lora_probe": lora_probe_stage,
            "finalize": finalize_stage,
        }
        details = stages[a.stage](a)
        if a.stage != "finalize":
            record = {"status": "PASS", "stage": a.stage,
                      "at_chicago": chicago(), "duration_seconds": round(time.monotonic() - start, 3),
                      "details": details}
            json_new(a.run_dir / f"{a.stage}.json", record)
            print(json.dumps({"status": "I01A_V3_STAGE_PASS", "stage": a.stage,
                              "at_chicago": record["at_chicago"],
                              "report": str(a.run_dir / f"{a.stage}.json")}), flush=True)
        else:
            print(json.dumps(details), flush=True)
        return 0
    except Exception as error:
        message = f"{type(error).__name__}:{error}"
        if a.stage != "finalize":
            record = {"status": "FAIL", "stage": a.stage, "reason": message,
                      "at_chicago": chicago(),
                      "duration_seconds": round(time.monotonic() - start, 3)}
            try:
                json_new(a.run_dir / f"{a.stage}.json", record)
            except Exception:
                pass
        print(json.dumps({"status": "I01A_V3_STAGE_FAIL", "stage": a.stage,
                          "error": message, "at_chicago": chicago()}),
              file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
