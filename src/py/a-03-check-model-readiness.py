#!/usr/bin/env python3
"""A-03a: non-downloading readiness audit for three open code base models."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

MODELS = ["bigcode/starcoder2-7b", "bigcode/starcoder2-15b", "codellama/CodeLlama-7b-hf"]
PACKAGES = ["torch", "transformers", "accelerate", "peft", "datasets", "huggingface-hub", "safetensors", "bitsandbytes", "sentencepiece", "scipy", "scikit-learn"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/a-03a")
    ap.add_argument("--offline", action="store_true", help="Do not query the Hub; environment-only audit")
    a=ap.parse_args()
    out=Path(a.output_dir).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    data={"timestamp_utc":datetime.now(timezone.utc).isoformat(), "python":sys.version.split()[0], "platform":platform.platform(), "packages":{}, "gpu":{}, "models":[], "notes":[]}
    for p in PACKAGES:
        try: data["packages"][p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: data["packages"][p]=None
    try:
        import torch
        data["gpu"]={"torch_cuda":torch.version.cuda, "cuda_available":torch.cuda.is_available(), "device_count":torch.cuda.device_count(), "devices":[]}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop=torch.cuda.get_device_properties(i)
                free,total=torch.cuda.mem_get_info(i)
                data["gpu"]["devices"].append({"index":i,"name":prop.name,"total_gib":round(total/2**30,2),"free_gib":round(free/2**30,2),"capability":f"{prop.major}.{prop.minor}"})
    except Exception as e: data["gpu"]={"error":str(e)}
    try:
        p=subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=8)
        data["gpu"]["compute_processes"] = p.stdout.strip().splitlines() if p.returncode==0 else []
    except Exception: pass
    if not a.offline:
        try:
            from huggingface_hub import HfApi
            api=HfApi()
        except Exception as e:
            api=None; data["notes"].append("Hub client unavailable: "+str(e))
    else: api=None
    for model in MODELS:
        record={"model_id":model,"hub_status":"not_checked" if a.offline else "unknown", "model_type":None,"tokenizer_class":None,"gated":None,"sha":None}
        if api:
            try:
                inf=api.model_info(model,timeout=15)
                record.update(hub_status="accessible",gated=inf.gated,sha=inf.sha)
                # metadata-only: model weights and tokenizers are NOT downloaded
                if inf.config:
                    record["model_type"]=inf.config.get("model_type")
                    record["tokenizer_class"]=inf.config.get("tokenizer_class")
            except Exception as e:
                record.update(hub_status="unavailable_or_restricted",error=f"{type(e).__name__}: {str(e)[:350]}")
        data["models"].append(record)
    data["notes"].extend(["A-03a does not download/load model weights or run inference.","StarCoder2 and CodeLlama checkpoints here are base code models, not instruction-tuned.","Hub metadata access does not guarantee successful weight downloads; verify license/gating before running.","Freeze Java/Python test pairs and compare models on the same held-out split; do not choose projects by observed test performance."])
    (out/"readiness.json").write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
    lines=["# A-03a model readiness", "",f"Timestamp (UTC): {data['timestamp_utc']}","", "## GPU", "",json.dumps(data['gpu'],indent=2),"", "## Dependencies", ""]
    lines.extend(f"- {p}: {v or 'MISSING'}" for p,v in data["packages"].items())
    lines.extend(["", "## Model metadata", ""])
    lines.extend(f"- {m['model_id']}: {m['hub_status']} (gated={m['gated']}, sha={m['sha'] or 'n/a'})" for m in data["models"])
    lines.extend(["", "No weights downloaded. This is not an inference or training result.",""])
    (out/"REPORT.md").write_text("\n".join(lines))
    print("A-03a readiness audit complete")
    print("GPUs:",data["gpu"].get("device_count","unknown"))
    for m in data["models"]: print(m["model_id"],m["hub_status"], "gated=",m["gated"])
    print("Output:",out)
if __name__=='__main__': main()
