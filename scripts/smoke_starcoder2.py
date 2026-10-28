import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--bigcode--starcoder2-15b-instruct-v0.1"
    / "snapshots"
    / "ffb8dd9776ba9a66d655ecd962e882f3013e9f7c"
)

def main():
    assert MODEL.exists(), f"Model not found: {MODEL}"

    print("Model:", MODEL)
    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))

    start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL),
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda:0")

    model.eval()
    print("Load time:", round(time.perf_counter() - start, 2), "s")

    instruction = """Summarize the following Python function
in one concise English sentence. Output only the summary.

```python
def add(a, b):
    return a + b
```"""

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = instruction

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0, inputs["input_ids"].shape[1]:]
    summary = tokenizer.decode(
        generated, skip_special_tokens=True
    ).strip()

    print("\n=== Generated Summary ===")
    print(summary)
    print("\nPeak GPU Memory (GiB):",
          round(torch.cuda.max_memory_allocated(0) / 1024**3, 2))

if __name__ == "__main__":
    main()
