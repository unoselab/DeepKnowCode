#!/usr/bin/env python3
"""Read-only audit of BCB vs adapted clone-detection prediction files."""

import argparse
from collections import Counter
from pathlib import Path

MODELS = {
    "codebert": (
        "saved_models_bcb/predictions.txt",
        "saved_models_combined_py/predictions.txt",
    ),
    "graphcodebert": (
        "saved_models_bcb/predictions.txt",
        "saved_models_combined_py/predictions.txt",
    ),
}


def load_pairs(path: Path) -> dict[tuple[str, str], int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    pairs = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) != 3 or fields[2] not in {"0", "1"}:
                raise ValueError(f"Invalid pair/label at {path}:{line_no}")
            a, b, label = fields
            key = (a, b)
            if key in pairs:
                raise ValueError(f"Duplicate ordered pair at {path}:{line_no}: {key}")
            pairs[key] = int(label)
    return pairs


def metrics(truth, predictions):
    tp = sum(truth[k] == 1 and predictions[k] == 1 for k in truth)
    fp = sum(truth[k] == 0 and predictions[k] == 1 for k in truth)
    fn = sum(truth[k] == 1 and predictions[k] == 0 for k in truth)
    tn = len(truth) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(truth) if truth else 0.0
    return precision, recall, f1, accuracy


def evaluate(model_name, root, truth):
    before_rel, after_rel = MODELS[model_name]
    before = load_pairs(root / model_name / before_rel)
    after = load_pairs(root / model_name / after_rel)

    print(f"\n=== {model_name.upper()} ===")
    for stage, predictions in (("Before", before), ("After", after)):
        missing = set(truth) - set(predictions)
        extra = set(predictions) - set(truth)
        print(f"{stage}: rows={len(predictions):,}, missing={len(missing):,}, extra={len(extra):,}")
        if missing or extra:
            print("SKIPPED: prediction pairs do not exactly match reference pairs")
            return

    changed = sum(before[key] != after[key] for key in truth)
    print(f"Changed predictions: {changed:,} / {len(truth):,}")
    print("Stage       Precision    Recall        F1   Accuracy")
    for stage, predictions in (("Before", before), ("After ", after)):
        p, r, f1, acc = metrics(truth, predictions)
        print(f"{stage}      {p:9.4f} {r:9.4f} {f1:9.4f} {acc:9.4f}")
    delta = metrics(truth, after)[2] - metrics(truth, before)[2]
    print(f"F1 difference (After - Before): {delta:+.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    truth = load_pairs(root / "dataset/org/test.txt")
    print(f"Reference test pairs: {len(truth):,}")
    print(f"Ground truth label counts: {dict(sorted(Counter(truth.values()).items()))}")
    for model_name in MODELS:
        evaluate(model_name, root, truth)
    print("\nNote: folder names alone do not verify checkpoint training history.")


if __name__ == "__main__":
    main()
