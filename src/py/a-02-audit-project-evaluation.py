#!/usr/bin/env python3
"""Read-only RQ4 audit for RQ3 prediction and RQ4 RCP paper reproduction.

Ground truth: ID.split('_')[1] equality (original evaluator).
RCP: per-pair negative penalties, divided by number of true clones (original evaluator).
Do not infer the paper's selected 20 Python projects from 24 files.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

MODELS = ("codebert", "codegpt", "codet5", "graphcodebert")
PAPER = {"java": {"projects": 48, "positive": 70262, "negative": 70262},
         "python": {"projects": 20, "positive": 7586, "negative": 7586}}


def write_csv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"pair_left_func", "pair_right_func", "clone_predict"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required columns in {path}: {reader.fieldnames}")
        return list(reader)


def gt(key):
    left, right = key
    a, b = left.split("_"), right.split("_")
    return int(a[1] == b[1]) if len(a) > 1 and len(b) > 1 else 0


def parse_label(value):
    if str(value).strip() not in ("0", "1"):
        raise ValueError(f"Invalid clone_predict={value!r}")
    return int(str(value).strip())


def index_rows(rows):
    result = {}
    duplicate = 0
    invalid = 0
    swapped = 0
    for row in rows:
        l, r = row["pair_left_func"].strip(), row["pair_right_func"].strip()
        if not l or not r:
            invalid += 1
            continue
        try:
            parse_label(row["clone_predict"])
        except ValueError:
            invalid += 1
            continue
        key = (l, r)
        if key in result:
            duplicate += 1
        else:
            result[key] = row
        if l > r:
            swapped += 1
    return result, duplicate, invalid, swapped


def rcp(row, truth):
    pred = parse_label(row["clone_predict"])
    penalty = -int(pred != truth)
    if pred == 1:
        penalty -= int(str(row.get("pair_left_in", "")) != str(row.get("pair_right_in", "")))
        penalty -= int(str(row.get("pair_left_out", "")) != str(row.get("pair_right_out", "")))
        penalty -= int(str(row.get("pair_inType_ok", "")).strip().lower() == "false")
        penalty -= int(str(row.get("pair_outType_ok", "")).strip().lower() == "false")
    return penalty


def scores(rows_by_key):
    tp = fp = tn = fn = 0
    total_penalty = 0
    for key, row in rows_by_key.items():
        y, yhat = gt(key), parse_label(row["clone_predict"])
        if y and yhat: tp += 1
        elif y and not yhat: fn += 1
        elif not y and yhat: fp += 1
        else: tn += 1
        total_penalty += rcp(row, y)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2*p*r/(p+r) if p+r else 0.0
    n = len(rows_by_key)
    positives = tp + fn
    return dict(rows=n, positive=positives, negative=fp+tn, tp=tp, fp=fp, tn=tn, fn=fn,
                precision=p, recall=r, f1=f, accuracy=(tp+tn)/n if n else 0.0,
                total_penalty=total_penalty,
                rcp_legacy=total_penalty/positives if positives else 0.0,
                rcp_per_pair=total_penalty/n if n else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rq4-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    root = args.rq4_root.resolve()
    out = args.output_dir.resolve()
    inventory, alignments, metrics, paper = [], [], [], []
    total = Counter()
    for language, sub in (("java", "data-java"), ("python", "data-py")):
        base = root / language / "data" / sub
        gt_dir = base / "ground_truth_test_refactorability"
        before_dir = base / "pred_before_adapt_refactorability"
        after_dir = base / "pred_after_adapt_refactorability"
        files = sorted(gt_dir.glob("*.csv"))
        if not files:
            print(f"WARNING: no ground truth CSVs in {gt_dir}")
        for reference in files:
            project = reference.name.removeprefix("pred_").removesuffix("_with_refactorability.csv")
            try:
                ref_rows = read_csv(reference)
                ref_map, ref_dups, ref_bad, _ = index_rows(ref_rows)
            except Exception as exc:
                inventory.append(dict(language=language, project=project, file=str(reference), status=f"READ_ERROR: {exc}"))
                continue
            ref_scores = scores(ref_map)
            disagreement = sum(parse_label(row["clone_predict"]) != gt(key) for key, row in ref_map.items())
            inventory.append(dict(language=language, project=project, file=str(reference), status="OK",
                                  rows_raw=len(ref_rows), unique_pairs=len(ref_map), duplicates=ref_dups,
                                  invalid_rows=ref_bad, positives=ref_scores["positive"], negatives=ref_scores["negative"],
                                  ref_label_disagreement=disagreement))
            total[(language, "projects")] += 1
            total[(language, "positive")] += ref_scores["positive"]
            total[(language, "negative")] += ref_scores["negative"]
            for model in MODELS:
                for stage, directory in (("before", before_dir), ("after", after_dir)):
                    candidate = directory / model / reference.name
                    record = dict(language=language, project=project, model=model, stage=stage,
                                  gt_path=str(reference), prediction_path=str(candidate))
                    if not candidate.is_file():
                        alignments.append({**record, "status": "MISSING_FILE"})
                        continue
                    try:
                        pred_rows = read_csv(candidate)
                        pred_map, dupl, bad, _ = index_rows(pred_rows)
                        missing_keys = ref_map.keys() - pred_map.keys()
                        extra_keys = pred_map.keys() - ref_map.keys()
                        shared = ref_map.keys() & pred_map.keys()
                        status = "EXACT_IDS" if not missing_keys and not extra_keys and not dupl and not bad and not ref_dups and not ref_bad else "PARTIAL_OR_INVALID"
                        alignments.append({**record, "status": status, "prediction_rows": len(pred_rows),
                                           "prediction_unique": len(pred_map), "duplicates": dupl, "invalid_rows": bad,
                                           "missing_ids": len(missing_keys), "extra_ids": len(extra_keys), "shared_ids": len(shared)})
                        if status != "EXACT_IDS":
                            continue
                        m = scores(pred_map)
                        metrics.append({**record, **m})
                    except Exception as exc:
                        alignments.append({**record, "status": f"ERROR: {exc}"})
        expected = PAPER[language]
        row = dict(language=language, found_projects=total[(language,"projects")], paper_projects=expected["projects"],
                   found_positive=total[(language,"positive")], paper_positive=expected["positive"],
                   found_negative=total[(language,"negative")], paper_negative=expected["negative"])
        row["project_count_match"] = row["found_projects"] == row["paper_projects"]
        row["pair_count_match"] = row["found_positive"] == row["paper_positive"] and row["found_negative"] == row["paper_negative"]
        paper.append(row)

    write_csv(out / "project_inventory.csv", inventory, ["language","project","file","status","rows_raw","unique_pairs","duplicates","invalid_rows","positives","negatives","ref_label_disagreement"])
    write_csv(out / "prediction_alignment.csv", alignments, ["language","project","model","stage","gt_path","prediction_path","status","prediction_rows","prediction_unique","duplicates","invalid_rows","missing_ids","extra_ids","shared_ids"])
    write_csv(out / "before_after_metrics.csv", metrics, ["language","project","model","stage","rows","positive","negative","tp","fp","tn","fn","precision","recall","f1","accuracy","total_penalty","rcp_legacy","rcp_per_pair","prediction_path","gt_path"])
    write_csv(out / "rcp_metrics.csv", metrics, ["language","project","model","stage","rows","positive","total_penalty","rcp_legacy","rcp_per_pair"])
    write_csv(out / "paper_sample_audit.csv", paper, list(paper[0]) if paper else ["language"])
    summary = {"rq4_root": str(root), "paper_reference": PAPER, "inventory": {l: [x for x in inventory if x["language"] == l] for l in ("java", "python")},
               "counts": {"ground_truth_files":len(inventory), "prediction_files_audited":len(alignments),"exact_alignments":sum(a["status"]=="EXACT_IDS" for a in alignments), "metric_rows":len(metrics)},
               "cautions": ["Original evaluator ground truth uses the second underscore-separated token of each function ID.",
                            "Ground-truth CSV clone_predict is NOT used as the reference label; disagreements are recorded.",
                            "Legacy RCP denominator is number of positive ground-truth pairs; RCP per pair is provided for comparison.",
                            "Do not infer which 20 Python projects were in the published paper from 24 CSVs.",
                            "Only EXACT_IDS cases are scored; files with duplicates or missing/extra IDs are flagged."]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("RQ4 audit complete", summary["counts"])
    for r in paper:
        print(r)
    print("Output:", out)

if __name__ == "__main__":
    main()
