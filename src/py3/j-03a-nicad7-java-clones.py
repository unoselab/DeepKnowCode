#!/usr/bin/env python3
"""J-03a v3: independent subject-level Java NiCad 7 detection and verified resume.

All experiment parameters are required CLI arguments supplied by the Shell wrapper.
No prepared training datasets are opened for writing.
"""
import argparse
import csv
import functools
import fnmatch
import math
import shutil
import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*__[a-z0-9][a-z0-9_.-]*$")
JAVA_SUFFIX = ".java"
CLONE_PAIR_FIELDS = [
    "pair_id", "fragment_a_id", "repository_a", "file_a", "start_line_a", "end_line_a", "commit_a",
    "source_sha256_a", "fragment_b_id", "repository_b", "file_b", "start_line_b", "end_line_b", "commit_b",
    "source_sha256_b", "same_repository", "nicad_similarity_raw", "nicad_nlines", "detector", "config", "xml_sha256",
]
FRAGMENT_FIELDS = ["fragment_id", "repository", "file_path", "start_line", "end_line", "commit", "source_sha256", "language"]
INVENTORY_FIELDS = ["repository", "origin_url", "head_commit", "shallow", "tracked_java_files", "status", "absolute_path"]
FILTER_FIELDS = ["repository", "relative_path", "status", "reason", "bytes", "source_sha256"]


def now():
    return datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="seconds")


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def write_json_new(path, obj):
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def write_csv_new(path, fields, rows):
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def git(repo, *args):
    cp = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if cp.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {cp.stderr.strip()}")
    return cp.stdout.strip()


def check_install(nicad_home, txl_home, granularity, language, config):
    lib = nicad_home / "lib" / "nicad"
    required = [nicad_home / "bin" / "nicad", lib / "scripts" / "nicadpair.sh",
                lib / "tools" / "clonepairs.x", lib / "txl" / f"{language}-extract-{granularity}.x",
                lib / "config" / f"{config}.cfg", txl_home / "bin" / "txl"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError("NiCad/OpenTxl installation incomplete: " + ", ".join(missing))
    return lib


def tracked_java_files(repo):
    """Read tracked Java paths directly from Git, without touching .git or untracked files."""
    out = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z", "--", "*.java"])
    for raw in out.split(b"\0"):
        if not raw:
            continue
        rel = Path(os.fsdecode(raw))
        if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.suffix != JAVA_SUFFIX:
            raise RuntimeError(f"Unexpected Git-tracked Java path in {repo}: {rel!r}")
        yield rel


def exclusion_reason(rel, args):
    """Conservative Java analogue of C-01's path/file Test exclusion policy."""
    parts = tuple(p.casefold() for p in rel.parts[:-1])
    for item in parts:
        if item in args.skip_dir:
            return "skip_dir:" + item
    filename = rel.name.casefold()
    for pattern in args.skip_file_glob:
        if fnmatch.fnmatchcase(filename, pattern.casefold()):
            return "skip_file_glob:" + pattern
    return ""


def select_repositories(args, records):
    if args.mode == "--inspect":
        return records
    selected = [r for r in records if r["repository"] == args.subject]
    if len(selected) != 1:
        raise RuntimeError(f"Subject not in inventory: {args.subject}")
    return selected


def candidate_records(args, inventory_records):
    for record in select_repositories(args, inventory_records):
        repo = args.repo_root / record["repository"]
        for rel in tracked_java_files(repo):
            src = repo / rel
            reason = exclusion_reason(rel, args)
            if not reason and src.is_symlink():
                reason = "symlink"
            if not reason and not src.is_file():
                raise RuntimeError(f"Git tracked Java file missing: {src}")
            yield record["repository"], rel, src, reason


def selection_summary(args, records, mode, audit_path=None):
    count = Counter()
    skipped = Counter()
    bytes_total = 0
    if audit_path:
        stream = audit_path.open("x", encoding="utf-8", newline="")
        writer = csv.DictWriter(stream, fieldnames=FILTER_FIELDS)
        writer.writeheader()
    else:
        stream = None
        writer = None
    try:
        for repository, rel, src, reason in candidate_records(args, records):
            count["tracked_java_files"] += 1
            if reason:
                count["excluded_java_files"] += 1
                skipped[reason] += 1
                size, sha = "", ""
            else:
                count["included_java_files"] += 1
                size = src.stat().st_size
                bytes_total += size
                sha = ""
            if writer is not None:
                writer.writerow({"repository": repository, "relative_path": rel.as_posix(),
                                 "status": "EXCLUDED" if reason else "INCLUDED", "reason": reason,
                                 "bytes": size, "source_sha256": sha})
    finally:
        if stream is not None:
            stream.close()
    return {
        "status": mode,
        "stage_root": str(args.stage_root),
        "repository_scope": [r["repository"] for r in select_repositories(args, records)],
        "excluded_directories": args.skip_dir,
        "excluded_file_globs": args.skip_file_glob,
        "in_file_test_method_policy": args.embedded_test_policy,
        "line_number_policy": "Byte-identical full file staging; no rewriting or line masking",
        "filter_limitations": ["Path/name heuristics may include nonstandard tests or omit production files",
                               "No Java AST-based masking inside retained production files"],
        "counts": dict(count), "excluded_by_reason": dict(sorted(skipped.items())),
        "estimated_staged_source_bytes": bytes_total,
    }


def stage(args):
    manifest_path = args.output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = args.repo_root.resolve(strict=True)
    if manifest.get("subject") != args.subject or manifest["repository_root"] != str(root) or manifest["source_filter_policy"]["excluded_directories"] != args.skip_dir or manifest["source_filter_policy"]["excluded_file_globs"] != args.skip_file_glob:
        raise RuntimeError("Stage configuration differs from original inventory")
    records = manifest["git_repositories"]
    for r in select_repositories(args, records):
        repo = root / r["repository"]
        if git(repo, "rev-parse", "HEAD") != r["head_commit"]:
            raise RuntimeError(f"Git HEAD changed: {repo}")
        if args.require_clean and git(repo, "status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError(f"Git checkout not clean: {repo}")
    summary = selection_summary(args, records, "STAGING_PLANNED")
    expected_preview = manifest["source_filter_preview"]["counts"]
    if summary["counts"] != expected_preview:
        raise RuntimeError(f"Source selection changed since inspection: {expected_preview} != {summary['counts']}")
    if not summary["counts"].get("included_java_files"):
        raise RuntimeError("No production Java files remain after filtering")
    needed = math.ceil(summary["estimated_staged_source_bytes"] * args.stage_disk_multiplier) + args.stage_reserve_bytes
    free = shutil.disk_usage(args.output).free
    print(f"stage_preflight included={summary['counts']['included_java_files']} excluded={summary['counts'].get('excluded_java_files', 0)} "
          f"estimated_bytes={summary['estimated_staged_source_bytes']} free_bytes={free} required_free_bytes={needed}", flush=True)
    if free < needed:
        raise RuntimeError(f"Insufficient disk for copy-based staging: free={free}, required={needed}")
    if args.stage_root.exists() or args.stage_root.is_symlink():
        raise FileExistsError(f"Stage root already exists; refusing overwrite: {args.stage_root}")
    args.stage_root.mkdir(mode=0o755, parents=False, exist_ok=False)
    out_audit = args.output / "source_filter_inventory.csv"
    count, skipped, bytes_written = Counter(), Counter(), 0
    with out_audit.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FILTER_FIELDS)
        writer.writeheader()
        for repository, rel, src, reason in candidate_records(args, records):
            count["tracked_java_files"] += 1
            size, sha = "", ""
            if reason:
                count["excluded_java_files"] += 1
                skipped[reason] += 1
            else:
                dest = args.stage_root / repository / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() or dest.is_symlink():
                    raise FileExistsError(f"Duplicate/colliding stage file: {dest}")
                # Copy file bytes, never hardlink or symlink into original Git checkout.
                with src.open("rb") as inf, dest.open("xb") as outf:
                    h = hashlib.sha256()
                    size = 0
                    for block in iter(lambda: inf.read(1024 * 1024), b""):
                        outf.write(block)
                        h.update(block)
                        size += len(block)
                sha = h.hexdigest()
                if size != src.stat().st_size or sha != digest_file(src):
                    raise RuntimeError(f"Source changed while staging: {src}")
                count["included_java_files"] += 1
                bytes_written += size
                if count["included_java_files"] % args.stage_progress_every == 0:
                    print(f"[{now()}] STAGING copied={count['included_java_files']} bytes={bytes_written}", flush=True)
            writer.writerow({"repository": repository, "relative_path": rel.as_posix(),
                             "status": "EXCLUDED" if reason else "INCLUDED", "reason": reason,
                             "bytes": size, "source_sha256": sha})
    if dict(count) != summary["counts"] or bytes_written != summary["estimated_staged_source_bytes"]:
        raise RuntimeError("Stage totals differ from preflight; source changed")
    summary.update({"status": "STAGED", "counts": dict(count), "excluded_by_reason": dict(sorted(skipped.items())),
                    "actual_staged_bytes": bytes_written, "source_filter_inventory_sha256": digest_file(out_audit)})
    write_json_new(args.output / "source_filter_summary.json", summary)
    print(f"stage=PASS included={count['included_java_files']} excluded={count['excluded_java_files']} "
          f"bytes={bytes_written} source={args.stage_root}", flush=True)


def inventory(args):
    root = args.repo_root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError("Repository root is not a directory")
    entries = sorted(p for p in root.iterdir() if p.is_dir())
    if len(entries) != args.expected_repos:
        raise RuntimeError(f"Expected {args.expected_repos} Git directories, found {len(entries)}")
    records = []
    for repo in entries:
        if repo.is_symlink() or not REPO_PATTERN.fullmatch(repo.name):
            raise RuntimeError(f"Unexpected noncanonical repository or symlink: {repo}")
        if not (repo / ".git").is_dir():
            raise RuntimeError(f"Missing .git directory: {repo}")
        head = git(repo, "rev-parse", "HEAD")
        url = git(repo, "remote", "get-url", "origin")
        if "github.com" not in url:
            raise RuntimeError(f"Non-GitHub origin for {repo.name}: {url}")
        status = git(repo, "status", "--porcelain", "--untracked-files=all")
        if args.require_clean and status:
            raise RuntimeError(f"Repository has changes/untracked files (first 300 chars): {repo}: {status[:300]}")
        shallow = git(repo, "rev-parse", "--is-shallow-repository")
        java_count = len([p for p in subprocess.check_output(
            ["git", "-C", str(repo), "ls-files", "-z", "--", "*.java"]).split(b"\0") if p])
        records.append({"repository": repo.name, "origin_url": url, "head_commit": head,
                        "shallow": shallow, "tracked_java_files": java_count,
                        "status": "CLEAN" if not status else "MODIFIED", "absolute_path": str(repo)})
    names = {r["repository"] for r in records}
    if args.smoke_repository not in names:
        raise RuntimeError(f"Smoke repository not in inventory: {args.smoke_repository}")
    source_root = (root if args.mode == "--inspect" else args.stage_root / args.subject)
    lib = check_install(args.nicad_home, args.txl_home, args.granularity, args.language, args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv_new(args.output / "repository_inventory.csv", INVENTORY_FIELDS, records)
    preview = selection_summary(args, records, "INSPECT_PREVIEW")
    if not preview["counts"].get("included_java_files"):
        raise RuntimeError("Source selection leaves no Java files")
    write_json_new(args.output / "source_filter_preview.json", preview)
    meta = {
        "experiment": args.experiment, "version": args.version, "mode": args.mode, "subject": args.subject,
        "timestamp_chicago": now(), "repository_root": str(root), "selected_source_root": str(source_root),
        "nicad_home": str(args.nicad_home.resolve()), "nicad_library": str(lib.resolve()),
        "opentxl_home": str(args.txl_home.resolve()), "config": args.config,
        "granularity": args.granularity, "language": args.language,
        "expected_repos": args.expected_repos, "actual_repos": len(records),
        "smoke_repository": args.smoke_repository, "subject": args.subject, "require_clean": args.require_clean,
        "strict_parse_failures": args.strict_parse_failures,
        "source_filter_policy": {"excluded_directories": args.skip_dir,
                                 "excluded_file_globs": args.skip_file_glob,
                                 "embedded_test_policy": args.embedded_test_policy,
                                 "staging_copy_method": "byte_identical_copy_no_links",
                                 "stage_disk_multiplier": args.stage_disk_multiplier,
                                 "stage_reserve_bytes": args.stage_reserve_bytes,
                                 "stage_progress_every": args.stage_progress_every},
        "source_filter_preview": preview,
        "source_filter_preview_sha256": digest_file(args.output / "source_filter_preview.json"),
        "repository_inventory_sha256": digest_file(args.output / "repository_inventory.csv"),
        "git_repositories": records,
        "source_policy": "Active checked-out Git HEAD, no history-based scan",
        "prepared_dataset_policy": "READ_ONLY_NO_MODIFICATIONS",
        "run_status": "INSPECT_ONLY" if args.mode == "--inspect" else "INVENTORY_READY",
    }
    write_json_new(args.output / "run_manifest.json", meta)
    print(f"inventory=PASS repositories={len(records)} source={source_root} java_files={sum(r['tracked_java_files'] for r in records)}", flush=True)
    print(f"shallow_repositories={sum(r['shallow'] == 'true' for r in records)}", flush=True)
    print(f"selected_source_root={source_root}", flush=True)
    print(f"filter_preview included={preview['counts']['included_java_files']} "
          f"excluded={preview['counts'].get('excluded_java_files', 0)} embedded_test_policy={args.embedded_test_policy}", flush=True)


@functools.lru_cache(maxsize=96)
def cached_source_lines(path):
    return Path(path).read_bytes().splitlines(keepends=True)


def resolve_source(raw, work_dir, stage_root, repo_root):
    raw = (raw or "").strip()
    if not raw or "\x00" in raw:
        raise ValueError(f"Empty/invalid NiCad source path: {raw!r}")
    p = Path(raw)
    candidates = [p] if p.is_absolute() else [work_dir / p, stage_root / p]
    for candidate in candidates:
        try:
            staged = candidate.resolve(strict=True)
            rel = staged.relative_to(stage_root)
            if staged.is_file() and len(rel.parts) >= 2 and rel.suffix == JAVA_SUFFIX:
                original = repo_root / rel
                if not original.is_file() or original.is_symlink():
                    raise RuntimeError(f"Original source missing or unsafe: {original}")
                return staged, original, rel
        except (FileNotFoundError, ValueError, OSError):
            continue
    raise ValueError(f"NiCad XML source not present in filtered staging: {raw!r}")


@functools.lru_cache(maxsize=128)
def verify_copy(staged, original):
    if digest_file(Path(staged)) != digest_file(Path(original)):
        raise RuntimeError(f"Staged source differs from original checkout: {original}")



def parse(args):
    work = args.work_dir.resolve(strict=True)
    repo_root = args.repo_root.resolve(strict=True)
    manifest = json.loads((args.output / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("subject") != args.subject or manifest["repository_root"] != str(repo_root) or manifest["config"] != args.config:
        raise RuntimeError("Input manifest/root/config mismatch")
    stage_root = args.stage_root.resolve(strict=True)
    summary_path = args.output / "source_filter_summary.json"
    filter_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if filter_summary.get("status") != "STAGED" or filter_summary.get("stage_root") != str(stage_root):
        raise RuntimeError("Missing/invalid verified source filter summary")
    if filter_summary["counts"] != manifest["source_filter_preview"]["counts"]:
        raise RuntimeError("Staging filter counts differ from inventory")
    if digest_file(args.output / "source_filter_inventory.csv") != filter_summary["source_filter_inventory_sha256"]:
        raise RuntimeError("Source filter audit hash mismatch")
    inv = {r["repository"]: r for r in manifest["git_repositories"]}
    for name, info in inv.items():
        if name != args.subject:
            continue
        repo = repo_root / name
        if git(repo, "rev-parse", "HEAD") != info["head_commit"]:
            raise RuntimeError(f"Git HEAD changed since inventory: {name}")
        if args.require_clean and git(repo, "status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError(f"Git working tree changed since inventory: {name}")
    xml_candidates = [p for p in (work / "nicadclones").rglob("*.xml")
                      if p.name.endswith(f"-clones-{args.nicad_threshold}.xml") and "-classes" not in p.name]
    if args.config not in ("default", "default-report"):
        # The NiCad configuration can change the suffix/threshold; identify unambiguously.
        xml_candidates = [p for p in (work / "nicadclones").rglob("*.xml") if "-clones-" in p.name
                          and "-classes" not in p.name and not p.name.endswith("-withsource.xml")]
    if len(xml_candidates) != 1:
        raise RuntimeError(f"Expected exactly one pair XML; found {len(xml_candidates)}: {xml_candidates[:10]}")
    xml = xml_candidates[0]
    xml_digest = digest_file(xml)
    pair_rows, fragments = [], {}
    seen_pairs = set()
    duplicates = 0
    cross_repo = 0
    pair_nodes = 0

    # Streaming XML parse: clear each clone node to avoid retaining all source code in memory.
    # Only source attributes, not XML source bodies, are used.
    for event, elem in ET.iterparse(xml, events=("end",)):
        if elem.tag != "clone":
            continue
        pair_nodes += 1
        sources = elem.findall("source")
        if len(sources) != 2:
            raise RuntimeError(f"Clone #{pair_nodes} had {len(sources)} sources, expected 2")
        sides = []
        for src in sources:
            staged, path, rel = resolve_source(src.get("file"), work, stage_root, repo_root)
            if exclusion_reason(Path(*rel.parts[1:]), args):
                raise RuntimeError(f"Excluded Test code appeared in NiCad XML: {rel}")
            verify_copy(str(staged), str(path))
            name = rel.parts[0]
            if name not in inv:
                raise RuntimeError(f"NiCad source repository not in inventory: {name}")
            if name != args.subject:
                raise RuntimeError(f"Unexpected repo in smoke run: {name}")
            try:
                start, end = int(src.attrib["startline"]), int(src.attrib["endline"])
            except (KeyError, ValueError) as exc:
                raise RuntimeError(f"Missing/invalid NiCad source coordinates #{pair_nodes}") from exc
            lines = cached_source_lines(str(path))
            if start < 1 or end < start or end > len(lines):
                raise RuntimeError(f"Out-of-range NiCad coordinates: {rel}:{start}-{end} (total {len(lines)})")
            source_hash = digest_bytes(b"".join(lines[start - 1:end]))
            frag = {"repository": name, "file_path": str(Path(*rel.parts[1:])),
                    "start_line": start, "end_line": end, "commit": inv[name]["head_commit"],
                    "source_sha256": source_hash, "language": args.language}
            frag_id = "j03af_" + digest_bytes(json.dumps(frag, sort_keys=True).encode("utf-8"))[:24]
            frag["fragment_id"] = frag_id
            fragments[frag_id] = frag
            sides.append(frag)
        sides.sort(key=lambda f: f["fragment_id"])
        aa, bb = sides
        if aa["fragment_id"] == bb["fragment_id"]:
            raise RuntimeError(f"Self clone fragment #{pair_nodes}")
        key = (aa["fragment_id"], bb["fragment_id"])
        if key in seen_pairs:
            duplicates += 1
            elem.clear()
            continue
        seen_pairs.add(key)
        same = aa["repository"] == bb["repository"]
        cross_repo += not same
        pair_id = "j03ap_" + digest_bytes(("\0".join(key) + "\0" + args.config).encode("utf-8"))[:24]
        pair_rows.append({
            "pair_id": pair_id, "fragment_a_id": aa["fragment_id"], "repository_a": aa["repository"],
            "file_a": aa["file_path"], "start_line_a": aa["start_line"], "end_line_a": aa["end_line"],
            "commit_a": aa["commit"], "source_sha256_a": aa["source_sha256"],
            "fragment_b_id": bb["fragment_id"], "repository_b": bb["repository"],
            "file_b": bb["file_path"], "start_line_b": bb["start_line"], "end_line_b": bb["end_line"],
            "commit_b": bb["commit"], "source_sha256_b": bb["source_sha256"],
            "same_repository": str(same).lower(), "nicad_similarity_raw": elem.attrib.get("similarity", ""),
            "nicad_nlines": elem.attrib.get("nlines", ""), "detector": "NiCad7/functions/java",
            "config": args.config, "xml_sha256": xml_digest,
        })
        elem.clear()

    logs = sorted((work / "nicadclones").rglob("*_functions-clones-*.log"))
    issues = Counter()
    for log in logs:
        s = log.read_text(encoding="utf-8", errors="replace")
        issues["TXL0191E"] += s.count("TXL0191E")
        issues["TXL0192E"] += s.count("TXL0192E")
    parse_failures = sum(issues.values())
    if parse_failures and args.strict_parse_failures:
        raise RuntimeError(f"NiCad reported {parse_failures} TXL parse failures; refusing export")
    repository_counts = {name: {"repository": name, "pair_endpoints": 0,
                                "within_repo_pairs": 0, "cross_repo_pair_endpoints": 0}
                         for name in sorted(inv)}
    for pair in pair_rows:
        a, b = pair["repository_a"], pair["repository_b"]
        repository_counts[a]["pair_endpoints"] += 1
        repository_counts[b]["pair_endpoints"] += 1
        if a == b:
            repository_counts[a]["within_repo_pairs"] += 1
        else:
            repository_counts[a]["cross_repo_pair_endpoints"] += 1
            repository_counts[b]["cross_repo_pair_endpoints"] += 1
    # A within-repository pair has two endpoints, both counted above.
    pair_rows.sort(key=lambda x: x["pair_id"])
    write_csv_new(args.output / "repository_clone_counts.csv",
                  ["repository", "pair_endpoints", "within_repo_pairs", "cross_repo_pair_endpoints"],
                  [repository_counts[name] for name in sorted(repository_counts)])
    write_csv_new(args.output / "clone_pairs.csv", CLONE_PAIR_FIELDS, pair_rows)
    write_csv_new(args.output / "clone_fragments.csv", FRAGMENT_FIELDS,
                  sorted(fragments.values(), key=lambda x: x["fragment_id"]))
    summary = {
        "experiment": args.experiment, "version": args.version, "mode": args.mode, "subject": args.subject,
        "timestamp_chicago": now(), "status": "PARSE_WARNINGS_REVIEW_REQUIRED" if parse_failures else "NICAD_XML_PARSED",
        "nicad_pair_xml": str(xml), "nicad_pair_xml_sha256": xml_digest,
        "input_repository_count": len(inv), "submitted_repository_count": 1, "subject": args.subject,
        "repositories_appearing_in_clone_pairs": len({x for pair in pair_rows for x in (pair["repository_a"], pair["repository_b"])}),
        "xml_clone_nodes": pair_nodes, "exported_unique_pairs": len(pair_rows),
        "duplicate_xml_pairs": duplicates, "unique_fragments": len(fragments),
        "cross_repository_pairs": cross_repo, "within_repository_pairs": len(pair_rows) - cross_repo,
        "parse_failures_detected_in_nicad_logs": parse_failures, "parse_failure_codes": dict(issues),
        "source_filter": {"status": filter_summary["status"],
                          "counts": filter_summary["counts"],
                          "excluded_by_reason": filter_summary["excluded_by_reason"],
                          "source_filter_inventory_sha256": filter_summary["source_filter_inventory_sha256"],
                          "in_file_test_method_policy": filter_summary["in_file_test_method_policy"]},
        "source_validation": "All exported locations validated against filtered staging and byte-identical Git HEAD originals; 1-based inclusive line ranges",
        "labels_policy": "NIcad candidates only; no ground-truth adjudication",
        "prepared_dataset_changed": False,
        "pair_csv_sha256": digest_file(args.output / "clone_pairs.csv"),
        "fragment_csv_sha256": digest_file(args.output / "clone_fragments.csv"),
        "repository_clone_counts_sha256": digest_file(args.output / "repository_clone_counts.csv"),
    }
    write_json_new(args.output / "run_summary.json", summary)
    print("J-03a", summary["status"], "pairs=", len(pair_rows), "fragments=", len(fragments),
          "cross_repo=", cross_repo, "TXL_parse_failures=", parse_failures, flush=True)
    print(f"clone_pairs={args.output / 'clone_pairs.csv'}", flush=True)
    print(f"clone_fragments={args.output / 'clone_fragments.csv'}", flush=True)



def verify(args):
    """Verify a successful subject attempt for --resume; never accept partial/edited outputs."""
    o = args.output.resolve(strict=True)
    summary_path = o / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads((o / "run_manifest.json").read_text(encoding="utf-8"))
    filt = json.loads((o / "source_filter_summary.json").read_text(encoding="utf-8"))
    if (summary.get("version") != args.version or summary.get("subject") != args.subject or
        manifest.get("subject") != args.subject or manifest.get("version") != args.version or
        summary.get("prepared_dataset_changed") is not False or
        summary.get("submitted_repository_count") != 1 or summary.get("cross_repository_pairs") != 0 or
        summary.get("status") not in ("NICAD_XML_PARSED", "PARSE_WARNINGS_REVIEW_REQUIRED")):
        raise RuntimeError("Invalid or mismatched subject summary")
    if (manifest["repository_root"] != str(args.repo_root.resolve()) or
        manifest["config"] != args.config or manifest["source_filter_policy"]["excluded_directories"] != args.skip_dir or
        manifest["source_filter_policy"]["excluded_file_globs"] != args.skip_file_glob or
        manifest["source_filter_policy"]["embedded_test_policy"] != args.embedded_test_policy or
        manifest["granularity"] != args.granularity or manifest["language"] != args.language or
        bool(manifest["strict_parse_failures"]) != args.strict_parse_failures):
        raise RuntimeError("Subject resume configuration changed")
    inv = {x["repository"]: x for x in manifest["git_repositories"]}
    if args.subject not in inv:
        raise RuntimeError("Subject absent from inventory")
    if git(args.repo_root / args.subject, "rev-parse", "HEAD") != inv[args.subject]["head_commit"]:
        raise RuntimeError("Subject Git HEAD changed")
    if args.require_clean and git(args.repo_root / args.subject, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Subject working tree modified")
    if filt["status"] != "STAGED" or filt["counts"] != manifest["source_filter_preview"]["counts"]:
        raise RuntimeError("Staging audit/manifest differs")
    checks = [
        ("clone_pairs.csv", "pair_csv_sha256"),
        ("clone_fragments.csv", "fragment_csv_sha256"),
        ("repository_clone_counts.csv", "repository_clone_counts_sha256"),
    ]
    for filename, key in checks:
        if digest_file(o / filename) != summary[key]:
            raise RuntimeError(f"Subject output SHA mismatch: {filename}")
    if digest_file(o / "source_filter_inventory.csv") != filt["source_filter_inventory_sha256"]:
        raise RuntimeError("Filter audit SHA mismatch")
    xml = Path(summary["nicad_pair_xml"])
    if not xml.is_file() or not xml.resolve().is_relative_to(o):
        raise RuntimeError("Missing or external XML")
    if digest_file(xml) != summary["nicad_pair_xml_sha256"]:
        raise RuntimeError("XML SHA mismatch")
    pairs = list(csv.DictReader((o / "clone_pairs.csv").open(newline="", encoding="utf-8")))
    frags = list(csv.DictReader((o / "clone_fragments.csv").open(newline="", encoding="utf-8")))
    if len(pairs) != summary["exported_unique_pairs"] or len(frags) != summary["unique_fragments"]:
        raise RuntimeError("Pair/fragment row counts differ")
    if len({x["pair_id"] for x in pairs}) != len(pairs):
        raise RuntimeError("Duplicate pair ids")
    fids = {x["fragment_id"] for x in frags}
    if len(fids) != len(frags):
        raise RuntimeError("Duplicate fragment ids")
    for row in pairs:
        if row["repository_a"] != args.subject or row["repository_b"] != args.subject:
            raise RuntimeError("Cross-repository or wrong-subject pair")
        if row["fragment_a_id"] not in fids or row["fragment_b_id"] not in fids:
            raise RuntimeError("Missing referenced fragment")
    for row in frags:
        if row["repository"] != args.subject:
            raise RuntimeError("Wrong-subject fragment")
    print(f"VERIFY PASS subject={args.subject} pairs={len(pairs)} fragments={len(frags)} attempt={o}", flush=True)


def aggregate(args):
    """Produce one immutable dated summary, taking only verified most-recent attempts."""
    root = args.run_root.resolve(strict=True)
    base = root / "subjects"
    inv = args.repo_root.resolve(strict=True)
    subjects = sorted(p.name for p in inv.iterdir() if p.is_dir() and (p / ".git").is_dir())
    if len(subjects) != args.expected_repos:
        raise RuntimeError("Repository count changed before summary")
    rows = []
    totals = Counter()
    for subject in subjects:
        attempts = sorted((base / subject / "attempts").glob("*")) if (base / subject / "attempts").exists() else []
        valid, reason = None, "not_started"
        for attempt in reversed(attempts):
            if not attempt.is_dir():
                continue
            old_output, old_subject = args.output, args.subject
            try:
                args.output, args.subject = attempt, subject
                verify(args)
                valid = attempt
                reason = "verified"
                break
            except Exception as exc:
                reason = f"latest_invalid_or_incomplete:{type(exc).__name__}"
            finally:
                args.output, args.subject = old_output, old_subject
        if valid:
            summary = json.loads((valid / "run_summary.json").read_text(encoding="utf-8"))
            pairs = summary["exported_unique_pairs"]
            fragments = summary["unique_fragments"]
            failures = summary["parse_failures_detected_in_nicad_logs"]
            totals.update({"pairs": pairs, "fragments": fragments, "parse_failures": failures, "complete": 1})
            status = summary["status"]
        else:
            pairs, fragments, failures, status = "", "", "", "INCOMPLETE"
        rows.append({"repository": subject, "status": status, "pairs": pairs,
                     "fragments": fragments, "parse_failures": failures,
                     "attempt": str(valid or ""), "detail": reason})
    suffix = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y%m%d-%H%M%S-%f")
    path = root / f"project_summary-{suffix}.csv"
    write_csv_new(path, ["repository", "status", "pairs", "fragments", "parse_failures", "attempt", "detail"], rows)
    result = {"experiment": args.experiment, "version": args.version, "timestamp_chicago": now(),
              "expected_subjects": args.expected_repos, "completed_subjects": totals["complete"],
              "incomplete_subjects": args.expected_repos - totals["complete"],
              "clone_pairs": totals["pairs"], "unique_fragments_subject_sum": totals["fragments"],
              "parse_failures": totals["parse_failures"], "subject_summary_csv": str(path),
              "scope": "NiCad invoked independently once per repository; no cross-repository comparisons",
              "source_policy": "Git HEAD, filtered Java source staging, read-only original repositories"}
    out = root / f"run_summary-{suffix}.json"
    write_json_new(out, result)
    print(f"AGGREGATE complete={totals['complete']}/{args.expected_repos} pairs={totals['pairs']} "
          f"parse_failures={totals['parse_failures']} summary={out}", flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("inventory", "stage", "parse", "verify", "aggregate"))
    for key in ("root", "repo-root", "nicad-home", "txl-home", "output", "experiment", "version", "mode",
                "expected-repos", "smoke-repository", "language", "granularity", "config",
                "require-clean", "strict-parse-failures", "work-dir", "stage-root",
                "stage-disk-multiplier", "stage-reserve-bytes", "stage-progress-every",
                "embedded-test-policy", "nicad-threshold", "subject", "run-root"):
        typ = (int if key in ("expected-repos", "stage-reserve-bytes", "stage-progress-every") else
               float if key == "stage-disk-multiplier" else
               Path if key in ("root", "repo-root", "nicad-home", "txl-home", "output", "work-dir", "stage-root", "run-root") else str)
        p.add_argument("--" + key, type=typ, required=True)
    p.add_argument("--skip-dir", action="append", required=True)
    p.add_argument("--skip-file-glob", action="append", required=True)
    args = p.parse_args()
    args.skip_dir = [x.casefold() for x in args.skip_dir]
    if (len(args.skip_dir) != len(set(args.skip_dir)) or
        not all(re.fullmatch(r"[a-z0-9_.-]+", x) for x in args.skip_dir)):
        p.error("Invalid or duplicate --skip-dir names")
    if not all(x and "/" not in x and "\\" not in x and ".." not in x for x in args.skip_file_glob):
        p.error("Invalid --skip-file-glob")
    if (args.stage_disk_multiplier < 1.0 or not math.isfinite(args.stage_disk_multiplier) or
        args.stage_reserve_bytes < 0 or args.stage_progress_every < 1):
        p.error("Invalid staging disk/progress parameters")
    if args.embedded_test_policy != "retain_audit_only":
        p.error("Unsupported embedded Java test method policy; no AST masking is implemented")
    if not re.fullmatch(r"0\.[0-9]{2}", args.nicad_threshold):
        p.error("Expected NiCad threshold like 0.30")
    if args.expected_repos < 1 or args.mode not in ("--inspect", "--subject"):
        p.error("Invalid expected-repos or mode")
    if args.require_clean not in ("true", "false") or args.strict_parse_failures not in ("true", "false"):
        p.error("Boolean arguments must be 'true' or 'false'")
    args.require_clean = args.require_clean == "true"
    args.strict_parse_failures = args.strict_parse_failures == "true"
    if args.granularity != "functions" or args.language != "java":
        p.error("J-03a v3 supports only functions/java")
    if not REPO_PATTERN.fullmatch(args.subject):
        p.error("Invalid subject name")
    if args.root.resolve() == args.output.resolve() or args.repo_root.resolve() == args.output.resolve():
        p.error("Output cannot overwrite project or input root")
    if args.action == "inventory":
        inventory(args)
    elif args.action == "stage":
        if args.mode == "--inspect":
            p.error("Cannot stage in --inspect mode")
        stage(args)
    elif args.action == "parse":
        if args.mode == "--inspect":
            p.error("Cannot parse without running NiCad")
        parse(args)
    elif args.action == "verify":
        verify(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[J-03a] FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)
