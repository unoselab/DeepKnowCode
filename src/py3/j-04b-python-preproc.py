#!/usr/bin/env python3
"""J-04b v2: immutable static-analysis comment and standalone-string masking.

The Shell wrapper owns all experiment configuration; every CLI input is required.
Input:  data/repositories/python-repos/{owner}__{repo}
Output: data/repositories/python-repo-preproc/{owner}__{repo}
Do not write to the original Git checkouts or clone-detection archives.
"""
from __future__ import annotations

import argparse
import ast
import codecs
import copy
import csv
import fnmatch
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import tokenize
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


SUBJECT = re.compile(r"^[a-z0-9][a-z0-9_.-]*__[a-z0-9][a-z0-9_.-]*$")
SHA = re.compile(r"^[0-9a-f]{40}$")
FILE_AUDIT_FIELDS = (
    "subject", "file", "status", "reason", "raw_sha256", "preproc_sha256",
    "bytes", "masked_comments", "masked_docstrings", "masked_standalone_strings",
    "pass_insertions", "preserved_special_comments", "preserved_uncertain_strings",
    "fallback", "source_encoding"
)
RANGE_FIELDS = ("subject", "file", "category", "start_line", "end_line", "start_byte", "end_byte", "action")


class TransformError(Exception):
    pass


def chicago() -> str:
    return datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="seconds")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_new(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, indent=2, sort_keys=True, ensure_ascii=False)
        output.write("\n")


def git(repo: Path, *arguments: str) -> str:
    cp = subprocess.run(["git", "-C", str(repo), *arguments], capture_output=True, text=True, check=False)
    if cp.returncode:
        raise RuntimeError(f"Git error in {repo}: {' '.join(arguments)}: {cp.stderr.strip()}")
    return cp.stdout.strip()


def git_tracked_python(repo: Path) -> list[Path]:
    cp = subprocess.run(["git", "-C", str(repo), "ls-files", "-z", "--", "*.py"], capture_output=True, check=False)
    if cp.returncode:
        raise RuntimeError(f"git ls-files failed: {repo}: {cp.stderr.decode(errors='replace')}")
    result = []
    for blob in cp.stdout.split(b"\0"):
        if not blob:
            continue
        path = Path(os.fsdecode(blob))
        if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".py":
            raise RuntimeError(f"Unsafe tracked Python path: {path}")
        result.append(path)
    return sorted(result)


def exclusion_reason(path: Path, source: Path, skip_dirs: list[str], skip_globs: list[str]) -> str:
    for part in path.parts[:-1]:
        if part.casefold() in skip_dirs:
            return f"skip_dir:{part.casefold()}"
    for pattern in skip_globs:
        if fnmatch.fnmatchcase(path.name.casefold(), pattern.casefold()):
            return f"skip_file_glob:{pattern}"
    if source.is_symlink():
        return "symlink"
    if not source.is_file():
        raise RuntimeError(f"Git-tracked file is missing: {source}")
    return ""


def safe_subject(name: str) -> None:
    if not SUBJECT.fullmatch(name):
        raise RuntimeError(f"Invalid repository identity: {name}")


def github_identity(remote: str) -> str:
    match = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?/?$", remote, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError(f"Unrecognized GitHub origin: {remote}")
    return f"{match.group(1).lower()}__{match.group(2).lower()}"


def baseline_records(args: argparse.Namespace) -> tuple[dict, str]:
    if not args.baseline_organizer.is_file() or args.baseline_organizer.is_symlink():
        raise RuntimeError(f"Baseline organizer missing/unsafe: {args.baseline_organizer}")
    data = json.loads(args.baseline_organizer.read_text(encoding="utf-8"))
    if data.get("status") != "COMPLETED" or data.get("selected") != args.expected_subjects:
        raise RuntimeError("Baseline organizer not complete or subject count mismatch")
    projects = data.get("projects")
    if not isinstance(projects, list) or len(projects) != args.expected_subjects:
        raise RuntimeError("Baseline organizer has unexpected number of projects")
    output = {}
    for item in projects:
        ident = f"{item['github_owner'].lower()}__{item['github_repository'].lower()}"
        safe_subject(ident)
        if ident in output or not SHA.fullmatch(item['commit']) or item.get("action") != "COPIED":
            raise RuntimeError(f"Invalid baseline subject record: {ident}")
        if item["github_slug"].casefold() != (item["github_owner"] + "/" + item["github_repository"]).casefold():
            raise RuntimeError(f"Inconsistent baseline GitHub slug: {ident}")
        output[ident] = {"git_head": item["commit"], "github_slug": item["github_slug"],
                         "remote": item["remote"], "c01_run_id": item["c01_run_id"]}
    return output, digest_file(args.baseline_organizer)


def repository_inventory(args: argparse.Namespace) -> tuple[list[dict], str]:
    if not args.repo_root.is_dir() or args.repo_root.is_symlink():
        raise RuntimeError(f"Repository root missing/unsafe: {args.repo_root}")
    baselines, baseline_sha = baseline_records(args)
    folders = sorted(p for p in args.repo_root.iterdir() if p.is_dir() and not p.is_symlink())
    if len(folders) != args.expected_subjects:
        raise RuntimeError(f"Subject inventory mismatch: found={len(folders)} expected={args.expected_subjects}")
    result = []
    for repo in folders:
        name = repo.name
        safe_subject(name)
        if name not in baselines:
            raise RuntimeError(f"Subject not present in C-01 baseline: {name}")
        if not (repo / ".git").exists():
            raise RuntimeError(f"Git checkout missing: {repo}")
        head = git(repo, "rev-parse", "HEAD")
        if not SHA.fullmatch(head):
            raise RuntimeError(f"Invalid Git HEAD: {name}: {head}")
        origin = git(repo, "remote", "get-url", "origin")
        identity = github_identity(origin)
        if identity != name:
            raise RuntimeError(f"Git origin identity mismatch: folder={name} remote={origin}")
        if args.require_clean and git(repo, "status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError(f"Git checkout dirty: {name}; do not modify input")
        baseline = baselines[name]
        if args.require_baseline_commit and head != baseline["git_head"]:
            raise RuntimeError(f"C-01 Git HEAD mismatch: subject={name} head={head} baseline={baseline['git_head']}")
        tracked = git_tracked_python(repo)
        if not tracked:
            raise RuntimeError(f"Repository has no Git-tracked Python sources: {name}")
        result.append({"subject": name, "repo": repo, "git_head": head, "origin": origin,
                       "tracked": tracked, "baseline": baseline})
        print(f"PASS REPOSITORY subject={name} tracked_py={len(tracked)} head={head} baseline_match={head == baseline['git_head']}", flush=True)
    if set(baselines) != {r["subject"] for r in result}:
        raise RuntimeError("C-01 subject set does not equal Python repository set")
    print(f"PASS INVENTORY subjects={len(result)} tracked_py={sum(len(x['tracked']) for x in result)}", flush=True)
    return result, baseline_sha


def _physical_lines(raw: bytes, text: str, encoding: str) -> tuple[list[bytes], list[str], list[int], int, str]:
    byte_lines = raw.splitlines(keepends=True)
    text_lines = text.splitlines(keepends=True)
    if len(byte_lines) != len(text_lines):
        raise TransformError("Encoding changes physical line interpretation")
    if encoding.lower().replace("_", "-") == "utf-8-sig":
        prefix = len(codecs.BOM_UTF8) if raw.startswith(codecs.BOM_UTF8) else 0
        per_line_enc = "utf-8"
    else:
        prefix = 0
        per_line_enc = encoding
    starts = []
    offset = 0
    for row, (b, s) in enumerate(zip(byte_lines, text_lines)):
        starts.append(offset)
        offset += len(b)
        if (codecs.BOM_UTF8 if row == 0 and prefix else b"") + s.encode(per_line_enc) != b:
            raise TransformError("Non-roundtrip source encoding")
    return byte_lines, text_lines, starts, prefix, per_line_enc


def _to_raw_offset(row: int, col: int, *, utf8_column: bool, text_lines: list[str],
                   offsets: list[int], bom_prefix: int, line_encoding: str) -> int:
    if row < 1 or row > len(text_lines):
        raise TransformError("Invalid AST/token line coordinate")
    line = text_lines[row - 1]
    if utf8_column:
        value = line.encode("utf-8")
        if col > len(value):
            raise TransformError("Invalid AST UTF8 column")
        try:
            lead = value[:col].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TransformError("AST column inside UTF8 character") from exc
    else:
        if col > len(line):
            raise TransformError("Invalid tokenize column")
        lead = line[:col]
    return offsets[row - 1] + (bom_prefix if row == 1 else 0) + len(lead.encode(line_encoding))


def _suites(tree: ast.AST):
    """Yield every AST suite that consists of statements, including nested control-flow suites."""
    for owner in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            value = getattr(owner, field, None)
            if isinstance(value, list) and value and all(isinstance(x, ast.stmt) for x in value):
                yield owner, field, value


def _is_triple(token_string: str) -> bool:
    return bool(re.match(r"(?i)^[rubf]*(?:\"\"\"|''')", token_string))


def _protected_comment(token: tokenize.TokenInfo, prefixes: list[str]) -> bool:
    body = token.string[1:].strip().casefold()
    if token.start[0] == 1 and token.start[1] == 0 and body.startswith("!"):
        return True
    if token.start[0] <= 2 and re.search(r"coding\s*[:=]", body):
        return True
    return any(body.startswith(p) for p in prefixes)


def _mask_raw(target: bytearray, begin: int, end: int, pass_token: bool = False) -> None:
    if not (0 <= begin < end <= len(target)):
        raise TransformError("Mask range outside file")
    for i in range(begin, end):
        if target[i] not in (10, 13):
            target[i] = 32
    if pass_token:
        if end - begin < 4 or any(target[i] in (10, 13) for i in range(begin, begin + 4)):
            raise TransformError("Cannot insert pass inside a short literal")
        target[begin:begin + 4] = b"pass"


def transform(raw: bytes, prefixes: list[str], *, mask_head_docstrings: bool,
              mask_standalone_triple_strings: bool) -> tuple[bytes, list[dict], dict, str]:
    """Return a position-preserving transform, audited ranges, counts, and detected encoding.

    On ambiguous source/token structure, raise TransformError so caller can preserve
    the entire original file and record a fail-closed fallback.
    """
    tokens = list(tokenize.tokenize(io.BytesIO(raw).readline))
    encoding = tokenize.detect_encoding(io.BytesIO(raw).readline)[0]
    text = raw.decode(encoding)
    byte_lines, text_lines, offsets, bom_prefix, line_enc = _physical_lines(raw, text, encoding)
    tree = ast.parse(text, type_comments=True)
    string_tokens_by_span = {}
    for tok in tokens:
        if tok.type != tokenize.STRING:
            continue
        a = _to_raw_offset(tok.start[0], tok.start[1], utf8_column=False,
                           text_lines=text_lines, offsets=offsets, bom_prefix=bom_prefix, line_encoding=line_enc)
        b = _to_raw_offset(tok.end[0], tok.end[1], utf8_column=False,
                           text_lines=text_lines, offsets=offsets, bom_prefix=bom_prefix, line_encoding=line_enc)
        string_tokens_by_span[(a, b)] = tok
    marks: dict[int, tuple[ast.stmt, str, int, int]] = {}
    uncertain = 0

    for owner, field, statements in _suites(tree):
        owner_is_doc_holder = isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and field == "body"
        for index, stmt in enumerate(statements):
            # A standalone string is an Expr(Constant(str)) statement. Strings
            # used by Assign / Return / Call / Yield / F-string cannot match.
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)):
                continue
            is_head_docstring = owner_is_doc_holder and index == 0
            value = stmt.value
            start = _to_raw_offset(value.lineno, value.col_offset, utf8_column=True,
                                   text_lines=text_lines, offsets=offsets, bom_prefix=bom_prefix, line_encoding=line_enc)
            end = _to_raw_offset(value.end_lineno, value.end_col_offset, utf8_column=True,
                                 text_lines=text_lines, offsets=offsets, bom_prefix=bom_prefix, line_encoding=line_enc)
            matching_token = string_tokens_by_span.get((start, end))
            if matching_token is None:
                # Token/span mismatch, implicit literal concatenation, etc.
                # Preserve rather than accidentally remove executable data.
                uncertain += 1
                continue
            if is_head_docstring:
                # A disabled head-docstring policy is an explicit preserve, not
                # an implicit fall-through into standalone-triple masking.
                if not mask_head_docstrings:
                    continue
                category = "DOCSTRING"
            elif mask_standalone_triple_strings and _is_triple(matching_token.string):
                # Uniform rule at module, class, function and nested-suite scope.
                # This is a syntactic decision, not a claim about author intent.
                category = "STANDALONE_TRIPLE_STRING"
            else:
                continue
            marks[id(stmt)] = (stmt, category, start, end)

    replacements: dict[int, str] = {}
    for owner, field, statements in _suites(tree):
        candidate = [s for s in statements if id(s) in marks]
        if not candidate:
            continue
        suite_emptied = len(candidate) == len(statements)
        for stmt in candidate:
            same_line = any(x is not stmt and x.lineno == stmt.lineno for x in statements)
            wants_pass = suite_emptied and stmt is candidate[0] or same_line
            _, _, start, end = marks[id(stmt)]
            if wants_pass and (end - start < 4 or b"\n" in raw[start:start + 4] or b"\r" in raw[start:start + 4]):
                uncertain += 1
                continue
            replacements[id(stmt)] = "PASS" if wants_pass else "MASK"
        # If a short docstring prevented the suite from becoming empty, a pass is not needed.
        if suite_emptied and any(id(s) not in replacements for s in candidate):
            for s in candidate:
                if id(s) in replacements:
                    replacements[id(s)] = "MASK"

    updated = bytearray(raw)
    ranges = []
    counts = Counter()
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        if _protected_comment(token, prefixes):
            counts["preserved_special_comments"] += 1
            continue
        start = _to_raw_offset(token.start[0], token.start[1], utf8_column=False,
                               text_lines=text_lines, offsets=offsets, bom_prefix=bom_prefix, line_encoding=line_enc)
        end = _to_raw_offset(token.end[0], token.end[1], utf8_column=False,
                             text_lines=text_lines, offsets=offsets, bom_prefix=bom_prefix, line_encoding=line_enc)
        _mask_raw(updated, start, end)
        ranges.append({"category": "REGULAR_COMMENT", "start_line": token.start[0], "end_line": token.end[0],
                       "start_byte": start, "end_byte": end, "action": "MASK"})
        counts["masked_comments"] += 1

    for stmt, category, start, end in marks.values():
        action = replacements.get(id(stmt))
        if not action:
            continue
        _mask_raw(updated, start, end, pass_token=action == "PASS")
        ranges.append({"category": category, "start_line": stmt.lineno, "end_line": stmt.end_lineno,
                       "start_byte": start, "end_byte": end, "action": action})
        counts["masked_docstrings" if category == "DOCSTRING" else "masked_standalone_strings"] += 1
        if action == "PASS":
            counts["pass_insertions"] += 1
    counts["preserved_uncertain_strings"] = uncertain

    # Build the expected AST by dropping only the explicitly selected statement
    # nodes, or replacing them with 'pass' where a suite must remain executable.
    expected = copy.deepcopy(tree)
    # deepcopy loses object identity; map original/remapped statements by exact
    # structural position in each suite, not by source text or line alone.
    def update_pair(original: ast.AST, cloned: ast.AST) -> None:
        for attr, value in ast.iter_fields(original):
            other = getattr(cloned, attr)
            if isinstance(value, list) and value and all(isinstance(s, ast.stmt) for s in value):
                new_items = []
                for o_stmt, c_stmt in zip(value, other):
                    decision = replacements.get(id(o_stmt))
                    if decision == "MASK":
                        continue
                    if decision == "PASS":
                        new_items.append(ast.copy_location(ast.Pass(), c_stmt))
                    else:
                        update_pair(o_stmt, c_stmt)
                        new_items.append(c_stmt)
                setattr(cloned, attr, new_items)
            elif isinstance(value, list):
                for original_child, cloned_child in zip(value, other):
                    if isinstance(original_child, ast.AST):
                        update_pair(original_child, cloned_child)
            elif isinstance(value, ast.AST):
                update_pair(value, other)
    update_pair(tree, expected)

    result = bytes(updated)
    if len(raw) != len(result):
        raise TransformError("Source byte length changed")
    if any(raw[i] != result[i] for i in range(len(raw)) if raw[i] in (10, 13)):
        raise TransformError("Line ending byte changed")
    try:
        actual_tree = ast.parse(result.decode(encoding), type_comments=True)
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise TransformError(f"Masked Python cannot be parsed: {exc}") from exc
    if ast.dump(actual_tree, include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise TransformError("AST mismatch outside intended documentation masking")
    return result, ranges, dict(counts), encoding


def effective_config(args: argparse.Namespace) -> dict:
    return {"experiment": args.experiment, "script_version": args.version,
            "preproc_data_version": args.preproc_data_version, "require_clean": args.require_clean,
            "require_baseline_commit": args.require_baseline_commit, "parse_failure_policy": args.parse_failure_policy,
            "skip_directories": args.skip_dir, "skip_file_globs": args.skip_glob,
            "protected_comment_prefixes": args.protected_comment_prefix,
            "mask_head_docstrings": args.mask_head_docstrings,
            "mask_standalone_triple_strings": args.mask_standalone_triple_strings,
            "expected_subjects": args.expected_subjects, "progress_every": args.progress_every,
            "repo_root": str(args.repo_root.resolve()), "preproc_root": str(args.preproc_root.resolve()),
            "baseline_organizer": str(args.baseline_organizer.resolve())}


def _write_csv(path: Path, fields: tuple[str, ...], records: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _verify_subject(args: argparse.Namespace, repo: dict, baseline_sha: str) -> dict:
    name = repo["subject"]
    final = args.preproc_root / name
    if final.is_symlink() or not final.is_dir():
        raise RuntimeError(f"Preprocessed subject is missing or unsafe: {final}")
    manifest_file = final / ".j04b-preproc-manifest.json"
    if not manifest_file.is_file() or manifest_file.is_symlink():
        raise RuntimeError(f"Missing preproc manifest: {manifest_file}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected = {"subject": name, "git_head": repo["git_head"], "baseline_organizer_sha256": baseline_sha,
                "baseline_git_head": repo["baseline"]["git_head"], "status": "PREPARED",
                "config": effective_config(args), "github_origin": repo["origin"]}
    for key, val in expected.items():
        if manifest.get(key) != val:
            raise RuntimeError(f"Preproc manifest mismatch: {name} key={key}")
    audit_file = Path(manifest["file_audit"])
    range_file = Path(manifest["range_audit"])
    for file, key in ((audit_file, "file_audit_sha256"), (range_file, "range_audit_sha256")):
        if not file.is_file() or file.is_symlink() or digest_file(file) != manifest[key]:
            raise RuntimeError(f"Audit missing or changed: {name} file={file}")
    included = []
    audited = []
    with audit_file.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            audited.append(row["file"])
            if row["status"] not in ("INCLUDED", "EXCLUDED"):
                raise RuntimeError(f"Unknown audit status: {name} {row['status']}")
            if row["status"] == "INCLUDED":
                included.append(row)
    if len(audited) != len(set(audited)) or set(audited) != {p.as_posix() for p in repo["tracked"]}:
        raise RuntimeError(f"Audited Git-tracked file inventory mismatch: {name}")
    expected_paths = set()
    for row in included:
        relative = Path(row["file"])
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
            raise RuntimeError(f"Unsafe preproc path: {name} {relative}")
        src, dst = repo["repo"] / relative, final / relative
        if not dst.is_file() or dst.is_symlink() or not src.is_file() or src.is_symlink():
            raise RuntimeError(f"Preproc file missing/unsafe: {name} {relative}")
        if digest_file(src) != row["raw_sha256"] or digest_file(dst) != row["preproc_sha256"]:
            raise RuntimeError(f"Preproc file SHA mismatch: {name} {relative}")
        if src.stat().st_size != dst.stat().st_size or src.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"Preproc file byte length mismatch: {name} {relative}")
        original_bytes, transformed_bytes = src.read_bytes(), dst.read_bytes()
        if any((a in (10, 13)) != (b in (10, 13)) or
               (a in (10, 13) and a != b)
               for a, b in zip(original_bytes, transformed_bytes)):
            raise RuntimeError(f"Preproc newline position mismatch: {name} {relative}")
        expected_paths.add(relative.as_posix())
    all_entries = list(final.rglob("*"))
    if any(p.is_symlink() for p in all_entries):
        raise RuntimeError(f"Output contains an unsafe symlink: {name}")
    actual_paths = {p.relative_to(final).as_posix() for p in all_entries if p.is_file()}
    if actual_paths != expected_paths | {".j04b-preproc-manifest.json"}:
        raise RuntimeError(f"Unexpected/missing preproc files: {name}")
    if len(included) != manifest["counts"].get("included", 0):
        raise RuntimeError(f"Incorrect preproc included-file count: {name}")
    if git(repo["repo"], "rev-parse", "HEAD") != repo["git_head"]:
        raise RuntimeError(f"Git HEAD changed since inventory: {name}")
    print(f"PASS VERIFY subject={name} files={len(included)}", flush=True)
    return manifest


def prepare_subject(args: argparse.Namespace, repo: dict, baseline_sha: str) -> None:
    name = repo["subject"]
    final = args.preproc_root / name
    if final.exists() or final.is_symlink():
        _verify_subject(args, repo, baseline_sha)
        print(f"PASS EXISTING_VERIFIED subject={name}", flush=True)
        return
    args.preproc_root.mkdir(parents=True, exist_ok=True)
    # Both the run artifacts and final subject are immutable. Retain failed staging
    # directories for diagnosis; never silently remove them.
    stage = Path(tempfile.mkdtemp(prefix=f".{name}.pending-", dir=args.preproc_root))
    file_audit: list[dict] = []
    range_audit: list[dict] = []
    counts: Counter = Counter({key: 0 for key in (
        "included", "excluded", "bytes", "changed_files", "passthrough_files",
        "masked_comments", "masked_docstrings", "masked_standalone_strings",
        "pass_insertions", "preserved_special_comments", "preserved_uncertain_strings"
    )})
    try:
        for rel in repo["tracked"]:
            src = repo["repo"] / rel
            reason = exclusion_reason(rel, src, args.skip_dir, args.skip_glob)
            record = {"subject": name, "file": rel.as_posix(), "status": "EXCLUDED" if reason else "INCLUDED",
                      "reason": reason, "raw_sha256": "", "preproc_sha256": "", "bytes": "",
                      "masked_comments": 0, "masked_docstrings": 0, "masked_standalone_strings": 0,
                      "pass_insertions": 0, "preserved_special_comments": 0,
                      "preserved_uncertain_strings": 0, "fallback": "", "source_encoding": ""}
            if reason:
                counts["excluded"] += 1
            else:
                raw = src.read_bytes()
                encoding = ""
                try:
                    transformed, ranges, metrics, encoding = transform(raw, args.protected_comment_prefix,
                                                           mask_head_docstrings=args.mask_head_docstrings,
                                                           mask_standalone_triple_strings=args.mask_standalone_triple_strings)
                    fallback = ""
                except (SyntaxError, UnicodeDecodeError, tokenize.TokenError, TransformError, LookupError) as exc:
                    if args.parse_failure_policy == "fail_closed":
                        raise RuntimeError(f"Preprocessing ambiguity: {name}/{rel}: {exc}") from exc
                    transformed, ranges, metrics = raw, [], {},
                    fallback = f"{type(exc).__name__}:{str(exc)[:240]}"
                    counts["passthrough_files"] += 1
                    print(f"[{chicago()}] PRESERVE_UNCERTAIN subject={name} file={rel.as_posix()} reason={fallback}", flush=True)
                if src.read_bytes() != raw:
                    raise RuntimeError(f"Source changed during processing: {src}")
                dst = stage / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                with dst.open("xb") as stream:
                    stream.write(transformed)
                if digest_file(dst) != digest_bytes(transformed):
                    raise RuntimeError(f"Written file SHA mismatch: {dst}")
                record.update({"raw_sha256": digest_bytes(raw), "preproc_sha256": digest_bytes(transformed),
                               "bytes": len(raw), "fallback": fallback, "source_encoding": encoding})
                for key, val in metrics.items():
                    record[key] = val
                    counts[key] += val
                for item in ranges:
                    range_audit.append({"subject": name, "file": rel.as_posix(), **item})
                counts["included"] += 1
                counts["bytes"] += len(raw)
                counts["changed_files"] += int(raw != transformed)
                if counts["included"] % args.progress_every == 0:
                    print(f"[{chicago()}] PREPROC_PROGRESS subject={name} included={counts['included']}", flush=True)
            file_audit.append(record)
        if git(repo["repo"], "rev-parse", "HEAD") != repo["git_head"]:
            raise RuntimeError(f"Git HEAD changed during preprocessing: {name}")
        if args.require_clean and git(repo["repo"], "status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError(f"Git checkout became dirty during preprocessing: {name}")
        audit_path = args.report_dir / f"{name}-file-audit.csv"
        range_path = args.report_dir / f"{name}-range-audit.csv"
        _write_csv(audit_path, FILE_AUDIT_FIELDS, file_audit)
        _write_csv(range_path, RANGE_FIELDS, range_audit)
        manifest = {
            "experiment": args.experiment, "script_version": args.version,
            "preproc_data_version": args.preproc_data_version,
            "subject": name, "timestamp_chicago": chicago(), "status": "PREPARED",
            "raw_source_root": str(repo["repo"].resolve()), "preproc_source_root": str(final.resolve()),
            "git_head": repo["git_head"], "github_origin": repo["origin"],
            "baseline_git_head": repo["baseline"]["git_head"],
            "baseline_github_slug": repo["baseline"]["github_slug"],
            "baseline_c01_run_id": repo["baseline"]["c01_run_id"],
            "baseline_organizer_sha256": baseline_sha,
            "file_audit": str(audit_path.resolve()), "file_audit_sha256": digest_file(audit_path),
            "range_audit": str(range_path.resolve()), "range_audit_sha256": digest_file(range_path),
            "counts": dict(counts), "config": effective_config(args),
            "representation": "static_clone_detection_comment_and_standalone_string_masked_not_runtime_equivalent",
            "limitations": ["Docstring removal can affect __doc__/inspect.getdoc(); no dynamic equivalence is claimed",
                            "Standalone triple-quoted string expressions are syntactically masked at all suite scopes",
                            "Ambiguous or unparseable source is preserved under configured policy"],
        }
        write_json_new(args.report_dir / f"{name}-preproc-summary.json", manifest)
        write_json_new(stage / ".j04b-preproc-manifest.json", manifest)
        if final.exists() or final.is_symlink():
            raise RuntimeError(f"Concurrent output exists; refusing overwrite: {final}")
        stage.rename(final)
        _verify_subject(args, repo, baseline_sha)
        print(f"PASS PREPARED subject={name} included={counts['included']} excluded={counts['excluded']} "
              f"comments={counts['masked_comments']} docstrings={counts['masked_docstrings']} "
              f"standalone_strings={counts['masked_standalone_strings']} passthrough={counts['passthrough_files']}", flush=True)
    except BaseException:
        print(f"FAIL PREPARE subject={name} staging={stage} output={final}", file=sys.stderr, flush=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo-root", "preproc-root", "baseline-organizer", "report-dir", "experiment", "version",
                "preproc-data-version", "mode", "subject", "require-clean", "require-baseline-commit",
                "parse-failure-policy", "expected-subjects", "progress-every",
                "mask-head-docstrings", "mask-standalone-triple-strings"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--skip-dir", action="append", required=True)
    parser.add_argument("--skip-glob", action="append", required=True)
    parser.add_argument("--protected-comment-prefix", action="append", required=True)
    args = parser.parse_args()
    args.repo_root = Path(args.repo_root)
    args.preproc_root = Path(args.preproc_root)
    args.baseline_organizer = Path(args.baseline_organizer)
    args.report_dir = Path(args.report_dir)
    for key in ("require_clean", "require_baseline_commit", "mask_head_docstrings", "mask_standalone_triple_strings"):
        value = getattr(args, key)
        if value not in ("true", "false"):
            parser.error(f"--{key.replace('_', '-')} must be true or false")
        setattr(args, key, value == "true")
    try:
        args.expected_subjects = int(args.expected_subjects)
        args.progress_every = int(args.progress_every)
    except ValueError:
        parser.error("--expected-subjects and --progress-every must be integers")
    if args.expected_subjects < 1 or args.progress_every < 1:
        parser.error("Expected subjects and progress interval must be positive")
    if args.parse_failure_policy not in ("preserve_original_audited", "fail_closed"):
        parser.error("Invalid --parse-failure-policy")
    if args.mode not in ("inspect", "dry-run", "smoke", "subject", "all", "verify"):
        parser.error("Invalid --mode")
    args.skip_dir = [x.casefold() for x in args.skip_dir]
    args.skip_glob = [x.casefold() for x in args.skip_glob]
    args.protected_comment_prefix = [x.casefold() for x in args.protected_comment_prefix]
    safe_subject(args.subject)
    if args.mode in ("all", "smoke", "subject") and (not args.report_dir.exists() or not args.report_dir.is_dir()):
        parser.error("Report directory must exist before preprocessing")
    return args


def main() -> int:
    args = parse_args()
    print(f"J04B START mode={args.mode} version={args.version} timestamp_chicago={chicago()}", flush=True)
    inventory, baseline_sha = repository_inventory(args)
    mapping = {item["subject"]: item for item in inventory}
    if args.mode == "inspect":
        print(f"PASS INSPECT subjects={len(mapping)}", flush=True)
        return 0
    if args.mode == "dry-run":
        blocked = []
        for item in inventory:
            output = args.preproc_root / item["subject"]
            if output.exists() or output.is_symlink():
                try:
                    _verify_subject(args, item, baseline_sha)
                except Exception as exc:
                    blocked.append(f"{item['subject']}: {exc}")
        if blocked:
            raise RuntimeError("Existing unverified output(s): " + "; ".join(blocked))
        print(f"PASS DRY_RUN subjects={len(mapping)} source={args.repo_root} destination={args.preproc_root}", flush=True)
        return 0
    if args.mode in ("smoke", "subject"):
        if args.subject not in mapping:
            raise RuntimeError(f"Requested subject not found: {args.subject}")
        selected = [mapping[args.subject]]
    else:
        selected = inventory
    if args.mode == "verify":
        for item in selected:
            _verify_subject(args, item, baseline_sha)
        print(f"PASS VERIFY_ALL subjects={len(selected)}", flush=True)
        return 0
    for item in selected:
        prepare_subject(args, item, baseline_sha)
    print(f"PASS PREPROC_ALL subjects={len(selected)} destination={args.preproc_root}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAIL J-04b: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)
