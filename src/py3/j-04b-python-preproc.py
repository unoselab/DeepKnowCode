#!/usr/bin/env python3
"""J-04b v8: immutable static masking with isolated Python 3.14 AST parsing.

The Shell wrapper owns all experiment configuration; every CLI input is required.
Input:  data/repositories/python-repos/{owner}__{repo}
Output: data/repositories/python-repo-preproc-v8/{owner}__{repo}
Do not write to the original Git checkouts or clone-detection archives.
"""
from __future__ import annotations

import argparse
import ast
import codecs
import csv
import fnmatch
import hashlib
import io
import json
import os
import signal
import re
import shutil
import subprocess
import sys
import traceback
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
    "fallback", "source_encoding", "parser_compat_pep758_headers", "parser_compat_pep758_commas", "ast_native_parsed"
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


def _pep758_parse_positions(tokens: list[tokenize.TokenInfo], text_lines: list[str]) -> tuple[list[int], int]:
    """Locate comma separators in except headers, without touching nested expression commas.

    Positions index Unicode code points of the decoded source; replacements preserve
    source length and all line/AST offsets. This supports PEP 758 on Python 3.12/3.13
    for parsing ONLY; original source/output bytes are never rewritten here.
    """
    starts = []
    offset = 0
    for line in text_lines:
        starts.append(offset)
        offset += len(line)
    positions = []
    headers = 0
    for index, token in enumerate(tokens):
        if token.type != tokenize.NAME or token.string != "except":
            continue
        depth = 0
        candidates = []
        for next_token in tokens[index + 1:]:
            if next_token.type in (tokenize.NEWLINE, tokenize.ENDMARKER):
                break
            if next_token.type == tokenize.OP:
                if next_token.string in ("(", "[", "{"):
                    depth += 1
                elif next_token.string in (")", "]", "}"):
                    depth -= 1
                    if depth < 0:
                        break
                elif depth == 0 and next_token.string == ":":
                    positions.extend(candidates)
                    headers += bool(candidates)
                    break
                elif depth == 0 and next_token.string == ",":
                    row, col = next_token.start
                    if 1 <= row <= len(starts):
                        candidates.append(starts[row - 1] + col)
            if depth == 0 and next_token.type == tokenize.NAME and next_token.string == "as":
                # `except E1, E2 as err:` -- candidates already collected.
                # Remaining syntax until ':' cannot contain more exception types.
                continue
    return sorted(set(positions)), headers


def _parse_source_with_compat(text: str, tokens: list[tokenize.TokenInfo],
                              text_lines: list[str], enabled: bool) -> tuple[ast.AST, list[int], int]:
    try:
        return ast.parse(text, type_comments=True), [], 0
    except SyntaxError as exc:
        if not enabled or "multiple exception types must be parenthesized" not in str(exc):
            raise
        positions, headers = _pep758_parse_positions(tokens, text_lines)
        if not positions or any(text[i] != "," for i in positions):
            raise
        patched = list(text)
        for pos in positions:
            patched[pos] = "|"
        # The normalized input is *never* used as output, only as parser input.
        return ast.parse("".join(patched), type_comments=True), positions, headers


def _ast_equivalent_after_mask(original: ast.AST, actual: ast.AST,
                               replacements: dict[int, str]) -> bool:
    """Iterative AST structural equality, interpreting selected MASK/PASS nodes.

    Replaces v5's deepcopy + recursive update_pair + ast.dump, all of which can
    overflow on valid deeply nested trees. It ignores AST position attributes
    exactly like ast.dump(include_attributes=False).
    """
    pending = [(original, actual)]
    while pending:
        old, new = pending.pop()
        if isinstance(old, ast.AST):
            if not isinstance(new, ast.AST) or type(old) is not type(new):
                return False
            old_fields = list(ast.iter_fields(old))
            new_fields = list(ast.iter_fields(new))
            if [x[0] for x in old_fields] != [x[0] for x in new_fields]:
                return False
            pending.extend((old_value, new_value) for (_, old_value), (_, new_value)
                           in zip(old_fields, new_fields))
        elif isinstance(old, list):
            if not isinstance(new, list):
                return False
            selected = [(node, replacements.get(id(node))) for node in old]
            selected = [(node, decision) for node, decision in selected if decision != "MASK"]
            if len(selected) != len(new):
                return False
            for (node, decision), actual_node in zip(selected, new):
                if decision == "PASS":
                    if not isinstance(actual_node, ast.Pass):
                        return False
                else:
                    pending.append((node, actual_node))
        elif old != new or type(old) is not type(new):
            return False
    return True


def transform(raw: bytes, prefixes: list[str], *, mask_head_docstrings: bool,
              mask_standalone_triple_strings: bool,
              parser_compat_pep758: bool) -> tuple[bytes, list[dict], dict, str]:
    """Return a position-preserving transform, audited ranges, counts, and detected encoding.

    On ambiguous source/token structure, raise TransformError so caller can preserve
    the entire original file and record a fail-closed fallback.
    """
    tokens = list(tokenize.tokenize(io.BytesIO(raw).readline))
    encoding = tokenize.detect_encoding(io.BytesIO(raw).readline)[0]
    text = raw.decode(encoding)
    byte_lines, text_lines, offsets, bom_prefix, line_enc = _physical_lines(raw, text, encoding)
    tree, pep758_positions, pep758_headers = _parse_source_with_compat(
        text, tokens, text_lines, parser_compat_pep758)
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

    # Validate changed source without cloning or recursively traversing the AST.
    # For PEP 758, compare the same analysis-only normalized interpretation.
    counts["parser_compat_pep758_headers"] = pep758_headers
    counts["parser_compat_pep758_commas"] = len(pep758_positions)

    result = bytes(updated)
    if len(raw) != len(result):
        raise TransformError("Source byte length changed")
    if any(raw[i] != result[i] for i in range(len(raw)) if raw[i] in (10, 13)):
        raise TransformError("Line ending byte changed")
    try:
        result_text = result.decode(encoding)
        if pep758_positions:
            if any(result_text[i] != "," for i in pep758_positions):
                raise TransformError("PEP 758 exception header changed unexpectedly")
            compat_text = list(result_text)
            for pos in pep758_positions:
                compat_text[pos] = "|"
            result_text = "".join(compat_text)
        actual_tree = ast.parse(result_text, type_comments=True)
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise TransformError(f"Masked Python cannot be parsed: {exc}") from exc
    if not _ast_equivalent_after_mask(tree, actual_tree, replacements):
        raise TransformError("AST mismatch outside intended documentation masking")
    return result, ranges, dict(counts), encoding





def transform_external(raw: bytes, prefixes: list[str], *, mask_head_docstrings: bool,
                       mask_standalone_triple_strings: bool,
                       parser: AstWorkerClient) -> tuple[bytes, list[dict], dict, str]:
    """v7 production: main 3.12 masks bytes; only native 3.14 worker parses AST.

    Receives source AST statement ranges and suite shape; never re-parses AST in
    main process. Native worker checks the masked AST against the original AST.
    """
    tokens = list(tokenize.tokenize(io.BytesIO(raw).readline))
    encoding = tokenize.detect_encoding(io.BytesIO(raw).readline)[0]
    text = raw.decode(encoding)
    _, text_lines, offsets, bom_prefix, line_enc = _physical_lines(raw, text, encoding)
    suites = parser.analyze(text)
    string_tokens_by_span = {}
    for tok in tokens:
        if tok.type != tokenize.STRING:
            continue
        a = _to_raw_offset(tok.start[0], tok.start[1], utf8_column=False,
                           text_lines=text_lines, offsets=offsets,
                           bom_prefix=bom_prefix, line_encoding=line_enc)
        b = _to_raw_offset(tok.end[0], tok.end[1], utf8_column=False,
                           text_lines=text_lines, offsets=offsets,
                           bom_prefix=bom_prefix, line_encoding=line_enc)
        string_tokens_by_span[(a, b)] = tok
    marks = {}
    uncertain = 0
    for suite in suites:
        sid = suite["id"]
        for candidate in suite["candidates"]:
            idx = candidate["index"]
            vrow, vcol = candidate["start"]
            erow, ecol = candidate["end"]
            start = _to_raw_offset(vrow, vcol, utf8_column=True,
                                   text_lines=text_lines, offsets=offsets,
                                   bom_prefix=bom_prefix, line_encoding=line_enc)
            end = _to_raw_offset(erow, ecol, utf8_column=True,
                                 text_lines=text_lines, offsets=offsets,
                                 bom_prefix=bom_prefix, line_encoding=line_enc)
            token = string_tokens_by_span.get((start, end))
            if token is None:
                uncertain += 1
                continue
            is_head_docstring = suite["doc_holder"] and idx == 0
            if is_head_docstring:
                if not mask_head_docstrings:
                    continue
                category = "DOCSTRING"
            elif mask_standalone_triple_strings and _is_triple(token.string):
                category = "STANDALONE_TRIPLE_STRING"
            else:
                continue
            marks[(sid, idx)] = (candidate, category, start, end)

    replacements = {}
    for suite in suites:
        sid = suite["id"]
        present = [candidate for candidate in suite["candidates"]
                   if (sid, candidate["index"]) in marks]
        if not present:
            continue
        lines = suite["statement_lines"]
        suite_emptied = len(present) == len(lines)
        for candidate in present:
            idx = candidate["index"]
            same_line = any(i != idx and n == candidate["stmt_line"]
                            for i, n in enumerate(lines))
            wants_pass = (suite_emptied and candidate is present[0]) or same_line
            start, end = marks[(sid, idx)][2:]
            if wants_pass and (end - start < 4 or b"\n" in raw[start:start + 4]
                               or b"\r" in raw[start:start + 4]):
                uncertain += 1
                continue
            replacements[(sid, idx)] = "PASS" if wants_pass else "MASK"
        if suite_emptied and any((sid, c["index"]) not in replacements for c in present):
            for candidate in present:
                key = (sid, candidate["index"])
                if key in replacements:
                    replacements[key] = "MASK"

    updated = bytearray(raw)
    ranges = []
    counts = Counter()
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        if _protected_comment(tok, prefixes):
            counts["preserved_special_comments"] += 1
            continue
        start = _to_raw_offset(tok.start[0], tok.start[1], utf8_column=False,
                               text_lines=text_lines, offsets=offsets,
                               bom_prefix=bom_prefix, line_encoding=line_enc)
        end = _to_raw_offset(tok.end[0], tok.end[1], utf8_column=False,
                             text_lines=text_lines, offsets=offsets,
                             bom_prefix=bom_prefix, line_encoding=line_enc)
        _mask_raw(updated, start, end)
        ranges.append({"category": "REGULAR_COMMENT", "start_line": tok.start[0],
                       "end_line": tok.end[0], "start_byte": start, "end_byte": end,
                       "action": "MASK"})
        counts["masked_comments"] += 1
    for (sid, idx), (candidate, category, start, end) in marks.items():
        action = replacements.get((sid, idx))
        if not action:
            continue
        _mask_raw(updated, start, end, pass_token=action == "PASS")
        ranges.append({"category": category, "start_line": candidate["stmt_line"],
                       "end_line": candidate["stmt_end_line"],
                       "start_byte": start, "end_byte": end, "action": action})
        counts["masked_docstrings" if category == "DOCSTRING" else "masked_standalone_strings"] += 1
        if action == "PASS":
            counts["pass_insertions"] += 1
    counts["preserved_uncertain_strings"] = uncertain
    # No PEP 758 comma->pipe interpretation; Python 3.14 parses the *original* text.
    counts["parser_compat_pep758_headers"] = 0
    counts["parser_compat_pep758_commas"] = 0
    counts["ast_native_parsed"] = 1

    result = bytes(updated)
    if len(raw) != len(result):
        raise TransformError("Source byte length changed")
    if any(raw[i] != result[i] for i in range(len(raw)) if raw[i] in (10, 13)):
        raise TransformError("Line ending byte changed")
    result_text = result.decode(encoding)
    decisions = [[sid, idx, action] for (sid, idx), action in sorted(replacements.items())]
    parser.verify(result_text, decisions)
    return result, ranges, dict(counts), encoding


class AstWorkerClient:
    """Persistent Python 3.14 parser subprocess; never silently falls back to 3.12.

    One JSON-line request at a time; runtime or protocol failure aborts the run.
    Only syntax/ambiguous-AST failures can cause audited per-file passthrough.
    """
    def __init__(self, *, executable: Path, worker_script: Path, min_version: str,
                 expected_version: str, suppress_syntax_warnings: bool):
        self.executable = executable.resolve(strict=True)
        self.worker_script = worker_script.resolve(strict=True)
        self.min_version = min_version
        self.expected_version = expected_version
        self.suppress_syntax_warnings = suppress_syntax_warnings
        self.proc = None
        self.runtime_executable = None
        self.runtime_version = None

    def _response(self) -> dict:
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError("AST worker not started")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"AST worker terminated unexpectedly exit={self.proc.poll()}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid AST worker JSON response") from exc
        if not isinstance(value, dict) or type(value.get("ok")) is not bool:
            raise RuntimeError("Malformed AST worker response")
        if not value["ok"]:
            if value.get("kind") == "source":
                raise TransformError("AST worker: " + str(value.get("error", "unknown source error")))
            raise RuntimeError("AST worker protocol failure: " + str(value.get("error", "unknown error")))
        return value

    def _request(self, value: dict) -> dict:
        if self.proc is None or self.proc.stdin is None or self.proc.poll() is not None:
            raise RuntimeError("AST worker is unavailable")
        try:
            self.proc.stdin.write(json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("AST worker write failed") from exc
        return self._response()

    def start(self):
        if self.proc is not None:
            raise RuntimeError("AST worker already started")
        if not self.worker_script.is_file() or self.worker_script.is_symlink():
            raise RuntimeError("AST worker script must be a regular non-symlink file")
        self.proc = subprocess.Popen(
            [str(self.executable), "-u", str(self.worker_script), "--serve",
             "--min-version", self.min_version, "--expected-version", self.expected_version,
             "--suppress-syntax-warnings", str(self.suppress_syntax_warnings).lower()],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", bufsize=1)
        hello = self._response()
        if hello.get("kind") != "hello" or hello.get("protocol") != "j04b-ast-v1":
            raise RuntimeError("Unexpected AST worker protocol")
        if hello.get("version") != self.expected_version:
            raise RuntimeError("Unexpected AST worker Python version: " + str(hello.get("version")))
        actual = str(Path(hello["executable"]).resolve(strict=True))
        if actual != str(self.executable):
            raise RuntimeError(f"AST worker Python executable mismatch: {actual}")
        self.runtime_version = hello["version"]
        self.runtime_executable = actual
        # Actual Python 3.14 native support is verified before input data is touched.
        if tuple(int(x) for x in self.min_version.split(".")) >= (3, 14):
            probe = "try:\n    pass\nexcept ValueError, TypeError:\n    pass\n"
            self.analyze(probe)
            self.verify(probe, [])
        print(f"PASS AST_WORKER protocol=j04b-ast-v1 python={self.runtime_version} "
              f"executable={self.runtime_executable} sha256={digest_file(self.worker_script)}", flush=True)
        return self

    def analyze(self, source: str) -> list[dict]:
        reply = self._request({"op": "analyze", "text": source})
        if reply.get("kind") != "analysis" or not isinstance(reply.get("suites"), list):
            raise RuntimeError("Malformed AST worker analysis response")
        return reply["suites"]

    def verify(self, result: str, decisions: list[list]) -> None:
        reply = self._request({"op": "verify", "text": result, "decisions": decisions})
        if reply.get("kind") != "verified":
            raise RuntimeError("Malformed AST worker equivalence response")

    def close(self):
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            try:
                rc = self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                rc = self.proc.wait(timeout=5)
            if rc != 0:
                raise RuntimeError(f"AST worker exited with code {rc}")
        finally:
            if self.proc.stdout:
                self.proc.stdout.close()
            self.proc = None


def effective_config(args: argparse.Namespace) -> dict:
    return {"experiment": args.experiment, "script_version": args.version,
            "preproc_data_version": args.preproc_data_version, "require_clean": args.require_clean,
            "require_baseline_commit": args.require_baseline_commit, "parse_failure_policy": args.parse_failure_policy,
            "skip_directories": args.skip_dir, "skip_file_globs": args.skip_glob,
            "protected_comment_prefixes": args.protected_comment_prefix,
            "mask_head_docstrings": args.mask_head_docstrings,
            "mask_standalone_triple_strings": args.mask_standalone_triple_strings,
            "parser_compat_pep758": args.parser_compat_pep758,
            "python_min_version": args.python_min_version,
            "main_python_expected_version": args.main_python_expected_version,
            "ast_python_expected_version": args.ast_python_expected_version,
            "ast_python_min_version": args.ast_python_min_version,
            "ast_python_executable": args.ast_worker.runtime_executable,
            "ast_python_runtime_version": args.ast_worker.runtime_version,
            "ast_worker_script": str(args.ast_worker.worker_script),
            "ast_worker_sha256": digest_file(args.ast_worker.worker_script),
            "ast_parser_mode": args.ast_parser_mode,
            "ast_suppress_syntax_warnings": args.ast_suppress_syntax_warnings,
            "python_runtime_version": sys.version.split()[0],
            "python_runtime_executable": str(Path(sys.executable).resolve()),
            "expected_subjects": args.expected_subjects, "progress_every": args.progress_every,
            "repo_root": str(args.repo_root.resolve()), "preproc_root": str(args.preproc_root.resolve()),
            "baseline_organizer": str(args.baseline_organizer.resolve()),
            "clone_input_root": str(args.clone_input_root.resolve()),
            "clone_input_policy": args.clone_input_policy,
            "clone_copy_mode": args.clone_copy_mode,
            "clone_expected_total_included": args.clone_expected_total_included,
            "clone_expected_unparsed_files": args.clone_expected_unparsed_files}


def _write_csv(path: Path, fields: tuple[str, ...], records: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def inventory_exclusions(args: argparse.Namespace, inventory: list[dict], baseline_sha: str) -> None:
    """Read-only comparison of legacy v3 rules with effective v6 directory exclusions."""
    candidate = set(args.candidate_dir)
    if not candidate:
        raise RuntimeError("Candidate directories required for inventory")
    if args.report_dir.is_symlink() or not args.report_dir.is_dir():
        raise RuntimeError("Unsafe or missing inventory report directory")
    rows = []
    directory_counter = Counter()
    subject_counter = Counter()
    candidate_counter = Counter()
    total = Counter()
    for repo in inventory:
        name = repo["subject"]
        for rel in repo["tracked"]:
            source = repo["repo"] / rel
            previous = exclusion_reason(rel, source, getattr(args, "inventory_baseline_skip_dir", args.skip_dir), args.skip_glob)
            effective = exclusion_reason(rel, source, args.skip_dir, args.skip_glob)
            directory_parts = [part.casefold() for part in rel.parts[:-1]]
            matches = [part for part in directory_parts if part in candidate]
            # A file may match multiple candidate dirs. Only the first is used for
            # uniquely counted impact; all hits remain in the path-level audit.
            primary = matches[0] if matches else ""
            extra = bool(primary and not previous)
            status = "EXCLUDED_V3" if previous else ("CANDIDATE_EXCLUDE" if extra else "INCLUDE")
            record = {
                "subject": name, "path": rel.as_posix(),
                "v3_status": "EXCLUDE" if previous else "INCLUDE",
                "v3_reason": previous, "candidate_directory": primary,
                "candidate_all_matches": ";".join(dict.fromkeys(matches)),
                "candidate_new_exclusion": "true" if extra else "false",
                "proposed_status": status,
                "v5_status": "EXCLUDE" if effective else "INCLUDE",
                "v5_reason": effective,
                "v6_status": "EXCLUDE" if effective else "INCLUDE",
                "v6_reason": effective,
            }
            rows.append(record)
            total["tracked"] += 1
            subject_counter[(name, "tracked")] += 1
            if previous:
                total["v3_excluded"] += 1
                subject_counter[(name, "v3_excluded")] += 1
            else:
                total["v3_included"] += 1
                subject_counter[(name, "v3_included")] += 1
            if extra:
                total["additional_candidate"] += 1
                subject_counter[(name, "additional_candidate")] += 1
                candidate_counter[(name, primary)] += 1
            elif primary and previous:
                total["already_excluded_candidate"] += 1
            # Count each directory name at most once per tracked file.
            for part in set(directory_parts):
                directory_counter[(name, part, "tracked")] += 1
                if previous:
                    directory_counter[(name, part, "v3_excluded")] += 1
                elif extra:
                    directory_counter[(name, part, "candidate_new_exclusion")] += 1
                else:
                    directory_counter[(name, part, "remaining_included")] += 1
    subject_rows = [{"subject": r["subject"], "git_head": r["git_head"],
                     "tracked_py": subject_counter[(r["subject"], "tracked")],
                     "v3_included": subject_counter[(r["subject"], "v3_included")],
                     "v3_excluded": subject_counter[(r["subject"], "v3_excluded")],
                     "new_candidate_exclusions": subject_counter[(r["subject"], "additional_candidate")],
                     "remaining_if_approved": subject_counter[(r["subject"], "v3_included")] - subject_counter[(r["subject"], "additional_candidate")]}
                    for r in inventory]
    dirs = sorted({(name, part) for name, part, metric in directory_counter})
    dir_rows = [{"subject": name, "directory_component": part,
                 "tracked_py": directory_counter[(name, part, "tracked")],
                 "v3_excluded": directory_counter[(name, part, "v3_excluded")],
                 "candidate_new_exclusion": directory_counter[(name, part, "candidate_new_exclusion")],
                 "remaining_included": directory_counter[(name, part, "remaining_included")],
                 "candidate_flag": "true" if part in candidate else "false"}
                for name, part in dirs]
    candidate_rows = [{"subject": name, "candidate_directory": part, "new_exclusions": count}
                      for (name, part), count in sorted(candidate_counter.items())]
    file_outputs = {
        "path_inventory": ("path-inventory.csv", tuple(rows[0].keys()) if rows else (), rows),
        "subject_impact": ("subject-impact.csv", tuple(subject_rows[0].keys()), subject_rows),
        "directory_components": ("directory-components.csv", tuple(dir_rows[0].keys()) if dir_rows else (), dir_rows),
        "candidate_impact": ("candidate-impact.csv", ("subject", "candidate_directory", "new_exclusions"), candidate_rows),
    }
    hashes = {}
    for label, (filename, fields, data) in file_outputs.items():
        path = args.report_dir / filename
        _write_csv(path, fields, data)
        hashes[label] = {"filename": filename, "sha256": digest_file(path), "rows": len(data)}
    summary = {
        "experiment": args.experiment, "version": args.version, "mode": args.mode,
        "timestamp_chicago": chicago(), "baseline_organizer_sha256": baseline_sha,
        "expected_subjects": args.expected_subjects, "actual_subjects": len(inventory),
        "v3_skip_directories": getattr(args, "inventory_baseline_skip_dir", args.skip_dir), "v3_skip_file_globs": args.skip_glob,
        "v5_effective_skip_directories": args.skip_dir,
        "v5_effective_skip_file_globs": args.skip_glob,
        "v6_effective_skip_directories": args.skip_dir,
        "v6_effective_skip_file_globs": args.skip_glob,
        "approved_nonproduction_directories": args.candidate_dir,
        "totals": dict(total), "files": hashes,
        "read_only": True, "preprocessing_executed": False,
        "source_root": str(args.repo_root.resolve()),
        "output_root_not_written": str(args.preproc_root.absolute()),
    }
    write_json_new(args.report_dir / "inventory-summary.json", summary)
    print(f"PASS EXCLUSION_INVENTORY subjects={len(inventory)} tracked_py={total['tracked']} "
          f"v3_included={total['v3_included']} v3_excluded={total['v3_excluded']} "
          f"additional_candidate={total['additional_candidate']} report={args.report_dir}", flush=True)
    for item in subject_rows:
        print(f"INVENTORY_SUBJECT subject={item['subject']} tracked={item['tracked_py']} "
              f"v3_included={item['v3_included']} additional_candidate={item['new_candidate_exclusions']}", flush=True)


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
    passthrough = sum(bool(row["fallback"]) for row in included)
    native = sum(row["ast_native_parsed"] == "1" for row in included)
    if (passthrough != manifest["counts"].get("passthrough_files", 0) or
            native != manifest["counts"].get("ast_native_parsed", 0) or
            passthrough + native != len(included) or
            any(row["ast_native_parsed"] not in ("0", "1") or
                bool(row["fallback"]) == (row["ast_native_parsed"] == "1")
                for row in included)):
        raise RuntimeError(f"Inconsistent native AST / passthrough audit: {name}")
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
        "pass_insertions", "preserved_special_comments", "preserved_uncertain_strings",
        "parser_compat_pep758_headers", "parser_compat_pep758_commas", "ast_native_parsed"
    )})
    current_rel = None
    try:
        for rel in repo["tracked"]:
            current_rel = rel
            src = repo["repo"] / rel
            reason = exclusion_reason(rel, src, args.skip_dir, args.skip_glob)
            record = {"subject": name, "file": rel.as_posix(), "status": "EXCLUDED" if reason else "INCLUDED",
                      "reason": reason, "raw_sha256": "", "preproc_sha256": "", "bytes": "",
                      "masked_comments": 0, "masked_docstrings": 0, "masked_standalone_strings": 0,
                      "pass_insertions": 0, "preserved_special_comments": 0,
                      "preserved_uncertain_strings": 0, "fallback": "", "source_encoding": "",
                      "parser_compat_pep758_headers": 0, "parser_compat_pep758_commas": 0, "ast_native_parsed": 0}
            if reason:
                counts["excluded"] += 1
            else:
                raw = src.read_bytes()
                encoding = ""
                try:
                    transformed, ranges, metrics, encoding = transform_external(
                        raw, args.protected_comment_prefix,
                        mask_head_docstrings=args.mask_head_docstrings,
                        mask_standalone_triple_strings=args.mask_standalone_triple_strings,
                        parser=args.ast_worker)
                    fallback = ""
                except (SyntaxError, UnicodeDecodeError, tokenize.TokenError, TransformError, LookupError, RecursionError) as exc:
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
        if counts["ast_native_parsed"] + counts["passthrough_files"] != counts["included"]:
            raise RuntimeError(f"Native AST / passthrough count invariant failed: {name}")
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
        print(f"FAIL PREPARE subject={name} file={current_rel} staging={stage} output={final}", file=sys.stderr, flush=True)
        raise


def _clone_safe_python_relative(raw: str) -> Path:
    """Accept only canonical relative Python file names from the verified file audit."""
    rel = Path(raw)
    if (not raw or rel.is_absolute() or '..' in rel.parts or rel.suffix != '.py'
            or rel.as_posix() != raw or any(part in ('', '.') for part in raw.split('/'))):
        raise RuntimeError(f"Unsafe audited Python path: {raw}")
    return rel


def _clone_collect(args: argparse.Namespace, inventory: list[dict], baseline_sha: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Collect verified native-AST files; auditable failure policy is fail-closed."""
    accepted: list[dict] = []
    rejected: list[dict] = []
    subjects: list[dict] = []
    for repo in inventory:
        manifest = _verify_subject(args, repo, baseline_sha)
        subject = repo['subject']
        manifest_path = args.preproc_root / subject / '.j04b-preproc-manifest.json'
        audit_path = Path(manifest['file_audit'])
        counter = Counter()
        with audit_path.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                if row['subject'] != subject:
                    raise RuntimeError(f"Mismatched subject in audited row: {subject}")
                relative = _clone_safe_python_relative(row['file'])
                if row['status'] == 'EXCLUDED':
                    counter['upstream_excluded'] += 1
                    continue
                if row['status'] != 'INCLUDED':
                    raise RuntimeError(f"Unknown audit status: {subject}/{relative}")
                counter['upstream_included'] += 1
                preprocessed = args.preproc_root / subject / relative
                if not preprocessed.is_file() or preprocessed.is_symlink():
                    raise RuntimeError(f"Missing/unsafe source for clone input: {preprocessed}")
                if digest_file(preprocessed) != row['preproc_sha256']:
                    raise RuntimeError(f"Source changed after verification: {preprocessed}")
                entry = {'subject': subject, 'file': relative.as_posix(), 'sha256': row['preproc_sha256']}
                if row['fallback']:
                    if row['ast_native_parsed'] != '0':
                        raise RuntimeError(f"Unparsed audit mismatch: {subject}/{relative}")
                    counter['unparsed_excluded'] += 1
                    rejected.append({**entry, 'reason': row['fallback']})
                else:
                    if row['ast_native_parsed'] != '1':
                        raise RuntimeError(f"Unverified AST file: {subject}/{relative}")
                    counter['clone_accepted'] += 1
                    accepted.append(entry)
        if (counter['upstream_included'] != manifest['counts']['included'] or
            counter['upstream_excluded'] != manifest['counts']['excluded'] or
            counter['unparsed_excluded'] != manifest['counts']['passthrough_files'] or
            counter['clone_accepted'] != manifest['counts']['ast_native_parsed']):
            raise RuntimeError(f"Subject audit totals mismatch: {subject}")
        subjects.append({'subject': subject, **counter,
                         'git_head': repo['git_head'],
                         'manifest_sha256': digest_file(manifest_path),
                         'file_audit_sha256': manifest['file_audit_sha256'],
                         'range_audit_sha256': manifest['range_audit_sha256']})
    if len(subjects) != args.expected_subjects or len(accepted) + len(rejected) != args.clone_expected_total_included:
        raise RuntimeError(f"Clone input inventory mismatch: subjects={len(subjects)} "
                           f"included={len(accepted)+len(rejected)}; "
                           f"expected_subjects={args.expected_subjects} "
                           f"expected_included={args.clone_expected_total_included}")
    if len(rejected) != args.clone_expected_unparsed_files:
        raise RuntimeError(f"Clone input unparsed count mismatch: actual={len(rejected)} "
                           f"expected={args.clone_expected_unparsed_files}")
    return accepted, rejected, subjects


def _clone_manifest(args: argparse.Namespace, accepted: list[dict], rejected: list[dict],
                    subjects: list[dict]) -> dict:
    return {
        'schema': 'deepknowcode-j04b-clone-input-v8',
        'experiment': args.experiment, 'version': args.version,
        'source_data_version': args.preproc_data_version,
        'timestamp_chicago': chicago(),
        'source_root': str(args.preproc_root.resolve()),
        'clone_input_root': str((args.clone_input_root / 'repositories').absolute()),
        'policy': args.clone_input_policy,
        'copy_mode': args.clone_copy_mode,
        'counts': {'subjects': len(subjects),
                   'upstream_included': len(accepted)+len(rejected),
                   'clone_included': len(accepted),
                   'unparsed_excluded': len(rejected)},
        'expected': {'subjects': args.expected_subjects,
                     'upstream_included': args.clone_expected_total_included,
                     'unparsed_excluded': args.clone_expected_unparsed_files},
        'subjects': subjects,
    }


def _clone_csv_bytes(fieldnames: tuple[str, ...], records: list[dict]) -> bytes:
    """Deterministic canonical CSV for independent verification."""
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue().encode('utf-8')


def _clone_check_destination(args: argparse.Namespace) -> None:
    source = args.preproc_root.absolute()
    target = args.clone_input_root.absolute()
    if source == target or source in target.parents or target in source.parents:
        raise RuntimeError(f"Clone input must be separate from preprocessing root: {target}")
    if target.is_symlink() or any(p.is_symlink() for p in target.parents):
        raise RuntimeError(f"Unsafe clone output path symlink: {target}")


def build_clone_input(args: argparse.Namespace, inventory: list[dict], baseline_sha: str) -> None:
    _clone_check_destination(args)
    if args.clone_input_root.exists() or args.clone_input_root.is_symlink():
        raise RuntimeError(f"Clone input output already exists; refusing overwrite: {args.clone_input_root}")
    accepted, rejected, subjects = _clone_collect(args, inventory, baseline_sha)
    args.clone_input_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{args.clone_input_root.name}.pending-',
                                  dir=args.clone_input_root.parent))
    print(f"[{chicago()}] START CLONE_FILTER accepted={len(accepted)} unparsed={len(rejected)} staging={stage}", flush=True)
    try:
        for index, record in enumerate(accepted, 1):
            relative = _clone_safe_python_relative(record['file'])
            source = args.preproc_root / record['subject'] / relative
            destination = stage / 'repositories' / record['subject'] / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as from_stream, destination.open('xb') as to_stream:
                shutil.copyfileobj(from_stream, to_stream)
            if digest_file(destination) != record['sha256'] or digest_file(source) != record['sha256']:
                raise RuntimeError(f"Clone copy SHA256 mismatch: {record['subject']}/{relative}")
            if index % args.progress_every == 0:
                print(f"[{chicago()}] CLONE_FILTER_PROGRESS copied={index}", flush=True)
        with (stage / 'clone-accepted.csv').open('xb') as stream:
            stream.write(_clone_csv_bytes(('subject', 'file', 'sha256'), accepted))
        with (stage / 'clone-excluded-parsing-failures.csv').open('xb') as stream:
            stream.write(_clone_csv_bytes(('subject', 'file', 'sha256', 'reason'), rejected))
        summary = _clone_manifest(args, accepted, rejected, subjects)
        summary['accepted_csv_sha256'] = digest_file(stage / 'clone-accepted.csv')
        summary['excluded_csv_sha256'] = digest_file(stage / 'clone-excluded-parsing-failures.csv')
        write_json_new(stage / 'clone-input-manifest.json', summary)
        if args.clone_input_root.exists() or args.clone_input_root.is_symlink():
            raise RuntimeError(f"Concurrent clone input output exists: {args.clone_input_root}")
        stage.rename(args.clone_input_root)
    except BaseException:
        print(f"FAIL CLONE_FILTER staging_retained={stage}", file=sys.stderr, flush=True)
        raise
    print(f"PASS CLONE_FILTER subjects={len(subjects)} upstream_included={len(accepted)+len(rejected)} "
          f"clone_included={len(accepted)} unparsed_excluded={len(rejected)} "
          f"destination={args.clone_input_root / 'repositories'}", flush=True)


def verify_clone_input(args: argparse.Namespace, inventory: list[dict], baseline_sha: str) -> None:
    _clone_check_destination(args)
    output = args.clone_input_root
    if not output.is_dir() or output.is_symlink():
        raise RuntimeError(f"Clone input directory missing/unsafe: {output}")
    accepted, rejected, subjects = _clone_collect(args, inventory, baseline_sha)
    manifest_file = output / 'clone-input-manifest.json'
    if not manifest_file.is_file() or manifest_file.is_symlink():
        raise RuntimeError(f"Missing clone input manifest: {manifest_file}")
    saved = json.loads(manifest_file.read_text(encoding='utf-8'))
    expected = _clone_manifest(args, accepted, rejected, subjects)
    expected.pop('timestamp_chicago')
    other = dict(saved)
    other.pop('timestamp_chicago', None)
    if other.pop('accepted_csv_sha256', None) != digest_file(output / 'clone-accepted.csv'):
        raise RuntimeError('Clone accepted CSV checksum mismatch')
    if other.pop('excluded_csv_sha256', None) != digest_file(output / 'clone-excluded-parsing-failures.csv'):
        raise RuntimeError('Clone excluded CSV checksum mismatch')
    if other != expected:
        raise RuntimeError('Clone input manifest does not match audited configuration')
    for filename, fields, records in (
        ('clone-accepted.csv', ('subject', 'file', 'sha256'), accepted),
        ('clone-excluded-parsing-failures.csv', ('subject', 'file', 'sha256', 'reason'), rejected),
    ):
        file_path = output / filename
        if file_path.is_symlink() or not file_path.is_file() or file_path.read_bytes() != _clone_csv_bytes(fields, records):
            raise RuntimeError(f"Clone input CSV mismatch: {filename}")
    expected_files = {'clone-input-manifest.json', 'clone-accepted.csv', 'clone-excluded-parsing-failures.csv'}
    for record in accepted:
        relative = Path('repositories') / record['subject'] / _clone_safe_python_relative(record['file'])
        file_path = output / relative
        if file_path.is_symlink() or not file_path.is_file() or digest_file(file_path) != record['sha256']:
            raise RuntimeError(f"Clone input missing/changed: {relative}")
        expected_files.add(relative.as_posix())
    observed_entries = list(output.rglob('*'))
    if any(p.is_symlink() for p in observed_entries):
        raise RuntimeError('Clone input contains symlinks')
    actual_files = {p.relative_to(output).as_posix() for p in observed_entries if p.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(f"Unexpected/missing clone input files: actual={len(actual_files)} expected={len(expected_files)}")
    print(f"PASS VERIFY_CLONE_INPUT subjects={len(subjects)} clone_included={len(accepted)} "
          f"unparsed_excluded={len(rejected)} destination={output / 'repositories'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo-root", "preproc-root", "baseline-organizer", "report-dir", "experiment", "version",
                "preproc-data-version", "mode", "subject", "require-clean", "require-baseline-commit",
                "parse-failure-policy", "expected-subjects", "progress-every",
                "mask-head-docstrings", "mask-standalone-triple-strings",
                "parser-compat-pep758", "python-min-version",
                "main-python-expected-version", "ast-python-bin", "ast-worker-script",
                "ast-python-expected-version", "ast-python-min-version",
                "ast-parser-mode", "ast-suppress-syntax-warnings", "clone-input-root",
                "clone-input-policy", "clone-copy-mode", "clone-expected-total-included",
                "clone-expected-unparsed-files", "exclude-unparsed-files"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--skip-dir", action="append", required=True)
    parser.add_argument("--skip-glob", action="append", required=True)
    parser.add_argument("--candidate-dir", action="append")
    parser.add_argument("--inventory-baseline-skip-dir", action="append")
    parser.add_argument("--protected-comment-prefix", action="append", required=True)
    args = parser.parse_args()
    args.repo_root = Path(args.repo_root)
    args.preproc_root = Path(args.preproc_root)
    args.baseline_organizer = Path(args.baseline_organizer)
    args.report_dir = Path(args.report_dir)
    args.clone_input_root = Path(args.clone_input_root)
    args.ast_python_bin = Path(args.ast_python_bin)
    args.ast_worker_script = Path(args.ast_worker_script)
    for key in ("require_clean", "require_baseline_commit", "mask_head_docstrings", "mask_standalone_triple_strings", "parser_compat_pep758",
                "ast_suppress_syntax_warnings", "exclude_unparsed_files"):
        value = getattr(args, key)
        if value not in ("true", "false"):
            parser.error(f"--{key.replace('_', '-')} must be true or false")
        setattr(args, key, value == "true")
    if not re.fullmatch(r"[0-9]+\.[0-9]+", args.python_min_version):
        parser.error("--python-min-version must be major.minor")
    minimum = tuple(map(int, args.python_min_version.split(".")))
    if tuple(sys.version_info[:2]) < minimum:
        parser.error(f"Python {args.python_min_version}+ required, running {sys.version.split()[0]}")
    if sys.version.split()[0] != args.main_python_expected_version:
        parser.error(f"Main Python must be {args.main_python_expected_version}; found {sys.version.split()[0]}")
    if args.clone_input_policy != 'native_ast_only_exclude_all_fallback':
        parser.error('Invalid --clone-input-policy')
    if args.clone_copy_mode != 'copy_verified':
        parser.error('Invalid --clone-copy-mode')
    try:
        args.clone_expected_total_included = int(args.clone_expected_total_included)
        args.clone_expected_unparsed_files = int(args.clone_expected_unparsed_files)
    except ValueError:
        parser.error('Clone expected counts must be integers')
    if (args.clone_expected_total_included < 1 or args.clone_expected_unparsed_files < 0 or
            args.clone_expected_unparsed_files > args.clone_expected_total_included):
        parser.error('Invalid expected clone inventory counts')
    if args.ast_parser_mode != "external_py314_native":
        parser.error("v8 production requires --ast-parser-mode external_py314_native")
    if args.parser_compat_pep758:
        parser.error("PEP 758 compatibility rewrites must be disabled for native AST mode")
    if not re.fullmatch(r"[0-9]+\.[0-9]+", args.ast_python_min_version):
        parser.error("--ast-python-min-version must be major.minor")
    if tuple(map(int, args.ast_python_min_version.split("."))) < (3, 14):
        parser.error("Native AST worker requires Python 3.14+ minimum")
    if not args.ast_python_bin.is_file() or not os.access(args.ast_python_bin, os.X_OK):
        parser.error("AST Python binary missing/not executable")
    if args.ast_worker_script.is_symlink() or not args.ast_worker_script.is_file():
        parser.error("AST worker script missing/unsafe")
    try:
        args.expected_subjects = int(args.expected_subjects)
        args.progress_every = int(args.progress_every)
    except ValueError:
        parser.error("--expected-subjects and --progress-every must be integers")
    if args.expected_subjects < 1 or args.progress_every < 1:
        parser.error("Expected subjects and progress interval must be positive")
    if args.parse_failure_policy not in ("preserve_original_audited", "fail_closed"):
        parser.error("Invalid --parse-failure-policy")
    if args.mode not in ("inspect", "inventory-exclusions", "dry-run", "smoke", "subject", "all", "verify", "exclude-unparsed-files"):
        parser.error("Invalid --mode")
    if args.exclude_unparsed_files and args.mode not in ('all', 'verify', 'exclude-unparsed-files'):
        parser.error('--exclude-unparsed-files is supported only with --all, --verify, or by itself')
    if args.mode == 'exclude-unparsed-files' and not args.exclude_unparsed_files:
        parser.error('--exclude-unparsed-files mode requires the flag true')
    args.skip_dir = [x.casefold() for x in args.skip_dir]
    args.skip_glob = [x.casefold() for x in args.skip_glob]
    args.candidate_dir = [x.casefold() for x in (args.candidate_dir or [])]
    args.inventory_baseline_skip_dir = [x.casefold() for x in (args.inventory_baseline_skip_dir or [])]
    if args.mode == "inventory-exclusions" and not args.inventory_baseline_skip_dir:
        parser.error("--inventory-exclusions requires --inventory-baseline-skip-dir supplied by Shell")
    if args.mode == "inventory-exclusions" and not args.candidate_dir:
        parser.error("--inventory-exclusions requires --candidate-dir supplied by Shell")
    args.protected_comment_prefix = [x.casefold() for x in args.protected_comment_prefix]
    safe_subject(args.subject)
    if args.mode in ("all", "smoke", "subject") and (not args.report_dir.exists() or not args.report_dir.is_dir()):
        parser.error("Report directory must exist before preprocessing")
    return args


def _run_pipeline(args: argparse.Namespace) -> int:
    print(f"J04B START mode={args.mode} version={args.version} timestamp_chicago={chicago()}", flush=True)
    inventory, baseline_sha = repository_inventory(args)
    mapping = {item["subject"]: item for item in inventory}
    if args.mode == "inventory-exclusions":
        inventory_exclusions(args, inventory, baseline_sha)
        return 0
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
    if args.mode == 'exclude-unparsed-files':
        build_clone_input(args, inventory, baseline_sha)
        return 0
    if args.mode == "verify":
        for item in selected:
            _verify_subject(args, item, baseline_sha)
        print(f"PASS VERIFY_ALL subjects={len(selected)}", flush=True)
        if args.exclude_unparsed_files:
            verify_clone_input(args, inventory, baseline_sha)
        return 0
    for item in selected:
        prepare_subject(args, item, baseline_sha)
    print(f"PASS PREPROC_ALL subjects={len(selected)} destination={args.preproc_root}", flush=True)
    if args.mode == 'all' and args.exclude_unparsed_files:
        build_clone_input(args, inventory, baseline_sha)
    return 0


def main() -> int:
    args = parse_args()
    worker = AstWorkerClient(executable=args.ast_python_bin,
                             worker_script=args.ast_worker_script,
                             min_version=args.ast_python_min_version,
                             expected_version=args.ast_python_expected_version,
                             suppress_syntax_warnings=args.ast_suppress_syntax_warnings)
    try:
        args.ast_worker = worker.start()
        return _run_pipeline(args)
    finally:
        worker.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAIL J-04b: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(2)
