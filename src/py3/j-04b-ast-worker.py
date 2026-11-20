#!/usr/bin/env python3
"""J-04b v7 isolated native-AST parser worker.

Only the external AST process executes ast.parse for the v7 production path.
One persistent JSON-lines protocol, one outstanding request at a time, no writes
of experiment datasets, and no dynamic execution of submitted source code.
All worker configuration must be passed from the Shell-owned main configuration.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import warnings

PROTOCOL = "j04b-ast-v1"


def suites(tree):
    for owner in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            value = getattr(owner, field, None)
            if isinstance(value, list) and value and all(isinstance(x, ast.stmt) for x in value):
                yield (isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                          ast.AsyncFunctionDef)) and field == "body", value)


def equivalent(original, actual, replacements):
    """Iterative structural check; ignore only explicitly marked MASK/PASS stmt nodes."""
    pending = [(original, actual)]
    while pending:
        old, new = pending.pop()
        if isinstance(old, ast.AST):
            if not isinstance(new, ast.AST) or type(old) is not type(new):
                return False
            old_fields, new_fields = list(ast.iter_fields(old)), list(ast.iter_fields(new))
            if [f for f, _ in old_fields] != [f for f, _ in new_fields]:
                return False
            pending.extend((v1, v2) for (_, v1), (_, v2) in zip(old_fields, new_fields))
        elif isinstance(old, list):
            if not isinstance(new, list):
                return False
            selected = [(node, replacements.get(id(node))) for node in old]
            selected = [(node, action) for node, action in selected if action != "MASK"]
            if len(selected) != len(new):
                return False
            for (node, action), actual_node in zip(selected, new):
                if action == "PASS":
                    if not isinstance(actual_node, ast.Pass):
                        return False
                else:
                    pending.append((node, actual_node))
        elif old != new or type(old) is not type(new):
            return False
    return True


def get_suites(tree):
    data, node_map = [], {}
    for suite_id, (doc_holder, statements) in enumerate(suites(tree)):
        candidates = []
        for index, stmt in enumerate(statements):
            node_map[(suite_id, index)] = stmt
            if not (isinstance(stmt, ast.Expr) and
                    isinstance(stmt.value, ast.Constant) and
                    isinstance(stmt.value.value, str)):
                continue
            value = stmt.value
            candidates.append({
                "index": index,
                "start": [value.lineno, value.col_offset],
                "end": [value.end_lineno, value.end_col_offset],
                "stmt_line": stmt.lineno, "stmt_end_line": stmt.end_lineno,
            })
        if candidates:
            data.append({"id": suite_id, "doc_holder": doc_holder,
                         "statement_lines": [s.lineno for s in statements],
                         "candidates": candidates})
    return data, node_map


def respond(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--min-version", required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--suppress-syntax-warnings", required=True,
                        choices=("true", "false"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]+\.[0-9]+", args.min_version):
        parser.error("invalid --min-version")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", args.expected_version):
        parser.error("invalid --expected-version")
    minimum = tuple(int(x) for x in args.min_version.split("."))
    if tuple(sys.version_info[:2]) < minimum:
        parser.error(f"Python >= {args.min_version} required")
    if sys.version.split()[0] != args.expected_version:
        parser.error(f"expected Python {args.expected_version}, found {sys.version.split()[0]}")
    if args.suppress_syntax_warnings == "true":
        warnings.filterwarnings("ignore", category=SyntaxWarning)
    return args


def main():
    parse_args()
    respond({"ok": True, "kind": "hello", "protocol": PROTOCOL,
             "version": sys.version.split()[0],
             "executable": os.path.realpath(sys.executable)})
    current_ast, current_nodes = None, None
    for line in sys.stdin:
        try:
            message = json.loads(line)
            operation = message.get("op")
            if operation == "analyze":
                current_ast, current_nodes = None, None
                if not isinstance(message.get("text"), str):
                    raise ValueError("analyze requires text")
                current_ast = ast.parse(message["text"], type_comments=True)
                metadata, current_nodes = get_suites(current_ast)
                respond({"ok": True, "kind": "analysis", "suites": metadata})
            elif operation == "verify":
                if current_ast is None or current_nodes is None:
                    raise ValueError("verify without preceding analysis")
                if not isinstance(message.get("text"), str):
                    raise ValueError("verify requires text")
                decisions = message.get("decisions")
                if not isinstance(decisions, list):
                    raise ValueError("verify requires decisions list")
                replacements = {}
                for entry in decisions:
                    if (not isinstance(entry, list) or len(entry) != 3 or
                            not all(type(x) is int for x in entry[:2]) or
                            entry[2] not in ("MASK", "PASS")):
                        raise ValueError("invalid replacement action")
                    node = current_nodes[(entry[0], entry[1])]
                    if id(node) in replacements:
                        raise ValueError("duplicate replacement action")
                    replacements[id(node)] = entry[2]
                actual = ast.parse(message["text"], type_comments=True)
                if not equivalent(current_ast, actual, replacements):
                    respond({"ok": False, "kind": "source", "error":
                             "AST mismatch outside intended documentation masking"})
                else:
                    respond({"ok": True, "kind": "verified"})
                current_ast, current_nodes = None, None
            else:
                raise ValueError(f"unsupported worker operation: {operation!r}")
        except (SyntaxError, IndentationError, RecursionError) as exc:
            current_ast, current_nodes = None, None
            respond({"ok": False, "kind": "source", "error":
                     f"{type(exc).__name__}:{str(exc)[:300]}"})
        except Exception as exc:
            current_ast, current_nodes = None, None
            respond({"ok": False, "kind": "protocol", "error":
                     f"{type(exc).__name__}:{str(exc)[:300]}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
