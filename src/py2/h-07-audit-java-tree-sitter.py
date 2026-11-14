#!/usr/bin/env python3
"""H-07: auditable Java Tree-sitter parse census. Does not assign clone types."""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def put_line(f, item):
    f.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")


def required_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-map', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-code-rows', type=int, required=True)
    p.add_argument('--expected-code-sha256', required=True)
    p.add_argument('--parser-package-version', required=True)
    p.add_argument('--grammar-package-version', required=True)
    p.add_argument('--timezone', required=True)
    p.add_argument('--max-examples', type=int, required=True)
    p.add_argument('--max-parse-failure-rate', type=float, required=True)
    a = p.parse_args()
    if a.expected_code_rows < 1 or a.max_examples < 0 or not 0 <= a.max_parse_failure_rate <= 1:
        p.error('Invalid expected rows, max examples or failure rate')
    if a.timezone != 'America/Chicago':
        p.error('Only America/Chicago configured for this experiment')
    if len(a.expected_code_sha256) != 64:
        p.error('Invalid expected sha256')
    return a


def collect_leaves(root, source):
    leaves = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in ('comment', 'line_comment', 'block_comment'):
            continue
        if n.child_count:
            stack.extend(reversed(n.children))
        else:
            leaves.append((n.type, source[n.start_byte:n.end_byte].decode('utf-8', errors='replace')))
    return leaves


def tokens_for_normalization(leaves):
    consistent = {}
    blind = []
    aligned = []
    for kind, content in leaves:
        if kind in ('identifier', 'type_identifier'):
            key = (kind, content)
            if key not in consistent:
                consistent[key] = f'{kind}:{len(consistent)}'
            aligned.append(consistent[key])
            blind.append(f'{kind}:*')
        elif kind in ('decimal_integer_literal', 'hex_integer_literal', 'octal_integer_literal',
                      'binary_integer_literal', 'decimal_floating_point_literal',
                      'hex_floating_point_literal', 'string_literal', 'character_literal',
                      'text_block') or kind.endswith('_literal'):
            aligned.append(f'{kind}:LITERAL')
            blind.append(f'{kind}:LITERAL')
        else:
            aligned.append(kind + ':' + content)
            blind.append(kind + ':' + content)
    return aligned, blind


def signature(leaves, mode):
    if mode == 'syntax':
        tokens = [k + ':' + t for k, t in leaves]
    else:
        consistent, blind = tokens_for_normalization(leaves)
        tokens = consistent if mode == 'consistent' else blind
    return digest(json.dumps(tokens, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


# Full input Java Compilation Units and single method/constructor snippets are
# distinct evaluation units. Do not silently pick one method from a whole class.
DECLARATION_TYPES = {'method_declaration', 'constructor_declaration', 'compact_constructor_declaration'}
CLASS_TYPES = {'class_declaration', 'interface_declaration', 'enum_declaration',
               'record_declaration', 'annotation_type_declaration'}


def has_parse_anomalies(node):
    """Treat missing nodes as errors even when a caller supplies an unusual CST."""
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == 'ERROR' or n.is_missing:
            return True
        stack.extend(n.children)
    return False


def pre_order_named(node):
    stack = [node]
    while stack:
        n = stack.pop()
        if n.is_named and n.type not in ('line_comment', 'block_comment', 'comment'):
            yield n.type
        stack.extend(reversed(n.children))


def direct_targets(root):
    """Only top-level input declarations, never anonymous/nested methods."""
    return [n for n in root.children if n.type in DECLARATION_TYPES]


def direct_class_targets(root):
    """Only direct children of the synthetic wrapper class body."""
    classes = [n for n in root.children if n.type == 'class_declaration']
    if len(classes) != 1:
        return []
    body = classes[0].child_by_field_name('body')
    return [n for n in body.children if n.type in DECLARATION_TYPES] if body is not None else []


def features(target, raw, mode, kind, eligible, nested_count):
    leaves = collect_leaves(target, raw)
    if not leaves:
        return None
    total = 0
    for _ in pre_order_named(target):
        total += 1
    named = list(pre_order_named(target))
    return {
        'parse_status': 'PASS', 'parse_mode': mode,
        'record_kind': kind, 'function_level_eligible': eligible,
        'method_node_type': target.type if eligible else None,
        'nested_method_count': nested_count,
        'target_start_byte': target.start_byte,
        'target_end_byte': target.end_byte,
        'ast_node_count': total, 'syntax_token_count': len(leaves),
        'type1_syntax_sha256': signature(leaves, 'syntax'),
        'type2_consistent_sha256': signature(leaves, 'consistent'),
        'type2_blind_sha256': signature(leaves, 'blind'),
        'ast_named_preorder_sha256': digest(json.dumps(named, separators=(',', ':')).encode()),
        'ast_named_node_count': len(named),
    }


def parse_strategy(parser, code):
    data = code.encode('utf-8')
    failures = []
    direct = parser.parse(data)
    root = direct.root_node
    if not root.has_error and not has_parse_anomalies(root):
        # A syntactically complete compilation unit may contain many methods;
        # it is valid Java but must not be reported as a function-level pair.
        units = [n for n in root.children if n.type in CLASS_TYPES]
        if units:
            f = features(root, data, 'compilation_unit', 'FULL_COMPILATION_UNIT', False,
                         sum(1 for n in walk(root) if n.type in DECLARATION_TYPES))
            if f:
                return f
        top = direct_targets(root)
        if len(top) == 1 and len(root.named_children) == 1:
            candidate = top[0]
            if candidate.child_by_field_name('body') is not None:
                f = features(candidate, data, 'direct_declaration', 'FUNCTION_OR_CONSTRUCTOR', True,
                             sum(1 for n in walk(candidate) if n.type in DECLARATION_TYPES)-1)
                if f:
                    return f
        failures.append({'mode': 'compilation_unit', 'record_kind': 'not_single_declaration',
                         'top_level_declarations': len(top), 'top_level_types': [n.type for n in root.named_children[:8]]})
    else:
        failures.append({'mode': 'compilation_unit', 'has_error': root.has_error,
                         'has_missing_or_error': has_parse_anomalies(root)})

    prefix = b'class H07SyntheticWrapper {\n'
    suffix = b'\n}\n'
    wrapped = prefix + data + suffix
    tree = parser.parse(wrapped)
    wr = tree.root_node
    if wr.has_error or has_parse_anomalies(wr):
        failures.append({'mode': 'method_in_class_wrapper', 'has_error': wr.has_error,
                         'has_missing_or_error': has_parse_anomalies(wr)})
    else:
        candidates = direct_class_targets(wr)
        if len(candidates) == 1:
            target = candidates[0]
            # Exactly one direct declaration inside wrapper, covering the input
            # except whitespace/comments; nested declarations remain part of AST.
            if target.child_by_field_name('body') is not None:
                other = [n for n in wr.children if n.type in CLASS_TYPES]
                if len(other) == 1:
                    start = target.start_byte - len(prefix)
                    end = target.end_byte - len(prefix)
                    if start >= 0 and end <= len(data):
                        before = data[:start].strip()
                        after = data[end:].strip()
                        if not before and not after:
                            f = features(target, wrapped, 'method_in_class_wrapper',
                                         'FUNCTION_OR_CONSTRUCTOR', True,
                                         sum(1 for n in walk(target) if n.type in DECLARATION_TYPES)-1)
                            if f:
                                f['target_start_byte'] = start
                                f['target_end_byte'] = end
                                return f
                        failures.append({'mode': 'method_in_class_wrapper', 'outside_target_nonwhitespace': True})
            failures.append({'mode': 'method_in_class_wrapper', 'missing_method_body': True})
        else:
            failures.append({'mode': 'method_in_class_wrapper', 'direct_declaration_count': len(candidates)})
    return {'parse_status': 'FAIL', 'parse_mode': 'UNRESOLVED',
            'record_kind': 'UNRESOLVED', 'function_level_eligible': False,
            'parse_failure_reasons': failures}


def walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def main():
    a = required_args()
    import importlib.metadata
    from tree_sitter import Language, Parser
    import tree_sitter_java

    ts_version = importlib.metadata.version('tree-sitter')
    java_version = importlib.metadata.version('tree-sitter-java')
    if ts_version != a.parser_package_version or java_version != a.grammar_package_version:
        raise RuntimeError(f'Unexpected parser versions {ts_version} {java_version}')
    if not a.code_map.is_file():
        raise FileNotFoundError(a.code_map)
    if not a.output.is_dir() or any(a.output.iterdir()):
        raise FileExistsError(f'Output must be a new empty directory: {a.output}')
    actual_sha = digest(a.code_map.read_bytes())
    if actual_sha != a.expected_code_sha256:
        raise RuntimeError(f'Code map sha256 mismatch: expected={a.expected_code_sha256} actual={actual_sha}')
    parser = Parser(Language(tree_sitter_java.language()))
    seen = set()
    counts = Counter()
    examples = []
    records_path = a.output / 'function_parse_records.jsonl'
    failures_path = a.output / 'parse_failures.jsonl'
    with records_path.open('x', encoding='utf-8') as records, failures_path.open('x', encoding='utf-8') as bad:
        with a.code_map.open(encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    raise RuntimeError(f'Blank JSONL line {lineno}')
                item = json.loads(line)
                if not isinstance(item, dict) or not isinstance(item.get('code'), str) or not str(item.get('id', '')).strip():
                    raise RuntimeError(f'Invalid code schema at line {lineno}')
                fid = str(item['id'])
                if fid in seen:
                    raise RuntimeError(f'Duplicate code ID {fid}')
                seen.add(fid)
                raw = item['code']
                result = {
                    'id': fid, 'code_sha256': digest(raw.encode('utf-8')),
                    'code_bytes': len(raw.encode('utf-8')),
                    **(parse_strategy(parser, raw) if raw.strip() else {
                        'parse_status': 'FAIL', 'parse_mode': 'UNRESOLVED',
                        'record_kind': 'UNRESOLVED', 'function_level_eligible': False,
                        'parse_failure_reasons': [{'empty_code': True}]}),
                }
                put_line(records, result)
                counts[result['parse_status']] += 1
                counts['mode_' + result['parse_mode']] += 1
                counts['kind_' + result.get('record_kind', 'UNRESOLVED')] += 1
                if result.get('function_level_eligible'):
                    counts['function_level_eligible'] += 1
                if result.get('nested_method_count', 0) > 0:
                    counts['with_nested_declarations'] += 1
                if result['parse_status'] == 'FAIL':
                    put_line(bad, result)
                    if len(examples) < a.max_examples:
                        examples.append({'id': fid, 'reasons': result['parse_failure_reasons']})
                if lineno % 5000 == 0:
                    print(f'H07_PROGRESS rows={lineno} pass={counts["PASS"]} fail={counts["FAIL"]}', flush=True)
    if len(seen) != a.expected_code_rows:
        raise RuntimeError(f'Unexpected code rows {len(seen)} expected {a.expected_code_rows}')
    rate = counts['FAIL'] / len(seen)
    status = 'H07_V2_PARSE_AUDIT_PASS' if rate <= a.max_parse_failure_rate else 'H07_V2_PARSE_QUALITY_GATE_FAIL'
    summary = {
        'experiment': 'h-07', 'version': 'v2', 'timezone': a.timezone,
        'at_chicago': datetime.now(ZoneInfo(a.timezone)).isoformat(),
        'status': status, 'code_map': str(a.code_map.resolve()),
        'code_map_sha256': actual_sha, 'code_rows': len(seen),
        'parse_pass': counts['PASS'], 'parse_fail': counts['FAIL'],
        'failure_rate': rate, 'max_failure_rate': a.max_parse_failure_rate,
        'parse_modes': {k.removeprefix('mode_'):v for k,v in counts.items() if k.startswith('mode_')},
        'record_kinds': {k.removeprefix('kind_'):v for k,v in counts.items() if k.startswith('kind_')},
        'function_level_eligible': counts['function_level_eligible'],
        'with_nested_declarations': counts['with_nested_declarations'],
        'parser_package_version': ts_version, 'grammar_package_version': java_version,
        'examples': examples, 'clone_type_certified': False,
        'notes': ['No clone type labels are produced.',
                  'Syntax hashes are parser-specific and include method name.',
                  'Full compilation units are parse-successful but not function-level eligible.',
                  'Named AST preorder hash alone is structural, not clone-type evidence.',
                  'Consistent/blind hashes are candidate normalization signatures, not independently verified clone labels.',
                  'Java fragments needing unsupported wrappers are reported as failures.'],
        'output_sha256': {p.name: digest(p.read_bytes()) for p in [records_path, failures_path]},
    }
    (a.output / 'parse_audit.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (a.output / 'manifest.json').write_text(json.dumps({
        'experiment': 'h-07', 'version': 'v2', 'input_sha256': actual_sha,
        'parse_audit_sha256': digest((a.output / 'parse_audit.json').read_bytes()),
        'records_sha256': summary['output_sha256'], 'status': status,
        'at_chicago': summary['at_chicago'],
    }, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('status','code_rows','parse_pass','parse_fail','failure_rate','parse_modes','record_kinds','function_level_eligible','with_nested_declarations')}, indent=2), flush=True)
    if status != 'H07_V2_PARSE_AUDIT_PASS':
        return 4
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f'H07_FATAL {type(e).__name__}: {e}', file=sys.stderr, flush=True)
        sys.exit(2)
