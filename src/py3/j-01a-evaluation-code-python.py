#!/usr/bin/env python3
"""J-01a v2: derive a Java-schema function collection AND a lossless pair map.

No training, model loading, truncation, or source modification. Fail closed.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def records(path):
    with path.open(encoding='utf-8') as f:
        for n, raw in enumerate(f, 1):
            if not raw.strip():
                raise ValueError(f'{path}:{n}: empty line')
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f'{path}:{n}: invalid JSON: {e}') from e
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{n}: must be JSON object')
            yield n, row


def text_value(obj, field, where):
    val = obj.get(field)
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f'{where}: expected nonblank string {field!r}, got {type(val).__name__}')
    return val


def get_function(side, side_key, n, code_field, id_field, id_policy, id_prefix):
    where = f'line {n}, {side_key}'
    if isinstance(side, str):
        code, original_id = side, None
    elif isinstance(side, dict):
        code = text_value(side, code_field, where)
        original_id = side.get(id_field) if id_field else None
    else:
        raise ValueError(f'{where}: expected object or code string, got {type(side).__name__}')
    if not code.strip():
        raise ValueError(f'{where}: empty code')
    if id_policy == 'source':
        if isinstance(original_id, bool) or not isinstance(original_id, (str, int)) or not str(original_id).strip():
            raise ValueError(f'{where}: source ID unavailable; select content-sha256 policy explicitly')
        fid = str(original_id)
    else:
        fid = id_prefix + hashlib.sha256(code.encode('utf-8')).hexdigest()
    return fid, code


def normalize_label(v, n):
    if type(v) is not int or v not in (0, 1):
        raise ValueError(f'line {n}: gold_label must be integer 0 or 1 (got {v!r})')
    return v


def args_parser():
    p = argparse.ArgumentParser()
    for k in ('mode','source','reference','output','pairs-output','report',
              'expected-pairs','left-field','right-field','code-field',
              'source-id-field','id-policy','id-prefix','label-field','pair-id-field',
              'expected-reference-keys','strict-pair-id-unique'):
        p.add_argument('--' + k, required=True)
    return p


def main():
    a = args_parser().parse_args()
    if a.mode not in ('inspect','convert'):
        raise ValueError('mode must be inspect or convert')
    if a.id_policy not in ('source', 'content-sha256'):
        raise ValueError('id-policy must be source or content-sha256')
    if a.strict_pair_id_unique not in ('true', 'false'):
        raise ValueError('strict-pair-id-unique must be true or false')
    if a.id_policy == 'content-sha256' and not a.id_prefix:
        raise ValueError('content hash ID requires nonblank ID prefix')
    expected = int(a.expected_pairs)
    if expected <= 0:
        raise ValueError('expected-pairs must be >0')
    src, ref, out, pairs, report = [Path(s).expanduser().resolve() for s in
                                  (a.source,a.reference,a.output,a.pairs_output,a.report)]
    paths = (src, ref, out, pairs, report)
    if len(set(paths)) != len(paths):
        raise ValueError('source, reference, outputs and report paths must be different')
    for x in (src, ref):
        if not x.is_file():
            raise FileNotFoundError(x)
    if a.mode == 'convert':
        for x in (out, pairs, report):
            if x.exists():
                raise FileExistsError(f'No-overwrite policy: {x}')
    try:
        _, java_first = next(records(ref))
        _, first = next(records(src))
    except StopIteration:
        raise ValueError('empty source or reference JSONL')
    expected_keys = json.loads(a.expected_reference_keys)
    if not isinstance(expected_keys, list) or not all(isinstance(x,str) for x in expected_keys):
        raise ValueError('expected-reference-keys must be a JSON array of strings')
    if set(java_first) != set(expected_keys) or len(expected_keys) != 2 or set(expected_keys) != {'id','code'}:
        raise ValueError(f'Java schema mismatch: actual={sorted(java_first)} expected={sorted(expected_keys)}')
    for k in (a.left_field,a.right_field,a.label_field,a.pair_id_field):
        if k not in first:
            raise ValueError(f'Python source missing field {k!r}')
    info = {'experiment':'j-01a','version':'v2','source':str(src),'reference':str(ref),
            'source_sha256':sha(src),'reference_sha256':sha(ref),'source_first_row_keys':sorted(first),
            'reference_first_row_keys':sorted(java_first),'expected_pairs':expected,
            'mode':a.mode,'id_policy':a.id_policy,'source_id_field':a.source_id_field,
            'code_field':a.code_field,'pair_id_field':a.pair_id_field,
            'label_field':a.label_field,'left_field':a.left_field,'right_field':a.right_field}
    if a.mode == 'inspect':
        info['status'] = 'INSPECTED'
        for key in (a.left_field,a.right_field):
            v = first[key]
            info[key + '_type'] = type(v).__name__
            info[key + '_keys'] = sorted(v) if isinstance(v,dict) else []
        print(json.dumps(info,indent=2,ensure_ascii=False))
        return
    function_map = {}  # canonical ID -> exact code
    pair_ids = set()
    pair_rows = []
    label_count = collections.Counter()
    for n, row in records(src):
        if set(row) != set(first):
            raise ValueError(f'line {n}: source schema drift')
        left_id, left_code = get_function(row[a.left_field], a.left_field, n,
                                          a.code_field,a.source_id_field,a.id_policy,a.id_prefix)
        right_id, right_code = get_function(row[a.right_field], a.right_field,n,
                                            a.code_field,a.source_id_field,a.id_policy,a.id_prefix)
        for fid, code in ((left_id,left_code),(right_id,right_code)):
            previous = function_map.get(fid)
            if previous is not None and previous != code:
                raise ValueError(f'line {n}: ID collision with different code for {fid!r}')
            function_map[fid] = code
        pid = row[a.pair_id_field]
        if isinstance(pid,bool) or not isinstance(pid,(str,int)) or not str(pid).strip():
            raise ValueError(f'line {n}: invalid pair ID')
        if a.strict_pair_id_unique == 'true' and str(pid) in pair_ids:
            raise ValueError(f'line {n}: duplicate pair ID {pid!r}')
        pair_ids.add(str(pid))
        label = normalize_label(row[a.label_field], n)
        label_count[label] += 1
        # Retain all original pair metadata, not just labels and references.
        pair_rows.append({'left_id':left_id,'right_id':right_id,
                          'source_row':row})
    if len(pair_rows) != expected:
        raise ValueError(f'Pair count mismatch actual={len(pair_rows)} expected={expected}')
    if not function_map:
        raise ValueError('no functions found')
    # Create both files in private temporary locations; no output files until all validation passes.
    for p in (out,pairs,report):
        p.parent.mkdir(parents=True,exist_ok=True)
    temps = []
    created = []
    try:
        for dest in (out,pairs):
            fd, tmpname = tempfile.mkstemp(prefix='.j-01a-v2-',suffix='.tmp',dir=dest.parent)
            temps.append(Path(tmpname))
            with os.fdopen(fd,'w',encoding='utf-8') as stream:
                if dest == out:
                    for fid, code in function_map.items():
                        stream.write(json.dumps({'id':fid,'code':code},ensure_ascii=False,separators=(',',':'))+'\n')
                else:
                    for row in pair_rows:
                        stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
                stream.flush(); os.fsync(stream.fileno())
        # Independently verify generated content and pair relations.
        verify_functions = {}
        for _, row in records(temps[0]):
            if set(row) != {'id','code'} or row['id'] in verify_functions:
                raise ValueError('generated function schema or duplicate ID failure')
            verify_functions[row['id']] = row['code']
        if verify_functions != function_map:
            raise ValueError('function verification mismatch')
        verified_pair_count = 0
        for (_, row),(_, original) in zip(records(temps[1]),records(src)):
            verified_pair_count += 1
            if row.get('source_row') != original:
                raise ValueError('pair metadata changed')
            if row.get('left_id') not in verify_functions or row.get('right_id') not in verify_functions:
                raise ValueError('pair references nonexistent function')
        if verified_pair_count != expected:
            raise ValueError('pair verification count mismatch')
        info.update(status='PASS',pair_count=len(pair_rows),function_count=len(function_map),
                    label_distribution={str(k):v for k,v in sorted(label_count.items())},
                    output=str(out),pairs_output=str(pairs),
                    output_sha256=sha(temps[0]),pairs_output_sha256=sha(temps[1]),
                    no_truncation=True,no_resampling=True,
                    notice='Function collection alone does not preserve pair connectivity/labels: use the pair sidecar for evaluation')
        # Fail-closed exclusive create (never overwrite). On error, remove only created outputs.
        for dest,tmp in zip((out,pairs),temps):
            with dest.open('xb') as w, tmp.open('rb') as r:
                created.append(dest)
                for part in iter(lambda:r.read(1024*1024),b''):
                    w.write(part)
        with report.open('x',encoding='utf-8') as w:
            created.append(report)
            json.dump(info,w,indent=2,ensure_ascii=False)
        print(json.dumps(info,indent=2,ensure_ascii=False))
    except Exception:
        for p in created:
            p.unlink(missing_ok=True)
        raise
    finally:
        for p in temps:
            p.unlink(missing_ok=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'J01A_V2_FAIL:{type(exc).__name__}:{exc}',file=sys.stderr)
        sys.exit(1)
