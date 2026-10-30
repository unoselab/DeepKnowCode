#!/usr/bin/env python3
"""Evaluate code clone classification on a fixed paired dataset."""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

def emit_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def targets(root,language):
    sub='data-java' if language=='java' else 'data-py'
    d=root/language/'data'/sub/'ground_truth_test_refactorability'
    if not d.is_dir(): raise FileNotFoundError(d)
    jobs=[];wanted=set()
    for fp in sorted(d.glob('*.csv')):
        project=fp.name.removeprefix('pred_').removesuffix('_with_refactorability.csv')
        with fp.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
        if not rows: continue
        for row in rows:
            for k in ['pair_left_func','pair_right_func']:
                if not row.get(k): raise ValueError(f'Missing {k}: {fp}')
                wanted.add(row[k].strip())
        jobs.append((project,rows))
    return jobs,wanted

def gt(a,b):
    x=a.split('_');y=b.split('_')
    if len(x)<2 or len(y)<2: raise ValueError(f'Unparseable IDs: {a} {b}')
    return int(x[1]==y[1])

def candidates(rq3,language):
    base=rq3/language
    if not base.is_dir():raise FileNotFoundError(base)
    # Only actual code corpora, not massive model/prediction artifacts.
    results=[]
    for root in [base/'code'/'Dataset-Preparation',base/'data']:
        if root.exists(): results.extend(root.rglob('data.jsonl'))
    return sorted(set(results))

def build_code_map(paths,wanted):
    found={}; conflicts=set();scanned=0; usable=0
    idkeys=('idx','id','index','func_id','function_id')
    codekeys=('func','code','source','source_code','function','body')
    for path in paths:
        with path.open(encoding='utf-8',errors='replace') as f:
            for line in f:
                scanned+=1
                try:d=json.loads(line)
                except (ValueError,TypeError):continue
                if not isinstance(d,dict):continue
                ix=next((str(d[k]).strip() for k in idkeys if k in d and d[k] is not None),'')
                if ix not in wanted:continue
                code=next((d[k] for k in codekeys if isinstance(d.get(k),str) and d[k].strip()),None)
                if not code: continue
                usable+=1
                if ix in found and found[ix]!=code:conflicts.add(ix)
                else:found[ix]=code
    return found,conflicts,scanned,usable

def preflight(args):
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    tasks=[];reports=[];bylang={}
    for lang in ('java','python'):
        gt_dir=args.data_root/'ground_truth'/lang
        code_path=args.data_root/'code'/f'{lang}.jsonl'
        if not gt_dir.is_dir() or not code_path.is_file():raise FileNotFoundError(f'Missing staged dataset: {gt_dir} / {code_path}; run data staging first')
        code_map={}
        with code_path.open(encoding='utf-8') as f:
            for line in f:
                d=json.loads(line)
                if d['id'] in code_map:raise ValueError(f'Duplicate staged code ID: {lang}/{d["id"]}')
                code_map[d['id']]=d['code']
        jobs=[];wanted=set()
        for fp in sorted(gt_dir.glob('*.csv')):
            project=fp.name.removeprefix('pred_').removesuffix('_with_refactorability.csv')
            with fp.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            jobs.append((project,rows))
            for row in rows:wanted.update((row['pair_left_func'].strip(),row['pair_right_func'].strip()))
        missing=wanted-set(code_map)
        for ix in sorted(missing):reports.append({'language':lang,'id':ix,'reason':'missing'})
        bylang[lang]={'projects':len(jobs),'requested_ids':len(wanted),'resolved_ids':len(wanted)-len(missing),'missing_ids':len(missing)}
        for project,rows in jobs:
            for i,row in enumerate(rows):
                a=row['pair_left_func'].strip();b=row['pair_right_func'].strip()
                tasks.append({'language':lang,'project':project,'row':i,'left_id':a,'right_id':b,'label':gt(a,b),'code_a':code_map.get(a),'code_b':code_map.get(b)})
        print(f'[A-06] {lang}: projects={len(jobs)} pairs={sum(map(lambda x:len(x[1]),jobs))} missing={len(missing)}',flush=True)
    emit_csv(out/'unresolved_ids.csv',reports,['language','id','reason'])
    (out/'preflight.json').write_text(json.dumps({'languages':bylang,'pairs':len(tasks)},indent=2)+'\n')
    if reports:raise RuntimeError(f'{len(reports)} missing IDs: see unresolved_ids.csv')
    p=out/'aligned_pairs.jsonl'
    with p.open('w',encoding='utf-8') as f:
        for t in tasks:f.write(json.dumps(t,ensure_ascii=False)+'\n')
    print('[A-06] Alignment complete:',len(tasks),'pairs',flush=True)

def window_ranges(token_count, capacity, overlap):
    """Return overlapping, exhaustive ranges over token positions."""
    if not 0 <= overlap < 1:
        raise ValueError('Window overlap must be in [0, 1).')
    if capacity < 1:
        raise ValueError('Window capacity must be positive.')
    if token_count <= capacity:
        return [(0, token_count)]
    stride = max(1, int(capacity * (1 - overlap)))
    last_start = token_count - capacity
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    ranges = [(start, start + capacity) for start in starts]
    if ranges[0][0] != 0 or ranges[-1][1] != token_count:
        raise AssertionError('Window coverage does not include both endpoints.')
    if any(ranges[i][1] < ranges[i + 1][0] for i in range(len(ranges) - 1)):
        raise AssertionError('Gap detected in sliding-window coverage.')
    return ranges


def code_token_offsets(tokenizer, source):
    """Map tokenizer token positions back to the unmodified source string."""
    if not tokenizer.is_fast:
        raise RuntimeError('Lossless source-text windows require a fast tokenizer.')
    enc = tokenizer(source, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc['input_ids']
    offsets = enc['offset_mapping']
    if len(ids) != len(offsets):
        raise RuntimeError('Inconsistent tokenizer offsets.')
    return offsets


def source_windows(source, offsets, capacity, overlap):
    chunks = []
    for start, end in window_ranges(len(offsets), capacity, overlap):
        # Slice ORIGINAL source text. Decoding partial byte/BPE tokens can corrupt Unicode.
        char_start = 0 if start == 0 else offsets[start][0]
        char_end = len(source) if end == len(offsets) else offsets[end][0]
        if not 0 <= char_start <= char_end <= len(source):
            raise RuntimeError('Invalid tokenizer text offsets.')
        chunks.append({'start': start, 'end': end, 'text': source[char_start:char_end]})
    return chunks


def allocate_code_budget(len_a, len_b, budget):
    """Keep the shorter fragment intact when its partner has room for a window."""
    minimum = 128
    if budget < minimum * 2:
        raise RuntimeError(f'Window budget is too small: {budget}')
    if len_a <= len_b and len_a <= budget - minimum:
        return max(1, len_a), budget - len_a
    if len_b < len_a and len_b <= budget - minimum:
        return budget - len_b, max(1, len_b)
    share = len_a / max(1, len_a + len_b)
    cap_a = max(minimum, min(budget - minimum, round(budget * share)))
    return cap_a, budget - cap_a


def iterate_chunks(plan, pair):
    for a in plan['chunks_a']:
        for b in plan['chunks_b']:
            edited = dict(pair, code_a=a['text'], code_b=b['text'])
            yield a, b, edited


def prepare_windows(pair, tokenizer, limit, overlap):
    """Plan full-coverage pair windows; never delete code to satisfy a limit."""
    if limit < 512:
        raise ValueError('Maximum tokens must be at least 512.')
    full_ids = tokenizer.encode(prompt(pair), add_special_tokens=False)
    original_length = len(full_ids)
    if original_length <= limit:
        return {
            'original_tokens': original_length,
            'windowed': False,
            'window_count': 1,
            'full_ids': full_ids,
        }

    offsets_a = code_token_offsets(tokenizer, pair['code_a'])
    offsets_b = code_token_offsets(tokenizer, pair['code_b'])
    empty = dict(pair, code_a='', code_b='')
    fixed_tokens = len(tokenizer.encode(prompt(empty), add_special_tokens=False))
    budget = limit - fixed_tokens - 128  # reserve for re-tokenization at boundaries
    if budget < 256:
        raise RuntimeError(f'Prompt format does not fit into {limit} tokens.')

    # Check every candidate's actual re-tokenized length before accepting a plan.
    for _ in range(48):
        cap_a, cap_b = allocate_code_budget(len(offsets_a), len(offsets_b), budget)
        chunks_a = source_windows(pair['code_a'], offsets_a, cap_a, overlap)
        chunks_b = source_windows(pair['code_b'], offsets_b, cap_b, overlap)
        plan = {
            'original_tokens': original_length,
            'windowed': True,
            'window_count': len(chunks_a) * len(chunks_b),
            'code_a_tokens': len(offsets_a),
            'code_b_tokens': len(offsets_b),
            'chunks_a': chunks_a,
            'chunks_b': chunks_b,
            'capacity_a': cap_a,
            'capacity_b': cap_b,
        }
        overflow = 0
        for _, _, edited in iterate_chunks(plan, pair):
            n = len(tokenizer.encode(prompt(edited), add_special_tokens=False))
            overflow = max(overflow, n - limit)
            if overflow:
                break
        if not overflow:
            return plan
        budget -= max(32, overflow + 32)
        if budget < 256:
            raise RuntimeError(f'Could not construct lossless windows within {limit} tokens.')
    raise RuntimeError('Window planner failed to converge.')


def window_inputs(pair, tokenizer, plan):
    """Yield tokenized prompts with source coverage metadata."""
    if not plan['windowed']:
        yield 0, plan['full_ids'], None, None
        return
    for i, (a, b, edited) in enumerate(iterate_chunks(plan, pair)):
        yield i, tokenizer.encode(prompt(edited), add_special_tokens=False), a, b


def audit_lengths(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, local_files_only=True)
    if not tok.is_fast:
        raise RuntimeError('A fast tokenizer is required for lossless source windows.')
    oversized = []
    max_tokens = 0
    count = 0
    total_windows = 0
    max_windows = 0
    with (args.output_dir / 'aligned_pairs.jsonl').open(encoding='utf-8') as f:
        for line in f:
            pair = json.loads(line)
            plan = prepare_windows(pair, tok, args.max_input_tokens, args.window_overlap)
            n = plan['original_tokens']
            max_tokens = max(max_tokens, n)
            count += 1
            total_windows += plan['window_count']
            max_windows = max(max_windows, plan['window_count'])
            if plan['windowed']:
                oversized.append({
                    'language': pair['language'], 'project': pair['project'],
                    'row': pair['row'], 'tokens': n,
                    'windows': plan['window_count'],
                    'code_a_tokens': plan['code_a_tokens'],
                    'code_b_tokens': plan['code_b_tokens'],
                })
            if count % 5000 == 0:
                print(f'[A-06] token audit {count} pairs, max={max_tokens}', flush=True)
    emit_csv(args.output_dir / 'oversized_inputs.csv', oversized,
             ['language', 'project', 'row', 'tokens', 'windows', 'code_a_tokens', 'code_b_tokens'])
    info = {
        'pairs': count, 'max_original_tokens': max_tokens,
        'window_limit': args.max_input_tokens,
        'window_overlap': args.window_overlap,
        'windowed_pairs': len(oversized),
        'total_evaluation_windows': total_windows,
        'max_windows_per_pair': max_windows,
        'aggregation': args.aggregation,
        'truncated_pairs': 0,
    }
    (args.output_dir / 'token_audit.json').write_text(json.dumps(info, indent=2) + '\n')
    print('[A-06] token audit:', info, flush=True)


def prompt(t):
    return ('Determine whether the following two code fragments are code clones. '
            'Answer YES if they are clones and NO otherwise.\n'
            f'Language: {t["language"]}\nCode A:\n{t["code_a"]}\n'
            f'Code B:\n{t["code_b"]}\nAnswer:')

def infer(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.manual_seed(2026)
    opts = {'revision': args.revision, 'local_files_only': True}
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **opts)
    if not tokenizer.is_fast:
        raise RuntimeError('A fast tokenizer is required for lossless source windows.')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.float16, low_cpu_mem_usage=True,
        device_map={'': 0}, **opts
    ).eval()
    choices = {}
    for label in ('YES', 'NO'):
        ids = tokenizer.encode(' ' + label, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f'Label {label} is not single token: {ids}')
        choices[label] = ids[0]

    inp = args.output_dir / 'aligned_pairs.jsonl'
    dest = args.output_dir / f'predictions-worker-{args.worker_index}.csv'
    window_dest = args.output_dir / f'window-predictions-worker-{args.worker_index}.csv'
    fields = [
        'language', 'project', 'row', 'left_id', 'right_id', 'true_label',
        'predicted_label', 'yes_logit', 'no_logit', 'worker', 'input_tokens',
        'original_input_tokens', 'truncated', 'windowed', 'window_count',
        'total_window_tokens', 'mean_logit_margin', 'max_logit_margin',
    ]
    window_fields = [
        'language', 'project', 'row', 'window_index', 'window_count',
        'a_token_start', 'a_token_end', 'b_token_start', 'b_token_end',
        'input_tokens', 'yes_logit', 'no_logit', 'logit_margin',
    ]
    print('[A-06] worker', args.worker_index, 'labels', choices, flush=True)
    with (inp.open(encoding='utf-8') as f,
          dest.open('w', newline='', encoding='utf-8') as o,
          window_dest.open('w', newline='', encoding='utf-8') as wo):
        writer = csv.DictWriter(o, fieldnames=fields)
        writer.writeheader()
        window_writer = csv.DictWriter(wo, fieldnames=window_fields)
        window_writer.writeheader()
        count = 0
        windowed_count = 0
        total_windows = 0
        for n, line in enumerate(f):
            if n % args.num_workers != args.worker_index:
                continue
            pair = json.loads(line)
            plan = prepare_windows(pair, tokenizer, args.max_input_tokens, args.window_overlap)
            if plan['windowed']:
                windowed_count += 1
            sum_yes = 0.0
            sum_no = 0.0
            max_margin = float('-inf')
            max_tokens = 0
            total_tokens = 0
            observed = 0
            for index, token_ids, a, b in window_inputs(pair, tokenizer, plan):
                if len(token_ids) > args.max_input_tokens:
                    raise RuntimeError('Window exceeds context limit')
                ids = torch.tensor([token_ids], device='cuda:0')
                with torch.inference_mode():
                    logits = model(input_ids=ids, use_cache=False).logits[0, -1]
                yes = float(logits[choices['YES']].item())
                no = float(logits[choices['NO']].item())
                del logits, ids
                sum_yes += yes
                sum_no += no
                max_margin = max(max_margin, yes - no)
                max_tokens = max(max_tokens, len(token_ids))
                total_tokens += len(token_ids)
                observed += 1
                if plan['windowed']:
                    window_writer.writerow({
                        'language': pair['language'], 'project': pair['project'],
                        'row': pair['row'], 'window_index': index,
                        'window_count': plan['window_count'],
                        'a_token_start': a['start'], 'a_token_end': a['end'],
                        'b_token_start': b['start'], 'b_token_end': b['end'],
                        'input_tokens': len(token_ids), 'yes_logit': yes,
                        'no_logit': no, 'logit_margin': yes - no,
                    })
            if observed != plan['window_count']:
                raise AssertionError('Window count mismatch')
            avg_yes = sum_yes / observed
            avg_no = sum_no / observed
            mean_margin = avg_yes - avg_no
            if args.aggregation != 'mean-margin':
                raise ValueError(f'Unknown window aggregation: {args.aggregation}')
            predicted = int(mean_margin > 0)
            writer.writerow({
                'language': pair['language'], 'project': pair['project'],
                'row': pair['row'], 'left_id': pair['left_id'],
                'right_id': pair['right_id'], 'true_label': pair['label'],
                'predicted_label': predicted, 'yes_logit': avg_yes,
                'no_logit': avg_no, 'worker': args.worker_index,
                'input_tokens': max_tokens,
                'original_input_tokens': plan['original_tokens'],
                'truncated': 'false', 'windowed': str(plan['windowed']).lower(),
                'window_count': observed, 'total_window_tokens': total_tokens,
                'mean_logit_margin': mean_margin, 'max_logit_margin': max_margin,
            })
            count += 1
            total_windows += observed
            if count % args.progress_every == 0:
                o.flush()
                wo.flush()
                print(f'[A-06] worker {args.worker_index}: {count} pairs, {total_windows} windows', flush=True)
        print(f'[A-06] worker {args.worker_index} complete: '
              f'{count} pairs, {windowed_count} windowed pairs, '
              f'{total_windows} windows; {dest}', flush=True)


def confusion_metrics(items):
    tp = sum(r['true_label'] == '1' and r['predicted_label'] == '1' for r in items)
    fp = sum(r['true_label'] == '0' and r['predicted_label'] == '1' for r in items)
    fn = sum(r['true_label'] == '1' and r['predicted_label'] == '0' for r in items)
    tn = len(items) - tp - fp - fn
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': tp / (tp + fp) if tp + fp else 0,
        'recall': tp / (tp + fn) if tp + fn else 0,
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
    }


def summarize(args):
    with (args.output_dir / 'aligned_pairs.jsonl').open(encoding='utf-8') as f:
        expected = sum(1 for _ in f)
    rows = []
    keys = set()
    for i in range(args.num_workers):
        path = args.output_dir / f'predictions-worker-{i}.csv'
        with path.open(newline='', encoding='utf-8') as f:
            for record in csv.DictReader(f):
                key = (record['language'], record['project'], record['row'])
                if key in keys:
                    raise RuntimeError(f'Duplicate pair: {key}')
                keys.add(key)
                rows.append(record)
    if len(rows) != expected:
        raise RuntimeError(f'Expected {expected} pair rows, got {len(rows)}')

    grouped = defaultdict(list)
    for record in rows:
        grouped[(record['language'], record['project'])].append(record)
    all_metrics = []
    unwindowed_metrics = []
    incomplete = 0
    for (language, project), records in sorted(grouped.items()):
        valid = [r for r in records if r['predicted_label'] in ('0', '1')]
        if len(valid) != len(records):
            incomplete += 1
        all_metrics.append({
            'language': language, 'project': project,
            'n_expected': len(records), 'n_scored': len(valid),
            **confusion_metrics(valid),
            'complete': len(records) == len(valid),
        })
        restricted = [r for r in valid if r['windowed'] == 'false']
        unwindowed_metrics.append({
            'language': language, 'project': project,
            'n_original': len(records), 'n_excluded_windowed': len(valid) - len(restricted),
            'n_scored': len(restricted), **confusion_metrics(restricted),
        })

    emit_csv(args.output_dir / 'project_metrics.csv', all_metrics,
             ['language', 'project', 'n_expected', 'n_scored', 'tp', 'fp',
              'fn', 'tn', 'precision', 'recall', 'f1', 'complete'])
    emit_csv(args.output_dir / 'project_metrics_unwindowed.csv', unwindowed_metrics,
             ['language', 'project', 'n_original', 'n_excluded_windowed',
              'n_scored', 'tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1'])
    status = 'COMPLETE' if not incomplete else 'INCOMPLETE_DO_NOT_COMPARE'
    summary = {
        'status': status,
        'expected_pairs': expected,
        'scored_pairs': sum(int(r['n_scored']) for r in all_metrics),
        'projects': len(all_metrics),
        'incomplete_projects': incomplete,
        'model': args.model_id,
        'revision': args.revision,
        'windowing': 'sliding-window',
        'window_limit': args.max_input_tokens,
        'window_overlap': args.window_overlap,
        'aggregation': args.aggregation,
        'windowed_pairs': sum(r['windowed'] == 'true' for r in rows),
        'truncated_pairs': sum(r['truncated'] == 'true' for r in rows),
        'total_windows': sum(int(r['window_count']) for r in rows),
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('[A-06]', status, 'projects=', len(all_metrics), 'expected_pairs=', expected,
          'windowed_pairs=', summary['windowed_pairs'], flush=True)
    if incomplete:
        raise SystemExit(4)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['preflight','audit-lengths','infer','summarize'])
    ap.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parents[2]/'data'/'evaluation')
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--model-id',default='bigcode/starcoder2-7b')
    ap.add_argument('--revision',default='bb9afde76d7945da5745592525db122d4d729eb1')
    ap.add_argument('--worker-index',type=int,default=0)
    ap.add_argument('--num-workers',type=int,default=3)
    ap.add_argument('--max-input-tokens',type=int,default=12000)
    ap.add_argument('--window-overlap',type=float,default=0.20)
    ap.add_argument('--aggregation',choices=['mean-margin'],default='mean-margin')
    ap.add_argument('--progress-every',type=int,default=100)
    args=ap.parse_args()
    if not 0 <= args.window_overlap < 1:
        ap.error('--window-overlap must be between 0 (inclusive) and 1 (exclusive)')
    if args.command=='preflight':preflight(args)
    elif args.command=='audit-lengths':audit_lengths(args)
    elif args.command=='infer':infer(args)
    else:summarize(args)
if __name__=='__main__':main()
