#!/usr/bin/env python3
"""H-08 v1: Java pair-level structural clone-type candidate audit (not gold)."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from zoneinfo import ZoneInfo


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def args():
    p = argparse.ArgumentParser(description=__doc__)
    for opt in ('--code-map','--h07-records','--h07-audit','--ground-truth-dir','--output','--expected-code-sha256','--expected-h07-records-sha256','--timezone','--tree-sitter-version','--tree-sitter-java-version','--thresholds','--primary-threshold'):
        p.add_argument(opt, required=True)
    for opt in ('--expected-code-rows','--expected-csv-files','--expected-pairs','--expected-positive','--expected-negative','--gram-size','--sample-per-type','--progress-interval','--max-ast-comparison-cells'):
        p.add_argument(opt, required=True, type=int)
    a = p.parse_args()
    for attr in ('code_map','h07_records','h07_audit','ground_truth_dir','output'):
        setattr(a, attr, Path(getattr(a, attr)).expanduser().resolve())
    a.thresholds = tuple(float(s) for s in a.thresholds.split(','))
    a.primary_threshold = float(a.primary_threshold)
    require(a.timezone == 'America/Chicago', 'Timezone must be America/Chicago')
    require(len(a.thresholds) > 0 and all(0 <= x <= 1 for x in a.thresholds), 'Invalid thresholds')
    require(0 <= a.primary_threshold <= 1, 'Invalid primary threshold')
    require(a.gram_size >= 1 and a.sample_per_type >= 0 and a.progress_interval >= 1 and a.max_ast_comparison_cells > 0, 'Invalid config')
    require(a.expected_code_rows > 0 and a.expected_pairs > 0 and a.expected_csv_files > 0, 'Invalid counts')
    require(a.expected_positive + a.expected_negative == a.expected_pairs, 'Label totals mismatch')
    return a


def jsonl(path):
    with path.open(encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            require(bool(line.strip()), f'Blank line {path}:{n}')
            yield json.loads(line)


def ancestry_named(node):
    stack = [node]
    out = []
    while stack:
        n = stack.pop()
        if n.is_named and n.type not in ('line_comment','block_comment','comment'):
            out.append(n.type)
        stack.extend(reversed(n.children))
    return tuple(out)


def grams(seq, k):
    if not seq:
        return frozenset()
    if len(seq) < k:
        return frozenset({seq})
    return frozenset(tuple(seq[i:i+k]) for i in range(len(seq)-k+1))


def jaccard(a,b):
    union = len(a | b)
    return len(a & b)/union if union else 1.0


def harmonic(x,y):
    return (2*x*y/(x+y)) if x+y else 0.0


def choose_type(left, right, score, threshold):
    if not left['function_level_eligible'] or not right['function_level_eligible']:
        return 'INELIGIBLE'
    if left['type1_syntax_sha256'] == right['type1_syntax_sha256']:
        return 'TYPE_1_SIGNATURE'
    if left['type2_consistent_sha256'] == right['type2_consistent_sha256']:
        return 'TYPE_2_CONSISTENT_SIGNATURE'
    if left['type2_blind_sha256'] == right['type2_blind_sha256']:
        return 'TYPE_2_BLIND_SIGNATURE'
    return 'STRUCTURAL_UNSCORED' if score is None else ('TYPE_3_STRUCTURAL_CANDIDATE' if score >= threshold else 'SEMANTIC_UNDETERMINED')


def main():
    a = args()
    import tree_sitter_java
    from tree_sitter import Language, Parser
    require(importlib.metadata.version('tree-sitter') == a.tree_sitter_version, 'tree-sitter version mismatch')
    require(importlib.metadata.version('tree-sitter-java') == a.tree_sitter_java_version, 'tree-sitter-java version mismatch')
    require(a.code_map.is_file() and a.h07_records.is_file() and a.h07_audit.is_file(), 'Missing input')
    require(a.ground_truth_dir.is_dir() and a.output.is_dir() and not any(a.output.iterdir()), 'Output must be new empty directory')
    code_sha = sha(a.code_map.read_bytes())
    require(code_sha == a.expected_code_sha256, 'Code-map SHA mismatch')
    rec_sha = sha(a.h07_records.read_bytes())
    require(rec_sha == a.expected_h07_records_sha256, 'H07-record SHA mismatch')
    audit = json.loads(a.h07_audit.read_text(encoding='utf-8'))
    require(audit['status'] == 'H07_V2_PARSE_AUDIT_PASS' and audit['code_map_sha256'] == code_sha, 'H07 provenance mismatch')
    require(audit['output_sha256']['function_parse_records.jsonl'] == rec_sha, 'H07 audit record SHA mismatch')
    require(audit['code_rows'] == a.expected_code_rows and audit['parse_fail'] == 0, 'H07 coverage mismatch')
    records = {}
    for item in jsonl(a.h07_records):
        require(item['id'] not in records, 'Duplicate H07 ID')
        records[item['id']] = item
    require(len(records) == a.expected_code_rows, 'H07 row count mismatch')
    parser = Parser(Language(tree_sitter_java.language()))
    features = {}
    code_seen = set()
    for i, item in enumerate(jsonl(a.code_map), 1):
        fid, code = str(item['id']), item['code']
        require(fid in records and fid not in code_seen and isinstance(code,str), f'Code map invalid ID {fid}')
        code_seen.add(fid)
        rec = records[fid]
        require(sha(code.encode('utf-8')) == rec['code_sha256'], f'H07 code drift: {fid}')
        if rec['function_level_eligible']:
            source = code.encode('utf-8')
            if rec['parse_mode'] == 'method_in_class_wrapper':
                prefix = b'class H07SyntheticWrapper {\n'
                source = prefix + source + b'\n}\n'
                root = parser.parse(source).root_node
                cl = next((x for x in root.children if x.type == 'class_declaration'), None)
                require(cl is not None, f'Wrapper missing {fid}')
                body = cl.child_by_field_name('body')
                targets = [x for x in body.children if x.type == rec['method_node_type']]
            else:
                root = parser.parse(source).root_node
                targets = [x for x in root.children if x.type == rec['method_node_type']]
            require(not root.has_error and len(targets) == 1, f'AST target mismatch {fid}')
            seq = ancestry_named(targets[0])
            require(len(seq) == rec['ast_named_node_count'], f'AST sequence length drift {fid}')
            sequence_sha = sha(json.dumps(list(seq),separators=(',',':')).encode('utf-8'))
            require(sequence_sha == rec['ast_named_preorder_sha256'], f'AST sequence SHA drift {fid}')
            features[fid] = (seq, grams(seq, a.gram_size))
        if i % a.progress_interval == 0:
            print(f'H08_AST_PROGRESS rows={i} eligible={len(features)}', flush=True)
    require(code_seen == set(records), 'Code-map/H07 ID coverage mismatch')
    files = sorted(a.ground_truth_dir.glob('pred_*_with_refactorability.csv'))
    require(len(files) == a.expected_csv_files, f'CSV count mismatch {len(files)}')
    columns = ['source_csv','source_row','left_id','right_id','id_derived_label','source_prediction','parse_mode_left','parse_mode_right','record_kind_left','record_kind_right','type1_equal','type2_consistent_equal','type2_blind_equal','ast_jaccard','ast_sequence_ratio','ast_combined','metric_status','primary_type']
    types = Counter(); labels = Counter(); by_label = defaultdict(Counter); cross = Counter()
    sensitivity = {str(t):Counter() for t in a.thresholds}
    sampled = defaultdict(list); processed = 0; eligible_pairs = 0; scored_cache = {}; unscored_pairs = 0
    output = a.output / 'pair_clone_types.csv'
    with output.open('x',newline='',encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        for file in files:
            with file.open(encoding='utf-8-sig',newline='') as inp:
                reader = csv.DictReader(inp)
                require({'pair_left_func','pair_right_func','clone_predict'} <= set(reader.fieldnames or []), f'Invalid CSV: {file}')
                for rownum, row in enumerate(reader, 2):
                    lid = (row['pair_left_func'] or '').strip(); rid = (row['pair_right_func'] or '').strip()
                    require(lid in records and rid in records, f'Unknown pair IDs {file.name}:{rownum}')
                    label = int(lid.split('_')[1] == rid.split('_')[1])
                    require(row['clone_predict'].strip() in ('0','1'), f'Invalid source prediction {file}:{rownum}')
                    le, ri = records[lid], records[rid]
                    score = None; jac = None; ratio = None
                    metric_status = 'INELIGIBLE'
                    if le['function_level_eligible'] and ri['function_level_eligible']:
                        eligible_pairs += 1
                        seq_a, g_a = features[lid]; seq_b, g_b = features[rid]
                        ha, hb = le['ast_named_preorder_sha256'], ri['ast_named_preorder_sha256']
                        cache_key = tuple(sorted((ha,hb)))
                        if ha == hb:
                            jac = ratio = score = 1.0
                            metric_status = 'SCORED_IDENTICAL_AST_NODE_TYPES'
                        elif len(seq_a)*len(seq_b) > a.max_ast_comparison_cells:
                            # Never silently compare prefixes or assign low similarity to unscored pairs.
                            metric_status = 'UNSCORED_AST_COMPARISON_LIMIT'
                            unscored_pairs += 1
                        elif cache_key in scored_cache:
                            jac, ratio, score = scored_cache[cache_key]
                            metric_status = 'SCORED_CACHED'
                        else:
                            # Descriptive structural similarity: token values are intentionally ignored here.
                            jac = jaccard(g_a,g_b)
                            ratio = SequenceMatcher(None,seq_a,seq_b,autojunk=False).ratio()
                            score = harmonic(jac,ratio)
                            scored_cache[cache_key] = (jac,ratio,score)
                            metric_status = 'SCORED'
                    typ = choose_type(le,ri,score,a.primary_threshold)
                    types[typ]+=1; labels[label]+=1; by_label[str(label)][typ]+=1
                    cross[(le['parse_mode'],ri['parse_mode'])]+=1
                    for t in a.thresholds:
                        sensitivity[str(t)][choose_type(le,ri,score,t)]+=1
                    result = {'source_csv':file.name,'source_row':rownum,'left_id':lid,'right_id':rid,'id_derived_label':label,'source_prediction':row['clone_predict'].strip(),
                              'parse_mode_left':le['parse_mode'],'parse_mode_right':ri['parse_mode'],'record_kind_left':le['record_kind'],'record_kind_right':ri['record_kind'],
                              'type1_equal':int(le.get('type1_syntax_sha256') == ri.get('type1_syntax_sha256')) if score is not None else '',
                              'type2_consistent_equal':int(le.get('type2_consistent_sha256') == ri.get('type2_consistent_sha256')) if score is not None else '',
                              'type2_blind_equal':int(le.get('type2_blind_sha256') == ri.get('type2_blind_sha256')) if score is not None else '',
                              'ast_jaccard':f'{jac:.8f}' if jac is not None else '',
                              'ast_sequence_ratio':f'{ratio:.8f}' if ratio is not None else '',
                              'ast_combined':f'{score:.8f}' if score is not None else '', 'metric_status':metric_status, 'primary_type':typ}
                    writer.writerow(result)
                    if len(sampled[typ]) < a.sample_per_type:
                        sampled[typ].append(result)
                    processed += 1
                    if processed % a.progress_interval == 0:
                        print(f'H08_PAIR_PROGRESS pairs={processed} eligible={eligible_pairs}',flush=True)
    require(processed == a.expected_pairs and labels[1] == a.expected_positive and labels[0] == a.expected_negative, f'Pair census mismatch {processed} {labels}')
    summary = {'experiment':'h-08','version':'v1','at_chicago':datetime.now(ZoneInfo(a.timezone)).isoformat(),'timezone':a.timezone,
               'status':'H08_V1_PAIR_ANALYSIS_PASS','clone_type_certified':False,'human_verified_gold':False,
               'metric_definition':{'ast_jaccard':f'Jaccard of named AST node-type {a.gram_size}-grams','ast_sequence_ratio':'difflib SequenceMatcher named AST node preorder (autojunk=False)','ast_combined':'harmonic mean of ast_jaccard and ast_sequence_ratio'},
               'primary_threshold':a.primary_threshold,'thresholds':a.thresholds,'code_rows':len(records),'h07_eligible_functions':len(features),
               'csv_files':len(files),'pairs':processed,'eligible_pairs':eligible_pairs,'ineligible_pairs':processed-eligible_pairs,'structural_unscored_pairs':unscored_pairs,
               'label_counts':dict(labels),'primary_type_counts':dict(types),'by_id_derived_label':{k:dict(v) for k,v in by_label.items()},
               'threshold_sensitivity':{k:dict(v) for k,v in sensitivity.items()},'parse_mode_pair_counts':{' / '.join(k):v for k,v in cross.items()},
               'code_map_sha256':code_sha,'h07_records_sha256':rec_sha,'h07_audit_sha256':sha(a.h07_audit.read_bytes()),
               'pair_output_sha256':sha(output.read_bytes()),
               'limitations':['ID-derived label is not independently human-verified ground truth.',
                              'Type-1/2 are parser-specific normalized signature equivalences, not certified clone labels.',
                              'Type-3 is a threshold-based structural candidate, not semantic confirmation.',
                              'Structure-only score 1.0 can reflect distinct programs with identical AST node-type preorder.',
                              'Large AST comparisons over the explicit cell budget are STRUCTURAL_UNSCORED, not Type-4.',
                              'SEMANTIC_UNDETERMINED does not imply Type-4.',
                              'AST node-type sequence does not preserve identifier or literal values; use Type-1/2 signatures for exact/normalized cases.',
                              'Full compilation units are marked INELIGIBLE for function-level comparison.']}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (a.output/'manual_review_samples.json').write_text(json.dumps(dict(sampled),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    manifest = {'experiment':'h-08','version':'v1','status':summary['status'],'at_chicago':summary['at_chicago'],
                'config':{'thresholds':a.thresholds,'primary_threshold':a.primary_threshold,'gram_size':a.gram_size,'sample_per_type':a.sample_per_type,'progress_interval':a.progress_interval,'max_ast_comparison_cells':a.max_ast_comparison_cells,
                          'expected_code_rows':a.expected_code_rows,'expected_csv_files':a.expected_csv_files,'expected_pairs':a.expected_pairs,'expected_positive':a.expected_positive,'expected_negative':a.expected_negative,
                          'tree_sitter_version':a.tree_sitter_version,'tree_sitter_java_version':a.tree_sitter_java_version},
                'inputs':{'code_map':str(a.code_map),'h07_records':str(a.h07_records),'h07_audit':str(a.h07_audit),'ground_truth_dir':str(a.ground_truth_dir)},
                'sha256':{p.name:sha(p.read_bytes()) for p in [output,a.output/'summary.json',a.output/'manual_review_samples.json']}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('status','pairs','eligible_pairs','ineligible_pairs','structural_unscored_pairs','label_counts','primary_type_counts','threshold_sensitivity')},indent=2),flush=True)

if __name__ == '__main__':
    main()
