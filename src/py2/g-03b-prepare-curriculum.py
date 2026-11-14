#!/usr/bin/env python3
"""G-03b v1: read-only M2/score provenance; deterministic M3 order candidate (NO TRAINING)."""
import argparse
import collections
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 'v1'
DOMAINS = ('java_source', 'java_mixed', 'python_mixed')
COMPONENTS = ('d_sim', 'd_cf', 'd_len', 'd_df')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def chi():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def json_bytes(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False,
                       indent=2, allow_nan=False) + '\n').encode('utf-8')


def lines_bytes(rows):
    return (''.join(canonical(row) + '\n' for row in rows)).encode('utf-8')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    result = []
    with Path(path).open('rb') as stream:
        for idx, line in enumerate(stream):
            require(line.strip() and line.endswith(b'\n'), f'INVALID_JSONL_LINE {path}:{idx+1}')
            obj = json.loads(line)
            require(isinstance(obj, dict), f'NON_OBJECT_JSONL {path}:{idx+1}')
            result.append((obj, sha_bytes(line.rstrip(b'\r\n'))))
    return result


def bound(path, expected, name):
    path = Path(path)
    require(path.is_file() and path.stat().st_size > 0, f'MISSING_OR_EMPTY {name} {path}')
    actual = sha(path)
    require(actual == expected, f'SHA256_MISMATCH {name} actual={actual} expected={expected}')
    return path


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('prepare', 'verify'), required=True)
    for key in ('root', 'run-dir', 'log-file', 'g03a-score-run', 'g02-pipeline',
                'f01c-prepared', 'm1-adapter'):
        p.add_argument('--' + key, type=Path, required=True)
    for key in ('version', 'model-id', 'model-revision', 'score-policy', 'score-provenance',
                'score-field', 'g03a-score-manifest-sha256', 'g03a-scores-sha256',
                'g03a-components-sha256', 'g03a-component-audit-sha256',
                'g03a-summary-sha256', 'g02-manifest-sha256', 'g02-train-sha256',
                'g02-validation-sha256', 'g02-standard-order-sha256', 'g02-fingerprint',
                'f01c-manifest-sha256', 'f01c-train-sha256', 'f01c-validation-sha256',
                'm1-adapter-sha256', 'expected-curriculum-order-sha256', 'curriculum-policy', 'tie-policy', 'review-policy',
                'training-approved'):
        p.add_argument('--' + key, required=True)
    for key in ('expected-train', 'expected-validation', 'source-train', 'java-train',
                'python-train', 'source-validation', 'java-validation', 'python-validation',
                'max-seq-length', 'global-batch', 'optimizer-steps', 'seed',
                'expected-imputed', 'expected-variants'):
        p.add_argument('--' + key, type=int, required=True)
    p.add_argument('--score-weights', type=float, nargs=4, required=True)
    p.add_argument('--score-tolerance', type=float, required=True)
    p.add_argument('--phase-names', nargs=3, required=True)
    for key in ('phase-steps', 'phase-source', 'phase-target'):
        p.add_argument('--' + key, type=int, nargs=3, required=True)
    return p.parse_args()


def effective(a):
    return {k: str(v) if isinstance(v, Path) else v
            for k, v in vars(a).items() if k not in ('mode', 'run_dir', 'log_file')}


def validate(a):
    require(a.version == VERSION, 'VERSION_MISMATCH')
    require(a.training_approved == 'false', 'G03B_TRAINING_APPROVAL_MUST_REMAIN_FALSE')
    require(a.curriculum_policy == 'three-phase-domain-balanced-easy-to-hard-v1',
            'CURRICULUM_POLICY_DRIFT')
    require(a.tie_policy == 'score-seeded-sha-row-v1', 'TIE_POLICY_DRIFT')
    require(a.score_policy == 'reduced-4d-externally-verified-v1' and
            a.score_field == 'reduced_4d_score', 'SCORE_POLICY_DRIFT')
    require(a.score_provenance == 'G03A_V1_RECOMPUTED_D04_EFFECTIVE_CODE_ON_G02_FROZEN',
            'SCORE_PROVENANCE_DRIFT')
    require(a.review_policy == 'PENDING_AND_NOT_INDEPENDENTLY_VERIFIED',
            'REVIEW_POLICY_DRIFT')
    for key in ('g03a_score_manifest_sha256', 'g03a_scores_sha256',
                'g03a_components_sha256', 'g03a_component_audit_sha256',
                'g03a_summary_sha256', 'g02_manifest_sha256', 'g02_train_sha256',
                'g02_validation_sha256', 'g02_standard_order_sha256', 'g02_fingerprint',
                'f01c_manifest_sha256', 'f01c_train_sha256', 'f01c_validation_sha256',
                'm1_adapter_sha256', 'expected_curriculum_order_sha256'):
        value = getattr(a, key)
        require(len(value) == 64 and all(ch in '0123456789abcdef' for ch in value),
                'INVALID_SHA256 ' + key)
    require(len(a.model_revision) == 40 and
            all(ch in '0123456789abcdef' for ch in a.model_revision),
            'INVALID_MODEL_REVISION')
    require(a.expected_train == a.source_train + a.java_train + a.python_train ==
            a.optimizer_steps * a.global_batch, 'TOTAL_TRAIN_BUDGET_MISMATCH')
    require(a.expected_validation == a.source_validation + a.java_validation +
            a.python_validation, 'TOTAL_VALIDATION_BUDGET_MISMATCH')
    require(a.global_batch > 0 and a.global_batch % 2 == 0 and a.max_seq_length > 0,
            'INVALID_BATCH_OR_SEQUENCE_LENGTH')
    require(all(n > 0 and n % 2 == 0 for n in
                (a.source_train, a.java_train, a.python_train)), 'INVALID_TRAIN_QUOTAS')
    require(a.expected_imputed >= 0 and a.expected_variants >= 0, 'INVALID_EXPECTED_COUNTS')
    require(all(math.isfinite(w) and w >= 0 for w in a.score_weights) and
            abs(sum(a.score_weights) - 1.0) < 1e-10, 'INVALID_SCORE_WEIGHTS')
    require(math.isfinite(a.score_tolerance) and 0 < a.score_tolerance <= 1e-4,
            'INVALID_SCORE_TOLERANCE')
    require(len(set(a.phase_names)) == 3 and all(a.phase_names), 'INVALID_PHASE_NAMES')
    require(all(s > 0 for s in a.phase_steps), 'INVALID_PHASE_STEPS')
    require(sum(a.phase_steps) == a.optimizer_steps, 'PHASE_STEP_TOTAL_MISMATCH')
    require(sum(a.phase_source) == a.source_train and
            sum(a.phase_target) == a.java_train + a.python_train,
            'PHASE_DOMAIN_TOTAL_MISMATCH')
    require(a.java_train == a.python_train, 'TARGET_DOMAIN_NOT_EQUAL')
    for ix in range(3):
        source, target, steps = (a.phase_source[ix], a.phase_target[ix], a.phase_steps[ix])
        require(source >= 0 and target >= 0 and source + target == steps * a.global_batch,
                f'PHASE_BUDGET_MISMATCH phase={ix}')
        require(source % 2 == 0 and target % 4 == 0,
                f'PHASE_CANNOT_BALANCE_DOMAIN_LABEL phase={ix}')
    # Domain adaptation policy must become more target-focused across phases.
    shares = [a.phase_target[i] / (a.phase_steps[i] * a.global_batch)
              for i in range(3)]
    require(shares[0] < shares[1] < shares[2], 'TARGET_FRACTION_NOT_INCREASING')
    require(a.root.is_dir() and a.g03a_score_run.is_dir() and
            a.g02_pipeline.is_dir() and a.f01c_prepared.is_dir(),
            'MISSING_SOURCE_DIRECTORY')


def load_sources(a):
    """Every input is read-only; both byte-level and row-level provenance are verified."""
    score_dir = a.g03a_score_run
    gprep = a.g02_pipeline / '00-prepare' / 'prepared'
    bound(score_dir / 'score_manifest.json', a.g03a_score_manifest_sha256, 'G03A_MANIFEST')
    bound(score_dir / 'verified_scores.jsonl', a.g03a_scores_sha256, 'G03A_SCORES')
    bound(score_dir / 'verified_d04_components.jsonl',
          a.g03a_components_sha256, 'G03A_COMPONENTS')
    bound(score_dir / 'component_audit.json',
          a.g03a_component_audit_sha256, 'G03A_AUDIT')
    bound(score_dir / 'complexity_summary.json', a.g03a_summary_sha256, 'G03A_SUMMARY')
    bound(gprep / 'manifest.json', a.g02_manifest_sha256, 'G02_MANIFEST')
    bound(gprep / 'train.jsonl', a.g02_train_sha256, 'G02_FROZEN_TRAIN')
    bound(gprep / 'validation.jsonl', a.g02_validation_sha256, 'G02_FROZEN_VALIDATION')
    bound(gprep / 'order.jsonl', a.g02_standard_order_sha256, 'G02_STANDARD_ORDER')
    bound(a.f01c_prepared / 'manifest.json', a.f01c_manifest_sha256, 'F01C_MANIFEST')
    bound(a.f01c_prepared / 'train.jsonl', a.f01c_train_sha256, 'F01C_TRAIN')
    bound(a.f01c_prepared / 'validation.jsonl', a.f01c_validation_sha256, 'F01C_VALIDATION')
    bound(a.m1_adapter, a.m1_adapter_sha256, 'COMMON_M1_ADAPTER')
    meta = load_json(score_dir / 'score_manifest.json')
    gm = load_json(gprep / 'manifest.json')
    fm = load_json(a.f01c_prepared / 'manifest.json')
    audit = load_json(score_dir / 'component_audit.json')
    summary = load_json(score_dir / 'complexity_summary.json')
    g02_pipeline_result = load_json(a.g02_pipeline / 'pipeline_result.json')
    g02_summary = load_json(a.g02_pipeline / '03-train' / 'training_summary.json')
    require(meta.get('status') == 'G03A_V1_COMPLEXITY_SCORE_PASS_NOT_M3_TRAINING'
            and meta.get('approved_for_training') is False
            and meta.get('train_order_generated') is False,
            'G03A_SCORE_STATUS_INVALID')
    for key, expected in (
        ('model_id', a.model_id), ('model_revision', a.model_revision),
        ('g02_pipeline', str(a.g02_pipeline.resolve())),
        ('g02_prepared_manifest_sha256', a.g02_manifest_sha256),
        ('g02_prepared_train_sha256', a.g02_train_sha256),
        ('g02_prepared_validation_sha256', a.g02_validation_sha256),
        ('g02_standard_mixed_order_sha256', a.g02_standard_order_sha256),
        ('g02_config_fingerprint', a.g02_fingerprint),
        ('f01c_frozen_manifest_sha256', a.f01c_manifest_sha256),
        ('f01c_frozen_train_sha256', a.f01c_train_sha256),
        ('f01c_frozen_validation_sha256', a.f01c_validation_sha256),
        ('m1_initial_adapter_sha256', a.m1_adapter_sha256),
        ('score_policy', a.score_policy), ('score_provenance', a.score_provenance),
        ('train_count', a.expected_train), ('validation_count', a.expected_validation),
        ('imputed_pairs', a.expected_imputed), ('variants_count', a.expected_variants),
        ('manual_label_review_status', a.review_policy),
        ('verified_scores_sha256', a.g03a_scores_sha256),
        ('verified_d04_components_sha256', a.g03a_components_sha256),
        ('component_audit_sha256', a.g03a_component_audit_sha256),
        ('complexity_summary_sha256', a.g03a_summary_sha256)):
        require(meta.get(key) == expected, 'G03A_MANIFEST_DRIFT ' + key)
    require(meta.get('m2_adapter_used_as_initial_m3_weights') is False,
            'M3_MUST_BRANCH_FROM_M1')
    require(all(abs(float(x) - float(y)) <= 1e-12 for x, y in
                zip(meta.get('weights', []), a.score_weights)) and
            len(meta.get('weights', [])) == 4, 'SCORE_WEIGHTS_DRIFT')
    require(gm.get('status') == 'G02_V2_PREPARE_PASS_NOT_TRAINED' and
            gm.get('config_fingerprint') == a.g02_fingerprint, 'G02_PREPARED_INVALID')
    require(gm.get('model', {}).get('model_id') == a.model_id and
            gm['model'].get('revision') == a.model_revision, 'G02_MODEL_DRIFT')
    require(gm.get('m1_initial_adapter', {}).get('adapter_sha256') == a.m1_adapter_sha256,
            'G02_M1_ADAPTER_IDENTITY_INVALID')
    require(gm.get('files', {}).get('train', {}).get('sha256') == a.g02_train_sha256 and
            gm['files']['validation'].get('sha256') == a.g02_validation_sha256 and
            gm['files']['order'].get('sha256') == a.g02_standard_order_sha256,
            'G02_DATASET_FILE_LINK_INVALID')
    require(gm.get('frozen_mixed', {}).get('f01c_frozen_sha256') ==
            {'train': a.f01c_train_sha256, 'validation': a.f01c_validation_sha256,
             'manifest': a.f01c_manifest_sha256}, 'F01C_PROVENANCE_DRIFT')
    require(gm['frozen_mixed'].get('m1_source_pair_identity') == 'TRAIN_EXACT_MATCH',
            'M1_SOURCE_PAIR_IDENTITY_INVALID')
    require(gm['frozen_mixed'].get('manual_label_review_status') == a.review_policy,
            'G02_LABEL_REVIEW_DRIFT')
    require(gm.get('domain_counts', {}).get('train') ==
            {'java_source': a.source_train, 'java_mixed': a.java_train,
             'python_mixed': a.python_train}, 'TRAIN_DOMAIN_COUNTS_DRIFT')
    require(gm['domain_counts'].get('validation') ==
            {'java_source': a.source_validation, 'java_mixed': a.java_validation,
             'python_mixed': a.python_validation}, 'VALIDATION_DOMAIN_COUNTS_DRIFT')
    require(gm.get('training_budget', {}).get('optimizer_steps') == a.optimizer_steps
            and gm['training_budget'].get('examples') == a.expected_train and
            gm['training_budget'].get('gradient_accumulation') == a.global_batch and
            gm['training_budget'].get('world_size') == 1,
            'G02_SINGLE_PROCESS_BUDGET_DRIFT')
    require(fm.get('status') == 'PREPARED_PENDING_LABEL_REVIEW' and
            fm.get('sample', {}).get('train', {}).get('sha256') == a.f01c_train_sha256 and
            fm['sample']['validation'].get('sha256') == a.f01c_validation_sha256,
            'F01C_LABEL_OR_SAMPLE_DRIFT')
    require(g02_pipeline_result.get('status') == 'G02_V2_FULL_M2_TRAINING_COMPLETE'
            and g02_pipeline_result.get('exit_code') == 0,
            'G02_REFERENCE_TRAINING_NOT_COMPLETE')
    require(g02_summary.get('status') == 'G02_V2_M2_STANDARD_MIXED_TRAINING_COMPLETE'
            and g02_summary.get('optimizer_steps') == a.optimizer_steps
            and g02_summary.get('train_sample_exposures') == a.expected_train
            and g02_summary.get('initial_m1_adapter_sha256') == a.m1_adapter_sha256,
            'G02_TRAINING_SUMMARY_DRIFT')
    require(audit.get('component_file_sha256') == a.g03a_components_sha256 and
            len(audit.get('median_fallback_pairs', [])) == a.expected_imputed and
            summary.get('count') == a.expected_train and
            summary.get('imputed_pairs') == a.expected_imputed,
            'G03A_COMPONENT_AUDIT_DRIFT')
    train = read_jsonl(gprep / 'train.jsonl')
    original = read_jsonl(a.f01c_prepared / 'train.jsonl')
    scores = read_jsonl(score_dir / 'verified_scores.jsonl')
    components = read_jsonl(score_dir / 'verified_d04_components.jsonl')
    standard = read_jsonl(gprep / 'order.jsonl')
    require(len(train) == len(original) == len(scores) == len(components) ==
            len(standard) == a.expected_train, 'ROW_COUNT_DRIFT')
    validation = read_jsonl(gprep / 'validation.jsonl')
    fvalidation = read_jsonl(a.f01c_prepared / 'validation.jsonl')
    require(len(validation) == len(fvalidation) == a.expected_validation,
            'VALIDATION_COUNT_DRIFT')
    index_seen = set()
    domain_counts = collections.Counter()
    imputed = 0
    variants = 0
    verified = []
    std_order = []
    for ix, (g, f, sc, co) in enumerate(zip(train, original, scores, components)):
        gr, ghash = g
        fr, fhash = f
        s, _ = sc
        c, _ = co
        require(type(gr.get('label')) is int and gr['label'] in (0, 1),
                'G02_LABEL_INVALID ' + str(ix))
        key = (gr.get('corpus'), str(gr.get('id1')), str(gr.get('id2')), gr['label'])
        original_key = (fr.get('corpus'), str(fr.get('id1')), str(fr.get('id2')), fr.get('label'))
        require(key == original_key and key[0] in DOMAINS and key not in index_seen,
                'PAIR_IDENTITY_DRIFT ' + str(ix))
        index_seen.add(key)
        require(s.get('row_index') == ix and c.get('row_index') == ix and
                tuple(s.get(k) for k in ('corpus', 'id1', 'id2', 'label')) == key,
                'SCORE_PAIR_IDENTITY_DRIFT ' + str(ix))
        require(s.get('g02_row_sha256') == ghash and c.get('g02_row_sha256') == ghash and
                s.get('source_row_sha256') == fhash and c.get('source_row_sha256') == fhash and
                s.get('g02_original_row_sha256') == gr.get('original_row_sha256') ==
                sha_bytes(canonical(fr).encode('utf-8')),
                'SCORE_ROW_HASH_DRIFT ' + str(ix))
        require(s.get('effective_code_a_sha256') == gr.get('code1_sha256') and
                s.get('effective_code_b_sha256') == gr.get('code2_sha256') and
                c.get('effective_code_a_sha256') == gr.get('code1_sha256') and
                c.get('effective_code_b_sha256') == gr.get('code2_sha256'),
                'EFFECTIVE_CODE_HASH_DRIFT ' + str(ix))
        require(s.get('score_provenance') == a.score_provenance and
                s.get('score_policy') == a.score_policy and
                s.get('component_policy') == 'D04_VERIFIED_EFFECTIVE_CODE_COMPONENTS',
                'SCORE_COMPONENT_POLICY_DRIFT ' + str(ix))
        for k in COMPONENTS:
            value = s.get(k)
            require(type(value) in (int, float) and math.isfinite(value)
                    and 0 <= value <= 1 and value == c.get(k),
                    'INVALID_COMPONENT '+k+' row='+str(ix))
        score = s.get(a.score_field)
        expect = sum(w * s[k] for k, w in zip(COMPONENTS, a.score_weights))
        require(type(score) in (int, float) and math.isfinite(score) and
                0 <= score <= 1 and math.isclose(score, expect, abs_tol=a.score_tolerance,
                                              rel_tol=0), 'SCORE_ARITHMETIC_INVALID '+str(ix))
        require(type(s.get('imputed_proxy')) is bool and
                s['imputed_proxy'] == c.get('imputed_proxy'), 'IMPUTATION_DRIFT '+str(ix))
        require(type(gr.get('n_tokens')) is int and 1 <= gr['n_tokens'] <= a.max_seq_length,
                'GPTOSS_TOKEN_LIMIT_DRIFT '+str(ix))
        original_fields = ('code_a_override', 'code_b_override')
        require(all((key in gr) == (key in fr) and (key not in gr or gr[key] == fr[key])
                    for key in original_fields), 'STRUCTURAL_OVERRIDE_DRIFT '+str(ix))
        variants += bool('code_a_override' in gr or 'code_b_override' in gr)
        domain_counts[(key[0], key[3])] += 1
        imputed += s['imputed_proxy']
        verified.append({'row_index': ix, 'corpus': key[0], 'id1': key[1],
                         'id2': key[2], 'label': key[3], 'score': score,
                         'g02_row_sha256': ghash, 'source_row_sha256': fhash,
                         'imputed_proxy': s['imputed_proxy']})
    require(variants == a.expected_variants and imputed == a.expected_imputed,
            'EXPECTED_VARIANT_OR_IMPUTED_DRIFT')
    for corpus, amount in (('java_source', a.source_train), ('java_mixed', a.java_train),
                           ('python_mixed', a.python_train)):
        require(domain_counts[(corpus, 0)] == domain_counts[(corpus, 1)] == amount // 2,
                'CORPUS_LABEL_QUOTA_DRIFT '+corpus)
    for (item, _), ix in zip(standard, range(a.expected_train)):
        require(type(item.get('index')) is int and 0 <= item['index'] < a.expected_train,
                'BAD_STANDARD_ORDER_INDEX '+str(ix))
        row = train[item['index']][0]
        require(item.get('row_sha256') == sha_bytes(canonical(row).encode('utf-8')),
                'STANDARD_ORDER_ROW_HASH_DRIFT '+str(ix))
        std_order.append(item['index'])
    require(len(set(std_order)) == a.expected_train, 'STANDARD_ORDER_NOT_PERMUTATION')
    print(json.dumps({'event':'G03B_SOURCE_AUDIT_PASS',
          'frozen_pairs':len(verified),'validation_unchanged':len(validation),
          'imputed_pairs':imputed,'variants':variants,
          'm1_adapter_sha256':a.m1_adapter_sha256, 'at_chicago':chi()}), flush=True)
    return verified, std_order


def allocation_order(domain_quota, steps, batch_half):
    """Deterministic proportional-deficit selection; no replacement or random sampling."""
    total = sum(domain_quota.values())
    require(total == steps * batch_half, 'LABEL_PHASE_QUOTA_INCONSISTENT')
    consumed = collections.Counter()
    seq = []
    for pos in range(total):
        candidates = [d for d in DOMAINS if consumed[d] < domain_quota[d]]
        require(bool(candidates), 'FAIR_SCHEDULER_EXHAUSTED')
        # Highest fractional deficit compared to ideal expected exposure at prefix.
        choice = max(candidates, key=lambda d: (
            (pos + 1) * domain_quota[d] - total * consumed[d],
            -DOMAINS.index(d)))
        consumed[choice] += 1
        seq.append(choice)
    require(dict(consumed) == {d:n for d,n in domain_quota.items() if n},
            'FAIR_SCHEDULER_WRONG_COUNTS')
    return seq


def make_order(a, source, standard):
    buckets = { (d, y): [] for d in DOMAINS for y in (0, 1)}
    for row in source:
        buckets[row['corpus'], row['label']].append(row)
    for key in buckets:
        buckets[key].sort(key=lambda r: (
            r['score'], sha_bytes((str(a.seed) + ':' + r['g02_row_sha256']).encode('utf-8')),
            r['row_index']))
    cursors = collections.Counter()
    records = []
    phases = []
    step = 0
    per_step = a.global_batch // 2
    for phase_idx, name in enumerate(a.phase_names):
        phase_start = len(records)
        steps = a.phase_steps[phase_idx]
        quotas = {'java_source':a.phase_source[phase_idx] // 2,
                  'java_mixed':a.phase_target[phase_idx] // 4,
                  'python_mixed':a.phase_target[phase_idx] // 4}
        per_label = {y: allocation_order(quotas, steps, per_step) for y in (0, 1)}
        hist = collections.Counter()
        diffs = []
        for local_step in range(steps):
            step += 1
            for position in range(a.global_batch):
                label = position % 2
                seq_pos = local_step * per_step + position // 2
                domain = per_label[label][seq_pos]
                k = (domain, label)
                offset = cursors[k]
                require(offset < len(buckets[k]), 'INSUFFICIENT_FROZEN_SCORE_POOL')
                data = buckets[k][offset]
                cursors[k] += 1
                hist[k] += 1
                diffs.append(data['score'])
                records.append({'position':len(records), 'optimizer_step':step,
                    'microbatch_index':position, 'phase_index':phase_idx,
                    'phase_name':name, 'row_index':data['row_index'],
                    'g02_row_sha256':data['g02_row_sha256'],
                    'source_row_sha256':data['source_row_sha256'],
                    'corpus':data['corpus'], 'label':label,
                    a.score_field:data['score'], 'imputed_proxy':data['imputed_proxy']})
        expected_hist = { (d, y): quotas[d] for d in DOMAINS for y in (0, 1)}
        require(all(hist[key] == wanted for key, wanted in expected_hist.items()),
                'PHASE_DOMAIN_LABEL_BUDGET_DRIFT '+name)
        phases.append({'name':name, 'phase_index':phase_idx,
                       'steps':steps, 'first_optimizer_step':step-steps+1,
                       'last_optimizer_step':step, 'rows':len(records)-phase_start,
                       'java_source':a.phase_source[phase_idx],
                       'java_mixed':a.phase_target[phase_idx] // 2,
                       'python_mixed':a.phase_target[phase_idx] // 2,
                       'corpus_label_counts':{d:{str(y):hist[d,y] for y in (0,1)}
                                              for d in DOMAINS},
                       'mean_heuristic_difficulty':statistics.fmean(diffs),
                       'minimum_heuristic_difficulty':min(diffs),
                       'maximum_heuristic_difficulty':max(diffs),
                       'target_fraction':a.phase_target[phase_idx] / (steps*a.global_batch)})
    require(len(records) == a.expected_train and step == a.optimizer_steps,
            'CURRICULUM_BUDGET_NOT_EXHAUSTIVE')
    require(all(cursors[k] == len(v) for k,v in buckets.items()),
            'DUPLICATE_OR_UNCONSUMED_SOURCE_SCORE')
    original_order = [r['row_index'] for r in records]
    require(len(set(original_order)) == a.expected_train and
            set(original_order) == set(range(a.expected_train)), 'ORDER_NOT_EXACT_PERMUTATION')
    for i in range(a.optimizer_steps):
        block = records[i*a.global_batch:(i+1)*a.global_batch]
        require(sum(r['label'] == 0 for r in block) == per_step and
                sum(r['label'] == 1 for r in block) == per_step and
                all(r['optimizer_step']==i+1 for r in block),
                'STEP_LABEL_BALANCE_OR_ORDER_DRIFT '+str(i+1))
    relocated = sum(i != j for i,j in zip(original_order,standard))
    require(relocated > 0, 'CURRICULUM_ORDER_IDENTICAL_TO_STANDARD')
    audit = {'status':'G03B_V1_CURRICULUM_ORDER_AUDIT_PASS_NOT_TRAINED',
        'version':VERSION, 'training_mode':'single-process-balanced-device-map-3gpu',
        'global_batch':a.global_batch, 'optimizer_steps':a.optimizer_steps,
        'total_samples':len(records), 'unique_source_rows':len(set(original_order)),
        'original_row_count':a.expected_train,
        'every_step_labels':{ '0':per_step, '1':per_step },
        'imputed_scored_pairs':sum(r['imputed_proxy'] for r in records),
        'phases':phases,
        'score_method':a.score_policy,
        'note':'Ordered by ascending Reduced-4D heuristic within each corpus/label stratum; phase domain fractions rise. Not independently validated difficulty.',
        'approved_for_training':False}
    compare = {'status':'G03B_V1_REFERENCE_STANDARD_ORDER_COMPARED',
        'g02_standard_order_sha256':a.g02_standard_order_sha256,
        'train_pairs':a.expected_train,'same_position_count':a.expected_train-relocated,
        'changed_position_count':relocated,
        'changed_position_fraction':relocated/a.expected_train,
        'permutation_equal':True,
        'frozen_dataset_reordered_not_modified':True}
    return records, audit, compare


def prepare(a):
    source, standard = load_sources(a)
    ordered, audit, compare = make_order(a,source,standard)
    require(sha_bytes(lines_bytes(ordered)) == a.expected_curriculum_order_sha256,
            'CURRICULUM_PRECOMPUTED_ORDER_SHA256_MISMATCH')
    require(not a.run_dir.exists(), 'REFUSE_OVERWRITE_EXISTING_RUN')
    a.run_dir.mkdir(parents=True,exist_ok=False)
    outputs = {
        'curriculum_order.jsonl':lines_bytes(ordered),
        'curriculum_phase_audit.json':json_bytes(audit),
        'standard_order_comparison.json':json_bytes(compare),
    }
    for filename, data in outputs.items():
        with (a.run_dir/filename).open('xb') as stream:stream.write(data)
    manifest = {
        'version':VERSION, 'status':'G03B_V1_CURRICULUM_PREPARED_AUDIT_PASS_NOT_TRAINED',
        'created_at_chicago':chi(),'run_dir':str(a.run_dir.resolve()),
        'g03a_score_run':str(a.g03a_score_run.resolve()),
        'g02_pipeline':str(a.g02_pipeline.resolve()),
        'model_id':a.model_id,'model_revision':a.model_revision,
        'common_frozen_m1_adapter_sha256':a.m1_adapter_sha256,
        'm2_adapter_used_to_initialize_m3':False,
        'source_g02_train_sha256':a.g02_train_sha256,
        'source_g02_validation_sha256':a.g02_validation_sha256,
        'source_g02_standard_order_sha256':a.g02_standard_order_sha256,
        'source_g03a_score_sha256':a.g03a_scores_sha256,
        'source_g03a_manifest_sha256':a.g03a_score_manifest_sha256,
        'expected_curriculum_order_sha256':a.expected_curriculum_order_sha256,
        'input_pinned_hashes':{key:getattr(a,key) for key in
            ('g03a_score_manifest_sha256','g03a_scores_sha256','g03a_components_sha256',
             'g03a_component_audit_sha256','g03a_summary_sha256',
             'g02_manifest_sha256','g02_train_sha256','g02_validation_sha256',
             'g02_standard_order_sha256','g02_fingerprint','f01c_manifest_sha256',
             'f01c_train_sha256','f01c_validation_sha256','m1_adapter_sha256',
             'expected_curriculum_order_sha256')},
        'output_sha256':{name:sha_bytes(data) for name,data in outputs.items()},
        'global_batch':a.global_batch,'optimizer_steps':a.optimizer_steps,
        'train_pairs':a.expected_train,'validation_pairs_unchanged':a.expected_validation,
        'imputed_pairs':a.expected_imputed,'expected_structural_variants':a.expected_variants,
        'curriculum_policy':a.curriculum_policy,'tie_policy':a.tie_policy,
        'phase_plan':audit['phases'],
        'manual_label_review_status':a.review_policy,
        'approved_for_training':False,
        'not_approved_reasons':['Independent Java mixed-label adjudication pending',
            'Heuristic difficulty scores not independently calibrated to prediction errors',
            'G-03c full training order consumption and checkpoint alignment not yet audited'],
        'frozen_train_modified':False,'frozen_validation_modified':False,
        'gpu_training_started':False,
        'effective_parameters':effective(a),
    }
    with (a.run_dir/'curriculum_manifest.json').open('xb') as stream:
        stream.write(json_bytes(manifest))
    print(json.dumps({'status':manifest['status'], 'run':str(a.run_dir.resolve()),
        'train_pairs':len(ordered),'steps':a.optimizer_steps,
        'order_sha256':manifest['output_sha256']['curriculum_order.jsonl'],
        'changed_from_standard':compare['changed_position_count'],
        'approved_for_training':False,'at_chicago':chi()}),flush=True)


def verify(a):
    require(a.run_dir.is_dir(),'MISSING_CURRICULUM_RUN')
    manifest_path=a.run_dir/'curriculum_manifest.json'
    manifest=load_json(manifest_path)
    require(manifest.get('status')=='G03B_V1_CURRICULUM_PREPARED_AUDIT_PASS_NOT_TRAINED' and
            manifest.get('approved_for_training') is False and
            manifest.get('gpu_training_started') is False and
            manifest.get('m2_adapter_used_to_initialize_m3') is False,
            'CURRICULUM_MANIFEST_STATUS_OR_LINEAGE_INVALID')
    require(manifest.get('run_dir')==str(a.run_dir.resolve()) and
            manifest.get('effective_parameters')==effective(a),
            'CURRICULUM_CONFIG_OR_PATH_DRIFT')
    require(manifest.get('model_id')==a.model_id and
            manifest.get('model_revision')==a.model_revision and
            manifest.get('common_frozen_m1_adapter_sha256')==a.m1_adapter_sha256,
            'CURRICULUM_MODEL_OR_M1_DRIFT')
    source, standard=load_sources(a)
    ordered, audit, compare=make_order(a,source,standard)
    require(sha_bytes(lines_bytes(ordered)) == a.expected_curriculum_order_sha256,
            'CURRICULUM_PRECOMPUTED_ORDER_SHA256_MISMATCH')
    expectations={
        'curriculum_order.jsonl':lines_bytes(ordered),
        'curriculum_phase_audit.json':json_bytes(audit),
        'standard_order_comparison.json':json_bytes(compare),
    }
    for name, expected in expectations.items():
        require(manifest.get('output_sha256',{}).get(name)==sha_bytes(expected),
                'CURRICULUM_MANIFEST_HASH_DRIFT '+name)
        require(bound(a.run_dir/name,sha_bytes(expected),name) is not None,
                'CURRICULUM_OUTPUT_MISSING '+name)
        require((a.run_dir/name).read_bytes()==expected,
                'NONDETERMINISTIC_OR_DRIFTED_CURRICULUM_OUTPUT '+name)
    require(manifest.get('phase_plan')==audit['phases'] and
            manifest.get('train_pairs')==a.expected_train and
            manifest.get('optimizer_steps')==a.optimizer_steps and
            manifest.get('source_g03a_score_sha256')==a.g03a_scores_sha256 and
            manifest.get('expected_curriculum_order_sha256') == a.expected_curriculum_order_sha256,
            'CURRICULUM_MANIFEST_PHASE_DRIFT')
    print(json.dumps({'status':'G03B_V1_INDEPENDENT_REBUILD_VERIFY_PASS',
        'run':str(a.run_dir.resolve()), 'train_pairs':len(ordered),
        'order_sha256':sha(a.run_dir/'curriculum_order.jsonl'),
        'at_chicago':chi()}),flush=True)


def main():
    a=cli()
    try:
        validate(a)
        if a.mode=='prepare': prepare(a)
        else:verify(a)
    except Exception as exc:
        print(json.dumps({'event':'G03B_FAIL','error_type':type(exc).__name__,
                          'error':str(exc),'at_chicago':chi()}),flush=True)
        raise


if __name__=='__main__':
    main()
