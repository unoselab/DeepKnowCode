#!/usr/bin/env python3
"""F-01c v4: read-only unused Java train-pair discovery and truncation risk audit.

Neither input corpus nor training dataset is mutated. Extra candidate files must be
explicitly confined to the training tree; validation/test paths are never searched.
"""
from pathlib import Path


def pair_key(x, y):
    return tuple(sorted((x, y)))


def enumerate_train_sources(root, java_train, patterns, max_files):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Pool discovery root must be a directory')
    found = set()
    for pattern in patterns:
        if not pattern.startswith('**/') or pattern.count('*') > 4 or '..' in pattern or '/' not in pattern:
            raise ValueError('Unsafe discovery glob: ' + pattern)
        for item in root.glob(pattern):
            if not item.is_file() or item.is_symlink():
                continue
            resolved = item.resolve(strict=True)
            if not resolved.is_relative_to(root):
                continue
            parts = [p.lower() for p in resolved.relative_to(root).parts]
            if any(x in ('validation','valid','val','test','heldout','evaluation') for x in parts):
                continue
            # The filename itself must be train-only, not test/validation.
            name = resolved.name.lower()
            if not name.endswith('.txt') or 'train' not in name or any(t in name for t in ('test','valid','heldout')):
                continue
            if resolved == Path(java_train).resolve():
                continue
            found.add(resolved)
            if len(found) > max_files:
                raise ValueError('Discovery file budget exceeded; narrow Shell patterns')
    return sorted(found)


def parse_external_pair_file(path, codes):
    """Strictly recognize rows with IDs present in the pinned java_mixed code map."""
    seen = set()
    invalid = unresolved = duplicates = 0
    with path.open('r', encoding='utf-8-sig') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = line.split()
            if len(row) != 3 or row[2] not in ('0','1'):
                invalid += 1
                continue
            x, y, label = row
            if label != '0':
                continue
            if x not in codes or y not in codes or x == y:
                unresolved += 1
                continue
            key = pair_key(x, y)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
    return seen, {'invalid_rows':invalid,'unresolved_negative_rows':unresolved,'duplicate_negative_rows':duplicates}


def analyze_pool(a, codes, source_hash, heldout_hashes, tok, answer_ids, contextual, existing_pool, read_eval_hashes):
    """Use only additional *train* files with both endpoints already in the pinned Java map."""
    java = codes['java_mixed']
    baseline_keys = {pair_key(x,y) for y in (0,1) for x,y,_ in existing_pool[y]}
    used_keys = set(baseline_keys)
    existing_train = a.java_target_dir / 'train.txt'
    validation_keys = {pair_key(x,y) for x,y,_ in __import__('f01c_shared').pair_rows(a.java_target_dir/'valid.txt', java)}
    train_files = enumerate_train_sources(a.pool_discovery_root, existing_train, a.pool_discovery_globs, a.pool_discovery_max_files)
    available = []
    sources = []
    skip = {'existing_pair':0,'validation_pair':0,'source_only':0,'heldout_function':0,'duplicate_across_sources':0,
            'overlength':0,'contextual_token_drift':0}
    for path in train_files:
        keys, meta = parse_external_pair_file(path, java)
        accepted = 0
        for x,y in sorted(keys):
            key = pair_key(x,y)
            if key in baseline_keys:
                skip['existing_pair'] += 1; continue
            if key in validation_keys:
                skip['validation_pair'] += 1; continue
            if key in used_keys:
                skip['duplicate_across_sources'] += 1; continue
            used_keys.add(key)
            hx, hy = ( __import__('f01c_shared').code_hash(java[z]) for z in (x,y))
            if hx in source_hash and hy in source_hash:
                skip['source_only'] += 1; continue
            if hx in heldout_hashes or hy in heldout_hashes:
                skip['heldout_function'] += 1; continue
            text = a.prompt_prefix + java[x] + a.prompt_b_separator + java[y] + a.prompt_suffix
            prefix, token = contextual(tok, text, a.answer_no)
            if token != answer_ids[0]:
                skip['contextual_token_drift'] += 1; continue
            length = len(prefix) + 1
            if length > a.max_seq_length:
                skip['overlength'] += 1; continue
            accepted += 1
            available.append({'id1':x,'id2':y,'label':0,'tokens':length,'source_file':str(path),
                              'source_sha256':__import__('f01c_shared').sha256(path),
                              'pair_ids_sha256':__import__('f01c_shared').canonical_hash([x,y,0])})
        sources.append({'file':str(path),'sha256':__import__('f01c_shared').sha256(path),
                        'resolved_negative_pairs':len(keys),'eligible_new_pairs':accepted,**meta})
    return {'discovery_root':str(a.pool_discovery_root),'discovery_patterns':a.pool_discovery_globs,
            'files_examined':len(train_files),'candidate_sources':sources,'extra_eligible_count':len(available),
            'extra_eligible_pairs':available,'excluded_counts':skip,
            'limitations':['Only train-named files inside explicitly configured root are searched.',
                           'External pairs must reference IDs in the current approved Java target code map.',
                           'Additional candidate label provenance and train/test disjointness across other datasets remain unverified.',
                           'Train/validation endpoint isolation and full deterministic sampling remain unverified.']}


def truncation_risks(a, existing_pool, java, tok, answer_ids, contextual):
    """Risk-ranked overlength audit: no truncation performed and no training approval."""
    negatives = []
    for x,y,_ in existing_pool[0]:
        prompt = a.prompt_prefix + java[x] + a.prompt_b_separator + java[y] + a.prompt_suffix
        prefix, answer = contextual(tok, prompt, a.answer_no)
        if answer != answer_ids[0]:
            raise ValueError('Contextual NO drift in proposed truncation input')
        length = len(prefix)+1
        if length <= a.max_seq_length:
            continue
        # Only suggest balanced CODE budgets; do not change the actual code.
        # Total prompt overhead is computed with the same model tokenizer.
        base = a.prompt_prefix + a.prompt_b_separator + a.prompt_suffix
        overhead = len(tok.encode(base, add_special_tokens=False)) + 1
        budget = max(0, a.max_seq_length-overhead)
        negatives.append({'id1':x,'id2':y,'label':0,'original_tokens':length,
                          'required_token_reduction_at_least':length-a.max_seq_length,
                          'illustrative_code_budget_each':budget//2,
                          'original_pair_sha256':__import__('f01c_shared').canonical_hash([x,y,0,java[x],java[y]]),
                          'review_status':'MANUAL_SEMANTIC_AND_LEAKAGE_REVIEW_REQUIRED'})
    negatives.sort(key=lambda r:(r['required_token_reduction_at_least'],r['id1'],r['id2']))
    return {'original_overlength_count':len(negatives),'risk_ranked_proposals':negatives,
            'truncation_applied':False,
            'limitations':['No truncated code is emitted: blind token clipping may change clone labels and Java validity.',
                           'Token budgets are estimates only; any reviewed excerpts must be re-tokenized with exact prompt.',
                           'M2/M3 must use byte-identical prepared inputs, with labeled transformations and immutable hashes.']}
