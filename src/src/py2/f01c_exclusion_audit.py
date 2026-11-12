#!/usr/bin/env python3
"""Read-only, exhaustive Java Target pair exclusion census.

It reproduces the original target_pool policy exactly: exclude a pair ONLY
when BOTH endpoint normalized fingerprints occur in Java Source. Diagnostic
heldout/validation/duplicate flags do NOT silently alter this baseline.
"""
import json
from collections import Counter
from pathlib import Path
from f01c_shared import code_hash, canonical_hash
from f01c_variants import contextual_continuation


def audit_pairs(a, codes, source_hash, heldout_hash, tokenizer, answer_ids):
    mapping = codes['java_mixed']
    function_hashes = {key: code_hash(body) for key, body in mapping.items()}
    valid_ids = set()
    valid_hashes = set()
    valid_pairs = set()
    validation_rows = Counter()
    for x,y,label in pair_rows(a.java_target_dir/'valid.txt',mapping):
        valid_ids.update((x,y))
        valid_hashes.update((function_hashes[x],function_hashes[y]))
        valid_pairs.add((tuple(sorted((function_hashes[x],function_hashes[y]))),label))
        validation_rows[str(label)] += 1
    report = {'policy': 'both-endpoints-in-source-excluded-v1',
              'policy_logic': 'retain iff NOT (source_has_A AND source_has_B)',
              'validation_rows_by_label': dict(validation_rows),
              'validation_unique_endpoints': len(valid_ids),
              'training': {}, 'limitations': [
                  'Source membership is normalized whitespace-free SHA256, not semantic equality.',
                  'Heldout, validation and duplicate diagnostics do not alter the baseline candidate pool.',
                  'Validation endpoint checks are diagnostics, not proof of feasible deterministic joint train/validation sampling.',
                  'No excluded pair is authorized for reuse and no label is changed.',
                  'Tokenization uses the exact frozen tokenizer and contextual answer token.']}
    for label in (0,1):
        report['training'][str(label)] = {'counts': {}, 'examples': {}}
    counters = {0:Counter(),1:Counter()}
    examples={0:{},1:{}}
    seen={0:set(),1:set()}
    candidate_length_counts={0:Counter(),1:Counter()}
    split_overlap={0:Counter(),1:Counter()}
    for n,(x,y,label) in enumerate(pair_rows(a.java_target_dir/'train.txt',mapping),1):
        c=counters[label]
        h1,h2=function_hashes[x],function_hashes[y]
        key=tuple(sorted((h1,h2)))
        membership=(h1 in source_hash)+(h2 in source_hash)
        c['total_rows']+=1
        c[f'source_endpoints_{membership}']+=1
        source_reject=(membership==2)
        if source_reject: c['excluded_both_source']+=1
        else: c['retained_by_actual_source_policy']+=1
        if x==y or h1==h2:c['identical_endpoint_code']+=1
        if key in seen[label]:c['duplicate_pair_hash_unordered']+=1
        else:seen[label].add(key)
        # Orthogonal flags: report even if a pair is source-excluded.
        held=(h1 in heldout_hash)+(h2 in heldout_hash)
        val=(h1 in valid_hashes)+(h2 in valid_hashes)
        if held:c['any_heldout_endpoint']+=1
        if held==2:c['both_heldout_endpoints']+=1
        if val:c['any_java_target_valid_endpoint']+=1
        if val==2:c['both_java_target_valid_endpoints']+=1
        if (key,label) in valid_pairs:c['identical_pair_in_java_target_validation']+=1
        if source_reject:
            if held:c['excluded_source_and_heldout']+=1
            if val:c['excluded_source_and_validation']+=1
            record_key='both_source_excluded'
        else:
            if held:c['retained_but_heldout_collision']+=1
            if val:c['retained_but_validation_endpoint_collision']+=1
            if (key,label) in valid_pairs:c['retained_but_validation_pair_collision']+=1
            prompt=a.prompt_prefix+mapping[x]+a.prompt_b_separator+mapping[y]+a.prompt_suffix
            answer=a.answer_yes if label else a.answer_no
            prefix,token=contextual_continuation(tokenizer,prompt,answer)
            if token != answer_ids[label]:raise RuntimeError(f'Contextual answer token drift: {label} at row {n}')
            tokens=len(prefix)+1
            for limit in a.audit_token_limits:
                if tokens<=limit:c[f'eligible_at_{limit}']+=1
            if tokens>a.max_seq_length:
                c['overlength']+=1
                record_key='retained_overlength'
            else:
                c['eligible_under_training_limit']+=1
                record_key='retained_eligible'
            if held and tokens<=a.max_seq_length:c['eligible_but_heldout_collision']+=1
            if val and tokens<=a.max_seq_length:c['eligible_but_validation_endpoint_collision']+=1
            candidate_length_counts[label][tokens]+=1
        if len(examples[label].get(record_key,[])) < a.exclusion_max_examples:
            examples[label].setdefault(record_key,[]).append({'id1':x,'id2':y,'line':n,
                'label':label,'source_endpoint_count':membership,
                'heldout_endpoint_count':held,'java_target_validation_endpoint_count':val,
                'pair_sha256':canonical_hash([x,y,label,h1,h2])})
    for label in (0,1):
        c=counters[label]
        if c['total_rows'] != c['excluded_both_source'] + c['retained_by_actual_source_policy']:
            raise RuntimeError('Source eligibility conservation failed')
        if c['retained_by_actual_source_policy'] != c['eligible_under_training_limit'] + c['overlength']:
            raise RuntimeError('Token eligibility conservation failed')
        if c['eligible_under_training_limit'] != c[f'eligible_at_{a.max_seq_length}']:
            raise RuntimeError('Token limit accounting inconsistent')
        required=a.java_target_train//2
        result=report['training'][str(label)]
        result['counts']=dict(sorted(c.items()))
        result['examples']=examples[label]
        result['required']=required
        result['shortfall_before_validation_isolation']=max(0,required-c['eligible_under_training_limit'])
        result['baseline_eligible_before_validation_isolation']=c['eligible_under_training_limit']
        result['unique_unordered_pair_hashes']=len(seen[label])
        lengths=sorted(candidate_length_counts[label].elements())
        result['minimum_token_limit_for_quota']=(lengths[required-1] if len(lengths)>=required else None)
    report['baseline_java_negative_eligible']=report['training']['0']['baseline_eligible_before_validation_isolation']
    report['baseline_shortfall']=report['training']['0']['shortfall_before_validation_isolation']
    return report


def pair_rows(path,mapping):
    with Path(path).open(encoding='utf-8-sig') as f:
        for i,line in enumerate(f,1):
            if not line.strip():continue
            part=line.split()
            if len(part)!=3 or part[2] not in ('0','1') or part[0] not in mapping or part[1] not in mapping:
                raise ValueError(f'Malformed pair row at {path}:{i}')
            yield part[0],part[1],int(part[2])
