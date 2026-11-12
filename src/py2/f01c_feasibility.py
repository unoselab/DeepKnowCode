#!/usr/bin/env python3
"""Read-only deterministic F-01c v18 train/validation feasibility reconstruction.

NEVER modifies the clean corpora, prepared samples, adapter, or training pipeline.
Transformation changes program behavior; negative labels require independent review.
"""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import f01c_shared as s
from f01c_syntax_audit import parse_java, require_parser, token_count, sha_text
from f01c_variants import contextual_continuation

POLICY = 'deterministic-training-feasibility-explicit-variants-v2'


def discover_source(results_dir, configured):
    """Explicit path or deterministically newest v13 structural report/proposal pair."""
    if configured != 'newest-valid':
        folder = Path(configured).resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing feasibility source directory: {folder}')
        return folder
    root = Path(results_dir)
    for folder in sorted(root.glob('*-structural-audit'), reverse=True):
        report = folder / 'structural_truncation_audit.json'
        proposals = folder / 'structural_truncation_proposals.jsonl'
        if report.is_file() and proposals.is_file():
            try:
                obj = s.read_json(report)
            except (ValueError, OSError):
                continue
            if obj.get('version') == 'v13' and obj.get('status') == 'F01C_V13_STRUCTURAL_TRUNCATION_MANUAL_REVIEW_REQUIRED':
                return folder
    raise FileNotFoundError('No complete v13 structural audit; set F01C_FEASIBILITY_SOURCE_RUN in Shell environment')


def verify_proposals(folder, report, codes, data_sha, max_seq_length, tok, args, answer_ids, source_hashes, heldout_hashes):
    """Fail closed: validate every proposal against its source, v13 audit, AST, and token budget."""
    if report.get('version') != 'v13' or report.get('approved_for_training') is not False:
        raise ValueError('Expected unapproved v13 structural report')
    if report.get('source_dataset_sha256') != data_sha:
        raise ValueError('Structural report dataset SHA256 differs from current clean corpora')
    if report.get('configuration', {}).get('max_seq_length') != max_seq_length:
        raise ValueError('Structural report max_seq_length differs from active configuration')
    if report.get('configuration', {}).get('model_id') != args.model_id or report.get('configuration', {}).get('model_revision') != args.model_revision:
        raise ValueError('Structural report model/revision mismatch')
    if report.get('configuration', {}).get('seed') != args.seed:
        raise ValueError('Structural report seed mismatch')
    index={(r['id1'], r['id2']):r for r in report['candidate_records']}
    parser=require_parser()
    accepted=[];seen=set()
    with (folder/'structural_truncation_proposals.jsonl').open(encoding='utf-8') as stream:
        for line_number,line in enumerate(stream,1):
            if not line.strip():continue
            row=json.loads(line)
            key=(row['id1'],row['id2'])
            if key in seen or key not in index:raise ValueError(f'Unlisted or duplicate proposal at line {line_number}')
            seen.add(key);record=index[key]
            if row['label']!=0 or record.get('label')!=0 or row.get('approved_for_training') is not False:
                raise ValueError('Nonnegative or already approved proposal')
            before_a,before_b=codes[key[0]],codes[key[1]]
            after_a,after_b=row['code_a_after'],row['code_b_after']
            if row['code_a_before']!=before_a or row['code_b_before']!=before_b:
                raise ValueError(f'Original code mismatch: {key}')
            if (sha_text(before_a)!=record['original_code_sha256']['id1'] or sha_text(before_b)!=record['original_code_sha256']['id2']
                or sha_text(after_a)!=record['processed_code_sha256']['id1'] or sha_text(after_b)!=record['processed_code_sha256']['id2']):
                raise ValueError(f'Code hashes mismatch: {key}')
            if (after_a==before_a and after_b==before_b) or parse_java(parser,after_a) is None or parse_java(parser,after_b) is None:
                raise ValueError(f'Unmodified or unparsable candidate: {key}')
            if any(s.code_hash(code) in source_hashes or s.code_hash(code) in heldout_hashes for code in (after_a,after_b)):
                raise ValueError(f'Source or heldout collision: {key}')
            actual=token_count(tok,contextual_continuation,args,after_a,after_b,answer_ids)
            # contextual one-token length must match exact audit function; input args are identical.
            if actual != row['transformed_tokens'] or actual != record['transformed_tokens'] or actual>max_seq_length:
                raise ValueError(f'Token count mismatch: {key}: actual={actual}')
            changed=[side for side,before,after in (('A',before_a,after_a),('B',before_b,after_b)) if before!=after]
            if changed!=row.get('changed_sides') or changed!=record.get('changed_sides'):
                raise ValueError(f'Changed-side mismatch: {key}')
            if len(row.get('removed_statements',[]))!=record['removed_count']:
                raise ValueError(f'Statement count mismatch: {key}')
            accepted.append({'id1':key[0],'id2':key[1],'label':0,'tokens':actual,'changed_sides':changed,
                             'removed_count':record['removed_count'],'original_tokens':record['original_tokens'],
                             'a':after_a,'b':after_b,'sha_a':sha_text(after_a),'sha_b':sha_text(after_b),
                             'original_pair_sha256':record['original_pair_sha256']})
    if len(accepted)!=report.get('manual_review_candidates'):
        raise ValueError('Proposal count differs from v13 report')
    return sorted(accepted,key=lambda r:(r['removed_count'],r['original_tokens']-r['tokens'],
                                         len(r['changed_sides']),r['id1'],r['id2']))


def variant_code_id(original_id, code, original_code):
    """Never overload an original source ID with changed bytes.

    The suffix identifies the concrete code instance; original_id is retained in
    the pair for train/validation ancestry checks and split provenance.
    """
    if code == original_code:
        return original_id
    return original_id + '@trunc-' + sha_text(code)


def add_variant_metadata(row, codes):
    """Return a detached row with explicit immutable code-instance identities."""
    r = dict(row)
    code_a = row.get('a', codes[row['id1']])
    code_b = row.get('b', codes[row['id2']])
    id_a = variant_code_id(row['id1'], code_a, codes[row['id1']])
    id_b = variant_code_id(row['id2'], code_b, codes[row['id2']])
    r['code_instance_id1'] = id_a
    r['code_instance_id2'] = id_b
    r['variant_parent_id1'] = row['id1']
    r['variant_parent_id2'] = row['id2']
    r['code_instance_sha256'] = {'A': sha_text(code_a), 'B': sha_text(code_b)}
    return r


def candidate_checks(selected, validation, proposals, codes, heldout_hashes, source_hashes):
    """Use variant IDs for byte identity and parent IDs for split isolation."""
    train_parents={idx for r in selected for idx in (r['id1'],r['id2'])}
    val_parents={idx for r in validation for idx in (r['id1'],r['id2'])}
    endpoint_overlap=sorted(train_parents & val_parents)
    assigned={}; conflicts=[]; pair_fingerprints=set(); pair_duplicates=[]
    missing_variant_provenance=[]; malformed_variant_ids=[]
    for r in selected:
        pair=(r['id1'],r['id2'],r['label'])
        if pair in pair_fingerprints:pair_duplicates.append(pair)
        pair_fingerprints.add(pair)
        for side,parent,code in (
            ('1',r['id1'],r.get('a',codes[r['id1']])),
            ('2',r['id2'],r.get('b',codes[r['id2']]))):
            instance=r.get('code_instance_id'+side)
            expected=variant_code_id(parent,code,codes[parent])
            if instance is None:
                missing_variant_provenance.append(parent)
                continue
            if instance != expected or r.get('variant_parent_id'+side)!=parent:
                malformed_variant_ids.append(parent)
            sha=sha_text(code)
            if instance in assigned and assigned[instance]!=sha:conflicts.append(instance)
            assigned[instance]=sha
    return {'train_validation_endpoint_overlap_ids':endpoint_overlap,
            'train_endpoint_conflicting_code_ids':sorted(set(conflicts)),
            'missing_variant_provenance':sorted(set(missing_variant_provenance)),
            'malformed_variant_ids':sorted(set(malformed_variant_ids)),
            'duplicate_train_pairs':pair_duplicates,
            'selected_modified_heldout_collisions':sum(
                s.code_hash(code) in heldout_hashes
                for r in selected if 'a' in r for code in (r['a'],r['b'])),
            'selected_modified_source_collisions':sum(
                s.code_hash(code) in source_hashes
                for r in selected if 'a' in r for code in (r['a'],r['b']))}


def select_java(args,codes,source_hashes,tok,answer_ids,proposals,heldout_hashes):
    """Reconstruct deterministic java train and validation under target_pool & shuffle rules."""
    import f01c_core as core
    pool_train=core.target_pool(args.java_target_dir/'train.txt',codes,source_hashes)
    pool_val=core.target_pool(args.java_target_dir/'valid.txt',codes,source_hashes)
    for split,pool in (('train',pool_train),('validation',pool_val)):
        rng=random.Random(args.seed+(0 if split=='train' else args.sample_val_seed_offset)+args.java_target_seed_offset)
        for y in (0,1):rng.shuffle(pool[y])
    pool_keys={tuple(x) for x in pool_train[0]}
    for p in proposals:
        if (p['id1'],p['id2'],0) not in pool_keys:
            raise ValueError(f'Proposed pair does not originate from java_mixed/train.txt: {p["id1"]}/{p["id2"]}')
    def length(a,b,y):
        return token_count(tok,contextual_continuation,args,a,b,answer_ids)
    baseline=[];baseline_overlong=0
    for x,z,y in pool_train[0]:
        n=length(codes[x],codes[z],y)
        if n<=args.max_seq_length:baseline.append({'id1':x,'id2':z,'label':y,'tokens':n,'origin':'original'})
        else:baseline_overlong+=1
    original_set={(r['id1'],r['id2']) for r in baseline}
    remaining=[p for p in proposals if (p['id1'],p['id2']) not in original_set]
    need=args.java_target_train//2
    # Same original shuffled sequence as select_domain; append ranked transformations as fallback.
    chosen=[add_variant_metadata(r,codes) for r in baseline[:need]]
    replacement=[]
    for p in remaining:
        if len(chosen)>=need:break
        replacement_row={'id1':p['id1'],'id2':p['id2'],'label':0,'tokens':p['tokens'],
                         'origin':'structural-v13','a':p['a'],'b':p['b'],
                         'changed_sides':p['changed_sides'],'removed_count':p['removed_count'],
                         'original_pair_sha256':p['original_pair_sha256']}
        variant_row=add_variant_metadata(replacement_row,codes)
        chosen.append(variant_row)
        replacement.append(variant_row)
    positive_count=0
    for x,z,y in pool_train[1]:
        if positive_count>=need:break
        n=length(codes[x],codes[z],y)
        if n<=args.max_seq_length:
            chosen.append(add_variant_metadata({'id1':x,'id2':z,'label':y,'tokens':n,'origin':'original'},codes))
            positive_count+=1
    counts=Counter(r['label'] for r in chosen)
    train_ids={idx for r in chosen for idx in (r['id1'],r['id2'])}
    validation=[];validation_skips=Counter()
    val_need=args.java_target_val//2
    for y in (0,1):
        per=0
        for x,z,label in pool_val[y]:
            if x in train_ids or z in train_ids:
                validation_skips['train_endpoint_overlap']+=1;continue
            n=length(codes[x],codes[z],y)
            if n>args.max_seq_length:
                validation_skips['overlength']+=1;continue
            validation.append({'id1':x,'id2':z,'label':y,'tokens':n,'origin':'original'})
            per+=1
            if per==val_need:break
    val_count=Counter(r['label'] for r in validation)
    issues=candidate_checks(chosen,validation,proposals,codes,heldout_hashes,source_hashes)
    feasible=(counts[0]==need and counts[1]==need and all(val_count[y]==val_need for y in (0,1))
              and not any(issues[key] for key in issues))
    report={'train_labels':{str(y):counts[y] for y in (0,1)},
            'validation_labels':{str(y):val_count[y] for y in (0,1)},
            'baseline_eligible_negatives':len(baseline),
            'baseline_overlength_negatives':baseline_overlong,
            'proposals_available':len(proposals), 'proposals_selected':len(replacement),
            'proposals_rejected_endpoint_conflict':[],
            'variant_identity_policy':'original-function-ancestry-plus-content-addressed-code-instance-v1',
            'selected_modified_code_instance_count':sum(
                1 for r in replacement for side in ('1','2')
                if r['code_instance_id'+side] != r['variant_parent_id'+side]),
            'validation_skips':dict(validation_skips),'checks':issues,
            'quota_feasible_under_review':feasible}
    return report,chosen,validation


def review_rows(rows):
    return [{key:value for key,value in r.items() if key not in ('a','b')} for r in rows]
