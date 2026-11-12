#!/usr/bin/env python3
"""Read-only structural truncation proposals: complete switch case groups + AST tail.

Java syntax and length are machine-checkable. Semantics, labels and compilation
are NEVER certified; no candidate is written to a training corpus.
"""
import hashlib
from importlib.metadata import version, PackageNotFoundError

import f01c_tail_audit as tail
from f01c_syntax_audit import require_parser, parse_java, token_count, sha_text, WRAP_HEAD
from f01c_shared import canonical_hash, code_hash

POLICY = 'java-switch-case-group-and-nested-tail-two-sided-review-v2'


def _descendants(root):
    stack=[root]
    while stack:
        node=stack.pop()
        yield node
        stack.extend(reversed(getattr(node,'children',())))


def _switch_groups(parser, code, max_groups):
    """Select contiguous complete colon-case groups nearest the switch end.

    Never delete `default`, fallthrough groups or the entire switch. Arrow-rule
    forms are excluded. A case group may contain declarations; only syntax is
    validated, and compile/semantic status remains unknown.
    """
    root=parse_java(parser,code)
    if root is None:
        return None
    raw=code.encode('utf8'); offset=len(WRAP_HEAD)
    choices=[]
    for sw in _descendants(root):
        if sw.type not in ('switch_expression','switch_statement'):
            continue
        if tail._is_inside_protected(sw):
            continue
        for block in getattr(sw,'named_children',()):
            if block.type not in ('switch_block','switch_expression_block'):
                continue
            groups=[]
            for node in getattr(block,'named_children',()):
                if node.type not in ('switch_block_statement_group','switch_block_statement_group'):
                    continue
                if not (offset<=node.start_byte<node.end_byte<=offset+len(raw)):
                    continue
                snippet=raw[node.start_byte-offset:node.end_byte-offset]
                labels=[c for c in getattr(node,'named_children',()) if c.type=='switch_label']
                # Treat default robustly even if tree-sitter exposes it as a token.
                if not labels or any(b'default' in raw[c.start_byte-offset:c.end_byte-offset] for c in labels):
                    continue
                statements=[c for c in getattr(node,'named_children',()) if c.type!='switch_label']
                if not statements or statements[-1].type not in ('break_statement','return_statement','throw_statement'):
                    continue
                # No trivial fallthrough and no cleanup special cases.
                if tail.CLEANUP_CALL_RE.search(snippet):
                    continue
                groups.append((node.start_byte-offset,node.end_byte-offset,'switch_case_group'))
            # Preserve at least one ordinary branch in each switch.
            if len(groups)<2:
                continue
            # Prefer suffixes adjacent to default; preserve remaining cases.
            for count in range(1,min(len(groups)-1,max_groups)+1):
                spans=list(reversed(groups[-count:]))
                choices.append({'kind':'switch_case_group_suffix','spans':spans,
                                'end':max(s[1] for s in spans),'count':count})
    choices.sort(key=lambda item:(item['count'],-item['end']))
    return choices


def _candidates(parser,code,args):
    switch=_switch_groups(parser,code,args.structural_max_case_groups)
    if switch is None:
        return None
    blocks=tail.tail_blocks(parser,code,args.tail_max_blocks_per_code,args.tail_max_statements_per_block)
    if blocks is None:
        return None
    for group in switch:
        yield group
    for block in blocks:
        for count in range(1,len(block['suffix'])+1):
            yield {'kind':'nested_tail_expression_suffix','spans':block['suffix'][:count],
                   'count':count,'end':block['end']}


def _removed(code, spans):
    raw=code.encode('utf8')
    return [{'utf8_span':[start,end],'node_type':kind,
             'sha256':hashlib.sha256(raw[start:end]).hexdigest(),
             'statement':raw[start:end].decode('utf8')} for start,end,kind in spans]


def _side_options(args, parser, code, source_hashes, heldout_hashes):
    """Generate AST-validated, deterministic candidate edits for one endpoint."""
    results=[]
    seen=set()
    for item in _candidates(parser,code,args):
        changed=tail.remove_suffix(code,item['spans'])
        if changed==code:
            continue
        digest=sha_text(changed)
        if digest in seen or parse_java(parser,changed) is None:
            continue
        # Endpoint-level leakage checks; full pair check follows below.
        hash_value=code_hash(changed)
        if hash_value in source_hashes or hash_value in heldout_hashes:
            continue
        seen.add(digest)
        removed=_removed(code,item['spans'])
        bytes_removed=sum(end-start for start,end,_ in item['spans'])
        results.append({'code':changed,'item':item,'removed':removed,
                        'bytes_removed':bytes_removed})
    results.sort(key=lambda v:(len(v['removed']),v['bytes_removed'],
                               0 if v['item']['kind']=='switch_case_group_suffix' else 1,
                               -v['item']['end'],sha_text(v['code'])))
    return results


def _diverse_shortlist(options, limit):
    """Keep both small edits and high-reduction edits under a shell-defined cap."""
    if len(options)<=limit:
        return options
    front=(limit+1)//2
    by_bytes=sorted(options,key=lambda x:(-x['bytes_removed'],len(x['removed']),sha_text(x['code'])))
    result=options[:front]
    selected={sha_text(x['code']) for x in result}
    for row in by_bytes:
        if len(result)>=limit:break
        if sha_text(row['code']) not in selected:
            result.append(row)
            selected.add(sha_text(row['code']))
    return result


def propose_pair(args,a,b,tok,answer_ids,contextual,parser,source_hashes,heldout_hashes):
    if parse_java(parser,a) is None or parse_java(parser,b) is None:
        return {'status':'ORIGINAL_JAVA_PARSE_ERROR','evaluated':0},None
    oa=_side_options(args,parser,a,source_hashes,heldout_hashes)
    ob=_side_options(args,parser,b,source_hashes,heldout_hashes)
    evaluated=0

    def evaluate(ca,cb):
        nonlocal evaluated
        evaluated+=1
        if not tail._valid_hashes(ca,cb,source_hashes,heldout_hashes):
            return None
        length=token_count(tok,contextual,args,ca,cb,answer_ids)
        return length if length<=args.max_seq_length else None

    def make_result(ra,rb,length,strategy):
        ca=ra['code'] if ra else a
        cb=rb['code'] if rb else b
        removed_a=ra['removed'] if ra else []
        removed_b=rb['removed'] if rb else []
        changed_sides=['A' if ra else None,'B' if rb else None]
        changed_sides=[x for x in changed_sides if x]
        return {'status':'STRUCTURAL_PROPOSAL_MANUAL_REVIEW_REQUIRED',
                'strategy':strategy,'side':'+'.join(changed_sides),
                'changed_sides':changed_sides,'removed_count':len(removed_a)+len(removed_b),
                'removed_statements':removed_a+removed_b,
                'removed_by_side':{'A':removed_a,'B':removed_b},
                'evaluated':evaluated,'transformed_tokens':length,
                'processed_code_sha256':{'id1':sha_text(ca),'id2':sha_text(cb)},
                'semantics_verified':False,'label_verified':False,'compile_verified':False,
                'approved_for_training':False}, {'code_a':ca,'code_b':cb}

    singles=[(x,None) for x in oa]+[(None,x) for x in ob]
    singles.sort(key=lambda pair:(sum(len(q['removed']) for q in pair if q),
                 sum(q['bytes_removed'] for q in pair if q),
                 0 if pair[0] else 1))
    for ra,rb in singles:
        ca=ra['code'] if ra else a
        cb=rb['code'] if rb else b
        length=evaluate(ca,cb)
        if length is not None:
            return make_result(ra,rb,length,(ra or rb)['item']['kind'])

    # Only attempt two-sided edits if neither individual endpoint can meet quota.
    # Avoid combinatorial explosion: bounded deterministic beam by shell policy.
    if args.two_side_max_candidates_per_side>0 and args.two_side_max_combinations>0:
        left=_diverse_shortlist(oa,args.two_side_max_candidates_per_side)
        right=_diverse_shortlist(ob,args.two_side_max_candidates_per_side)
        combinations=[(ra,rb) for ra in left for rb in right]
        combinations.sort(key=lambda pair:(len(pair[0]['removed'])+len(pair[1]['removed']),
                           pair[0]['bytes_removed']+pair[1]['bytes_removed'],
                           abs(pair[0]['bytes_removed']-pair[1]['bytes_removed']),
                           sha_text(pair[0]['code']),sha_text(pair[1]['code'])))
        for ra,rb in combinations[:args.two_side_max_combinations]:
            length=evaluate(ra['code'],rb['code'])
            if length is not None:
                return make_result(ra,rb,length,'two_sided_'+ra['item']['kind']+'__'+rb['item']['kind'])
    return {'status':'NO_STRUCTURAL_PROPOSAL_UNDER_LIMIT','evaluated':evaluated,
            'semantics_verified':False,'label_verified':False,'approved_for_training':False},None


def audit_pairs(args,negatives,codes,tok,answer_ids,contextual,source_hashes,heldout_hashes):
    parser=require_parser()
    rows=[];previews=[];baseline=0;overlength=0
    for id1,id2,label in negatives:
        if label!=0:raise ValueError('Only original negative Java training pairs are accepted')
        a,b=codes[id1],codes[id2]
        orig=token_count(tok,contextual,args,a,b,answer_ids)
        if orig<=args.max_seq_length:
            baseline+=1;continue
        overlength+=1
        result,preview=propose_pair(args,a,b,tok,answer_ids,contextual,parser,source_hashes,heldout_hashes)
        row={'id1':id1,'id2':id2,'label':0,'original_tokens':orig,
             'minimum_reduction':orig-args.max_seq_length,
             'original_pair_sha256':canonical_hash([id1,id2,0,a,b]),
             'original_code_sha256':{'id1':sha_text(a),'id2':sha_text(b)},**result}
        rows.append(row)
        if preview:
            previews.append({'id1':id1,'id2':id2,'label':0,'original_tokens':orig,
                 'transformed_tokens':result['transformed_tokens'],
                 'strategy':result['strategy'],'code_a_before':a,'code_b_before':b,
                 'code_a_after':preview['code_a'],'code_b_after':preview['code_b'],
                 'removed_statements':result['removed_statements'],
                 'removed_by_side':result['removed_by_side'],
                 'changed_sides':result['changed_sides'],
                 'approved_for_training':False,'manual_semantic_review_required':True})
    rows.sort(key=lambda r:(r['original_tokens'],r['id1'],r['id2']))
    vers={}
    for name in ('tree-sitter','tree-sitter-java'):
        try:vers[name]=version(name)
        except PackageNotFoundError:vers[name]='unknown'
    return {'baseline_eligible':baseline,'overlength_count':overlength,
            'manual_review_candidates':len(previews),'candidate_records':rows,
            'previews':previews,'parser_versions':vers,'parser_backend':'tree-sitter-java'}
