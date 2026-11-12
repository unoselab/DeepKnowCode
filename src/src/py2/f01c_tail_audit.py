#!/usr/bin/env python3
"""F-01c v9: read-only last-block suffix statement truncation proposals.

Whole AST statement removal is *not* semantic-preserving, even when Java parses.
Never writes or authorizes training data. Java tree-sitter parse is a syntax check
only; type checks, control-flow effects and negative labels remain unverified.
"""
import hashlib
import re
from importlib.metadata import version, PackageNotFoundError

from f01c_syntax_audit import require_parser, parse_java, token_count, sha_text
from f01c_shared import code_hash, canonical_hash

POLICY = 'java-last-block-consecutive-tail-expressions-review-v1'
# An expression statement can mutate state. This whitelist bounds syntactic edits,
# NOT behavioral effects. The transformed code ALWAYS requires human review.
REMOVABLE = frozenset({'expression_statement'})
# Conservative lexical safety net: never propose truncating conventional cleanup.
# This is NOT an exhaustive side-effect or semantic analysis.
CLEANUP_CALL_RE = re.compile(rb'\b(?:close|commit|rollback|unlatch|unlock|release|shutdown|flush)\s*\(',re.I)
PROTECTED_ANCESTORS = frozenset({
    'finally_clause', 'catch_clause', 'catch_formal_parameter',
    'synchronized_statement', 'lambda_expression', 'class_body',
})


def _named(node):
    return list(getattr(node,'named_children',[]))


def _is_inside_protected(node):
    p = node.parent
    while p is not None:
        # The synthetic class itself has class_body ancestor; stop at the
        # enclosing original method, not at the wrapper class_body.
        if p.type in ('method_declaration','constructor_declaration'):
            return False
        if p.type in PROTECTED_ANCESTORS:
            return True
        p = p.parent
    return False


def tail_blocks(parser, code, max_blocks, max_tail_statements):
    """Identify consecutive removable expression suffixes, latest blocks first.

    Only function/method body blocks are eligible; ignores class/anonymous-class
    bodies, catch/finally, and nested lambda bodies. Protects final return/throw,
    local declarations, and all control-flow nodes by stopping before them.
    """
    root=parse_java(parser,code)
    if root is None:
        return None
    raw=code.encode('utf-8')
    from f01c_syntax_audit import WRAP_HEAD
    offset=len(WRAP_HEAD)
    blocks=[]
    pending=[root]
    while pending:
        node=pending.pop()
        pending.extend(reversed(node.children))
        if node.type!='block' or node.start_byte<offset or node.end_byte>offset+len(raw):
            continue
        if _is_inside_protected(node):
            continue
        named=_named(node)
        if not named or named[-1].type not in REMOVABLE:
            continue
        suffix=[]
        for statement in reversed(named):
            if statement.type not in REMOVABLE:
                break
            if statement.start_byte < offset or statement.end_byte > offset+len(raw):
                break
            if CLEANUP_CALL_RE.search(raw[statement.start_byte-offset:statement.end_byte-offset]):
                break
            suffix.append((statement.start_byte-offset,statement.end_byte-offset,statement.type))
            if len(suffix)>=max_tail_statements:
                break
        if suffix:
            blocks.append({'start':node.start_byte-offset,'end':node.end_byte-offset,
                           'suffix':suffix,'available_suffix_statements':len(suffix)})
    blocks.sort(key=lambda b:(-b['end'],-b['start']))
    return blocks[:max_blocks]


def remove_suffix(code, spans):
    """Remove a contiguous tail of AST expression statements by UTF-8 bytes."""
    raw=code.encode('utf-8')
    ordered=sorted(spans)
    if not ordered or any(not (0<=s<e<=len(raw)) for s,e,*_ in ordered):
        raise ValueError('Invalid statement spans')
    if any(ordered[i][1] > ordered[i+1][0] for i in range(len(ordered)-1)):
        raise ValueError('Overlapping AST spans')
    for start,end,*_ in reversed(ordered):
        raw=raw[:start]+b' ; '+raw[end:]
    return raw.decode('utf-8')


def _valid_hashes(a,b,source,heldout):
    x,y=code_hash(a),code_hash(b)
    return x!=y and x not in source and y not in source and x not in heldout and y not in heldout


def propose_pair(args,a,b,tok,answer_ids,contextual,parser,source_hashes,heldout_hashes):
    """Evaluate last viable blocks and increasing suffix lengths, no training writes."""
    blocks_a=tail_blocks(parser,a,args.tail_max_blocks_per_code,args.tail_max_statements_per_block)
    blocks_b=tail_blocks(parser,b,args.tail_max_blocks_per_code,args.tail_max_statements_per_block)
    if blocks_a is None or blocks_b is None:
        return {'status':'ORIGINAL_JAVA_PARSE_ERROR','evaluated':0},None
    choices=[(0,item) for item in blocks_a]+[(1,item) for item in blocks_b]
    # Prefer the block closest to the end of the later function; attempt both sides.
    choices.sort(key=lambda t:(-t[1]['end'],t[0],-t[1]['start']))
    tests=0
    for side, block in choices:
        for count in range(1,len(block['suffix'])+1):
            spans=block['suffix'][:count]
            transformed=remove_suffix(a if side==0 else b,spans)
            ca,cb=(transformed,b) if side==0 else (a,transformed)
            tests+=1
            if parse_java(parser,ca) is None or parse_java(parser,cb) is None:
                continue
            if not _valid_hashes(ca,cb,source_hashes,heldout_hashes):
                continue
            length=token_count(tok,contextual,args,ca,cb,answer_ids)
            if length>args.max_seq_length:
                continue
            original=(a if side==0 else b).encode('utf-8')
            removed=[{'utf8_span':[s,e], 'node_type':typ,
                      'sha256':hashlib.sha256(original[s:e]).hexdigest(),
                      'statement':original[s:e].decode('utf-8')} for s,e,typ in spans]
            return {'status':'AST_PARSEABLE_MANUAL_SEMANTIC_REVIEW_REQUIRED',
                    'side':'A' if side==0 else 'B',
                    'block_utf8_span':[block['start'],block['end']],
                    'removed_count':count, 'removed_statements':removed,
                    'evaluated':tests, 'transformed_tokens':length,
                    'processed_code_sha256':{'id1':sha_text(ca),'id2':sha_text(cb)},
                    'semantics_verified':False,'label_verified':False,
                    'compile_verified':False,'approved_for_training':False}, {'code_a':ca,'code_b':cb}
    return {'status':'NO_LAST_BLOCK_TAIL_PROPOSAL_UNDER_LIMIT',
            'evaluated':tests,'available_blocks_A':len(blocks_a),
            'available_blocks_B':len(blocks_b),'semantics_verified':False,
            'approved_for_training':False},None


def audit_pairs(args, negatives, codes, tok,answer_ids,contextual,source_hashes,heldout_hashes):
    parser=require_parser()
    rows=[];previews=[];baseline=0;overlength=0
    for id1,id2,label in negatives:
        if label!=0:
            raise ValueError('Only labeled negative Java training pairs accepted')
        a,b=codes[id1],codes[id2]
        orig=token_count(tok,contextual,args,a,b,answer_ids)
        if orig<=args.max_seq_length:
            baseline+=1
            continue
        overlength+=1
        outcome,preview=propose_pair(args,a,b,tok,answer_ids,contextual,parser,source_hashes,heldout_hashes)
        row={'id1':id1,'id2':id2,'label':0,'original_tokens':orig,
             'minimum_reduction':orig-args.max_seq_length,
             'original_pair_sha256':canonical_hash([id1,id2,0,a,b]),
             'original_code_sha256':{'id1':sha_text(a),'id2':sha_text(b)},**outcome}
        rows.append(row)
        if preview:
            previews.append({'id1':id1,'id2':id2,'label':0,
                             'original_tokens':orig,'transformed_tokens':outcome['transformed_tokens'],
                             'code_a_before':a,'code_b_before':b,
                             'code_a_after':preview['code_a'],'code_b_after':preview['code_b'],
                             'removed_statements':outcome['removed_statements'],
                             'approved_for_training':False,'manual_semantic_review_required':True})
    rows.sort(key=lambda r:(r['original_tokens'],r['id1'],r['id2']))
    vers={}
    for name in ('tree-sitter','tree-sitter-java'):
        try:vers[name]=version(name)
        except PackageNotFoundError:vers[name]='unknown'
    return {'baseline_eligible':baseline,'overlength_count':overlength,
            'manual_review_candidates':len(previews),'candidate_records':rows,
            'previews':previews,'parser_versions':vers,'parser_backend':'tree-sitter-java'}
