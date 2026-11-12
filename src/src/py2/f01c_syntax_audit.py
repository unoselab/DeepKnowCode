#!/usr/bin/env python3
"""F-01c v5: read-only Java AST statement-removal proposals (never training input).

A parseable Java method is necessary, but not sufficient, to preserve semantics.
Every emitted proposal is MANUAL_SEMANTIC_REVIEW_REQUIRED and never approved.
Requires tree-sitter and tree-sitter-java installed in the server Python env.
"""
import hashlib
import json
from importlib.metadata import version, PackageNotFoundError

POLICY = 'tree-sitter-java-statement-review-v1'
# Standalone statements in an explicit block, never declarations at class scope or
# control-flow constructs. These statements may have side effects; human review required.
REVIEWABLE_NODES = frozenset(('expression_statement', 'local_variable_declaration'))
WRAP_HEAD = b'class F01cSyntaxAuditWrapper {\n'
WRAP_TAIL = b'\n}\n'


def sha_text(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def require_parser():
    try:
        import tree_sitter
        import tree_sitter_java
    except ImportError as e:
        raise RuntimeError('Syntax audit requires tree-sitter and tree-sitter-java: '
                           'python -m pip install tree-sitter tree-sitter-java') from e
    try:
        language = tree_sitter.Language(tree_sitter_java.language())
        try:
            parser = tree_sitter.Parser(language)
        except TypeError:
            parser = tree_sitter.Parser()
            parser.language = language
    except Exception as e:
        raise RuntimeError('Incompatible tree-sitter/tree-sitter-java installations') from e
    return parser


def parse_java(parser, code):
    """Parse method/function body as member of a synthetic class; never patch source."""
    if not isinstance(code, str) or not code.strip():
        return None
    encoded = code.encode('utf-8')
    wrapped = WRAP_HEAD + encoded + WRAP_TAIL
    tree = parser.parse(wrapped)
    if tree.root_node.has_error:
        return None
    return tree.root_node


def candidate_spans(parser, code):
    """Return UTF-8 byte spans of complete, standalone method-body statements."""
    root = parse_java(parser, code)
    if root is None:
        return None
    head_len = len(WRAP_HEAD)
    end_len = head_len + len(code.encode('utf-8'))
    spans = []
    pending = [root]
    while pending:
        node = pending.pop()
        if (node.type in REVIEWABLE_NODES and node.parent is not None
                and node.parent.type == 'block' and node.start_byte >= head_len
                and node.end_byte <= end_len):
            spans.append((node.start_byte-head_len, node.end_byte-head_len, node.type))
        pending.extend(reversed(node.children))
    return sorted(set(spans), key=lambda r:(r[0],r[1],r[2]))


def replace_statement(code, span):
    """Replace one complete statement with an empty Java statement, not token clipping."""
    start,end,_ = span
    raw = code.encode('utf-8')
    if not (0 <= start < end <= len(raw)):
        raise ValueError('Invalid AST byte span')
    # Extra whitespace surrounding replacement keeps adjacent lexical tokens separate.
    return (raw[:start] + b' ; ' + raw[end:]).decode('utf-8')


def is_safe_hash_distance(code_a, code_b, source_hashes, heldout_hashes, hash_function):
    hashes = (hash_function(code_a),hash_function(code_b))
    return (hashes[0] != hashes[1] and
            not any(x in source_hashes or x in heldout_hashes for x in hashes))


def token_count(tok, contextual, args, a, b, answer_ids):
    text = args.prompt_prefix + a + args.prompt_b_separator + b + args.prompt_suffix
    prefix, token = contextual(tok,text,args.answer_no)
    if token != answer_ids[0]:
        raise ValueError('Pinned contextual NO token drift')
    return len(prefix)+1


def propose_one(args, code_a, code_b, tok, answer_ids, contextual, parser,
                source_hashes, heldout_hashes, hash_function):
    """Try replacing one AST-identified full statement at a time; no training writes."""
    a_spans = candidate_spans(parser,code_a)
    b_spans = candidate_spans(parser,code_b)
    if a_spans is None or b_spans is None:
        return {'status':'UNPARSEABLE_JAVA_FRAGMENT', 'evaluated':0}, None
    spans = [(0,x) for x in a_spans]+[(1,x) for x in b_spans]
    # Most conservative first: prefer small bytes removed to avoid dramatic edits.
    spans.sort(key=lambda item:(item[1][1]-item[1][0],item[0],item[1][0]))
    tested=0
    for side,span in spans[:args.syntax_max_statements_per_pair]:
        if side==0:
            cand_a,cand_b=replace_statement(code_a,span),code_b
        else:
            cand_a,cand_b=code_a,replace_statement(code_b,span)
        tested+=1
        if parse_java(parser,cand_a) is None or parse_java(parser,cand_b) is None:
            continue
        if not is_safe_hash_distance(cand_a,cand_b,source_hashes,heldout_hashes,hash_function):
            continue
        length=token_count(tok,contextual,args,cand_a,cand_b,answer_ids)
        if length>args.max_seq_length:
            continue
        meta={'status':'AST_PARSEABLE_MANUAL_SEMANTIC_REVIEW_REQUIRED',
              'statement_side':'A' if side==0 else 'B',
              'statement_type':span[2], 'span_utf8_bytes':[span[0],span[1]],
              'removed_statement_sha256':hashlib.sha256((code_a if side==0 else code_b).encode('utf-8')[span[0]:span[1]]).hexdigest(),
              'transformed_tokens':length,'processed_code_sha256':{'id1':sha_text(cand_a),'id2':sha_text(cand_b)},
              'evaluated':tested,'semantics_verified':False,'label_verified':False}
        return meta, {'code_a':cand_a,'code_b':cand_b}
    return {'status':'NO_PARSEABLE_SINGLE_STATEMENT_PROPOSAL', 'evaluated':tested,
            'available_statement_nodes':len(spans)}, None


def audit_pairs(args, negatives, java, tok, answer_ids, contextual, source_hashes,heldout_hashes):
    parser=require_parser()
    from f01c_shared import code_hash, canonical_hash
    records=[];previews=[];baseline=0;overlength=0;eligible=0
    for id1,id2,label in negatives:
        if label!=0:
            raise ValueError('Syntax audit accepts only negative training pairs')
        a,b=java[id1],java[id2]
        original_tokens=token_count(tok,contextual,args,a,b,answer_ids)
        if original_tokens<=args.max_seq_length:
            baseline+=1
            continue
        overlength+=1
        result, preview=propose_one(args,a,b,tok,answer_ids,contextual,parser,
                                    source_hashes,heldout_hashes,code_hash)
        row={'id1':id1,'id2':id2,'label':0,'original_tokens':original_tokens,
             'minimum_reduction':original_tokens-args.max_seq_length,
             'original_code_sha256':{'id1':sha_text(a),'id2':sha_text(b)},
             'original_pair_sha256':canonical_hash([id1,id2,0,a,b]),**result}
        records.append(row)
        if preview is not None:
            eligible+=1
            previews.append({'id1':id1,'id2':id2,'label':0,
               'audit_only_not_training_data':True,'manual_semantic_review_required':True,
               'code_a':preview['code_a'],'code_b':preview['code_b'],
               'original_code_sha256':row['original_code_sha256'],
               'processed_code_sha256':result['processed_code_sha256'],
               'transformed_tokens':result['transformed_tokens'],
               'original_tokens':original_tokens})
    records.sort(key=lambda x:(x['original_tokens'],x['id1'],x['id2']))
    versions={}
    for package in ('tree-sitter','tree-sitter-java'):
        try: versions[package]=version(package)
        except PackageNotFoundError: versions[package]='unknown'
    return {'parser_versions':versions,'baseline_eligible':baseline,'overlength_count':overlength,
      'eligible_proposal_pairs':eligible,'candidate_records':records,'previews':previews,
      'parser_backend':'tree-sitter-java'}
