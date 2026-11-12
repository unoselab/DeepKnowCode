#!/usr/bin/env python3
"""Conservative Java comment stripper for F-01c audit only.

Never mutate the source dataset. Fail closed on ambiguous Java Unicode escapes,
malformed literals/comments and text blocks. Preserves actual string/char/text
block bytes. Removes comments as whitespace separators, preserving newlines.
Not a proof of Java semantic equivalence (comments can carry directives).
"""
import re

POLICY = 'java-lexer-preserve-literals-v1'
UNICODE_ESCAPE = re.compile(r'\\u+[0-9a-fA-F]{4}')


class UnsafeJavaLexing(ValueError):
    pass


def strip_java_comments(code):
    if not isinstance(code, str):
        raise TypeError('Java code must be a str')
    # Java processes Unicode escapes before lexical analysis; reject instead of
    # incorrectly interpreting escaped comment delimiters, quotes or newlines.
    if UNICODE_ESCAPE.search(code):
        raise UnsafeJavaLexing('Java Unicode escape requires pre-lexical translation')
    result = []
    i = 0
    n = len(code)
    removed = 0
    while i < n:
        if code.startswith('//', i):
            j = i + 2
            while j < n and code[j] not in '\r\n':
                j += 1
            result.append(' ')
            removed += 1
            i = j
            continue
        if code.startswith('/*', i):
            j = code.find('*/', i + 2)
            if j < 0:
                raise UnsafeJavaLexing('Unterminated block comment')
            chunk = code[i:j+2]
            result.append(' ' + ''.join(c for c in chunk if c in '\r\n'))
            removed += 1
            i = j + 2
            continue
        if code.startswith('"""', i):
            j = i + 3
            while j < n:
                if code[j] == '\\':
                    j += 2
                    continue
                if code.startswith('"""', j):
                    j += 3
                    break
                j += 1
            else:
                raise UnsafeJavaLexing('Unterminated Java text block')
            result.append(code[i:j])
            i = j
            continue
        if code[i] in ('"', "'"):
            quote = code[i]
            j = i + 1
            while j < n:
                if code[j] in '\r\n':
                    raise UnsafeJavaLexing('Newline in Java quoted literal')
                if code[j] == '\\':
                    j += 2
                    continue
                if code[j] == quote:
                    j += 1
                    break
                j += 1
            else:
                raise UnsafeJavaLexing('Unterminated Java quoted literal')
            if j > n:
                raise UnsafeJavaLexing('Dangling escape in quoted literal')
            result.append(code[i:j])
            i = j
            continue
        result.append(code[i])
        i += 1
    return ''.join(result), removed
