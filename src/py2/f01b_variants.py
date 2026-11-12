#!/usr/bin/env python3
"""F-01b v1 model-family interfaces; experiment choices come from Shell/CLI.

Only implementation policies live here, not model IDs, revisions, budgets, or
architecture-specific experiment parameters.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPolicy:
    name: str
    yes_text: str
    no_text: str
    expected_yes: str
    expected_no: str


def contextual_continuation(tokenizer, prompt, answer):
    """Return (prompt token IDs, continuation), rejecting token-boundary drift."""
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    joined = tokenizer.encode(prompt + answer, add_special_tokens=False)
    if not prefix or joined[:len(prefix)] != prefix:
        raise ValueError('Prompt prefix is not stable when answer is appended')
    tail = joined[len(prefix):]
    if len(tail) != 1:
        raise ValueError(f'Answer is not one contextual token: {answer!r} => {tail!r}')
    return prefix, tail[0]


def resolve_answer_ids(tokenizer, prompt, policy):
    if policy.name not in ('standalone-one-token', 'contextual-one-token'):
        raise ValueError('Unsupported token policy: ' + policy.name)
    values = {}
    evidence = {}
    for label, answer, expected in (
        (1, policy.yes_text, policy.expected_yes),
        (0, policy.no_text, policy.expected_no),
    ):
        standalone = tokenizer.encode(answer, add_special_tokens=False)
        _, contextual = contextual_continuation(tokenizer, prompt, answer)
        if policy.name == 'standalone-one-token':
            if len(standalone) != 1 or standalone[0] != contextual:
                raise ValueError(f'Standalone/contextual policy mismatch: {answer!r} '
                                 f'{standalone!r} vs {contextual!r}')
            token = standalone[0]
        else:
            token = contextual
        if expected != 'derive':
            try:
                pinned = int(expected)
            except (TypeError, ValueError) as exc:
                raise ValueError('Expected answer token must be numeric or derive') from exc
            if token != pinned:
                raise ValueError(f'Pinned token drift for {answer!r}: {token} != {pinned}')
        values[label] = token
        evidence['YES' if label else 'NO'] = {
            'answer': answer, 'standalone_ids': standalone,
            'contextual_id': contextual, 'selected_id': token,
            'expected_id': expected,
        }
    if values[0] == values[1]:
        raise ValueError('YES and NO token IDs unexpectedly coincide')
    return values, evidence


def checked_training_tokens(tokenizer, prompt, answer, expected_id, max_seq_length):
    prefix, token = contextual_continuation(tokenizer, prompt, answer)
    if token != expected_id:
        raise ValueError(f'Contextual answer drift: expected={expected_id}, actual={token}')
    if len(prefix) + 1 > max_seq_length:
        return None
    return {'input_ids': prefix + [token],
            'attention_mask': [1] * (len(prefix) + 1),
            'labels': [-100] * len(prefix) + [token]}
