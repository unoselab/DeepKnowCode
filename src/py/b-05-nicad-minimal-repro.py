#!/usr/bin/env python3
"""B-05 v7 | NiCad Python grammar: minimal reproduction, failure localization, ablation.

Phases
  inventory  NO NiCad: list syntactic constructs per staged file, find constructs that occur only in
             NiCad-failing files, and predict how many files each fix would recover.
  calibrate  Find a safe --workers value: run the same cases at several worker counts,
             verify identical results, measure speedup and memory.
  synthetic  Run a catalog of small single-feature Python sources through NiCad.
  localize   Reduce each NiCad-parse-failing Salt file to a small failing snippet.
  ablate     Test which source rewrites (annotations, async, walrus, ...) restore parsing.
  all        synthetic + localize + ablate

READ ONLY: never modifies the Salt repository, B-01 staging, B-02/B-03/B-04 results,
or the NiCad cache of other experiments (only b05_probe_w<N> paths are touched).
"""
import argparse
import ast
import csv
import hashlib
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

VERSION = 'v7'
EXPERIMENT = 'B-05'
EXPECTED_COMMIT = '0fa7bec68485f8d89b7de2bd6b8743d040fb956b'

# NiCad prints "1 source file failed to parse" (singular) and "N source files ..." (plural).
# B-02 matched only the plural form; both are handled here.
PARSE_RE = re.compile(r'(\d+)\s+source\s+files?\s+failed\s+to\s+parse', re.I)
EXTRACT_RE = re.compile(r'Extracted\s+(\d+)\s+functions?', re.I)
EXCERPT_RE = re.compile(r'warn|error|fail|parse|extract|txl|stack|memory', re.I)
CATS = ('SUCCESS', 'NICAD_PARSE_FAILURE', 'NO_FUNCTIONS_EXTRACTED',
        'OTHER_EXECUTION_ERROR', 'TIMEOUT', 'UNKNOWN')

_print_lock = threading.Lock()


def local_now():
    """Current time in America/Chicago (DST handled by date); falls back to UTC if tz data is missing."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo('America/Chicago'))
    except Exception:
        return datetime.now(timezone.utc)


def P(*a):
    with _print_lock:
        print(*a, flush=True)


# --------------------------------------------------------------------------- NiCad probe
def classify(out, rc, timed_out=False):
    if timed_out:
        return 'TIMEOUT', 0, None
    m = PARSE_RE.findall(out)
    failed = int(m[-1]) if m else 0
    e = EXTRACT_RE.findall(out)
    extracted = int(e[-1]) if e else None
    if failed > 0:
        cat = 'NICAD_PARSE_FAILURE'
    elif extracted == 0:
        cat = 'NO_FUNCTIONS_EXTRACTED'
    elif rc == 0 and extracted:
        cat = 'SUCCESS'
    elif rc != 0:
        cat = 'OTHER_EXECUTION_ERROR'
    else:
        cat = 'UNKNOWN'
    return cat, failed, extracted


class Prober:
    """Runs `nicad functions py <dir> <config>` on one file; owns only b05_probe_w<slot>*."""

    def __init__(self, nicad_home, txl_home, work, slot, timeout, config='default'):
        self.nicad = Path(nicad_home)
        self.txl = Path(txl_home)
        self.work = Path(work)
        self.slot = slot
        self.timeout = timeout
        self.config = config
        self.base = 'b05_probe_w%d' % slot
        self.input_dir = self.work / self.base
        self.cache_root = self.nicad / 'nicadclones'
        self.env = os.environ.copy()
        self.env['PATH'] = str(self.txl / 'bin') + os.pathsep + self.env.get('PATH', '')
        self.memo = {}
        self.calls = 0
        self.invocations = 0
        self.seconds = 0.0

    def _owned(self, p):
        n = p.name
        return n == self.base or n.startswith(self.base + '_') or n.startswith(self.base + '.')

    def _rm(self, p):
        p = Path(p)
        if not self._owned(p):
            raise RuntimeError('refusing to delete non-B-05 path: %s' % p)
        parents = {self.work.resolve(), self.cache_root.resolve()}
        if p.parent.resolve() not in parents:
            raise RuntimeError('refusing to delete outside probe roots: %s' % p)
        if p.is_symlink():
            raise RuntimeError('refusing to delete symlink: %s' % p)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()

    def clean(self):
        for root in (self.work, self.cache_root):
            if not root.is_dir():
                continue
            for p in list(root.iterdir()):
                if self._owned(p):
                    self._rm(p)

    def probe(self, data, filename=None):
        # NiCad's extract.sh writes pyindent output to a FIXED /tmp/<basename>-pyindent path.
        # A per-worker file name keeps concurrent workers from overwriting each other's file.
        if filename is None:
            filename = 'target_w%d.py' % self.slot
        if isinstance(data, str):
            data = data.encode('utf-8')
        self.invocations += 1
        key = hashlib.sha256(data).hexdigest()
        if key in self.memo:
            return self.memo[key]
        self.clean()
        self.input_dir.mkdir(parents=True)
        (self.input_dir / filename).write_bytes(data)
        cmd = [str(self.nicad / 'bin/nicad'), 'functions', 'py', str(self.input_dir), self.config]
        t0 = time.monotonic()
        timed_out = False
        out, rc = '', 0
        try:
            p = subprocess.run(cmd, cwd=str(self.nicad), env=self.env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=self.timeout, errors='replace')
            out, rc = p.stdout, p.returncode
        except subprocess.TimeoutExpired:
            timed_out, rc = True, 124
        finally:
            self.clean()
        sec = time.monotonic() - t0
        cat, failed, extracted = classify(out, rc, timed_out)
        excerpt = ' | '.join(l.strip() for l in out.splitlines() if EXCERPT_RE.search(l))[-500:]
        res = dict(category=cat, failed_count=failed, extracted=extracted, exit=rc,
                   seconds=round(sec, 3), excerpt=excerpt)
        self.memo[key] = res
        self.calls += 1
        self.seconds += sec
        return res


# --------------------------------------------------------------------------- synthetic catalog
def F(body, header='def probe(a, b):', pad=14, pre=None, post=None):
    """Wrap feature lines in a function with >=14 lines so NiCad (minline=10) extracts it."""
    out = list(pre or [])
    out.append(header)
    for l in body:
        out.append(('    ' + l) if l else '')
    for i in range(max(0, pad - len(body))):
        out.append('    pad_%d = a + %d' % (i, i))
    out.append('    return a')
    out.extend(post or [])
    return '\n'.join(out) + '\n'


CASES = []


def add(cid, group, desc, src):
    CASES.append(dict(id=cid, group=group, desc=desc, src=src))


def build_catalog():
    CASES.clear()
    # ---- controls
    add('C00', 'control', 'plain function (must be SUCCESS)',
        F(['x = a + b', 'y = x * 2', 'z = [i for i in range(y)]', 'total = sum(z)',
           'if total > 10:', '    total -= 1', 'else:', '    total += 1', 'w = total']))
    add('C01', 'control', 'plain class method',
        'class Probe:\n' + '\n'.join(
            ['    def method(self, a):'] + ['        v_%d = a + %d' % (i, i) for i in range(12)]
            + ['        return a']) + '\n')
    # ---- strings
    add('S01', 'string', 'f-string simple', F(['s = f"{a} and {b}"']))
    add('S02', 'string', 'f-string format spec', F(['s = f"{a:>10} {b:.2f}"']))
    add('S03', 'string', 'f-string conversion !r', F(['s = f"{a!r} {b!s}"']))
    add('S04', 'string', 'f-string self-doc {a=}', F(['s = f"{a=}"']))
    add('S05', 'string', 'f-string dict subscript old quotes', F(['d = {}', 's = f"{d[\'k\']}"']))
    add('S06', 'string', 'f-string nested same quote (PEP 701, 3.12)', F(['d = {}', 's = f"{d["k"]}"']))
    add('S07', 'string', 'f-string nested f-string', F(['s = f"{f\'{a}\'}"']))
    add('S08', 'string', 'f-string triple-quoted multiline', F(['s = f"""{a}', '{b}"""']))
    add('S09', 'string', 'f-string escaped braces', F(['s = f"{{literal}} {a}"']))
    add('S10', 'string', 'f-string backslash in expr (PEP 701, 3.12)', F(['s = f"{\'\\n\'.join(a)}"']))
    add('S11', 'string', 'rf-string prefix', F(['s = rf"\\d{a}"']))
    add('S12', 'string', 'bytes / rb literals', F(['s = b"abc"', 't = rb"\\x00"']))
    add('S13', 'string', 'raw string backslash', F(['s = r"\\\\d+\\s*"']))
    add('S14', 'string', 'implicit string concatenation', F(['s = ("a"', '     "b")']))
    add('S15', 'string', 'docstring with embedded quotes',
        F(['"""Doc with "quotes" and \'single\'."""']))
    add('S16', 'string', 'non-ASCII string literal', F(['s = "h\u00e9llo \u2013 \u00fc \u4e2d\u6587"']))
    add('S17', 'string', 'non-ASCII identifier', F(['caf\u00e9 = 1']))
    add('S18', 'string', 'percent formatting', F(['s = "%s-%d" % (a, 1)']))
    add('S19', 'string', 'backslash line continuation', F(['s = a + \\', '    b']))
    add('S20', 'string', 'hash char inside string', F(['s = "# not a comment"']))
    add('S21', 'string', 'u-prefix string', F(['s = u"text"']))
    # ---- definitions / decorators / async / annotations
    add('D01', 'def', 'nested function', F(['def inner(x):', '    return x + 1', 'v = inner(a)']))
    add('D02', 'def', 'nested class', F(['class Inner:', '    attr = 1', 'v = Inner()']))
    add('D03', 'def', 'closure with nonlocal',
        F(['count = 0', 'def inc():', '    nonlocal count', '    count += 1', 'inc()']))
    add('D04', 'def', 'lambda with default', F(['fn = lambda x, y=1: x + y']))
    add('D05', 'decorator', 'simple decorator', F(['return a'], header='@dec\ndef probe(a, b):'))
    add('D06', 'decorator', 'decorator with arguments',
        F(['return a'], header='@dec(1, x=2)\ndef probe(a, b):'))
    add('D07', 'decorator', 'dotted decorator', F(['return a'], header='@a.b.c\ndef probe(a, b):'))
    add('D08', 'decorator', 'arbitrary-expression decorator (3.9)',
        F(['return a'], header='@buttons[0].clicked.connect\ndef probe(a, b):'))
    add('D09', 'decorator', 'stacked decorators',
        F(['return a'], header='@one\n@two(3)\n@three.x\ndef probe(a, b):'))
    add('D10', 'decorator', 'class decorator',
        '@dec\nclass Probe:\n' + '\n'.join(['    v_%d = %d' % (i, i) for i in range(12)]) + '\n'
        + '    def m(self):\n' + '\n'.join(['        w_%d = %d' % (i, i) for i in range(10)]) + '\n')
    add('D11', 'async', 'async def', F(['x = a'], header='async def probe(a, b):'))
    add('D12', 'async', 'await expression', F(['x = await a'], header='async def probe(a, b):'))
    add('D13', 'async', 'async for', F(['async for x in a:', '    y = x'], header='async def probe(a, b):'))
    add('D14', 'async', 'async with', F(['async with a as x:', '    y = x'], header='async def probe(a, b):'))
    add('D15', 'async', 'async comprehension', F(['x = [i async for i in a]'], header='async def probe(a, b):'))
    add('D16', 'async', 'async method in class',
        'class Probe:\n' + '\n'.join(
            ['    async def method(self, a):'] + ['        v_%d = a + %d' % (i, i) for i in range(12)]
            + ['        return await a']) + '\n')
    add('D17', 'annotation', 'param + return annotations',
        F(['x = a'], header='def probe(a: int, b: str = "x") -> bool:'))
    add('D18', 'annotation', 'return-only annotation', F(['x = a'], header='def probe(a, b) -> int:'))
    add('D19', 'annotation', 'variable annotations', F(['x: int = 1', 'y: List[int]', 'z: "Foo" = None']))
    add('D20', 'annotation', '*args/**kwargs annotations',
        F(['x = a'], header='def probe(a, *args: int, **kwargs: str):'))
    add('D21', 'annotation', 'string forward reference', F(['x = a'], header='def probe(a: "Foo") -> "Bar":'))
    add('D22', 'annotation', 'union pipe (3.10)', F(['x = a'], header='def probe(a: int | None, b: str | bytes = None):'))
    add('D23', 'annotation', 'subscripted generics',
        F(['x = a'], header='def probe(a: Dict[str, List[int]]) -> Optional[Callable[[int], str]]:'))
    add('D24', 'signature', 'positional-only marker /', F(['x = a'], header='def probe(a, /, b):'))
    add('D25', 'signature', 'keyword-only marker *', F(['x = a'], header='def probe(a, *, b):'))
    add('D26', 'signature', 'keyword-only with default', F(['x = a'], header='def probe(*, a=1, b=2):'))
    add('D27', 'pep695', 'generic function def f[T] (3.12)', F(['x = a'], header='def probe[T](a: T) -> T:'))
    add('D28', 'pep695', 'type alias statement (3.12)',
        F(['x = a'], pre=['type Alias = int']))
    add('D29', 'pep695', 'generic class C[T] (3.12)',
        'class Probe[T]:\n    def method(self, a):\n'
        + '\n'.join(['        v_%d = a + %d' % (i, i) for i in range(12)]) + '\n        return a\n')
    add('D30', 'call', 'star args in call', F(['r = fn(*a, **b)']))
    add('D31', 'generator', 'yield', F(['yield a', 'yield b']))
    add('D32', 'generator', 'yield from', F(['yield from a']))
    # ---- expressions / statements
    add('E01', 'walrus', 'walrus in if', F(['if (n := len(a)) > 1:', '    x = n']))
    add('E02', 'walrus', 'walrus in while', F(['while (line := a.read()):', '    x = line']))
    add('E03', 'walrus', 'walrus in comprehension', F(['xs = [y for x in a if (y := x)]']))
    add('E04', 'walrus', 'walrus as call argument', F(['r = fn(y := 1)']))
    add('E05', 'match', 'match/case statement (3.10)',
        F(['match a:', '    case 1:', '        x = 1', '    case _:', '        x = 2']))
    add('E06', 'match', 'soft keyword `match`/`case` as identifiers',
        F(['match = re.match(r"x", a)', 'case = 1', 'if match:', '    x = match.group(0)']))
    add('E07', 'match', '`type` as identifier', F(['type = 1', 'x = type(a)']))
    add('E08', 'stmt', 'try/except* (3.11)',
        F(['try:', '    x = 1', 'except* ValueError:', '    x = 2']))
    add('E09', 'stmt', 'parenthesized with (3.10)',
        F(['with (open("a") as f, open("b") as g):', '    x = f']))
    add('E10', 'stmt', 'multiple with items', F(['with open("a") as f, open("b") as g:', '    x = f']))
    add('E11', 'expr', 'star assignment', F(['x, *y = a']))
    add('E12', 'expr', 'star in subscript (3.11)', F(['x = a[*b]']))
    add('E13', 'expr', 'matmul operators', F(['x = a @ b', 'x @= b']))
    add('E14', 'expr', 'Ellipsis', F(['x = ...']))
    add('E15', 'expr', 'dict/set comprehensions', F(['d = {k: v for k, v in a}', 's = {k for k in a}']))
    add('E16', 'expr', 'generator expression', F(['g = (x for x in a)']))
    add('E17', 'expr', 'conditional expression', F(['x = a if b else None']))
    add('E18', 'expr', 'chained comparison', F(['x = 0 < a <= b != 3']))
    add('E19', 'expr', 'numeric literals (underscore, complex, hex)', F(['x = 1_000 + 0xFF + 0o7 + 0b1 + 1j']))
    add('E20', 'stmt', 'global statement', F(['global G', 'G = a']))
    add('E21', 'stmt', 'del statement', F(['x = [1]', 'del x[0]']))
    add('E22', 'stmt', 'assert with message', F(['assert a, "message"']))
    add('E23', 'stmt', 'raise ... from', F(['raise ValueError("x") from a']))
    add('E24', 'stmt', 'print / exec as functions', F(['print(a)', 'exec("x = 1")']))
    add('E25', 'expr', 'slices', F(['x = a[1:2, ::3]']))
    add('E26', 'expr', 'dict and list unpacking', F(['d = {**a, "k": 1}', 'l = [*a, *b]']))
    add('E27', 'stmt', 'semicolon-separated statements', F(['x = 1; y = 2']))
    add('E28', 'stmt', 'one-line compound statement', F(['if a: x = 1', 'for i in a: y = i']))
    add('E29', 'stmt', 'for/while else', F(['for i in a:', '    x = i', 'else:', '    x = 0']))
    add('E30', 'stmt', 'try/except tuple as e',
        F(['try:', '    x = 1', 'except (ValueError, TypeError) as e:', '    x = e']))
    add('E31', 'stmt', 'try/finally + else',
        F(['try:', '    x = 1', 'except Exception:', '    x = 2', 'else:', '    x = 3', 'finally:', '    y = 4']))
    add('E32', 'stmt', 'import forms',
        F(['import os', 'import os.path as osp', 'from . import x', 'from .. import y as z',
           'from a.b import (c, d)']))
    add('E33', 'stmt', '__future__ import',
        F(['x = a'], pre=['from __future__ import annotations']))
    add('E34', 'stmt', 'main guard block',
        F(['x = a'], post=['if __name__ == "__main__":', '    probe(1, 2)']))
    add('E35', 'stmt', 'conditional lambda/ternary chain', F(['f = lambda x: 1 if x else 2 if a else 3']))
    # ---- file-level / lexical
    base = F(['x = a + b', 'y = x * 2', 'z = [i for i in range(y)]', 'total = sum(z)'])
    add('L01', 'lexical', 'CRLF line endings', base.replace('\n', '\r\n'))
    add('L02', 'lexical', 'no trailing newline', base.rstrip('\n'))
    add('L03', 'lexical', 'UTF-8 BOM', ('\ufeff' + base).encode('utf-8'))
    add('L04', 'lexical', 'tab indentation', base.replace('    ', '\t'))
    add('L05', 'lexical', 'coding cookie + shebang', '#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n' + base)
    add('L06', 'lexical', 'type-comment', F(['x = a  # type: int', 'y = b  # type: ignore']))
    add('L07', 'lexical', 'form feed character', base.replace('def probe', '\x0c\ndef probe', 1))
    add('L08', 'lexical', 'trailing whitespace and blank lines', base.replace('\n', '   \n\n'))
    add('L09', 'lexical', 'comment-only lines inside function', F(['# comment', 'x = a', '# another']))
    add('L10', 'lexical', 'very long line (4000 chars)', F(['x = "%s"' % ('a' * 4000)]))
    add('L11', 'lexical', 'large file (600 functions)',
        '\n'.join(base.replace('def probe', 'def probe_%d' % i) for i in range(600)))
    # ---- v4: constructs discovered in the 282 real failing files
    def CLS(header, extra=None):
        body = ['    def method(self, a):'] + ['        v_%d = a + %d' % (i, i) for i in range(12)] + ['        return a']
        return '\n'.join([header] + (extra or []) + body) + '\n'
    add('E36', 'set', 'set literal {a, b}', F(['s = {a, b}']))
    add('E37', 'set', 'set literal single element {a}', F(['s = {a}']))
    add('E38', 'set', 'frozenset({"a", "b"})', F(['s = frozenset({"a", "b"})']))
    add('E44', 'set', 'set comprehension only', F(['s = {x for x in a}']))
    add('E45', 'set', 'dict comprehension only', F(['d = {k: v for k, v in a}']))
    add('E40', 'expr', 'dict unpack {**a} only', F(['d = {**a}']))
    add('E41', 'expr', 'list unpack [*a] only', F(['l = [*a]']))
    add('E39', 'expr', 'star arg after keyword in call', F(['r = fn(a=1, *b)']))
    add('E42', 'expr', 'nested list comprehension', F(['x = [[j for j in i] for i in a]']))
    add('T01', 'trailing', 'def plain trailing comma', F(['x = a'], header='def probe(a, b,):'))
    add('T02', 'trailing', 'def *args trailing comma', F(['x = a'], header='def probe(a, *args,):'))
    add('T03', 'trailing', 'def **kwargs trailing comma', F(['x = a'], header='def probe(a, **kwargs,):'))
    add('T04', 'trailing', 'def multi-line black style with **kwargs,',
        F(['x = a'], header='def probe(\n    a,\n    b=None,\n    **kwargs,\n):'))
    add('T05', 'trailing', 'call trailing comma', F(['r = fn(a, b,)']))
    add('T06', 'trailing', 'call **kwargs trailing comma', F(['r = fn(a, **b,)']))
    add('T07', 'trailing', 'tuple / list / dict trailing commas', F(['t = (1, 2,)', 'l = [1, 2,]', 'd = {1: 2,}']))
    add('T08', 'trailing', 'parenthesized import trailing comma', F(['from os import (path, sep,)']))
    add('T09', 'trailing', 'multi-line def plain params trailing comma',
        F(['x = a'], header='def probe(\n    a,\n    b=None,\n):'))
    add('P01', 'strprefix', 'b"" prefix', F(['s = b"abc"']))
    add('P02', 'strprefix', 'rb"" prefix', F(['s = rb"abc"']))
    add('P03', 'strprefix', 'br"" prefix', F(['s = br"abc"']))
    add('P04', 'strprefix', 'rf"" prefix', F(['s = rf"{a}"']))
    add('P05', 'strprefix', 'fr"" prefix', F(['s = fr"{a}"']))
    add('P06', 'strprefix', 'uppercase prefixes B R F', F(['s = B"a"', 't = R"b"', 'u = F"{a}"']))
    add('P07', 'strprefix', 'r"" prefix alone', F(['s = r"abc"']))
    add('P08', 'strprefix', 'f"" prefix alone (reference)', F(['s = f"{a}"']))
    add('N01', 'number', 'underscore number 1_000', F(['x = 1_000']))
    add('N02', 'number', 'hex literal 0xFF', F(['x = 0xFF']))
    add('N03', 'number', 'octal literal 0o17', F(['x = 0o17']))
    add('N04', 'number', 'binary literal 0b101', F(['x = 0b101']))
    add('N05', 'number', 'complex literal 1j', F(['x = 1j']))
    add('N06', 'number', 'float exponent', F(['x = 1e5 + 2.5E-3']))
    add('K01', 'class', 'class keyword arg (metaclass=)', CLS('class Probe(metaclass=Meta):'))
    add('K02', 'class', 'class bases + metaclass=', CLS('class Probe(Base, metaclass=Meta):'))
    add('K03', 'class', 'class bases trailing comma', CLS('class Probe(Base,):'))
    add('K04', 'class', 'class annotated attribute (x: int = 0)', CLS('class Probe:', ['    attr: int = 0']))
    add('D33', 'signature', 'keyword-only after *args', F(['x = a'], header='def probe(*args, b=2):'))
    add('D34', 'annotation', 'variable annotation inside function only', F(['x: int = 1']))
    add('E46', 'callorder', 'call: *args then keyword', F(['r = fn(*a, k=1)']))
    add('E47', 'callorder', 'call: *args then positional', F(['r = fn(*a, b)']))
    add('E48', 'callorder', 'call: **kw then keyword', F(['r = fn(**a, k=1)']))
    add('E49', 'callorder', 'call: positional, *args, positional, keyword', F(['r = fn(a, *b, c, k=1)']))
    add('E50', 'header', 'comment after def header colon', F(['x = a'], header='def probe(a, b):  # comment'))
    add('E51', 'header', 'multi-line def, comment after ):',
        F(['x = a'], header='def probe(\n    a, b\n):  # pylint: disable=x'))
    # ---- v5: call-argument shapes, decorator/annotation shapes, walrus contexts
    add('A01', 'callshape', 'positional, *star', F(['r = fn(a, *b)']))
    add('A02', 'callshape', 'positional, *star, positional', F(['r = fn(a, *b, c)']))
    add('A03', 'callshape', 'positional, *star, keyword', F(['r = fn(a, *b, k=1)']))
    add('A04', 'callshape', 'positional, **dstar, keyword', F(['r = fn(a, **b, k=1)']))
    add('A05', 'callshape', 'positional, *star, **dstar', F(['r = fn(a, *b, **c)']))
    add('A06', 'callshape', 'positional, keyword, *star', F(['r = fn(a, k=1, *b)']))
    add('A07', 'callshape', 'two *star arguments', F(['r = fn(*a, *b)']))
    add('A08', 'callshape', '*(parenthesized expr) only', F(['r = fn(*(a or ()))']))
    add('A09', 'callshape', 'positional, *(parenthesized expr), positional', F(['r = fn(a, *(b or ()), c)']))
    add('A10', 'callshape', 'two positional, *star, two positional', F(['r = fn(a, b, *c, d, e)']))
    add('A11', 'callshape', 'positional, keyword, **dstar (baseline)', F(['r = fn(a, b=1, **c)']))
    add('A12', 'callshape', 'positional, *star, keyword, **dstar', F(['r = fn(a, *b, c=1, **d)']))
    add('H01', 'header', 'annotated param + trailing comma, multi-line', F(['x = a'], header='def probe(\n    a: int,\n):'))
    add('H02', 'header', 'annotated param with default + trailing comma',
        F(['x = a'], header='def probe(\n    a: int,\n    b: str = "x",\n):'))
    add('H03', 'header', 'decorator: dotted name + call', F(['x = a'], header='@a.b(name="x")\ndef probe(a, b):'))
    add('H04', 'header', 'decorator: multi-line call, nested dict, trailing commas',
        F(['x = a'], header='@a.b(\n    name="s",\n    arguments={\n        "p": {\n            "h": "x",\n        },\n    },\n)\ndef probe(a, b):'))
    add('H05', 'header', 'def with annotated params + default + trailing comma (ci_failure shape)',
        F(['x = a'], header='def probe(\n    ctx: Context,\n    pr: int,\n    repository: str = "x/y",\n):'))
    add('H06', 'header', 'whole ci_failure shape: H04 decorator + H05 def',
        F(['x = a'], header='@a.b(\n    name="s",\n    arguments={\n        "p": {\n            "h": "x",\n        },\n    },\n)\ndef probe(\n    ctx: Context,\n    pr: int,\n    repository: str = "x/y",\n):'))
    add('H07', 'header', 'nested dict with trailing commas as call keyword value',
        F(['r = fn(name="s", arguments={"p": {"h": "x",},},)']))
    add('H08', 'header', 'decorator: simple name, multi-line call', F(['x = a'], header='@dec(\n    name="x",\n)\ndef probe(a, b):'))
    add('W01', 'walrus', 'bare walrus as while test', F(['while x := fn():', '    y = x']))
    add('W02', 'walrus', 'bare walrus as if test', F(['if x := fn():', '    y = x']))
    add('W03', 'walrus', 'parenthesized walrus as if test', F(['if (x := fn()):', '    y = x']))
    add('W04', 'walrus', 'bare walrus in if + elif', F(['if x := fn():', '    y = x', 'elif z := fn():', '    y = z']))
    add('W05', 'walrus', 'walrus inside boolean expression in if', F(['if a and (x := fn()):', '    y = x']))
    add('W06', 'walrus', 'walrus as assignment value', F(['y = (x := fn())']))
    add('L12', 'lexical', 'non-ASCII only in a comment', F(['# héllo 中文', 'x = a']))
    add('L13', 'lexical', 'non-ASCII only in a docstring', F(['"""héllo 中文"""', 'x = a']))
    # ---- v7: attribute/name `print` (leaf statement of the localized ci_failure.py was ctx.print(""))
    add('Y01', 'attrname', 'method call named print: ctx.print("")', F(['ctx.print("")']))
    add('Y02', 'attrname', 'bare print("x") call', F(['print("x")']))
    add('Y03', 'attrname', 'attribute read named print', F(['r = obj.print']))
    add('Y04', 'attrname', 'method call named exec', F(['obj.exec("x")']))
    add('Y05', 'attrname', 'ci_failure shape: dotted decorator + annotated def + ctx.print("")',
        F(['ctx.print("")'], header='@grp.command(\n    name="summary",\n    arguments={\n        "pr": {\n            "help": "x",\n        },\n    },\n)\ndef probe(\n    ctx: Context,\n    pr: int,\n    repository: str = "a/b",\n):'))
    # ---- realistic combinations
    add('R01', 'combo', 'decorator + async + annotations',
        F(['x = await a'], header='@dec(1)\nasync def probe(a: int, b: str = "x") -> bool:'))
    add('R02', 'combo', 'method with decorator + annotations in class',
        'class Probe:\n    @staticmethod\n    def method(a: int, b: str = "") -> None:\n'
        + '\n'.join('        v_%d = f"{a}-{b}"' % i for i in range(12)) + '\n')
    return CASES


def py_ast_status(data):
    try:
        ast.parse(data)
        return 'PASS', ''
    except (SyntaxError, ValueError, UnicodeError) as e:
        return 'FAIL', str(e)[:200]


def run_synthetic(a, out, probers):
    cases = build_catalog()
    if a.only:
        wanted = {x.strip() for x in a.only.split(',') if x.strip()}
        cases = [c for c in cases if c['id'] in wanted or c['group'] in wanted]
    if a.max_cases > 0:
        cases = cases[:a.max_cases]
    sdir = out / 'synthetic'
    (sdir / 'cases').mkdir(parents=True, exist_ok=True)
    P('')
    P('=' * 96)
    P('[SYNTHETIC] %d single-feature cases | NiCad config=%s | workers=%d' % (len(cases), a.config, len(probers)))
    P('=' * 96)
    P('%-5s %-10s %-4s %-24s %5s  %s' % ('ID', 'Group', 'AST', 'NiCad category', 'Func', 'Description'))
    q = queue.Queue()
    for p in probers:
        q.put(p)

    def work(c):
        prober = q.get()
        try:
            data = c['src'].encode('utf-8') if isinstance(c['src'], str) else c['src']
            (sdir / 'cases' / ('%s.py' % c['id'])).write_bytes(data)
            ast_s, ast_e = py_ast_status(data)
            r = prober.probe(data)
            row = dict(id=c['id'], group=c['group'], description=c['desc'], python_ast=ast_s,
                       python_error=ast_e, nicad_category=r['category'], nicad_failed_count=r['failed_count'],
                       nicad_extracted_functions=r['extracted'], nicad_exit=r['exit'],
                       seconds=r['seconds'], nicad_excerpt=r['excerpt'], size_bytes=len(data))
            P('%-5s %-10s %-4s %-24s %5s  %s' % (c['id'], c['group'], ast_s, r['category'],
                                                  '' if r['extracted'] is None else r['extracted'], c['desc']))
            return row
        finally:
            q.put(prober)

    with ThreadPoolExecutor(max_workers=len(probers)) as ex:
        rows = list(ex.map(work, cases))
    fields = ['id', 'group', 'description', 'python_ast', 'python_error', 'nicad_category',
              'nicad_failed_count', 'nicad_extracted_functions', 'nicad_exit', 'seconds',
              'size_bytes', 'nicad_excerpt']
    write_csv(sdir / 'synthetic_results.csv', rows, fields)
    ctrl = [r for r in rows if r['id'] == 'C00']
    ctrl_ok = (not ctrl) or ctrl[0]['nicad_category'] == 'SUCCESS'
    bad = [r for r in rows if r['nicad_category'] == 'NICAD_PARSE_FAILURE' and r['python_ast'] == 'PASS']
    by_group = defaultdict(lambda: Counter())
    for r in rows:
        by_group[r['group']][r['nicad_category']] += 1
    P('')
    P('[SYNTHETIC SUMMARY] categories: ' + ', '.join('%s=%d' % kv for kv in
                                                    sorted(Counter(r['nicad_category'] for r in rows).items())))
    if not ctrl_ok:
        P('!! CONTROL C00 DID NOT SUCCEED (%s): the NiCad harness/environment is suspect; '
          'ignore the feature results below.' % ctrl[0]['nicad_category'])
    P('')
    P('[NiCad PARSE FAILURES on syntax that Python %s accepts] %d' % (sys.version.split()[0], len(bad)))
    for r in bad:
        P('  %-5s %-10s %s' % (r['id'], r['group'], r['description']))
    nof = [r for r in rows if r['nicad_category'] == 'NO_FUNCTIONS_EXTRACTED']
    if nof:
        P('[NO_FUNCTIONS_EXTRACTED despite a >=14-line function] %d (feature may hide the function from the extractor)'
          % len(nof))
        for r in nof:
            P('  %-5s %-10s %s' % (r['id'], r['group'], r['description']))
    other = [r for r in rows if r['nicad_category'] in ('OTHER_EXECUTION_ERROR', 'TIMEOUT', 'UNKNOWN')]
    if other:
        P('[OTHER / TIMEOUT / UNKNOWN] %d' % len(other))
        for r in other:
            P('  %-5s %-10s %-22s %s' % (r['id'], r['group'], r['nicad_category'], r['description']))
    return dict(cases=len(rows), categories=dict(Counter(r['nicad_category'] for r in rows)),
                unsupported=[r['id'] for r in bad], control_ok=ctrl_ok)


# --------------------------------------------------------------------------- calibration
def mem_available_mb():
    try:
        for l in Path('/proc/meminfo').read_text().splitlines():
            if l.startswith('MemAvailable:'):
                return int(l.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return None


def run_calibrate(a, out, nicad, txl, work):
    import resource
    levels = sorted({int(x) for x in a.calibrate_workers.split(',') if x.strip()})
    if not levels or levels[0] < 1:
        raise SystemExit('ERROR: --calibrate-workers needs positive integers, e.g. 1,2,4,8')
    if 1 not in levels:
        levels.insert(0, 1)  # workers=1 is the reference for result equivalence
    cases = build_catalog()
    seen, uniq = set(), []
    for c in cases:  # dedupe by content: the per-prober memo must not hide NiCad calls
        if c['id'] == 'L11':  # one ~50s job would bound wall time at every worker count
            continue
        data = c['src'].encode('utf-8') if isinstance(c['src'], str) else c['src']
        h = hashlib.sha256(data).hexdigest()
        if h not in seen:
            seen.add(h)
            uniq.append((c, data))
    n = min(a.calibrate_cases, len(uniq))
    step = len(uniq) / float(n)
    chosen = [uniq[int(i * step)] for i in range(n)]
    # always include the control
    ids = {c['id'] for c, _ in chosen}
    for want in ('C00',):
        if want not in ids:
            for c, d in uniq:
                if c['id'] == want:
                    chosen.append((c, d))
    ncpu = os.cpu_count() or 1
    P('')
    P('=' * 96)
    P('[CALIBRATE] %d cases | worker levels %s | cpu_count=%d | MemAvailable=%s MB | loadavg=%s'
      % (len(chosen), levels, ncpu, mem_available_mb(),
         ' '.join('%.2f' % x for x in os.getloadavg())))
    P('=' * 96)
    ref = {}
    rows = []
    for w in levels:
        probers = [Prober(nicad, txl, work, i, a.probe_timeout, a.config) for i in range(w)]
        for p in probers:
            p.clean()
        q = queue.Queue()
        for p in probers:
            q.put(p)
        rss_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

        def one(item):
            c, data = item
            pr = q.get()
            try:
                r = pr.probe(data)
                return c['id'], r['category'], r['extracted'], r['seconds']
            finally:
                q.put(pr)

        t0 = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=w) as ex:
                res = list(ex.map(one, chosen))
        finally:
            wall = time.monotonic() - t0
            for p in probers:
                p.clean()
        rss_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss  # largest single child so far
        leftovers = [p.name for p in (nicad / 'nicadclones').iterdir() if p.name.startswith('b05_probe_w')] \
            if (nicad / 'nicadclones').is_dir() else []
        leftovers += [p.name for p in work.iterdir() if p.name.startswith('b05_probe_w')]
        got = {cid: (cat, ext) for cid, cat, ext, _ in res}
        if w == levels[0]:
            ref = dict(got)
        diffs = sorted(cid for cid in got if got[cid] != ref.get(cid))
        timeouts = sum(1 for _, cat, _, _ in res if cat == 'TIMEOUT')
        base_wall = rows[0]['wall_seconds'] if rows else wall
        speedup = base_wall / wall if wall > 0 else 0.0
        rows.append(dict(workers=w, cases=len(res), wall_seconds=round(wall, 2),
                         throughput_per_min=round(60.0 * len(res) / wall, 1) if wall > 0 else 0,
                         speedup=round(speedup, 2), efficiency=round(speedup / w, 2),
                         result_mismatches=len(diffs), mismatch_ids=';'.join(diffs),
                         timeouts=timeouts, leftover_probe_dirs=len(leftovers),
                         max_child_rss_mb=round(rss_kb / 1024.0, 1),
                         est_peak_mem_mb=round(w * rss_kb / 1024.0, 1),
                         mem_available_mb=mem_available_mb(),
                         loadavg1=round(os.getloadavg()[0], 2)))
        r = rows[-1]
        P('workers=%-3d wall=%7.1fs  %6.1f cases/min  speedup=%.2fx  eff=%.2f  mismatches=%d  timeouts=%d  '
          'leftovers=%d  maxRSS/proc=%.0fMB  load1=%.1f'
          % (w, wall, r['throughput_per_min'], speedup, r['efficiency'], len(diffs), timeouts,
             len(leftovers), r['max_child_rss_mb'], r['loadavg1']))
        if diffs:
            P('   !! results differ from workers=%d for: %s' % (levels[0], ', '.join(diffs)))
    fields = ['workers', 'cases', 'wall_seconds', 'throughput_per_min', 'speedup', 'efficiency',
              'result_mismatches', 'mismatch_ids', 'timeouts', 'leftover_probe_dirs',
              'max_child_rss_mb', 'est_peak_mem_mb', 'mem_available_mb', 'loadavg1']
    write_csv(out / 'calibration.csv', rows, fields)
    ctrl_ok = ref.get('C00', ('SUCCESS',))[0] == 'SUCCESS'
    mem0 = mem_available_mb()
    safe = []
    for r in rows:
        ok = (r['result_mismatches'] == 0 and r['timeouts'] == 0 and r['leftover_probe_dirs'] == 0
              and r['efficiency'] >= a.min_efficiency
              and (mem0 is None or r['est_peak_mem_mb'] <= 0.7 * mem0)
              and r['workers'] <= ncpu)
        if ok:
            safe.append(r['workers'])
    rec = max(safe) if safe else 1
    P('')
    if not ctrl_ok:
        P('!! Control C00 did not succeed at workers=1: environment suspect, do not trust this calibration.')
    P('[RECOMMENDATION] --workers %d' % rec)
    P('  Criteria: identical results to workers=%d, no timeouts, no leftover probe dirs, efficiency >= %.2f, '
      'workers <= cpu_count (%d), est. memory <= 70%% of MemAvailable.' % (levels[0], a.min_efficiency, ncpu))
    P('  Efficiency = speedup/workers. Falling efficiency means CPU/IO contention; any mismatch or leftover means '
      'NiCad/TXL shares state between runs, so use fewer workers.')
    P('  Run calibration while the server is otherwise quiet; loadavg above is part of the evidence.')
    return dict(levels=rows, recommended_workers=rec, cpu_count=ncpu, control_ok=ctrl_ok)


# --------------------------------------------------------------------------- construct inventory (no NiCad)
import keyword as _kw
import tokenize as _tok

_SKIP_TOK = {_tok.NL, _tok.COMMENT, _tok.NEWLINE, _tok.INDENT, _tok.DEDENT, _tok.ENCODING, _tok.ENDMARKER}


def _decorator_is_simple(d):
    if isinstance(d, ast.Call):
        d = d.func
    while isinstance(d, ast.Attribute):
        d = d.value
    return isinstance(d, ast.Name)


def _match_close(sig, i):
    depth = 0
    for j in range(i, len(sig)):
        if sig[j].type == _tok.OP and sig[j].string in '([{' and len(sig[j].string) == 1:
            depth += 1
        elif sig[j].type == _tok.OP and sig[j].string in ')]}' and len(sig[j].string) == 1:
            depth -= 1
            if depth == 0:
                return j
    return None


def inventory_constructs(text):
    """Set of syntactic constructs in `text`; None if Python itself cannot parse it."""
    c = set()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    bare_tests = set()
    src_lines = text.split('\n')

    def _preceded_by_paren(node):
        line = src_lines[node.lineno - 1].encode('utf-8')[:node.col_offset].decode('utf-8', 'replace')
        if line.strip():
            return line.rstrip().endswith('(')
        k = node.lineno - 2
        while k >= 0 and not src_lines[k].strip():
            k -= 1
        return k >= 0 and src_lines[k].rstrip().endswith('(')

    for n in ast.walk(tree):
        if isinstance(n, (ast.If, ast.While)) and isinstance(n.test, ast.NamedExpr) \
                and not _preceded_by_paren(n.test):
            bare_tests.add(id(n.test))  # `if x := f():` ; parenthesized forms count as walrus_nested
    for n in ast.walk(tree):
        if isinstance(n, ast.DictComp):
            c.add('dict_comprehension')
        elif isinstance(n, ast.SetComp):
            c.add('set_comprehension')
        elif isinstance(n, ast.Set):
            c.add('set_literal')
        elif isinstance(n, ast.YieldFrom):
            c.add('yield_from')
        elif isinstance(n, ast.Await):
            c.add('await_expr')
        elif isinstance(n, ast.AnnAssign):
            c.add('variable_annotation')
        elif isinstance(n, ast.NamedExpr):
            c.add('walrus_bare_test' if id(n) in bare_tests else 'walrus_nested')
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult):
            c.add('matmul')
        elif isinstance(n, ast.AugAssign) and isinstance(n.op, ast.MatMult):
            c.add('matmul')
        elif isinstance(n, ast.Constant) and n.value is Ellipsis:
            c.add('ellipsis')
        elif isinstance(n, ast.Raise) and n.cause is not None:
            c.add('raise_from')
        elif isinstance(n, ast.With) and len(n.items) > 1:
            c.add('multi_item_with')
        elif isinstance(n, ast.AsyncFor):
            c.add('async_for')
        elif isinstance(n, ast.AsyncWith):
            c.add('async_with')
        elif isinstance(n, ast.comprehension) and n.is_async:
            c.add('async_comprehension')
        elif isinstance(n, ast.JoinedStr):
            c.add('fstring')
        elif hasattr(ast, 'Match') and isinstance(n, ast.Match):
            c.add('match_stmt')
        elif hasattr(ast, 'TryStar') and isinstance(n, ast.TryStar):
            c.add('except_star')
        elif hasattr(ast, 'TypeAlias') and isinstance(n, ast.TypeAlias):
            c.add('type_alias_stmt')
        elif isinstance(n, ast.Dict) and any(k is None for k in n.keys):
            c.add('dict_unpack_display')
        elif isinstance(n, (ast.List, ast.Tuple)) and isinstance(getattr(n, 'ctx', None), ast.Store) \
                and any(isinstance(e, ast.Starred) for e in n.elts):
            c.add('star_assignment')
        elif isinstance(n, (ast.List, ast.Tuple, ast.Set)) and isinstance(getattr(n, 'ctx', None), ast.Load) \
                and any(isinstance(e, ast.Starred) for e in n.elts):
            c.add('star_in_display')
        elif isinstance(n, ast.Subscript) and any(isinstance(x, ast.Starred) for x in ast.walk(n.slice)):
            c.add('star_in_subscript')
        if isinstance(n, ast.Call):
            pos_ = lambda x: (x.lineno, x.col_offset)
            stars = [x for x in n.args if isinstance(x, ast.Starred)]
            if stars:
                fs = min(pos_(x) for x in stars)
                before = [x for x in n.args if not isinstance(x, ast.Starred) and pos_(x) < fs]
                after = [x for x in n.args if not isinstance(x, ast.Starred) and pos_(x) > fs]
                named_after = [k for k in n.keywords if k.arg is not None and pos_(k) > fs]
                if before:
                    c.add('call_pos_then_star')
                    if after:
                        c.add('call_pos_star_pos')
                    if named_after:
                        c.add('call_pos_star_kw')
                if any(pos_(x) > fs for x in n.args if not isinstance(x, ast.Starred)):
                    c.add('call_star_then_positional')
                for k in n.keywords:
                    if k.arg is not None:
                        c.add('call_star_then_keyword' if pos_(k) > fs else 'call_keyword_then_star')
            dst = [k for k in n.keywords if k.arg is None]
            if dst:
                fd = min(pos_(k) for k in dst)
                if any(k.arg is not None and pos_(k) > fd for k in n.keywords):
                    c.add('call_dstar_then_keyword')
                    if any(pos_(x) < fd for x in n.args):
                        c.add('call_pos_dstar_kw')
        if isinstance(n, ast.Attribute) and n.attr in ('print', 'exec'):
            c.add('attr_name_print' if n.attr == 'print' else 'attr_name_exec')
        if isinstance(n, ast.ClassDef):
            if n.keywords:
                c.add('class_keyword_arg')
            if any(not _decorator_is_simple(d) for d in n.decorator_list):
                c.add('decorator_expression')
            if getattr(n, 'type_params', None):
                c.add('pep695_generics')
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if isinstance(n, ast.AsyncFunctionDef):
                c.add('async_def')
            if any(not _decorator_is_simple(d) for d in n.decorator_list):
                c.add('decorator_expression')
            if any(isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) for d in n.decorator_list):
                c.add('decorator_dotted_call')
            if getattr(n, 'type_params', None):
                c.add('pep695_generics')
            a_ = n.args
            if a_.posonlyargs:
                c.add('positional_only')
            if a_.kwonlyargs:
                c.add('kwonly_after_vararg' if a_.vararg else 'bare_star_kwonly')
            for x in (a_.vararg, a_.kwarg):
                if x is not None and x.annotation is not None:
                    c.add('varargs_annotation')
    # ---- token level
    try:
        toks = list(_tok.generate_tokens(io.StringIO(text).readline))
    except (_tok.TokenError, IndentationError, SyntaxError):
        return c
    fstart = getattr(_tok, 'FSTRING_START', None)
    fend = getattr(_tok, 'FSTRING_END', None)
    fq = []
    for t in toks:
        if fstart is not None and t.type == fstart:
            q = re.sub(r'^[A-Za-z]*', '', t.string)
            if fq and len(fq[-1]) == 1 and q[:1] == fq[-1]:
                c.add('fstring_nested_same_quote')
            fq.append(q)
        elif fend is not None and t.type == fend:
            if fq:
                fq.pop()
        elif t.type == _tok.STRING and fq:
            q = re.sub(r'^[A-Za-z]*', '', t.string)[:1]
            if len(fq[-1]) == 1 and q == fq[-1]:
                c.add('fstring_nested_same_quote')
    for t in toks:
        if t.type == _tok.STRING or (fstart is not None and t.type == fstart):
            m = re.match(r'[A-Za-z]*', t.string)
            pre = m.group().lower() if m else ''
            if pre:
                c.add('str_prefix_' + pre)
            if t.type == _tok.STRING and not t.string.isascii():
                c.add('non_ascii_string')
        elif t.type == _tok.NUMBER:
            x = t.string.lower()
            if '_' in x:
                c.add('num_underscore')
            if x.startswith('0x'):
                c.add('num_hex')
            elif x.startswith('0o'):
                c.add('num_octal')
            elif x.startswith('0b'):
                c.add('num_binary')
            elif x.endswith('j'):
                c.add('num_complex')
            elif 'e' in x:
                c.add('num_exponent')
        elif t.type == _tok.NAME and not t.string.isascii():
            c.add('non_ascii_identifier')
        elif t.type == _tok.COMMENT and not t.string.isascii():
            c.add('non_ascii_comment')
    sig = [t for t in toks if t.type not in _SKIP_TOK]
    stack = []
    for i, t in enumerate(sig):
        s_ = t.string
        is_op = t.type == _tok.OP
        if is_op and s_ in ('(', '[', '{'):
            prev = sig[i - 1] if i else None
            prev2 = sig[i - 2] if i > 1 else None
            kind = 'group' if s_ == '(' else ('list' if s_ == '[' else 'brace')
            if s_ == '(':
                if prev is not None and prev.type == _tok.NAME:
                    if prev2 is not None and prev2.string == 'def':
                        kind = 'def'
                    elif prev2 is not None and prev2.string == 'class':
                        kind = 'classbases'
                    elif prev.string == 'import':
                        kind = 'import'
                    elif not _kw.iskeyword(prev.string):
                        kind = 'call'
                elif prev is not None and prev.string in (')', ']'):
                    kind = 'call'
            elif s_ == '[':
                if prev is not None and ((prev.type == _tok.NAME and not _kw.iskeyword(prev.string))
                                         or prev.string in (')', ']') or prev.type == _tok.STRING):
                    kind = 'subscript'
            if stack:
                stack[-1]['seg'].append(t)
            stack.append(dict(kind=kind, seg=[], last=[]))
        elif is_op and s_ in (')', ']', '}'):
            if stack:
                e = stack.pop()
                if i and sig[i - 1].string == ',' and e['last']:
                    form = ''
                    if e['kind'] in ('def', 'call'):
                        f0 = e['last'][0].string
                        form = ':kwargs' if f0 == '**' else (':bare_star' if f0 == '*' and len(e['last']) == 1
                                                              else (':args' if f0 == '*' else ':plain'))
                        if e['kind'] == 'def' and form == ':plain' and any(t_.string == ':' for t_ in e['last']):
                            form = ':annotated_default' if any(t_.string == '=' for t_ in e['last']) else ':annotated'
                    c.add('trailing_comma:%s%s' % (e['kind'], form))
                if stack:
                    stack[-1]['seg'].append(t)
        elif is_op and s_ == ',' and stack:
            stack[-1]['last'] = stack[-1]['seg']
            stack[-1]['seg'] = []
        elif stack:
            stack[-1]['seg'].append(t)
        # parenthesized with-items: `with (a as x, b as y):`
        if t.type == _tok.NAME and s_ == 'with' and i + 1 < len(sig) and sig[i + 1].string == '(':
            j = _match_close(sig, i + 1)
            if j is not None and j + 1 < len(sig) and sig[j + 1].string == ':':
                depth, has_as = 0, False
                for k in range(i + 1, j + 1):
                    if sig[k].type == _tok.OP and sig[k].string in ('(', '[', '{'):
                        depth += 1
                    elif sig[k].type == _tok.OP and sig[k].string in (')', ']', '}'):
                        depth -= 1
                    elif depth == 1 and sig[k].string == 'as':
                        has_as = True
                if has_as:
                    c.add('with_parenthesized_items')
    return c



# Synthetic cases that isolate one inventory construct. A construct is CONFIRMED_UNSUPPORTED when any of its
# isolating cases fails in NiCad; SUPPORTED_ALONE when all pass (a failure in real files is then confounded).
ISOLATING_CASES = {
    'dict_comprehension': ['E45'], 'set_comprehension': ['E44'], 'set_literal': ['E36', 'E37', 'E38'],
    'trailing_comma:def:kwargs': ['T03', 'T04'], 'trailing_comma:def:args': ['T02'],
    'trailing_comma:def:plain': ['T01', 'T09'], 'trailing_comma:def:annotated': ['H01'],
    'trailing_comma:def:annotated_default': ['H02', 'H05'], 'trailing_comma:call:plain': ['T05'],
    'trailing_comma:call:kwargs': ['T06'], 'trailing_comma:classbases': ['K03'],
    'trailing_comma:brace': ['T07'], 'trailing_comma:import': ['T08'],
    'variable_annotation': ['D19', 'D34', 'K04'], 'varargs_annotation': ['D20'],
    'await_expr': ['D12'], 'yield_from': ['D32'], 'raise_from': ['E23'], 'dict_unpack_display': ['E40'],
    'star_in_display': ['E41'], 'bare_star_kwonly': ['D25'], 'kwonly_after_vararg': ['D33'],
    'class_keyword_arg': ['K01', 'K02'], 'ellipsis': ['E14'], 'num_underscore': ['N01'], 'num_binary': ['N04'],
    'num_hex': ['N02'], 'num_octal': ['N03'], 'num_complex': ['N05'], 'num_exponent': ['N06'],
    'str_prefix_rf': ['P04'], 'str_prefix_fr': ['P05'], 'str_prefix_rb': ['P02'], 'str_prefix_br': ['P03'],
    'str_prefix_b': ['P01'], 'str_prefix_r': ['P07'], 'str_prefix_f': ['P08'],
    'walrus_bare_test': ['W01', 'W02', 'W04'], 'walrus_nested': ['E01', 'E02', 'E03', 'E04', 'W03', 'W05', 'W06'],
    'match_stmt': ['E05'], 'except_star': ['E08'], 'positional_only': ['D24'], 'star_assignment': ['E11'],
    'star_in_subscript': ['E12'], 'matmul': ['E13'], 'multi_item_with': ['E10'],
    'with_parenthesized_items': ['E09'], 'pep695_generics': ['D27', 'D29'], 'type_alias_stmt': ['D28'],
    'async_comprehension': ['D15'], 'async_def': ['D11'], 'async_with': ['D14'], 'async_for': ['D13'],
    'decorator_expression': ['D08'], 'decorator_dotted_call': ['H03', 'H04'],
    'non_ascii_identifier': ['S17'], 'non_ascii_comment': ['L12'], 'fstring_nested_same_quote': ['S06'],
    'call_pos_then_star': ['A01'], 'call_pos_star_pos': ['A02', 'A09', 'A10'], 'call_pos_star_kw': ['A03', 'A12'],
    'call_pos_dstar_kw': ['A04'], 'call_star_then_keyword': ['E46'], 'call_star_then_positional': ['E47'],
    'call_dstar_then_keyword': ['E48'], 'call_keyword_then_star': ['A06'],
    'attr_name_print': ['Y01', 'Y03', 'Y05'], 'attr_name_exec': ['Y04'],
}


def load_isolated_results(a, out):
    """case id -> NiCad category from synthetic runs under results/b-05 (descriptions must match the catalog)."""
    if a.synthetic_result == 'none':
        return {}, []
    desc = {c['id']: c['desc'] for c in build_catalog()}
    if a.synthetic_result == 'auto':
        files = sorted(out.parent.glob('*/synthetic/synthetic_results.csv'), key=lambda p: p.stat().st_mtime)
    else:
        pth = Path(a.synthetic_result)
        files = [pth] if pth.is_file() else [pth / 'synthetic/synthetic_results.csv']
    res, used = {}, []
    for f in files:
        try:
            with f.open(newline='', encoding='utf-8') as h:
                for r in csv.DictReader(h):
                    if r.get('id') in desc and r.get('description') == desc[r['id']]:
                        res[r['id']] = r['nicad_category']
            used.append(str(f))
        except (OSError, UnicodeError, csv.Error):
            continue
    return res, used


def run_inventory(a, out, cats, stage):
    idir = out / 'inventory'
    idir.mkdir(parents=True, exist_ok=True)
    files = sorted(r for r in cats if (stage / r).is_file())
    rows, per = [], {}
    P('')
    P('=' * 96)
    P('[INVENTORY] syntactic constructs in %d staged files (NO NiCad calls)' % len(files))
    P('=' * 96)
    for rel in files:
        _, text = read_text(stage / rel)
        cs = inventory_constructs(text)
        per[rel] = cs
        rows.append(dict(file=rel, label=cats[rel], constructs='AST_ERROR' if cs is None else ';'.join(sorted(cs))))
    write_csv(idir / 'file_constructs.csv', rows, ['file', 'label', 'constructs'])
    fail = [r for r in files if cats[r] == 'NICAD_PARSE_FAILURE' and per[r] is not None]
    okf = [r for r in files if cats[r] in ('SUCCESS', 'NO_FUNCTIONS_EXTRACTED') and per[r] is not None]
    P('parse failures: %d | parsed without failure (SUCCESS + NO_FUNCTIONS_EXTRACTED): %d' % (len(fail), len(okf)))
    cf, co = Counter(), Counter()
    for r in fail:
        cf.update(per[r])
    for r in okf:
        co.update(per[r])
    names = sorted(set(cf) | set(co))
    iso_res, iso_files = load_isolated_results(a, out)
    have_iso = bool(iso_res)
    P('synthetic evidence: %d case results from %d run(s)%s' % (
        len(iso_res), len(iso_files), '' if have_iso else '  (none found: run --phase synthetic first for confirmed results)'))
    for f_ in iso_files[-6:]:
        P('   source: ' + f_ + '   (newest run wins per case; use --synthetic-result <dir> to pin one run)')
    summ = []
    for n in names:
        ids = [i for i in ISOLATING_CASES.get(n, []) if i in iso_res]
        iso = None
        if ids:
            iso = 'FAIL' if any(iso_res[i] == 'NICAD_PARSE_FAILURE' for i in ids) else 'PASS'
        if iso == 'FAIL':
            verdict = 'CONFIRMED_UNSUPPORTED'
        elif iso == 'PASS':
            verdict = 'SUPPORTED' if co[n] > 0 else ('SUPPORTED_ALONE' if cf[n] > 0 else 'SUPPORTED')
        elif co[n] > 0:
            verdict = 'SUPPORTED'
        elif cf[n] >= a.min_evidence:
            verdict = 'UNCONFIRMED_LIKELY' if have_iso else 'LIKELY_UNSUPPORTED'
        elif cf[n] > 0:
            verdict = 'UNCONFIRMED_WEAK' if have_iso else 'WEAK_EVIDENCE'
        else:
            verdict = 'UNOBSERVED'
        summ.append(dict(construct=n, in_failing_files=cf[n], in_parsed_ok_files=co[n], verdict=verdict,
                         isolated_cases=','.join('%s=%s' % (i, 'F' if iso_res[i] == 'NICAD_PARSE_FAILURE' else 'P') for i in ids)))
    summ.sort(key=lambda r: (-r['in_failing_files'], r['construct']))
    write_csv(idir / 'construct_summary.csv', summ, ['construct', 'in_failing_files', 'in_parsed_ok_files', 'verdict', 'isolated_cases'])
    P('')
    P('Logic: NiCad rejects a whole file if ANY construct is unsupported, so a construct that occurs in a')
    P('file NiCad parsed is SUPPORTED. A construct that occurs in failing files and never in parsed files is')
    P('a LIKELY_UNSUPPORTED candidate (evidence, not proof; verify with synthetic cases).')
    P('')
    P('%-38s %10s %10s  %-22s %s' % ('Construct', 'in failing', 'in parsed', 'Verdict', 'isolated synthetic'))
    for r in summ:
        if r['in_failing_files'] > 0:
            P('%-38s %10d %10d  %-22s %s' % (r['construct'], r['in_failing_files'], r['in_parsed_ok_files'], r['verdict'], r['isolated_cases']))
    if have_iso:
        cand = {r['construct'] for r in summ if r['verdict'] == 'CONFIRMED_UNSUPPORTED'}
        P('')
        P('Recovery below counts ONLY constructs confirmed unsupported by an isolated synthetic case.')
    else:
        cand = {r['construct'] for r in summ if r['verdict'] in ('LIKELY_UNSUPPORTED', 'WEAK_EVIDENCE')}
    need = {r: set(per[r]) & cand for r in fail}
    unexplained = sorted(r for r in fail if not need[r])
    explained = [r for r in fail if need[r]]
    write_csv(idir / 'unexplained_files.csv', [dict(file=r) for r in unexplained], ['file'])
    remaining = {r: set(need[r]) for r in explained}
    curve, fixed = [], []
    while True:
        score = Counter()
        for r, rem in remaining.items():
            if rem:
                w = 1.0 / len(rem)
                for x in rem:
                    score[x] += w
        if not score:
            break
        best = max(score, key=lambda x: (round(score[x], 9), cf[x], x))
        fixed.append(best)
        for rem in remaining.values():
            rem.discard(best)
        rec = sum(1 for rem in remaining.values() if not rem)
        curve.append(dict(step=len(fixed), add_support_for=best, files_recovered=rec,
                          pct_of_failing=round(100.0 * rec / max(1, len(fail)), 1)))
    write_csv(idir / 'recovery_curve.csv', curve, ['step', 'add_support_for', 'files_recovered', 'pct_of_failing'])
    P('')
    P('[RECOVERY PREDICTION] files that would parse if the listed constructs were supported (greedy order)')
    P('Assumes every other construct is already supported and the candidate list is complete.')
    P('%-5s %-38s %16s %10s' % ('Step', 'Add support for', 'Files recovered', '% of fail'))
    for r in curve[:a.curve_steps]:
        P('%-5d %-38s %16d %9.1f%%' % (r['step'], r['add_support_for'], r['files_recovered'], r['pct_of_failing']))
    P('')
    P('Failing files with NO confirmed-unsupported construct: %d of %d' % (len(unexplained), len(fail)))
    vmap = {r['construct']: r['verdict'] for r in summ}
    for r in unexplained[:15]:
        sus = sorted(x for x in per[r] if vmap.get(x, '').startswith(('UNCONFIRMED', 'LIKELY', 'WEAK', 'SUPPORTED_ALONE')))
        P('  %s   unconfirmed/confounded constructs: %s' % (r, ', '.join(sus) or '(none: undetected construct or a combination)'))
    if have_iso:
        todo = [r for r in summ if r['verdict'].startswith('UNCONFIRMED') and r['in_failing_files'] > 0]
        if todo:
            P('')
            P('Candidates with no isolating synthetic case yet (add a case or use localize on a file containing them):')
            for r in todo[:12]:
                P('  %-38s failing=%d parsed=%d' % (r['construct'], r['in_failing_files'], r['in_parsed_ok_files']))
    return dict(failing=len(fail), parsed_ok=len(okf), synthetic_cases_used=len(iso_res), candidates=sorted(cand), unexplained=len(unexplained),
                greedy_order=fixed, curve=curve)


# --------------------------------------------------------------------------- helpers
def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def split_lines(text):
    return io.StringIO(text, newline='').readlines()


def read_text(path):
    data = Path(path).read_bytes()
    return data, data.decode('utf-8', errors='replace')


def row_category(row):
    for v in row.values():
        if isinstance(v, str) and v.strip() in CATS:
            return v.strip()
    exc = row.get('nicad_error_excerpt') or ''
    if PARSE_RE.search(exc):
        return 'NICAD_PARSE_FAILURE'
    st = (row.get('nicad_status') or '').strip().lower()
    if st == 'passed':
        return 'SUCCESS'
    if st == 'failed':
        return 'NICAD_PARSE_FAILURE'
    ext = (row.get('nicad_extracted_functions') or '').strip()
    if st in ('error', 'unknown') and ext == '0':
        return 'NO_FUNCTIONS_EXTRACTED'
    return None


def load_categories(input_dir):
    """Find the CSV (B-02 file_parse_audit.csv or a B-03 classification CSV) that labels each file."""
    best = (0, None, {})
    seen = []
    for p in sorted(Path(input_dir).rglob('*.csv')):
        try:
            with p.open(newline='', encoding='utf-8') as h:
                rd = csv.DictReader(h)
                if not rd.fieldnames or 'file' not in rd.fieldnames:
                    continue
                seen.append('%s: %s' % (p.name, ','.join(rd.fieldnames)))
                cats = {}
                for row in rd:
                    c = row_category(row)
                    if c:
                        cats[row['file']] = c
        except (OSError, UnicodeError, csv.Error):
            continue
        if len(cats) > best[0]:
            best = (len(cats), p, cats)
    if best[1] is None:
        raise SystemExit('ERROR: no CSV with a `file` column and recognizable categories under %s\n  headers seen: %s'
                         % (input_dir, ' || '.join(seen) or '(none)'))
    return best[2], best[1]


def default_input_result(root):
    for sub in ('b-03', 'b-02'):
        base = root / 'results' / sub
        if base.is_dir():
            dirs = sorted(d for d in base.iterdir() if d.is_dir() and not d.name.startswith('_'))
            if dirs:
                return dirs[-1]
    return None


SIZE_BUCKETS = [(0, 2048, '<2KB'), (2048, 5120, '2-5KB'), (5120, 10240, '5-10KB'),
                (10240, 20480, '10-20KB'), (20480, 51200, '20-50KB'), (51200, 102400, '50-100KB'),
                (102400, 10 ** 12, '>=100KB')]


def size_audit(cats, stage, out):
    rows = []
    stats = {b[2]: [0, 0] for b in SIZE_BUCKETS}
    for rel, c in cats.items():
        p = stage / rel
        if not p.is_file():
            continue
        sz = p.stat().st_size
        for lo, hi, name in SIZE_BUCKETS:
            if lo <= sz < hi:
                stats[name][0] += 1
                stats[name][1] += (c == 'NICAD_PARSE_FAILURE')
                break
    P('')
    P('[SIZE AUDIT] NiCad parse-failure rate by staged file size (read from B-02/B-03 labels)')
    P('%-10s %8s %8s %8s' % ('Bucket', 'Files', 'Failed', 'Rate'))
    for _, _, name in SIZE_BUCKETS:
        n, f = stats[name]
        rate = (100.0 * f / n) if n else 0.0
        rows.append(dict(bucket=name, files=n, parse_failures=f, failure_rate_pct=round(rate, 1)))
        P('%-10s %8d %8d %7.1f%%' % (name, n, f, rate))
    write_csv(out / 'size_failure_rate.csv', rows, ['bucket', 'files', 'parse_failures', 'failure_rate_pct'])
    P('  (Failure rate vs size only. Larger files contain more constructs, so a rising rate does not by itself')
    P('   distinguish a grammar gap from a resource limit: use the inventory/synthetic phases for that.)')
    return rows


# --------------------------------------------------------------------------- feature hints
def detect_features(text):
    f = set()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return f

    def ann_union(n):
        return any(isinstance(x, ast.BinOp) and isinstance(x.op, ast.BitOr) for x in ast.walk(n))

    def visit(n, in_func):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if isinstance(n, ast.AsyncFunctionDef):
                f.add('async_def')
            if n.decorator_list:
                f.add('decorator')
            if in_func:
                f.add('nested_function')
            a = n.args
            allargs = a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]
            if a.posonlyargs:
                f.add('positional_only')
            if a.kwonlyargs:
                f.add('keyword_only')
            for x in allargs:
                if x.annotation is not None:
                    f.add('annotation')
                    if ann_union(x.annotation):
                        f.add('union_pipe')
            if n.returns is not None:
                f.add('annotation')
            if getattr(n, 'type_params', None):
                f.add('pep695_generics')
            in_func = True
        elif isinstance(n, ast.ClassDef):
            if n.decorator_list:
                f.add('decorator')
            if in_func:
                f.add('nested_class')
            if getattr(n, 'type_params', None):
                f.add('pep695_generics')
        elif isinstance(n, ast.Await):
            f.add('await')
        elif isinstance(n, ast.AsyncFor):
            f.add('async_for')
        elif isinstance(n, ast.AsyncWith):
            f.add('async_with')
        elif isinstance(n, ast.comprehension) and n.is_async:
            f.add('async_comprehension')
        elif isinstance(n, ast.NamedExpr):
            f.add('walrus')
        elif isinstance(n, ast.JoinedStr):
            f.add('fstring')
        elif isinstance(n, ast.FormattedValue):
            if n.format_spec is not None:
                f.add('fstring_format_spec')
            if n.conversion != -1:
                f.add('fstring_conversion')
        elif isinstance(n, ast.AnnAssign):
            f.add('annotation_assign')
        elif hasattr(ast, 'Match') and isinstance(n, ast.Match):
            f.add('match')
        elif hasattr(ast, 'TryStar') and isinstance(n, ast.TryStar):
            f.add('except_star')
        elif hasattr(ast, 'TypeAlias') and isinstance(n, ast.TypeAlias):
            f.add('type_alias')
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult):
            f.add('matmul')
        elif isinstance(n, ast.Subscript) and any(isinstance(x, ast.Starred) for x in ast.walk(n.slice)):
            f.add('star_index')
        elif isinstance(n, ast.With) and len(n.items) > 1:
            f.add('multi_with')
        elif isinstance(n, ast.Lambda):
            f.add('lambda')
        elif isinstance(n, (ast.Yield, ast.YieldFrom)):
            f.add('yield')
        elif isinstance(n, ast.Constant) and isinstance(n.value, bytes):
            f.add('bytes')
        elif isinstance(n, ast.Constant) and n.value is Ellipsis:
            f.add('ellipsis')
        elif isinstance(n, ast.Global) or isinstance(n, ast.Nonlocal):
            f.add('global_nonlocal')
        for c in ast.iter_child_nodes(n):
            visit(c, in_func)

    visit(tree, False)
    if re.search(r'[^\x00-\x7f]', text):
        f.add('non_ascii')
    if re.search(r'\\\r?\n', text):
        f.add('backslash_continuation')
    if re.search(r'^\t', text, re.M):
        f.add('tab_indent')
    if '\r\n' in text:
        f.add('crlf')
    if '\x0c' in text:
        f.add('form_feed')
    if re.search(r'^\s*(match|case|type)\s*=', text, re.M):
        f.add('soft_keyword_identifier')
    return f


# --------------------------------------------------------------------------- localization
def stmt_candidates(tree, lines):
    cands = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        if isinstance(node, ast.Pass) and node.lineno == node.end_lineno:
            continue
        start = node.lineno
        for d in getattr(node, 'decorator_list', []):
            start = min(start, d.lineno)
        end = node.end_lineno
        first = lines[start - 1]
        indent = re.match(r'[ \t]*', first).group()
        # the statement must start at the line's indentation and be alone on its last line
        if node.lineno == start and node.col_offset != len(indent.encode('utf-8')):
            continue
        tail = lines[end - 1].encode('utf-8')[node.end_col_offset:].decode('utf-8', 'replace')
        if not re.fullmatch(r'\s*(#.*)?', tail, re.S):
            continue
        cands.append((end - start, start, end, indent))
    return cands


def replace_span(lines, start, end, indent):
    first = lines[start - 1]
    eol = '\r\n' if first.endswith('\r\n') else ('\r' if first.endswith('\r') else '\n')
    return ''.join(lines[:start - 1] + [indent + 'pass' + eol] + lines[end:])


def reduce_failure(prober, text, max_probes):
    """Greedy statement-level reduction preserving NICAD_PARSE_FAILURE (largest spans first)."""
    cur = text
    rejected = set()
    used = 0
    steps = 0
    exhausted = False
    while True:
        try:
            tree = ast.parse(cur)
        except (SyntaxError, ValueError):
            break
        lines = split_lines(cur)
        cands = sorted(stmt_candidates(tree, lines), key=lambda c: (-c[0], c[1]))
        progressed = False
        for span, start, end, indent in cands:
            key = hashlib.sha1(''.join(lines[start - 1:end]).encode('utf-8', 'replace')).hexdigest()
            if key in rejected:
                continue
            new = replace_span(lines, start, end, indent)
            if new == cur:
                rejected.add(key)
                continue
            try:
                ast.parse(new)
            except (SyntaxError, ValueError):
                rejected.add(key)
                continue
            if used >= max_probes:
                exhausted = True
                break
            used += 1
            r = prober.probe(new)
            if r['category'] == 'NICAD_PARSE_FAILURE':
                cur = new
                steps += 1
                progressed = True
                break
            rejected.add(key)
        if exhausted or not progressed:
            break
    return cur, used, steps, exhausted


def leaf_statements(text):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return 0
    compound = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.For, ast.AsyncFor,
                ast.While, ast.With, ast.AsyncWith, ast.Try)
    extra = tuple(getattr(ast, n) for n in ('TryStar', 'Match') if hasattr(ast, n))
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(node, compound + extra + (ast.Pass,)):
            n += 1
    return n


def slugify(rel):
    return re.sub(r'[^A-Za-z0-9_.-]+', '__', rel)[:120]


def pick_files(cats, stage, a):
    files = sorted(r for r, c in cats.items() if c == 'NICAD_PARSE_FAILURE' and (stage / r).is_file())
    missing = sorted(r for r, c in cats.items() if c == 'NICAD_PARSE_FAILURE' and not (stage / r).is_file())
    if a.files_from:
        want = {l.strip() for l in Path(a.files_from).read_text().splitlines() if l.strip()}
        files = [f for f in files if f in want]
    if a.file_filter:
        files = [f for f in files if a.file_filter in f]
    if a.max_files > 0 and len(files) > a.max_files:
        step = len(files) / float(a.max_files)
        files = [files[int(i * step)] for i in range(a.max_files)]
    return files, missing


def run_localize(a, out, probers, cats, stage, files):
    ldir = out / 'localize'
    (ldir / 'minimal_cases').mkdir(parents=True, exist_ok=True)
    P('')
    P('=' * 96)
    P('[LOCALIZE] reducing %d NiCad-parse-failing files | probe budget/file=%d | workers=%d'
      % (len(files), a.max_probes_per_file, len(probers)))
    P('=' * 96)
    q = queue.Queue()
    for p in probers:
        q.put(p)
    counter = [0]

    def work(item):
        idx, rel = item
        prober = q.get()
        try:
            data, text = read_text(stage / rel)
            row = dict(file=rel, size_bytes=len(data), lines=len(split_lines(text)),
                       rerun_category='', outcome='', probes=0, accepted_steps=0,
                       minimal_leaf_statements=0, minimal_lines=0, features='',
                       minimal_path='', preview='')
            ast_s, ast_e = py_ast_status(data)
            if ast_s != 'PASS':
                row['outcome'] = 'AST_FAIL_SKIPPED'
                return row, ''
            base = prober.probe(data)
            row['rerun_category'] = base['category']
            if base['category'] != 'NICAD_PARSE_FAILURE':
                row['outcome'] = 'NOT_REPRODUCED'
                return row, ''
            cur, used, steps, exhausted = reduce_failure(prober, text, a.max_probes_per_file)
            leaf = leaf_statements(cur)
            nonpass = [l for l in split_lines(cur) if l.strip() and l.strip() != 'pass']
            feats = sorted(detect_features(cur))
            if steps == 0:
                outcome = 'IRREDUCIBLE'
            elif exhausted:
                outcome = 'BUDGET_EXHAUSTED'
            elif leaf <= 3:
                outcome = 'LOCALIZED'
            else:
                outcome = 'PARTIAL'
            mp = ldir / 'minimal_cases' / ('%04d__%s' % (idx, slugify(rel)))
            mp.write_text(cur, encoding='utf-8')
            prev = ' / '.join(l.strip() for l in nonpass[:3])[:200]
            row.update(outcome=outcome, probes=used, accepted_steps=steps, minimal_leaf_statements=leaf,
                       minimal_lines=len(nonpass), features=';'.join(feats),
                       minimal_path=str(mp.relative_to(out)), preview=prev)
            with _print_lock:
                counter[0] += 1
                print('[LOCALIZE %d/%d] %-16s leaf=%-3d lines=%-4d probes=%-3d %s | %s'
                      % (counter[0], len(files), outcome, leaf, len(nonpass), used, rel, ';'.join(feats)),
                      flush=True)
            return row, cur
        finally:
            q.put(prober)

    with ThreadPoolExecutor(max_workers=len(probers)) as ex:
        results = list(ex.map(work, list(enumerate(files, 1))))
    rows = [r for r, _ in results]
    snippets = {r['file']: s for r, s in results}
    fields = ['file', 'size_bytes', 'lines', 'rerun_category', 'outcome', 'probes', 'accepted_steps',
              'minimal_leaf_statements', 'minimal_lines', 'features', 'minimal_path', 'preview']
    write_csv(ldir / 'minimal_cases.csv', rows, fields)
    outcomes = Counter(r['outcome'] for r in rows)
    feat = Counter()
    sigs = defaultdict(list)
    for r in rows:
        if r['outcome'] in ('LOCALIZED', 'PARTIAL', 'BUDGET_EXHAUSTED') and r['features']:
            fs = r['features'].split(';')
            feat.update(fs)
            sigs[tuple(fs)].append(r['file'])
    write_csv(ldir / 'feature_counts.csv', [dict(feature=k, minimal_cases=v) for k, v in feat.most_common()],
              ['feature', 'minimal_cases'])
    write_csv(ldir / 'signature_clusters.csv',
              [dict(signature=';'.join(k), files=len(v), example=v[0]) for k, v in
               sorted(sigs.items(), key=lambda kv: -len(kv[1]))], ['signature', 'files', 'example'])
    P('')
    P('[LOCALIZE SUMMARY] ' + ', '.join('%s=%d' % kv for kv in sorted(outcomes.items())))
    P('')
    P('[FEATURES PRESENT IN MINIMAL FAILING SNIPPETS] (the failing snippet contains this syntax)')
    P('%-26s %8s' % ('Feature', 'Cases'))
    for k, v in feat.most_common(20):
        P('%-26s %8d' % (k, v))
    P('')
    P('[TOP SIGNATURE CLUSTERS]')
    for k, v in sorted(sigs.items(), key=lambda kv: -len(kv[1]))[:10]:
        P('  %3d files | %-60s | e.g. %s' % (len(v), ';'.join(k) or '(no tracked feature)', v[0]))
    shown = 0
    P('')
    P('[MINIMAL FAILING SNIPPETS] up to %d LOCALIZED cases (real code that makes NiCad fail to parse)' % a.show)
    for r in rows:
        if shown >= a.show:
            break
        if r['outcome'] != 'LOCALIZED':
            continue
        shown += 1
        P('\n--- %d/%d: %s  [%s]' % (shown, a.show, r['file'], r['features']))
        code = [l.rstrip('\r\n') for l in split_lines(snippets[r['file']]) if l.strip() and l.strip() != 'pass']
        for l in code[:14]:
            P('    ' + l[:160])
    return dict(outcomes=dict(outcomes), features=dict(feat.most_common()),
                clusters=len(sigs), files=len(rows))


# --------------------------------------------------------------------------- ablation
class _StripAnn(ast.NodeTransformer):
    def _args(self, a):
        for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
            if x is not None:
                x.annotation = None

    def visit_FunctionDef(self, n):
        self.generic_visit(n)
        self._args(n.args)
        n.returns = None
        return n

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_AnnAssign(self, n):
        self.generic_visit(n)
        if n.value is None:
            return ast.copy_location(ast.Pass(), n)
        return ast.copy_location(ast.Assign(targets=[n.target], value=n.value, type_comment=None), n)


class _SyncAsync(ast.NodeTransformer):
    def visit_AsyncFunctionDef(self, n):
        self.generic_visit(n)
        new = ast.FunctionDef(name=n.name, args=n.args, body=n.body, decorator_list=n.decorator_list,
                              returns=n.returns, type_comment=None, type_params=getattr(n, 'type_params', []))
        return ast.copy_location(new, n)

    def visit_Await(self, n):
        self.generic_visit(n)
        return n.value

    def visit_AsyncFor(self, n):
        self.generic_visit(n)
        return ast.copy_location(ast.For(target=n.target, iter=n.iter, body=n.body, orelse=n.orelse,
                                         type_comment=None), n)

    def visit_AsyncWith(self, n):
        self.generic_visit(n)
        return ast.copy_location(ast.With(items=n.items, body=n.body, type_comment=None), n)

    def visit_comprehension(self, n):
        self.generic_visit(n)
        n.is_async = 0
        return n


class _DeWalrus(ast.NodeTransformer):
    def visit_NamedExpr(self, n):
        self.generic_visit(n)
        return n.value


class _StripDecorators(ast.NodeTransformer):
    def _strip(self, n):
        self.generic_visit(n)
        n.decorator_list = []
        return n

    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip


class _SimplifyFString(ast.NodeTransformer):
    def visit_JoinedStr(self, n):
        return ast.copy_location(ast.Constant(value='FSTRING'), n)


class _NewSyntax(ast.NodeTransformer):
    """match -> pass, except* -> except, PEP 695 type params / type alias removed."""

    def visit_Match(self, n):
        return ast.copy_location(ast.Pass(), n)

    def visit_TryStar(self, n):
        self.generic_visit(n)
        return ast.copy_location(ast.Try(body=n.body, handlers=n.handlers, orelse=n.orelse,
                                         finalbody=n.finalbody), n)

    def visit_TypeAlias(self, n):
        return ast.copy_location(ast.Pass(), n)

    def _tp(self, n):
        self.generic_visit(n)
        if hasattr(n, 'type_params'):
            n.type_params = []
        return n

    visit_FunctionDef = _tp
    visit_AsyncFunctionDef = _tp
    visit_ClassDef = _tp


TX = [('annotations', _StripAnn), ('async', _SyncAsync), ('walrus', _DeWalrus),
      ('decorators', _StripDecorators), ('fstrings', _SimplifyFString), ('newsyntax', _NewSyntax)]
TXD = dict(TX)


def apply_tx(text, names):
    tree = ast.parse(text)
    for n in names:
        tree = TXD[n]().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + '\n'


def restored(cat):
    return cat in ('SUCCESS', 'NO_FUNCTIONS_EXTRACTED')


def run_ablate(a, out, probers, files, stage):
    adir = out / 'ablate'
    adir.mkdir(parents=True, exist_ok=True)
    names = [n for n, _ in TX]
    P('')
    P('=' * 96)
    P('[ABLATE] %d files | transforms: %s' % (len(files), ', '.join(names)))
    P('  T0 = ast.unparse only (formatting/tokenization control). A file restored by T0 alone is a')
    P('  lexical/formatting problem, not a grammar-feature problem. Rewrites change line numbers: this')
    P('  phase is DIAGNOSTIC only (it identifies the cause); it does not produce analysis input.')
    P('=' * 96)
    q = queue.Queue()
    for p in probers:
        q.put(p)
    counter = [0]

    def try_set(prober, text, subset):
        try:
            new = apply_tx(text, subset)
        except (SyntaxError, ValueError, TypeError, AttributeError, RecursionError) as e:
            return 'TX_ERROR:' + type(e).__name__
        return prober.probe(new)['category']

    def work(rel):
        prober = q.get()
        try:
            data, text = read_text(stage / rel)
            row = dict(file=rel, size_bytes=len(data), base='', t0_unparse='', all_transforms='',
                       minimal_set='', outcome='')
            for n in names:
                row['only_' + n] = ''
            if py_ast_status(data)[0] != 'PASS':
                row['outcome'] = 'AST_FAIL_SKIPPED'
                return row
            base = prober.probe(data)['category']
            row['base'] = base
            if base != 'NICAD_PARSE_FAILURE':
                row['outcome'] = 'NOT_REPRODUCED'
                return row
            row['t0_unparse'] = try_set(prober, text, [])
            if restored(row['t0_unparse']):
                row['outcome'] = 'FIXED_BY_UNPARSE_ONLY'
                row['minimal_set'] = ''
            else:
                for n in names:
                    row['only_' + n] = try_set(prober, text, [n])
                row['all_transforms'] = try_set(prober, text, names)
                if not restored(row['all_transforms']):
                    row['outcome'] = 'UNRESOLVED'
                else:
                    cur = list(names)
                    for n in names:
                        trial = [x for x in cur if x != n]
                        if restored(try_set(prober, text, trial)):
                            cur = trial
                    row['minimal_set'] = '+'.join(cur)
                    singles = [n for n in names if restored(row['only_' + n])]
                    row['outcome'] = ('FIXED_BY_SINGLE:' + singles[0]) if len(cur) == 1 else 'FIXED_BY_COMBINATION'
            with _print_lock:
                counter[0] += 1
                print('[ABLATE %d/%d] %-26s min=%-28s %s' % (counter[0], len(files), row['outcome'],
                                                            row['minimal_set'] or '-', rel), flush=True)
            return row
        finally:
            q.put(prober)

    with ThreadPoolExecutor(max_workers=len(probers)) as ex:
        rows = list(ex.map(work, files))
    fields = ['file', 'size_bytes', 'base', 't0_unparse'] + ['only_' + n for n in names] + \
             ['all_transforms', 'minimal_set', 'outcome']
    write_csv(adir / 'ablation_results.csv', rows, fields)
    oc = Counter(r['outcome'] for r in rows)
    need = Counter()
    sole = Counter()
    for r in rows:
        if r['minimal_set']:
            for t in r['minimal_set'].split('+'):
                need[t] += 1
        for n in names:
            if restored(r.get('only_' + n, '')):
                sole[n] += 1
    write_csv(adir / 'transform_effect.csv',
              [dict(transform=n, in_minimal_set=need[n], restores_alone=sole[n]) for n in names],
              ['transform', 'in_minimal_set', 'restores_alone'])
    P('')
    P('[ABLATION SUMMARY]')
    for k, v in sorted(oc.items(), key=lambda kv: -kv[1]):
        P('  %-28s %5d' % (k, v))
    P('')
    P('%-14s %16s %16s' % ('Transform', 'In minimal set', 'Restores alone'))
    for n in names:
        P('%-14s %16d %16d' % (n, need[n], sole[n]))
    P('  (restores alone = file parses after that single rewrite; in minimal set = needed in the greedy-minimal combination)')
    return dict(outcomes=dict(oc), in_minimal_set=dict(need), restores_alone=dict(sole), files=len(rows))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument('--phase', choices=['calibrate', 'inventory', 'synthetic', 'localize', 'ablate', 'all'], default='synthetic')
    ap.add_argument('--calibrate-workers', default='1,2,4,8', help='calibrate: worker counts to compare')
    ap.add_argument('--calibrate-cases', type=int, default=24, help='calibrate: cases per level')
    ap.add_argument('--synthetic-result', default='auto', help="inventory: synthetic results to confirm constructs: 'auto' (all runs under results/b-05), 'none', a result dir, or a synthetic_results.csv")
    ap.add_argument('--min-evidence', type=int, default=3, help='inventory: failing files needed for LIKELY_UNSUPPORTED')
    ap.add_argument('--curve-steps', type=int, default=30, help='inventory: rows of the recovery curve to print')
    ap.add_argument('--min-efficiency', type=float, default=0.6, help='calibrate: minimum speedup/workers to accept')
    ap.add_argument('--input-result', help='B-03 (or B-02) result dir with per-file NiCad labels (default: newest b-03, else b-02)')
    ap.add_argument('--stage', help='staged source root (default: results/b-01/_staging/salt)')
    ap.add_argument('--nicad-home')
    ap.add_argument('--txl-home')
    ap.add_argument('--config', default='default')
    ap.add_argument('--probe-timeout', type=int, default=100)
    ap.add_argument('--workers', type=int, default=1, help='parallel NiCad probes (each gets its own b05_probe_w<N> dir)')
    ap.add_argument('--max-files', type=int, default=0, help='0=all; otherwise evenly spaced sample of failing files')
    ap.add_argument('--files-from', help='text file listing staged-relative paths to analyze')
    ap.add_argument('--file-filter', help='substring filter on staged-relative path')
    ap.add_argument('--max-probes-per-file', type=int, default=80)
    ap.add_argument('--only', help='synthetic: comma-separated case IDs or groups')
    ap.add_argument('--max-cases', type=int, default=0)
    ap.add_argument('--show', type=int, default=12, help='LOCALIZED snippets printed to console')
    ap.add_argument('--list-cases', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()

    if a.list_cases:
        for c in build_catalog():
            print('%-5s %-10s %s' % (c['id'], c['group'], c['desc']))
        return 0

    root = Path(a.root).resolve()
    nicad = Path(a.nicad_home or os.environ.get('NICAD_HOME') or root / 'tools/nicad-7.0.1-linux-x86_64').resolve()
    txl = Path(a.txl_home or os.environ.get('TXL_HOME') or root / 'tools/opentxl-11.3.7-linux-x64').resolve()
    if not (nicad / 'bin/nicad').is_file():
        raise SystemExit('ERROR: NiCad not found: %s/bin/nicad' % nicad)
    if not (txl / 'bin/txl').exists():
        raise SystemExit('ERROR: OpenTxl not found: %s/bin/txl' % txl)
    (nicad / 'nicadclones').mkdir(exist_ok=True)

    started = local_now()
    stamp = started.strftime('%Y%m%d-%H%M%S-%f')  # America/Chicago local time
    out = Path(a.out).resolve() if a.out else root / 'results/b-05' / stamp
    out.mkdir(parents=True, exist_ok=False)
    work = root / 'results/b-05/_probe'
    work.mkdir(parents=True, exist_ok=True)
    nw = max(1, a.workers)
    probers = [Prober(nicad, txl, work, i, a.probe_timeout, a.config) for i in range(nw)]
    for p in probers:
        p.clean()

    commit = ''
    repo = root / 'data/repositories/salt'
    if (repo / '.git').exists() or (repo / '.git').is_file():
        try:
            commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        except (subprocess.CalledProcessError, OSError):
            commit = ''
    print('=' * 96)
    print('B-05 %s | NiCad Python grammar minimal reproduction | phase=%s | READ ONLY' % (VERSION, a.phase))
    print('=' * 96)
    print('Python: %s | NiCad: %s | OpenTxl: %s' % (sys.version.split()[0], nicad, txl))
    print('Salt commit: %s%s' % (commit or '(not available)',
                                 '' if (not commit or commit == EXPECTED_COMMIT) else '  !! differs from %s' % EXPECTED_COMMIT))
    print('Output: %s' % out, flush=True)

    summary = dict(status='RUNNING', experiment=EXPERIMENT, version=VERSION, phase=a.phase,
                   python_version=sys.version.split()[0], nicad_home=str(nicad), txl_home=str(txl),
                   started_local=started.isoformat(timespec='seconds'),
                   started_utc=started.astimezone(timezone.utc).isoformat(timespec='seconds'),
                   salt_commit=commit, workers=nw, nicad_config=a.config, probe_timeout=a.probe_timeout)
    rc = 0
    t0 = time.monotonic()
    try:
        if a.phase == 'calibrate':
            s = run_calibrate(a, out, nicad, txl, work)
            summary['calibrate'] = s
            if not s['control_ok']:
                rc = 3
        if a.phase in ('synthetic', 'all'):
            s = run_synthetic(a, out, probers)
            summary['synthetic'] = s
            if not s['control_ok']:
                rc = 3
        if a.phase in ('inventory', 'localize', 'ablate', 'all'):
            inp = Path(a.input_result).resolve() if a.input_result else default_input_result(root)
            if inp is None or not inp.is_dir():
                raise SystemExit('ERROR: no --input-result given and no results/b-03 or results/b-02 found')
            stage = Path(a.stage).resolve() if a.stage else root / 'results/b-01/_staging/salt'
            if not stage.is_dir():
                raise SystemExit('ERROR: staged source missing: %s' % stage)
            cats, src_csv = load_categories(inp)
            cc = Counter(cats.values())
            print('\nInput labels: %s (%d files) -> %s' % (src_csv, len(cats),
                                                         ', '.join('%s=%d' % kv for kv in sorted(cc.items()))))
            summary['input_result'] = str(inp)
            summary['input_labels'] = dict(cc)
            size_audit(cats, stage, out)
            if a.phase in ('inventory', 'all'):
                summary['inventory'] = run_inventory(a, out, cats, stage)
            if a.phase == 'inventory':
                files, missing = [], []
            else:
                files, missing = pick_files(cats, stage, a)
            if missing:
                print('WARNING: %d failing files missing from staging (skipped), e.g. %s' % (len(missing), missing[0]))
            summary['files_selected'] = len(files)
            summary['files_missing_in_stage'] = len(missing)
            if a.phase in ('localize', 'all'):
                summary['localize'] = run_localize(a, out, probers, cats, stage, files)
            if a.phase in ('ablate', 'all'):
                summary['ablate'] = run_ablate(a, out, probers, files, stage)
        summary['status'] = 'COMPLETE' if rc == 0 else 'COMPLETE_WITH_WARNINGS'
    finally:
        for p in probers:
            try:
                p.clean()
            except RuntimeError:
                pass
        summary['nicad_invocations'] = sum(p.calls for p in probers)
        summary['nicad_seconds'] = round(sum(p.seconds for p in probers), 1)
        summary['wall_seconds'] = round(time.monotonic() - t0, 1)
        summary['limitations'] = [
            'A minimal failing snippet shows what remains necessary at statement granularity; it is evidence, not a proof of the NiCad grammar rule.',
            'ablate rewrites use ast.unparse (changes line numbers); it diagnoses causes and is not analysis input.',
            'Per-file NiCad probes can differ from a whole-project run; compare totals before claiming exact reproduction.',
        ]
        (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        lines = ['# B-05 %s NiCad Python grammar minimal reproduction' % VERSION, '',
                 '- Phase: `%s`' % a.phase, '- Salt commit: `%s`' % (commit or 'n/a'),
                 '- NiCad invocations: %d (%.1fs NiCad time, %.1fs wall)' % (
                     summary['nicad_invocations'], summary['nicad_seconds'], summary['wall_seconds']), '']
        for k in ('calibrate', 'inventory', 'synthetic', 'localize', 'ablate'):
            if k in summary:
                lines += ['## %s' % k, '```json', json.dumps(summary[k], indent=2, ensure_ascii=False), '```', '']
        lines += ['## Limitations'] + ['- ' + x for x in summary['limitations']] + ['']
        (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n[B-05 %s] %s OUTPUT_DIR=%s' % (VERSION, summary['status'], out))
    return rc


if __name__ == '__main__':
    sys.exit(main())
