r"""Scoring for the `expression` answer format: MATH-500 golds that are not a single plain number.

GSM8K and JEEBench answers are option letters or numbers, so `common.normalize_answer` could reduce every answer
to a letter set or a Decimal. 184 of the 500 MATH-500 problems answer with a LaTeX *expression* instead:
`\frac{14}{3}`, `3\sqrt{13}`, `90^\circ`, `(6,31,-1)`, `(-\infty, 2) \cup (3, \infty)`, `52_8`,
`\begin{pmatrix} -1/3 \\ 2/3 \\ 5/3 \end{pmatrix}`, `\text{Evelyn}`. This module is the only place that knows how
to compare those, and `common` routes the `expression` format here.

Two functions:
  normalize(value)  ->  a canonical string, or None when the answer is empty / unreadable / the NONE sentinel.
                        The canonical string is what gets stored in records and used as a dict key, so it must be
                        stable and idempotent: normalize(normalize(x)) == normalize(x).
  match(a, b)       ->  whether two answers are the same, in four escalating steps (see below).

`match` is deliberately ordered from cheapest and safest to most permissive:
  1. exact equality of canonical strings, then case-insensitively (`\text{East}` vs `east`);
  2. structural comparison, element by element, for anything that is a comma-separated list, tuple, interval, set
     or column vector - and the delimiters must agree, so `(3,4]` never equals `[3,4]`;
  3. exact numeric equality when both sides are plain numbers;
  4. symbolic equality via sympy, on a conservative subset: the LaTeX is converted to sympy syntax and the attempt
     is abandoned unless the conversion consumed every backslash and left no `_`, `,`, `|`, `=` or bracket. So
     `\frac{1}{2}` matches `0.5` and `(a+5)(b+2)` matches `ab+2a+5b+10`, while `52_8` (base 8) is never compared
     numerically with `52`, and `1 \pm \sqrt{19}` is only ever compared as a string.

What is deliberately NOT done: no decimal approximation of an exact value is accepted (`4.67` does not match
`\frac{14}{3}`), because "give the exact value" is part of what these problems ask. Equivalence checking on free
text is never perfect; it is applied identically to every method's answers and to the gold, so no arm is favoured.

Self-check (no API calls):  python experiments/tests/test_expression_answers.py
"""

import re
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Tuple

# A model answer that resolves to one of these is treated as "no answer" and scored incorrect, like an unreadable
# number elsewhere. `NONE` is the sentinel the perturbation verifiers are told to write.
UNREADABLE = {"", "none", "n/a", "na", "null", "nan", "?", "-", "--"}

# sympy is only consulted on short, fully converted expressions: a cap keeps a pathological `simplify` from hanging
# a worker thread, and nothing in the pipeline depends on the symbolic step succeeding.
MAX_SYMPY_CHARS = 90
MAX_SYMPY_DIGITS = 8  # reject 2^{1000}-style inputs before sympy expands them
NUMERIC_TOLERANCE = 1e-9

# ---- small LaTeX helpers ---------------------------------------------------------------------------

_ANSWER_PREFIX = re.compile(r"^\s*(?:so\s+|thus\s+|therefore\s+)?(?:the\s+)?(?:final\s+|correct\s+)?answer\s*(?:is|are|=|:)\s*", re.IGNORECASE)
# Commands whose single braced argument is kept and whose name is dropped: \boxed{3} -> 3.
_UNWRAP = ("boxed", "textbf", "textit", "mathrm", "mathbf", "mathit", "mathsf", "operatorname")
# Text commands, which MATH-500 uses for two different things:
#   the whole answer is a word  ->  \text{Evelyn}, \text{ellipse}, \text{east}: keep the word.
#   a unit stuck onto a value   ->  \frac{270}7\text{ degrees}, 864 \mbox{ inches}^2, 5.4 \text{ cents}: drop it,
#                                   together with any exponent attached to it, so "inches^2" does not leave "864^2".
# The problems are asked with "no units", so a gold that carries one must shed it or no answer could ever match.
_TEXT = ("text", "textrm", "mbox", "hbox")
_WHOLE_TEXT = re.compile(r"^\\(?:%s)\{(.*)\}$" % "|".join(_TEXT), re.DOTALL)
# Spacing and layout commands with no mathematical content.
_DROP = (r"\left", r"\right", r"\displaystyle", r"\limits", r"\!", r"\,", r"\;", r"\:", r"\ ", r"\quad", r"\qquad",
         r"\degree", r"\%", "%", "~")
_MATRIX_ENVS = ("pmatrix", "bmatrix", "vmatrix", "Bmatrix", "matrix", "array")

_OPENERS, _CLOSERS = "([{", ")]}"


def _find_group(text: str, start: int) -> Tuple[str, int]:
    """text[start] is '{'. Returns (contents, index just past the matching '}')."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    return text[start + 1:], len(text)  # unbalanced input: keep the rest


def _read_arg(text: str, i: int) -> Tuple[str, int]:
    r"""The argument of a command at position i: `{...}`, a single character (\frac12) or a command (\frac\pi2)."""
    while i < len(text) and text[i] == " ":
        i += 1
    if i >= len(text):
        return "", i
    if text[i] == "{":
        return _find_group(text, i)
    command = re.match(r"\\[A-Za-z]+", text[i:])
    if command:
        return command.group(0), i + command.end()
    return text[i], i + 1


def _unwrap(text: str, names: Tuple[str, ...]) -> str:
    r"""Drops each command in `names`, keeping its argument: \text{east} -> east. Longest name first."""
    names = tuple(sorted(names, key=len, reverse=True))
    out, i = [], 0
    while i < len(text):
        for name in names:
            if text.startswith("\\" + name, i):
                argument, end = _read_arg(text, i + len(name) + 1)
                out.append(_unwrap(argument, names))
                i = end
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _resolve_text_groups(text: str) -> str:
    r"""Applies the two readings of \text{...} described at _TEXT: the whole answer keeps its word, an annotation
    attached to a value is dropped with its exponent."""
    whole = _WHOLE_TEXT.match(text.strip())
    if whole:
        return whole.group(1).strip()
    out, i = [], 0
    while i < len(text):
        for name in sorted(_TEXT, key=len, reverse=True):
            if text.startswith("\\" + name, i):
                _, end = _read_arg(text, i + len(name) + 1)
                exponent = re.match(r"\^\s*(?:\{[^{}]*\}|\w)", text[end:])  # "inches^2" -> drop the ^2 as well
                i = end + (exponent.end() if exponent else 0)
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _canonical_commands(text: str) -> str:
    r"""Rewrites \frac and \sqrt into one spelling: \frac12 -> \frac{1}{2}, \sqrt3 -> \sqrt{3}, recursively."""
    out, i = [], 0
    while i < len(text):
        if text.startswith(r"\frac", i):
            numerator, j = _read_arg(text, i + 5)
            denominator, k = _read_arg(text, j)
            out.append(r"\frac{%s}{%s}" % (_canonical_commands(numerator), _canonical_commands(denominator)))
            i = k
        elif text.startswith(r"\sqrt", i):
            j, root = i + 5, ""
            if j < len(text) and text[j] == "[":
                end = text.find("]", j)
                if end > 0:
                    root, j = text[j + 1:end], end + 1
            radicand, k = _read_arg(text, j)
            out.append(r"\sqrt%s{%s}" % (f"[{root}]" if root else "", _canonical_commands(radicand)))
            i = k
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _flatten_matrices(text: str) -> str:
    r"""\begin{pmatrix} 1 \\ 2 \end{pmatrix} -> (1,2), so a column vector compares element by element."""
    for env in _MATRIX_ENVS:
        pattern = re.compile(r"\\begin\{%s\*?\}(.*?)\\end\{%s\*?\}" % (env, env), re.DOTALL)
        text = pattern.sub(lambda m: "(" + re.sub(r"\\\\|&", ",", m.group(1)).strip() + ")", text)
    return text


def _strip_thousands(text: str) -> str:
    """1{,}000 and 10,080 (after \\! removal) -> 1000, but only when the WHOLE answer is one such number, so that a
    tuple like (6,31,-1) is never silently joined into a number."""
    text = re.sub(r"\{,\}", "", text)
    return text.replace(",", "") if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text) else text


def as_number(text: str) -> Optional[Decimal]:
    """The Decimal value of `text` only when the WHOLE string is a plain number; never a number found inside it
    (`52_8` is base 8, not 52)."""
    if not re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)", text or ""):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


# ---- normalization --------------------------------------------------------------------------------

def normalize(value) -> Optional[str]:
    """Canonical string form of an expression answer, or None if there is no readable answer."""
    text = str(value if value is not None else "").strip()
    text = _ANSWER_PREFIX.sub("", text).strip()
    text = text.replace("\\$", "").replace("$", "").replace("**", "").strip()
    text = text.replace(r"\{", "{").replace(r"\}", "}")
    text = _unwrap(text, _UNWRAP)          # \boxed{...} first, so a \text{...} inside it can be seen as the whole answer
    text = _resolve_text_groups(text.strip())
    text = _flatten_matrices(text)
    for token in _DROP:
        text = text.replace(token, "")
    text = re.sub(r"\\[dtc]frac", r"\\frac", text)
    text = _canonical_commands(text)
    text = re.sub(r"\^\s*\{?\s*\\?circ\s*\}?", "", text)  # 90^\circ and 90^{\circ} -> 90
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"_\{(\w+)\}", r"_\1", text)            # 4210_{5} -> 4210_5 (a base, kept as written)
    text = _strip_thousands(text)
    text = text.rstrip(".,;")
    # "x=5" -> "5", but "5x-7y+11z+4=0" keeps its equation: only a bare one- or two-character left side is dropped.
    halves = text.split("=")
    if len(halves) == 2 and re.fullmatch(r"[A-Za-z]\w?|\\[A-Za-z]+", halves[0]) and halves[1]:
        text = halves[1]
    if text.lower() in UNREADABLE:
        return None
    number = as_number(text)
    if number is not None:
        return format(number.normalize(), "f")  # 2.50 -> 2.5, so the canonical form of a number is unique
    return text or None


# ---- structural comparison ------------------------------------------------------------------------

def _split_top_level(text: str, separator: str = ",") -> List[str]:
    parts, depth, current = [], 0, []
    for ch in text:
        if ch in _OPENERS:
            depth += 1
        elif ch in _CLOSERS:
            depth -= 1
        if ch == separator and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _components(text: str) -> Optional[Tuple[str, str, List[str]]]:
    """(opening delimiter, closing delimiter, elements) when `text` is a list, tuple, interval, set or vector;
    None when it is a single expression. `(a+5)(b+2)` is NOT a tuple: its first bracket closes before the end."""
    opener = closer = ""
    body = text
    if text[:1] in _OPENERS and text[-1:] in _CLOSERS:
        depth = 0
        for i, ch in enumerate(text):
            depth += (ch in _OPENERS) - (ch in _CLOSERS)
            if depth == 0:
                if i != len(text) - 1:
                    return None  # the opening bracket closes early, so the whole string is not one bracketed group
                break
        opener, closer, body = text[0], text[-1], text[1:-1]
    elements = _split_top_level(body)
    if len(elements) < 2:
        return None
    return opener, closer, elements


# ---- symbolic comparison --------------------------------------------------------------------------

_SYMPY_FUNCTIONS = ("arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "sin", "cos", "tan", "cot", "sec", "csc",
                    "log", "ln", "exp")
_SYMPY_CONSTANTS = ((r"\pi", "pi"), (r"\infty", "oo"), (r"\cdot", "*"), (r"\times", "*"), (r"\div", "/"))


def _to_sympy_text(text: str) -> Optional[str]:
    r"""LaTeX -> sympy syntax, or None when anything is left that must not be compared symbolically.

    Bailing out is the point: whatever this function does not fully understand is compared as a string instead.
    """
    out, i = [], 0
    while i < len(text):  # \frac{a}{b} -> ((a)/(b)), \sqrt[n]{a} -> ((a)**(1/(n)))
        if text.startswith(r"\frac{", i):
            numerator, j = _find_group(text, i + 5)
            denominator, k = _find_group(text, j)
            a, b = _to_sympy_text(numerator), _to_sympy_text(denominator)
            if a is None or b is None:
                return None
            out.append(f"(({a})/({b}))")
            i = k
        elif text.startswith(r"\sqrt", i):
            j, root = i + 5, None
            if j < len(text) and text[j] == "[":
                end = text.find("]", j)
                if end < 0:
                    return None
                root, j = text[j + 1:end], end + 1
            if j >= len(text) or text[j] != "{":
                return None
            radicand, k = _find_group(text, j)
            inner = _to_sympy_text(radicand)
            converted_root = _to_sympy_text(root) if root else None
            if inner is None or (root and converted_root is None):
                return None
            out.append(f"(({inner})**(1/({converted_root})))" if root else f"sqrt({inner})")
            i = k
        else:
            out.append(text[i])
            i += 1
    converted = "".join(out)
    for name in _SYMPY_FUNCTIONS:  # \sin{x} and \sin x and \cot x -> sin(x), cot(x)
        converted = re.sub(r"\\%s\{([^{}]*)\}" % name, r"%s(\1)" % name, converted)
        converted = re.sub(r"\\%s\(" % name, "%s(" % name, converted)
        converted = re.sub(r"\\%s([A-Za-z0-9])" % name, r"%s(\1)" % name, converted)
    converted = converted.replace("ln(", "log(")
    for latex, python in _SYMPY_CONSTANTS:
        converted = converted.replace(latex, python)
    converted = converted.replace("{", "(").replace("}", ")").replace("^", "**")
    converted = re.sub(r"(?<![A-Za-z])i(?![A-Za-z])", "I", converted)      # 6-5i -> 6-5*I
    # Implicit multiplication (3sqrt(13), 2k, (a+5)(b+2), ab) is left to sympy's own parser: see _sympy_match.
    # Anything below means the expression is not safe to compare symbolically: a surviving LaTeX command, a base
    # subscript, a list, a relation, an absolute value, or a number big enough to make sympy expensive.
    if "\\" in converted or re.search(r"[_,|<>=!\[\]]", converted):
        return None
    if any(len(run) > MAX_SYMPY_DIGITS for run in re.findall(r"\d+", converted)):
        return None
    return converted


def _sympy_match(a: str, b: str) -> bool:
    if max(len(a), len(b)) > MAX_SYMPY_CHARS:
        return False
    left, right = _to_sympy_text(a), _to_sympy_text(b)
    if left is None or right is None or not left or not right:
        return False
    try:
        import sympy
        from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations
        # implicit_multiplication_application is what makes `3sqrt(13)`, `2k`, `(a+5)(b+2)` and `ab` parse the way
        # they are written in LaTeX, while leaving `pi`, `oo`, `sqrt` and the trig functions alone.
        transformations = standard_transformations + (implicit_multiplication_application,)
        first = parse_expr(left, transformations=transformations)
        second = parse_expr(right, transformations=transformations)
        if not (first.free_symbols or second.free_symbols):  # a pure value: settle it numerically, never symbolically
            difference = complex(first.evalf() - second.evalf())
            return abs(difference) <= NUMERIC_TOLERANCE * max(1.0, abs(complex(first.evalf())))
        if first.free_symbols != second.free_symbols:
            return False
        return bool(sympy.simplify(first - second) == 0)
    except Exception:  # unparsable, or sympy is not installed: fall back to "not equal"
        return False


# ---- the comparison used by common.answers_match --------------------------------------------------

def match(a: Optional[str], b: Optional[str]) -> bool:
    """Whether two expression answers are the same. Inputs are normally already normalized; normalizing again is
    idempotent, so callers may pass raw strings."""
    if a is None or b is None:
        return False
    a, b = normalize(a), normalize(b)
    if a is None or b is None:
        return False
    if a == b or a.lower() == b.lower():
        return True
    left, right = _components(a), _components(b)
    if left and right:
        if left[0] != right[0] or left[1] != right[1] or len(left[2]) != len(right[2]):
            return False
        return all(match(x, y) for x, y in zip(left[2], right[2]))
    if bool(left) != bool(right):
        return False  # a tuple is never equal to a single expression
    first, second = as_number(a), as_number(b)
    if first is not None and second is not None:
        return first == second
    return _sympy_match(a, b)
