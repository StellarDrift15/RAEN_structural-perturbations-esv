#!/usr/bin/env python3
r"""Unit checks for the `expression` answer format (experiments/expression_answers.py). No API calls, no network.

Three groups:
  1. normalize() on the shapes MATH-500 actually uses, and its idempotence;
  2. match() on pairs that MUST be equal (the same value written differently) and pairs that MUST NOT be
     (a different value, a different base, a different interval end, a decimal approximation of an exact value);
  3. every gold in data/math500/source.json, if it has been pulled: each one must normalize to something readable,
     normalize idempotently, and match itself.

Run: python experiments/tests/test_expression_answers.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import expression_answers as expr  # noqa: E402

# (input, expected canonical form)
NORMALIZE = [
    (r"\frac{14}{3}", r"\frac{14}{3}"),
    (r"\dfrac{17}{50}", r"\frac{17}{50}"),
    (r"\tfrac12", r"\frac{1}{2}"),
    (r"\frac43", r"\frac{4}{3}"),
    (r"\boxed{\frac{3}{2}}", r"\frac{3}{2}"),
    (r"$3\sqrt{13}$", r"3\sqrt{13}"),
    (r"\sqrt3", r"\sqrt{3}"),
    (r"90^\circ", "90"),
    (r"145^{\circ}", "145"),
    (r"\text{Evelyn}", "Evelyn"),
    (r"\text{east}", "east"),
    (r"\boxed{\text{ellipse}}", "ellipse"),
    (r"\frac{270}7\text{ degrees}", r"\frac{270}{7}"),   # a unit annotation is dropped ...
    (r"864 \mbox{ inches}^2", "864"),                    # ... together with the exponent that belongs to it
    (r"5.4 \text{ cents}", "5.4"),
    (r"\left( 3, \frac{\pi}{2} \right)", r"(3,\frac{\pi}{2})"),
    (r"\begin{pmatrix} -1/3 \\ 2/3 \\ 5/3 \end{pmatrix}", "(-1/3,2/3,5/3)"),
    (r"\begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}", "(-1,0,0,-1)"),
    (r"10,\!080", "10080"),
    ("58,500", "58500"),
    (r"\$32,\!348", "32348"),
    (r"\$18.90", "18.9"),
    ("x=5", "5"),
    ("y = 2x + 3", "2x+3"),
    ("5x - 7y + 11z + 4 = 0", "5x-7y+11z+4=0"),  # a real equation is kept whole
    ("4210_{5}", "4210_5"),
    ("52_8", "52_8"),
    (r"1 \pm \sqrt{19}", r"1\pm\sqrt{19}"),
    (r"(-\infty, 2) \cup (3, \infty)", r"(-\infty,2)\cup(3,\infty)"),
    (r"\{1\pm\sqrt{5},-2\}", r"{1\pm\sqrt{5},-2}"),
    ("2.50", "2.5"),
    ("The answer is \\frac{1}{2}", r"\frac{1}{2}"),
    ("  7  ", "7"),
]

UNREADABLE = ["", "   ", "NONE", "none", "n/a", None, "null"]

EQUAL = [
    (r"\frac{1}{2}", "0.5"),                       # sympy, pure value
    (r"\dfrac{14}{3}", r"\frac{14}{3}"),
    (r"\frac{3\sqrt{3}}{4}", r"\frac{3}{4}\sqrt{3}"),
    ("3\\sqrt{13}", r"\sqrt{117}"),
    (r"90^\circ", "90"),
    (r"\text{East}", "east"),                      # case-insensitive
    ("(a+5)(b+2)", "ab+2a+5b+10"),
    ("2k+2", "2*k+2"),
    (r"x^5 - x^4 + x^3 - x^2 + x - 1", "x^5-x^4+x^3-x^2+x-1"),
    (r"\left( 3, \frac{\pi}{2} \right)", r"(3, \frac{\pi}{2})"),
    ("(6,31,-1)", "(6, 31, -1)"),
    (r"\begin{pmatrix} 1/5 \\ -18/5 \end{pmatrix}", "(1/5,-18/5)"),
    ("x=5", "5"),
    (r"\frac{\pi}{2}", r"\dfrac{\pi}{2}"),
    (r"\frac{270}7\text{ degrees}", r"\frac{270}{7}"),   # the gold carries a unit, the solver was told not to
    (r"864 \mbox{ inches}^2", "864"),
]

NOT_EQUAL = [
    (r"\frac{14}{3}", "4.67"),                     # a decimal approximation is not the exact value asked for
    (r"\frac{14}{3}", r"\frac{3}{14}"),
    ("52_8", "52"),                                # base 8, never compared as a plain number
    ("52_8", "40_9"),
    ("(3,4]", "[3,4]"),                            # the interval ends differ
    ("(3,4]", "(3,4)"),
    ("(6,31,-1)", "(6,-1,31)"),                    # order matters in a coordinate tuple
    ("(6,31,-1)", "(6,31)"),
    ("(6,31,-1)", "36"),                           # a tuple is never a single value
    (r"1 \pm \sqrt{19}", r"1 + \sqrt{19}"),
    (r"\text{even}", r"\text{odd}"),
    (r"3\sqrt{13}", r"3\sqrt{14}"),
    ("x^2-4", "x^2+4"),
    (r"(-\infty, 2) \cup (3, \infty)", r"(-\infty, 2) \cup (4, \infty)"),
]


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        raise SystemExit(1)


def main() -> int:
    failures = []
    for raw, want in NORMALIZE:
        got = expr.normalize(raw)
        if got != want:
            failures.append(f"normalize({raw!r}) = {got!r}, expected {want!r}")
        elif expr.normalize(got) != got:
            failures.append(f"normalize is not idempotent on {raw!r}: {got!r} -> {expr.normalize(got)!r}")
    check(not failures, f"normalize() on {len(NORMALIZE)} MATH-500 answer shapes\n       " + "\n       ".join(failures))

    check(all(expr.normalize(v) is None for v in UNREADABLE), "empty / NONE / n/a answers normalize to None")
    check(all(not expr.match(v, "7") for v in UNREADABLE), "an unreadable answer never matches")

    failures = [f"{a!r} should equal {b!r}" for a, b in EQUAL if not expr.match(a, b)]
    failures += [f"{b!r} should equal {a!r} (not symmetric)" for a, b in EQUAL if not expr.match(b, a)]
    check(not failures, f"match() accepts {len(EQUAL)} equivalent pairs\n       " + "\n       ".join(failures))

    failures = [f"{a!r} must NOT equal {b!r}" for a, b in NOT_EQUAL if expr.match(a, b)]
    failures += [f"{b!r} must NOT equal {a!r} (not symmetric)" for a, b in NOT_EQUAL if expr.match(b, a)]
    check(not failures, f"match() rejects {len(NOT_EQUAL)} non-equivalent pairs\n       " + "\n       ".join(failures))

    source = Path(__file__).resolve().parents[1] / "data" / "math500" / "source.json"
    if not source.exists():
        print(f"SKIP  {source} not pulled yet; run: python experiments/benchmarks.py math500")
        return 0
    rows = json.loads(source.read_text(encoding="utf-8"))
    failures = []
    for row in rows:
        gold = row["gold"]
        canonical = expr.normalize(gold)
        if canonical is None:
            failures.append(f"{row['source_index']}: gold {gold!r} normalizes to None")
        elif expr.normalize(canonical) != canonical:
            failures.append(f"{row['source_index']}: not idempotent, {gold!r} -> {canonical!r} -> {expr.normalize(canonical)!r}")
        elif not expr.match(gold, gold):
            failures.append(f"{row['source_index']}: gold {gold!r} does not match itself")
    check(not failures, f"every one of the {len(rows)} MATH-500 golds is readable, stable and self-matching\n       "
                        + "\n       ".join(failures[:20]))

    # The perturbation pipeline rejects a generated answer that does not have the shape its format requires, so
    # every published gold must pass that same gate - otherwise no perturbation of such a problem could be accepted.
    import build_dataset  # noqa: E402  (imported here: it pulls in tqdm, which the checks above do not need)
    import common  # noqa: E402
    failures = [f"{r['source_index']}: {r['gold']!r} ({r['type']})" for r in rows
                if not build_dataset.valid_answer(common.normalize_answer(r["gold"], common.answer_format_of(r["type"])),
                                                  common.answer_format_of(r["type"]))]
    check(not failures, f"every gold passes build_dataset.valid_answer for its format\n       " + "\n       ".join(failures[:20]))

    expressions = [expr.normalize(r["gold"]) for r in rows if r["type"] == "Expression"]
    longest_word = max((len(w) for e in expressions for w in re.findall(r"[A-Za-z]+", e)), default=0)
    print(f"\n{len(rows)} golds: {len(set(expr.normalize(r['gold']) for r in rows))} distinct canonical forms; "
          f"{len(expressions)} Expression golds, longest {max(len(e) for e in expressions)} chars, "
          f"longest run of letters {longest_word}")
    print("All expression-format checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
