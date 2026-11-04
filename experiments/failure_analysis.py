#!/usr/bin/env python3
"""Where ESV v4 loses: a taxonomy of its consistent-failure runs, and free (no API) tests of fixes for ESV v2.

Reads only the frozen Experiment 1 records (results_frozen/jeebench/experiment1_esv_v4), so the numbers cannot drift.

  python experiments/failure_analysis.py      -> results/analysis/v4_failure_analysis.md
"""

import json
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import EXPERIMENT_DIR, answers_match, is_choice_format, normalize_answer  # noqa: E402

FROZEN_DIR = EXPERIMENT_DIR / "results_frozen" / "jeebench" / "experiment1_esv_v4"
MODELS = {"qwen-2.5-7b-instruct": "Qwen 2.5 7B", "mistral-nemo": "Mistral Nemo 12B", "gemma-3-4b-it": "Gemma 3 4B"}
DERIVATIONS = ("esv_independent_solve", "esv_recovery", "esv_cross_check")
OUTCOMES = ("recovered", "lost: independent re-solve only", "lost: adjudication chose wrong",
            "stubborn: repeated known-wrong", "recycled: another CoT wrong answer", "near miss", "other wrong")


def load_runs() -> List[Dict[str, Any]]:
    runs = []
    for slug, name in MODELS.items():
        records = json.loads((FROZEN_DIR / slug / "records.json").read_text(encoding="utf-8"))["instances"]
        for r in records.values():
            if r.get("esv"):
                r["model_name"] = name
                runs.append(r)
    return runs


def norm(r: Dict[str, Any], answer: Any) -> Optional[str]:
    return normalize_answer(answer, r["answer_format"])


def gold(r: Dict[str, Any]) -> Optional[str]:
    return norm(r, r["gold"])


def stage_answer(r: Dict[str, Any], stage: str) -> Optional[str]:
    s = r["esv"]["stages"].get(stage)
    return s.get("normalized_answer") if s else None


def is_near_miss(r: Dict[str, Any], answer: Optional[str]) -> bool:
    """One option off on a multiple-correct MCQ, or within 5% on a number."""
    g, fmt_ = gold(r), r["answer_format"]
    if answer is None or g is None:
        return False
    if fmt_ == "multiple":
        return len(set(answer) ^ set(g)) == 1
    if fmt_ in ("integer", "numeric"):
        try:
            a, b = Decimal(answer), Decimal(g)
            return b != 0 and abs(a - b) / abs(b) <= Decimal("0.05")
        except InvalidOperation:
            return False
    return False


def outcome(r: Dict[str, Any]) -> str:
    e, fmt_ = r["esv"], r["answer_format"]
    if e["final_correct"]:
        return "recovered"
    if e["independent_correct"] and not e["recovery_correct"] and not e["cross_check_correct"]:
        return "lost: independent re-solve only"
    if e["recovery_correct"] or e["cross_check_correct"]:
        return "lost: adjudication chose wrong"
    final = e["final_normalized"]
    if e["final_repeated_initial"]:
        return "stubborn: repeated known-wrong"
    if any(answers_match(final, s["normalized_answer"], fmt_) for s in r["samples"]["cot"]):
        return "recycled: another CoT wrong answer"
    if is_near_miss(r, final):
        return "near miss"
    return "other wrong"


def vote(answers: List[Optional[str]], fmt_: str, exclude: Optional[str] = None) -> Optional[str]:
    """Most common readable answer (first-listed wins ties), optionally never the known-wrong one."""
    pool = [a for a in answers if a is not None and not (exclude and answers_match(a, exclude, fmt_))]
    if not pool:
        return None
    counts = Counter(pool)
    best = max(counts.values())
    return next(a for a in pool if counts[a] == best)


def option_vote(answers: List[Optional[str]], threshold: float = 0.5) -> Optional[str]:
    """Multiple-correct MCQ: keep each option chosen by more than `threshold` of the readable answers."""
    pool = [a for a in answers if a]
    if not pool:
        return None
    return "".join(o for o in "ABCD" if sum(o in a for a in pool) / len(pool) > threshold) or None


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0"


def table(header: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def main() -> int:
    common.select_benchmark("jeebench")  # Experiment 1 ran on JEEBench only
    runs = load_runs()
    names = list(MODELS.values())
    by_model = {n: [r for r in runs if r["model_name"] == n] for n in names}
    out: List[str] = ["# ESV v4 failure analysis (JEEBench dev set, Experiment 1)", "",
                      f"{len(runs)} ESV runs on consistent failures (all 4 CoT samples wrong). Source: `{FROZEN_DIR.relative_to(EXPERIMENT_DIR)}`.", ""]

    # 1. Outcome taxonomy
    out += ["## 1. What happened on each failure", "",
            "Mutually exclusive, assigned in this order. *Lost* = a correct answer was produced at some stage but not kept.", ""]
    rows = []
    for o in OUTCOMES:
        rows.append([o] + [pct(sum(outcome(r) == o for r in by_model[n]), len(by_model[n])) for n in names]
                    + [pct(sum(outcome(r) == o for r in runs), len(runs))])
    out += [table(["Outcome"] + names + ["Pooled"], rows), ""]
    rows = []
    for fmt_ in ("single", "multiple", "integer", "numeric"):
        sub = [r for r in runs if r["answer_format"] == fmt_]
        c = Counter(outcome(r) for r in sub)
        rows.append([fmt_, len(sub)] + [c[o] for o in OUTCOMES])
    out += ["By answer format (pooled counts):", "", table(["Format", "n"] + list(OUTCOMES), rows), ""]

    # 2. Gold-free aggregation over what ESV already produced (no new API calls)
    out += ["## 2. Free test: would a different final rule have done better?", "",
            "Each rule picks a final answer from outputs ESV v4 already produced, without the gold answer. "
            "*Excl. known-wrong* never returns the answer the model was told is wrong.", ""]

    def answers_of(r):
        return [stage_answer(r, s) for s in DERIVATIONS]

    rules = {
        "v4 as run (agree, else adjudicate)": lambda r: r["esv"]["final_normalized"],
        "recovery only (original lifecycle)": lambda r: stage_answer(r, "esv_recovery"),
        "vote of 3 derivations": lambda r: vote(answers_of(r), r["answer_format"]),
        "vote of 3 derivations, excl. known-wrong": lambda r: vote(answers_of(r), r["answer_format"], exclude=gold_free_wrong(r)),
        "vote of 3 derivations + adjudicator, excl. known-wrong":
            lambda r: vote(answers_of(r) + [stage_answer(r, "esv_final_answer")], r["answer_format"], exclude=gold_free_wrong(r)),
        "v4, but option-vote on multiple-MCQ": lambda r: option_vote(answers_of(r)) if r["answer_format"] == "multiple" else r["esv"]["final_normalized"],
        "retry control": lambda r: r["retry"].get("normalized_answer"),
        "ceiling: any derivation correct": None,
    }
    rows = []
    for label, rule in rules.items():
        cells = []
        for group in [by_model[n] for n in names] + [runs]:
            if rule is None:
                k = sum(any(r["esv"].get(f) for f in ("independent_correct", "recovery_correct", "cross_check_correct", "final_correct")) for r in group)
            else:
                k = sum(answers_match(rule(r), gold(r), r["answer_format"]) for r in group)
            cells.append(pct(k, len(group)))
        rows.append([label] + cells)
    out += [table(["Final-answer rule"] + names + ["Pooled"], rows), ""]

    # 3. Multiple-correct MCQs: how far off are the answers?
    multi = [r for r in runs if r["answer_format"] == "multiple"]
    out += ["## 3. Multiple-correct MCQs (the largest group)", ""]
    rows = []
    for label, get in (("ESV final", lambda r: r["esv"]["final_normalized"]), ("retry", lambda r: r["retry"].get("normalized_answer")),
                       ("initial (known-wrong)", lambda r: gold_free_wrong(r))):
        diffs = Counter(len(set(get(r) or "") ^ set(gold(r))) for r in multi)
        option_acc = sum(sum((o in (get(r) or "")) == (o in gold(r)) for o in "ABCD") for r in multi) / (4 * len(multi))
        rows.append([label, f"{100 * option_acc:.1f}%"] + [diffs[k] for k in range(5)])
    out += [f"{len(multi)} runs. Options wrong = size of the symmetric difference with the gold set.", "",
            table(["Answer", "per-option accuracy", "0 wrong", "1 wrong", "2 wrong", "3 wrong", "4 wrong"], rows), ""]
    gsize = Counter(len(gold(r)) for r in multi)
    fsize = Counter(len(r["esv"]["final_normalized"] or "") for r in multi)
    out += [f"Gold set size: {dict(sorted(gsize.items()))}; ESV final set size: {dict(sorted(fsize.items()))}.", ""]

    # 4. Stage behaviour
    out += ["## 4. Stage behaviour", ""]
    rows = []
    for n in names + ["Pooled"]:
        g = runs if n == "Pooled" else by_model[n]
        fmt_ok = lambda r, s: r["esv"]["stages"][s].get("normalized_answer") is not None
        rec_wrong = sum(answers_match(stage_answer(r, "esv_recovery"), gold_free_wrong(r), r["answer_format"]) for r in g)
        cross_wrong = sum(answers_match(stage_answer(r, "esv_cross_check"), gold_free_wrong(r), r["answer_format"]) for r in g)
        indep_wrong = sum(answers_match(stage_answer(r, "esv_independent_solve"), gold_free_wrong(r), r["answer_format"]) for r in g)
        not_recog = sum(r["esv"]["recognized"] is False for r in g)
        self_unverified = sum(r["esv"].get("strategy_self_verified") is False for r in g)
        all3 = sum(len({a for a in answers_of(r) if a}) == 1 and all(answers_of(r)) for r in g)
        rows.append([n, pct(indep_wrong, len(g)), pct(rec_wrong, len(g)), pct(cross_wrong, len(g)), pct(not_recog, len(g)),
                     pct(self_unverified, len(g)), pct(all3, len(g))])
    out += [table(["Model", "indep. = known-wrong", "recovery = known-wrong", "cross-check = known-wrong", "recognition said 'no failure'",
                   "strategy failed own check", "all 3 derivations agree"], rows), ""]

    # 5. Judge verdicts vs outcome, pooled
    out += ["## 5. Does the quality of diagnosis / strategy matter? (judge-graded, pooled)", ""]
    rows = []
    for d in ("CORRECT", "PARTIAL", "INCORRECT"):
        for s in ("VALID", "INVALID"):
            g = [r for r in runs if r["esv"].get("diagnosis_verdict") == d and r["esv"].get("strategy_verdict") == s]
            if g:
                rows.append([d, s, len(g), pct(sum(r["esv"]["final_correct"] for r in g), len(g)),
                             pct(sum(bool(r["esv"]["recovery_correct"]) for r in g), len(g)),
                             pct(sum(bool(r["retry"]["correct"]) for r in g), len(g))])
    out += [table(["Diagnosis", "Strategy", "n", "ESV final", "recovery stage", "retry"], rows), ""]
    reasons = Counter()
    for r in runs:
        if r["esv"].get("strategy_verdict") == "INVALID":
            text = ((r["esv"]["stages"]["judge_diagnosis"].get("parsed") or {}).get("strategy_explanation") or "").lower()
            for key, words in (("same as failed method", ("same", "identical", "repeat", "essentially the")),
                               ("misses the changed/critical condition", ("condition", "ignore", "overlook", "miss")),
                               ("mathematically wrong / leads to wrong answer", ("incorrect", "wrong", "error", "invalid"))):
                if any(w in text for w in words):
                    reasons[key] += 1
    n_invalid = sum(r["esv"].get("strategy_verdict") == "INVALID" for r in runs)
    out += [f"Keywords in the judge's explanation for the {n_invalid} INVALID strategies (not exclusive): "
            + ", ".join(f"{k} {v}" for k, v in reasons.most_common()) + ".", ""]

    # 6. Cost per stage
    out += ["## 6. Output tokens per ESV run, by stage (pooled mean)", ""]
    tok = Counter()
    for r in runs:
        for name, s in r["esv"]["stages"].items():
            tok[name] += (s.get("usage") or {}).get("completion_tokens", 0)
        tok["retry (for comparison)"] += (r["retry"].get("usage") or {}).get("completion_tokens", 0)
    out += [table(["Stage", "mean output tokens"], [[k, round(v / len(runs))] for k, v in tok.items()]), ""]

    path = common.RESULTS_DIR / "analysis" / "v4_failure_analysis.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nSaved {path}")
    return 0


def gold_free_wrong(r: Dict[str, Any]) -> Optional[str]:
    """The answer the model was told is wrong (its modal CoT answer); known without the gold answer."""
    return norm(r, r["esv"]["initial_answer"])


if __name__ == "__main__":
    raise SystemExit(main())
