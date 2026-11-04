#!/usr/bin/env python3
"""Report for the gold-free evaluation (run_open.py): ESV and its controls vs Direct QA and CoT on every instance.

Every comparison is paired on the instances that have an open result. Against CoT maj@4 (the answer every method starts
from), a *fix* is wrong -> right and a *break* is right -> wrong; McNemar exact p on fixes vs breaks. Recognition is
scored as a detector of wrong initial answers. One model set per report, so pooled numbers never mix sizes.

  python experiments/analyze_open.py                 # current (large) solvers -> open_report_large.md
  python experiments/analyze_open.py --models small  # Qwen 7B / Mistral Nemo / Gemma 4B -> open_report_small.md
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import mcnemar_exact  # noqa: E402
import common  # noqa: E402
from common import SMALL_SOLVER_MODELS, SOLVER_MODELS, majority_answer, model_slug, score_answer  # noqa: E402
from run_open import METHODS, OPEN_VERSION, VOTE_RULE, rescore  # noqa: E402

LABELS = {"dqa_pass1": "Direct QA pass@1", "dqa_maj": "Direct QA maj@4", "cot_pass1": "CoT pass@1",
          "cot_maj": "CoT maj@4 (initial)", "self_refine": "Self-Refine (control)",
          "recheck3": "Recognition + 3 re-solves (control)", "esv": "ESV"}
ARMS = ("original", "breaking", "preserving")


def tokens(calls) -> int:
    return sum((c.get("usage") or {}).get("completion_tokens", 0) for c in calls if c)


def row_of(o: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    """Per-instance correctness (pass@1 as a fraction) and solver tokens for every method."""
    samples = base["samples"]
    fmt_ = o["answer_format"]

    def maj(method):
        return bool(score_answer(majority_answer([s["normalized_answer"] for s in samples[method]], fmt_), o["gold"], fmt_)["correct"])

    m = o["methods"]
    arm = "original" if o["problem_type"] == "original" else o["perturbation_type"]
    return {
        "arm": arm, "format": fmt_, "flagged": o["flagged"], "initial": bool(o["initial"]["correct"]),
        "correct": {"dqa_pass1": sum(s["correct"] for s in samples["dqa"]) / len(samples["dqa"]), "dqa_maj": maj("dqa"),
                    "cot_pass1": sum(s["correct"] for s in samples["cot"]) / len(samples["cot"]), "cot_maj": bool(o["initial"]["correct"]),
                    **{k: bool(m[k]["final_correct"]) for k in METHODS}},
        "refine_flagged": m["self_refine"]["flagged"],
        "tokens": {"esv": tokens([o["recognition"], *m["esv"]["stages"].values()]),
                   "recheck3": tokens([o["recognition"], *m["recheck3"]["stages"].values()]),
                   "self_refine": tokens(m["self_refine"]["stages"].values())},
        "esv": m["esv"],
    }


def pct(n: float, d: int) -> str:
    return f"{n:g}/{d} ({100 * n / d:.1f}%)" if d else "-"


def rate(n: float, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "-"


def table(header: List[str], rows: List[List[Any]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def paired(rows: List[Dict[str, Any]], a: str, b: str) -> Tuple[int, int, str]:
    """(a right & b wrong, b right & a wrong, McNemar p)."""
    wins = sum(r["correct"][a] and not r["correct"][b] for r in rows)
    losses = sum(r["correct"][b] and not r["correct"][a] for r in rows)
    p = mcnemar_exact(wins, losses)
    return wins, losses, "-" if p is None else f"{p:.3f}"


def accuracy_table(rows: List[Dict[str, Any]]) -> str:
    out = []
    for key in LABELS:
        acc = sum(r["correct"][key] for r in rows)
        if key in ("dqa_pass1", "cot_pass1", "cot_maj"):
            out.append([LABELS[key], rate(acc, len(rows)), "", "", "", ""])
            continue
        fixes, breaks, p = paired(rows, key, "cot_maj")
        tok = round(sum(r["tokens"][key] for r in rows) / len(rows)) if key in METHODS else ""
        out.append([LABELS[key], rate(acc, len(rows)), fixes, breaks, f"{fixes - breaks:+d} (p={p})", tok])
    return table(["Method", "accuracy", "fixes", "breaks", "net vs CoT maj@4", "tokens/instance"], out)


def detector_table(rows: List[Dict[str, Any]]) -> str:
    wrong = [r for r in rows if not r["initial"]]
    right = [r for r in rows if r["initial"]]
    out = []
    for label, flag in (("ESV recognition", lambda r: r["flagged"]), ("Self-Refine verdict", lambda r: r["refine_flagged"])):
        caught = sum(flag(r) for r in wrong)
        alarms = sum(flag(r) for r in right)
        out.append([label, pct(caught, len(wrong)), pct(alarms, len(right)), rate(caught, caught + alarms)])
    return table(["Detector", "catch rate (flags a wrong initial)", "false alarms (flags a right initial)", "precision"], out)


def after_flag_table(rows: List[Dict[str, Any]]) -> str:
    out = []
    for key, flag in (("esv", lambda r: r["flagged"]), ("recheck3", lambda r: r["flagged"]), ("self_refine", lambda r: r["refine_flagged"])):
        fw = [r for r in rows if flag(r) and not r["initial"]]
        fr = [r for r in rows if flag(r) and r["initial"]]
        out.append([LABELS[key], pct(sum(r["correct"][key] for r in fw), len(fw)), pct(sum(not r["correct"][key] for r in fr), len(fr))])
    return table(["Method", "flagged & wrong: fixed", "flagged & right: broken"], out)


def breakdown(rows: List[Dict[str, Any]], key: Callable, values) -> str:
    out = []
    for v in values:
        sub = [r for r in rows if key(r) == v]
        if not sub:
            continue
        out.append([v, len(sub)] + [rate(sum(r["correct"][k] for r in sub), len(sub)) for k in ("cot_maj", "self_refine", "recheck3", "esv")]
                   + [f"{paired(sub, 'esv', 'cot_maj')[0]} / {paired(sub, 'esv', 'cot_maj')[1]}"])
    return table(["group", "n", "CoT maj@4", "Self-Refine", "Recheck x3", "ESV", "ESV fixes / breaks"], out)


def section(name: str, rows: List[Dict[str, Any]]) -> List[str]:
    lines = [f"## {name} ({len(rows)} instances)", "", accuracy_table(rows), "",
             "ESV head to head (ESV-only / other-only, McNemar p): " + "; ".join(
                 f"vs {LABELS[b]} {w} / {l} (p={p})" for b in ("cot_maj", "self_refine", "recheck3") for w, l, p in [paired(rows, "esv", b)]),
             "", detector_table(rows), "", after_flag_table(rows), ""]
    return lines


def lifecycle(rows: List[Dict[str, Any]]) -> List[str]:
    """ESV stages on the instances where they matter: flagged, and the initial answer really is wrong."""
    sub = [r for r in rows if r["flagged"] and not r["initial"]]
    if not sub:
        return []
    lines = ["### ESV lifecycle on flagged wrong answers (pooled)", "", f"{len(sub)} instances. Judge verdicts are measurement only.", ""]
    out = []
    for v in ("CORRECT", "PARTIAL", "INCORRECT", None):
        s = [r for r in sub if r["esv"].get("diagnosis_verdict") == v]
        if s:
            out.append([f"diagnosis {v or 'ungraded'}", len(s), pct(sum(r["correct"]["esv"] for r in s), len(s))])
    for label, test in (("some strategy VALID", True), ("no strategy VALID", False)):
        s = [r for r in sub if ("VALID" in (r["esv"].get("strategy_verdicts") or [])) == test]
        out.append([label, len(s), pct(sum(r["correct"]["esv"] for r in s), len(s))])
    s = [r for r in sub if any(r["esv"]["derivations_correct"])]
    out.append(["some derivation correct (ceiling)", len(s), pct(sum(r["correct"]["esv"] for r in s), len(s))])
    lines += [table(["ESV stage outcome", "n", "fixed"], out), "",
              f"- strategies kept per run: {dict(sorted(Counter(len(r['esv']['strategies']) for r in sub).items()))}", ""]
    return lines


MODEL_SETS = {"large": SOLVER_MODELS, "small": SMALL_SOLVER_MODELS}


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Report for the gold-free evaluation.")
    common.add_benchmark_arg(parser)
    parser.add_argument("--models", choices=list(MODEL_SETS), default="large")
    args = parser.parse_args(argv)
    common.select_benchmark(args.benchmark)
    per_model: Dict[str, List[Dict[str, Any]]] = {}
    coverage = []
    for model in MODEL_SETS[args.models]:
        path = common.RESULTS_DIR / model_slug(model) / "open_records.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        base = json.loads((path.parent / "records.json").read_text(encoding="utf-8"))["instances"]
        opened = [rescore(o) for o in data["instances"].values() if o.get("version") == OPEN_VERSION]
        model = data["metadata"]["model"]
        per_model[model] = [row_of(o, base[o["instance_id"]]) for o in opened]
        coverage.append([model, f"{len(opened)} / {len(base)}"])
    if not per_model:
        print(f"No open_records.json for the {args.models} models; run run_open.py first.", file=sys.stderr)
        return 2

    pooled = [r for rows in per_model.values() for r in rows]
    out = [f"# Gold-free evaluation: ESV on every instance ({common.benchmark_info()['label']}, {args.models} models)", "",
           f"Version `{OPEN_VERSION}`, vote rule `{VOTE_RULE}` (the initial answer votes in ESV's and Recheck x3's final "
           "vote and wins ties). Every method starts from the CoT majority answer (CoT maj@4) and decides on its own "
           "whether to change it; the gold answer is used only to score the final answers. *Fixes* / *breaks* = instances "
           "the method turned wrong -> right / right -> wrong relative to CoT maj@4. Tokens = mean solver output tokens per "
           "instance beyond the 4 CoT samples (judge excluded). The two controls share ESV's answer checks; Recheck x3 "
           "also shares ESV's recognition decision, so ESV - Recheck x3 isolates the diagnosis and strategies.", "",
           table(["Model", "instances with open results"], coverage), ""]
    for model, rows in per_model.items():
        out += section(model, rows)
    out += section("Pooled", pooled)
    out += ["### Pooled, by arm", "", breakdown(pooled, lambda r: r["arm"], ARMS), "",
            "### Pooled, by answer format", "", breakdown(pooled, lambda r: r["format"], ("single", "multiple", "integer", "numeric")), ""]
    out += lifecycle(pooled)

    path = common.RESULTS_DIR / "analysis" / f"open_report_{args.models}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nSaved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
