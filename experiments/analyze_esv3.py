#!/usr/bin/env python3
"""Stage-by-stage report for ESV v3 (run_esv3.py) -> results/analysis/esv3_report.md.

Stage 1 needs only the CoT samples (records.json), so it is reported even before run_esv3.py has run. Stages 2-3 are
reported from esv3_records.json. Every comparison is paired on the same instances; fixes / breaks are vs CoT maj@4,
the answer every method starts from.

  python experiments/analyze_esv3.py                    # current (large) solvers
  python experiments/analyze_esv3.py --limit-pairs 30   # stage 1 on the first 30 pairs only
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import (  # noqa: E402
    SMALL_SOLVER_MODELS, SOLVER_MODELS, answers_match, majority_answer, mcnemar_exact, model_slug, normalize_answer,
    score_answer,
)
from run_esv3 import DECISION_RULE, V3_VERSION, candidates_of, decide, is_sound, redecide  # noqa: E402

MODEL_SETS = {"large": SOLVER_MODELS, "small": SMALL_SOLVER_MODELS}
ARMS = ("original", "breaking", "preserving")
# Fixed before the 30-pair dev run (2026-11-03): ESV v3 must beat CoT maj@4 by >= 3 pp AND beat USC.
GO_MARGIN_PP = 3.0
LABELS = {"dqa_maj": "Direct QA maj@4", "cot_pass1": "CoT pass@1", "cot_maj": "CoT maj@4 (initial)", "cot_pass4": "CoT pass@4 (oracle choice)",
          "usc": "Universal Self-Consistency", "esv3": "ESV v3 (stages 1-3)"}


def pct(n: float, d: int) -> str:
    return f"{n:g}/{d} ({100 * n / d:.1f}%)" if d else "-"


def rate(n: float, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "-"


def table(header: List[str], rows: List[List[Any]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def tokens(call) -> int:
    return ((call or {}).get("usage") or {}).get("completion_tokens", 0)


def naive_answers(dataset: Dict[str, Any]) -> Dict[Any, Optional[str]]:
    """pair_id -> the trap answer of a breaking perturbation (what the original strategy S yields on P'), normalized."""
    out = {}
    for p in dataset.get("pairs", []):
        if p["perturbation_type"] != "breaking":
            continue
        values = [(p["validation"].get(k) or {}).get("naive_answer") for k in ("model_b", "model_c", "model_a")]
        value = next((v for v in values if v and str(v).upper() != "NONE"), None)
        out[p["id"]] = normalize_answer(value, p["answer_format"]) if value else None
    return out


def base_row(inst: Dict[str, Any], naive: Optional[str] = None) -> Dict[str, Any]:
    fmt_, gold = inst["answer_format"], inst["gold"]
    cot, dqa = inst["samples"]["cot"], inst["samples"]["dqa"]
    norm = [s["normalized_answer"] for s in cot]
    maj = majority_answer(norm, fmt_)
    arm = "original" if inst["problem_type"] == "original" else inst["perturbation_type"]
    trap = naive is not None and arm == "breaking" and inst["problem_type"] == "perturbed"
    return {
        "id": inst["instance_id"], "format": fmt_, "arm": arm,
        "trap_majority": trap and answers_match(maj, naive, fmt_),  # the majority answer IS the trap answer
        "trap_samples": sum(answers_match(a, naive, fmt_) for a in norm) if trap else None,
        "unanimous": len(candidates_of(cot, fmt_)) < 2,  # the same grouping run_esv3.py gates on
        "correct": {"dqa_maj": bool(score_answer(majority_answer([s["normalized_answer"] for s in dqa], fmt_), gold, fmt_)["correct"]),
                    "cot_pass1": sum(s["correct"] for s in cot) / len(cot),
                    "cot_maj": bool(score_answer(maj, gold, fmt_)["correct"]),
                    "cot_pass4": any(s["correct"] for s in cot)},
        "tokens": {"cot4": sum(tokens(s) for s in cot)},
        "calls": {},
    }


def load(models, limit_pairs, v3_file: str = "esv3_records.json", min_pair=None, max_pair=None):
    per_model: Dict[str, List[Dict[str, Any]]] = {}
    naive = naive_answers(json.loads(common.FROZEN_DATASET.read_text(encoding="utf-8")) if common.FROZEN_DATASET.exists() else {})
    for model in models:
        folder = common.RESULTS_DIR / model_slug(model)
        if not (folder / "records.json").exists():
            continue
        base = json.loads((folder / "records.json").read_text(encoding="utf-8"))["instances"]
        v3 = {}
        if (folder / v3_file).exists():
            v3 = {i: redecide(r) for i, r in json.loads((folder / v3_file).read_text(encoding="utf-8"))["instances"].items()
                  if r.get("version") == V3_VERSION}
        complete = [i for i, r in base.items() if all(s is not None for s in r["samples"]["cot"]) and all(s is not None for s in r["samples"]["dqa"])]
        ids = sorted(v3) if v3 else [i for i in complete if limit_pairs is None or int(base[i]["pair_id"]) < limit_pairs]
        ids = [i for i in ids if not common.is_disputed(i)]
        if min_pair is not None:
            ids = [i for i in ids if int(base[i]["pair_id"]) >= min_pair]
        if max_pair is not None:
            ids = [i for i in ids if int(base[i]["pair_id"]) < max_pair]
        rows = []
        for i in ids:
            row = base_row(base[i], naive.get(base[i]["pair_id"]))
            if i in v3:
                r = v3[i]
                row["v3"] = r
                row["correct"].update(esv3=bool(r["methods"]["esv3"]["final_correct"]), usc=bool(r["methods"]["usc"]["final_correct"]))
                row["tokens"].update(esv3=tokens(r["methods"]["esv3"]["call"]), usc=tokens(r["methods"]["usc"]["call"]))
                row["calls"].update(esv3=int(r["methods"]["esv3"]["call"] is not None), usc=int(r["methods"]["usc"]["call"] is not None))
            rows.append(row)
        if rows:
            per_model[model] = rows
    return per_model


def stage1(rows: List[Dict[str, Any]]) -> List[List[Any]]:
    n = len(rows)
    una = [r for r in rows if r["unanimous"]]
    dis = [r for r in rows if not r["unanimous"]]
    return [[n, pct(len(una), n), pct(sum(not r["correct"]["cot_maj"] for r in una), len(una)), pct(len(dis), n),
             sum(r["correct"]["cot_maj"] for r in dis), sum(r["correct"]["cot_pass4"] and not r["correct"]["cot_maj"] for r in dis),
             sum(not r["correct"]["cot_pass4"] for r in dis), rate(sum(r["correct"]["cot_maj"] for r in rows), n),
             rate(sum(r["correct"]["cot_pass4"] for r in rows), n)]]


def failure_kinds(rows: List[Dict[str, Any]]) -> List[Any]:
    """What the wrong CoT maj@4 answers are: on untouched originals, on preserving P', or on breaking P' (the trap
    answer the original strategy yields, or some other wrong answer). ESV's premise needs the trap column to be large."""
    wrong = [r for r in rows if not r["correct"]["cot_maj"]]
    kinds = Counter("breaking: TRAP answer" if r["trap_majority"] else f"{r['arm']}" + (": other wrong answer" if r["arm"] == "breaking" else "")
                    for r in wrong)
    breaking = [r for r in rows if r["arm"] == "breaking" and r["trap_samples"] is not None]
    trap_samples = sum(r["trap_samples"] for r in breaking)
    return [len(wrong)] + [pct(kinds[k], len(wrong)) for k in ("original", "preserving", "breaking: TRAP answer", "breaking: other wrong answer")] \
        + [rate(sum(r["correct"]["cot_maj"] for r in [x for x in rows if x["arm"] == "original"]), sum(r["arm"] == "original" for r in rows)),
           rate(sum(r["correct"]["cot_maj"] for r in breaking), len(breaking)), pct(trap_samples, 4 * len(breaking))]


FAILURE_HEADER = ["wrong CoT maj@4 answers", "on originals", "on preserving P'", "on breaking P': the TRAP answer",
                  "on breaking P': other wrong", "CoT maj@4 on originals", "CoT maj@4 on breaking P'", "CoT samples giving the trap answer"]


STAGE1_HEADER = ["instances", "unanimous (no call)", "unanimous but wrong (lost)", "disagreeing (ESV runs)",
                 "disagreeing, majority right (can break)", "disagreeing, fixable (majority wrong, a candidate right)",
                 "disagreeing, no candidate right", "CoT maj@4", "CoT pass@4 (ceiling)"]


def paired(rows, a, b):
    wins = sum(r["correct"][a] and not r["correct"][b] for r in rows)
    losses = sum(r["correct"][b] and not r["correct"][a] for r in rows)
    p = mcnemar_exact(wins, losses)
    return wins, losses, "-" if p is None else f"{p:.3f}"


def accuracy_table(rows: List[Dict[str, Any]]) -> str:
    out = []
    for key, label in LABELS.items():
        have = [r for r in rows if key in r["correct"]]
        if len(have) != len(rows):
            continue
        acc = rate(sum(r["correct"][key] for r in rows), len(rows))
        if key in ("cot_pass1", "cot_maj", "cot_pass4"):
            out.append([label, acc, "", "", "", ""])
            continue
        fixes, breaks, p = paired(rows, key, "cot_maj")
        tok = round(statistics.mean(r["tokens"][key] for r in rows)) if key in rows[0]["tokens"] else ""
        calls = f"{statistics.mean(r['calls'][key] for r in rows):.2f}" if key in rows[0]["calls"] else ""
        out.append([label, acc, fixes, breaks, f"{fixes - breaks:+d} (p={p})", f"{tok} / {calls}" if tok != "" else ""])
    return table(["Method", "accuracy", "fixes", "breaks", "net vs CoT maj@4", "extra tokens / calls per instance"], out)


def candidate_table(rows: List[Dict[str, Any]]) -> str:
    """Stages 2-3 as a detector, per candidate: is the right candidate called SOUND and the wrong ones FLAWED?"""
    out = []
    pairs = [(c, k) for r in rows if "v3" in r for c, k in zip(r["v3"]["candidates"], r["v3"]["methods"]["esv3"]["checks"])]
    for label, strict in (("strict (used)", True), ("verdict only", False)):
        right = [k for c, k in pairs if c["correct"]]
        wrong = [k for c, k in pairs if not c["correct"]]
        sr, sw = sum(is_sound(k, strict) for k in right), sum(is_sound(k, strict) for k in wrong)
        out.append([label, pct(sr, len(right)), pct(sw, len(wrong)), rate(sr, sr + sw)])
    return table(["SOUND rule", "right candidates called SOUND", "wrong candidates called SOUND", "precision of SOUND"], out)


def check_table(rows: List[Dict[str, Any]]) -> str:
    pairs = [(c, k) for r in rows if "v3" in r for c, k in zip(r["v3"]["candidates"], r["v3"]["methods"]["esv3"]["checks"])]
    out = []
    for label, sub in (("right candidates", [k for c, k in pairs if c["correct"]]), ("wrong candidates", [k for c, k in pairs if not c["correct"]])):
        out.append([label, len(sub), pct(sum(k["mismatches"] > 0 for k in sub), len(sub)), pct(sum(k["violations"] > 0 for k in sub), len(sub)),
                    pct(sum(k["verdict"] is None for k in sub), len(sub)),
                    ", ".join(f"{t}: {n}" for t, n in Counter(k["failure_type"] for k in sub if k["verdict"] == "FLAWED").most_common())])
    return table(["", "n", "some binding MISMATCH", "some condition VIOLATED", "verdict unreadable", "failure types when FLAWED"], out)


def decision_table(rows: List[Dict[str, Any]]) -> str:
    out = []
    for d in ("unanimous", "one_sound", "several_sound", "none_sound", "unreadable"):
        sub = [r for r in rows if "v3" in r and r["v3"]["methods"]["esv3"]["decision"] == d]
        if not sub:
            continue
        fixes, breaks, _ = paired(sub, "esv3", "cot_maj")
        out.append([d, len(sub), rate(sum(r["correct"]["cot_maj"] for r in sub), len(sub)), rate(sum(r["correct"]["esv3"] for r in sub), len(sub)),
                    fixes, breaks, sum(r["correct"]["cot_pass4"] for r in sub)])
    return table(["decision", "n", "CoT maj@4", "ESV v3", "fixes", "breaks", "a candidate is right"], out)


def selection_table(rows: List[Dict[str, Any]]) -> str:
    dis = [r for r in rows if not r["unanimous"] and "v3" in r]
    if not dis:
        return ""
    chance = sum(sum(c["correct"] for c in r["v3"]["candidates"]) / len(r["v3"]["candidates"]) for r in dis)
    lenient = 0
    for r in dis:
        e = r["v3"]
        _, idx = decide(e["candidates"], e["methods"]["esv3"]["checks"], strict=False)
        lenient += e["candidates"][idx]["correct"] if idx is not None else r["correct"]["cot_maj"]
    out = [["majority vote (CoT maj@4)", pct(sum(r["correct"]["cot_maj"] for r in dis), len(dis))],
           ["uniform random candidate (chance)", pct(round(chance, 1), len(dis))],
           ["Universal Self-Consistency", pct(sum(r["correct"]["usc"] for r in dis), len(dis))],
           ["ESV v3, strict SOUND (used)", pct(sum(r["correct"]["esv3"] for r in dis), len(dis))],
           ["ESV v3, verdict-only SOUND (secondary)", pct(lenient, len(dis))],
           ["oracle choice (CoT pass@4)", pct(sum(r["correct"]["cot_pass4"] for r in dis), len(dis))]]
    return table([f"Choice among the candidates ({len(dis)} disagreeing instances)", "right"], out)


def breakdown(rows: List[Dict[str, Any]], keys) -> str:
    out = []
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if sub:
            out.append([arm, len(sub)] + [rate(sum(r["correct"][k] for r in sub), len(sub)) for k in keys]
                       + [f"{paired(sub, 'esv3', 'cot_maj')[0]} / {paired(sub, 'esv3', 'cot_maj')[1]}" if "esv3" in keys else ""])
    return table(["arm", "n"] + [LABELS[k] for k in keys] + ["ESV v3 fixes / breaks"], out)


def go_no_go(rows: List[Dict[str, Any]]) -> str:
    n = len(rows)
    d_maj = 100 * (sum(r["correct"]["esv3"] for r in rows) - sum(r["correct"]["cot_maj"] for r in rows)) / n
    d_usc = 100 * (sum(r["correct"]["esv3"] for r in rows) - sum(r["correct"]["usc"] for r in rows)) / n
    ok = d_maj >= GO_MARGIN_PP and d_usc > 0
    return (f"**Go / no-go (fixed before the run): {'GO' if ok else 'NO-GO'}.** ESV v3 - CoT maj@4 = {d_maj:+.1f} pp "
            f"(needs >= +{GO_MARGIN_PP:g}); ESV v3 - USC = {d_usc:+.1f} pp (needs > 0).")


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="ESV v3 stage-by-stage report.")
    common.add_benchmark_arg(parser)
    parser.add_argument("--models", choices=list(MODEL_SETS), default="large")
    parser.add_argument("--limit-pairs", type=int, default=None, help="stage 1 only, before run_esv3.py has run")
    parser.add_argument("--verifier-model", default=None,
                        help="Read the esv3_records.<verifier>.json written by run_esv3.py --verifier-model")
    parser.add_argument("--min-pair", type=int, default=None,
                        help="Score only pairs >= this id (e.g. 60 = the held-out split; 0-59 is the dev set)")
    parser.add_argument("--max-pair", type=int, default=None,
                        help="Score only pairs < this id (e.g. --max-pair 60 = the dev split, pairs 0-59)")
    args = parser.parse_args(argv)
    common.select_benchmark(args.benchmark)
    vslug = model_slug(args.verifier_model) if args.verifier_model else None
    per_model = load(MODEL_SETS[args.models], args.limit_pairs, f"esv3_records.{vslug}.json" if vslug else "esv3_records.json",
                     args.min_pair, args.max_pair)
    if not per_model:
        print(f"No records.json for the {args.models} models.", file=sys.stderr)
        return 2
    pooled = [r for rows in per_model.values() for r in rows]
    disputed = common.DISPUTED_INSTANCES.get(common.BENCHMARK, frozenset())
    has_v3 = all("v3" in r for r in pooled)

    verifier_note = f", verifier {args.verifier_model}" if args.verifier_model else ""
    if args.min_pair is not None:
        verifier_note += f", HELD-OUT pairs >= {args.min_pair}"
    if args.max_pair is not None:
        verifier_note += f", DEV pairs < {args.max_pair}"
    out = [f"# ESV v3, stage by stage ({common.benchmark_info()['label']}, {args.models} models{verifier_note})", "",
           f"Version `{V3_VERSION}`, decision rule `{DECISION_RULE}`. Gold is used only to score. Design and motivation: "
           "`esv_bottleneck_analysis.md`.", "",
           "| Stage | Status |", "|---|---|",
           "| 1. Disagreement gate | built, measured below |",
           f"| 2-3. Candidate verification + choice | {'run, measured below' if has_v3 else 'built, not run yet'} |",
           "| 4. Diagnosis-driven repair | not built (decision `none_sound` keeps the initial answer) |", "",
           *([f"Excluded from every score: {len(disputed)} instances whose published gold is wrong or ambiguous "
              f"(`common.DISPUTED_INSTANCES`): {', '.join(sorted(disputed))}.", ""] if disputed else []),
           "## Stage 1: disagreement gate (no calls)", "",
           "Unanimous = fewer than 2 distinct readable answers among the 4 CoT samples: ESV keeps the answer. Only the "
           "disagreeing instances can change, and only the fixable ones can gain.", "",
           table(["model"] + STAGE1_HEADER, [[m] + stage1(rows)[0] for m, rows in per_model.items()] + [["**pooled**"] + stage1(pooled)[0]]), "",
           "### What the wrong answers are", "",
           "ESV targets the *trap*: the answer the original strategy S yields when applied blindly to a strategy-breaking P'. "
           "The premise holds only where that column is a large share of the failures.", "",
           table(["model"] + FAILURE_HEADER, [[m] + failure_kinds(rows) for m, rows in per_model.items()] + [["**pooled**"] + failure_kinds(pooled)]), ""]
    if has_v3:
        out += ["## Stages 2-3: candidate verification and choice", "", go_no_go(pooled), "",
                "### All instances", ""]
        for model, rows in per_model.items():
            out += [f"**{model}** ({len(rows)} instances)", "", accuracy_table(rows), ""]
        out += [f"**Pooled** ({len(pooled)} instances)", "", accuracy_table(pooled), "",
                "ESV v3 vs USC (ESV-only / USC-only right, McNemar p): " + "{} / {} (p={})".format(*paired(pooled, "esv3", "usc")), "",
                "### The choice, on disagreeing instances only (pooled)", "", selection_table(pooled), "",
                "### Stages 2-3 as a detector, per candidate (pooled)", "", candidate_table(pooled), "", check_table(pooled), "",
                "### Decisions (pooled)", "", decision_table(pooled), "",
                "`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.", "",
                "### By arm (pooled)", "", breakdown(pooled, ("cot_maj", "usc", "esv3")), ""]
    name = "esv3_report.md" if args.models == "large" else f"esv3_report_{args.models}.md"
    if vslug:
        name = name[:-3] + f".{vslug}.md"
    if args.min_pair is not None:
        name = name[:-3] + f".from{args.min_pair}.md"
    if args.max_pair is not None:
        name = name[:-3] + f".upto{args.max_pair}.md"
    path = common.RESULTS_DIR / "analysis" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nSaved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
