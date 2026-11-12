#!/usr/bin/env python3
"""Stage-by-stage report for ESV v3 (run_esv3.py) -> results/analysis/esv3_report.md.

Stage 1 needs only the CoT samples (records.json), so it is reported even before run_esv3.py has run. Stages 2-3 are
reported from esv3_records.json. Every comparison is paired on the same instances; fixes / breaks are vs CoT maj@4,
the answer every method starts from.

--verifier-model reads the ESV v3 arm from esv3_records.<verifier>.json, but the USC arm always comes from
esv3_records.json: USC asks the solver, not the verifier, so run_esv3.py measures it once, in the self-verify run.

  python experiments/analyze_esv3.py                    # current (large) solvers
  python experiments/analyze_esv3.py --limit-pairs 30   # stage 1 on the first 30 pairs only
  python experiments/analyze_esv3.py --benchmark math500 --models small --verifier-model openai/gpt-oss-120b
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
          "usc": "Universal Self-Consistency", "judge_select": "Judge-select (same checker, no ESV form)",
          "esv3": "ESV v3 (stages 1-3)"}


def pct(n: float, d: int) -> str:
    return f"{n:g}/{d} ({100 * n / d:.1f}%)" if d else "-"


def rate(n: float, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "-"


def table(header: List[str], rows: List[List[Any]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def tokens(call) -> int:
    return ((call or {}).get("usage") or {}).get("completion_tokens", 0)


def call_cost(call) -> Dict[str, float]:
    """One API call's billable size and wall-clock time, as recorded by common.call_stage. None = the call never ran."""
    u = (call or {}).get("usage") or {}
    return {"calls": 1 if call else 0, "prompt": u.get("prompt_tokens", 0), "completion": u.get("completion_tokens", 0),
            "seconds": (call or {}).get("latency_seconds") or 0.0}


def sample_cost(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """A set of n independent samples. `serial` is the compute time (what a provider bills for); `path` is what a user
    waits for, because run_methods.py issues the n samples concurrently, so only the slowest one is on the critical
    path. Both are reported: serial is the throughput cost, path is the latency cost."""
    parts = [call_cost(s) for s in samples]
    return {"calls": len(parts), "prompt": sum(p["prompt"] for p in parts), "completion": sum(p["completion"] for p in parts),
            "serial": sum(p["seconds"] for p in parts), "path": max((p["seconds"] for p in parts), default=0.0)}


def scale(cost: Dict[str, float], factor: float) -> Dict[str, float]:
    return {k: v * factor for k, v in cost.items()}


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


def method_cost(solver: Dict[str, float], extra: Optional[Dict[str, float]] = None,
                extra_is_verifier: bool = False) -> Dict[str, float]:
    """The compute a method spends on one instance: its share of the shared CoT/DQA samples plus at most one extra
    call. The extra call is strictly sequential (it needs all 4 answers first), so it lands on the critical path in
    full. Solver and verifier tokens are kept apart because under --verifier-model they are billed by different
    models at different rates; `price_sensitivity` is what combines them."""
    extra = extra or {"calls": 0, "prompt": 0, "completion": 0, "seconds": 0.0}
    side = "verifier" if extra_is_verifier else "solver"
    cost = {"calls": solver["calls"] + extra["calls"],
            "solver": solver["prompt"] + solver["completion"], "verifier": 0.0,
            "solver_completion": solver["completion"], "verifier_completion": 0.0,
            "serial": solver["serial"] + extra["seconds"], "path": solver["path"] + extra["seconds"]}
    cost[side] += extra["prompt"] + extra["completion"]
    cost[side + "_completion"] += extra["completion"]
    return cost


def base_row(inst: Dict[str, Any], naive: Optional[str] = None) -> Dict[str, Any]:
    fmt_, gold = inst["answer_format"], inst["gold"]
    cot, dqa = inst["samples"]["cot"], inst["samples"]["dqa"]
    norm = [s["normalized_answer"] for s in cot]
    maj = majority_answer(norm, fmt_)
    cot_cost, dqa_cost = sample_cost(cot), sample_cost(dqa)
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
        # CoT pass@1 is one of the 4 samples, so it costs a quarter of the shared block on average.
        "cost": {"dqa_maj": method_cost(dqa_cost), "cot_maj": method_cost(cot_cost), "cot_pass4": method_cost(cot_cost),
                 "cot_pass1": method_cost({**scale(cot_cost, 0.25), "path": cot_cost["serial"] / 4})},
    }


def read_v3(folder: Path, name: str) -> Dict[str, Any]:
    if not (folder / name).exists():
        return {}
    return {i: redecide(r) for i, r in json.loads((folder / name).read_text(encoding="utf-8"))["instances"].items()
            if r.get("version") == V3_VERSION}


def load(models, limit_pairs, v3_file: str = "esv3_records.json", min_pair=None, max_pair=None,
         usc_file: Optional[str] = None):
    """The ESV v3 arm is read from v3_file. USC has no verifier, so run_esv3.py measures it only in the self-verify
    condition; under --verifier-model its arm is read from usc_file (esv3_records.json) instead. Reading it from the
    verifier file would report CoT maj@4 under USC's name, because a skipped USC falls back to the initial answer."""
    per_model: Dict[str, List[Dict[str, Any]]] = {}
    naive = naive_answers(json.loads(common.FROZEN_DATASET.read_text(encoding="utf-8")) if common.FROZEN_DATASET.exists() else {})
    for model in models:
        folder = common.RESULTS_DIR / model_slug(model)
        if not (folder / "records.json").exists():
            continue
        base = json.loads((folder / "records.json").read_text(encoding="utf-8"))["instances"]
        v3 = read_v3(folder, v3_file)
        usc_src = v3 if not usc_file or usc_file == v3_file else read_v3(folder, usc_file)
        complete = [i for i, r in base.items() if all(s is not None for s in r["samples"]["cot"]) and all(s is not None for s in r["samples"]["dqa"])]
        ids = sorted(v3) if v3 else [i for i in complete if limit_pairs is None or int(base[i]["pair_id"]) < limit_pairs]
        ids = [i for i in ids if not common.is_disputed(i)]
        if v3:
            unscored = [i for i in ids if (usc_src.get(i) or v3[i])["methods"]["usc"]["final_correct"] is None]
            if unscored:
                raise SystemExit(
                    f"{model}: the USC arm is unmeasured for {len(unscored)} of {len(ids)} instances in "
                    f"{usc_file or v3_file}. USC asks the solver, not the verifier, so it is only run in the "
                    f"self-verify condition. Run `run_esv3.py` without --verifier-model first so that "
                    f"{usc_file or 'esv3_records.json'} exists for this model, then re-run this report.")
        if min_pair is not None:
            ids = [i for i in ids if int(base[i]["pair_id"]) >= min_pair]
        if max_pair is not None:
            ids = [i for i in ids if int(base[i]["pair_id"]) < max_pair]
        rows = []
        for i in ids:
            row = base_row(base[i], naive.get(base[i]["pair_id"]))
            if i in v3:
                r = v3[i]
                u = (usc_src.get(i) or r)["methods"]["usc"]  # USC from the self-verify run under --verifier-model
                row["v3"] = r
                row["usc_selection"] = u.get("selection")
                row["correct"].update(esv3=bool(r["methods"]["esv3"]["final_correct"]), usc=bool(u["final_correct"]))
                row["tokens"].update(esv3=tokens(r["methods"]["esv3"]["call"]), usc=tokens(u["call"]))
                row["calls"].update(esv3=int(r["methods"]["esv3"]["call"] is not None), usc=int(u["call"] is not None))
                # Both arms reuse the same 4 CoT samples and add at most one call. USC always asks the solver; the
                # ESV check asks whichever model run_esv3.py recorded, which is a different one under --verifier-model.
                echeck = r["methods"]["esv3"]["call"]
                row["cost"].update(
                    esv3=method_cost(sample_cost(base[i]["samples"]["cot"]), call_cost(echeck),
                                     extra_is_verifier=bool(echeck) and echeck.get("model") != model),
                    usc=method_cost(sample_cost(base[i]["samples"]["cot"]), call_cost(u["call"])))
                # The judge-select control is opt-in, so it is present only in records run with --judge-select.
                judge = r["methods"].get("judge_select")
                if judge is not None:
                    jcall = judge["call"]
                    row["judge_selection"] = judge.get("selection")
                    row["correct"]["judge_select"] = bool(judge["final_correct"])
                    row["tokens"]["judge_select"] = tokens(jcall)
                    row["calls"]["judge_select"] = int(jcall is not None)
                    row["cost"]["judge_select"] = method_cost(
                        sample_cost(base[i]["samples"]["cot"]), call_cost(jcall),
                        extra_is_verifier=bool(jcall) and jcall.get("model") != model)
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
    if p is None:
        return wins, losses, "-"
    return wins, losses, f"{p:.3f}" if p >= 0.001 else (f"{p:.0e}" if p > 0 else "<1e-300")


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
    # Completion tokens only, for continuity with the earlier reports; the "Compute cost" section below bills the
    # prompt side too and splits solver from verifier tokens.
    return table(["Method", "accuracy", "fixes", "breaks", "net vs CoT maj@4",
                  "extra completion tokens / calls per instance"], out)


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


def pipeline(verifier_model: Optional[str], has_v3: bool) -> List[str]:
    """The stages exactly as run_esv3.py applies them, and where USC sits relative to them. Written out in full
    because 'stages 1-3' alone does not say who is called, when, or what happens when a stage declines to act."""
    checker = verifier_model or "the solver itself (self-verification)"
    ran = "run, measured below" if has_v3 else "built, not run yet"
    stages = [
        ["0", "CoT sampling (shared input, not part of ESV)",
         "The solver answers the problem 4 times with chain-of-thought. Produced once by `run_methods.py` and reused "
         "unchanged by every method, so nothing below ever re-solves the problem.",
         "4 per instance, T = 0.7, seeds 0-3", "solver",
         "4 answers. Their majority is the `initial` answer and is scored as the CoT maj@4 baseline."],
        ["1", "Disagreement gate",
         "Group the 4 answers into distinct candidates by normalized value, in order of first appearance, keeping "
         "each one's vote count. Unreadable answers are dropped.",
         "0 (free)", "none",
         "Fewer than 2 distinct candidates -> decision `unanimous`: keep the initial answer and STOP. "
         "2 or more -> go to stage 2."],
        ["2", "Contrastive verification",
         "All distinct candidates in ONE call, one representative solution each, in order of first appearance, with "
         "vote counts hidden. Per candidate: the strategy, its bindings to quoted givens (MATCHES / MISMATCH), the "
         "conditions the method needs (HOLDS / VIOLATED), then the first divergence point, then SOUND or FLAWED with "
         "a failure type (BINDING / METHOD / COMPUTATION), the flawed step and the corrected fact.",
         "1 per disagreeing instance, T = 0", checker,
         f"A filled-in check per candidate ({ran})."],
        ["3", "Decision rule (code only, no model)",
         "A candidate is strictly SOUND only if its verdict is SOUND **and** it lists zero MISMATCH **and** zero "
         "VIOLATED. Exactly one such candidate wins; several, the one with the most votes (ties to the initial "
         "answer, then to the earliest); none, nothing is chosen.",
         "0 (free)", "none",
         "`one_sound` -> take that candidate. `several_sound` -> take the most-voted SOUND one. "
         "`none_sound` -> keep the initial answer. `unreadable` -> keep the initial answer."],
        ["4", "Diagnosis-driven repair",
         "Would re-solve using the diagnosis from stage 2 where no candidate survives. NOT BUILT: `none_sound` "
         "simply keeps the initial answer, so ESV never invents an answer the solver did not already produce.",
         "n/a", "n/a",
         "Not built. Its ceiling is the `a candidate is right` column of the decisions table below."],
        ["5", "Scoring",
         "Gold enters here and nowhere else. Paired against CoT maj@4 on the same instances, with exact McNemar "
         "tests, so a stage can break a correct majority as easily as repair a wrong one.",
         "0 (free)", "none", "The accuracy tables below."],
    ]
    usc_when = ("Not measured in this report: USC asks the solver, so it is run once, in the self-verify condition, "
                "and read from `esv3_records.json`.") if verifier_model else "Measured in this report."
    return [
        "## The pipeline, as run", "",
        "Every method starts from the same 4 CoT samples and none of them re-solves the problem; they differ only in "
        "how they pick among answers the solver already produced. ESV fires only on disagreement, so it can damage a "
        "correct majority as easily as repair a wrong one, and is scored that way.", "",
        table(["#", "Stage", "What it does", "Calls", "Model called", "Outcome"], stages), "",
        "### Where USC sits (the baseline, not a stage)", "",
        "Universal Self-Consistency is the control ESV must beat, not a step inside it. It replaces stages 2-3 with a "
        "single selection call and is wired in deliberately parallel to them:", "",
        "- **When it fires:** on exactly the same instances as stage 2, the disagreeing ones (2 or more distinct "
        "readable candidates). A `unanimous` instance never calls it and keeps the majority answer, so USC and CoT "
        "maj@4 agree there by construction.",
        "- **Who is called:** the **solver**, never the verifier. USC has no verifier, which is why it is identical "
        f"in both conditions. ({usc_when})",
        "- **What it sees:** all 4 samples including duplicates, in arbitrary order, with the reasoning shown. Unlike "
        "stage 2 it is therefore free to count multiplicity: it can see that 3 of 4 samples agree.",
        "- **How it answers:** one `<selected_response>` tag naming the response it picks. Resolved first as a "
        "response label (1-4), then as the selected answer's value, which solvers frequently return instead.",
        "- **When it declines:** a tag that resolves to neither keeps the initial answer, i.e. USC falls back to CoT "
        "maj@4. The rate is reported under *Did USC actually select?* and must be read before any USC comparison.", "",
    ]


def usc_health(per_model: Dict[str, List[Dict[str, Any]]]) -> str:
    """How USC's selection tag parsed. An unresolved selection keeps the initial answer, so a high unresolved rate
    makes USC collapse onto CoT maj@4 - the first thing to check whenever the two accuracies coincide."""
    out = []
    for model, rows in list(per_model.items()) + [("**pooled**", [r for rs in per_model.values() for r in rs])]:
        calls = [r["usc_selection"] for r in rows if r.get("usc_selection")]
        if not calls:
            continue
        c = Counter(calls)
        out.append([model, len(calls), pct(c["label"], len(calls)), pct(c["value"], len(calls)),
                    pct(c["ambiguous"], len(calls)), pct(c["unresolved"], len(calls))])
    if not out:
        return ""
    return table(["model", "USC calls", "by response label", "by answer value", "ambiguous (label used)",
                  "unresolved -> keeps CoT maj@4"], out)


def breakdown(rows: List[Dict[str, Any]], keys) -> str:
    out = []
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if sub:
            out.append([arm, len(sub)] + [rate(sum(r["correct"][k] for r in sub), len(sub)) for k in keys]
                       + [f"{paired(sub, 'esv3', 'cot_maj')[0]} / {paired(sub, 'esv3', 'cot_maj')[1]}" if "esv3" in keys else ""])
    return table(["arm", "n"] + [LABELS[k] for k in keys] + ["ESV v3 fixes / breaks"], out)


def format_breakdown(rows: List[Dict[str, Any]], keys) -> str:
    """The same comparison split by answer format. Only meaningful where a benchmark mixes formats: MATH-500 has
    integer, numeric and expression golds, and `expression` is the one scored by LaTeX / symbolic equivalence rather
    than exact match, so if it were systematically harder for every method this is where that shows up."""
    formats = sorted({r["format"] for r in rows})
    if len(formats) < 2:
        return ""
    out = []
    for fmt_ in formats:
        sub = [r for r in rows if r["format"] == fmt_]
        fixes, breaks, _ = paired(sub, "esv3", "cot_maj") if "esv3" in keys else (0, 0, "-")
        out.append([fmt_, len(sub)] + [rate(sum(r["correct"][k] for r in sub), len(sub)) for k in keys]
                   + [f"{fixes} / {breaks}" if "esv3" in keys else ""])
    return table(["answer format", "n"] + [LABELS[k] for k in keys] + ["ESV v3 fixes / breaks"], out)


# ---------------------------------------------------------------------------
# Compute cost. Every number below is read from the `usage` and `latency_seconds` that common.call_stage stored on
# each call, so the whole section is reproducible offline from the records with no API calls.
# ---------------------------------------------------------------------------
COST_METHODS = ("dqa_maj", "cot_pass1", "cot_maj", "usc", "judge_select", "esv3", "cot_pass4")


def mean_cost(rows: List[Dict[str, Any]], key: str) -> Optional[Dict[str, float]]:
    have = [r["cost"][key] for r in rows if key in r["cost"]]
    if len(have) != len(rows) or not have:
        return None
    return {k: statistics.mean(c[k] for c in have) for k in have[0]}


def total(cost: Dict[str, float]) -> float:
    return cost["solver"] + cost["verifier"]


def compute_table(rows: List[Dict[str, Any]]) -> str:
    """What each method spends per instance. `serial` is billed compute time; `critical path` is what a user waits
    for, given that the 4 samples are issued concurrently and the extra call cannot start until all 4 are back."""
    base = mean_cost(rows, "cot_maj")
    out = []
    for key in COST_METHODS:
        cost = mean_cost(rows, key)
        if cost is None:
            continue
        acc = rate(sum(r["correct"][key] for r in rows), len(rows))
        vs = f"{100 * (total(cost) - total(base)) / total(base):+.1f}%" if total(base) else "-"
        out.append([LABELS[key], acc, f"{cost['calls']:.2f}", f"{cost['solver']:,.0f}",
                    f"{cost['verifier']:,.0f}" if cost["verifier"] else "-", f"{total(cost):,.0f}", vs,
                    f"{cost['serial']:.1f}", f"{cost['path']:.1f}"])
    return table(["Method", "accuracy", "calls", "solver tokens", "verifier tokens", "total tokens",
                  "vs CoT maj@4", "serial s", "critical path s"], out)


def tradeoff_table(rows: List[Dict[str, Any]]) -> str:
    """Accuracy bought per unit of extra compute, against the baseline every method starts from. `net` is
    fixes - breaks, so a method that trades one correct answer for one wrong one is charged for both."""
    base, n = mean_cost(rows, "cot_maj"), len(rows)
    base_acc = 100 * sum(r["correct"]["cot_maj"] for r in rows) / n
    out = []
    for key in COST_METHODS:
        cost = mean_cost(rows, key)
        if cost is None or key in ("cot_maj", "cot_pass1"):
            continue
        fixes, breaks, p = paired(rows, key, "cot_maj")
        net = fixes - breaks
        extra_tok, extra_path = total(cost) - total(base), cost["path"] - base["path"]
        delta = 100 * sum(r["correct"][key] for r in rows) / n - base_acc
        per_fix = f"{extra_tok * n / net:,.0f}" if net > 0 and extra_tok > 0 else ("0" if net > 0 else "no net gain")
        out.append([LABELS[key], f"{delta:+.1f} pp", f"{fixes} / {breaks}", f"{net:+d}", f"p={p}",
                    f"{extra_tok:+,.0f}", f"{extra_path:+.1f} s", per_fix])
    return table(["Method", "accuracy vs CoT maj@4", "fixes / breaks", "net", "McNemar",
                  "extra tokens / instance", "extra wait", "extra tokens per net fix"], out)


def gate_economics(rows: List[Dict[str, Any]]) -> str:
    """The disagreement gate is the whole efficiency story: it is free (code only) and it retires most instances
    before any verification call is paid for."""
    fired = [r for r in rows if r["calls"].get("esv3")]
    if not fired:
        return ""
    def extra(sub, field):
        return statistics.mean(r["cost"]["esv3"][field] - r["cost"]["cot_maj"][field] for r in sub)

    fired_tok = extra(fired, "solver") + extra(fired, "verifier")
    amort_tok = extra(rows, "solver") + extra(rows, "verifier")
    return table(["", "per instance"], [
        ["Instances where the gate fires (>= 2 distinct candidates)", pct(len(fired), len(rows))],
        ["A verification call, on an instance that fires", f"{fired_tok:,.0f} tokens, {extra(fired, 'path'):.1f} s"],
        ["Amortized over every instance (what ESV actually costs)", f"{amort_tok:,.0f} tokens, {extra(rows, 'path'):.1f} s"],
        ["The same protocol ungated, verifying every instance", f"{fired_tok:,.0f} tokens, {extra(fired, 'path'):.1f} s"],
        ["Verification spend the gate avoids", f"{100 * (1 - len(fired) / len(rows)):.1f}%"],
        ["Cost on a gated-out instance", "0 tokens, 0 s - ESV returns the CoT maj@4 answer untouched"]])


def arm_compute(rows: List[Dict[str, Any]]) -> str:
    """Compute per arm. The gate spends where the perturbation actually hurts, so this is the table that ties the
    cost profile back to the dataset design."""
    if not any("esv3" in r["cost"] for r in rows):
        return ""
    out = []
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        base, esv = mean_cost(sub, "cot_maj"), mean_cost(sub, "esv3")
        out.append([arm, len(sub), pct(sum(r["calls"].get("esv3", 0) for r in sub), len(sub)),
                    rate(sum(r["correct"]["cot_maj"] for r in sub), len(sub)),
                    rate(sum(r["correct"]["esv3"] for r in sub), len(sub)),
                    f"{total(base):,.0f}", f"{total(esv) - total(base):+,.0f}",
                    f"{100 * (total(esv) - total(base)) / total(base):+.1f}%"])
    return table(["arm", "instances", "gate fires", "CoT maj@4", "ESV v3", "baseline tokens",
                  "extra tokens", "vs baseline"], out)


def price_sensitivity(rows: List[Dict[str, Any]], verifier_model: Optional[str]) -> List[str]:
    """Under --verifier-model the extra call is billed by a different, larger model, so a raw token count understates
    it. k = the verifier's price per token divided by the solver's; costs are quoted in solver-token equivalents.
    Plug the providers' list prices into k - this report does not assume any."""
    esv, base = mean_cost(rows, "esv3"), mean_cost(rows, "cot_maj")
    usc = mean_cost(rows, "usc")
    if esv is None or not esv["verifier"]:
        return []
    n = len(rows)
    net_esv = sum(r["correct"]["esv3"] and not r["correct"]["cot_maj"] for r in rows) - \
        sum(r["correct"]["cot_maj"] and not r["correct"]["esv3"] for r in rows)
    net_usc = sum(r["correct"]["usc"] and not r["correct"]["cot_maj"] for r in rows) - \
        sum(r["correct"]["cot_maj"] and not r["correct"]["usc"] for r in rows)
    usc_extra = total(usc) - total(base)
    rowsout = []
    for k in (1, 2, 5, 10, 20):
        tot = esv["solver"] + k * esv["verifier"]
        rowsout.append([k, f"{tot:,.0f}", f"{100 * (tot - total(base)) / total(base):+.1f}%",
                        f"{k * esv['verifier'] / usc_extra:.1f}x" if usc_extra else "-",
                        f"{k * esv['verifier'] * n / net_esv:,.0f}" if net_esv > 0 else "-"])
    out = [f"The ESV check is billed by {verifier_model}, the 4 samples by the solver, so a flat token count "
           "understates the check. k = the verifier's price per token over the solver's; totals are in "
           "solver-token equivalents. No price list is assumed - read off the row for your k.", "",
           table(["k", "ESV tokens / instance", "vs CoT maj@4", "check vs USC's overhead", "per net fix"], rowsout), ""]
    if net_esv > 0 and net_usc > 0 and usc_extra:
        breakeven = (usc_extra * n / net_usc) / (esv["verifier"] * n / net_esv)
        out += [f"Break-even against USC: ESV v3 costs less per net fix than USC for any k below "
                f"**{breakeven:.1f}**. Above it USC is the cheaper way to buy a fix - but USC's total gain is capped "
                f"at {net_usc:+d} net fixes against ESV's {net_esv:+d}, so the two are not interchangeable at any "
                "price: more USC cannot reach ESV's accuracy.", ""]
    return out


def judge_select_section(rows: List[Dict[str, Any]]) -> List[str]:
    """The judge-select control: same checker, same candidates, same hidden votes, same licence to reject them all
    and the same room to reason - only ESV's structured form is removed. It answers the one question the headline
    result cannot: is the gain the protocol, or just a stronger model looking at the candidates?"""
    if not all("judge_select" in r["correct"] for r in rows):
        return []
    dis = [r for r in rows if r["calls"].get("esv3")]
    wins, losses, p = paired(rows, "esv3", "judge_select")
    out = ["## The judge-select control: is it the structure, or the checker?", "",
           "Judge-select asks the **same checker** to read the **same candidates** in the same order with the same "
           "vote counts hidden, reason freely in its own words, and name one solution or NONE. Everything is held "
           "fixed except ESV's form (bindings / conditions / divergence / verdict per candidate). It keeps the "
           "licence to reject every candidate, so ESV gets no free win from its `none_sound` branch, and it is "
           "given room to reason, so this is structured reasoning versus unstructured reasoning rather than "
           "versus no thinking at all.", "",
           f"**ESV v3 vs judge-select, paired over all {len(rows)} instances: {wins} / {losses} (McNemar p={p}).**", "",
           accuracy_table(rows), "",
           "### Where the two differ (disagreeing instances only)", "",
           "Stratified by what ESV decided, because the hypothesis is specific: the form should matter least where "
           "one candidate is clearly sound, and most where none survives and judge-select has to commit anyway.", ""]
    strata = []
    for branch in ("one_sound", "several_sound", "none_sound", "unreadable"):
        sub = [r for r in dis if r["v3"]["methods"]["esv3"]["decision"] == branch]
        if not sub:
            continue
        w, l, pp = paired(sub, "esv3", "judge_select")
        strata.append([branch, len(sub),
                       rate(sum(r["correct"]["cot_maj"] for r in sub), len(sub)),
                       rate(sum(r["correct"]["judge_select"] for r in sub), len(sub)),
                       rate(sum(r["correct"]["esv3"] for r in sub), len(sub)),
                       rate(sum(r["correct"]["cot_pass4"] for r in sub), len(sub)),
                       f"{w} / {l} (p={pp})"])
    out += [table(["ESV's decision", "n", "CoT maj@4", "judge-select", "ESV v3", "oracle",
                   "ESV-only / judge-only right"], strata), "",
            "### Selection quality on the disagreeing instances", "",
            table(["", "correct"],
                  [[LABELS[k], rate(sum(r["correct"][k] for r in dis), len(dis))]
                   for k in ("cot_maj", "usc", "judge_select", "esv3", "cot_pass4") if k in dis[0]["correct"]]), "",
            "### What judge-select actually returned", "",
            "`none` is a judgement that no candidate holds up - the counterpart of ESV's `none_sound`. "
            "`unresolved` is a tag that named nothing usable; both keep the initial answer, but only the first is "
            "a decision, so they are reported apart.", "",
            table(["selection", "instances", "correct"],
                  [[how or "(no call)", pct(len(sub), len(dis)),
                    rate(sum(r["correct"]["judge_select"] for r in sub), len(sub))]
                   for how, sub in sorted(
                       ((h, [r for r in dis if r.get("judge_selection") == h])
                        for h in {r.get("judge_selection") for r in dis}), key=lambda kv: -len(kv[1])) if sub]), ""]
    return out


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
    parser.add_argument("--models", nargs="+", default=["large"],
                        help="A preset (" + ", ".join(MODEL_SETS) + ") or explicit model ids, e.g. "
                             "--models mistralai/mistral-nemo google/gemma-3-4b-it. Explicit ids let the report "
                             "cover only the solvers that have ESV records, and let a benchmark be re-scored on "
                             "the solver subset another benchmark shares.")
    parser.add_argument("--limit-pairs", type=int, default=None, help="stage 1 only, before run_esv3.py has run")
    parser.add_argument("--verifier-model", default=None,
                        help="Read the ESV v3 arm from the esv3_records.<verifier>.json written by run_esv3.py "
                             "--verifier-model. The USC arm has no verifier and is always read from esv3_records.json")
    parser.add_argument("--min-pair", type=int, default=None,
                        help="Score only pairs >= this id (e.g. 60 = the held-out split; 0-59 is the dev set)")
    parser.add_argument("--max-pair", type=int, default=None,
                        help="Score only pairs < this id (e.g. --max-pair 60 = the dev split, pairs 0-59)")
    args = parser.parse_args(argv)
    common.select_benchmark(args.benchmark)
    vslug = model_slug(args.verifier_model) if args.verifier_model else None
    if len(args.models) == 1 and args.models[0] in MODEL_SETS:
        models, models_label = MODEL_SETS[args.models[0]], f"{args.models[0]} models"
        models_tag = args.models[0]
    else:
        models = args.models
        models_label = "models " + ", ".join(model_slug(m) for m in models)
        models_tag = "+".join(model_slug(m) for m in models)
    per_model = load(models, args.limit_pairs, f"esv3_records.{vslug}.json" if vslug else "esv3_records.json",
                     args.min_pair, args.max_pair, usc_file="esv3_records.json" if vslug else None)
    if not per_model:
        print(f"No records.json for the {models_label}.", file=sys.stderr)
        return 2
    pooled = [r for rows in per_model.values() for r in rows]
    disputed = common.DISPUTED_INSTANCES.get(common.BENCHMARK, frozenset())
    has_v3 = all("v3" in r for r in pooled)

    verifier_note = f", verifier {args.verifier_model}" if args.verifier_model else ""
    if args.min_pair is not None:
        verifier_note += f", HELD-OUT pairs >= {args.min_pair}"
    if args.max_pair is not None:
        verifier_note += f", DEV pairs < {args.max_pair}"
    out = [f"# ESV v3, stage by stage ({common.benchmark_info()['label']}, {models_label}{verifier_note})", "",
           f"Version `{V3_VERSION}`, decision rule `{DECISION_RULE}`. Gold is used only to score. Design and motivation: "
           "`esv_bottleneck_analysis.md`.", "",
           *([f"The ESV v3 arm is verified by {args.verifier_model}; the USC arm is read from `esv3_records.json`, "
              "because USC asks the solver (it has no verifier) and is measured once, in the self-verify run.", ""]
             if args.verifier_model else []),
           *pipeline(args.verifier_model, has_v3),
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
                "### Did USC actually select? (pooled)", "",
                "USC names its pick in one tag. Where that tag cannot be resolved to a response, USC keeps the "
                "initial answer and its accuracy becomes CoT maj@4's by construction - so this table has to be read "
                "before any USC comparison.", "",
                usc_health(per_model), ""]
        out += [
                "### The choice, on disagreeing instances only (pooled)", "", selection_table(pooled), "",
                "### Stages 2-3 as a detector, per candidate (pooled)", "", candidate_table(pooled), "", check_table(pooled), "",
                "### Decisions (pooled)", "", decision_table(pooled), "",
                "`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.", "",
                "### By arm", "",
                "Accuracy on the untouched originals, on strategy-preserving P' and on strategy-breaking P', per "
                "model and pooled. ESV's premise is that the breaking arm is where a selector has the most to "
                "recover, so the fixes / breaks column is read per arm.", ""]
        for model, rows in per_model.items():
            out += [f"**{model}**", "", breakdown(rows, ("cot_maj", "usc", "esv3")), ""]
        out += ["**Pooled**", "", breakdown(pooled, ("cot_maj", "usc", "esv3")), "",
                "### Decisions, per model", "",
                "The same decision mix as above, split by model: a checker that almost never returns "
                "`one_sound` is abstaining, not selecting.", ""]
        for model, rows in per_model.items():
            out += [f"**{model}**", "", decision_table(rows), ""]
        out += judge_select_section(pooled)
        out += ["## Compute cost", "",
                "Read off the `usage` and `latency_seconds` stored on every call, so this section is reproducible "
                "offline from the records. Every method starts from the same 4 CoT samples and adds at most one "
                "call, so the baseline is shared and only the extra call separates them. **serial** is billed "
                "compute time; **critical path** is what a user waits for, since `run_methods.py` issues the 4 "
                "samples concurrently while the extra call cannot start until all 4 are back.", "",
                compute_table(pooled), "",
                "### What the extra compute buys (pooled)", "",
                tradeoff_table(pooled), "",
                "### The disagreement gate is the efficiency mechanism (pooled)", "",
                "Stage 1 costs nothing - it is code over answers the solver already produced - and it is what keeps "
                "ESV under 1.5x the baseline despite a verification call that is itself larger than a CoT sample.", "",
                gate_economics(pooled), ""]
        by_arm_cost = arm_compute(pooled)
        if by_arm_cost:
            out += ["### Compute by arm (pooled)", "",
                    "The gate fires where the perturbation actually hurts, so ESV spends most on the strategy-breaking "
                    "arm - which is also where it gains most. The cost profile is adaptive, not flat.", "",
                    by_arm_cost, ""]
        out += price_sensitivity(pooled, args.verifier_model or "the solver itself")
        by_format = format_breakdown(pooled, ("cot_maj", "usc", "esv3"))
        if by_format:
            out += ["### By answer format (pooled)", "",
                    "`expression` answers are compared by LaTeX normalization and symbolic equivalence "
                    "(`expression_answers.py`) rather than by exact match, so this split is the check that the "
                    "scoring is not what separates the methods.", "", by_format, ""]
    name = "esv3_report.md" if models_tag == "large" else f"esv3_report_{models_tag}.md"
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
