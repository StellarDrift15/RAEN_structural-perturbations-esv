#!/usr/bin/env python3
"""Compares the phase-3 methods (ESV v2, its ablation, the retry controls) with ESV v4 and the single retry.

Every comparison uses only the consistent failures on which ALL listed methods have a result, so the counts are
paired. Reads results/<model>/records.json and writes results/analysis/method_comparison.md.

  python experiments/compare_methods.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import mcnemar_exact  # noqa: E402
import common  # noqa: E402
from common import ALL_SOLVER_MODELS, model_slug  # noqa: E402
from run_methods import EXTRA_METHODS, EXTRA_VERSION  # noqa: E402

LABELS = {"retry": "Retry (1 try, v4 control)", "esv_v4": "ESV v4", "retry3": "Retry x3 (compute-matched)",
          "retry3_matched": "Retry x3 + checks + guard", "esv_v2_k1": "ESV v2, 1 strategy", "esv_v2": "ESV v2"}
BASELINES = ("retry3", "retry3_matched")


def outcome(r: Dict[str, Any], method: str) -> Optional[Dict[str, Any]]:
    """{correct, tokens, repeated} for one method on one record, or None if it has no (current) result."""
    if method == "retry":
        c = r["retry"]
        return {"correct": bool(c["correct"]), "tokens": (c.get("usage") or {}).get("completion_tokens", 0), "repeated": None}
    if method == "esv_v4":
        e = r["esv"]
        tokens = sum((s.get("usage") or {}).get("completion_tokens", 0) for k, s in e["stages"].items() if not k.startswith("judge"))
        return {"correct": bool(e["final_correct"]), "tokens": tokens, "repeated": bool(e["final_repeated_initial"])}
    x = (r.get("extra") or {}).get(method)
    if not x or x.get("version") != EXTRA_VERSION:
        return None
    tokens = sum((s.get("usage") or {}).get("completion_tokens", 0) for k, s in x["stages"].items() if not k.startswith("judge"))
    return {"correct": bool(x["final_correct"]), "tokens": tokens, "repeated": bool(x["final_repeated_initial"])}


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "-"


def table(header: List[str], rows: List[List[Any]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def compare(records: List[Dict[str, Any]], methods: List[str]) -> List[List[Any]]:
    rows = []
    for m in methods:
        res = [outcome(r, m) for r in records]
        row = [LABELS[m], pct(sum(o["correct"] for o in res), len(res)),
               round(sum(o["tokens"] for o in res) / max(1, len(res)))]
        for b in BASELINES:
            if b not in methods or b == m:
                row.append("")
                continue
            base = [outcome(r, b) for r in records]
            wins = sum(o["correct"] and not q["correct"] for o, q in zip(res, base))
            losses = sum(q["correct"] and not o["correct"] for o, q in zip(res, base))
            p = mcnemar_exact(wins, losses)
            row.append(f"{wins} / {losses} (p={p:.3f})" if p is not None else f"{wins} / {losses}")
        rows.append(row)
    return rows


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Experiment 2 report.")
    common.add_benchmark_arg(parser)
    args = parser.parse_args(argv)
    common.select_benchmark(args.benchmark)
    extras = [m for m in EXTRA_METHODS if m in LABELS]
    per_model: Dict[str, List[Dict[str, Any]]] = {}
    present: Counter = Counter()
    for model in ALL_SOLVER_MODELS:
        path = common.RESULTS_DIR / model_slug(model) / "records.json"
        if not path.exists():
            continue
        recs = [r for r in json.loads(path.read_text(encoding="utf-8"))["instances"].values() if r.get("consistent_failure") and r.get("esv")]
        per_model[model] = recs
        for m in extras:
            present[m] += sum(outcome(r, m) is not None for r in recs)
    methods = ["retry", "esv_v4"] + [m for m in ("retry3", "retry3_matched", "esv_v2_k1", "esv_v2") if present[m]]
    out = [f"# Phase 3 method comparison (consistent failures, {common.benchmark_info()['label']})", "",
           f"Phase-3 version `{EXTRA_VERSION}`. Each table uses only the failures on which every listed method has a result. "
           "*Wins / losses* = failures this method solved and the baseline did not / the reverse (McNemar exact p). "
           "Tokens = mean solver output tokens per failure (judge excluded).", ""]
    pooled: List[Dict[str, Any]] = []
    for model, recs in per_model.items():
        shared = [r for r in recs if all(outcome(r, m) is not None for m in methods)]
        pooled += shared
        out += [f"## {model} ({len(shared)} of {len(recs)} failures)", "",
                table(["Method", "recovered", "tokens"] + [f"vs {LABELS[b]}" for b in BASELINES], compare(shared, methods)), ""]
    out += [f"## Pooled ({len(pooled)} failures)", "",
            table(["Method", "recovered", "tokens"] + [f"vs {LABELS[b]}" for b in BASELINES], compare(pooled, methods)), ""]

    for group, key in (("answer format", lambda r: r["answer_format"]),
                       ("arm", lambda r: r["problem_type"] if r["problem_type"] == "original" else f"perturbed-{r['perturbation_type']}")):
        rows = []
        for value in sorted({key(r) for r in pooled}):
            sub = [r for r in pooled if key(r) == value]
            rows.append([value, len(sub)] + [sum(outcome(r, m)["correct"] for r in sub) for m in methods])
        out += [f"### Pooled, by {group} (recovered counts)", "", table([group, "n"] + [LABELS[m] for m in methods], rows), ""]

    for m in ("esv_v2", "esv_v2_k1"):
        runs = [r["extra"][m] for r in pooled if m in methods]
        if not runs:
            continue
        verdicts = [v for x in runs for v in x["strategy_verdicts"]]
        any_valid = [x for x in runs if "VALID" in x["strategy_verdicts"]]
        out += [f"### {LABELS[m]}: stages (pooled)", "",
                f"- recognition said a failure occurred: {pct(sum(x['recognized'] for x in runs), len(runs))}",
                f"- diagnosis verdicts: {dict(Counter(x['diagnosis_verdict'] for x in runs))}",
                f"- strategies kept per run: {dict(sorted(Counter(len(x['strategies']) for x in runs).items()))}; "
                f"regenerated for similarity: {sum(x['strategy_attempts'] > 1 for x in runs)}",
                f"- strategies judged VALID: {pct(verdicts.count('VALID'), len(verdicts))}; runs with at least one VALID: "
                f"{pct(len(any_valid), len(runs))}, of which recovered {pct(sum(x['final_correct'] for x in any_valid), len(any_valid))}",
                f"- some derivation correct (ceiling): {pct(sum(any(x['derivations_correct']) for x in runs), len(runs))}; "
                f"final repeated known-wrong: {pct(sum(x['final_repeated_initial'] for x in runs), len(runs))}", ""]

    path = common.RESULTS_DIR / "analysis" / "method_comparison.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nSaved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
