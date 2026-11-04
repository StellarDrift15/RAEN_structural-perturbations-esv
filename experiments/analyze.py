#!/usr/bin/env python3
"""Cross-model analysis of Direct QA, CoT and ESV on the frozen pairs of a benchmark. No API calls.

Reads results/<model>/records.json for every solver and data/frozen_dataset.json, and writes results/analysis/:
  metrics.json      every number in the report (rates stored as n / d / rate, with 95% bootstrap CIs)
  report.md         the full write-up: dataset, headline table, ESV vs controls, perturbation types, stages
  instances.csv     one row per (model, instance)
  fig_*.png         figures

How each method is scored per instance (4 samples per method, temperature 0.7):
  pass@1     mean correctness of the 4 samples (the expected accuracy of a single run)
  maj@4      the majority answer of the 4 samples (ties: earliest sample)
  pass@4     at least one of the 4 samples is correct (an oracle picks the right sample)
  ESV        correct if any of the 4 CoT samples is correct; otherwise (a consistent failure) the ESV final answer.
             ESV runs only when all 4 samples are wrong, so ESV >= CoT pass@4 by construction, and
             ESV - CoT pass@4 is exactly the share of instances that resampling never solved but ESV did.
  Retry      the same, with the retry-with-feedback control instead of ESV (same gold-gated trigger, same
             "your answer is wrong" information, no ESV stages).
  ESV / Retry (maj) gold-free answer choice for non-failures: CoT maj@4 when not a consistent failure.

Run: python experiments/analyze.py
"""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import (  # noqa: E402
    ALL_SOLVER_MODELS, NUM_SAMPLES, answers_match, fmt, fmt_pp, is_choice_format, load_json,
    majority_answer, model_slug, ratio, save_json, utc_now,
)



def out_dir() -> Path:
    return common.RESULTS_DIR / "analysis"

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 7

# Display names. "Original" and "perturbed" are problem types; "breaking"/"preserving" are perturbation types.
MODEL_NAMES = {"qwen/qwen-2.5-7b-instruct": "Qwen 2.5 7B", "mistralai/mistral-nemo": "Mistral Nemo 12B",
               "google/gemma-3-4b-it": "Gemma 3 4B", "qwen/qwen-2.5-72b-instruct": "Qwen 2.5 72B",
               "mistralai/mistral-small-3.2-24b-instruct": "Mistral Small 3.2 24B",
               "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B"}

# Chance of guessing right after one known-wrong answer is excluded, per answer format (for the recovery baseline):
# single-correct MCQ: 1 of the 3 remaining options; multiple-correct: 1 of the 14 remaining non-empty subsets;
# integer: most JEEBench integer answers are single digits, so a gold of 0-9 counts as 1 of the 9 remaining digits
# (larger integers such as 281 are treated as unguessable, see chance_after_exclusion); numeric: not guessable.
CHANCE_AFTER_EXCLUSION = {"single": 1 / 3, "multiple": 1 / 14, "integer": 1 / 9, "numeric": 0.0}


def chance_after_exclusion(answer_format: str, gold: str) -> float:
    if answer_format == "integer" and not (gold.strip().isdigit() and int(gold) <= 9):
        return 0.0
    return CHANCE_AFTER_EXCLUSION[answer_format]


# ---- Per-instance outcomes -------------------------------------------------------------------------

def sample_outcomes(samples: List[Dict[str, Any]], answer_format: str, gold_normalized: Optional[str]) -> Dict[str, Any]:
    correct = [bool(s["correct"]) for s in samples]
    majority = majority_answer([s["normalized_answer"] for s in samples], answer_format)
    return {
        "pass1": sum(correct) / len(correct),
        "first": correct[0],
        "maj": answers_match(majority, gold_normalized, answer_format),
        "pass_k": any(correct),
        "n_correct": sum(correct),
        "unreadable": sum(not s["answer_readable"] for s in samples),
    }


def instance_row(model: str, r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One analysis row, or None if the record is incomplete (missing samples or a missing ESV run)."""
    samples = r["samples"]
    if any(s is None for m in ("dqa", "cot") for s in samples[m]):
        return None
    cf = not any(s["correct"] for s in samples["cot"])
    if cf and (r.get("esv") is None or r.get("retry") is None):
        return None
    gold_n = samples["cot"][0]["normalized_gold"]
    dqa = sample_outcomes(samples["dqa"], r["answer_format"], gold_n)
    cot = sample_outcomes(samples["cot"], r["answer_format"], gold_n)
    esv_fix = bool(cf and r["esv"]["final_correct"])
    retry_fix = bool(cf and r["retry"]["correct"])
    row = {
        "model": model, "instance_id": r["instance_id"], "pair_id": r["pair_id"], "problem_type": r["problem_type"],
        "perturbation_type": r["perturbation_type"], "question_type": r["question_type"], "answer_format": r["answer_format"],
        "gold": r["gold"],
        "dqa_pass1": dqa["pass1"], "dqa_maj": dqa["maj"], "dqa_pass4": dqa["pass_k"],
        "cot_pass1": cot["pass1"], "cot_first": cot["first"], "cot_maj": cot["maj"], "cot_pass4": cot["pass_k"],
        "cot_n_correct": cot["n_correct"],
        "consistent_failure": cf,
        "esv": cot["pass_k"] or esv_fix,
        "esv_maj": esv_fix if cf else cot["maj"],
        "retry": cot["pass_k"] or retry_fix,
        "retry_maj": retry_fix if cf else cot["maj"],
        "esv_recovered": esv_fix, "retry_recovered": retry_fix,
        "dqa_unreadable": dqa["unreadable"], "cot_unreadable": cot["unreadable"],
        "chance_recovery": chance_after_exclusion(r["answer_format"], r["gold"]) if cf else 0.0,
    }
    if cf:
        e = r["esv"]
        row.update({
            "esv_recognized": e["recognized"], "esv_diagnosis": e["diagnosis_verdict"], "esv_strategy": e["strategy_verdict"],
            "esv_independent_correct": e["independent_correct"], "esv_recovery_correct": e["recovery_correct"],
            "esv_cross_correct": e["cross_check_correct"], "esv_agree": e["derivations_agree"],
            "esv_repeated_wrong": e["final_repeated_initial"], "esv_self_verified": e["strategy_self_verified"],
            "initial_answer": e["initial_answer"], "esv_final_answer": e["final_answer"], "retry_answer": r["retry"]["answer"],
        })
    return row


# ---- Statistics ------------------------------------------------------------------------------------

def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def bootstrap_ci(rows: List[Dict], stat: Callable[[List[Dict]], Optional[float]], seed: int = BOOTSTRAP_SEED) -> Tuple[Optional[float], Optional[float]]:
    """95% percentile CI, resampling PAIRS (an original and its perturbation are not independent)."""
    clusters: Dict[Any, List[Dict]] = {}
    for row in rows:
        clusters.setdefault(row["pair_id"], []).append(row)
    keys = list(clusters)
    if len(keys) < 2:
        return None, None
    rng = random.Random(seed)
    values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [row for k in rng.choices(keys, k=len(keys)) for row in clusters[k]]
        v = stat(sample)
        if v is not None:
            values.append(v)
    if not values:
        return None, None
    values.sort()
    return values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]


def mcnemar_exact(b: int, c: int) -> Optional[float]:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return None
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def poisson_binomial_tail(ps: List[float], k: int) -> float:
    """P(X >= k) for a sum of independent Bernoulli(p_i): chance of at least k recoveries by guessing."""
    dist = [1.0]
    for p in ps:
        new = [0.0] * (len(dist) + 1)
        for i, q in enumerate(dist):
            new[i] += q * (1 - p)
            new[i + 1] += q * p
        dist = new
    return sum(dist[k:]) if k < len(dist) else 0.0


def fisher_exact(a: int, b: int, c: int, d: int) -> Optional[float]:
    if min(a + b, c + d) == 0:
        return None
    try:
        from scipy.stats import fisher_exact as fe
        return float(fe([[a, b], [c, d]])[1])
    except ImportError:
        return None


# ---- Metric blocks ---------------------------------------------------------------------------------

ACCURACY_COLUMNS = [
    ("dqa_pass1", "Direct QA pass@1"), ("dqa_maj", "Direct QA maj@4"), ("dqa_pass4", "Direct QA pass@4"),
    ("cot_pass1", "CoT pass@1"), ("cot_maj", "CoT maj@4"), ("cot_pass4", "CoT pass@4"),
    ("retry", "Retry control (any-correct gate)"), ("esv", "ESV (any-correct gate)"),
    ("retry_maj", "Retry control (maj gate)"), ("esv_maj", "ESV (maj gate)"),
]


def acc_block(rows: List[Dict], key: str) -> Dict[str, Any]:
    stat = lambda rs: mean([float(r[key]) for r in rs])  # noqa: E731
    value = stat(rows)
    lo, hi = bootstrap_ci(rows, stat) if rows else (None, None)
    return {"rate": None if value is None else round(value, 4), "n_instances": len(rows),
            "ci95": [None if lo is None else round(lo, 4), None if hi is None else round(hi, 4)]}


def delta_block(rows: List[Dict], a: str, b: str) -> Dict[str, Any]:
    stat = lambda rs: mean([float(r[a]) - float(r[b]) for r in rs])  # noqa: E731
    value = stat(rows)
    lo, hi = bootstrap_ci(rows, stat) if rows else (None, None)
    a_only = sum(bool(r[a]) and not bool(r[b]) for r in rows)
    b_only = sum(bool(r[b]) and not bool(r[a]) for r in rows)
    return {"delta_pp": None if value is None else round(100 * value, 2),
            "ci95_pp": [None if lo is None else round(100 * lo, 2), None if hi is None else round(100 * hi, 2)],
            f"{a}_only": a_only, f"{b}_only": b_only, "mcnemar_p": mcnemar_exact(a_only, b_only)}


def recovery_block(rows: List[Dict]) -> Dict[str, Any]:
    """On consistent failures only: ESV vs retry control vs chance."""
    cf = [r for r in rows if r["consistent_failure"]]
    esv_n, retry_n = sum(r["esv_recovered"] for r in cf), sum(r["retry_recovered"] for r in cf)
    esv_only = sum(r["esv_recovered"] and not r["retry_recovered"] for r in cf)
    retry_only = sum(r["retry_recovered"] and not r["esv_recovered"] for r in cf)
    chance = [r["chance_recovery"] for r in cf]
    return {
        "consistent_failures": len(cf),
        "esv_recovery": ratio(esv_n, len(cf)),
        "retry_recovery": ratio(retry_n, len(cf)),
        "esv_only": esv_only, "retry_only": retry_only,
        "esv_vs_retry_mcnemar_p": mcnemar_exact(esv_only, retry_only),
        "chance_expected": round(sum(chance), 2),
        "esv_vs_chance_p": round(poisson_binomial_tail(chance, esv_n), 6) if cf else None,
        "retry_vs_chance_p": round(poisson_binomial_tail(chance, retry_n), 6) if cf else None,
    }


def stage_block(rows: List[Dict]) -> Dict[str, Any]:
    cf = [r for r in rows if r["consistent_failure"]]
    rec = [r for r in cf if r["esv_recognized"]]
    diag = [r for r in cf if r["esv_diagnosis"] == "CORRECT"]
    strat = [r for r in cf if r["esv_strategy"] == "VALID"]
    agreed = [r for r in cf if r["esv_agree"]]
    return {
        "failures": len(cf),
        "recognition": ratio(len(rec), len(cf)),
        "independent_resolve_correct": ratio(sum(r["esv_independent_correct"] for r in cf), len(cf)),
        "diagnosis_correct": ratio(len(diag), len(cf)),
        "diagnosis_partial": sum(r["esv_diagnosis"] == "PARTIAL" for r in cf),
        "strategy_valid": ratio(len(strat), len(cf)),
        "strategy_self_verified": ratio(sum(r["esv_self_verified"] is True for r in cf), len(cf)),
        "recovery_stage_correct": ratio(sum(r["esv_recovery_correct"] for r in cf), len(cf)),
        "cross_check_correct": ratio(sum(r["esv_cross_correct"] for r in cf), len(cf)),
        "derivations_agree": ratio(len(agreed), len(cf)),
        "accuracy_when_agree": ratio(sum(r["esv_recovered"] for r in agreed), len(agreed)),
        "accuracy_when_adjudicated": ratio(sum(r["esv_recovered"] for r in cf if not r["esv_agree"]), len(cf) - len(agreed)),
        "final_correct": ratio(sum(r["esv_recovered"] for r in cf), len(cf)),
        "recovered_given_recognized": ratio(sum(r["esv_recovered"] for r in rec), len(rec)),
        "recovered_given_valid_strategy": ratio(sum(r["esv_recovered"] for r in strat), len(strat)),
        "recovered_given_correct_diagnosis": ratio(sum(r["esv_recovered"] for r in diag), len(diag)),
        "repeated_known_wrong_answer": ratio(sum(r["esv_repeated_wrong"] for r in cf), len(cf)),
        "funnel": {"failures": len(cf), "recognized": len(rec),
                   "recognized_diagnosed": sum(r in diag for r in rec),
                   "recognized_diagnosed_valid": sum(r in diag and r in strat for r in rec),
                   "recognized_diagnosed_valid_recovered": sum(r in diag and r in strat and r["esv_recovered"] for r in rec)},
    }


def subsets(rows: List[Dict]) -> Dict[str, List[Dict]]:
    return {
        "overall": rows,
        "original": [r for r in rows if r["problem_type"] == "original"],
        "perturbed": [r for r in rows if r["problem_type"] == "perturbed"],
        "perturbed_breaking": [r for r in rows if r["problem_type"] == "perturbed" and r["perturbation_type"] == "breaking"],
        "perturbed_preserving": [r for r in rows if r["problem_type"] == "perturbed" and r["perturbation_type"] == "preserving"],
        "original_of_breaking": [r for r in rows if r["problem_type"] == "original" and r["perturbation_type"] == "breaking"],
        "original_of_preserving": [r for r in rows if r["problem_type"] == "original" and r["perturbation_type"] == "preserving"],
    }


def model_metrics(rows: List[Dict]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, sub in subsets(rows).items():
        out[name] = {
            "instances": len(sub),
            "consistent_failures": sum(r["consistent_failure"] for r in sub),
            "accuracy": {key: acc_block(sub, key) for key, _ in ACCURACY_COLUMNS},
            "esv_minus_cot_pass4": delta_block(sub, "esv", "cot_pass4"),
            "esv_minus_cot_pass1": delta_block(sub, "esv", "cot_pass1") if sub else None,
            "esv_minus_retry": delta_block(sub, "esv", "retry"),
            "esv_maj_minus_cot_maj": delta_block(sub, "esv_maj", "cot_maj"),
            "recovery": recovery_block(sub),
            "stages": stage_block(sub),
        }
    # Robustness: accuracy drop original -> perturbed, per method, per perturbation type
    drops = {}
    for arm, (o_key, p_key) in {"all": ("original", "perturbed"), "breaking": ("original_of_breaking", "perturbed_breaking"),
                                "preserving": ("original_of_preserving", "perturbed_preserving")}.items():
        drops[arm] = {}
        for key, _ in ACCURACY_COLUMNS:
            o, p = out[o_key]["accuracy"][key]["rate"], out[p_key]["accuracy"][key]["rate"]
            drops[arm][key] = None if o is None or p is None else round(100 * (o - p), 2)
    out["accuracy_drop_pp"] = drops
    # Breaking vs preserving: is ESV's recovery rate different between the arms? (perturbed instances only)
    b, p = out["perturbed_breaking"]["recovery"], out["perturbed_preserving"]["recovery"]
    out["breaking_vs_preserving_recovery_fisher_p"] = fisher_exact(
        b["esv_recovery"]["n"], b["consistent_failures"] - b["esv_recovery"]["n"],
        p["esv_recovery"]["n"], p["consistent_failures"] - p["esv_recovery"]["n"])
    by_qtype = {}
    for qt in sorted({r["question_type"] for r in rows}):
        sub = [r for r in rows if r["question_type"] == qt]
        by_qtype[qt] = {"instances": len(sub), "cot_pass1": acc_block(sub, "cot_pass1")["rate"],
                        "cot_pass4": acc_block(sub, "cot_pass4")["rate"], "esv": acc_block(sub, "esv")["rate"],
                        "retry": acc_block(sub, "retry")["rate"], "recovery": recovery_block(sub)}
    out["by_question_type"] = by_qtype
    return out


def pooled_metrics(all_rows: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Across models: pooled discordant counts (a stratified sign test) and direction consistency."""
    out: Dict[str, Any] = {}
    for name in ("overall", "original", "perturbed", "perturbed_breaking", "perturbed_preserving"):
        esv_only = retry_only = recovered = failures = 0
        chance: List[float] = []
        for rows in all_rows.values():
            cf = [r for r in subsets(rows)[name] if r["consistent_failure"]]
            failures += len(cf)
            recovered += sum(r["esv_recovered"] for r in cf)
            esv_only += sum(r["esv_recovered"] and not r["retry_recovered"] for r in cf)
            retry_only += sum(r["retry_recovered"] and not r["esv_recovered"] for r in cf)
            chance += [r["chance_recovery"] for r in cf]
        out[name] = {"consistent_failures": failures, "esv_recovered": recovered, "esv_only_vs_retry": esv_only,
                     "retry_only_vs_esv": retry_only, "esv_vs_retry_mcnemar_p": mcnemar_exact(esv_only, retry_only),
                     "chance_expected": round(sum(chance), 2),
                     "esv_vs_chance_p": round(poisson_binomial_tail(chance, recovered), 6) if failures else None}
    return out


# ---- Figures ---------------------------------------------------------------------------------------
# Reference palette (dataviz skill), categorical slots in fixed order; validated light-mode (all checks pass, contrast
# WARN on slots 3-5 -> every bar carries a direct value label, and every figure has a table in report.md).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


METHOD_COLORS = {"Direct QA pass@1": SERIES[0], "CoT pass@1": SERIES[1], "CoT pass@4": SERIES[2],
                 "Retry control": SERIES[3], "ESV": SERIES[4]}  # color follows the method in every figure


def grouped_bars(ax, groups: List[str], series: List[Tuple[str, List[Optional[float]]]], ylabel: str, ymax: float = 100) -> None:
    n = len(series)
    width = 0.8 / n
    for i, (label, values) in enumerate(series):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        heights = [0 if v is None else v for v in values]
        bars = ax.bar(xs, heights, width=width, color=METHOD_COLORS[label], label=label, edgecolor=SURFACE, linewidth=2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.01, "n/a" if v is None else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, color=INK, fontsize=9)
    ax.set_ylim(0, ymax * 1.1)  # headroom so a value label on a full-height bar clears the title
    ax.set_yticks(range(0, int(ymax) + 1, 20))
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    style_axes(ax)


def make_figures(per_model: Dict[str, Dict[str, Any]]) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    models = list(per_model)
    names = [MODEL_NAMES.get(m, m) for m in models]
    written = []

    def pct(m: str, subset: str, key: str) -> Optional[float]:
        v = per_model[m][subset]["accuracy"][key]["rate"]
        return None if v is None else 100 * v

    # 1. Overall accuracy by method, per model
    for subset, title in (("overall", "all instances"), ("perturbed", "perturbed problems")):
        fig, ax = plt.subplots(figsize=(8.5, 4), facecolor=SURFACE)
        keys = [("dqa_pass1", "Direct QA pass@1"), ("cot_pass1", "CoT pass@1"), ("cot_pass4", "CoT pass@4"),
                ("retry", "Retry control"), ("esv", "ESV")]
        grouped_bars(ax, names, [(label, [pct(m, subset, k) for m in models]) for k, label in keys], "Accuracy (%)")
        ax.set_title(f"Accuracy by method, {title} ({common.benchmark_info()['label']})", color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper left", bbox_to_anchor=(0, -0.08), labelcolor=INK2)
        fig.tight_layout()
        path = out_dir() / f"fig_accuracy_{subset}.png"
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        written.append(path.name)

    # 2. Recovery rate on consistent failures: ESV vs retry control, small multiples by instance group
    groups = [("original", "Original problems"), ("perturbed_breaking", "Perturbed: strategy-breaking"),
              ("perturbed_preserving", "Perturbed: strategy-preserving")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), facecolor=SURFACE, sharey=True)
    for ax, (subset, title) in zip(axes, groups):
        def rec(m, key):
            r = per_model[m][subset]["recovery"][key]["rate"]
            return None if r is None else 100 * r
        grouped_bars(ax, names, [("ESV", [rec(m, "esv_recovery") for m in models]),
                                 ("Retry control", [rec(m, "retry_recovery") for m in models])], "", ymax=100)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.tick_params(axis="x", labelsize=8)
        for i, m in enumerate(models):
            ax.text(i, -12, f"n={per_model[m][subset]['recovery']['consistent_failures']}", ha="center", fontsize=7.5, color=INK2)
    axes[0].set_ylabel("Recovered (% of consistent failures)", color=INK2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=2, loc="upper right", labelcolor=INK2)
    fig.suptitle("Recovery on consistent failures (all 4 CoT samples wrong)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out_dir() / "fig_recovery_by_perturbation.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    written.append(path.name)

    # 3. ESV stage funnel, one panel per model (single series, so a single hue)
    stages = [("failures", "Consistent failures"), ("recognized", "Recognized (blind)"), ("recognized_diagnosed", "+ diagnosed"),
              ("recognized_diagnosed_valid", "+ valid strategy"), ("recognized_diagnosed_valid_recovered", "+ recovered")]
    fig, axes = plt.subplots(1, len(models), figsize=(3.8 * len(models), 3.4), facecolor=SURFACE, sharex=False)
    axes = axes if len(models) > 1 else [axes]
    for ax, m, name in zip(axes, models, names):
        f = per_model[m]["overall"]["stages"]["funnel"]
        values = [f[k] for k, _ in stages]
        ys = list(range(len(stages)))[::-1]
        ax.barh(ys, values, color=METHOD_COLORS["ESV"], height=0.6, edgecolor=SURFACE, linewidth=2)
        top = max(values) if values and max(values) else 1
        for y, v in zip(ys, values):
            ax.text(v + top * 0.02, y, str(v), va="center", fontsize=8, color=INK2)
        ax.set_yticks(ys)
        ax.set_yticklabels([label for _, label in stages], fontsize=8, color=INK)
        ax.set_xlim(0, top * 1.18)
        ax.set_title(name, color=INK, fontsize=10, loc="left")
        style_axes(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    fig.suptitle("ESV stage funnel (judge-graded diagnosis and strategy)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    path = out_dir() / "fig_esv_funnel.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    written.append(path.name)
    return written


# ---- Report ----------------------------------------------------------------------------------------

def pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def pct_ci(block: Dict[str, Any]) -> str:
    lo, hi = block["ci95"]
    ci = "" if lo is None else f" [{100 * lo:.0f}, {100 * hi:.0f}]"
    return f"{pct(block['rate'])}{ci}"


def pval(p: Optional[float]) -> str:
    return "n/a" if p is None else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def delta_text(d: Dict[str, Any]) -> str:
    lo, hi = d["ci95_pp"]
    ci = "" if lo is None else f" [{lo:+.1f}, {hi:+.1f}]"
    return f"{fmt_pp(d['delta_pp'])}{ci}"


def build_report(dataset: Dict[str, Any], per_model: Dict[str, Dict[str, Any]], coverage: Dict[str, Dict[str, int]],
                 pooled: Dict[str, Any], figures: List[str]) -> str:
    models = list(per_model)
    names = {m: MODEL_NAMES.get(m, m) for m in models}
    split = dataset.get("perturbation_split", {})
    lines = [
        f"# {common.benchmark_info()['label']}: Direct QA vs CoT vs ESV across three solver models", "",
        f"Generated {utc_now()} by `experiments/analyze.py`.", "",
        "## Dataset", "",
        f"- {dataset.get('problems_sampled')} problems sampled at random from {common.benchmark_info()['label']} (seed {dataset.get('sample_seed')}); "
        f"{dataset.get('problems_processed')} processed; **{dataset.get('num_pairs')} frozen (P, P') pairs**.",
        f"- Perturbation split: **{split.get('breaking', 0)} strategy-breaking / {split.get('preserving', 0)} strategy-preserving**; "
        f"problems attempted per mode {dataset.get('problems_attempted_per_mode')}, acceptance rate {dataset.get('acceptance_rate_per_mode')}.",
        f"- Generator {dataset.get('models', {}).get('a')}; verifiers {dataset.get('models', {}).get('b')}, {dataset.get('models', {}).get('c')}. "
        "P' keeps the original answer format (single-correct MCQ, multiple-correct MCQ, integer, numeric).",
    ]
    comp = {}
    for p in dataset.get("pairs", []):
        comp.setdefault(p["perturbation_type"], {}).setdefault(p["type"], 0)
        comp[p["perturbation_type"]][p["type"]] += 1
    if comp:
        lines.append(f"- Question-type composition per arm: {comp}")
    lines += ["", "## Scoring", "",
              f"Every method gets {NUM_SAMPLES} samples per instance at temperature 0.7. **pass@1** = mean correctness of the samples; "
              "**maj@4** = majority answer; **pass@4** = any sample correct. ESV runs only on *consistent failures* (all 4 CoT "
              "samples wrong). The headline reports what ESV does on those instances, and a realistic accuracy that uses the "
              "CoT majority vote wherever ESV did not run. Under the any-correct (oracle) gate, ESV is scored as correct if any "
              "CoT sample is correct or the ESV final answer is correct, so "
              "**ESV - CoT pass@4 is exactly the share of instances that resampling never solved but ESV did.** The retry "
              "control uses the same trigger and the same \"your answer is wrong\" information without the ESV stages, so "
              "**ESV - Retry isolates the ESV stages themselves.** 95% CIs: bootstrap over pairs.", ""]

    lines += ["## Coverage", "", "| Model | Instances scored | Incomplete (excluded) |", "|---|---|---|"]
    for m in models:
        lines.append(f"| {names[m]} | {coverage[m]['scored']} / {coverage[m]['total']} | {coverage[m]['incomplete']} |")
    lines.append("")

    # Headline: (1) what ESV does on consistent failures, against chance and the retry control; (2) realistic,
    # gold-free accuracy (majority vote wherever ESV did not run). Any-correct-gated numbers follow as supporting rows.
    lines += ["## Headline 1: recovery on consistent failures (all 4 CoT samples wrong)", "",
              "Of the instances the model failed 4 out of 4 times, the share whose final answer is correct. Chance = expected "
              "recoveries from guessing uniformly among the answers left after the known-wrong one is excluded.", "",
              "| Group | Metric | " + " | ".join(names[m] for m in models) + " |", "|---|---|" + "---|" * len(models)]
    for subset, label in (("overall", "all"), ("original", "original"), ("perturbed_breaking", "perturbed, breaking"),
                          ("perturbed_preserving", "perturbed, preserving")):
        recs = {m: per_model[m][subset]["recovery"] for m in models}
        lines.append(f"| {label} | ESV recovered | " + " | ".join(fmt(recs[m]["esv_recovery"]) for m in models) + " |")
        lines.append(f"| {label} | Retry control recovered | " + " | ".join(fmt(recs[m]["retry_recovery"]) for m in models) + " |")
        lines.append(f"| {label} | Expected by chance | " + " | ".join(f"{recs[m]['chance_expected']}" for m in models) + " |")
        lines.append(f"| {label} | p (ESV vs chance) / p (ESV vs Retry) | " + " | ".join(
            f"{pval(recs[m]['esv_vs_chance_p'])} / {pval(recs[m]['esv_vs_retry_mcnemar_p'])}" for m in models) + " |")
    lines += ["", "## Headline 2: realistic accuracy (no gold answer used to pick among the 4 samples)", "",
              "ESV (maj gate) = CoT majority vote where ESV did not run, ESV's final answer where it did. The trigger itself "
              "(all 4 samples wrong) still uses the gold answer.", "",
              "| Method | " + " | ".join(names[m] for m in models) + " |", "|---|" + "---|" * len(models)]
    for key, label in (("dqa_pass1", "Direct QA pass@1"), ("cot_pass1", "CoT pass@1"), ("cot_maj", "CoT maj@4"),
                       ("retry_maj", "Retry control (maj gate)"), ("esv_maj", "ESV (maj gate)")):
        lines.append(f"| {label} | " + " | ".join(pct_ci(per_model[m]["overall"]["accuracy"][key]) for m in models) + " |")
    lines.append("| ESV (maj gate) - CoT maj@4 | " + " | ".join(delta_text(per_model[m]["overall"]["esv_maj_minus_cot_maj"]) for m in models) + " |")
    lines += ["", "The sections below give every scoring view per group, including the any-correct (oracle) gate.", ""]

    for subset, title in (("overall", "all instances"), ("original", "original problems"), ("perturbed", "perturbed problems"),
                          ("perturbed_breaking", "perturbed, strategy-breaking"), ("perturbed_preserving", "perturbed, strategy-preserving")):
        lines += [f"## Accuracy: {title}", "", "| Method | " + " | ".join(names[m] for m in models) + " |",
                  "|---|" + "---|" * len(models)]
        for key, label in ACCURACY_COLUMNS:
            lines.append(f"| {label} | " + " | ".join(pct_ci(per_model[m][subset]["accuracy"][key]) for m in models) + " |")
        lines.append("| Consistent failures (ESV triggered) | " + " | ".join(
            f"{per_model[m][subset]['consistent_failures']} / {per_model[m][subset]['instances']}" for m in models) + " |")
        lines.append("")

    lines += ["## ESV vs the fair comparisons", "",
              "| Comparison (overall) | " + " | ".join(names[m] for m in models) + " |", "|---|" + "---|" * len(models)]
    for key, label in (("esv_minus_cot_pass4", "ESV - CoT pass@4 (gain beyond resampling)"),
                       ("esv_minus_retry", "ESV - Retry control (gain from the ESV stages)"),
                       ("esv_maj_minus_cot_maj", "ESV (maj gate) - CoT maj@4"),
                       ("esv_minus_cot_pass1", "ESV - CoT pass@1 (NOT like-for-like: ESV sees 4 samples + gold gate)")):
        lines.append(f"| {label} | " + " | ".join(delta_text(per_model[m]["overall"][key]) for m in models) + " |")
    lines += ["", "### Recovery on consistent failures", "",
              "| Group | Model | Failures | ESV recovered | Retry recovered | ESV-only / Retry-only | McNemar p (ESV vs Retry) | Chance expected | p (ESV vs chance) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for subset in ("overall", "original", "perturbed", "perturbed_breaking", "perturbed_preserving"):
        for m in models:
            r = per_model[m][subset]["recovery"]
            lines.append(f"| {subset} | {names[m]} | {r['consistent_failures']} | {fmt(r['esv_recovery'])} | {fmt(r['retry_recovery'])} | "
                         f"{r['esv_only']} / {r['retry_only']} | {pval(r['esv_vs_retry_mcnemar_p'])} | {r['chance_expected']} | {pval(r['esv_vs_chance_p'])} |")
    lines += ["", "Pooled over the three models (stratified sign test on the discordant instances):", "",
              "| Group | Failures | ESV recovered | ESV-only / Retry-only | McNemar p | Chance expected | p (vs chance) |", "|---|---|---|---|---|---|---|"]
    for subset, r in pooled.items():
        lines.append(f"| {subset} | {r['consistent_failures']} | {r['esv_recovered']} | {r['esv_only_vs_retry']} / {r['retry_only_vs_esv']} | "
                     f"{pval(r['esv_vs_retry_mcnemar_p'])} | {r['chance_expected']} | {pval(r['esv_vs_chance_p'])} |")
    lines += ["", "Breaking vs preserving (perturbed instances), Fisher exact test on ESV recovery rates: " +
              ", ".join(f"{names[m]} p = {pval(per_model[m]['breaking_vs_preserving_recovery_fisher_p'])}" for m in models), ""]

    lines += ["## Robustness: accuracy drop original -> perturbed (pp)", "",
              "| Method | Arm | " + " | ".join(names[m] for m in models) + " |", "|---|---|" + "---|" * len(models)]
    for key, label in ACCURACY_COLUMNS:
        if key in ("dqa_maj", "dqa_pass4", "retry_maj", "esv_maj"):
            continue
        for arm in ("all", "breaking", "preserving"):
            lines.append(f"| {label} | {arm} | " + " | ".join(fmt_pp(per_model[m]["accuracy_drop_pp"][arm][key]) for m in models) + " |")
    lines.append("")

    lines += ["## ESV stages (consistent failures, overall)", "", "| Stage metric | " + " | ".join(names[m] for m in models) + " |",
              "|---|" + "---|" * len(models)]
    for key, label in (("recognition", "Recognition (blind)"), ("independent_resolve_correct", "Independent re-solve correct"),
                       ("diagnosis_correct", "Diagnosis correct (judge)"), ("strategy_valid", "Alternative strategy valid (judge)"),
                       ("strategy_self_verified", "Strategy self-verified by the model"),
                       ("recovery_stage_correct", "Correct after recovery stage"), ("cross_check_correct", "Cross-check correct"),
                       ("derivations_agree", "Recovery and cross-check agree"), ("accuracy_when_agree", "... accuracy when they agree"),
                       ("accuracy_when_adjudicated", "... accuracy when adjudicated"), ("final_correct", "Final answer correct (recovery rate)"),
                       ("recovered_given_recognized", "Recovered, given recognized"),
                       ("recovered_given_correct_diagnosis", "Recovered, given correct diagnosis"),
                       ("recovered_given_valid_strategy", "Recovered, given valid strategy"),
                       ("repeated_known_wrong_answer", "Repeated the known-wrong answer")):
        lines.append(f"| {label} | " + " | ".join(fmt(per_model[m]["overall"]["stages"][key]) for m in models) + " |")
    lines += ["", "### By perturbation type (perturbed instances)", "",
              "| Stage metric | Arm | " + " | ".join(names[m] for m in models) + " |", "|---|---|" + "---|" * len(models)]
    for key, label in (("recognition", "Recognition"), ("diagnosis_correct", "Diagnosis correct"),
                       ("strategy_valid", "Strategy valid"), ("final_correct", "Recovered")):
        for arm in ("perturbed_breaking", "perturbed_preserving"):
            lines.append(f"| {label} | {arm.split('_')[1]} | " + " | ".join(fmt(per_model[m][arm]["stages"][key]) for m in models) + " |")
    lines.append("")

    lines += ["## By question type (all instances)", "",
              "| Question type | Model | Instances | CoT pass@1 | CoT pass@4 | Retry | ESV | ESV recovered |", "|---|---|---|---|---|---|---|---|"]
    for qt in sorted({qt for m in models for qt in per_model[m]["by_question_type"]}):
        for m in models:
            q = per_model[m]["by_question_type"].get(qt)
            if q:
                lines.append(f"| {qt} | {names[m]} | {q['instances']} | {pct(q['cot_pass1'])} | {pct(q['cot_pass4'])} | {pct(q['retry'])} | "
                             f"{pct(q['esv'])} | {fmt(q['recovery']['esv_recovery'])} |")
    lines.append("")
    if figures:
        lines += ["## Figures", ""] + [f"![{f}]({f})" for f in figures] + [""]
    lines += ["## Notes on validity", "",
              "- ESV and the retry control are triggered by the gold answer (all 4 CoT samples wrong). Their accuracies are "
              "therefore oracle-gated and must be compared with CoT pass@4 (also oracle) or with each other, never with pass@1.",
              "- From diagnosis onward ESV is told that one answer (its most common wrong answer) is incorrect; for a "
              "single-correct MCQ that alone leaves a 1-in-3 guess. The chance column and the retry control quantify how much "
              "of the recovery that information explains.",
              "- Diagnosis and strategy verdicts come from an LLM judge (gpt-oss-120b); spot-check them in records.json.",
              "- Gemma 2 9B and Mistral 7B v0.3 were unavailable on OpenRouter (no providers), so Gemma 3 4B and Mistral Nemo "
              "12B were used instead.",
              ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Experiment 1 report (oracle-gated ESV v4 vs CoT and the retry control).")
    common.add_benchmark_arg(parser)
    args = parser.parse_args(argv)
    common.select_benchmark(args.benchmark)
    dataset = load_json(common.FROZEN_DATASET, {}) or {}
    pair_ids = {p["id"] for p in dataset.get("pairs", [])}
    out_dir().mkdir(parents=True, exist_ok=True)
    all_rows: Dict[str, List[Dict]] = {}
    coverage: Dict[str, Dict[str, int]] = {}
    for model in ALL_SOLVER_MODELS:
        data = load_json(common.RESULTS_DIR / model_slug(model) / "records.json")
        if not data or not any(r.get("esv") for r in data["instances"].values()):
            continue  # no oracle-gated ESV results (e.g. a --samples-only run)
        recs = [r for r in data["instances"].values() if r["pair_id"] in pair_ids and not common.is_disputed(r["instance_id"])]
        rows = [row for row in (instance_row(model, r) for r in recs) if row]
        coverage[model] = {"total": len(recs), "scored": len(rows), "incomplete": len(recs) - len(rows)}
        if rows:
            all_rows[model] = rows
    if not all_rows:
        print("No scorable results found; run run_methods.py first.")
        return 1
    per_model = {m: model_metrics(rows) for m, rows in all_rows.items()}
    pooled = pooled_metrics(all_rows)
    figures = make_figures(per_model)
    save_json(out_dir() / "metrics.json", {"generated_at": utc_now(), "coverage": coverage, "per_model": per_model, "pooled": pooled})
    report = build_report(dataset, per_model, {m: coverage[m] for m in per_model}, pooled, figures)
    (out_dir() / "report.md").write_text(report, encoding="utf-8")
    with (out_dir() / "instances.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        rows = [row for rs in all_rows.values() for row in rs]
        columns = sorted({k for row in rows for k in row}, key=lambda k: (k not in rows[0], k))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(report)
    print(f"\nSaved to {out_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
