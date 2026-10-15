#!/usr/bin/env python3
"""Direct QA and Chain-of-Thought baselines on the frozen paired dataset, aligned with ESV.

Everything except the prompting strategy is shared with methods/esv.py by IMPORTING it, so the three methods
cannot drift apart:
  - same solver model, temperature 0, max_tokens cap (doubled once on truncation), retries and nudges (call_stage)
  - same question text (extract_question strips the dataset wrapper) and answer-format note (answer_format_note)
  - same tag-based answer parsing (parse_tagged / extract_answer)
  - same deterministic scorer (score_answer), so all three methods are graded by identical code

The two baselines:
  cot  Chain-of-Thought = ESV's own Stage 0 prompt (build_initial_prompt), which already asks for step-by-step
       reasoning. ESV's final answer therefore differs from CoT ONLY by the five ESV stages run on failures.
  dqa  Direct QA = same format note, tags and format rules, but the model must give ONLY the final answer.

Outputs (default: results/ beside the frozen dataset, one set per solver model):
  baseline_results_<model>.json   every call: raw response, parsed answer, finish reason, score
  baseline_report_<model>.md      accuracy table for Direct QA / CoT / ESV, robustness, paired comparison

If results/esv_results_<model>.json exists (from methods/esv.py with the same --model), ESV is added to the
report automatically, including a paired instance-by-instance comparison of ESV vs CoT.

Run (from the project root, same .env as esv.py):
  python methods/baselines.py                       # both baselines, full dataset (resumes automatically)
  python methods/baselines.py --methods cot         # only CoT
  python methods/baselines.py --limit-pairs 2       # small pilot
  python methods/baselines.py --report-only         # rebuild the report from saved results; no API calls
"""

import argparse
import json
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import esv  # noqa: E402  (loads the same .env, model defaults and settings as the ESV run)
from esv import (  # noqa: E402
    DEFAULT_INPUT, DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_REASONING, ENV_FILE, REQUEST_TIMEOUT,
    SOLVER_MAX_TOKENS, TAG_RULES, answer_format_note, build_initial_prompt, call_stage, extract_answer,
    extract_question, fmt, fmt_pp, is_choice_gold, model_slug, ratio, reasoning_setting, save_json,
    score_answer, utc_now, value_as_text,
)

# Tied to ESV's prompt version: if the ESV prompts or scorer change, saved baseline records are rerun too.
BASELINE_VERSION = f"baselines_v1|{esv.PROMPT_VERSION}"

DIRECT_ANSWER_INSTRUCTION = """Solve the following problem. Give ONLY the final answer: do not write any reasoning, working or explanation.

Return your response using exactly this tag:
<final_answer>The option letter(s) for MCQs, or the final numerical answer otherwise</final_answer>
For MCQs, always give the option letter(s) in final_answer, even when the answer is numerical. Do not omit the final answer."""


def build_direct_prompt(question: str, format_note: str) -> str:
    # Same layout as esv.build_initial_prompt; only the instruction differs.
    return f"{DIRECT_ANSWER_INSTRUCTION}\n{TAG_RULES}\n\n{format_note}\n\nProblem:\n{question}"


METHODS = {
    "dqa": {"label": "Direct QA", "build": build_direct_prompt, "fields": {"final_answer": ""}},
    "cot": {"label": "CoT", "build": build_initial_prompt, "fields": {"reasoning": "", "final_answer": ""}},
}


def make_instance(pair: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """Mirrors esv.Runner.initial_solve exactly: same gold, question text and answer-format note."""
    gold = value_as_text(pair.get("original_gold") if kind == "original" else pair.get("answer_prime"))
    answer_format = "choice" if is_choice_gold(gold) else "numeric"
    single_correct = kind == "original" and pair.get("type") == "MCQ"
    return {
        "pair_id": pair.get("id"),
        "problem_type": kind,
        "question_type": pair.get("type"),
        "answer_format": answer_format,
        "question": extract_question(pair.get("original_problem" if kind == "original" else "perturbed_problem", "")),
        "format_note": answer_format_note(answer_format, single_correct),
        "gold": gold,
    }


class BaselineRunner:
    def __init__(self, client, model: str, reasoning: Optional[Dict], max_tokens: int):
        self.client, self.model, self.reasoning, self.max_tokens = client, model, reasoning, max_tokens

    def run(self, method: str, pair: Dict[str, Any], kind: str) -> Dict[str, Any]:
        spec = METHODS[method]
        inst = make_instance(pair, kind)
        call = call_stage(self.client, self.model, f"baseline_{method}", spec["build"](inst["question"], inst["format_note"]),
                          spec["fields"], ["final_answer"], self.reasoning, self.max_tokens)
        answer = extract_answer(call)
        score = score_answer(answer, inst["gold"])
        call.update(score)
        return {
            "record_id": f"{method}:{pair.get('id')}_{kind}",
            "instance_id": f"{pair.get('id')}_{kind}",
            "method": method,
            **{k: inst[k] for k in ("pair_id", "problem_type", "question_type", "answer_format", "gold")},
            "model": self.model,
            "version": BASELINE_VERSION,
            "answer": answer,
            **score,
            "status": call["status"],
            "finish_reason": call.get("finish_reason"),
            "call": call,
            "finished_at": utc_now(),
        }


# ---- Metrics ---------------------------------------------------------------------------------------

def accuracy_block(correct_by_id: Dict[str, bool]) -> Dict[str, Any]:
    orig = {k: v for k, v in correct_by_id.items() if k.endswith("_original")}
    pert = {k: v for k, v in correct_by_id.items() if k.endswith("_perturbed")}
    pairs = sorted({k.rsplit("_", 1)[0] for k in orig} & {k.rsplit("_", 1)[0] for k in pert})
    o_ok = [correct_by_id[f"{p}_original"] for p in pairs]
    p_ok = [correct_by_id[f"{p}_perturbed"] for p in pairs]
    acc_o, acc_p = ratio(sum(o_ok), len(pairs)), ratio(sum(p_ok), len(pairs))
    return {
        "overall": ratio(sum(correct_by_id.values()), len(correct_by_id)),
        "original": ratio(sum(orig.values()), len(orig)),
        "perturbed": ratio(sum(pert.values()), len(pert)),
        "drop_pp": round(100 * (acc_o["rate"] - acc_p["rate"]), 2) if pairs else None,
        "pairs": {"both": sum(a and b for a, b in zip(o_ok, p_ok)), "original_only": sum(a and not b for a, b in zip(o_ok, p_ok)),
                  "perturbed_only": sum(b and not a for a, b in zip(o_ok, p_ok)), "neither": sum(not a and not b for a, b in zip(o_ok, p_ok))},
    }


def mcnemar_exact(b: int, c: int) -> Optional[float]:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_esv(path: Path) -> Optional[Dict[str, Dict[str, Any]]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    usable = [r for r in data.get("results", []) if not r.get("fatal_error") and "initial" in r and "final" in r]
    return {r["instance_id"]: r for r in usable}


def build_report(records: List[Dict], esv_records: Optional[Dict[str, Dict]], esv_path: Path, model: str,
                 dataset: Dict[str, Any]) -> str:
    rows = []  # (label, {instance_id: correct}, extra)
    for method in ("dqa", "cot"):
        recs = [r for r in records if r["method"] == method]
        if recs:
            rows.append((METHODS[method]["label"], {r["instance_id"]: r["correct"] for r in recs}, recs))
    if esv_records:
        rows.append(("ESV Stage 0 (its initial CoT solve)", {k: r["initial"]["correct"] for k, r in esv_records.items()}, None))
        rows.append(("ESV (final, after recovery)", {k: r["final"]["correct"] for k, r in esv_records.items()}, None))

    lines = [f"# Baseline report: `{model}`", "",
             f"- Baseline version: `{BASELINE_VERSION}` (prompts, call settings and scorer imported from methods/esv.py)",
             f"- ESV results: `{esv_path}`" + ("" if esv_records else " (not found: run methods/esv.py with the same --model)"),
             f"- Generated {utc_now()}", "",
             "## Accuracy", "",
             "| Method | Overall | Original | Perturbed | Drop (orig - pert) | Pairs both / orig only / pert only / neither |",
             "|---|---|---|---|---|---|"]
    for label, correct, _ in rows:
        a = accuracy_block(correct)
        p = a["pairs"]
        lines.append(f"| {label} | {fmt(a['overall'])} | {fmt(a['original'])} | {fmt(a['perturbed'])} | {fmt_pp(a['drop_pp'])} | "
                     f"{p['both']} / {p['original_only']} / {p['perturbed_only']} / {p['neither']} |")
    lines.append("")

    lines += ["## Answer extraction and truncation", "",
              "| Method | No readable answer | Call status counts | Hit token cap on final attempt |", "|---|---|---|---|"]
    for label, _, recs in rows:
        if recs is None:
            continue
        status: Dict[str, int] = {}
        for r in recs:
            status[r["status"]] = status.get(r["status"], 0) + 1
        lines.append(f"| {label} | {sum(not r['answer_readable'] for r in recs)}/{len(recs)} | {status} | "
                     f"{sum(r.get('finish_reason') == 'length' for r in recs)} |")
    lines.append("")

    cot = {r["instance_id"]: r["correct"] for r in records if r["method"] == "cot"}
    if esv_records and cot:
        shared = sorted(set(cot) & set(esv_records))
        e_final = {k: esv_records[k]["final"]["correct"] for k in shared}
        e_init = {k: esv_records[k]["initial"]["correct"] for k in shared}
        both = sum(cot[k] and e_final[k] for k in shared)
        esv_only = sum(e_final[k] and not cot[k] for k in shared)
        cot_only = sum(cot[k] and not e_final[k] for k in shared)
        p = mcnemar_exact(esv_only, cot_only)
        agree0 = sum(cot[k] == e_init[k] for k in shared)
        lines += ["## Paired comparison: ESV vs CoT (same instances)", "",
                  f"- Instances in both: {len(shared)}",
                  f"- Both correct: {both} | ESV only: {esv_only} | CoT only: {cot_only} | neither: {len(shared) - both - esv_only - cot_only}",
                  f"- Exact McNemar test on the discordant instances: p = {p:.3f}" if p is not None else "- No discordant instances.",
                  f"- Sanity check: the CoT baseline and ESV Stage 0 use the identical prompt at temperature 0, and agree on "
                  f"correctness for {agree0}/{len(shared)} instances. Some disagreement is normal (providers are not fully deterministic at temperature 0); a large share would mean the runs were not set up the same way.", ""]

    fmt_change = [p["id"] for p in dataset.get("pairs", [])
                  if is_choice_gold(p.get("answer_prime")) and not is_choice_gold(p.get("original_gold"))]
    lines += ["## Notes on validity", "",
              "- All methods share the solver model, temperature, token cap and retries, question text, answer-format note, "
              "answer parsing and scorer (imported from methods/esv.py). Only the instruction differs.",
              "- ESV runs its stages only when the initial answer is wrong according to the gold answer, so its final "
              "accuracy can never fall below its Stage 0 accuracy. Report this alongside any ESV vs CoT comparison.",
              ]
    if fmt_change:
        lines.append(f"- Pairs {fmt_change} have integer-answer originals but 4-option multiple-choice perturbed versions, "
                     "so original vs perturbed accuracy is not like-for-like on those pairs.")
    lines.append("")
    return "\n".join(lines)


# ---- Driver ----------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Run Direct QA and CoT baselines aligned with ESV, and compare with ESV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Solver model (must match the ESV run to compare)")
    parser.add_argument("--reasoning", default=DEFAULT_REASONING, choices=["default", "off", "low", "medium", "high", "xhigh"])
    parser.add_argument("--max-tokens", type=int, default=SOLVER_MAX_TOKENS, help="Same cap as ESV (doubled once on truncation)")
    parser.add_argument("--methods", nargs="+", default=["dqa", "cot"], choices=list(METHODS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Rebuild the report from saved results; no API calls")
    parser.add_argument("--esv-results", type=Path, default=None, help="Override the path to esv_results_<model>.json")
    args = parser.parse_args()

    try:
        dataset = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: unable to read frozen dataset {args.input}: {exc}", file=sys.stderr)
        return 2
    pairs = dataset.get("pairs") or []
    if args.limit_pairs is not None:
        pairs = pairs[: max(0, args.limit_pairs)]

    slug = model_slug(args.model)
    results_path = args.output_dir / f"baseline_results_{slug}.json"
    report_path = args.output_dir / f"baseline_report_{slug}.md"
    esv_path = args.esv_results or args.output_dir / f"esv_results_{slug}.json"

    tasks = [(m, pair, kind) for m in args.methods for pair in pairs for kind in ("original", "perturbed")]
    order = [f"{m}:{pair.get('id')}_{kind}" for m, pair, kind in tasks]
    records: Dict[str, Dict] = {}
    if not args.no_resume and results_path.exists():
        try:
            saved = json.loads(results_path.read_text(encoding="utf-8")).get("results", [])
            records = {r["record_id"]: r for r in saved if isinstance(r, dict) and r.get("record_id")}
        except (OSError, json.JSONDecodeError):
            print(f"Warning: could not read {results_path}; starting fresh.", file=sys.stderr)
    lock = threading.Lock()
    metadata = {"model": args.model, "reasoning": args.reasoning, "max_tokens": args.max_tokens, "version": BASELINE_VERSION,
                "input_file": str(args.input.resolve()), "methods": args.methods}

    def write_outputs() -> str:
        ordered = [records[k] for k in order if k in records and not records[k].get("fatal_error")]
        esv_records = load_esv(esv_path)
        if esv_records is not None:
            wanted = {k.split(":", 1)[1] for k in order}
            esv_records = {k: v for k, v in esv_records.items() if k in wanted}
        save_json(results_path, {"metadata": metadata, "results": [records[k] for k in order if k in records]})
        report = build_report(ordered, esv_records, esv_path, args.model, {"pairs": pairs})
        report_path.write_text(report, encoding="utf-8")
        return report

    if not args.report_only:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print(f"ERROR: OPENROUTER_API_KEY is not set (.env used: {ENV_FILE or 'none found'}).", file=sys.stderr)
            return 2
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=0)
        runner = BaselineRunner(client, args.model, reasoning_setting(args.reasoning), args.max_tokens)

        def reusable(r: Optional[Dict]) -> bool:
            return bool(r) and r.get("version") == BASELINE_VERSION and r.get("model") == args.model and not r.get("fatal_error")

        pending = [(m, p, k) for m, p, k in tasks if not reusable(records.get(f"{m}:{p.get('id')}_{k}"))]
        print(f"Solver: {args.model} (reasoning={args.reasoning}, max_tokens={args.max_tokens}) | Methods: {args.methods}")
        print(f"Calls: {len(tasks)} | To run: {len(pending)} | Reused: {len(tasks) - len(pending)}")

        def run_one(m: str, p: Dict, k: str) -> Dict:
            try:
                return runner.run(m, p, k)
            except Exception as exc:  # one broken call must not kill the run
                return {"record_id": f"{m}:{p.get('id')}_{k}", "method": m, "model": args.model,
                        "version": BASELINE_VERSION, "fatal_error": f"{type(exc).__name__}: {exc}"}

        executor = ThreadPoolExecutor(max(1, args.workers))
        try:
            futures = [executor.submit(run_one, m, p, k) for m, p, k in pending]
            for done, fut in enumerate(as_completed(futures), start=1):
                r = fut.result()
                with lock:
                    records[r["record_id"]] = r
                    write_outputs()
                if r.get("fatal_error"):
                    print(f"[{done}/{len(pending)}] {r['record_id']}: FATAL {r['fatal_error']}")
                else:
                    flag = "" if r["status"] == "completed" else f" ({r['status']})"
                    print(f"[{done}/{len(pending)}] {r['record_id']}: answer={r['normalized_answer']} gold={r['normalized_gold']} "
                          f"{'Y' if r['correct'] else 'N'}{flag}")
        except KeyboardInterrupt:
            print("\nInterrupted: saving completed calls (rerun to resume).")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    with lock:
        report = write_outputs()
    print("\n" + report)
    print(f"Saved:\n  {results_path.resolve()}\n  {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())