#!/usr/bin/env python3
"""Experiment 3, gold-free evaluation: ESV runs on EVERY instance, and the gold answer is used only to score the end.

Starts from the 4 CoT samples already in results/<model>/records.json (run_methods.py phase 1); no new samples. Per
instance (every original and every perturbed problem):
  initial      the CoT majority answer (= CoT maj@4) and the reasoning of the first sample that gave it
  esv          ESV v2 without an oracle: blind recognition (stages 2-3). If it finds no failure, the initial answer is
               kept. Otherwise diagnosis, K = 3 strategies, one solve per strategy with answer checks, and a vote in
               which the initial answer also has one vote and wins ties: ESV switches only when some new answer
               gets more votes than every other. The later stages are told the failure was self-flagged, never that
               it is confirmed, so there is no known-wrong guard.
  recheck3     control: the SAME recognition decision, then 3 re-solves (T = 0.7) with the same answer checks and
               the same vote. Differs from esv only by the diagnosis and the strategies.
  self_refine  control (Self-Refine style): one "review your solution, correct it if wrong" call with the same answer
               checks; no strategy verification, diagnosis or strategies.
A method can therefore fix a wrong initial answer AND break a correct one; analyze_open.py counts both. The judge
grades the diagnosis and strategies only where ESV flagged an initial answer that is in fact wrong: a measurement
of the stages, run after every answer is final, that never affects one.

Output: results/<model>/open_records.json (records.json is only read). Report: analyze_open.py.

Run (from the repo root):
  python experiments/run_open.py --limit-pairs 30        # pilot: first 30 pairs, all three solvers
  python experiments/run_open.py                         # all pairs (resumes automatically)
  python experiments/run_open.py --models google/gemma-3-4b-it --workers 3
"""

import argparse
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import (  # noqa: E402
    JUDGE_MODEL, SAMPLE_TEMPERATURE, SOLVER_MODELS, answers_match, bool_value, call_stage,
    load_json, make_client, majority_answer, modal_index, model_slug, reasoning_setting, save_json, score_answer, utc_now,
    verdict_value,
)
import method_prompts as mp  # noqa: E402
import run_methods as rm  # noqa: E402

# Bump whenever a prompt, sampling setting or decision rule changes: resume only reuses records of the same version.
OPEN_VERSION = "jee_open_v1_2026-11-03"
STRATEGIES = 3
RETRIES = 3
METHODS = ("esv", "recheck3", "self_refine")
# Final vote of esv and recheck3. Changing it needs no new calls: rescore() recomputes the finals from the stored
# derivations of any record decided under another rule.
VOTE_RULE = "initial_votes_v1_2026-11-03"
VOTED = ("esv", "recheck3")


def decide(initial: Optional[str], answers: List[Optional[str]], answer_format: str) -> Optional[str]:
    """The initial answer votes first, so it wins every tie (all answers differ, or 2 against 2)."""
    return majority_answer([initial, *answers], answer_format)


def final_fields(gold: str, answer_format: str, initial: Dict[str, Any], final_answer: Optional[str]) -> Dict[str, Any]:
    """Scores a final answer against gold, only here. An empty final answer keeps the initial one."""
    final = final_answer if final_answer else initial["answer"]
    score = score_answer(final, gold, answer_format)
    return {"final_answer": final, "final_normalized": score["normalized_answer"], "final_correct": score["correct"],
            "changed": not answers_match(score["normalized_answer"], initial["normalized"], answer_format)}


def outcome(inst: Dict[str, Any], initial: Dict[str, Any], flagged: bool, final_answer: Optional[str],
            derivations: List[Dict[str, Any]], stages: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "flagged": flagged,
        "stages": stages,
        "derivation_answers": [d["normalized_answer"] for d in derivations],
        "derivations_correct": [bool(d["correct"]) for d in derivations],
        **final_fields(inst["gold"], inst["answer_format"], initial, final_answer),
    }


def rescore(record: Dict[str, Any]) -> Dict[str, Any]:
    """Re-decides esv and recheck3 under VOTE_RULE from the stored derivations (no calls); unflagged runs keep the initial."""
    if record.get("vote_rule") != VOTE_RULE:
        init = record["initial"]
        for m in VOTED:
            x = record["methods"][m]
            final = decide(init["normalized"], x["derivation_answers"], record["answer_format"]) if record["flagged"] else None
            x.update(final_fields(record["gold"], record["answer_format"], init, final))
        record["vote_rule"] = VOTE_RULE
    return record


class OpenRunner(rm.Runner):
    def run(self, inst: Dict[str, Any], cot: List[Dict[str, Any]]) -> Dict[str, Any]:
        fmt_ = inst["answer_format"]
        chosen = modal_index([s["normalized_answer"] for s in cot], fmt_)
        ctx = {**inst, "initial_reasoning": cot[chosen]["parsed"].get("reasoning", ""), "initial_answer": cot[chosen]["answer"]}
        ctx["failure_note"] = mp.self_flagged_note(ctx)
        start = score_answer(ctx["initial_answer"], inst["gold"], fmt_)
        initial = {"context_sample": chosen, "answer": ctx["initial_answer"], "normalized": start["normalized_answer"],
                   "correct": start["correct"]}

        recognition = self.solver("esv_recognition", mp.build_v2_recognition_prompt(ctx), list(mp.V2_RECOGNITION_FIELDS),
                                  ["failure_recognized"])
        flagged = bool_value(recognition["parsed"].get("failure_recognized")) is True  # unreadable decision: keep

        esv_stages: Dict[str, Any] = {}
        esv_derivations: List[Dict[str, Any]] = []
        diagnosis: Dict[str, str] = {}
        strategies: List[str] = []
        attempts = 0
        if flagged:
            call = self.solver("esv_diagnosis", mp.build_v2_diagnosis_prompt(ctx, recognition["parsed"]),
                               list(mp.V2_DIAGNOSIS_FIELDS), ["failed_step", "broken_condition"])
            esv_stages["esv_diagnosis"] = call
            diagnosis = call["parsed"]
            strategies, attempts = self.propose_strategies(ctx, diagnosis, STRATEGIES, esv_stages)
            esv_derivations = [self.guarded_solve(f"esv_solve_{i}", mp.build_v2_solve_prompt(ctx, diagnosis, s),
                                                  list(mp.SOLVE_FIELDS), ctx, None, False, esv_stages)
                               for i, s in enumerate(strategies, start=1)]
        esv = outcome(inst, initial, flagged, decide(initial["normalized"], [d["normalized_answer"] for d in esv_derivations], fmt_)
                      if flagged else None, esv_derivations, esv_stages)
        esv.update(strategies=strategies, strategy_attempts=attempts)

        retry_stages: Dict[str, Any] = {}
        retries: List[Dict[str, Any]] = []
        if flagged:
            retries = [self.guarded_solve(f"recheck_{k}", mp.build_matched_retry_prompt(ctx), list(mp.SOLVE_FIELDS), ctx,
                                          None, False, retry_stages, SAMPLE_TEMPERATURE, seed=k)
                       for k in range(RETRIES)]
        recheck3 = outcome(inst, initial, flagged, decide(initial["normalized"], [d["normalized_answer"] for d in retries], fmt_)
                           if flagged else None, retries, retry_stages)

        refine = self.scored(self.solver("self_refine", mp.build_self_refine_prompt(ctx), list(mp.SELF_REFINE_FIELDS),
                                         ["final_answer"]), inst)
        verdict = verdict_value(refine["parsed"].get("verdict"), ("INCORRECT", "CORRECT"))
        self_refine = outcome(inst, initial, verdict == "INCORRECT", refine["answer"], [refine], {"self_refine": refine})
        self_refine["verdict"] = verdict

        judge = None
        if flagged and not initial["correct"]:  # measurement only: every answer above is already final
            judge = call_stage(self.client, self.judge_model, "judge_diagnosis", mp.build_v2_judge_prompt(ctx, diagnosis, strategies),
                               list(mp.V2_JUDGE_FIELDS), ["diagnosis_verdict", "strategy_verdicts"], self.judge_reasoning,
                               rm.JUDGE_MAX_TOKENS)
            if rm.is_api_failure(judge):
                raise rm.ApiFailure("judge_diagnosis")
            verdicts = re.findall(r"\b(INVALID|VALID)\b", (judge["parsed"].get("strategy_verdicts") or "").upper())
            esv.update(diagnosis_verdict=verdict_value(judge["parsed"].get("diagnosis_verdict"), ("INCORRECT", "CORRECT", "PARTIAL")),
                       strategy_verdicts=verdicts[: len(strategies)])

        return {
            "instance_id": inst["instance_id"], "pair_id": inst["pair_id"], "problem_type": inst["problem_type"],
            "perturbation_type": inst["perturbation_type"], "answer_format": inst["answer_format"], "gold": inst["gold"],
            "model": self.model, "version": OPEN_VERSION, "vote_rule": VOTE_RULE,
            "initial": initial, "recognition": recognition, "flagged": flagged,
            "methods": {"esv": esv, "recheck3": recheck3, "self_refine": self_refine},
            "judge": judge,
        }


def run_model(client, model: str, pairs: List[Dict[str, Any]], args) -> None:
    out_dir = common.RESULTS_DIR / model_slug(model)
    base = (load_json(out_dir / "records.json", {}) or {}).get("instances", {})
    path = out_dir / "open_records.json"
    saved = (load_json(path, {}) or {}).get("instances", {})
    instances = [rm.make_instance(p, kind) for p in pairs for kind in ("original", "perturbed")]
    by_id = {i["instance_id"]: i for i in instances}
    records = {iid: rescore(r) for iid, r in saved.items() if r.get("version") == OPEN_VERSION and iid in by_id}
    untouched = {iid: r for iid, r in saved.items() if iid not in by_id}  # outside --limit-pairs: kept as is
    missing = [iid for iid in by_id if any(s is None for s in (base.get(iid) or {}).get("samples", {}).get("cot", [None]))]
    todo = [iid for iid in by_id if iid not in records and iid not in missing]
    print(f"\n=== {model} | {len(instances)} instances | done {len(records)} | to run {len(todo)}"
          + (f" | {len(missing)} skipped: CoT samples incomplete (run run_methods.py first)" if missing else ""), flush=True)

    metadata = {"model": model, "version": OPEN_VERSION, "vote_rule": VOTE_RULE, "judge_model": args.judge_model, "strategies": STRATEGIES,
                "retries": RETRIES, "frozen_dataset": str(common.FROZEN_DATASET), "pairs": len(pairs)}
    lock = threading.Lock()
    last_save = [time.time()]

    def save(force: bool = False) -> None:
        if force or time.time() - last_save[0] > 20:
            save_json(path, {"metadata": {**metadata, "saved_at": utc_now()}, "instances": {**untouched, **records}})
            last_save[0] = time.time()

    runner = OpenRunner(client, model, reasoning_setting(args.reasoning), args.judge_model, reasoning_setting(args.judge_reasoning))

    def job(iid: str) -> Optional[Dict[str, Any]]:
        if rm.STOP.is_set():
            return None
        try:
            return runner.run(by_id[iid], base[iid]["samples"]["cot"])
        except rm.ApiFailure as exc:
            print(f"    API failure ({exc}); will retry on the next run", flush=True)
        except Exception as exc:  # one broken call must not kill the run
            print(f"    unexpected error {type(exc).__name__}: {exc}", flush=True)
        return None

    yn = lambda ok: "Y" if ok else "N"
    with ThreadPoolExecutor(max(1, args.workers)) as ex:
        futures = {ex.submit(job, iid): iid for iid in todo}
        for done, fut in enumerate(as_completed(futures), start=1):
            out = fut.result()
            if out is None:
                continue
            with lock:
                records[out["instance_id"]] = out
                save()
            m = out["methods"]
            print(f"  [{done}/{len(todo)}] {out['instance_id']}: initial={out['initial']['normalized']} {yn(out['initial']['correct'])} "
                  f"flagged={yn(out['flagged'])} | esv={m['esv']['final_normalized']} {yn(m['esv']['final_correct'])} "
                  f"recheck3={yn(m['recheck3']['final_correct'])} self_refine={yn(m['self_refine']['final_correct'])}", flush=True)
    save(force=True)


def main(argv: Optional[List[str]] = None, client=None) -> int:
    parser = argparse.ArgumentParser(description="Gold-free evaluation: ESV and its controls on every instance.")
    common.add_benchmark_arg(parser)
    parser.add_argument("--models", nargs="+", default=list(SOLVER_MODELS))
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--reasoning", default="default", choices=["default", "off", "low", "medium", "high"])
    parser.add_argument("--judge-reasoning", default="medium", choices=["default", "off", "low", "medium", "high"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-pairs", type=int, default=None)
    args = parser.parse_args(argv)
    common.select_benchmark(args.benchmark)

    dataset = load_json(common.FROZEN_DATASET)
    if not dataset or not dataset.get("pairs"):
        print(f"ERROR: no pairs in {common.FROZEN_DATASET}; run build_dataset.py --benchmark {common.BENCHMARK} first.", file=sys.stderr)
        return 2
    pairs = dataset["pairs"][: args.limit_pairs] if args.limit_pairs else dataset["pairs"]
    client = client or make_client()
    for model in args.models:
        run_model(client, model, pairs, args)
        if rm.STOP.is_set():
            print("\nStopped: the API key was rejected (401/expired). Fix the key and rerun to resume.")
            return 1
    print("\nDone. Build the report with: python experiments/analyze_open.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
