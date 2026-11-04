#!/usr/bin/env python3
"""Runs Direct QA, CoT and ESV on the frozen JEEBench pairs, for each solver model.

Per instance (every original and every perturbed problem):
  Phase 1  n = 4 Direct QA samples and n = 4 CoT samples at temperature 0.7 (seeds 0-3). The 4 CoT samples ARE ESV's
           4 initial solves (ESV Stage 0 uses the CoT prompt), so CoT and ESV are compared on identical samples.
  Phase 2  Only if ALL 4 CoT samples are wrong (scored against the frozen gold): the model failed consistently, with
           no signal that resampling would fix. Then:
             ESV      stages 1-5 of the pilot (independent re-solve + recognition, diagnosis, alternative strategy,
                      recovery, cross-check + adjudication), at temperature 0, starting from the sample that gave the
                      most common wrong answer; the judge grades the diagnosis and the strategy.
             RETRY    control: one CoT re-solve at temperature 0 that is told the same thing ESV is told (that
                      answer is incorrect) and nothing else. Isolates the ESV stages from the "you were wrong" signal.
           If at least one sample is correct the instance is NOT a consistent failure and ESV does not run.
  Phase 3  Optional (--extra-methods): further methods on the same consistent failures, from the same context sample,
           stored under record["extra"][name] and versioned separately (EXTRA_VERSION), so they can be iterated
           without touching phases 1-2:
             esv_v2          ESV v2: recognition, diagnosis, K = 3 strategies that must differ from the failed one,
                             one solve per strategy with concrete answer checks, known-wrong guard, majority vote
             esv_v2_k1       ablation: ESV v2 with a single strategy
             retry3          compute-matched control: 3 retries (T = 0.7), same vote as ESV v2
             retry3_matched  retry3 plus ESV v2's answer checks and known-wrong guard: differs from esv_v2 only by
                             the ESV stages (recognition, diagnosis, strategies)

Outputs: results/<model>/records.json (every call, raw responses included). Metrics and reports: analyze.py
(phases 1-2) and compare_methods.py (phase 3).

Run (from the repo root):
  python experiments/run_methods.py --limit-pairs 2                    # smoke test, all three solvers
  python experiments/run_methods.py                                    # full run (resumes automatically)
  python experiments/run_methods.py --models qwen/qwen-2.5-7b-instruct # one solver
  python experiments/run_methods.py --extra-methods esv_v2 retry3 retry3_matched   # phase 3
  python experiments/run_methods.py --samples-only --limit-pairs 30   # samples for run_open.py only
"""

import argparse
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import (  # noqa: E402
    JUDGE_MODEL, NUM_SAMPLES, SAMPLE_TEMPERATURE, SOLVER_MODELS, answer_format_note,
    answers_match, bool_value, call_stage, extract_answer, load_json, majority_answer, make_client, modal_index, model_slug,
    reasoning_setting, save_json, score_answer, utc_now, verdict_value,
)
import method_prompts as mp  # noqa: E402

# Bump whenever a prompt, sampling setting or scoring rule changes: resume only reuses records of the same version.
RUN_VERSION = "jee_methods_v1_2026-11-01"
SOLVER_MAX_TOKENS = 4000  # non-reasoning solvers; a truncated reply is retried once with the cap doubled
JUDGE_MAX_TOKENS = 8000
MAX_STRATEGY_ATTEMPTS = 2
METHODS = ("dqa", "cot")

# Phase 3. Bump EXTRA_VERSION whenever a phase-3 prompt or rule changes: stale phase-3 results are then rerun.
EXTRA_VERSION = "jee_extra_v1_2026-11-02"
EXTRA_METHODS: Dict[str, Dict[str, Any]] = {
    "esv_v2": {"kind": "esv_v2", "strategies": 3, "guard": True},
    "esv_v2_k1": {"kind": "esv_v2", "strategies": 1, "guard": True},
    "retry3": {"kind": "retry", "attempts": 3, "matched": False},
    "retry3_matched": {"kind": "retry", "attempts": 3, "matched": True},
}
# A strategy is "the same method again" when its content words overlap the failed one's this much (Jaccard).
SIMILAR_TO_FAILED, SIMILAR_TO_EACH_OTHER = 0.5, 0.6

STOP = threading.Event()  # set on an invalid API key


def make_instance(pair: Dict[str, Any], kind: str) -> Dict[str, Any]:
    gold = pair["original_gold"] if kind == "original" else pair["answer_prime"]
    return {
        "instance_id": f"{pair['id']}_{kind}",
        "pair_id": pair["id"],
        "problem_type": kind,
        "perturbation_type": pair["perturbation_type"],
        "question_type": pair["type"],
        "answer_format": pair["answer_format"],
        "source": pair.get("source"),
        "source_index": pair.get("source_index"),
        "question": pair["original_problem"] if kind == "original" else pair["perturbed_problem"],
        "format_note": answer_format_note(pair["answer_format"]),
        "gold": str(gold),
        # Frozen annotations are used only by the judge, never in solver prompts.
        "frozen_pair_annotations": {"perturbation_type": pair["perturbation_type"],
                                    **{k: pair.get(k) for k in ("original_strategy", "changed_condition", "alternative_strategy")}}
        if kind == "perturbed" else {},
    }


def is_api_failure(call: Dict[str, Any]) -> bool:
    if call["status"] != "api_error":
        return False
    if any("401" in e or "expired" in e.lower() for e in call["errors"]):
        STOP.set()
    return True


class ApiFailure(Exception):
    pass


def content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3}


def similarity(a: str, b: str) -> float:
    x, y = content_words(a), content_words(b)
    return len(x & y) / len(x | y) if x and y else 0.0


def vote(calls: List[Dict[str, Any]], answer_format: str, wrong: Optional[str]) -> Optional[str]:
    """Majority of the derivations' answers, never the known-wrong one (ties: earliest derivation)."""
    usable = [c["normalized_answer"] for c in calls if not (wrong is not None and answers_match(c["normalized_answer"], wrong, answer_format))]
    return majority_answer(usable, answer_format) or (calls[0]["normalized_answer"] if calls else None)


class Runner:
    def __init__(self, client, model: str, reasoning: Optional[Dict], judge_model: str, judge_reasoning: Optional[Dict]):
        self.client, self.model, self.reasoning = client, model, reasoning
        self.judge_model, self.judge_reasoning = judge_model, judge_reasoning

    def solver(self, stage: str, prompt: str, fields: List[str], required: List[str], temperature: float = 0.0,
               seed: Optional[int] = None) -> Dict[str, Any]:
        call = call_stage(self.client, self.model, stage, prompt, fields, required, self.reasoning, SOLVER_MAX_TOKENS,
                          temperature=temperature, seed=seed)
        if is_api_failure(call):
            raise ApiFailure(f"{stage}: {call['errors'][-1] if call['errors'] else 'api error'}")
        return call

    def scored(self, call: Dict[str, Any], inst: Dict[str, Any]) -> Dict[str, Any]:
        call["answer"] = extract_answer(call)
        call.update(score_answer(call["answer"], inst["gold"], inst["answer_format"]))
        return call

    # Phase 1 -----------------------------------------------------------------------------------------
    def sample(self, inst: Dict[str, Any], method: str, k: int) -> Dict[str, Any]:
        if method == "dqa":
            prompt, fields = mp.build_direct_prompt(inst["question"], inst["format_note"], inst["answer_format"]), mp.DIRECT_FIELDS
        else:
            prompt, fields = mp.build_cot_prompt(inst["question"], inst["format_note"], inst["answer_format"]), mp.COT_FIELDS
        call = self.solver(f"{method}_sample_{k}", prompt, fields, ["final_answer"], SAMPLE_TEMPERATURE, seed=k)
        return self.scored(call, inst)

    # Phase 2 -----------------------------------------------------------------------------------------
    @staticmethod
    def failure_context(inst: Dict[str, Any], cot: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], int]:
        """The instance plus the sample that gave the model's most common wrong answer, which every method starts from."""
        chosen = modal_index([s["normalized_answer"] for s in cot], inst["answer_format"])
        return {**inst, "initial_reasoning": cot[chosen]["parsed"].get("reasoning", ""), "initial_answer": cot[chosen]["answer"]}, chosen

    def consistent_failure(self, inst: Dict[str, Any], cot: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ESV and the retry control on an instance whose 4 CoT samples are all wrong."""
        ctx, chosen = self.failure_context(inst, cot)
        retry = self.scored(self.solver("retry_with_feedback", mp.build_retry_prompt(ctx), mp.COT_FIELDS, ["final_answer"]), inst)
        esv = self.run_esv(ctx)
        esv["context_sample"] = chosen
        return {"retry": retry, "esv": esv}

    # Phase 3 -----------------------------------------------------------------------------------------
    def guarded_solve(self, stage: str, prompt: str, fields: List[str], inst: Dict[str, Any], wrong: Optional[str],
                      guard: bool, stages: Dict[str, Any], temperature: float = 0.0, seed: Optional[int] = None) -> Dict[str, Any]:
        """One derivation; with guard, a reply that repeats the known-wrong answer gets one follow-up."""
        call = self.scored(self.solver(stage, prompt, fields, ["final_answer"], temperature, seed), inst)
        stages[stage] = call
        if guard and wrong is not None and answers_match(call["normalized_answer"], wrong, inst["answer_format"]):
            call = self.scored(self.solver(f"{stage}_guard", prompt + mp.known_wrong_followup(call["answer"]), fields,
                                           ["final_answer"], temperature, seed), inst)
            stages[f"{stage}_guard"] = call
        return call

    def finish(self, inst: Dict[str, Any], stages: Dict[str, Any], derivations: List[Dict[str, Any]], wrong: Optional[str]) -> Dict[str, Any]:
        fmt_ = inst["answer_format"]
        final = vote(derivations, fmt_, wrong)
        score = score_answer(final, inst["gold"], fmt_)
        return {
            "version": EXTRA_VERSION,
            "stages": stages,
            "initial_answer": inst["initial_answer"],
            "derivation_answers": [d["normalized_answer"] for d in derivations],
            "derivations_correct": [bool(d["correct"]) for d in derivations],
            "final_normalized": score["normalized_answer"],
            "final_correct": score["correct"],
            "final_repeated_initial": wrong is not None and answers_match(score["normalized_answer"], wrong, fmt_),
        }

    def run_retry_k(self, ctx: Dict[str, Any], attempts: int, matched: bool) -> Dict[str, Any]:
        wrong = score_answer(ctx["initial_answer"], ctx["gold"], ctx["answer_format"])["normalized_answer"]
        prompt, fields = (mp.build_matched_retry_prompt(ctx), list(mp.SOLVE_FIELDS)) if matched else (mp.build_retry_prompt(ctx), mp.COT_FIELDS)
        stages: Dict[str, Any] = {}
        derivations = [self.guarded_solve(f"retry_{k}", prompt, fields, ctx, wrong, matched, stages, SAMPLE_TEMPERATURE, seed=k)
                       for k in range(attempts)]
        return self.finish(ctx, stages, derivations, wrong)

    def propose_strategies(self, ctx: Dict[str, Any], diagnosis: Dict[str, str], k: int, stages: Dict[str, Any]) -> Tuple[List[str], int]:
        """K strategies; ones too close to the failed method or to each other are rejected and regenerated once."""
        too_similar: List[str] = []
        kept: List[str] = []
        for attempt in range(1, MAX_STRATEGY_ATTEMPTS + 1):
            call = self.solver("esv_strategies", mp.build_v2_strategy_prompt(ctx, diagnosis, k, too_similar),
                               list(mp.v2_strategy_fields(k)), ["strategy_1"])
            call["strategy_attempt"] = attempt
            stages["esv_strategies" if attempt == 1 else f"esv_strategies_{attempt}"] = call
            failed = call["parsed"].get("failed_strategy", "")
            proposed = [s for s in (call["parsed"].get(f"strategy_{i}", "").strip() for i in range(1, k + 1)) if s]
            kept, too_similar = [], []
            for s in proposed:
                if similarity(s, failed) > SIMILAR_TO_FAILED or any(similarity(s, t) > SIMILAR_TO_EACH_OTHER for t in kept):
                    too_similar.append(s)
                else:
                    kept.append(s)
            if len(kept) == k:
                return kept, attempt
        return kept or proposed or ["Solve the problem afresh with a method different from the failed one."], MAX_STRATEGY_ATTEMPTS

    def run_esv_v2(self, ctx: Dict[str, Any], k: int, guard: bool) -> Dict[str, Any]:
        fmt_ = ctx["answer_format"]
        wrong = score_answer(ctx["initial_answer"], ctx["gold"], fmt_)["normalized_answer"]
        stages: Dict[str, Any] = {}
        recognition = self.solver("esv_recognition", mp.build_v2_recognition_prompt(ctx), list(mp.V2_RECOGNITION_FIELDS),
                                  ["failure_recognized"])
        stages["esv_recognition"] = recognition
        diagnosis = self.solver("esv_diagnosis", mp.build_v2_diagnosis_prompt(ctx, recognition["parsed"]),
                                list(mp.V2_DIAGNOSIS_FIELDS), ["failed_step", "broken_condition"])
        stages["esv_diagnosis"] = diagnosis
        strategies, attempts = self.propose_strategies(ctx, diagnosis["parsed"], k, stages)
        derivations = [self.guarded_solve(f"esv_solve_{i}", mp.build_v2_solve_prompt(ctx, diagnosis["parsed"], s),
                                          list(mp.SOLVE_FIELDS), ctx, wrong, guard, stages)
                       for i, s in enumerate(strategies, start=1)]
        judge = call_stage(self.client, self.judge_model, "judge_diagnosis", mp.build_v2_judge_prompt(ctx, diagnosis["parsed"], strategies),
                           list(mp.V2_JUDGE_FIELDS), ["diagnosis_verdict", "strategy_verdicts"], self.judge_reasoning, JUDGE_MAX_TOKENS)
        if is_api_failure(judge):
            raise ApiFailure("judge_diagnosis")
        stages["judge_diagnosis"] = judge
        verdicts = re.findall(r"\b(INVALID|VALID)\b", (judge["parsed"].get("strategy_verdicts") or "").upper())
        return {
            **self.finish(ctx, stages, derivations, wrong),
            "recognized": bool_value(recognition["parsed"].get("failure_recognized")) is True,
            "diagnosis_verdict": verdict_value(judge["parsed"].get("diagnosis_verdict"), ("INCORRECT", "CORRECT", "PARTIAL")),
            "strategies": strategies,
            "strategy_attempts": attempts,
            "strategy_verdicts": verdicts[: len(strategies)],
        }

    def extra(self, name: str, inst: Dict[str, Any], cot: List[Dict[str, Any]]) -> Dict[str, Any]:
        config = EXTRA_METHODS[name]
        ctx, chosen = self.failure_context(inst, cot)
        if config["kind"] == "esv_v2":
            out = self.run_esv_v2(ctx, config["strategies"], config["guard"])
        else:
            out = self.run_retry_k(ctx, config["attempts"], config["matched"])
        return {**out, "config": config, "context_sample": chosen}

    def run_esv(self, inst: Dict[str, Any]) -> Dict[str, Any]:
        """The pilot's five ESV stages, unchanged; every stage runs on every consistent failure."""
        stages: Dict[str, Dict[str, Any]] = {}
        fmt_ = inst["answer_format"]
        wrong = score_answer(inst["initial_answer"], inst["gold"], fmt_)["normalized_answer"]

        independent = self.scored(self.solver("esv_independent_solve", mp.build_independent_prompt(inst),
                                              list(mp.INDEPENDENT_FIELDS), ["final_answer"]), inst)
        stages["esv_independent_solve"] = independent
        answers_disagree = independent["answer_readable"] and wrong is not None and not answers_match(independent["normalized_answer"], wrong, fmt_)

        recognition = self.solver("esv_recognition", mp.build_recognition_prompt(inst, independent["parsed"]),
                                  list(mp.RECOGNITION_FIELDS), ["failure_recognized"])
        stages["esv_recognition"] = recognition
        model_flagged = bool_value(recognition["parsed"].get("failure_recognized")) is True

        diagnosis = self.solver("esv_diagnosis", mp.build_diagnosis_prompt(inst, recognition["parsed"]),
                                list(mp.DIAGNOSIS_FIELDS), ["failed_step", "root_cause"])
        stages["esv_diagnosis"] = diagnosis

        rejected: List[Dict[str, str]] = []
        strategy: Dict[str, Any] = {}
        for attempt in range(1, MAX_STRATEGY_ATTEMPTS + 1):
            strategy = self.solver("esv_alternative_strategy", mp.build_strategy_prompt(inst, diagnosis["parsed"], rejected),
                                   list(mp.STRATEGY_FIELDS), ["alternative_strategy", "strategy_verified"])
            strategy["strategy_attempt"] = attempt
            if rejected:
                strategy["rejected_strategies"] = rejected
            if bool_value(strategy["parsed"].get("strategy_verified")) is not False:
                break
            rejected.append(strategy["parsed"])
        stages["esv_alternative_strategy"] = strategy

        recovery = self.scored(self.solver("esv_recovery", mp.build_recovery_prompt(inst, diagnosis["parsed"], strategy["parsed"]),
                                           list(mp.RECOVERY_FIELDS), ["final_answer"]), inst)
        stages["esv_recovery"] = recovery
        cross = self.scored(self.solver("esv_cross_check", mp.build_cross_check_prompt(inst, diagnosis["parsed"], strategy["parsed"]),
                                        list(mp.RECOVERY_FIELDS), ["final_answer"]), inst)
        stages["esv_cross_check"] = cross
        agreement = recovery["answer_readable"] and answers_match(cross["normalized_answer"], recovery["normalized_answer"], fmt_)
        if agreement:
            final_answer = recovery["answer"]
        else:
            solutions = [s["parsed"] or {"final_answer": s["answer"]} for s in (recovery, cross) if s["answer"]]
            final = self.scored(self.solver("esv_final_answer", mp.build_final_prompt(inst, solutions),
                                            list(mp.FINAL_FIELDS), ["final_answer"]), inst)
            stages["esv_final_answer"] = final
            # If adjudication yields no answer, keep the recovered one rather than abstaining.
            final_answer = final["answer"] or recovery["answer"] or cross["answer"]
        final_score = score_answer(final_answer, inst["gold"], fmt_)

        judge = call_stage(self.client, self.judge_model, "judge_diagnosis", mp.build_judge_prompt(inst, diagnosis["parsed"], strategy["parsed"]),
                           list(mp.JUDGE_FIELDS), ["diagnosis_verdict", "strategy_verdict"], self.judge_reasoning, JUDGE_MAX_TOKENS)
        if is_api_failure(judge):
            raise ApiFailure("judge_diagnosis")
        stages["judge_diagnosis"] = judge

        return {
            "stages": stages,
            "initial_answer": inst["initial_answer"],
            "recognized": model_flagged or answers_disagree,
            "recognized_by_strategy_check": model_flagged,
            "recognized_by_disagreement": answers_disagree,
            "independent_correct": independent["correct"],
            "diagnosis_verdict": verdict_value(judge["parsed"].get("diagnosis_verdict"), ("INCORRECT", "CORRECT", "PARTIAL")),
            "strategy_verdict": verdict_value(judge["parsed"].get("strategy_verdict"), ("INVALID", "VALID")),
            "strategy_attempts": strategy.get("strategy_attempt"),
            "strategy_self_verified": bool_value(strategy["parsed"].get("strategy_verified")),
            "recovery_answer": recovery["answer"],
            "recovery_correct": recovery["correct"],
            "cross_check_correct": cross["correct"],
            "derivations_agree": agreement,
            "final_answer": final_answer,
            "final_normalized": final_score["normalized_answer"],
            "final_correct": final_score["correct"],
            "final_repeated_initial": wrong is not None and answers_match(final_score["normalized_answer"], wrong, fmt_),
        }


def run_model(client, model: str, pairs: List[Dict[str, Any]], args) -> Dict[str, Any]:
    out_dir = common.RESULTS_DIR / model_slug(model)
    path = out_dir / "records.json"
    if args.no_resume and path.exists():  # never silently discard a finished run
        backup = path.with_name(f"records.backup-{time.strftime('%Y%m%d-%H%M%S')}.json")
        path.replace(backup)
        print(f"  --no-resume: previous results moved to {backup}")
    saved = {} if args.no_resume else (load_json(path, {}) or {}).get("instances", {})
    instances = [make_instance(p, kind) for p in pairs for kind in ("original", "perturbed")]
    records: Dict[str, Dict[str, Any]] = {}
    for inst in instances:
        old = saved.get(inst["instance_id"])
        reusable = old and old.get("run_version") == RUN_VERSION and old.get("model") == model and old.get("gold") == inst["gold"]
        records[inst["instance_id"]] = old if reusable else {
            **{k: v for k, v in inst.items() if k not in ("question", "format_note", "frozen_pair_annotations")},
            "model": model, "run_version": RUN_VERSION, "samples": {m: [None] * NUM_SAMPLES for m in METHODS},
            "consistent_failure": None, "esv": None, "retry": None,
        }
    metadata = {"model": model, "run_version": RUN_VERSION, "judge_model": args.judge_model, "num_samples": NUM_SAMPLES,
                "sample_temperature": SAMPLE_TEMPERATURE, "solver_max_tokens": SOLVER_MAX_TOKENS, "reasoning": args.reasoning,
                "frozen_dataset": str(common.FROZEN_DATASET), "pairs": len(pairs)}
    lock = threading.Lock()
    last_save = [time.time()]

    # Saved records outside this run (e.g. beyond --limit-pairs) are written back untouched, never dropped.
    untouched = {iid: r for iid, r in saved.items() if iid not in records}

    def save(force: bool = False) -> None:
        if force or time.time() - last_save[0] > 20:
            save_json(path, {"metadata": {**metadata, "saved_at": utc_now()}, "instances": {**untouched, **records}})
            last_save[0] = time.time()

    runner = Runner(client, model, reasoning_setting(args.reasoning), args.judge_model, reasoning_setting(args.judge_reasoning))
    by_id = {i["instance_id"]: i for i in instances}

    def guarded(fn, *a):
        if STOP.is_set():
            return None
        try:
            return fn(*a)
        except ApiFailure as exc:
            print(f"    API failure ({exc}); will retry on the next run", flush=True)
        except Exception as exc:  # one broken call must not kill the run
            print(f"    unexpected error {type(exc).__name__}: {exc}", flush=True)
        return None

    # Phase 1: samples
    jobs = [(iid, m, k) for iid, r in records.items() for m in METHODS for k in range(NUM_SAMPLES) if r["samples"][m][k] is None]
    print(f"\n=== {model} | {len(instances)} instances | phase 1: {len(jobs)} sample calls to run ===", flush=True)
    with ThreadPoolExecutor(max(1, args.workers)) as ex:
        futures = {ex.submit(guarded, runner.sample, by_id[iid], m, k): (iid, m, k) for iid, m, k in jobs}
        for done, fut in enumerate(as_completed(futures), start=1):
            iid, m, k = futures[fut]
            call = fut.result()
            if call is not None:
                with lock:
                    records[iid]["samples"][m][k] = call
                    save()
            if done % 50 == 0 or done == len(jobs):
                print(f"  phase 1: {done}/{len(jobs)}", flush=True)
    save(force=True)
    if STOP.is_set() or args.samples_only:
        return records

    # Phase 2: ESV + retry on consistent failures
    todo = []
    for iid, r in records.items():
        cot = r["samples"]["cot"]
        if any(s is None for s in cot):
            continue  # incomplete samples (API failures): decided on the next run
        r["consistent_failure"] = not any(s["correct"] for s in cot)
        if r["consistent_failure"] and r["esv"] is None:
            todo.append(iid)
    n_fail = sum(r["consistent_failure"] is True for r in records.values())
    print(f"  consistent failures (all {NUM_SAMPLES} CoT samples wrong): {n_fail} | phase 2 (ESV + retry) to run: {len(todo)}", flush=True)

    def phase2(iid: str) -> Tuple[str, Dict[str, Any]]:
        return iid, runner.consistent_failure(by_id[iid], records[iid]["samples"]["cot"])

    with ThreadPoolExecutor(max(1, args.workers)) as ex:
        futures = [ex.submit(guarded, phase2, iid) for iid in todo]
        for done, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            if result is None:
                continue
            iid, out = result
            with lock:
                records[iid]["esv"], records[iid]["retry"] = out["esv"], out["retry"]
                save()
            e = out["esv"]
            print(f"  [{done}/{len(todo)}] {iid}: recognized={e['recognized']} diagnosis={e['diagnosis_verdict']} "
                  f"strategy={e['strategy_verdict']} ESV={'Y' if e['final_correct'] else 'N'} | retry={'Y' if out['retry']['correct'] else 'N'}",
                  flush=True)
    save(force=True)
    if STOP.is_set() or not args.extra_methods:
        return records

    # Phase 3: extra methods on the same consistent failures
    jobs3 = [(iid, name) for iid, r in records.items() if r["consistent_failure"] and r["esv"] is not None
             for name in args.extra_methods if (r.get("extra") or {}).get(name, {}).get("version") != EXTRA_VERSION]
    print(f"  phase 3 ({', '.join(args.extra_methods)}): {len(jobs3)} runs to do", flush=True)
    with ThreadPoolExecutor(max(1, args.workers)) as ex:
        futures = {ex.submit(guarded, runner.extra, name, by_id[iid], records[iid]["samples"]["cot"]): (iid, name) for iid, name in jobs3}
        for done, fut in enumerate(as_completed(futures), start=1):
            iid, name = futures[fut]
            out = fut.result()
            if out is None:
                continue
            with lock:
                records[iid].setdefault("extra", {})[name] = out
                save()
            print(f"  [{done}/{len(jobs3)}] {iid} {name}: answers={out['derivation_answers']} "
                  f"final={out['final_normalized']} {'Y' if out['final_correct'] else 'N'}", flush=True)
    save(force=True)
    return records


def main(argv: Optional[List[str]] = None, client=None) -> int:
    parser = argparse.ArgumentParser(description="Direct QA, CoT and ESV (n = 4 initial solves) on the frozen pairs of a benchmark.")
    common.add_benchmark_arg(parser)
    parser.add_argument("--models", nargs="+", default=list(SOLVER_MODELS))
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--reasoning", default="default", choices=["default", "off", "low", "medium", "high"],
                        help="Solver reasoning setting ('default' sends nothing; required for non-reasoning models)")
    parser.add_argument("--judge-reasoning", default="medium", choices=["default", "off", "low", "medium", "high"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--samples-only", action="store_true",
                        help="Phase 1 only (the DQA and CoT samples that run_open.py starts from); no oracle-gated ESV")
    parser.add_argument("--extra-methods", nargs="*", default=[], choices=list(EXTRA_METHODS),
                        help="Phase 3: further methods on the consistent failures (see the module docstring)")
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
        if STOP.is_set():
            print("\nStopped: the API key was rejected (401/expired). Fix the key and rerun to resume.")
            return 1
    print("\nDone. Build the report with: python experiments/analyze.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
