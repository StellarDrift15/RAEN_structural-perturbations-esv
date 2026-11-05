#!/usr/bin/env python3
"""ESV v3, gold-free, built stage by stage (design: results/analysis/esv_bottleneck_analysis.md).

Starts from the 4 CoT samples in results/<model>/records.json; no new samples. Per instance:
  stage 1     group the 4 CoT answers into distinct candidates. If they all agree (fewer than 2 readable candidates),
              keep the CoT majority answer: no call.
  stages 2-4  one call (T = 0) on the distinct candidates, one representative solution each, in order of first
              appearance and without vote counts: per candidate the strategy, its bindings to the problem text
              (MATCHES / MISMATCH), the method's conditions (HOLDS / VIOLATED), then the divergence point, then SOUND or
              FLAWED with the failure type (BINDING / METHOD / COMPUTATION), the first flawed step and the corrected fact.
  decision    (code, no call) a candidate counts as SOUND only if its verdict is SOUND and it lists no MISMATCH and no
              VIOLATED. Exactly one SOUND -> choose it; several -> the one with most votes (then the initial answer);
              none -> stage 4, NOT BUILT YET: the initial answer is kept.
  usc         control, Universal Self-Consistency: the same disagreeing instances, all 4 samples (duplicates
              included), one "select the most consistent response" call.
The gold answer is used only to score the final answers and the candidates (for the report), never in a decision.

Output: results/<model>/esv3_records.json. Report: analyze_esv3.py.

Run (from the repo root):
  python experiments/run_esv3.py --limit-pairs 30      # dev pilot, the three current solvers
  python experiments/run_esv3.py --limit-pairs 60 --verifier-model openai/gpt-oss-120b   # independent-verifier test
                                                       # (writes esv3_records.<verifier>.json next to the self-verify file)
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
    SOLVER_MODELS, answers_match, final_fields, load_json, majority_answer, make_client, modal_index,
    model_slug, reasoning_setting, save_json, score_answer, utc_now, verdict_value,
)
import method_prompts as mp  # noqa: E402
import run_methods as rm  # noqa: E402

# Bump when a prompt or sampling setting changes: resume only reuses records of the same version.
V3_VERSION = "jee_esv3_v1_2026-11-03"
# The decision is recomputed from the stored verdicts on every load, so changing it needs no calls.
DECISION_RULE = "stages1-3_strict_v1_2026-11-03"
METHODS = ("esv3", "usc")


def candidates_of(cot: List[Dict[str, Any]], answer_format: str) -> List[Dict[str, Any]]:
    """Distinct readable CoT answers in order of first appearance, with their vote counts."""
    initial = majority_answer([s["normalized_answer"] for s in cot], answer_format)
    out: List[Dict[str, Any]] = []
    for i, s in enumerate(cot):
        a = s["normalized_answer"]
        if a is None:
            continue
        for c in out:
            if answers_match(c["normalized"], a, answer_format):
                c["votes"] += 1
                break
        else:
            out.append({"normalized": a, "answer": s["answer"], "votes": 1, "sample_index": i})
    for c in out:
        c["is_initial"] = answers_match(c["normalized"], initial, answer_format)
    return out


def parse_checks(parsed: Dict[str, str], n: int) -> List[Dict[str, Any]]:
    out = []
    for i in range(1, n + 1):
        get = lambda name: parsed.get(f"s{i}_{name}", "") or ""
        out.append({
            "verdict": verdict_value(get("verdict"), ("FLAWED", "SOUND")),
            "failure_type": verdict_value(get("failure_type"), ("COMPUTATION", "BINDING", "METHOD")),
            "matches": len(re.findall(r"\bMATCHES\b", get("bindings").upper())),
            "mismatches": len(re.findall(r"\bMISMATCH(?:ES)?\b", get("bindings").upper())),
            "holds": len(re.findall(r"\bHOLDS\b", get("conditions").upper())),
            "violations": len(re.findall(r"\bVIOLATED\b", get("conditions").upper())),
            "flawed_step": get("flawed_step"),
            "corrected_fact": get("corrected_fact"),
        })
    return out


def is_sound(check: Dict[str, Any], strict: bool = True) -> bool:
    """SOUND as the prompt defines it: the verdict, and (strict) no MISMATCH binding and no VIOLATED condition."""
    return check["verdict"] == "SOUND" and (not strict or (check["mismatches"] == 0 and check["violations"] == 0))


def decide(candidates: List[Dict[str, Any]], checks: List[Dict[str, Any]], strict: bool = True) -> Tuple[str, Optional[int]]:
    """(decision, index of the chosen candidate or None = keep the initial answer). Pure: no gold, no calls."""
    if len(candidates) < 2:
        return "unanimous", None
    if not checks or all(c["verdict"] is None for c in checks):
        return "unreadable", None
    sound = [i for i, c in enumerate(checks) if is_sound(c, strict)]
    if len(sound) == 1:
        return "one_sound", sound[0]
    if sound:
        return "several_sound", max(sound, key=lambda i: (candidates[i]["votes"], candidates[i]["is_initial"], -i))
    return "none_sound", None  # stage 4 (diagnosis-driven repair) is not built yet


def usc_choice(call: Optional[Dict[str, Any]], cot: List[Dict[str, Any]]) -> Optional[int]:
    match = re.search(r"\d+", ((call or {}).get("parsed") or {}).get("selected_response", ""))
    index = int(match.group(0)) - 1 if match else -1
    return index if 0 <= index < len(cot) else None


def redecide(record: Dict[str, Any]) -> Dict[str, Any]:
    """Recomputes both finals from the stored calls under DECISION_RULE (no calls)."""
    fmt_, gold, init = record["answer_format"], record["gold"], record["initial"]
    cands = record["candidates"]
    e = record["methods"]["esv3"]
    e["decision"], e["chosen"] = decide(cands, e["checks"])
    e.update(final_fields(gold, fmt_, init, cands[e["chosen"]]["answer"] if e["chosen"] is not None else None))
    u = record["methods"]["usc"]
    pick = usc_choice(u["call"], record["cot_answers"]) if u["call"] else None
    u["chosen_sample"] = pick
    u.update(final_fields(gold, fmt_, init, record["cot_answers"][pick]["answer"] if pick is not None else None))
    record["decision_rule"] = DECISION_RULE
    return record


class V3Runner(rm.Runner):
    # None = self-verification (the solver checks its own candidates). Set to another model to test whether an
    # independent checker separates right from wrong candidates better than the solver itself does.
    verifier_model: Optional[str] = None
    verifier_reasoning: Optional[Dict[str, Any]] = None

    def verify_call(self, prompt: str, fields: List[str], required: List[str]) -> Dict[str, Any]:
        if not self.verifier_model:
            return self.solver("esv3_verify", prompt, fields, required)
        call = rm.call_stage(self.client, self.verifier_model, "esv3_verify", prompt, fields, required,
                             self.verifier_reasoning, rm.SOLVER_MAX_TOKENS)
        if rm.is_api_failure(call):
            raise rm.ApiFailure(f"esv3_verify[{self.verifier_model}]: {call['errors'][-1] if call['errors'] else 'api error'}")
        return call

    def run(self, inst: Dict[str, Any], cot: List[Dict[str, Any]]) -> Dict[str, Any]:
        fmt_ = inst["answer_format"]
        chosen = modal_index([s["normalized_answer"] for s in cot], fmt_)
        start = score_answer(cot[chosen]["answer"], inst["gold"], fmt_)
        initial = {"context_sample": chosen, "answer": cot[chosen]["answer"], "normalized": start["normalized_answer"],
                   "correct": start["correct"]}
        cands = candidates_of(cot, fmt_)
        verify = usc = None
        checks: List[Dict[str, Any]] = []
        if len(cands) >= 2:
            solutions = [{"reasoning": cot[c["sample_index"]]["parsed"].get("reasoning", ""), "answer": c["answer"]} for c in cands]
            fields = list(mp.v3_verify_fields(len(cands)))
            verify = self.verify_call(mp.build_v3_verify_prompt(inst, solutions), fields,
                                      [f"s{i}_verdict" for i in range(1, len(cands) + 1)])
            checks = parse_checks(verify["parsed"], len(cands))
            if not self.verifier_model:  # USC stays a self-model control; rerunning it would just duplicate records
                responses = [{"reasoning": s["parsed"].get("reasoning", ""), "answer": s["answer"]} for s in cot]
                usc = self.solver("usc", mp.build_usc_prompt(inst, responses), list(mp.USC_FIELDS), ["selected_response"])
        for c in cands:  # for the report only: scored after every call is made
            c["correct"] = score_answer(c["answer"], inst["gold"], fmt_)["correct"]
        record = {
            "instance_id": inst["instance_id"], "pair_id": inst["pair_id"], "problem_type": inst["problem_type"],
            "perturbation_type": inst["perturbation_type"], "answer_format": fmt_, "gold": inst["gold"],
            "model": self.model, "verifier_model": self.verifier_model, "version": V3_VERSION, "initial": initial, "candidates": cands,
            "cot_answers": [{"answer": s["answer"], "normalized": s["normalized_answer"]} for s in cot],
            "methods": {"esv3": {"call": verify, "checks": checks}, "usc": {"call": usc}},
        }
        return redecide(record)


def run_model(client, model: str, pairs: List[Dict[str, Any]], args) -> None:
    out_dir = common.RESULTS_DIR / model_slug(model)
    base = (load_json(out_dir / "records.json", {}) or {}).get("instances", {})
    vslug = model_slug(args.verifier_model) if args.verifier_model else None
    path = out_dir / (f"esv3_records.{vslug}.json" if vslug else "esv3_records.json")
    saved = (load_json(path, {}) or {}).get("instances", {})
    instances = [rm.make_instance(p, kind) for p in pairs for kind in ("original", "perturbed")]
    by_id = {i["instance_id"]: i for i in instances}
    records = {iid: redecide(r) for iid, r in saved.items() if r.get("version") == V3_VERSION and iid in by_id}
    untouched = {iid: r for iid, r in saved.items() if iid not in by_id}
    missing = [iid for iid in by_id if any(s is None for s in (base.get(iid) or {}).get("samples", {}).get("cot", [None]))]
    todo = [iid for iid in by_id if iid not in records and iid not in missing]
    print(f"\n=== {model} | {len(instances)} instances | done {len(records)} | to run {len(todo)}"
          + (f" | {len(missing)} skipped: CoT samples incomplete (run run_methods.py --samples-only first)" if missing else ""), flush=True)

    metadata = {"model": model, "verifier_model": args.verifier_model, "version": V3_VERSION,
                "decision_rule": DECISION_RULE, "frozen_dataset": str(common.FROZEN_DATASET), "pairs": len(pairs)}
    lock = threading.Lock()
    last_save = [time.time()]

    def save(force: bool = False) -> None:
        if force or time.time() - last_save[0] > 20:
            save_json(path, {"metadata": {**metadata, "saved_at": utc_now()}, "instances": {**untouched, **records}})
            last_save[0] = time.time()

    runner = V3Runner(client, model, reasoning_setting(args.reasoning))
    runner.verifier_model = args.verifier_model
    runner.verifier_reasoning = reasoning_setting(args.verifier_reasoning)

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
            e, u = out["methods"]["esv3"], out["methods"]["usc"]
            answers = "/".join(f"{c['normalized']}x{c['votes']}" for c in out["candidates"]) or "-"
            print(f"  [{done}/{len(todo)}] {out['instance_id']}: candidates {answers} initial={out['initial']['normalized']} "
                  f"{yn(out['initial']['correct'])} | esv3 {e['decision']} -> {e['final_normalized']} {yn(e['final_correct'])} | "
                  f"usc -> {u['final_normalized']} {yn(u['final_correct'])}", flush=True)
    save(force=True)


def main(argv: Optional[List[str]] = None, client=None) -> int:
    parser = argparse.ArgumentParser(description="ESV v3 (gold-free) and Universal Self-Consistency.")
    common.add_benchmark_arg(parser)
    parser.add_argument("--models", nargs="+", default=list(SOLVER_MODELS))
    parser.add_argument("--reasoning", default="default", choices=["default", "off", "low", "medium", "high"])
    parser.add_argument("--verifier-model", default=None,
                        help="Check the candidates with THIS model instead of the solver itself (independent verification)")
    parser.add_argument("--verifier-reasoning", default="medium", choices=["default", "off", "low", "medium", "high"])
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
    print("\nDone. Build the report with: python experiments/analyze_esv3.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
