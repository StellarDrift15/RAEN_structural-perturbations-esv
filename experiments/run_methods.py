#!/usr/bin/env python3
"""Draws the n = 4 Direct QA and n = 4 CoT samples every method starts from, for each solver model.

Per instance (every original and every perturbed problem of the frozen pairs): 4 Direct QA samples and 4 CoT
samples at temperature 0.7 (seeds 0-3). The 4 CoT samples are the shared starting point: CoT maj@4 is the
baseline, and ESV v3 (run_esv3.py) re-reads these same samples, so every method is compared on identical draws.

Outputs: results/<benchmark>/<model>/records.json (every call, raw responses included). Reports: analyze_esv3.py.

Run (from the repo root):
  python experiments/run_methods.py --benchmark gsm8k --limit-pairs 2    # smoke test, all three solvers
  python experiments/run_methods.py --benchmark gsm8k                    # full run (resumes automatically)
  python experiments/run_methods.py --benchmark gsm8k --models qwen/qwen-2.5-7b-instruct   # one solver
"""

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import (  # noqa: E402
    NUM_SAMPLES, SAMPLE_TEMPERATURE, SOLVER_MODELS, answer_format_note, call_stage, extract_answer, load_json,
    make_client, model_slug, reasoning_setting, save_json, score_answer, utc_now,
)
import method_prompts as mp  # noqa: E402

# Bump whenever a prompt, sampling setting or scoring rule changes: resume only reuses records of the same version.
RUN_VERSION = "jee_methods_v1_2026-11-01"
SOLVER_MAX_TOKENS = 4000  # non-reasoning solvers; a truncated reply is retried once with the cap doubled
METHODS = ("dqa", "cot")

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
    }


def is_api_failure(call: Dict[str, Any]) -> bool:
    """True when the call never reached the model; an expired or rejected key also stops the whole run."""
    if call["status"] != "api_error":
        return False
    if any("401" in e or "expired" in e.lower() for e in call["errors"]):
        STOP.set()
    return True


class ApiFailure(Exception):
    """Raised when a stage could not be completed; the instance is left unfinished for the next run."""


class Runner:
    """One solver model. Subclassed by run_esv3.V3Runner, which reuses solver() and scored()."""

    def __init__(self, client, model: str, reasoning: Optional[Dict] = None):
        self.client, self.model, self.reasoning = client, model, reasoning

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

    def sample(self, inst: Dict[str, Any], method: str, k: int) -> Dict[str, Any]:
        if method == "dqa":
            prompt, fields = mp.build_direct_prompt(inst["question"], inst["format_note"], inst["answer_format"]), mp.DIRECT_FIELDS
        else:
            prompt, fields = mp.build_cot_prompt(inst["question"], inst["format_note"], inst["answer_format"]), mp.COT_FIELDS
        call = self.solver(f"{method}_sample_{k}", prompt, fields, ["final_answer"], SAMPLE_TEMPERATURE, seed=k)
        return self.scored(call, inst)


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
            **{k: v for k, v in inst.items() if k not in ("question", "format_note")},
            "model": model, "run_version": RUN_VERSION, "samples": {m: [None] * NUM_SAMPLES for m in METHODS},
        }
    metadata = {"model": model, "run_version": RUN_VERSION, "num_samples": NUM_SAMPLES,
                "sample_temperature": SAMPLE_TEMPERATURE, "solver_max_tokens": SOLVER_MAX_TOKENS,
                "reasoning": args.reasoning, "frozen_dataset": str(common.FROZEN_DATASET), "pairs": len(pairs)}
    lock = threading.Lock()
    last_save = [time.time()]

    # Saved records outside this run (e.g. beyond --limit-pairs) are written back untouched, never dropped.
    untouched = {iid: r for iid, r in saved.items() if iid not in records}

    def save(force: bool = False) -> None:
        if force or time.time() - last_save[0] > 20:
            save_json(path, {"metadata": {**metadata, "saved_at": utc_now()}, "instances": {**untouched, **records}})
            last_save[0] = time.time()

    runner = Runner(client, model, reasoning_setting(args.reasoning))
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

    jobs = [(iid, m, k) for iid, r in records.items() for m in METHODS for k in range(NUM_SAMPLES) if r["samples"][m][k] is None]
    print(f"\n=== {model} | {len(instances)} instances | {len(jobs)} sample calls to run ===", flush=True)
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
                print(f"  {done}/{len(jobs)}", flush=True)
    save(force=True)
    return records


def main(argv: Optional[List[str]] = None, client=None) -> int:
    parser = argparse.ArgumentParser(description="The n = 4 Direct QA and CoT samples on the frozen pairs of a benchmark.")
    common.add_benchmark_arg(parser)
    parser.add_argument("--models", nargs="+", default=list(SOLVER_MODELS))
    parser.add_argument("--reasoning", default="default", choices=["default", "off", "low", "medium", "high"],
                        help="Solver reasoning setting ('default' sends nothing; required for non-reasoning models)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--samples-only", action="store_true",
                        help="Kept for backward compatibility with older command lines; sampling is all this script does.")
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
    print(f"\nDone. Next: python experiments/analyze_esv3.py --benchmark {common.BENCHMARK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
