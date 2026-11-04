#!/usr/bin/env python3
"""Builds the frozen (P, P') dataset of a benchmark: 200 random problems -> two-stage perturbation -> frozen pairs.

Stage 1  Model A (generator) writes P' in the assigned mode (strategy-breaking or strategy-preserving), keeping the
         original answer format. Cheap deterministic checks run before any verifier is called.
Stage 2  Models B and C (verifiers, different families) first solve P' WITHOUT seeing P or the strategy; they must
         judge it well-posed, agree on the answer and be confident. Then each applies the original strategy S to P':
           breaking   -> both must show that S gives a different answer than the verified one
           preserving -> both must show that S, applied unchanged, reaches the verified answer
Accepted pairs are frozen into data/frozen_dataset.json for Direct QA, CoT and ESV.

Balancing the two modes (roughly 50/50 accepted pairs)
  Each problem is assigned ONE mode when it starts, and keeps it for all its attempts. The balancer picks the mode
  whose expected accepted count is lower: accepted + in_flight * estimated acceptance rate (Laplace-smoothed).
  Ties are broken by a seeded RNG. The assignment depends only on earlier outcomes, never on the problem's content,
  and a problem that fails in one mode is NOT retried in the other (that would push the hardest-to-break problems
  into the preserving arm).

Progress is appended to data/generation_log.jsonl after every problem, so the run can be stopped and resumed.

Run (from the repo root):
  python experiments/build_dataset.py --limit 4 --workers 2   # small smoke test
  python experiments/build_dataset.py                         # all 200 problems (resumes automatically)
  python experiments/build_dataset.py --freeze-only           # rebuild frozen_dataset.json from the log
  python experiments/build_dataset.py --benchmark gsm8k       # another benchmark (data/gsm8k/, results/gsm8k/)
"""

import argparse
import json
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from benchmarks import load_source  # noqa: E402
from common import (  # noqa: E402
    GENERATOR_MODEL, NUM_ORIGINAL_PROBLEMS, SAMPLE_SEED, VERIFIER_MODELS, answer_format_of, answers_match, bool_value,
    call_stage, is_choice_format, load_json, make_client, normalize_answer, provider_prefs, save_json, to_confidence, utc_now,
)
from perturb_prompts import (  # noqa: E402
    A_FIELDS, SOLVE_FIELDS, STRATEGY_FIELDS, generate_prompt_a, generate_prompt_solve, generate_prompt_strategy,
)



def sample_path() -> Path:
    return common.DATA_DIR / "sample_200.json"


def log_path() -> Path:
    return common.DATA_DIR / "generation_log.jsonl"


MODES = ("breaking", "preserving")
PIPELINE_VERSION = "jee_perturb_v1_2026-11-01"

CONFIDENCE_THRESHOLD = 0.6
MAX_ATTEMPTS = 3  # perturbation attempts per original problem (same mode each time, with feedback)
# Both verifiers' application of S must show the required outcome. For strategy-breaking, a verifier answering NONE
# with status INVALID/INSUFFICIENT ("S cannot be carried out") counts as a break only if this is True AND the other
# verifier shows a concrete wrong answer. The generator prompt defines a break as "gives a WRONG answer or cannot be
# completed"; the pilot nevertheless treated NONE as unknown. Off = the pilot's stricter rule. Turned on 2026-11-01
# after the 6-problem test, where it was the sole cause of a breaking perturbation failing all 3 attempts.
NONE_INVALID_COUNTS_AS_BREAK = True
REJECT_UNCHANGED_OPTIONS = True

MODEL_A, MODEL_B, MODEL_C = GENERATOR_MODEL, *VERIFIER_MODELS
MODEL_REASONING = {MODEL_A: {"effort": "high"}, MODEL_B: {"effort": "medium"}, MODEL_C: {"effort": "medium"}}
MODEL_MAX_TOKENS = {MODEL_A: 16000, MODEL_B: 12000, MODEL_C: 12000}

STOP = threading.Event()  # set on an invalid API key: every later call would fail the same way


# ---- Progress display ------------------------------------------------------------------------------

class StageTracker:
    """Live view of the pipeline on the progress bar: accepted pairs per mode, and how many problems are at each
    stage right now. Stage results are printed above the bar unless --quiet."""

    STAGES = ("generating", "checks", "verifiers solving", "strategy check")

    def __init__(self):
        self.lock = threading.Lock()
        self.current: Dict[int, str] = {}
        self.bar: Optional[tqdm] = None
        self.balancer: Optional["ModeBalancer"] = None
        self.verbose = True

    def set(self, position: int, stage: str) -> None:
        with self.lock:
            self.current[position] = stage
            self.refresh()

    def clear(self, position: int) -> None:
        with self.lock:
            self.current.pop(position, None)
            self.refresh()

    def refresh(self) -> None:
        if self.bar is None:
            return
        counts = Counter(self.current.values())
        now = ", ".join(f"{stage} {counts[stage]}" for stage in self.STAGES if counts[stage]) or "idle"
        b = self.balancer
        accepted = f"breaking {b.accepted['breaking']}/{b.finished['breaking']}, preserving {b.accepted['preserving']}/{b.finished['preserving']}" if b else ""
        self.bar.set_postfix_str(f"accepted: {accepted} | now: {now}", refresh=True)

    def log(self, tag: str, message: str, always: bool = False) -> None:
        if self.verbose or always:
            tqdm.write(f"  {tag} | {message}")


TRACKER = StageTracker()


# ---- Source problems -------------------------------------------------------------------------------

def clean_question(text: str) -> str:
    # Same LaTeX clean-up as the pilot's create_inference_prompt.
    return str(text or "").strip().replace("\\mathrm{~mL}", " mL").replace("\\mathrm", "")


def load_sample(n: int = NUM_ORIGINAL_PROBLEMS, seed: int = SAMPLE_SEED) -> List[Dict[str, Any]]:
    """The fixed random sample of n math problems. Written once to data/sample_200.json and reused afterwards."""
    saved = load_json(sample_path())
    if saved and saved.get("seed") == seed and saved.get("n") == n:
        return saved["problems"]
    pool = load_source(common.BENCHMARK)
    chosen = random.Random(seed).sample(pool, min(n, len(pool)))
    problems = [{
        "sample_position": position,
        "source": p["source"],
        "source_index": int(p["source_index"]),
        "type": p["type"],
        "answer_format": answer_format_of(p["type"]),
        "question": clean_question(p["question"]),
        "gold": str(p["gold"]).strip(),
    } for position, p in enumerate(chosen)]
    save_json(sample_path(), {"benchmark": common.BENCHMARK, "subject": common.benchmark_info()["subject"], "seed": seed,
                              "n": n, "pool_size": len(pool), "created_at": utc_now(), "problems": problems})
    return problems


# ---- Balancer --------------------------------------------------------------------------------------

class ModeBalancer:
    """Assigns strategy-breaking / strategy-preserving so the ACCEPTED pairs end up roughly 50/50."""

    def __init__(self, seed: int):
        self.lock = threading.Lock()
        self.rng = random.Random(seed + 1)
        self.accepted = {m: 0 for m in MODES}
        self.finished = {m: 0 for m in MODES}
        self.in_flight = {m: 0 for m in MODES}

    def rate(self, mode: str) -> float:
        return (self.accepted[mode] + 1) / (self.finished[mode] + 2)

    def expected(self, mode: str) -> float:
        return self.accepted[mode] + self.in_flight[mode] * self.rate(mode)

    def assign(self) -> str:
        with self.lock:
            scores = {m: self.expected(m) for m in MODES}
            low = min(scores.values())
            mode = self.rng.choice(sorted(m for m in MODES if abs(scores[m] - low) < 1e-9))
            self.in_flight[mode] += 1
            return mode

    def record(self, mode: str, accepted: bool, in_flight: bool = True) -> None:
        with self.lock:
            if in_flight:
                self.in_flight[mode] -= 1
            self.finished[mode] += 1
            self.accepted[mode] += int(accepted)

    def release(self, mode: str) -> None:
        """A problem that stopped on an API error: it is retried on the next run, so it does not count."""
        with self.lock:
            self.in_flight[mode] -= 1

    def summary(self) -> str:
        return " | ".join(f"{m}: {self.accepted[m]}/{self.finished[m]} accepted" for m in MODES)


# ---- P' assembly and cheap checks ------------------------------------------------------------------

def has_options(text: str) -> bool:
    return all(re.search(rf"\(\s*{letter}\s*\)|(?:^|\n)\s*{letter}\s*[.)]", text, re.IGNORECASE) for letter in "ABCD")


def normalize_math_delimiters(text: str) -> str:
    r"""Converts \( ... \) to $...$ and \[ ... \] to $$...$$ so P' uses the same LaTeX style as the original problems."""
    text = re.sub(r"(?<!\\)\\\((.*?)\\\)", lambda m: f"${m.group(1).strip()}$", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)\\\[(.*?)\\\]", lambda m: f"$${m.group(1).strip()}$$", text, flags=re.DOTALL)
    return text


def format_option(text: str, letter: str = "") -> str:
    t = str(text).strip()
    stray = re.search(r"</?[A-Za-z_][A-Za-z_0-9]*>", t)  # model glitch like "380</option_C>380"
    if stray:
        t = t[:stray.start()].strip()
    if letter:  # the model repeated its own label, e.g. "(A) $(P)..." -> "(A) (A) $(P)..." once assembled
        t = re.sub(rf"^\(?{letter}[).]\s*", "", t)
    if len(t) > 1 and t.startswith("$") and t.endswith("$") and "\\(" in t:
        t = t[1:-1].strip()
    t = normalize_math_delimiters(t)
    return t if "$" in t or "\\" not in t else f"${t}$"  # bare LaTeX like \frac{7\pi}{9} -> $\frac{7\pi}{9}$


def assemble_problem(a: Dict[str, str], answer_format: str) -> Optional[str]:
    """P' stem plus (A)-(D) options for MCQs; the stem alone otherwise. None if MCQ options are missing."""
    stem = normalize_math_delimiters(str(a.get("P_prime", "")).strip())
    if not stem:
        return None
    if not is_choice_format(answer_format):
        return stem
    if has_options(stem):
        return stem
    options = {letter: a.get(f"option_{letter}", "") for letter in "ABCD"}
    if not all(str(options[letter]).strip() and "(empty)" not in str(options[letter]) for letter in "ABCD"):
        return None
    return stem + "\n\n" + "\n\n".join(f"({letter}) {format_option(options[letter], letter)}" for letter in "ABCD")


def parse_options(text: str) -> Optional[Dict[str, str]]:
    options = {}
    for m in re.finditer(r"\(\s*([A-D])\s*\)\s*(.*?)(?=\n\s*\(\s*[A-D]\s*\)|\Z)", str(text), re.DOTALL):
        options[m.group(1)] = re.sub(r"[\s$]+", "", m.group(2)).lower()
    return options if len(options) == 4 else None


def options_unchanged(problem: str, perturbed: str) -> bool:
    original, new = parse_options(problem), parse_options(perturbed)
    return bool(original and new and all(original[l] == new[l] for l in "ABCD"))


def valid_answer(value: Optional[str], answer_format: str) -> bool:
    if value is None:
        return False
    if answer_format == "single":
        return len(value) == 1
    if answer_format == "integer":
        return re.fullmatch(r"\d+", value) is not None
    return True


# ---- One attempt -----------------------------------------------------------------------------------

class ApiFailure(Exception):
    pass


def check_api(call: Dict[str, Any]) -> None:
    if call["status"] == "api_error":
        if any("401" in e or "expired" in e.lower() for e in call["errors"]):
            STOP.set()
        raise ApiFailure(f"{call['stage']} ({call['model']}): {call['errors'][-1] if call['errors'] else 'api error'}")


def verifier_call(client, model: str, stage: str, prompt: str, fields: List[str], required: List[str]) -> Dict[str, Any]:
    call = call_stage(client, model, stage, prompt, fields, required, MODEL_REASONING[model], MODEL_MAX_TOKENS[model],
                      temperature=None)
    check_api(call)
    return call


def summarize_call(call: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """What the log keeps of a verifier/generator call: parsed fields, status, usage (raw text kept for model A)."""
    if call is None:
        return None
    keep = {k: call[k] for k in ("stage", "model", "status", "parsed", "finish_reason", "attempts", "usage", "latency_seconds")}
    return keep


def run_attempt(client, problem: Dict[str, Any], mode: str, feedback: Optional[str], tag: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Returns (rejection reason or None, attempt record)."""
    fmt_ = problem["answer_format"]
    position = problem["sample_position"]
    record: Dict[str, Any] = {"mode": mode, "feedback_given": feedback}
    TRACKER.set(position, "generating")
    a_prompt = generate_prompt_a(problem["question"], problem["gold"], mode, fmt_, feedback)
    a_call = call_stage(client, MODEL_A, "generator", a_prompt, A_FIELDS, ["status"], MODEL_REASONING[MODEL_A],
                        MODEL_MAX_TOKENS[MODEL_A], temperature=None)
    check_api(a_call)
    if a_call["finish_reason"] == "length" and not a_call["parsed"].get("answer_prime"):
        # Cut off by the token cap after <status> (hidden reasoning counts toward it): one retry with the cap doubled,
        # instead of losing the whole attempt to a truncated candidate.
        TRACKER.log(tag, f"generator reply cut off at {MODEL_MAX_TOKENS[MODEL_A]} tokens; retrying with {2 * MODEL_MAX_TOKENS[MODEL_A]}")
        a_call = call_stage(client, MODEL_A, "generator", a_prompt, A_FIELDS, ["status"], MODEL_REASONING[MODEL_A],
                            2 * MODEL_MAX_TOKENS[MODEL_A], temperature=None)
        check_api(a_call)
    record["model_a"] = summarize_call(a_call)
    a = a_call["parsed"]
    TRACKER.log(tag, f"generator: {a.get('status', 'no status') or 'no status'}, answer={a.get('answer_prime')!r}, "
                     f"naive={a.get('naive_answer')!r} ({a_call['latency_seconds']:.0f}s, {a_call['usage']['completion_tokens']} tok)")
    TRACKER.set(position, "checks")
    if "CANDIDATE" not in a.get("status", "").upper():
        return "model A returned no candidate (REJECT or unparsable)", record
    missing = [k for k in ("P_prime", "naive_answer", "answer_prime", "original_strategy", "changed_condition") if not a.get(k)]
    if missing:
        return f"model A's candidate is missing {missing}", record
    perturbed = assemble_problem(a, fmt_)
    if perturbed is None:
        return "P' is missing the (A)-(D) options", record
    record["perturbed_problem"] = perturbed
    a_answer, a_naive = normalize_answer(a.get("answer_prime"), fmt_), normalize_answer(a.get("naive_answer"), fmt_)
    record["model_a"].update({"answer_prime": a_answer, "naive_answer": a_naive})

    # Cheap deterministic checks before any verifier call
    if not valid_answer(a_answer, fmt_):
        return f"model A's answer_prime {a.get('answer_prime')!r} is not a valid {fmt_} answer", record
    if mode == "breaking" and a_naive is not None and answers_match(a_naive, a_answer, fmt_):
        return "model A's own naive_answer equals its answer_prime (the perturbation does not break the strategy)", record
    if mode == "preserving" and not answers_match(a_naive, a_answer, fmt_):
        return "model A's own naive_answer differs from its answer_prime (the perturbation breaks the strategy; it must preserve it)", record
    if REJECT_UNCHANGED_OPTIONS and is_choice_format(fmt_) and options_unchanged(problem["question"], perturbed):
        return "P' reuses the original four options unchanged; recompute the options for P'", record
    if not is_choice_format(fmt_) and answers_match(a_answer, normalize_answer(problem["gold"], fmt_), fmt_):
        return "the answer of P' equals the original answer; change the problem so the answer must be recomputed", record

    # Stage 2a: independent solves of P' alone
    if STOP.is_set():
        raise ApiFailure("stopped")
    TRACKER.log(tag, "cheap checks passed; verifiers solving P' blind")
    TRACKER.set(position, "verifiers solving")
    # Both verifiers must pass every check, so C (fast) runs first and B (slow) only if C has not already rejected P'.
    # The accept/reject outcome is the same as running both; only B's time is saved on attempts C rejects.
    solve_prompt = generate_prompt_solve(perturbed, fmt_)

    def solve(model: str, name: str) -> Dict[str, Any]:
        call = verifier_call(client, model, "verifier_solve", solve_prompt, SOLVE_FIELDS, ["valid_problem", "answer_prime"])
        p = call["parsed"]
        record[name] = info = {"valid_problem": bool_value(p.get("valid_problem")), "answer_prime": normalize_answer(p.get("answer_prime"), fmt_),
                               "answer_prime_raw": p.get("answer_prime"), "confidence": to_confidence(p.get("confidence_level")),
                               "reason": p.get("reason"), "call": summarize_call(call)}
        TRACKER.log(tag, f"verifier {name[-1].upper()}: answer={info['answer_prime']} valid={info['valid_problem']} "
                         f"conf={info['confidence']:.2f} ({call['latency_seconds']:.0f}s)")
        return info

    def solve_rejection(s: Dict[str, Any]) -> Optional[str]:
        if s["valid_problem"] is not True:
            return "a verifier judged P' ill-posed, ambiguous or without a correct answer"
        if not valid_answer(s["answer_prime"], fmt_):
            return f"a verifier found no valid {fmt_} answer for P'"
        if s["confidence"] < CONFIDENCE_THRESHOLD:
            return "verifier confidence below threshold"
        return None

    c_info = solve(MODEL_C, "model_c")
    if solve_rejection(c_info):
        return solve_rejection(c_info) + " (verifier C; B skipped)", record
    b_info = solve(MODEL_B, "model_b")
    if solve_rejection(b_info):
        return solve_rejection(b_info), record
    if not answers_match(b_info["answer_prime"], c_info["answer_prime"], fmt_):
        return "answer_prime mismatch between the two verifiers", record
    solvers = [b_info, c_info]
    answer_prime = b_info["answer_prime"]
    if not is_choice_format(fmt_) and answers_match(answer_prime, normalize_answer(problem["gold"], fmt_), fmt_):
        return "the verified answer of P' equals the original answer (cosmetic change)", record

    # Stage 2b: each verifier applies S to P' (same prompt in both modes: the verifier is not told the expected outcome)
    if STOP.is_set():
        raise ApiFailure("stopped")
    TRACKER.set(position, "strategy check")

    def strategy_vote(model: str, s: Dict[str, Any]) -> str:
        """'yes' = shows the outcome this mode requires, 'soft' = S could not be carried out (see
        NONE_INVALID_COUNTS_AS_BREAK), 'no' = contradicts it."""
        call = verifier_call(client, model, "verifier_strategy",
                             generate_prompt_strategy(problem["question"], perturbed, a["original_strategy"], a["changed_condition"],
                                                      s["answer_prime"], fmt_), STRATEGY_FIELDS, ["strategy_status", "naive_answer"])
        p = call["parsed"]
        status = str(p.get("strategy_status", "")).upper()
        naive = normalize_answer(p.get("naive_answer"), fmt_)
        s.update({"strategy_status": status, "strategy_reason": p.get("strategy_reason"), "naive_answer": naive,
                  "naive_answer_raw": p.get("naive_answer"), "strategy_call": summarize_call(call)})
        if mode == "breaking":
            if naive is not None:
                vote = "yes" if not answers_match(naive, s["answer_prime"], fmt_) else "no"
            else:  # NONE: "S cannot be carried out". Unknown by default (a model may simply have failed), as in the pilot.
                vote = "soft" if NONE_INVALID_COUNTS_AS_BREAK and ("INVALID" in status or "INSUFFICIENT" in status) else "no"
        else:
            vote = "yes" if answers_match(naive, s["answer_prime"], fmt_) and "INVALID" not in status else "no"
        TRACKER.log(tag, f"strategy check {model.split('/')[-1]}: naive={naive} status={status or '-'} -> {vote} "
                         f"({call['latency_seconds']:.0f}s)")
        return vote

    # Acceptance: no 'no' vote, and at least one concrete 'yes' (a 'soft' vote alone never accepts).
    wanted = "S fails" if mode == "breaking" else "S still works"
    c_vote = strategy_vote(MODEL_C, c_info)
    votes = [c_vote]
    if c_vote != "no":
        votes.append(strategy_vote(MODEL_B, b_info))
    if "no" in votes or "yes" not in votes:
        skipped = " (verifier C; B skipped)" if c_vote == "no" else ""
        if mode == "breaking":
            return "the original strategy still works on P' (naive_answer equals answer_prime) or could not be shown to fail" + skipped, record
        return "the original strategy does not reach the verified answer of P' (the perturbation is not strategy-preserving)" + skipped, record
    TRACKER.log(tag, f"strategy votes {votes} for '{wanted}'")

    record["accepted"] = {
        "perturbed_problem": perturbed,
        "answer_prime": answer_prime,
        "original_strategy": a.get("original_strategy"),
        "changed_condition": a.get("changed_condition"),
        "failure_reason": a.get("failure_reason"),
        "alternative_strategy": a.get("alternative_strategy"),
    }
    return None, record


def build_feedback(reason: str, record: Dict[str, Any]) -> str:
    notes = [str(record[n].get(k)) for n in ("model_b", "model_c") if isinstance(record.get(n), dict)
             for k in ("reason", "strategy_reason") if record[n].get(k)]
    return reason + (". Verifier notes: " + " | ".join(notes) if notes else "")


def process_problem(client, problem: Dict[str, Any], balancer: ModeBalancer) -> Optional[Dict[str, Any]]:
    """Runs up to MAX_ATTEMPTS attempts in one assigned mode. Returns the log row, or None on an API failure."""
    mode = balancer.assign()
    tag = f"#{problem['sample_position']} ({problem['type']}, {mode})"
    attempts, feedback, started = [], None, time.perf_counter()
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            reason, record = run_attempt(client, problem, mode, feedback, f"{tag} try {attempt}/{MAX_ATTEMPTS}")
            record["attempt"] = attempt
            record["rejection_reason"] = reason
            attempts.append(record)
            if reason is None:
                break
            TRACKER.log(f"{tag} try {attempt}/{MAX_ATTEMPTS}", f"REJECTED: {reason}", always=True)
            feedback = build_feedback(reason, record)
    except ApiFailure as exc:
        balancer.release(mode)
        TRACKER.clear(problem["sample_position"])
        TRACKER.log(tag, f"API failure, will retry on the next run: {exc}", always=True)
        return None
    accepted = attempts[-1].get("accepted")
    balancer.record(mode, accepted is not None)
    TRACKER.clear(problem["sample_position"])
    TRACKER.log(tag, f"{'ACCEPTED' if accepted else 'gave up'} after {len(attempts)} attempt(s), "
                     f"{time.perf_counter() - started:.0f}s", always=True)
    return {
        "pipeline_version": PIPELINE_VERSION,
        "sample_position": problem["sample_position"],
        "source_index": problem["source_index"],
        "mode": mode,
        "accepted": accepted is not None,
        "attempts_used": len(attempts),
        "attempts": attempts,
        "seconds": round(time.perf_counter() - started, 1),
        "finished_at": utc_now(),
    }


# ---- Log and freeze --------------------------------------------------------------------------------

def read_log() -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    if log_path().exists():
        for line in log_path().read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("pipeline_version") == PIPELINE_VERSION:
                    rows[row["sample_position"]] = row
    return rows


def freeze(problems: List[Dict[str, Any]], log: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    pairs = []
    for problem in problems:  # sample order, so ids are stable whatever order the workers finished in
        row = log.get(problem["sample_position"])
        if not row or not row["accepted"]:
            continue
        final = row["attempts"][-1]
        acc = final["accepted"]
        validation = {
            "model_a": {"answer_prime": final["model_a"].get("answer_prime"), "naive_answer": final["model_a"].get("naive_answer")},
            **{name: {k: final[name].get(k) for k in ("valid_problem", "answer_prime", "confidence", "naive_answer", "strategy_status")}
               for name in ("model_b", "model_c")},
        }
        pairs.append({
            "id": len(pairs),
            "source": problem["source"],
            "source_index": problem["source_index"],
            "sample_position": problem["sample_position"],
            "type": problem["type"],
            "answer_format": problem["answer_format"],
            "perturbation_type": row["mode"],
            "original_problem": problem["question"],
            "original_gold": problem["gold"],
            # Rows logged before the doubled-label fix in format_option still carry "(A) (A) ..."
            "perturbed_problem": re.sub(r"(?m)^\(([A-D])\) \(\1\)[).]?\s*", r"(\1) ", acc["perturbed_problem"]),
            "answer_prime": acc["answer_prime"],
            "attempts_used": row["attempts_used"],
            "original_strategy": acc["original_strategy"],
            "changed_condition": acc["changed_condition"],
            "failure_reason": acc["failure_reason"],
            "alternative_strategy": acc["alternative_strategy"],
            "validation": validation,
        })
    processed = [log[p["sample_position"]] for p in problems if p["sample_position"] in log]
    split = {m: sum(p["perturbation_type"] == m for p in pairs) for m in MODES}
    attempted = {m: sum(r["mode"] == m for r in processed) for m in MODES}
    dataset = {
        "benchmark": common.BENCHMARK,
        "subject": common.benchmark_info()["subject"],
        "pipeline_version": PIPELINE_VERSION,
        "models": {"a": MODEL_A, "b": MODEL_B, "c": MODEL_C},
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "none_invalid_counts_as_break": NONE_INVALID_COUNTS_AS_BREAK,
        "provider_routing": {m: provider_prefs(m) for m in (MODEL_A, MODEL_B, MODEL_C)},
        "sample_seed": SAMPLE_SEED,
        "problems_sampled": len(problems),
        "problems_processed": len(processed),
        "num_pairs": len(pairs),
        "perturbation_split": split,
        "problems_attempted_per_mode": attempted,
        "acceptance_rate_per_mode": {m: round(split[m] / attempted[m], 4) if attempted[m] else None for m in MODES},
        "frozen_at": utc_now(),
        "pairs": pairs,
    }
    save_json(common.FROZEN_DATASET, dataset)
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen (P, P') dataset of a benchmark with balanced perturbation types.")
    common.add_benchmark_arg(parser)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N sampled problems (smoke test)")
    parser.add_argument("--workers", type=int, default=16, help="Problems processed in parallel (lower if you hit 429s)")
    parser.add_argument("--freeze-only", action="store_true", help="Rebuild frozen_dataset.json from the log; no API calls")
    parser.add_argument("--quiet", action="store_true", help="Only the progress bar and accept/reject lines, no per-stage lines")
    args = parser.parse_args()
    common.select_benchmark(args.benchmark)
    TRACKER.verbose = not args.quiet

    problems = load_sample()
    log = read_log()
    if args.freeze_only:
        dataset = freeze(problems, log)
        print(f"Frozen {dataset['num_pairs']} pairs from {dataset['problems_processed']} processed problems: {dataset['perturbation_split']}")
        return 0

    todo = [p for p in problems[: args.limit or len(problems)] if p["sample_position"] not in log]
    balancer = ModeBalancer(SAMPLE_SEED)
    for row in log.values():  # resume: earlier outcomes steer the balance of the remaining problems
        balancer.record(row["mode"], row["accepted"], in_flight=False)
    print(f"Benchmark: {common.BENCHMARK} | sample: {len(problems)} problems (seed {SAMPLE_SEED}) | done: {len(log)} | to run: {len(todo)}")
    print(f"Generator: {MODEL_A} | Verifiers: {MODEL_B}, {MODEL_C} | {balancer.summary()}")
    if not todo:
        dataset = freeze(problems, log)
        print(f"Nothing to run. Frozen {dataset['num_pairs']} pairs: {dataset['perturbation_split']}")
        return 0

    client = make_client()
    lock = threading.Lock()
    log_path().parent.mkdir(parents=True, exist_ok=True)
    executor = ThreadPoolExecutor(max(1, args.workers))

    def guarded(problem):
        if STOP.is_set():
            return None
        try:
            return process_problem(client, problem, balancer)
        except Exception as exc:  # never let one problem kill the run; it is retried on the next run
            TRACKER.clear(problem["sample_position"])
            TRACKER.log(f"#{problem['sample_position']}", f"unexpected error {type(exc).__name__}: {exc}", always=True)
            return None

    started = time.perf_counter()
    bar = tqdm(total=len(todo), desc="Problems", unit="problem", dynamic_ncols=True)
    TRACKER.bar, TRACKER.balancer = bar, balancer
    TRACKER.refresh()
    try:
        # The mode is assigned when a worker STARTS a problem (not at submission), so the balancer has seen every
        # outcome finished so far.
        futures = [executor.submit(guarded, p) for p in todo]
        for future in as_completed(futures):
            row = future.result()
            bar.update(1)
            if row is None:
                continue
            with lock:
                with log_path().open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                log[row["sample_position"]] = row
                try:
                    freeze(problems, log)
                except OSError as exc:  # the log already holds the row; the final freeze rewrites the dataset
                    TRACKER.log("freeze", f"could not update {common.FROZEN_DATASET.name} ({exc}); continuing", always=True)
            TRACKER.refresh()
    except KeyboardInterrupt:
        tqdm.write("\nInterrupted: finished problems are saved; rerun to resume.")
        STOP.set()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        bar.close()
        TRACKER.bar = None

    dataset = freeze(problems, log)
    if STOP.is_set():
        print("\nStopped early: the API key was rejected (401/expired). Fix the key in the env file and rerun to resume.")
    print(f"\nFrozen dataset: {dataset['num_pairs']} pairs from {dataset['problems_processed']} processed problems "
          f"| split {dataset['perturbation_split']} | acceptance {dataset['acceptance_rate_per_mode']}")
    print(f"Saved: {common.FROZEN_DATASET} | log: {log_path()} | {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
