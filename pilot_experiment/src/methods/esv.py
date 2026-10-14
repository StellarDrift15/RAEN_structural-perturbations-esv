#!/usr/bin/env python3
"""End-to-end Explicit Strategy Verification (ESV) evaluation for a frozen paired dataset.

Expected input schema: frozen_dataset.json with a top-level `pairs` list. Each pair has
`original_problem`, `original_gold`, `perturbed_problem`, and `answer_prime`.

Environment:
  OPENROUTER_API_KEY=...
  ESV_MODEL=qwen/qwen3.5-9b   (optional override)

Run from any working directory:
  python methods/esv.py
  python methods/esv.py --input frozen_dataset.json --model qwen/qwen3.5-9b

The script writes checkpointed esv_results.json and esv_metrics.json beside the input
file by default. It does not expose gold answers to the solving model; gold answers are
used only locally for correctness evaluation. Stage-level diagnosis/strategy quality is
logged as model self-assessment, not claimed as independent ground truth.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the project root (parent of methods/) and current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
load_dotenv(PROJECT_DIR / ".env")
load_dotenv()

DEFAULT_INPUT = PROJECT_DIR / "frozen_dataset.json"
DEFAULT_RESULTS = PROJECT_DIR / "esv_results.json"
DEFAULT_METRICS = PROJECT_DIR / "esv_metrics.json"
DEFAULT_MODEL = os.getenv("ESV_MODEL", "deepseek/deepseek-v4-flash-0731")

# Keep this initial instruction synchronized with the Direct QA baseline prompt.
# Only the question text differs between original and perturbed instances.
DIRECT_QA_INSTRUCTION = (
    """Solve the following problem carefully. Work through the mathematical reasoning step by step before selecting your final answer.

        Identify the relevant concepts, derive the necessary equations, and verify that your reasoning satisfies the problem's conditions. For multiple-choice questions, evaluate the options as needed and select the single correct option.

        Return your response as valid JSON using exactly this schema:
        {
        "reasoning": "Your step-by-step mathematical solution",
        "final_answer": "The option letter and its corresponding value for MCQs, or the final numerical answer otherwise"
        }
        For MCQs, always include the option letter in final_answer, even when the answer is numerical. Do not omit the final answer. Ensure the JSON is complete and valid.
    """
)

MAX_TOKENS = int(os.getenv("ESV_MAX_TOKENS", "1800"))
MAX_RETRIES = int(os.getenv("ESV_MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.getenv("ESV_TIMEOUT_SECONDS", "120"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_question(problem: str) -> str:
    """Remove the frozen original's dataset-specific wrapper, if present."""
    text = str(problem or "").strip()
    marker = "### Question:"
    if marker in text:
        text = text.split(marker, 1)[1].strip()
    # The original entries sometimes end with several newlines; normalize only edges.
    return text.strip()


def build_direct_prompt(problem: str) -> str:
    question = extract_question(problem)
    return (
        f"{DIRECT_QA_INSTRUCTION}\n\n"
        f"Problem:\n{question}\n\n"
        "Answer:"
    )


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON even if the model wraps it in a Markdown code fence."""
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    # Recover the first JSON object from surrounding text.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(cleaned[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def call_model(client: OpenAI, model: str, prompt: str, stage: str) -> Dict[str, Any]:
    """Call OpenRouter with retry handling; preserve usage and latency metadata."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0,
            )
            elapsed = round(time.perf_counter() - started, 3)
            choice = response.choices[0]
            content = choice.message.content or ""
            usage = getattr(response, "usage", None)
            usage_data = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            }
            return {
                "ok": True,
                "stage": stage,
                "model": model,
                "raw_response": content,
                "parsed": parse_json_object(content),
                "finish_reason": getattr(choice, "finish_reason", None),
                "usage": usage_data,
                "latency_seconds": elapsed,
                "attempts": attempt,
                "error": None,
            }
        except Exception as exc:  # API/provider errors are recorded, not silently swallowed.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_RETRIES:
                time.sleep(min(2 * attempt, 6))
    return {
        "ok": False,
        "stage": stage,
        "model": model,
        "raw_response": "",
        "parsed": None,
        "finish_reason": None,
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        "latency_seconds": None,
        "attempts": MAX_RETRIES,
        "error": last_error,
    }


def value_as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def extract_final_answer(call: Dict[str, Any]) -> str:
    parsed = call.get("parsed") or {}
    for key in ("final_answer", "answer", "recovered_answer"):
        if parsed.get(key) is not None and value_as_text(parsed[key]):
            return value_as_text(parsed[key])
    raw = value_as_text(call.get("raw_response"))
    # Fallback for providers that fail to follow the JSON output instruction.
    matches = re.findall(r"(?:final\s+answer|answer)\s*[:=]\s*(.+)", raw, flags=re.IGNORECASE)
    return matches[-1].strip().splitlines()[0] if matches else raw.strip().splitlines()[-1] if raw.strip() else ""


def expected_is_choice(gold: Any) -> bool:
    return bool(re.fullmatch(r"\s*[A-Da-d](?:[\s,;&]+[A-Da-d])*\s*", str(gold or ""))) or bool(
        re.fullmatch(r"\s*[A-Da-d]{1,4}\s*", str(gold or ""))
    )


def normalize_choice(value: Any) -> Optional[str]:
    """Normalize choice answers such as '(A), C' or 'AC' to sorted unique letters."""
    text = str(value or "").strip().upper()
    # Prefer a compact all-letters answer, which avoids collecting letters from explanations.
    compact = re.sub(r"[\s,;&()\[\]{}.*]+", "", text)
    compact = re.sub(r"^(?:OPTION|OPTIONS|ANSWER|FINALANSWER)+", "", compact)
    if re.fullmatch(r"[A-D]{1,4}", compact):
        return "".join(sorted(set(compact)))
    # Otherwise read explicit option markers, e.g. "Option A and Option C".
    matches = re.findall(r"(?:OPTION\s*|\()([A-D])(?:\)|\b)", text)
    if matches:
        return "".join(sorted(set(matches)))
    return None


def normalize_decimal(value: Any) -> Optional[Decimal]:
    text = str(value or "").strip().replace(",", "")
    # Simple fractions are supported (e.g. 3/4).
    frac = re.fullmatch(r"\s*([-+]?\d+)\s*/\s*(\d+)\s*", text)
    if frac:
        denominator = Decimal(frac.group(2))
        if denominator == 0:
            return None
        return Decimal(frac.group(1)) / denominator
    # If there is surrounding text, extract the first numeric token from the answer field.
    match = re.search(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return None


def compare_answer(prediction: Any, gold: Any) -> Tuple[Optional[bool], str, str]:
    """Compare answer against frozen gold. Returns (correct, normalized_pred, normalized_gold)."""
    pred_text = value_as_text(prediction)
    gold_text = value_as_text(gold)
    if not pred_text or not gold_text:
        return None, pred_text, gold_text
    if expected_is_choice(gold_text):
        pred_norm = normalize_choice(pred_text)
        gold_norm = normalize_choice(gold_text)
        if pred_norm is None or gold_norm is None:
            return None, pred_norm or pred_text, gold_norm or gold_text
        return pred_norm == gold_norm, pred_norm, gold_norm
    pred_num = normalize_decimal(pred_text)
    gold_num = normalize_decimal(gold_text)
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) <= Decimal("1e-8"), str(pred_num), str(gold_num)
    # Conservative text fallback: normalize whitespace and case, without guessing equivalence.
    pred_norm = re.sub(r"\s+", "", pred_text).lower()
    gold_norm = re.sub(r"\s+", "", gold_text).lower()
    return pred_norm == gold_norm, pred_norm, gold_norm


def bool_value(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().upper()
    if text in {"TRUE", "YES", "1"}:
        return True
    if text in {"FALSE", "NO", "0"}:
        return False
    return None


def json_prompt(task: str, schema: Dict[str, str]) -> str:
    fields = ",\n".join(f'  "{key}": "{description}"' for key, description in schema.items())
    return task + "\n\nReturn exactly one valid JSON object, with these fields:\n{\n" + fields + "\n}\nDo not use Markdown fences. Keep explanations concise."


def run_stage(client: OpenAI, model: str, stage: str, prompt: str) -> Dict[str, Any]:
    result = call_model(client, model, prompt, stage)
    parsed = result.get("parsed") or {}
    result["stage_status"] = "completed" if result["ok"] and parsed else (
        "invalid_json" if result["ok"] else "api_error"
    )
    return result


def run_esv_for_instance(
    client: OpenAI,
    model: str,
    pair: Dict[str, Any],
    problem_kind: str,
    problem: str,
    gold: Any,
    pair_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    instance_id = f"{pair.get('id', 'unknown')}_{problem_kind}"
    question = extract_question(problem)
    record: Dict[str, Any] = {
        "instance_id": instance_id,
        "pair_id": pair.get("id"),
        "source": pair.get("source"),
        "source_index": pair.get("source_index"),
        "problem_type": problem_kind,
        "question_type": pair.get("type"),
        "problem": question,
        "gold_answer": value_as_text(gold),
        "model": model,
        "started_at": utc_now(),
        "stages": {},
        "esv_activated": False,
        "initial_correct": None,
        "final_correct": None,
        "frozen_pair_annotations": pair_metadata,
    }

    # Stage 1: same initial instruction intended for Direct QA; the gold is not in the prompt.
    initial_prompt = json_prompt(
        build_direct_prompt(problem),
        {"reasoning": "brief reasoning, if useful", "final_answer": "final answer only, in the problem's requested format"},
    )
    initial = run_stage(client, model, "initial_direct_qa", initial_prompt)
    initial_answer = extract_final_answer(initial)
    initial_correct, pred_norm, gold_norm = compare_answer(initial_answer, gold)
    initial.update({
        "extracted_answer": initial_answer,
        "normalized_answer": pred_norm,
        "normalized_gold": gold_norm,
        "correct": initial_correct,
    })
    record["stages"]["initial_direct_qa"] = initial
    record["initial_answer"] = initial_answer
    record["initial_correct"] = initial_correct

    if initial_correct is True:
        record["esv_activated"] = False
        record["final_answer"] = initial_answer
        record["final_correct"] = True
        record["end_reason"] = "initial_answer_correct; recovery stages skipped"
        record["finished_at"] = utc_now()
        return record

    # A missing/unparseable answer is not treated as correct; the ESV stages are still attempted.
    record["esv_activated"] = True
    prior_answer = initial_answer or "[No answer could be extracted]"

    # Stage 2: recognition. Since the dataset checker triggered ESV, the true state is failure.
    recognition_prompt = json_prompt(
        "You are performing the failure-recognition stage of ESV. Re-examine the problem and your initial answer. "
        "Do not see or guess any reference answer. Decide whether your initial answer likely contains an error. "
        "Set recognized_failure to TRUE if you recognize a likely failure, otherwise FALSE.\n\n"
        f"Problem:\n{question}\n\nYour initial answer:\n{prior_answer}",
        {"recognized_failure": "TRUE or FALSE", "reason": "brief reason for the decision"},
    )
    recognition = run_stage(client, model, "recognition", recognition_prompt)
    recognition_value = bool_value((recognition.get("parsed") or {}).get("recognized_failure"))
    recognition["recognized_failure_bool"] = recognition_value
    # All triggered cases are objectively incorrect (or unscorable), so TRUE is recognition success.
    recognition["recognition_success"] = (recognition_value is True) if initial_correct is False else None
    record["stages"]["recognition"] = recognition

    # Stage 3: diagnosis. This remains an attempted stage even if recognition returned FALSE,
    # so we can measure downstream recovery potential independently of the recognition gate.
    diagnosis_prompt = json_prompt(
        "You are performing the diagnosis stage of ESV. Diagnose the likely mathematical error in the initial answer. "
        "Do not use or request the reference answer. Point to a concrete faulty step, assumption, or missed condition. "
        "If you cannot identify a specific error, say so plainly.\n\n"
        f"Problem:\n{question}\n\nInitial answer:\n{prior_answer}\n\n"
        f"Recognition output:\n{json.dumps(recognition.get('parsed') or {}, ensure_ascii=False)}",
        {
            "diagnosis": "specific suspected error or NONE_IDENTIFIED",
            "failed_assumption_or_step": "the condition or reasoning step that failed, if any",
            "diagnosis_self_assessment": "TRUE if you identified a concrete plausible error, otherwise FALSE",
        },
    )
    diagnosis = run_stage(client, model, "diagnosis", diagnosis_prompt)
    diagnosis["diagnosis_self_assessment_bool"] = bool_value((diagnosis.get("parsed") or {}).get("diagnosis_self_assessment"))
    diagnosis["quality_is_self_assessed"] = True
    record["stages"]["diagnosis"] = diagnosis

    # Stage 4: alternative strategy. It receives the diagnosis, but not the gold answer.
    diagnosis_payload = diagnosis.get("parsed") or {}
    alternative_prompt = json_prompt(
        "You are performing the alternative-strategy stage of ESV. Propose a mathematically valid approach "
        "that can solve the problem and avoids the suspected failure. State the core idea and why it applies. "
        "Do not provide or use the reference answer.\n\n"
        f"Problem:\n{question}\n\nInitial answer:\n{prior_answer}\n\n"
        f"Diagnosis:\n{json.dumps(diagnosis_payload, ensure_ascii=False)}",
        {
            "alternative_strategy": "concise alternative method",
            "why_it_applies": "why this method fits the problem and avoids the suspected error",
            "strategy_self_assessment": "TRUE if the proposed method appears valid and sufficient, otherwise FALSE",
        },
    )
    alternative = run_stage(client, model, "alternative_strategy", alternative_prompt)
    alternative["strategy_self_assessment_bool"] = bool_value((alternative.get("parsed") or {}).get("strategy_self_assessment"))
    alternative["quality_is_self_assessed"] = True
    record["stages"]["alternative_strategy"] = alternative

    # Stage 5: recovery. Ask for a revised solution using the alternative strategy.
    alt_payload = alternative.get("parsed") or {}
    recovery_prompt = json_prompt(
        "You are performing the recovery stage of ESV. Re-solve the problem from scratch using the proposed "
        "alternative strategy where appropriate. Do not use a reference answer. Show a concise derivation and "
        "give a tentative answer in the requested format.\n\n"
        f"Problem:\n{question}\n\nInitial answer:\n{prior_answer}\n\n"
        f"Diagnosis:\n{json.dumps(diagnosis_payload, ensure_ascii=False)}\n\n"
        f"Alternative strategy:\n{json.dumps(alt_payload, ensure_ascii=False)}",
        {"recovery_reasoning": "concise revised derivation", "recovered_answer": "tentative final answer in requested format"},
    )
    recovery = run_stage(client, model, "recovery", recovery_prompt)
    recovery_answer = extract_final_answer(recovery)
    recovery["extracted_answer"] = recovery_answer
    recovery_correct, recovery_norm, recovery_gold_norm = compare_answer(recovery_answer, gold)
    recovery["normalized_answer"] = recovery_norm
    recovery["normalized_gold"] = recovery_gold_norm
    recovery["correct"] = recovery_correct
    record["stages"]["recovery"] = recovery

    # Stage 6: final answer normalization. This is a separate short call to make the answer
    # field standardized; the local frozen-answer checker evaluates it objectively.
    final_prompt = json_prompt(
        "You are performing the final-answer stage of ESV. Based only on the recovered solution below, "
        "return the final answer in the exact format requested by the problem. Do not introduce a new solution "
        "or use a reference answer. For multiple-choice questions, return only the selected option letter(s); "
        "for integer/numeric questions, return only the number.\n\n"
        f"Problem:\n{question}\n\nRecovered solution and tentative answer:\n"
        f"{json.dumps(recovery.get('parsed') or {'raw_response': recovery.get('raw_response', '')}, ensure_ascii=False)}",
        {"final_answer": "answer only, in the requested format"},
    )
    final_stage = run_stage(client, model, "final_answer", final_prompt)
    final_answer = extract_final_answer(final_stage)
    final_correct, final_norm, final_gold_norm = compare_answer(final_answer, gold)
    final_stage.update({
        "extracted_answer": final_answer,
        "normalized_answer": final_norm,
        "normalized_gold": final_gold_norm,
        "correct": final_correct,
    })
    record["stages"]["final_answer"] = final_stage
    record["final_answer"] = final_answer
    record["final_correct"] = final_correct
    record["end_reason"] = "ESV stages attempted after initial incorrect or unscorable answer"

    # Frozen annotations are retained for later research analysis, never included in solver prompts.
    record["finished_at"] = utc_now()
    return record


def stage_token_totals(records: list) -> Dict[str, Any]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "known_calls": 0}
    for record in records:
        for stage in record.get("stages", {}).values():
            usage = stage.get("usage") or {}
            if usage.get("total_tokens") is not None:
                totals["known_calls"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if isinstance(usage.get(key), (int, float)):
                    totals[key] += usage[key]
    return totals


def safe_rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def group_metrics(records: list) -> Dict[str, Any]:
    total = len(records)
    initial_scored = [r for r in records if r.get("initial_correct") is not None]
    initially_correct = sum(r.get("initial_correct") is True for r in initial_scored)
    initially_incorrect = sum(r.get("initial_correct") is False for r in initial_scored)
    activated = [r for r in records if r.get("esv_activated")]
    recognition_scored = [r for r in activated if r.get("stages", {}).get("recognition", {}).get("recognized_failure_bool") is not None]
    recognition_true = sum(r["stages"]["recognition"].get("recognized_failure_bool") is True for r in recognition_scored)
    diagnosis_assessed = [r for r in activated if r.get("stages", {}).get("diagnosis", {}).get("diagnosis_self_assessment_bool") is not None]
    diagnosis_self_true = sum(r["stages"]["diagnosis"].get("diagnosis_self_assessment_bool") is True for r in diagnosis_assessed)
    strategy_assessed = [r for r in activated if r.get("stages", {}).get("alternative_strategy", {}).get("strategy_self_assessment_bool") is not None]
    strategy_self_true = sum(r["stages"]["alternative_strategy"].get("strategy_self_assessment_bool") is True for r in strategy_assessed)
    recovery_scored = [r for r in activated if r.get("stages", {}).get("recovery", {}).get("correct") is not None]
    recovery_correct = sum(r["stages"]["recovery"].get("correct") is True for r in recovery_scored)
    final_scored = [r for r in records if r.get("final_correct") is not None]
    final_correct = sum(r.get("final_correct") is True for r in final_scored)
    api_errors = sum(1 for r in records for s in r.get("stages", {}).values() if not s.get("ok"))
    invalid_json = sum(1 for r in records for s in r.get("stages", {}).values() if s.get("stage_status") == "invalid_json")
    latencies = [s["latency_seconds"] for r in records for s in r.get("stages", {}).values() if isinstance(s.get("latency_seconds"), (int, float))]
    return {
        "instances": total,
        "initial_accuracy": {
            "correct": initially_correct,
            "incorrect": initially_incorrect,
            "unscored": total - len(initial_scored),
            "scored_denominator": len(initial_scored),
            "rate": safe_rate(initially_correct, len(initial_scored)),
        },
        "esv_activation": {
            "activated": len(activated),
            "not_activated_initially_correct": sum(r.get("esv_activated") is False and r.get("initial_correct") is True for r in records),
            "rate_of_all_instances": safe_rate(len(activated), total),
            "rate_of_initially_incorrect": safe_rate(len(activated), initially_incorrect),
        },
        "recognition": {
            "true_recognitions": recognition_true,
            "false_recognitions": len(recognition_scored) - recognition_true,
            "unscored": len(activated) - len(recognition_scored),
            "scored_denominator": len(recognition_scored),
            "recognition_rate": safe_rate(recognition_true, len(recognition_scored)),
            "note": "Only initially incorrect/scoring-triggered cases are eligible; TRUE counts as recognized failure.",
        },
        "diagnosis": {
            "completed_calls": sum(bool(r.get("stages", {}).get("diagnosis", {}).get("ok")) for r in activated),
            "self_assessed_plausible": diagnosis_self_true,
            "self_assessed_denominator": len(diagnosis_assessed),
            "self_assessed_plausibility_rate": safe_rate(diagnosis_self_true, len(diagnosis_assessed)),
            "note": "Self-assessment only; not an independent correctness judgment.",
        },
        "alternative_strategy": {
            "completed_calls": sum(bool(r.get("stages", {}).get("alternative_strategy", {}).get("ok")) for r in activated),
            "self_assessed_valid": strategy_self_true,
            "self_assessed_denominator": len(strategy_assessed),
            "self_assessed_validity_rate": safe_rate(strategy_self_true, len(strategy_assessed)),
            "note": "Self-assessment only; not an independent correctness judgment.",
        },
        "recovery": {
            "correct": recovery_correct,
            "incorrect": len(recovery_scored) - recovery_correct,
            "unscored": len(activated) - len(recovery_scored),
            "scored_denominator": len(recovery_scored),
            "recovery_success_rate": safe_rate(recovery_correct, len(recovery_scored)),
        },
        "final_accuracy": {
            "correct": final_correct,
            "incorrect": len(final_scored) - final_correct,
            "unscored": total - len(final_scored),
            "scored_denominator": len(final_scored),
            "rate": safe_rate(final_correct, len(final_scored)),
        },
        "accuracy_change_vs_initial": {
            "initial_correct_on_final_scored_subset": sum(r.get("initial_correct") is True for r in final_scored),
            "final_correct_on_same_subset": final_correct,
            "absolute_change_pp": round(100 * (final_correct - sum(r.get("initial_correct") is True for r in final_scored)) / len(final_scored), 2) if final_scored else None,
            "note": "Percentage-point change on instances with a scored final answer; initial successes remain unchanged.",
        },
        "execution": {
            "api_error_calls": api_errors,
            "invalid_json_calls": invalid_json,
            "mean_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "total_latency_seconds": round(sum(latencies), 3) if latencies else 0,
            "tokens": stage_token_totals(records),
        },
    }


def calculate_metrics(records: list, metadata: Dict[str, Any]) -> Dict[str, Any]:
    originals = [r for r in records if r.get("problem_type") == "original"]
    perturbed = [r for r in records if r.get("problem_type") == "perturbed"]
    return {
        "experiment": {
            "method": "ESV",
            "model": metadata.get("model"),
            "input_file": metadata.get("input_file"),
            "expected_instances": metadata.get("expected_instances"),
            "completed_instances": len(records),
            "started_at": metadata.get("started_at"),
            "last_updated_at": utc_now(),
            "prompt_version": "esv_v1_2026-10-13",
            "direct_qa_instruction": DIRECT_QA_INSTRUCTION,
        },
        "overall": group_metrics(records),
        "original_problems": group_metrics(originals),
        "perturbed_problems": group_metrics(perturbed),
        "metric_interpretation": {
            "objective_metrics": ["initial_accuracy", "recovery.recovery_success_rate", "final_accuracy", "accuracy_change_vs_initial"],
            "recognition_metric": "Recognition is scored against the local frozen-answer checker: on an objectively incorrect initial answer, TRUE means the model recognized likely failure.",
            "diagnosis_and_strategy_metrics": "These are explicitly labeled model self-assessments. Do not report them as objectively verified correctness without a separate judge or deterministic rubric.",
            "gold_answer_policy": "Gold answers are used only in local evaluation and are never included in model prompts.",
        },
    }


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def load_existing_results(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("results", []) if isinstance(data, dict) else []
        return {r["instance_id"]: r for r in rows if isinstance(r, dict) and r.get("instance_id")}
    except (json.JSONDecodeError, OSError):
        print(f"Warning: could not read existing results at {path}; starting a new results file.", file=sys.stderr)
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the six-stage ESV pipeline on frozen original/perturbed pairs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to frozen_dataset.json")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Output detailed results JSON")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="Output aggregate metrics JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model ID (or set ESV_MODEL)")
    parser.add_argument("--limit-pairs", type=int, default=None, help="Optional small pilot limit")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing result records and rerun all instances")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is not set. Put it in the project .env or environment.", file=sys.stderr)
        return 2
    if not args.input.exists():
        print(f"ERROR: frozen dataset not found: {args.input}", file=sys.stderr)
        return 2

    try:
        dataset = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: unable to read frozen dataset: {exc}", file=sys.stderr)
        return 2
    pairs = dataset.get("pairs") if isinstance(dataset, dict) else None
    if not isinstance(pairs, list):
        print("ERROR: expected a top-level 'pairs' list in the frozen dataset.", file=sys.stderr)
        return 2
    if args.limit_pairs is not None:
        pairs = pairs[: max(0, args.limit_pairs)]

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=REQUEST_TIMEOUT,
        max_retries=0,  # explicit retries above are logged and controlled here
    )
    records_by_id = {} if args.no_resume else load_existing_results(args.results)
    expected_instances = len(pairs) * 2
    started_at = utc_now()

    tasks = []
    for pair in pairs:
        pair_id = pair.get("id", len(tasks) // 2)
        tasks.append((pair, "original", pair.get("original_problem", ""), pair.get("original_gold"), f"{pair_id}_original"))
        tasks.append((pair, "perturbed", pair.get("perturbed_problem", ""), pair.get("answer_prime"), f"{pair_id}_perturbed"))

    print(f"ESV model: {args.model}")
    print(f"Pairs: {len(pairs)} | Problem instances: {expected_instances}")
    print(f"Results: {args.results}\nMetrics: {args.metrics}")

    for index, (pair, kind, problem, gold, instance_id) in enumerate(tasks, start=1):
        if instance_id in records_by_id and records_by_id[instance_id].get("model") == args.model:
            print(f"[{index}/{expected_instances}] SKIP {instance_id} (already recorded for this model; use --no-resume to rerun)")
            continue
        print(f"[{index}/{expected_instances}] Running {instance_id}...")
        pair_metadata = {
            "original_strategy": pair.get("original_strategy"),
            "changed_condition": pair.get("changed_condition"),
            "alternative_strategy": pair.get("alternative_strategy"),
        } if kind == "perturbed" else {}
        try:
            record = run_esv_for_instance(client, args.model, pair, kind, problem, gold, pair_metadata)
        except KeyboardInterrupt:
            print("\nInterrupted. Saving completed results before exit.")
            break
        except Exception as exc:
            record = {
                "instance_id": instance_id,
                "pair_id": pair.get("id"),
                "problem_type": kind,
                "problem": extract_question(problem),
                "gold_answer": value_as_text(gold),
                "model": args.model,
                "initial_correct": None,
                "esv_activated": None,
                "final_correct": None,
                "fatal_error": f"{type(exc).__name__}: {exc}",
                "stages": {},
                "finished_at": utc_now(),
            }
        records_by_id[instance_id] = record
        ordered_records = [records_by_id[k] for _, _, _, _, k in tasks if k in records_by_id]
        metadata = {
            "model": args.model,
            "input_file": str(args.input.resolve()),
            "expected_instances": expected_instances,
            "started_at": started_at,
        }
        save_json(args.results, {"metadata": metadata, "results": ordered_records})
        save_json(args.metrics, calculate_metrics(ordered_records, metadata))
        print(f"    initial_correct={record.get('initial_correct')} | ESV={record.get('esv_activated')} | final_correct={record.get('final_correct')}")

    ordered_records = [records_by_id[k] for _, _, _, _, k in tasks if k in records_by_id]
    metadata = {
        "model": args.model,
        "input_file": str(args.input.resolve()),
        "expected_instances": expected_instances,
        "started_at": started_at,
    }
    save_json(args.results, {"metadata": metadata, "results": ordered_records})
    save_json(args.metrics, calculate_metrics(ordered_records, metadata))
    print(f"\nFinished with {len(ordered_records)}/{expected_instances} instances recorded.")
    print(f"Detailed results: {args.results.resolve()}")
    print(f"Metrics: {args.metrics.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
