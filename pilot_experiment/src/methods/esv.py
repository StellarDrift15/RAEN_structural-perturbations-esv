#!/usr/bin/env python3
"""End-to-end Explicit Strategy Verification (ESV) evaluation on the frozen paired dataset.

Pipeline, for every original and perturbed instance:

  Stage 0  initial_direct_qa         The model solves the problem. The answer is scored locally against the frozen
                                     gold answer. If it is correct, the instance is done.

  If the initial answer is wrong (a genuine failure), ESV runs all five stages:
  Stage 1  esv_recognition           Blind: the model is NOT told the answer is wrong. It names the strategy it used,
                                     lists the conditions that strategy needs, verifies each against the problem's
                                     stated conditions, and decides whether its solution failed.
  Stage 2  esv_diagnosis             The model is told the initial answer is incorrect (never the correct answer) and
                                     locates the failed step and the problem condition it conflicts with.
  Stage 3  esv_alternative_strategy  A strategy that differs from the failed one, with every precondition explicitly
                                     verified against the problem; one retry if it fails its own verification.
  Stage 4  esv_recovery              A fresh solve with the verified strategy. It does not see the failed reasoning,
                                     so it cannot anchor on it; it is told which answer is known to be wrong.
  Stage 5  esv_final_answer          Verifies the recovered answer against every problem condition (each option, for
                                     MCQs) and commits the final answer.

  Evaluation (local only; gold answers and frozen annotations never reach the solver)
    Answers are scored deterministically against the frozen gold answer.
    judge_diagnosis                  An independent judge model grades the diagnosis and the alternative strategy,
                                     using the gold answer and, for perturbed problems, the frozen changed-condition
                                     annotation.

Outputs (default: results/ beside the frozen dataset, one set per solver model):
  esv_results_<model>.json   full per-instance audit log (raw responses of every stage)
  esv_metrics_<model>.json   structured metrics (every rate stored as n / d / rate)
  esv_report_<model>.md      human-readable report of the research metrics
  esv_instances_<model>.csv  one row per instance, for spreadsheets

Environment (.env or env file in a parent folder of the repo, or the path in RAEN_ENV_FILE):
  OPENROUTER_API_KEY=...   optional: ESV_MODEL, ESV_JUDGE_MODEL

Run:
  python methods/esv.py                                  # full run (resumes automatically)
  python methods/esv.py --model meta-llama/llama-3.3-70b-instruct   # another solver, separate files
  python methods/esv.py --model deepseek/deepseek-v4-flash-0731 --max-tokens 16000   # reasoning solver
  python methods/esv.py --limit-pairs 2                  # small pilot
  python methods/esv.py --report-only                    # rebuild metrics/report, no API calls
"""

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Windows consoles default to cp1252; never let a stray symbol in a model answer crash a run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


ENV_FILE_NAMES = (".env", "env")  # Windows Explorer can save the file without its leading dot


def find_env_file() -> Optional[Path]:
    """The .env (OPENROUTER_API_KEY) is kept outside the repo, e.g. C:\\Ekansh_Folders\\Ekansh_Folders\\2026\\env.
    Uses the RAEN_ENV_FILE environment variable if set, else the nearest .env/env file in a parent folder."""
    if os.getenv("RAEN_ENV_FILE"):
        return Path(os.environ["RAEN_ENV_FILE"])
    return next((folder / name for folder in SCRIPT_DIR.parents for name in ENV_FILE_NAMES if (folder / name).is_file()), None)


ENV_FILE = find_env_file()
if ENV_FILE:
    load_dotenv(ENV_FILE)

DEFAULT_INPUT = PROJECT_DIR / "frozen_dataset.json"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results"
# Chosen ESV solver: small enough to fail on part of the dataset, so ESV has failures to recover. It is from a
# different family than the perturbation generator (DeepSeek) and the verifier model gpt-oss-120b.
DEFAULT_MODEL = os.getenv("ESV_MODEL", "qwen/qwen-2.5-7b-instruct")
# qwen-2.5-7b-instruct is not a reasoning model and does not accept OpenRouter's reasoning parameter, so send none.
DEFAULT_REASONING = os.getenv("ESV_REASONING", "default")
# Deliberately a different model family from the solver, so the judge does not grade its own reasoning style.
DEFAULT_JUDGE_MODEL = os.getenv("ESV_JUDGE_MODEL", "openai/gpt-oss-120b")

# Bump this whenever a prompt or scoring rule changes: --resume only reuses records with the same version.
PROMPT_VERSION = "esv_v4_2026-10-14"

# A response cut off by the cap is retried once with the cap doubled. 4000 suits a non-reasoning solver and stops a
# small model that loops from burning tokens; reasoning solvers (hidden thinking counts toward max_tokens) need far
# more, e.g. --max-tokens 16000 for deepseek-v4-flash.
SOLVER_MAX_TOKENS = int(os.getenv("ESV_MAX_TOKENS", "4000"))
JUDGE_MAX_TOKENS = int(os.getenv("ESV_JUDGE_MAX_TOKENS", "8000"))
MAX_RETRIES = int(os.getenv("ESV_MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.getenv("ESV_TIMEOUT_SECONDS", "300"))
MAX_STRATEGY_ATTEMPTS = 2  # alternative strategies proposed if the first fails its own verification
CONTEXT_CHAR_LIMIT = 6000  # initial reasoning passed to later stages is clipped to this length

# ---- Prompts ---------------------------------------------------------------------------------------
# Models answer in XML-style tags rather than JSON (same convention as prompts/prompt_functions.py):
# LaTeX backslashes break JSON strings, and a cut-off tagged reply can still be partially recovered.

# Keep this initial instruction synchronized with the Direct QA baseline prompt.
# Only the question text (and the answer-format note derived from the dataset) differs between instances.
DIRECT_QA_INSTRUCTION = """Solve the following problem carefully. Work through the mathematical reasoning step by step before selecting your final answer.

Identify the relevant concepts, derive the necessary equations, and verify that your reasoning satisfies the problem's conditions. For multiple-choice questions, evaluate the options as needed and select the correct option(s).

Return your response using exactly these tags:
<reasoning>Your step-by-step mathematical solution</reasoning>
<final_answer>The option letter(s) for MCQs, or the final numerical answer otherwise</final_answer>
For MCQs, always give the option letter(s) in final_answer, even when the answer is numerical. Do not omit the final answer."""

TAG_RULES = """
Format rules (strict):
- Output ONLY the tags listed, in order, each with an opening and a closing tag. No JSON, no Markdown fences, no text outside the tags.
- Write LaTeX normally with single backslashes; do not escape anything.
- Be concise: keep the written text focused (well under 400 words in total) and never leave a tag unfinished."""

TRUNCATION_NUDGE = "\n\nYour previous reply was cut off before all tags were written. Be much more concise this time and output ALL the tags."
FORMAT_NUDGE = "\n\nYour previous reply was missing required tags. Output ALL the tags listed above, each opened and closed."


def tag_block(fields: Dict[str, str]) -> str:
    return "\n".join(f"<{name}>{description}</{name}>" for name, description in fields.items())


def format_fields(values: Dict[str, str], fields: Dict[str, str]) -> str:
    return "\n".join(f"<{name}>{values.get(name, '')}</{name}>" for name in fields) if values else "[stage produced no output]"


def answer_format_note(answer_format: str, single_correct: bool) -> str:
    if answer_format == "numeric":
        return "Answer format: this question has a single numerical (integer) answer. final_answer must be just the number."
    if single_correct:
        return "Answer format: multiple choice, exactly one option is correct. final_answer must be the option letter, e.g. B."
    return ("Answer format: multiple choice, one or more options may be correct. Select every correct option. "
            "final_answer must be the letter(s) of all correct options, e.g. B or ACD.")


def build_initial_prompt(question: str, format_note: str) -> str:
    return f"{DIRECT_QA_INSTRUCTION}\n{TAG_RULES}\n\n{format_note}\n\nProblem:\n{question}"


def problem_block(instance: Dict[str, Any]) -> str:
    return f"Problem:\n{instance['question']}\n\n{instance['format_note']}"


def solution_context(instance: Dict[str, Any]) -> str:
    reasoning = instance["initial_reasoning"] or "[no reasoning was produced]"
    if len(reasoning) > CONTEXT_CHAR_LIMIT:
        reasoning = reasoning[:CONTEXT_CHAR_LIMIT] + " ...[truncated]"
    answer = instance["initial_answer"] or "[no final answer was produced]"
    return (f"{problem_block(instance)}\n\n"
            f"Your initial solution:\n<reasoning>{reasoning}</reasoning>\n<final_answer>{answer}</final_answer>")


def known_wrong_note(instance: Dict[str, Any]) -> str:
    if instance["initial_answer"]:
        return f"The initial final answer ({instance['initial_answer']}) has been checked and is INCORRECT."
    return "The initial solution has been checked and did not reach a correct final answer."


INDEPENDENT_FIELDS = {
    "method": "the method you used, in one sentence",
    "reasoning": "your step-by-step solution",
    "final_answer": "your answer in the required format",
}


def build_independent_prompt(instance: Dict[str, Any]) -> str:
    # Recognition step 1a: a fresh solve that never sees the initial solution, used as a consistency check.
    return (
        "You are performing Explicit Strategy Verification (ESV), stage 1a: independent re-solve.\n"
        "Solve the problem below from scratch. Before using any formula or case split, check that its conditions hold for "
        "the conditions stated in THIS problem. Before answering, check the result against every condition and, for MCQs, "
        "against each option.\n\n"
        f"{problem_block(instance)}\n{TAG_RULES}\n\nOutput exactly these tags:\n{tag_block(INDEPENDENT_FIELDS)}"
    )


RECOGNITION_FIELDS = {
    "strategy": "the key method or step the initial solution relied on",
    "assumption_checks": "one line per condition the strategy needs: condition - HOLDS or VIOLATED - one-line justification",
    "comparison": "whether the independent re-solve agrees with the initial answer and, if not, which one the checks support",
    "failure_recognized": "true or false",
    "reason": "one or two sentences explaining the decision",
}


def build_recognition_prompt(instance: Dict[str, Any], independent: Dict[str, str]) -> str:
    reasoning = (independent.get("reasoning") or "")[:CONTEXT_CHAR_LIMIT]
    return (
        "You are performing Explicit Strategy Verification (ESV), stage 1b: failure recognition.\n"
        "Verify the STRATEGY used in the initial solution:\n"
        "1. State the strategy the initial solution relied on (the key method, formula, case choice or step).\n"
        "2. List every condition that strategy needs in order to be valid: signs, domains, intervals, monotonicity, "
        "existence or uniqueness, the hypotheses of any formula or theorem used, which case applies, and so on.\n"
        "3. Check each condition against the conditions actually stated in THIS problem, not those of a similar familiar "
        "problem. Mark each HOLDS or VIOLATED with a one-line justification.\n"
        "4. Check that the final answer is consistent with the problem's conditions and, for MCQs, with the options.\n"
        "5. Compare with the independent re-solve below. If the answers differ, at least one is wrong: use your checks to "
        "decide which.\n"
        "Then decide whether the initial solution contains a failure.\n\n"
        f"{solution_context(instance)}\n\n"
        f"Independent re-solve:\n<method>{independent.get('method', '')}</method>\n<reasoning>{reasoning}</reasoning>\n"
        f"<final_answer>{independent.get('final_answer', '[no answer]')}</final_answer>\n{TAG_RULES}\n\n"
        f"Output exactly these tags:\n{tag_block(RECOGNITION_FIELDS)}"
    )


DIAGNOSIS_FIELDS = {
    "failed_step": "the specific step of the initial solution that is wrong",
    "root_cause": "the violated assumption, missed condition, misapplied formula or computational error behind it",
    "conflicting_condition": "the condition in the problem statement that the faulty step conflicts with, or NONE",
    "impact": "how this error changes the answer",
}


def build_diagnosis_prompt(instance: Dict[str, Any], recognition: Dict[str, str]) -> str:
    return (
        "You are performing Explicit Strategy Verification (ESV), stage 2: failure diagnosis.\n"
        f"{known_wrong_note(instance)} Identify precisely where and why the solution fails. Re-check the solution step "
        "by step: for each step, check whether it is justified by the conditions actually stated in the problem. Name the "
        "first step that is wrong, the assumption, condition or computation behind it, and the condition in the problem "
        "statement it conflicts with. Your earlier strategy check is included below; it may have missed the error.\n\n"
        f"{solution_context(instance)}\n\n"
        f"Your earlier strategy check:\n{format_fields(recognition, RECOGNITION_FIELDS)}\n{TAG_RULES}\n\n"
        f"Output exactly these tags:\n{tag_block(DIAGNOSIS_FIELDS)}"
    )


STRATEGY_FIELDS = {
    "failed_strategy": "the method the initial solution used, in one sentence",
    "candidate_strategies": "three DIFFERENT candidate methods, numbered 1-3, each with its key preconditions marked HOLDS or VIOLATED",
    "alternative_strategy": "the selected candidate (the best verified one), in a few sentences (do not carry out the full computation)",
    "precondition_checks": "one line per condition the selected method needs: condition - HOLDS or VIOLATED - justification from the problem text",
    "avoids_failure_because": "why the selected method avoids the diagnosed failure",
    "backup_strategy": "a second verified candidate, different from the selected one, for an independent cross-check",
    "strategy_verified": "true only if the selected method differs from the failed one, every precondition HOLDS, and it can reach the answer; otherwise false",
}


def build_strategy_prompt(instance: Dict[str, Any], diagnosis: Dict[str, str], rejected: List[Dict[str, str]]) -> str:
    previous = ""
    if rejected:
        previous = "\n\nThese strategies FAILED verification; propose a genuinely different one:\n" + "\n".join(
            f"- {r.get('alternative_strategy', '')} (checks: {r.get('precondition_checks', '')})" for r in rejected
        )
    return (
        "You are performing Explicit Strategy Verification (ESV), stage 3: alternative strategy with explicit verification.\n"
        f"{known_wrong_note(instance)} Propose a strategy that avoids the diagnosed failure and verify it BEFORE it is used:\n"
        "1. State the failed strategy.\n"
        "2. Propose THREE candidate methods that differ from the failed one and from each other: different techniques, "
        "case analyses, or ways of using the problem's conditions. Redoing the same method 'more carefully' does not count.\n"
        "3. For each candidate, check its key conditions against the conditions stated in the problem (HOLDS or VIOLATED). "
        "Be strict: mark VIOLATED whenever the problem does not guarantee the condition.\n"
        "4. Select the best verified candidate, list all its conditions in full, and explain how it avoids the diagnosed "
        "failure. Name a second verified candidate as the backup strategy.\n"
        "Do not carry out the full computation yet.\n\n"
        f"{solution_context(instance)}\n\n"
        f"Diagnosis:\n{format_fields(diagnosis, DIAGNOSIS_FIELDS)}{previous}\n{TAG_RULES}\n\n"
        f"Output exactly these tags:\n{tag_block(STRATEGY_FIELDS)}"
    )


RECOVERY_FIELDS = {
    "recovery_reasoning": "the complete step-by-step solution using the verified strategy",
    "final_answer": "your answer in the required format",
}


def build_recovery_prompt(instance: Dict[str, Any], diagnosis: Dict[str, str], strategy: Dict[str, str]) -> str:
    # The failed reasoning is deliberately left out so the model solves afresh instead of re-deriving it.
    return (
        "You are performing Explicit Strategy Verification (ESV), stage 4: recovery.\n"
        "Solve the problem from scratch, step by step, using the verified strategy below. Apply every condition stated in "
        f"the problem. {known_wrong_note(instance)} Do not return that answer unless your new derivation proves it.\n\n"
        f"{problem_block(instance)}\n\n"
        f"Diagnosed failure of the previous attempt:\n{format_fields(diagnosis, DIAGNOSIS_FIELDS)}\n\n"
        f"Verified strategy to use:\n{format_fields(strategy, STRATEGY_FIELDS)}\n{TAG_RULES}\n\n"
        f"Output exactly these tags:\n{tag_block(RECOVERY_FIELDS)}"
    )


def build_cross_check_prompt(instance: Dict[str, Any], diagnosis: Dict[str, str], strategy: Dict[str, str]) -> str:
    # Stage 5a: an independent second derivation with the backup strategy; it does not see the recovery.
    backup = strategy.get("backup_strategy") or "a method different from the one below"
    return (
        "You are performing Explicit Strategy Verification (ESV), stage 5a: independent cross-check.\n"
        "Solve the problem from scratch, step by step, using the BACKUP strategy below (not the selected one). Apply every "
        f"condition stated in the problem. {known_wrong_note(instance)} Do not return that answer unless your derivation "
        "proves it.\n\n"
        f"{problem_block(instance)}\n\n"
        f"Diagnosed failure of the first attempt:\n{format_fields(diagnosis, DIAGNOSIS_FIELDS)}\n\n"
        f"Backup strategy to use:\n{backup}\n\nSelected strategy (do NOT use this one):\n{strategy.get('alternative_strategy', '')}\n"
        f"{TAG_RULES}\n\nOutput exactly these tags:\n{tag_block(RECOVERY_FIELDS)}"
    )


FINAL_FIELDS = {
    "answer_checks": "one line per check: for MCQs each option - CORRECT or INCORRECT - reason; otherwise each problem condition - SATISFIED or VIOLATED",
    "final_answer": "the verified final answer in the required format",
}


def build_final_prompt(instance: Dict[str, Any], solutions: List[Dict[str, str]]) -> str:
    # Stage 5b: only reached when the recovery and the cross-check disagree (or one produced no answer).
    shown = "\n\n".join(f"Derivation {i}:\n{format_fields(s, RECOVERY_FIELDS)}" for i, s in enumerate(solutions, start=1))
    return (
        "You are performing Explicit Strategy Verification (ESV), stage 5b: final answer adjudication.\n"
        "The derivations below were produced independently and do not agree. Check each one step by step against the "
        "problem's conditions. For a multiple-choice question, check EVERY option independently and keep exactly the "
        "options that are correct. Otherwise, substitute the candidate answers back into every condition of the problem. "
        f"Commit to the answer the checks support. {known_wrong_note(instance)}\n\n"
        f"{problem_block(instance)}\n\n{shown}\n{TAG_RULES}\n\n"
        f"Output exactly these tags:\n{tag_block(FINAL_FIELDS)}"
    )


JUDGE_FIELDS = {
    "diagnosis_verdict": "CORRECT, PARTIAL or INCORRECT",
    "diagnosis_explanation": "one or two sentences",
    "strategy_verdict": "VALID or INVALID",
    "strategy_explanation": "one or two sentences",
}


def build_judge_prompt(instance: Dict[str, Any], diagnosis: Dict[str, str], strategy: Dict[str, str]) -> str:
    annotation = instance.get("frozen_pair_annotations") or {}
    reference_analysis = ""
    if annotation.get("changed_condition"):
        reference_analysis = (
            "\n\nReference failure analysis (this problem is a structural perturbation of a standard problem):\n"
            f"- Standard strategy that worked for the original problem: {annotation.get('original_strategy')}\n"
            f"- Condition that was changed: {annotation.get('changed_condition')}\n"
            f"- A correct alternative strategy for this problem: {annotation.get('alternative_strategy')}"
        )
    return (
        "You are an expert mathematics grader. A model solved the problem below and got the WRONG answer. Without seeing "
        "the reference answer, it then diagnosed its own failure and proposed an alternative strategy. Grade both.\n\n"
        f"Problem:\n{instance['question']}\n\nReference answer: {instance['gold']}{reference_analysis}\n\n"
        f"Model's initial (incorrect) solution:\n<reasoning>{(instance['initial_reasoning'] or '[none]')[:CONTEXT_CHAR_LIMIT]}</reasoning>\n"
        f"<final_answer>{instance['initial_answer'] or '[none]'}</final_answer>\n\n"
        f"Model's diagnosis:\n{format_fields(diagnosis, DIAGNOSIS_FIELDS)}\n\n"
        f"Model's alternative strategy:\n{format_fields(strategy, STRATEGY_FIELDS)}\n\n"
        "Grading rules:\n"
        "- diagnosis_verdict CORRECT: it identifies a genuine error in the initial solution that actually explains why its "
        "answer is wrong. For perturbed problems, recognising that the standard strategy was misapplied because of the "
        "changed condition counts, as does any other genuine, decisive error.\n"
        "- PARTIAL: it points at the right step or area but the stated cause is vague or only partly right.\n"
        "- INCORRECT: it flags something that is not an error, an irrelevant issue, or nothing concrete.\n"
        "- strategy_verdict VALID: the alternative strategy, executed correctly, leads to the reference answer. Otherwise INVALID.\n"
        f"{TAG_RULES}\n\nOutput exactly these tags:\n{tag_block(JUDGE_FIELDS)}"
    )


# ---- Model calls and parsing -----------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def value_as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Fallback for models that answer in JSON despite the tag instructions."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = cleaned[start:end + 1]
    for attempt in (candidate, re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", candidate)):  # 2nd try: escape raw LaTeX
        try:
            value = json.loads(attempt, strict=False)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def parse_tagged(text: str, fields: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Returns ({field: value}, [fields recovered from an unclosed tag]). Takes the last occurrence of each tag."""
    if not isinstance(text, str) or not text.strip():
        return {}, []
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    out: Dict[str, str] = {}
    for name in fields:
        matches = re.findall(rf"<{name}>(.*?)</{name}>", text, re.DOTALL | re.IGNORECASE)
        if matches and matches[-1].strip():
            out[name] = matches[-1].strip()
    partial = []
    # A reply cut off mid-field leaves the last tag unclosed; keep what was written.
    next_tag = re.compile(r"<(?:%s)>" % "|".join(fields), re.IGNORECASE)
    for name in fields:
        if name in out:
            continue
        openings = list(re.finditer(rf"<{name}>", text, re.IGNORECASE))
        if openings:
            tail = text[openings[-1].end():]
            following = next_tag.search(tail)
            value = (tail[:following.start()] if following else tail).strip()
            if value:
                out[name] = value
                partial.append(name)
    if not out:
        parsed_json = parse_json_object(text)
        if parsed_json:
            out = {name: value_as_text(parsed_json[name]) for name in fields if value_as_text(parsed_json.get(name))}
    return out, partial


def call_stage(
    client: OpenAI,
    model: str,
    stage: str,
    prompt: str,
    fields: Dict[str, str],
    required: List[str],
    reasoning: Optional[Dict[str, Any]],
    max_tokens: int,
) -> Dict[str, Any]:
    """Calls the model, retrying on API errors, missing tags and truncation. Token usage is summed over attempts."""
    names = list(fields)
    cap, suffix = max_tokens, ""
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    latency_total, errors = 0.0, []
    last: Optional[Dict[str, Any]] = None
    for attempt in range(1, MAX_RETRIES + 1):
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt + suffix}],
            "max_tokens": cap,
            "temperature": 0,
        }
        if reasoning is not None:
            kwargs["extra_body"] = {"reasoning": reasoning}
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:  # API/provider errors are recorded, not silently swallowed.
            latency_total += time.perf_counter() - started
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            time.sleep(min(3 * attempt, 10))
            continue
        latency_total += time.perf_counter() - started
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = getattr(response, "usage", None)
        for key in usage_total:
            value = getattr(usage, key, None) if usage else None
            if isinstance(value, (int, float)):
                usage_total[key] += value
        parsed, partial = parse_tagged(content, names)
        finish_reason = getattr(choice, "finish_reason", None)
        last = {"raw_response": content, "parsed": parsed, "partial_fields": partial, "finish_reason": finish_reason, "max_tokens": cap}
        complete = all(parsed.get(name) for name in required) and not any(name in partial for name in required)
        if complete:
            break
        if finish_reason == "length":
            cap = min(cap * 2, max_tokens * 2)
            suffix = TRUNCATION_NUDGE
        else:
            suffix = FORMAT_NUDGE
    if last is None:
        status = "api_error"
        last = {"raw_response": "", "parsed": {}, "partial_fields": [], "finish_reason": None, "max_tokens": cap}
    elif all(last["parsed"].get(name) for name in required):
        status = "completed" if not any(name in last["partial_fields"] for name in required) else "partial"
    else:
        status = "truncated" if last["finish_reason"] == "length" else "missing_fields"
    return {
        "stage": stage,
        "model": model,
        "status": status,
        **last,
        "attempts": attempt,
        "errors": errors,
        "usage": usage_total,
        "latency_seconds": round(latency_total, 3),
    }


def bool_value(value: Any) -> Optional[bool]:
    words = re.findall(r"[a-z]+", str(value or "").lower())
    if not words:
        return None
    if words[0] in {"true", "yes"}:
        return True
    if words[0] in {"false", "no"}:
        return False
    return None


def verdict_value(value: Any, labels: Tuple[str, ...]) -> Optional[str]:
    """First matching label; list longer labels first ('INCORRECT' before 'CORRECT')."""
    text = str(value or "").upper()
    for label in labels:
        if re.search(rf"\b{label}\b", text):
            return label
    return None


# ---- Answer scoring --------------------------------------------------------------------------------

def is_choice_gold(gold: Any) -> bool:
    return bool(re.fullmatch(r"\s*[A-Da-d]{1,4}\s*", str(gold or "")))


def strip_markup(text: str) -> str:
    text = re.sub(r"\\(?:boxed|text|mathrm|textbf)\s*\{([^{}]*)\}", r"\1", text)
    return text.replace("$", "").replace("**", "").strip()


CHOICE_PREFIX = re.compile(r"^\s*(?:the\s+)?(?:final\s+)?(?:correct\s+)?(?:answer|options?|choices?)?\s*(?:is|are)?\s*[:=\-]?\s*", re.IGNORECASE)
CHOICE_LETTER = re.compile(r"\(?\s*([A-D])\s*\)?(?![A-Za-z0-9])")
CHOICE_SEPARATOR = re.compile(r"\s*(?:,|;|&|/|\band\b|\s)\s*(?:options?\s*)?", re.IGNORECASE)


def normalize_choice(value: Any) -> Optional[str]:
    """'C) -2 tan θ' -> 'C', 'B and C (h(0)=16)' -> 'BC', '(A), (B), (C)' -> 'ABC', 'AC' -> 'AC'."""
    text = strip_markup(str(value or ""))
    compact = re.sub(r"[\s,;&()\[\]{}.*]+", "", text)
    if re.fullmatch(r"[A-Da-d]{1,4}", compact):
        return "".join(sorted(set(compact.upper())))
    # Read a leading run of option letters, stopping at the first non-letter content (an option's value).
    position = CHOICE_PREFIX.match(text).end()
    letters = []
    while True:
        match = CHOICE_LETTER.match(text, position)
        if not match:
            break
        letters.append(match.group(1))
        position = match.end()
        separator = CHOICE_SEPARATOR.match(text, position)
        if not separator:
            break
        position = separator.end()
    if letters:
        return "".join(sorted(set(letters)))
    markers = re.findall(r"(?:option\s*|\()([A-D])(?:\)|\b)", text, flags=re.IGNORECASE)
    return "".join(sorted(set(m.upper() for m in markers))) if markers else None


def normalize_number(value: Any) -> Optional[Decimal]:
    text = strip_markup(str(value or "")).replace(",", "")
    text = re.sub(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    if "=" in text:
        text = text.rsplit("=", 1)[1]
    fraction = re.search(r"\(?\s*([-+]?\d+(?:\.\d+)?)\s*\)?\s*/\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?", text)
    if fraction:
        denominator = Decimal(fraction.group(2))
        return Decimal(fraction.group(1)) / denominator if denominator else None
    number = re.search(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not number:
        return None
    try:
        return Decimal(number.group(0))
    except InvalidOperation:
        return None


def score_answer(prediction: str, gold: str) -> Dict[str, Any]:
    """An empty or unreadable answer is scored as incorrect (a failure to answer is a failure), and flagged."""
    if is_choice_gold(gold):
        predicted, expected = normalize_choice(prediction), normalize_choice(gold)
        correct = predicted is not None and predicted == expected
    else:
        predicted, expected = normalize_number(prediction), normalize_number(gold)
        correct = predicted is not None and expected is not None and abs(predicted - expected) <= Decimal("1e-6")
        predicted = None if predicted is None else str(predicted)
        expected = None if expected is None else str(expected)
    return {"normalized_answer": predicted, "normalized_gold": expected, "correct": correct, "answer_readable": predicted is not None}


def extract_answer(call: Dict[str, Any]) -> str:
    answer = (call.get("parsed") or {}).get("final_answer", "")
    if answer:
        return answer
    # Last resort for replies that ignore the tag format: an explicit "final answer: ..." line only.
    matches = re.findall(r"final\s+answer\s*(?:is)?\s*[:=]\s*(.+)", call.get("raw_response") or "", flags=re.IGNORECASE)
    return matches[-1].strip() if matches else ""


# ---- Pipeline --------------------------------------------------------------------------------------

def extract_question(problem: str) -> str:
    """Remove the frozen original's dataset-specific wrapper, if present."""
    text = str(problem or "").strip()
    marker = "### Question:"
    if marker in text:
        text = text.split(marker, 1)[1]
    return text.strip()


class Runner:
    def __init__(self, client: OpenAI, model: str, judge_model: str, reasoning: Optional[Dict], judge_reasoning: Optional[Dict]):
        self.client = client
        self.model = model
        self.judge_model = judge_model
        self.reasoning = reasoning
        self.judge_reasoning = judge_reasoning
        self.max_tokens = SOLVER_MAX_TOKENS

    def solver(self, stage: str, prompt: str, fields: Dict[str, str], required: List[str]) -> Dict[str, Any]:
        return call_stage(self.client, self.model, stage, prompt, fields, required, self.reasoning, self.max_tokens)

    def judge(self, prompt: str) -> Dict[str, Any]:
        return call_stage(self.client, self.judge_model, "judge_diagnosis", prompt, JUDGE_FIELDS,
                          ["diagnosis_verdict", "strategy_verdict"], self.judge_reasoning, JUDGE_MAX_TOKENS)

    def initial_solve(self, pair: Dict[str, Any], kind: str) -> Dict[str, Any]:
        """Builds the instance record and runs stage 0. Used by both the full pipeline and --screen."""
        gold = value_as_text(pair.get("original_gold") if kind == "original" else pair.get("answer_prime"))
        answer_format = "choice" if is_choice_gold(gold) else "numeric"
        # Original JEE 'MCQ' items have exactly one correct option; perturbed items were generated under
        # "one or more options may be correct", so that is the honest instruction for them.
        single_correct = kind == "original" and pair.get("type") == "MCQ"
        instance: Dict[str, Any] = {
            "instance_id": f"{pair.get('id')}_{kind}",
            "pair_id": pair.get("id"),
            "problem_type": kind,
            "source": pair.get("source"),
            "source_index": pair.get("source_index"),
            "question_type": pair.get("type"),
            "answer_format": answer_format,
            "question": extract_question(pair.get("original_problem" if kind == "original" else "perturbed_problem", "")),
            "gold": gold,
            "model": self.model,
            "judge_model": self.judge_model,
            "prompt_version": PROMPT_VERSION,
            "started_at": utc_now(),
            # Frozen annotations are used only by the judge, never in solver prompts.
            "frozen_pair_annotations": {key: pair.get(key) for key in ("original_strategy", "changed_condition", "alternative_strategy")}
            if kind == "perturbed" else {},
            "stages": {},
        }
        instance["format_note"] = answer_format_note(answer_format, single_correct)

        initial = self.solver("initial_direct_qa", build_initial_prompt(instance["question"], instance["format_note"]),
                              {"reasoning": "", "final_answer": ""}, ["final_answer"])
        instance["stages"]["initial_direct_qa"] = initial
        instance["initial_reasoning"] = initial["parsed"].get("reasoning", "")
        instance["initial_answer"] = extract_answer(initial)
        initial_score = score_answer(instance["initial_answer"], gold)
        initial.update(initial_score)
        instance["initial"] = {"answer": instance["initial_answer"], **initial_score}
        return instance

    def run_instance(self, pair: Dict[str, Any], kind: str) -> Dict[str, Any]:
        instance = self.initial_solve(pair, kind)
        if instance["initial"]["correct"]:
            instance["esv"] = None  # ESV is only applied to genuine failures
            instance["final"] = {"answer": instance["initial_answer"], "correct": True}
        else:
            instance["esv"] = self.run_esv(instance)
            instance["final"] = {"answer": instance["esv"]["final_answer"], "correct": instance["esv"]["final_correct"]}
        instance["finished_at"] = utc_now()
        return instance

    def run_esv(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        """All five stages run on every failure, so each stage is measured on every failure."""
        stages, gold = instance["stages"], instance["gold"]
        wrong = instance["initial"]["normalized_answer"]

        # Stage 1: recognition, blind. A fresh re-solve gives a consistency check; then the strategy check.
        independent = self.solver("esv_independent_solve", build_independent_prompt(instance), INDEPENDENT_FIELDS, ["final_answer"])
        stages["esv_independent_solve"] = independent
        independent_answer = extract_answer(independent)
        independent.update(score_answer(independent_answer, gold))
        answers_disagree = independent["answer_readable"] and wrong is not None and independent["normalized_answer"] != wrong

        recognition = self.solver("esv_recognition", build_recognition_prompt(instance, independent["parsed"]),
                                  RECOGNITION_FIELDS, ["failure_recognized"])
        stages["esv_recognition"] = recognition
        model_flagged = bool_value(recognition["parsed"].get("failure_recognized")) is True

        diagnosis = self.solver("esv_diagnosis", build_diagnosis_prompt(instance, recognition["parsed"]),
                                DIAGNOSIS_FIELDS, ["failed_step", "root_cause"])
        stages["esv_diagnosis"] = diagnosis

        rejected: List[Dict[str, str]] = []
        for attempt in range(1, MAX_STRATEGY_ATTEMPTS + 1):
            strategy = self.solver("esv_alternative_strategy", build_strategy_prompt(instance, diagnosis["parsed"], rejected),
                                   STRATEGY_FIELDS, ["alternative_strategy", "strategy_verified"])
            strategy["strategy_attempt"] = attempt
            if rejected:
                strategy["rejected_strategies"] = rejected
            if bool_value(strategy["parsed"].get("strategy_verified")) is not False:
                break
            rejected.append(strategy["parsed"])
        stages["esv_alternative_strategy"] = strategy

        recovery = self.solver("esv_recovery", build_recovery_prompt(instance, diagnosis["parsed"], strategy["parsed"]),
                               RECOVERY_FIELDS, ["final_answer"])
        stages["esv_recovery"] = recovery
        recovery_answer = extract_answer(recovery)
        recovery.update(score_answer(recovery_answer, gold))

        # Stage 5: final answer by agreement of two independent derivations; adjudicate only if they disagree.
        cross_check = self.solver("esv_cross_check", build_cross_check_prompt(instance, diagnosis["parsed"], strategy["parsed"]),
                                  RECOVERY_FIELDS, ["final_answer"])
        stages["esv_cross_check"] = cross_check
        cross_answer = extract_answer(cross_check)
        cross_check.update(score_answer(cross_answer, gold))
        agreement = recovery["answer_readable"] and cross_check["normalized_answer"] == recovery["normalized_answer"]
        if agreement:
            final_answer = recovery_answer
        else:
            solutions = [s["parsed"] or {"final_answer": a} for s, a in ((recovery, recovery_answer), (cross_check, cross_answer)) if a]
            final = self.solver("esv_final_answer", build_final_prompt(instance, solutions), FINAL_FIELDS, ["final_answer"])
            stages["esv_final_answer"] = final
            # If adjudication yields no answer, keep the recovered one rather than abstaining.
            final_answer = extract_answer(final) or recovery_answer or cross_answer
            final.update(score_answer(final_answer, gold))
        final_score = score_answer(final_answer, gold)

        judge = self.judge(build_judge_prompt(instance, diagnosis["parsed"], strategy["parsed"]))
        stages["judge_diagnosis"] = judge

        return {
            # Recognized if the strategy check flags a failure or the independent re-solve contradicts the initial answer.
            "recognized": model_flagged or answers_disagree,
            "recognized_by_strategy_check": model_flagged,
            "recognized_by_disagreement": answers_disagree,
            "independent_answer": independent_answer,
            "independent_correct": independent["correct"],
            "cross_check_answer": cross_answer,
            "cross_check_correct": cross_check["correct"],
            "derivations_agree": agreement,
            "diagnosis_verdict": verdict_value(judge["parsed"].get("diagnosis_verdict"), ("INCORRECT", "CORRECT", "PARTIAL")),
            "strategy_verdict": verdict_value(judge["parsed"].get("strategy_verdict"), ("INVALID", "VALID")),
            "strategy_attempts": strategy["strategy_attempt"],
            "strategy_self_verified": bool_value(strategy["parsed"].get("strategy_verified")),
            "recovery_answer": recovery_answer,
            "recovery_correct": recovery["correct"],
            "recovery_repeated_initial": wrong is not None and recovery["normalized_answer"] == wrong,
            "final_answer": final_answer,
            "final_correct": final_score["correct"],
            "final_repeated_initial": wrong is not None and final_score["normalized_answer"] == wrong,
        }


# ---- Metrics ---------------------------------------------------------------------------------------

def ratio(n: int, d: int) -> Dict[str, Any]:
    return {"n": n, "d": d, "rate": round(n / d, 4) if d else None}


def group_metrics(records: List[Dict]) -> Dict[str, Any]:
    total = len(records)
    failures = [r for r in records if not r["initial"]["correct"]]
    recognized = [r for r in failures if r["esv"]["recognized"]]
    diagnosed = [r for r in failures if r["esv"]["diagnosis_verdict"] == "CORRECT"]
    valid_strategy = [r for r in failures if r["esv"]["strategy_verdict"] == "VALID"]
    recovered = [r for r in failures if r["esv"]["final_correct"]]
    agreed = [r for r in failures if r["esv"]["derivations_agree"]]
    return {
        "instances": total,
        "initial_solving": {
            "answer_accuracy": ratio(total - len(failures), total),
            "failure_rate": ratio(len(failures), total),
            "no_readable_answer": sum(not r["initial"]["answer_readable"] for r in records),
        },
        "failure_recognition": {
            "recognition_accuracy": ratio(len(recognized), len(failures)),
            "flagged_by_strategy_check": ratio(sum(r["esv"]["recognized_by_strategy_check"] for r in failures), len(failures)),
            "flagged_by_disagreement": ratio(sum(r["esv"]["recognized_by_disagreement"] for r in failures), len(failures)),
            "independent_resolve_correct": ratio(sum(r["esv"]["independent_correct"] for r in failures), len(failures)),
        },
        "failure_diagnosis": {
            "diagnosis_accuracy": ratio(len(diagnosed), len(failures)),
            "partial_diagnoses": sum(r["esv"]["diagnosis_verdict"] == "PARTIAL" for r in failures),
            "diagnosis_accuracy_given_recognized": ratio(sum(r in diagnosed for r in recognized), len(recognized)),
        },
        "alternative_strategy": {
            "strategy_accuracy": ratio(len(valid_strategy), len(failures)),
            "strategy_accuracy_given_correct_diagnosis": ratio(sum(r in valid_strategy for r in diagnosed), len(diagnosed)),
            "self_verified_rate": ratio(sum(r["esv"]["strategy_self_verified"] is True for r in failures), len(failures)),
            "needed_second_attempt": sum(r["esv"]["strategy_attempts"] > 1 for r in failures),
        },
        "failure_recovery": {
            "recovery_rate": ratio(len(recovered), len(failures)),
            "conditional_recovery_rate": ratio(sum(r["esv"]["final_correct"] for r in recognized), len(recognized)),
            "recovery_rate_given_valid_strategy": ratio(sum(r["esv"]["final_correct"] for r in valid_strategy), len(valid_strategy)),
            "correct_after_recovery_stage": ratio(sum(r["esv"]["recovery_correct"] for r in failures), len(failures)),
            "cross_check_correct": ratio(sum(r["esv"]["cross_check_correct"] for r in failures), len(failures)),
            "derivations_agree": ratio(len(agreed), len(failures)),
            "accuracy_when_derivations_agree": ratio(sum(r["esv"]["final_correct"] for r in agreed), len(agreed)),
            "accuracy_when_adjudicated": ratio(sum(r["esv"]["final_correct"] for r in failures if r not in agreed), len(failures) - len(agreed)),
            "correct_after_final_answer_stage": ratio(len(recovered), len(failures)),
            "final_stage_fixed": sum(r["esv"]["final_correct"] and not r["esv"]["recovery_correct"] for r in failures),
            "final_stage_broke": sum(r["esv"]["recovery_correct"] and not r["esv"]["final_correct"] for r in failures),
            "repeated_initial_wrong_answer": ratio(sum(r["esv"]["final_repeated_initial"] for r in failures), len(failures)),
        },
        "stage_funnel": {
            "failures": len(failures),
            "recognized": len(recognized),
            "recognized_and_diagnosed": sum(r in diagnosed for r in recognized),
            "recognized_diagnosed_valid_strategy": sum(r in diagnosed and r in valid_strategy for r in recognized),
            "recognized_diagnosed_valid_strategy_recovered": sum(r in diagnosed and r in valid_strategy and r["esv"]["final_correct"] for r in recognized),
        },
        "final_accuracy": {
            "initial": ratio(total - len(failures), total),
            "after_esv": ratio(sum(r["final"]["correct"] for r in records), total),
        },
    }


def robustness_metrics(originals: List[Dict], perturbed: List[Dict]) -> Dict[str, Any]:
    by_pair = {r["pair_id"]: r for r in originals}
    paired = [(by_pair[r["pair_id"]], r) for r in perturbed if r["pair_id"] in by_pair]
    out: Dict[str, Any] = {"complete_pairs": len(paired)}
    for label, correct in (("initial", lambda r: r["initial"]["correct"]), ("after_esv", lambda r: r["final"]["correct"])):
        acc_original = ratio(sum(correct(o) for o, _ in paired), len(paired))
        acc_perturbed = ratio(sum(correct(p) for _, p in paired), len(paired))
        drop = (round(100 * (acc_original["rate"] - acc_perturbed["rate"]), 2)
                if acc_original["rate"] is not None and acc_perturbed["rate"] is not None else None)
        out[label] = {
            "accuracy_original": acc_original,
            "accuracy_perturbed": acc_perturbed,
            "accuracy_drop_pp": drop,
            "pairs_both_correct": sum(correct(o) and correct(p) for o, p in paired),
            "pairs_original_only": sum(correct(o) and not correct(p) for o, p in paired),
            "pairs_perturbed_only": sum(correct(p) and not correct(o) for o, p in paired),
            "pairs_neither": sum(not correct(o) and not correct(p) for o, p in paired),
        }
    return out


def execution_metrics(records: List[Dict]) -> Dict[str, Any]:
    stages = [s for r in records for s in r.get("stages", {}).values()]
    status_counts: Dict[str, int] = {}
    for s in stages:
        status_counts[s.get("status", "unknown")] = status_counts.get(s.get("status", "unknown"), 0) + 1
    tokens = {key: sum((s.get("usage") or {}).get(key) or 0 for s in stages) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    return {
        "stage_calls": len(stages),
        "api_requests_including_retries": sum(s.get("attempts", 0) for s in stages),
        "stage_status_counts": status_counts,
        "tokens": tokens,
        "total_latency_seconds": round(sum(s.get("latency_seconds") or 0 for s in stages), 1),
    }


def scorable(records: List[Dict]) -> List[Dict]:
    return [r for r in records if not r.get("fatal_error") and "initial" in r and "final" in r]


def calculate_metrics(records: List[Dict], metadata: Dict[str, Any]) -> Dict[str, Any]:
    usable = scorable(records)
    originals = [r for r in usable if r["problem_type"] == "original"]
    perturbed = [r for r in usable if r["problem_type"] == "perturbed"]
    return {
        "experiment": {
            **metadata,
            "prompt_version": PROMPT_VERSION,
            "completed_instances": len(usable),
            "failed_instances": [r["instance_id"] for r in records if r not in usable],
            "last_updated_at": utc_now(),
        },
        "groups": {
            "overall": group_metrics(usable),
            "original": group_metrics(originals),
            "perturbed": group_metrics(perturbed),
        },
        "perturbation_robustness": robustness_metrics(originals, perturbed),
        "execution": execution_metrics(records),
        "definitions": {
            "actual_failure": "Initial answer does not match the frozen gold answer (an unreadable/missing answer counts as a failure). ESV runs only on these.",
            "recognized": "Stage 1 (blind: the model is not told its answer is wrong): the strategy check returned failure_recognized=true, or the independent re-solve gave a different answer than the initial one.",
            "correctly_diagnosed": "Independent judge graded the stage 2 diagnosis CORRECT, given the gold answer and (for perturbed problems) the frozen changed-condition annotation.",
            "valid_strategy": "Independent judge graded the stage 3 alternative strategy VALID (it leads to the gold answer when executed correctly).",
            "recovered": "A failure whose stage 5 final answer matches the gold answer.",
            "conditional_recovery_rate": "Recovered failures / failures recognized in stage 1.",
            "accuracy_drop_pp": "Accuracy on originals minus accuracy on perturbed, over complete pairs, in percentage points.",
        },
    }


# ---- Report ----------------------------------------------------------------------------------------

def fmt(r: Optional[Dict[str, Any]]) -> str:
    if not r:
        return "-"
    return "n/a (0/0)" if r["rate"] is None else f"{100 * r['rate']:.1f}% ({r['n']}/{r['d']})"


def fmt_pp(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+.1f} pp"


def build_report(metrics: Dict[str, Any], records: List[Dict]) -> str:
    g = metrics["groups"]
    cols = ("overall", "original", "perturbed")
    exp = metrics["experiment"]
    lines = [
        f"# ESV evaluation report: `{exp.get('model')}`",
        "",
        f"- Dataset: `{exp.get('input_file')}` ({exp.get('completed_instances')}/{exp.get('expected_instances')} instances scored)",
        f"- Judge model (grades diagnosis and alternative strategy only): `{exp.get('judge_model')}`",
        f"- Prompt version: `{exp.get('prompt_version')}`, last updated {exp.get('last_updated_at')}",
        "- Pipeline: the model solves each problem; every genuine failure then goes through all five ESV stages: "
        "recognition (strategy check + independent re-solve), diagnosis, alternative strategy (3 verified candidates), "
        "recovery, and final answer (agreement of two independent derivations, adjudicated if they disagree).",
        "",
    ]
    if exp.get("failed_instances"):
        lines += [f"> Instances that hit a fatal error and are excluded: {', '.join(exp['failed_instances'])}", ""]

    def table(rows: List[Tuple[str, Any]]) -> None:
        lines.extend(["| Metric | Overall | Original | Perturbed |", "|---|---|---|---|"])
        for label, getter in rows:
            lines.append(f"| {label} | " + " | ".join(getter(g[c]) for c in cols) + " |")
        lines.append("")

    o = g["overall"]
    failures = o["initial_solving"]["failure_rate"]["n"]
    lines += [
        "## Headline",
        "",
        f"- Initial accuracy: **{fmt(o['initial_solving']['answer_accuracy'])}**, so **{failures}** failures went through ESV.",
        f"- Recognition **{fmt(o['failure_recognition']['recognition_accuracy'])}**, diagnosis **{fmt(o['failure_diagnosis']['diagnosis_accuracy'])}**, "
        f"valid alternative strategy **{fmt(o['alternative_strategy']['strategy_accuracy'])}**, "
        f"recovery **{fmt(o['failure_recovery']['recovery_rate'])}**.",
        f"- Final accuracy: **{fmt(o['final_accuracy']['initial'])} -> {fmt(o['final_accuracy']['after_esv'])}** after ESV.",
        "",
    ]
    if failures < 10:
        lines += [f"> **Caution:** only {failures} failures; the stage rates below rest on that small denominator.", ""]

    lines += ["## 1. Initial solving performance", "", "Answer Accuracy = #Correct / #Total. Failure Rate = #Incorrect / #Total.", ""]
    table([
        ("Answer accuracy", lambda x: fmt(x["initial_solving"]["answer_accuracy"])),
        ("Failure rate", lambda x: fmt(x["initial_solving"]["failure_rate"])),
        ("No readable answer (counted as failure)", lambda x: str(x["initial_solving"]["no_readable_answer"])),
    ])

    lines += ["## 2. Failure recognition", "",
              "Recognition Accuracy = #Correctly recognized failures / #Actual failures. The model is not told its answer is wrong. "
              "A failure is recognized if the strategy check flags it or an independent re-solve contradicts the initial answer.", ""]
    table([
        ("Recognition accuracy", lambda x: fmt(x["failure_recognition"]["recognition_accuracy"])),
        ("... flagged by the strategy check", lambda x: fmt(x["failure_recognition"]["flagged_by_strategy_check"])),
        ("... flagged by re-solve disagreement", lambda x: fmt(x["failure_recognition"]["flagged_by_disagreement"])),
        ("Independent re-solve already correct", lambda x: fmt(x["failure_recognition"]["independent_resolve_correct"])),
    ])

    lines += ["## 3. Failure diagnosis", "",
              "Diagnosis Accuracy = #Correctly diagnosed failures / #Actual failures (judge-graded).", ""]
    table([
        ("Diagnosis accuracy", lambda x: fmt(x["failure_diagnosis"]["diagnosis_accuracy"])),
        ("Partial diagnoses", lambda x: str(x["failure_diagnosis"]["partial_diagnoses"])),
        ("Diagnosis accuracy, given recognized", lambda x: fmt(x["failure_diagnosis"]["diagnosis_accuracy_given_recognized"])),
    ])

    lines += ["## 4. Alternative strategy", "",
              "Strategy Accuracy = #Failures with a valid alternative strategy / #Actual failures (judge-graded). "
              "Self-verified = the model's own verification passed.", ""]
    table([
        ("Strategy accuracy", lambda x: fmt(x["alternative_strategy"]["strategy_accuracy"])),
        ("Strategy accuracy, given correct diagnosis", lambda x: fmt(x["alternative_strategy"]["strategy_accuracy_given_correct_diagnosis"])),
        ("Self-verified by the model", lambda x: fmt(x["alternative_strategy"]["self_verified_rate"])),
        ("Needed a second strategy", lambda x: str(x["alternative_strategy"]["needed_second_attempt"])),
    ])

    lines += ["## 5. Failure recovery", "",
              "Recovery Rate = #Recovered failures / #Actual failures. Conditional Recovery Rate = #Recovered / #Correctly recognized failures.", ""]
    table([
        ("Recovery rate", lambda x: fmt(x["failure_recovery"]["recovery_rate"])),
        ("Conditional recovery rate", lambda x: fmt(x["failure_recovery"]["conditional_recovery_rate"])),
        ("Recovery rate, given valid strategy", lambda x: fmt(x["failure_recovery"]["recovery_rate_given_valid_strategy"])),
        ("Correct after recovery stage", lambda x: fmt(x["failure_recovery"]["correct_after_recovery_stage"])),
        ("Cross-check (backup strategy) correct", lambda x: fmt(x["failure_recovery"]["cross_check_correct"])),
        ("Recovery and cross-check agree", lambda x: fmt(x["failure_recovery"]["derivations_agree"])),
        ("... accuracy when they agree", lambda x: fmt(x["failure_recovery"]["accuracy_when_derivations_agree"])),
        ("... accuracy when adjudicated", lambda x: fmt(x["failure_recovery"]["accuracy_when_adjudicated"])),
        ("Correct after final answer stage", lambda x: fmt(x["failure_recovery"]["correct_after_final_answer_stage"])),
        ("Final stage fixed / broke", lambda x: f"{x['failure_recovery']['final_stage_fixed']} / {x['failure_recovery']['final_stage_broke']}"),
        ("Repeated the known-wrong answer", lambda x: fmt(x["failure_recovery"]["repeated_initial_wrong_answer"])),
    ])

    lines += ["## 6. Stage funnel", "", "How many failures make it through each stage in sequence.", ""]
    table([
        ("Failures", lambda x: str(x["stage_funnel"]["failures"])),
        ("Recognized", lambda x: str(x["stage_funnel"]["recognized"])),
        ("... and correctly diagnosed", lambda x: str(x["stage_funnel"]["recognized_and_diagnosed"])),
        ("... and valid strategy", lambda x: str(x["stage_funnel"]["recognized_diagnosed_valid_strategy"])),
        ("... and recovered", lambda x: str(x["stage_funnel"]["recognized_diagnosed_valid_strategy_recovered"])),
    ])

    rob = metrics["perturbation_robustness"]
    lines += ["## 7. Perturbation robustness", "",
              f"Perturbation Accuracy Drop = Accuracy(original) - Accuracy(perturbed), over {rob['complete_pairs']} complete pairs.", "",
              "| Stage | Acc. original | Acc. perturbed | Drop | Pairs: both / orig only / pert only / neither |",
              "|---|---|---|---|---|"]
    for label, name in (("initial", "Initial"), ("after_esv", "After ESV")):
        r = rob[label]
        lines.append(f"| {name} | {fmt(r['accuracy_original'])} | {fmt(r['accuracy_perturbed'])} | {fmt_pp(r['accuracy_drop_pp'])} | "
                     f"{r['pairs_both_correct']} / {r['pairs_original_only']} / {r['pairs_perturbed_only']} / {r['pairs_neither']} |")
    lines.append("")

    usable = scorable(records)
    failed = [r for r in usable if r["esv"]]
    lines += ["## Failures through ESV", "",
              "`recog`: recognized in stage 1. `diag`/`strat`: judge verdicts. Y = correct.", "",
              "| Instance | Gold | Initial | recog | diag | strat | Recovery answer | Final answer | Recovered |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in failed:
        e = r["esv"]
        lines.append("| " + " | ".join([
            r["instance_id"], r["gold"], cell(r["initial"]["answer"]), yn(e["recognized"]), e["diagnosis_verdict"] or "-",
            e["strategy_verdict"] or "-", cell(e["recovery_answer"]), cell(e["final_answer"]), yn(e["final_correct"]),
        ]) + " |")
    solved = [r["instance_id"] for r in usable if not r["esv"]]
    lines += ["", f"Solved correctly at the start (no ESV needed): {', '.join(solved) or 'none'}", ""]

    ex = metrics["execution"]
    lines += ["## Execution", "",
              f"- Stage calls: {ex['stage_calls']} ({ex['api_requests_including_retries']} API requests including retries)",
              f"- Stage status counts: {ex['stage_status_counts']}",
              f"- Tokens: {ex['tokens']}",
              f"- Total model latency: {ex['total_latency_seconds']} s",
              "",
              "## Notes on validity", "",
              "- The solver never sees the gold answer or the frozen annotations; they are used only for local scoring and by the judge.",
              "- Recognition is blind. From diagnosis onward the model is told that its initial answer is incorrect (but never "
              "what the correct answer is), because ESV is applied only to failures. State this when reporting recovery rates.",
              "- Diagnosis and strategy accuracy rely on an LLM judge; spot-check `judge_diagnosis` entries in the results JSON.",
              ""]
    return "\n".join(lines)


def yn(value: Any) -> str:
    return "Y" if value is True else "N" if value is False else "-"


def cell(text: Any, limit: int = 40) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).replace("|", "\\|")
    return (text[:limit] + "...") if len(text) > limit else (text or "-")


def write_csv(path: Path, records: List[Dict]) -> None:
    columns = ["instance_id", "pair_id", "problem_type", "question_type", "answer_format", "gold", "initial_answer",
               "initial_correct", "esv_recognized", "judge_diagnosis", "judge_strategy", "strategy_attempts",
               "strategy_self_verified", "recovery_answer", "recovery_correct", "final_answer", "final_correct"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for r in scorable(records):
            e = r["esv"] or {}
            writer.writerow([
                r["instance_id"], r["pair_id"], r["problem_type"], r["question_type"], r["answer_format"], r["gold"],
                r["initial"]["answer"], r["initial"]["correct"], e.get("recognized"), e.get("diagnosis_verdict"),
                e.get("strategy_verdict"), e.get("strategy_attempts"), e.get("strategy_self_verified"),
                e.get("recovery_answer"), e.get("recovery_correct"), r["final"]["answer"], r["final"]["correct"],
            ])


# ---- Screening -------------------------------------------------------------------------------------
# Runs only stage 0 (the identical initial solve) for several candidate solvers, to pick models whose
# initial accuracy leaves headroom for recovery. Model specs are "id" or "id@reasoning", e.g. openai/gpt-oss-20b@low.

def parse_model_spec(spec: str, default_reasoning: str) -> Tuple[str, str]:
    model, _, reasoning = spec.partition("@")
    return model, reasoning or default_reasoning


def screen_models(client: OpenAI, specs: List[str], default_reasoning: str, pairs: List[Dict], output_dir: Path, workers: int,
                  max_tokens: int) -> int:
    tasks = [(pair, kind) for pair in pairs for kind in ("original", "perturbed")]
    jobs = []
    for spec in specs:
        model, reasoning = parse_model_spec(spec, default_reasoning)
        runner = Runner(client, model, "", reasoning_setting(reasoning), None)
        runner.max_tokens = max_tokens
        jobs += [(spec, runner, pair, kind) for pair, kind in tasks]
    print(f"Screening {len(specs)} models x {len(tasks)} instances = {len(jobs)} initial solves")

    def solve(spec: str, runner: Runner, pair: Dict, kind: str) -> Tuple[str, Dict]:
        try:
            return spec, runner.initial_solve(pair, kind)
        except Exception as exc:
            return spec, {"instance_id": f"{pair.get('id')}_{kind}", "problem_type": kind, "fatal_error": f"{type(exc).__name__}: {exc}"}

    by_model: Dict[str, List[Dict]] = {spec: [] for spec in specs}
    with ThreadPoolExecutor(max(1, workers)) as executor:
        futures = [executor.submit(solve, *job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            spec, record = future.result()
            by_model[spec].append(record)
            if done % 20 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} solves finished")

    summary = []
    for spec, records in by_model.items():
        ok = [r for r in records if not r.get("fatal_error")]
        stages = [r["stages"]["initial_direct_qa"] for r in ok]
        row = {"model": spec, "instances": len(records), "fatal_errors": len(records) - len(ok)}
        for label, subset in (("overall", ok), ("original", [r for r in ok if r["problem_type"] == "original"]),
                              ("perturbed", [r for r in ok if r["problem_type"] == "perturbed"])):
            row[f"accuracy_{label}"] = ratio(sum(r["initial"]["correct"] for r in subset), len(subset))
        row["unreadable_answers"] = sum(not r["initial"]["answer_readable"] for r in ok)
        row["incomplete_calls"] = sum(s["status"] != "completed" for s in stages)
        row["mean_completion_tokens"] = round(sum(s["usage"]["completion_tokens"] for s in stages) / len(stages)) if stages else None
        row["mean_latency_seconds"] = round(sum(s["latency_seconds"] for s in stages) / len(stages), 1) if stages else None
        acc = row["accuracy_overall"]["rate"]
        row["in_headroom_zone"] = acc is not None and 0.40 <= acc <= 0.75
        summary.append(row)
        records.sort(key=lambda r: (r.get("instance_id") or ""))
    summary.sort(key=lambda r: -(r["accuracy_overall"]["rate"] or 0))

    save_json(output_dir / "screening_results.json", {"prompt_version": PROMPT_VERSION, "summary": summary, "records": by_model})
    lines = [
        "# Solver screening (initial solve only)", "",
        f"{len(tasks)} instances from the frozen dataset; prompt version `{PROMPT_VERSION}`. "
        "Headroom zone = 40-75% overall accuracy: enough failures for ESV to recover, and enough competence to recover them.", "",
        "| Model | Overall | Original | Perturbed | Drop | Unreadable | Incomplete calls | Mean tokens | Mean latency | Headroom |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary:
        o, p = row["accuracy_original"]["rate"], row["accuracy_perturbed"]["rate"]
        drop = fmt_pp(round(100 * (o - p), 2)) if o is not None and p is not None else "n/a"
        lines.append(f"| `{row['model']}` | {fmt(row['accuracy_overall'])} | {fmt(row['accuracy_original'])} | {fmt(row['accuracy_perturbed'])} | "
                     f"{drop} | {row['unreadable_answers']} | {row['incomplete_calls']} | {row['mean_completion_tokens']} | "
                     f"{row['mean_latency_seconds']} s | {'yes' if row['in_headroom_zone'] else 'no'} |")
    report = "\n".join(lines) + "\n"
    (output_dir / "screening_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved: {(output_dir / 'screening_report.md').resolve()}")
    return 0


# ---- Driver ----------------------------------------------------------------------------------------

def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.split("/")[-1])


def reasoning_setting(value: str) -> Optional[Dict[str, Any]]:
    if value == "default":
        return None  # send nothing: the provider's default behaviour
    if value == "off":
        return {"enabled": False}
    return {"effort": value}


def load_records(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"Warning: could not read {path}; starting fresh.", file=sys.stderr)
        return {}
    return {r["instance_id"]: r for r in data.get("results", []) if isinstance(r, dict) and r.get("instance_id")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve the frozen original/perturbed pairs, run the five ESV stages on every failure, and report the research metrics.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to frozen_dataset.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for results, metrics, report and CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Solver model (OpenRouter ID)")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model for diagnosis grading (OpenRouter ID)")
    parser.add_argument("--reasoning", default=DEFAULT_REASONING, choices=["default", "off", "low", "medium", "high", "xhigh"],
                        help="Solver reasoning setting sent to OpenRouter ('default' sends nothing)")
    parser.add_argument("--max-tokens", type=int, default=SOLVER_MAX_TOKENS,
                        help="Solver output cap per call (doubled once on truncation); use ~16000 for reasoning solvers")
    parser.add_argument("--judge-reasoning", default="medium", choices=["default", "off", "low", "medium", "high", "xhigh"])
    parser.add_argument("--workers", type=int, default=4, help="Instances processed in parallel (lower if you hit 429s)")
    parser.add_argument("--limit-pairs", type=int, default=None, help="Only run the first N pairs (pilot)")
    parser.add_argument("--no-resume", action="store_true", help="Rerun every instance instead of reusing matching records")
    parser.add_argument("--report-only", action="store_true", help="Rebuild metrics/report/CSV from existing results; no API calls")
    parser.add_argument("--screen", nargs="+", metavar="MODEL[@REASONING]",
                        help="Only run the initial solve for these candidate solvers and write screening_report.md")
    args = parser.parse_args()

    try:
        dataset = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: unable to read frozen dataset {args.input}: {exc}", file=sys.stderr)
        return 2
    pairs = dataset.get("pairs") if isinstance(dataset, dict) else None
    if not isinstance(pairs, list):
        print("ERROR: expected a top-level 'pairs' list in the frozen dataset.", file=sys.stderr)
        return 2
    if args.limit_pairs is not None:
        pairs = pairs[: max(0, args.limit_pairs)]

    slug = model_slug(args.model)
    paths = {
        "results": args.output_dir / f"esv_results_{slug}.json",
        "metrics": args.output_dir / f"esv_metrics_{slug}.json",
        "report": args.output_dir / f"esv_report_{slug}.md",
        "csv": args.output_dir / f"esv_instances_{slug}.csv",
    }
    tasks = [(pair, kind) for pair in pairs for kind in ("original", "perturbed")]
    order = [f"{pair.get('id')}_{kind}" for pair, kind in tasks]
    metadata = {
        "method": "ESV (recognition, diagnosis, alternative strategy, recovery, final answer) on initial failures",
        "model": args.model,
        "judge_model": args.judge_model,
        "reasoning": args.reasoning,
        "judge_reasoning": args.judge_reasoning,
        "solver_max_tokens": args.max_tokens,
        "input_file": str(args.input.resolve()),
        "perturbation_generator_models": dataset.get("models"),
        "expected_instances": len(tasks),
        "started_at": utc_now(),
    }
    records = {} if args.no_resume else load_records(paths["results"])
    lock = threading.Lock()

    def write_outputs() -> List[Dict]:
        ordered = [records[k] for k in order if k in records]
        metrics = calculate_metrics(ordered, metadata)
        save_json(paths["results"], {"metadata": metadata, "results": ordered})
        save_json(paths["metrics"], metrics)
        paths["report"].write_text(build_report(metrics, ordered), encoding="utf-8")
        write_csv(paths["csv"], ordered)
        return ordered

    if args.report_only:
        ordered = write_outputs()
        print(f"Rebuilt outputs from {len(ordered)} records:\n" + "\n".join(f"  {p.resolve()}" for p in paths.values()))
        return 0

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print(f"ERROR: OPENROUTER_API_KEY is not set (.env used: {ENV_FILE or 'none found'}). Set RAEN_ENV_FILE to the "
              "path of your .env, or place the .env in a parent folder of this script.", file=sys.stderr)
        return 2
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=0)
    if args.screen:
        return screen_models(client, args.screen, args.reasoning, pairs, args.output_dir, args.workers, args.max_tokens)
    runner = Runner(client, args.model, args.judge_model, reasoning_setting(args.reasoning), reasoning_setting(args.judge_reasoning))
    runner.max_tokens = args.max_tokens

    def reusable(record: Optional[Dict]) -> bool:
        return bool(record) and record.get("prompt_version") == PROMPT_VERSION and record.get("model") == args.model \
            and record.get("judge_model") == args.judge_model and not record.get("fatal_error")

    pending = [(pair, kind) for pair, kind in tasks if not reusable(records.get(f"{pair.get('id')}_{kind}"))]
    print(f"Solver: {args.model} (reasoning={args.reasoning}, max_tokens={args.max_tokens}) | Judge: {args.judge_model}")
    print(f"Pairs: {len(pairs)} | Instances: {len(tasks)} | To run: {len(pending)} | Reused: {len(tasks) - len(pending)}")
    print(f"Outputs: {args.output_dir.resolve()}")

    def run_one(pair: Dict[str, Any], kind: str) -> Dict[str, Any]:
        try:
            return runner.run_instance(pair, kind)
        except Exception as exc:  # one broken instance must not kill the run
            return {"instance_id": f"{pair.get('id')}_{kind}", "pair_id": pair.get("id"), "problem_type": kind,
                    "model": args.model, "prompt_version": PROMPT_VERSION, "fatal_error": f"{type(exc).__name__}: {exc}",
                    "stages": {}, "finished_at": utc_now()}

    done = 0
    executor = ThreadPoolExecutor(max(1, args.workers))
    try:
        futures = {executor.submit(run_one, pair, kind): f"{pair.get('id')}_{kind}" for pair, kind in pending}
        for future in as_completed(futures):
            record = future.result()
            done += 1
            with lock:
                records[record["instance_id"]] = record
                write_outputs()
            if record.get("fatal_error"):
                print(f"[{done}/{len(pending)}] {record['instance_id']}: FATAL {record['fatal_error']}")
            elif not record["esv"]:
                print(f"[{done}/{len(pending)}] {record['instance_id']}: initial=Y (no ESV needed)")
            else:
                e = record["esv"]
                print(f"[{done}/{len(pending)}] {record['instance_id']}: initial=N -> recognized={yn(e['recognized'])} "
                      f"diagnosis={e['diagnosis_verdict'] or '-'} strategy={e['strategy_verdict'] or '-'} "
                      f"recovered={yn(e['final_correct'])}")
    except KeyboardInterrupt:
        print("\nInterrupted: saving completed instances (rerun to resume).")
        executor.shutdown(wait=False, cancel_futures=True)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        with lock:
            ordered = write_outputs()

    print(f"\nRecorded {len(scorable(ordered))}/{len(tasks)} instances.")
    print("\n".join(f"  {name}: {path.resolve()}" for name, path in paths.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
