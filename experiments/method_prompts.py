"""Prompts for Direct QA, CoT, ESV and the retry-with-feedback control.

The Direct QA, CoT and ESV stage prompts are the pilot's (pilot_experiment/src/methods/esv.py and
DQV_CoT_Code_strict.py) unchanged in wording. What changed:
  - the answer-format note comes from the question type (common.answer_format_note), so perturbed problems get the
    same instruction as their originals;
  - RETRY is new: the control that receives exactly the information ESV receives (the initial answer is wrong) but
    no ESV stages;
  - the judge's reference analysis distinguishes strategy-breaking from strategy-preserving perturbations;
  - ESV v2 (phase 3) has its own prompts below the v4 ones, which stay unchanged so Experiment 1 is reproducible;
  - the gold-free evaluation (run_open.py) reuses the v2 prompts with a self-flagged failure note, plus a Self-Refine
    control prompt.
"""

from typing import Any, Dict, List

import common

CONTEXT_CHAR_LIMIT = 6000  # initial reasoning passed to later stages is clipped to this length

TAG_RULES = """
Format rules (strict):
- Output ONLY the tags listed, in order, each with an opening and a closing tag. No JSON, no Markdown fences, no text outside the tags.
- Write LaTeX normally with single backslashes; do not escape anything.
- Be concise: keep the written text focused (well under 400 words in total) and never leave a tag unfinished."""

# CoT = ESV's own Stage 0 prompt. ESV's final answer therefore differs from CoT only by the ESV stages.
# The MCQ sentences below are sent only for choice formats. On an integer/numeric benchmark they contradict the
# answer-format note and a weak model follows them instead: Gemma 3 4B answered a bare option letter ("A" in 138
# of 168 cases) on GSM8K, losing 10.6% of its samples (checked 2026-11-04). The MCQ text is unchanged, so JEE
# MCQ prompts are byte-identical; the integer/numeric variants changed, so re-run those to compare with a
# frozen JEE experiment.
COT_INSTRUCTION = """Solve the following problem carefully. Work through the mathematical reasoning step by step before selecting your final answer.

Identify the relevant concepts, derive the necessary equations, and verify that your reasoning satisfies the problem's conditions. For multiple-choice questions, evaluate the options as needed and select the correct option(s).

Return your response using exactly these tags:
<reasoning>Your step-by-step mathematical solution</reasoning>
<final_answer>The option letter(s) for MCQs, or the final numerical answer otherwise</final_answer>
For MCQs, always give the option letter(s) in final_answer, even when the answer is numerical. Do not omit the final answer."""

DIRECT_INSTRUCTION = """Solve the following problem. Give ONLY the final answer: do not write any reasoning, working or explanation.

Return your response using exactly this tag:
<final_answer>The option letter(s) for MCQs, or the final numerical answer otherwise</final_answer>
For MCQs, always give the option letter(s) in final_answer, even when the answer is numerical. Do not omit the final answer."""

COT_INSTRUCTION_NUMERIC = """Solve the following problem carefully. Work through the mathematical reasoning step by step before selecting your final answer.

Identify the relevant concepts, derive the necessary equations, and verify that your reasoning satisfies the problem's conditions.

Return your response using exactly these tags:
<reasoning>Your step-by-step mathematical solution</reasoning>
<final_answer>The final answer as a number, and nothing else</final_answer>
Do not omit the final answer."""

DIRECT_INSTRUCTION_NUMERIC = """Solve the following problem. Give ONLY the final answer: do not write any reasoning, working or explanation.

Return your response using exactly this tag:
<final_answer>The final answer as a number, and nothing else</final_answer>
Do not omit the final answer."""


def cot_instruction(answer_format: str) -> str:
    """The CoT instruction for this answer format: the MCQ wording is sent only where options exist."""
    return COT_INSTRUCTION if common.is_choice_format(answer_format) else COT_INSTRUCTION_NUMERIC


def direct_instruction(answer_format: str) -> str:
    return DIRECT_INSTRUCTION if common.is_choice_format(answer_format) else DIRECT_INSTRUCTION_NUMERIC


COT_FIELDS = ["reasoning", "final_answer"]
DIRECT_FIELDS = ["final_answer"]


def build_cot_prompt(question: str, format_note: str, answer_format: str) -> str:
    return f"{cot_instruction(answer_format)}\n{TAG_RULES}\n\n{format_note}\n\nProblem:\n{question}"


def build_direct_prompt(question: str, format_note: str, answer_format: str) -> str:
    # Same layout as build_cot_prompt; only the instruction differs.
    return f"{direct_instruction(answer_format)}\n{TAG_RULES}\n\n{format_note}\n\nProblem:\n{question}"


def tag_block(fields: Dict[str, str]) -> str:
    return "\n".join(f"<{name}>{description}</{name}>" for name, description in fields.items())


def format_fields(values: Dict[str, str], fields: Dict[str, str]) -> str:
    return "\n".join(f"<{name}>{values.get(name, '')}</{name}>" for name in fields) if values else "[stage produced no output]"


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


# ---- Retry-with-feedback control -------------------------------------------------------------------

def build_retry_prompt(instance: Dict[str, Any]) -> str:
    """Control: the CoT prompt plus the SAME known-wrong note ESV gives its later stages, and nothing else."""
    return (f"{cot_instruction(instance['answer_format'])}\n{TAG_RULES}\n\n{instance['format_note']}\n\n"
            f"Note: {known_wrong_note(instance)} Solve the problem again from scratch.\n\nProblem:\n{instance['question']}")


# ---- ESV stages (pilot wording) --------------------------------------------------------------------

INDEPENDENT_FIELDS = {
    "method": "the method you used, in one sentence",
    "reasoning": "your step-by-step solution",
    "final_answer": "your answer in the required format",
}


def build_independent_prompt(instance: Dict[str, Any]) -> str:
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


# ---- ESV v2 ----------------------------------------------------------------------------------------
# Built from the v4 failure analysis (results/analysis/v4_failure_analysis.md): the proposal's six stages, with
# stage 5 producing K strategies that must differ from the failed one, and stage 6 solving once per strategy, checking
# the answer concretely (each option / substitution) and voting. No independent re-solve, cross-check or adjudicator.

def failure_note(instance: Dict[str, Any]) -> str:
    """What the stages after recognition are told about the initial answer: by default that it is known to be wrong
    (phase 3, gold-gated); in the gold-free evaluation, that the model's own check flagged it (instance["failure_note"])."""
    return instance.get("failure_note") or known_wrong_note(instance)


def answer_check_instruction(answer_format: str) -> str:
    """Concrete answer verification, shared by ESV v2 and the matched retry control."""
    if answer_format == "multiple":
        return ("Before answering, check EACH option (A), (B), (C), (D) independently against the problem and mark it TRUE or "
                "FALSE with a one-line reason. ONE OR MORE options may be correct: your final answer must contain every "
                "option you marked TRUE, so do not stop after finding the first one.")
    if answer_format == "single":
        return ("Before answering, check EACH option (A), (B), (C), (D) against the problem and mark it TRUE or FALSE with a "
                "one-line reason. Exactly one option is correct.")
    return ("Before answering, substitute your answer back into every condition and equation of the problem and confirm "
            "each one holds (or recompute the final quantity a second way). If a check fails, fix the solution first.")


V2_RECOGNITION_FIELDS = {
    "strategy": "the key method the initial solution relied on, in one sentence",
    "condition_checks": "one line per condition that method needs: condition - HOLDS or VIOLATED in THIS problem - one-line reason",
    "failure_recognized": "true or false",
    "reason": "one or two sentences explaining the decision",
}


def build_v2_recognition_prompt(instance: Dict[str, Any]) -> str:
    # Proposal stages 2-3 (strategy verification, failure recognition). Blind: not told the answer is wrong.
    return (
        "You are performing Explicit Strategy Verification (ESV), stages 2-3: strategy verification and failure recognition.\n"
        "1. State the method the initial solution relied on (the key formula, theorem, case choice or step).\n"
        "2. List every condition that method needs to be valid: signs, domains, intervals, existence, which case applies, "
        "the hypotheses of each formula or theorem used.\n"
        "3. Check each condition against what THIS problem actually states, not what a similar familiar problem states. "
        "Mark each HOLDS or VIOLATED with a one-line reason.\n"
        "4. Decide whether the initial solution fails.\n\n"
        f"{solution_context(instance)}\n{TAG_RULES}\n\nOutput exactly these tags:\n{tag_block(V2_RECOGNITION_FIELDS)}"
    )


V2_DIAGNOSIS_FIELDS = {
    "failed_step": "the first step of the initial solution that is wrong",
    "root_cause": "why it is wrong: violated assumption, ignored condition, misapplied formula or computational slip",
    "broken_condition": "the condition of THIS problem, quoted or paraphrased from the statement, that the failed step ignores or violates; COMPUTATION if it is only an arithmetic or algebra slip",
    "impact": "how this error changes the answer",
}


def build_v2_diagnosis_prompt(instance: Dict[str, Any], recognition: Dict[str, str]) -> str:
    return (
        "You are performing Explicit Strategy Verification (ESV), stage 4: failure diagnosis.\n"
        f"{failure_note(instance)} Find where and why the solution fails. Re-check it step by step against the "
        "conditions actually stated in the problem. Name the first wrong step, the cause, and the specific condition of the "
        "problem it breaks. Your earlier strategy check is below; it may have missed the error.\n\n"
        f"{solution_context(instance)}\n\n"
        f"Your earlier strategy check:\n{format_fields(recognition, V2_RECOGNITION_FIELDS)}\n{TAG_RULES}\n\n"
        f"Output exactly these tags:\n{tag_block(V2_DIAGNOSIS_FIELDS)}"
    )


def v2_strategy_fields(k: int) -> Dict[str, str]:
    fields = {"failed_strategy": "the method the initial solution used, in one sentence"}
    for i in range(1, k + 1):
        fields[f"strategy_{i}"] = ("a method in 2-4 sentences: what it does, how it respects the broken condition, and its "
                                   "key preconditions each marked HOLDS (do not carry out the computation)")
    return fields


def build_v2_strategy_prompt(instance: Dict[str, Any], diagnosis: Dict[str, str], k: int, too_similar: List[str]) -> str:
    feedback = ""
    if too_similar:
        feedback = ("\n\nYour previous proposal was rejected because these strategies were too close to the failed method "
                    "or to each other:\n" + "\n".join(f"- {s}" for s in too_similar) +
                    "\nPropose genuinely different methods this time.")
    count = "ONE method" if k == 1 else f"{k} methods"
    return (
        "You are performing Explicit Strategy Verification (ESV), stage 5: alternative strategies.\n"
        f"{failure_note(instance)} Propose {count} for solving the problem that avoid the diagnosed failure.\n"
        "Requirements for every method:\n"
        "- It must NOT be the failed method again. Redoing the same method 'more carefully' does not count.\n"
        "- It must explicitly respect the broken condition named in the diagnosis.\n"
        "- Its key preconditions must all HOLD for the conditions stated in this problem.\n"
        + ("- The methods must differ from each other: different techniques, case analyses or ways of using the "
           "problem's conditions.\n" if k > 1 else "")
        + "Do not carry out the computation.\n\n"
        f"{solution_context(instance)}\n\n"
        f"Diagnosis:\n{format_fields(diagnosis, V2_DIAGNOSIS_FIELDS)}{feedback}\n{TAG_RULES}\n\n"
        f"Output exactly these tags:\n{tag_block(v2_strategy_fields(k))}"
    )


SOLVE_FIELDS = {
    "reasoning": "the complete step-by-step solution",
    "answer_checks": "the checks described above, one line each",
    "final_answer": "your answer in the required format",
}


def build_v2_solve_prompt(instance: Dict[str, Any], diagnosis: Dict[str, str], strategy: str) -> str:
    # Proposal stage 6, once per strategy. The failed reasoning is left out so the model solves afresh.
    return (
        "You are performing Explicit Strategy Verification (ESV), stage 6: recovery.\n"
        "Solve the problem from scratch, step by step, using the strategy below. Apply every condition stated in the "
        f"problem. {failure_note(instance)} Do not return that answer unless your new derivation proves it.\n"
        f"{answer_check_instruction(instance['answer_format'])}\n\n"
        f"{problem_block(instance)}\n\n"
        f"Diagnosed failure of the previous attempt:\n{format_fields(diagnosis, V2_DIAGNOSIS_FIELDS)}\n\n"
        f"Strategy to use:\n{strategy}\n{TAG_RULES}\n\nOutput exactly these tags:\n{tag_block(SOLVE_FIELDS)}"
    )


def build_matched_retry_prompt(instance: Dict[str, Any]) -> str:
    """Retry control with ESV v2's answer checks: the same known-wrong note and checks, but no ESV stages."""
    return (f"{cot_instruction(instance['answer_format'])}\n{TAG_RULES}\n\n{instance['format_note']}\n\n"
            f"Note: {failure_note(instance)} Solve the problem again from scratch. "
            f"{answer_check_instruction(instance['answer_format'])}\n\n"
            f"Problem:\n{instance['question']}\n\nOutput exactly these tags:\n{tag_block(SOLVE_FIELDS)}")


def known_wrong_followup(answer: str) -> str:
    """Appended once when a derivation returns the answer already known to be wrong (ESV v2 and matched retry)."""
    return (f"\n\nYour previous reply gave the final answer {answer}, which is the answer already known to be INCORRECT. "
            "Find the mistake and solve again; your final answer must be different and justified.")


V2_JUDGE_FIELDS = {
    "diagnosis_verdict": "CORRECT, PARTIAL or INCORRECT",
    "diagnosis_explanation": "one or two sentences",
    "strategy_verdicts": "one line per strategy, in order: VALID or INVALID",
    "strategy_explanation": "one or two sentences",
}


JUDGE_FIELDS = {
    "diagnosis_verdict": "CORRECT, PARTIAL or INCORRECT",
    "diagnosis_explanation": "one or two sentences",
    "strategy_verdict": "VALID or INVALID",
    "strategy_explanation": "one or two sentences",
}


JUDGE_DIAGNOSIS_RULES = (
    "- diagnosis_verdict CORRECT: it identifies a genuine error in the initial solution that actually explains why its "
    "answer is wrong. For perturbed problems, recognising that the standard strategy was misapplied because of the "
    "changed condition counts, as does any other genuine, decisive error.\n"
    "- PARTIAL: it points at the right step or area but the stated cause is vague or only partly right.\n"
    "- INCORRECT: it flags something that is not an error, an irrelevant issue, or nothing concrete.\n"
)


def judge_context(instance: Dict[str, Any]) -> str:
    """Problem, reference answer and analysis, and the initial solution: shared by the v4 and v2 judge prompts."""
    annotation = instance.get("frozen_pair_annotations") or {}
    reference_analysis = ""
    if annotation.get("perturbation_type") == "breaking":
        reference_analysis = (
            "\n\nReference failure analysis (this problem is a structural perturbation of a standard problem that breaks "
            "the standard strategy):\n"
            f"- Standard strategy that worked for the original problem: {annotation.get('original_strategy')}\n"
            f"- Condition that was changed: {annotation.get('changed_condition')}\n"
            f"- A correct alternative strategy for this problem: {annotation.get('alternative_strategy')}"
        )
    elif annotation.get("perturbation_type") == "preserving":
        reference_analysis = (
            "\n\nReference analysis (this problem is a perturbation of a standard problem in which the standard strategy "
            "still works unchanged, so a wrong answer comes from misapplying or mis-executing it):\n"
            f"- Standard strategy, which applies to this problem: {annotation.get('original_strategy')}\n"
            f"- What was changed from the original: {annotation.get('changed_condition')}"
        )
    return (
        f"Problem:\n{instance['question']}\n\nReference answer: {instance['gold']}{reference_analysis}\n\n"
        f"Model's initial (incorrect) solution:\n<reasoning>{(instance['initial_reasoning'] or '[none]')[:CONTEXT_CHAR_LIMIT]}</reasoning>\n"
        f"<final_answer>{instance['initial_answer'] or '[none]'}</final_answer>"
    )


def build_judge_prompt(instance: Dict[str, Any], diagnosis: Dict[str, str], strategy: Dict[str, str]) -> str:
    return (
        "You are an expert mathematics grader. A model solved the problem below and got the WRONG answer. Without seeing "
        "the reference answer, it then diagnosed its own failure and proposed an alternative strategy. Grade both.\n\n"
        f"{judge_context(instance)}\n\n"
        f"Model's diagnosis:\n{format_fields(diagnosis, DIAGNOSIS_FIELDS)}\n\n"
        f"Model's alternative strategy:\n{format_fields(strategy, STRATEGY_FIELDS)}\n\n"
        f"Grading rules:\n{JUDGE_DIAGNOSIS_RULES}"
        "- strategy_verdict VALID: the alternative strategy, executed correctly, leads to the reference answer. Otherwise INVALID.\n"
        f"{TAG_RULES}\n\nOutput exactly these tags:\n{tag_block(JUDGE_FIELDS)}"
    )


def build_v2_judge_prompt(instance: Dict[str, Any], diagnosis: Dict[str, str], strategies: List[str]) -> str:
    shown = "\n".join(f"Strategy {i}: {s}" for i, s in enumerate(strategies, start=1)) or "[no strategy was produced]"
    return (
        "You are an expert mathematics grader. A model solved the problem below and got the WRONG answer. Without seeing "
        "the reference answer, it then diagnosed its own failure and proposed alternative strategies. Grade them.\n\n"
        f"{judge_context(instance)}\n\n"
        f"Model's diagnosis:\n{format_fields(diagnosis, V2_DIAGNOSIS_FIELDS)}\n\n"
        f"Model's alternative strategies:\n{shown}\n\n"
        f"Grading rules:\n{JUDGE_DIAGNOSIS_RULES}"
        "- strategy_verdicts: for each strategy in order, VALID if that strategy, executed correctly, leads to the reference "
        "answer; otherwise INVALID.\n"
        f"{TAG_RULES}\n\nOutput exactly these tags:\n{tag_block(V2_JUDGE_FIELDS)}"
    )


# ---- Gold-free evaluation (run_open.py) ------------------------------------------------------------
# ESV runs on every instance, so nothing downstream of recognition may claim the answer is wrong: it was only flagged.

def self_flagged_note(instance: Dict[str, Any]) -> str:
    answer = f" (final answer {instance['initial_answer']})" if instance["initial_answer"] else ""
    return f"Your own strategy check concluded that the initial solution{answer} fails; this has not been confirmed."


SELF_REFINE_FIELDS = {
    "review": "check each step of the initial solution against the conditions stated in the problem; name any error found",
    "verdict": "CORRECT or INCORRECT",
    "reasoning": "if INCORRECT, the corrected step-by-step solution; if CORRECT, write KEEP",
    "answer_checks": "the checks described above, one line each",
    "final_answer": "the initial answer if CORRECT, otherwise the corrected answer, in the required format",
}


def build_self_refine_prompt(instance: Dict[str, Any]) -> str:
    """Control (Self-Refine style): review the solution and correct it if wrong, with the same answer checks as ESV,
    but no strategy verification, diagnosis or alternative strategies."""
    return (
        "Review your initial solution below carefully. Check every step and every condition the problem states. If you "
        "find an error, correct the solution; if the solution is correct, keep its answer.\n"
        f"{answer_check_instruction(instance['answer_format'])}\n\n"
        f"{solution_context(instance)}\n{TAG_RULES}\n\nOutput exactly these tags:\n{tag_block(SELF_REFINE_FIELDS)}"
    )


# ---- ESV v3 (run_esv3.py) --------------------------------------------------------------------------
# Built from results/analysis/esv_bottleneck_analysis.md. ESV runs only where the 4 CoT samples disagree (gold-free),
# and verifies the competing solutions the model already produced instead of re-solving: one call covers stages 2-4
# for every candidate. Universal Self-Consistency (Chen et al., 2023) is the control that sees the same samples but
# only picks the most consistent one.

V3_TAG_RULES = """
Format rules (strict):
- Output ONLY the tags listed, in order, each with an opening and a closing tag. No JSON, no Markdown fences, no text outside the tags.
- Write LaTeX normally with single backslashes; do not escape anything.
- Be concise: about 150 words per solution in total, one short line per check, and never leave a tag unfinished."""

V3_PER_SOLUTION_CHECKS = {
    "strategy": "the method this solution uses, in one sentence",
    "bindings": "one line per given of the problem the method depends on: \"quote from the problem\" -> how this solution used it -> MATCHES or MISMATCH",
    "conditions": "one line per condition the method needs: condition - HOLDS or VIOLATED in THIS problem - one-line reason",
}
V3_PER_SOLUTION_VERDICT = {
    "verdict": "SOUND or FLAWED",
    "failure_type": "BINDING, METHOD or COMPUTATION; NONE if SOUND",
    "flawed_step": "the first flawed step, quoted word for word from the solution; NONE if SOUND",
    "corrected_fact": "what that step should say instead; NONE if SOUND",
}


def v3_verify_fields(n: int) -> Dict[str, str]:
    fields = {f"s{i}_{name}": f"Solution {i}: {text}" for i in range(1, n + 1) for name, text in V3_PER_SOLUTION_CHECKS.items()}
    fields["divergence"] = ("the first step where the solutions disagree: quote each version and say which one the problem "
                            "statement supports, and why")
    fields.update({f"s{i}_{name}": f"Solution {i}: {text}" for i in range(1, n + 1) for name, text in V3_PER_SOLUTION_VERDICT.items()})
    return fields


def clipped(text: str) -> str:
    text = text or "[no reasoning was produced]"
    return text[:CONTEXT_CHAR_LIMIT] + " ...[truncated]" if len(text) > CONTEXT_CHAR_LIMIT else text


def build_v3_verify_prompt(instance: Dict[str, Any], solutions: List[Dict[str, str]]) -> str:
    """ESV v3 stages 2-4 on every distinct candidate. solutions: [{"reasoning", "answer"}], in order of first appearance;
    how many samples reached each answer is deliberately not shown."""
    shown = "\n\n".join(f"Solution {i}:\n<reasoning>{clipped(s['reasoning'])}</reasoning>\n<final_answer>{s['answer']}</final_answer>"
                        for i, s in enumerate(solutions, start=1))
    return (
        "You are performing Explicit Strategy Verification (ESV) on several solutions to the same problem: stage 2 "
        "(strategy verification), stage 3 (failure recognition) and stage 4 (diagnosis).\n"
        "The solutions reach DIFFERENT final answers, so at most one of them is correct, and all of them may be wrong. "
        "They are listed in arbitrary order; how many attempts reached each answer is not shown and is not evidence.\n\n"
        "For EACH solution:\n"
        "1. strategy: the method it uses, in one sentence.\n"
        "2. bindings: read the PROBLEM STATEMENT itself (not the solution's restatement of it) and quote, word for word, "
        "every given the method depends on: definitions and labels of variables, points and sides, ranges and intervals, "
        "constraints, special conditions, and exactly what is asked. For each quote, say how this solution used it and "
        "mark MATCHES (used exactly as stated) or MISMATCH (misread, replaced by a familiar convention, dropped or changed).\n"
        "3. conditions: each condition the method needs (hypotheses of each theorem or formula, which case applies, "
        "domains, signs), marked HOLDS or VIOLATED for THIS problem.\n"
        "Then divergence: find the first step where the solutions disagree, quote each version, and say which one the "
        "problem statement supports and why.\n"
        "Then, for EACH solution, the verdict:\n"
        "- SOUND only if every binding MATCHES, every condition HOLDS, and recomputing its key steps finds no error.\n"
        "- FLAWED otherwise, with failure_type BINDING (a given of the problem is misused), METHOD (the method cannot work "
        "for this problem) or COMPUTATION (only a calculation or algebra error); the first flawed step quoted word for word "
        "from that solution; and the corrected fact (what that step should say).\n"
        "For a SOUND solution write NONE in failure_type, flawed_step and corrected_fact.\n\n"
        f"{problem_block(instance)}\n\n{shown}\n{V3_TAG_RULES}\n\n"
        f"Output exactly these tags:\n{tag_block(v3_verify_fields(len(solutions)))}"
    )


USC_FIELDS = {"selected_response": "the number of the most consistent response"}


def build_usc_prompt(instance: Dict[str, Any], responses: List[Dict[str, str]]) -> str:
    """Universal Self-Consistency (Chen et al., 2023): all samples, duplicates included, and one selection."""
    shown = "\n\n".join(f"Response {i}:\n<reasoning>{clipped(r['reasoning'])}</reasoning>\n"
                        f"<final_answer>{r['answer'] or '[no final answer]'}</final_answer>"
                        for i, r in enumerate(responses, start=1))
    return (
        "I have generated the following responses to the question below. Evaluate these responses. Select the most "
        "consistent response based on majority consensus.\n\n"
        f"{problem_block(instance)}\n\n{shown}\n{TAG_RULES}\n\nOutput exactly this tag:\n{tag_block(USC_FIELDS)}"
    )
