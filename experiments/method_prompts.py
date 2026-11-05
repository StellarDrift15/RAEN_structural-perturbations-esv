"""Prompts for Direct QA, CoT and ESV v3.

The Direct QA and CoT prompts are the pilot's (pilot_experiment/src/methods/esv.py and DQV_CoT_Code_strict.py)
unchanged in wording, except that the answer-format note comes from the question type
(common.answer_format_note), so perturbed problems get the same instruction as their originals.

ESV v3 and its Universal Self-Consistency control are at the bottom: the only verification prompts still in use.
The superseded ESV v4 / v2 / judge / Self-Refine prompts were removed together with their runners; see git history
for run_open.py, analyze.py and the phase 2-3 code of run_methods.py.
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


def problem_block(instance: Dict[str, Any]) -> str:
    return f"Problem:\n{instance['question']}\n\n{instance['format_note']}"


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


def usc_fields(n: int = 4) -> Dict[str, str]:
    """"the number OF the most consistent response" was read as "the number THAT IS the most consistent response":
    solvers returned the answer value (35, 200, ...) instead of the response index, and usc_choice then had nothing
    to select with. The label must name the range and rule the answer value out explicitly."""
    return {"selected_response": f"ONE integer from 1 to {n}: the label of the response you select, "
                                 f"NOT its answer value"}


USC_FIELDS = usc_fields()


def build_usc_prompt(instance: Dict[str, Any], responses: List[Dict[str, str]]) -> str:
    """Universal Self-Consistency: all samples, duplicates included, and one selection."""
    n = len(responses)
    shown = "\n\n".join(f"Response {i}:\n<reasoning>{clipped(r['reasoning'])}</reasoning>\n"
                        f"<final_answer>{r['answer'] or '[no final answer]'}</final_answer>"
                        for i, r in enumerate(responses, start=1))
    return (
        "I have generated the following responses to the question below. Evaluate these responses. Select the most "
        "consistent response based on majority consensus.\n\n"
        f"{problem_block(instance)}\n\n{shown}\n{TAG_RULES}\n"
        f"- Identify your selection by its Response LABEL, the integer 1-{n} printed above it. Do not write the "
        f"answer itself: `<selected_response>2</selected_response>` means Response 2, not the answer 2.\n\n"
        f"Output exactly this tag:\n{tag_block(usc_fields(n))}"
    )
